--!strict
-- Parallel Luau support (Actors) for RoNetV4.1.
--
-- Single source of truth for training/eval compute that can run EITHER in the
-- calling thread ("single") OR on a pool of Actor workers ("multi"). The mode
-- is a user-facing toggle:
--     Parallel.setMode("single")   -- default; everything in-process
--     Parallel.setMode("multi")    -- dispatch jobs to N Actor workers
--     Parallel.setMode("auto")     -- multi on a Roblox runtime, else single
--
-- Grounded in the official Parallel Luau docs / API reference:
--   * Each Actor owns a separate Luau VM. ModuleScripts required inside an
--     Actor are re-required per VM, so every worker gets a private copy of the
--     RoNet graph and module-level state NEVER races across VMs.
--   * require() is illegal in a desynchronized (parallel) phase, so the worker
--     script requires RoNet at its top level (serial phase), keeps the refs,
--     then registers BindToMessageParallel. All math runs inside the parallel
--     handler using already-loaded refs.
--   * Instance mutation is illegal in parallel, so JOBS ARE PURE COMPUTE:
--     they only build Lua tables and run Tensor/Serialize/math.* calls. The
--     disposable MLP a job builds is fully local to the worker VM.
--   * Instances cannot cross VM boundaries, so payloads travel as plain tables
--     via Actor:SendMessage (copied by value), models travel as weight strings
--     (Serialize.dump/load), and results come back through a pool-owned
--     BindableEvent whose instance the worker resolves through the DataModel.
--   * SINGLE/MULTI PARITY IS BY CONSTRUCTION: the same runJob() does the math.
--     multi just slices the batch across workers and re-aggregates.
--
-- Ops (identical code path in both modes):
--   forward  : outputs for a batch of rows
--   loss     : scalar loss over the batch (row-count-weighted mean across shards)
--   backward : scalar loss + per-param gradients; the pool writes SUMMED grads
--              into the master model's params so the caller's optimizer:step()
--              works afterwards
--   scoreGenomes : per-genome loss (GA fitness) - the headline op for running
--              an evolutionary population eval in parallel
--
-- "multi" requires a real Roblox runtime (game + Instance + Actor + task). In
-- the Luau CLI (or anywhere those globals are absent) every mode falls back to
-- "single", so this module always compiles and runs.
return function(deps: any): any
	local Tensor = deps.Tensor
	local Serialize = deps.Serialize
	local Module = deps.Module
	local PRNG = deps.PRNG
	local MLP = deps.MLP
	local Losses = deps.Losses
	local CrossEntropy = deps.CrossEntropy

	-- Engine globals via rawget so this file compiles & runs in the CLI.
	local G = rawget(_G, "game")
	local InstanceC = rawget(_G, "Instance")
	local ActorC = rawget(_G, "Actor")
	local task = rawget(_G, "task")
	local hasTask = type(task) == "table"
	local warnOut = if type(rawget(_G, "warn")) == "function" then rawget(_G, "warn") else function(_m: any) end

	local function isRobloxRuntime(): boolean
		return type(G) == "table"
			and type(InstanceC) == "table"
			and type(ActorC) == "table"
			and hasTask
	end

	local Parallel: any = {}

	Parallel.MODE_SINGLE = "single"
	Parallel.MODE_MULTI = "multi"
	Parallel.MODE_AUTO = "auto"

	Parallel._mode = Parallel.MODE_SINGLE -- default: deterministic/off
	Parallel._workers = 4
	Parallel._parent = nil -- optional Instance parent for the pool folder

	--- True when running on a real Roblox runtime (game, Instance, Actor and
	--- task are all present), i.e. when "multi" mode can actually spawn Actor
	--- workers. Always false in the Luau CLI.
	function Parallel.available(): boolean
		return isRobloxRuntime()
	end

	--- Set execution mode. Unknown modes error; "multi" without a runtime
	--- downgrades to "single" with a warning.
	function Parallel.setMode(m: string, workers: number?)
		if m ~= Parallel.MODE_SINGLE and m ~= Parallel.MODE_MULTI and m ~= Parallel.MODE_AUTO then
			error("Parallel.setMode: unknown mode '" .. tostring(m) .. "'", 2)
		end
		if workers ~= nil and workers >= 1 then
			Parallel._workers = math.floor(workers)
		end
		Parallel._mode = m
		if m == Parallel.MODE_MULTI and not isRobloxRuntime() then
			warnOut("Parallel.setMode('multi') requires a Roblox runtime; falling back to 'single'")
			Parallel._mode = Parallel.MODE_SINGLE
		end
	end

	--- The configured mode string ("single", "multi" or "auto"), exactly as
	--- last set. Use Parallel.effective() for the mode that will actually run.
	function Parallel.getMode(): string
		return Parallel._mode
	end

	--- Set the worker count (>= 1) used by newPool/attach when no explicit
	--- worker count is given.
	function Parallel.setWorkers(n: number)
		if n < 1 then error("Parallel.setWorkers: need >= 1 worker", 2) end
		Parallel._workers = math.floor(n)
	end

	--- The configured worker count (>= 1).
	function Parallel.getWorkers(): number
		return Parallel._workers
	end

	--- Optional parent instance the pool folder is created under.
	function Parallel.setParent(p: any?)
		Parallel._parent = p
	end

	--- Effective mode right now ("auto" resolves against the runtime).
	function Parallel.effective(): string
		local m = Parallel._mode
		if m == Parallel.MODE_AUTO then
			return if isRobloxRuntime() then Parallel.MODE_MULTI else Parallel.MODE_SINGLE
		end
		return m
	end

	--==========================================================================
	-- SHARED COMPUTE - the ONLY place math lives. Runs identically in the
	-- calling thread and inside each worker VM.
	--==========================================================================

	-- Row arrays -> a batched input tensor.
	local function rowsToTensor(rows: { { number } }): any
		local n = #rows
		local f = #rows[1]
		local flat = table.create(n * f)
		for r = 1, n do
			local base = (r - 1) * f
			local row = rows[r]
			for c = 1, f do
				flat[base + c] = row[c]
			end
		end
		return Tensor.fromData(flat, { n, f }, false)
	end

	-- targets: flat {number} of scalars/class-ids, or a table of {number} rows.
	local function scalarLoss(lossName: string, out: any, targets: any): any
		local n = out.shape[1]
		local f = out.shape[2]
		if lossName == "mse" then
			local yt = if type(targets[1]) == "table"
				then rowsToTensor(targets)
				else Tensor.fromData(targets, { n, f }, false)
			local diff = Tensor.sub(out, yt)
			return Tensor.mean(Tensor.mul(diff, diff), -1)
		elseif lossName == "bce" then
			local yt = Tensor.fromData(targets, { n, f }, false)
			return Losses.binaryCrossEntropy(out, yt)
		elseif lossName == "crossentropy" then
			local flat: { number } = {}
			if type(targets[1]) == "table" then
				for r = 1, n do flat[r] = targets[r][1] end
			else
				for r = 1, n do flat[r] = targets[r] end
			end
			return CrossEntropy.crossEntropy(out, flat, 0.0)
		end
		error("Parallel: unknown lossName '" .. tostring(lossName) .. "'", 2)
	end

	-- Flatten per-param gradient buffers (arrays) for the job result.
	local function grabGrads(params: { any }): { { number } }
		local grads: { { number } } = {}
		for i = 1, #params do
			local pg = Tensor.gradData(params[i])
			if pg then
				grads[i] = table.clone(pg)
			else
				grads[i] = table.create(#params[i].data, 0)
			end
		end
		return grads
	end

	--- Portable weights string for a param list (Serialize.dump). Models (and
	--- GA genomes) travel across the Actor VM boundary as such strings.
	function Parallel.weightsString(params: { any }): string
		return Serialize.dump(params)
	end

	--- Rebuild a fresh MLP from a portable weights string (used on workers and
	--- by the genome op). act must match the model the string came from.
	function Parallel.modelFromWeights(layerSizes: { number }, act: string, weightsStr: string): any
		local model = MLP.new("par", layerSizes, PRNG.new(1), { act = act })
		Serialize.load(Module.collectParams(model), weightsStr)
		return model
	end

	--- GA genome {layerSizes, weights, biases, act?} -> a Serialize.dump string
	--- (genomes therefore cross the VM boundary as ONE string each).
	function Parallel.genomeToWeightsString(genome: any): string
		local model = MLP.new("gen", genome.layerSizes, PRNG.new(1), { act = genome.act or "tanh" })
		local params = Module.collectParams(model)
		local li = 0
		for i = 1, #model.layers do
			local layer = model.layers[i]
			if layer.W ~= nil and layer.b ~= nil then
				li += 1
				local inN = layer.W.shape[2]
				local outN = layer.W.shape[1]
				local w = genome.weights[li]
				local b = genome.biases[li]
				local flatW = table.create(inN * outN)
				for r = 1, outN do
					local base = (r - 1) * inN
					for c = 1, inN do
						flatW[base + c] = w[r][c]
					end
				end
				layer.W.data = flatW
				local flatB = table.create(outN)
				for r = 1, outN do flatB[r] = b[r] end
				layer.b.data = flatB
			end
		end
		return Serialize.dump(params)
	end

	--- THE shared job implementation. Payload fields:
	---   op: "forward" | "loss" | "backward" | "genome"
	---   layerSizes / act / weightsStr : the single model everyone shares
	---   rows        : this shard's rows ({ {number} })
	---   targets     : {number} or { {number} } slice (loss/backward/genome)
	---   lossName    : "mse" | "bce" | "crossentropy" (loss/backward/genome)
	---   genomes     : { string } weight-strings (genome only)
	--- Returns a plain-Lua table (numbers/tables only): safe to copy back.
	function Parallel.runJob(job: any): any
		local model = Parallel.modelFromWeights(job.layerSizes, job.act or "tanh", job.weightsStr)
		local xs = rowsToTensor(job.rows)

		if job.op == "forward" then
			local out = model:forward(xs)
			return { shape = table.clone(out.shape), data = table.clone(out.data) }
		end

		if job.op == "genome" then
			local scores: { number } = {}
			for g = 1, #job.genomes do
				local m2 = Parallel.modelFromWeights(job.layerSizes, job.act or "tanh", job.genomes[g])
				local l = scalarLoss(job.lossName or "mse", m2:forward(xs), job.targets)
				scores[g] = Tensor.item(l)
			end
			return { scores = scores }
		end

		local loss = scalarLoss(job.lossName or "mse", model:forward(xs), job.targets)
		if job.op == "loss" then
			return { loss = Tensor.item(loss) }
		end

		if job.op == "backward" then
			-- Scale by this shard's row count so each shard contributes RAW
			-- row-sums of gradients (cancelling the 1/n_k inside scalarLoss).
			-- The pool then divides the summed grads by the total row count,
			-- exactly reproducing the full-batch mean-loss gradient no matter
			-- how the batch was split.
			local scaled = Tensor.mul(loss, Tensor.scalar(#job.rows, false))
			Tensor.backward(scaled, false)
			return { loss = Tensor.item(loss), nrows = #job.rows, grads = grabGrads(Module.collectParams(model)) }
		end

		error("Parallel: unknown op '" .. tostring(job.op) .. "'", 2)
	end

	--==========================================================================
	-- ORCHESTRATION
	--==========================================================================

	-- Derive MLP sizes from a live model's weights (W is {out, in}).
	local function sizesFromParams(params: { any }): { number }
		local sizes: { number } = {}
		local pri: string = "in"
		for i = 1, #params do
			local sh = params[i].shape
			if #sh == 2 then
				if pri == "in" then
					table.insert(sizes, sh[2])
					pri = "out"
				end
				table.insert(sizes, sh[1])
			end
		end
		return sizes
	end

	-- Contiguous slice of a list [lo..hi].
	local function sliceRows(rows: { any }, lo: number, hi: number): { any }
		local out: { any } = {}
		for i = lo, hi do out[i - lo + 1] = rows[i] end
		return out
	end

	-- Contiguous [lo, hi] ranges partitioning `n` items into `shards` chunks.
	local function rangesFor(n: number, shards: number): { { number } }
		local out: { { number } } = {}
		for si = 1, shards do
			local lo = 1 + math.floor((si - 1) * n / shards)
			local hi = math.min(n, math.floor(si * n / shards))
			if lo <= hi then table.insert(out, { lo, hi }) end
		end
		return out
	end

	-- Frozen master-model description (weights string computed once per op so
	-- every worker evaluates the EXACT same weights).
	local function descOf(model: any): any
		local params = Module.collectParams(model)
		return { sizes = sizesFromParams(params), act = "tanh", weightsStr = Serialize.dump(params) }
	end

	--------------------------------------------------------------------------------
	-- Worker script source. string.format placeholders:
	--   %s pool folder DataModel path, %s RoNet module DataModel path,
	--   %d worker index. Top level runs SERIAL (require + bind), the "Job"
	--   handler runs PARALLEL and is pure compute.
	--------------------------------------------------------------------------------
	local WORKER_SRC = [[
local actor = script:GetActor()
local function resolveFromGame(path)
	local node = game
	for token in string.gmatch(path, "[^.]+") do
		local child = node:FindFirstChild(token)
		if not child then return nil end
		node = child
	end
	return node
end
local poolFolder = resolveFromGame("%s")
if not poolFolder then
	error("RoNet worker: pool folder not found at %s")
end
local resultEvent = poolFolder:WaitForChild("Results")
local RoNetRoot = resolveFromGame("%s")
if not RoNetRoot then
	error("RoNet worker: RoNet module not found")
end
local RoNet = require(RoNetRoot)
if not RoNet.Parallel or not RoNet.Parallel.runJob then
	error("RoNet worker: Parallel.runJob missing on the RoNet module")
end
actor:BindToMessageParallel("Job", function(jobId, idx, payload)
	local ok, res = pcall(RoNet.Parallel.runJob, payload)
	resultEvent:Fire("result", jobId, idx, ok, res)
end)
resultEvent:Fire("ready", %d)
]]

	local function newWorker(actor: any, poolFolderPath: string, modulePath: string, index: number): any
		local src = string.format(WORKER_SRC, poolFolderPath, poolFolderPath, modulePath, index)
		local script = InstanceC.new("Script")
		script.Name = "Worker" .. index
		script.Source = src
		script.Parent = actor
		return script
	end

	local function waitUntil(cond: () -> boolean, timeout: number): boolean
		local t0 = os.clock()
		while not cond() do
			if os.clock() - t0 > timeout then return false end
			task.wait()
		end
		return true
	end

	--- Build a pool. opts: { workers?, mode?, parent?, modulePath? }
	--- modulePath: DataModel path of the requireable RoNet root ModuleScript
	--- (default "ReplicatedStorage.RoNet").
	function Parallel.newPool(opts: any?): any
		local o = opts or {}
		local workers = o.workers or Parallel._workers
		local mode = o.mode or Parallel.effective()

		if mode == Parallel.MODE_SINGLE then
			return {
				mode = Parallel.MODE_SINGLE,
				workers = 1,
				dispatch = function(_self: any, jobs: { any }): { any }
					local results: { any } = {}
					for i = 1, #jobs do
						results[i] = Parallel.runJob(jobs[i])
					end
					return results
				end,
				shutdown = function() end,
			}
		end

		if not isRobloxRuntime() then
			warnOut("Parallel.newPool: 'multi' needs a Roblox runtime; using a single-mode pool")
			return Parallel.newPool({ workers = workers, mode = Parallel.MODE_SINGLE })
		end

		local modulePath = o.modulePath or "ReplicatedStorage.RoNet"
		local parent = o.parent or Parallel._parent or G:GetService("ReplicatedStorage")

		local poolFolder = InstanceC.new("Folder")
		poolFolder.Name = "RoNet_WorkerPool"
		poolFolder.Parent = parent
		local event = InstanceC.new("BindableEvent")
		event.Name = "Results"
		event.Parent = poolFolder
		local poolFolderPath = parent.Name .. "." .. poolFolder.Name

		local actors: { any } = {}
		for i = 1, workers do
			local actor = InstanceC.new("Actor")
			actor.Name = "RoNet_Worker" .. i
			actor.Parent = poolFolder
			newWorker(actor, poolFolderPath, modulePath, i)
			table.insert(actors, actor)
		end

		local ready = 0
		local results: { [number]: { any } } = {}
		local inflight = false
		local conn = event.Event:Connect(function(topic: string, ...)
			if topic == "ready" then
				ready += 1
			elseif topic == "result" then
				local args = { ... }
				local jobId = args[1]
				local idx = args[2]
				local ok = args[3]
				local res = args[4]
				local bucket: any = results[jobId]
				if bucket then
					bucket.out[idx] = { ok = ok, res = res }
					bucket.count += 1
				end
			end
		end)

		local pool: any = {
			mode = Parallel.MODE_MULTI,
			workers = workers,
			folder = poolFolder,
			actors = actors,
			event = event,
			_conn = conn,
			_ready = false,
		}

		-- Wait for the worker VMs to come up and register their handlers. This
		-- is the only place "multi" yields at construction time.
		pool._ready = waitUntil(function()
			return ready >= workers
		end, 15)

		--- Tear the pool down: disconnects the result listener and destroys the
		--- worker folder (and its Actor instances).
		function pool:shutdown()
			local conn: any = self._conn
			if conn then
				conn:Disconnect()
				self._conn = nil
			end
			local folder: any = self.folder
			if folder then
				folder:Destroy()
				self.folder = nil
			end
		end

		--- Run a list of jobs on the pool and block until every result is back.
		--- Jobs run in shard order across workers; results are returned in the
		--- same order as the jobs argument. Only one dispatch may be in flight
		--- at a time. In "single" mode this just runs the jobs in-process.
		function pool:dispatch(jobs: { any }): { any }
			if not self._ready then
				error("Parallel pool: workers did not become ready (are the Scripts running?)", 2)
			end
			if inflight then
				error("Parallel pool: only one dispatch may be in flight at a time", 2)
			end
			inflight = true
			local jobId = #results + 1
			local bucket = { count = 0, out = {}, jobs = jobs }
			results[jobId] = bucket
			for i = 1, math.min(#jobs, self.workers) do
				self.actors[i]:SendMessage("Job", jobId, i, jobs[i])
			end
			local ok = waitUntil(function()
				return bucket.count >= #jobs
			end, 30)
			inflight = false
			if not ok then
				error("Parallel pool: dispatch timed out waiting for workers", 2)
			end
			-- Recover shard order: results keyed by their job index.
			local out: { any } = {}
			for i = 1, #jobs do
				local entry = bucket.out[i]
				if not entry then
					error(string.format("Parallel pool: missing result for job %d", i), 2)
				end
				if not entry.ok then
					error("Parallel pool worker error: " .. tostring(entry.res), 2)
				end
				out[i] = entry.res
			end
			results[jobId] = nil
			return out
		end

		return pool
	end

	--==========================================================================
	-- OP FACADE - one implementation used by both modes.
	--==========================================================================

	-- Forward: rows -> outputs. Returns { shape, rows = {{number}} }.
	local function poolForward(pool: any, model: any, rows: { { number } }, opts: any?): any
		local o = opts or {}
		local desc = descOf(model)
		desc.act = o.act or desc.act
		local n = #rows
		local ranges = rangesFor(n, pool.workers)
		local jobs: { any } = {}
		for si = 1, #ranges do
			local lo, hi = ranges[si][1], ranges[si][2]
			jobs[si] = {
				op = "forward",
				layerSizes = desc.sizes,
				act = desc.act,
				weightsStr = desc.weightsStr,
				rows = sliceRows(rows, lo, hi),
			}
		end
		local results = pool:dispatch(jobs)
		local shape = results[1].shape
		local total = 0
		for i = 1, #results do total += #results[i].data end
		local data: { number } = {}
		local ptr = 0
		for i = 1, #results do
			for j = 1, #results[i].data do
				ptr += 1
				data[ptr] = results[i].data[j]
			end
		end
		local f = shape[2]
		local outRows: { { number } } = {}
		for i = 1, n do
			local row: { number } = {}
			local base = (i - 1) * f
			for c = 1, f do row[c] = data[base + c] end
			outRows[i] = row
		end
		return { shape = table.clone(shape), rows = outRows }
	end

	-- Slice targets to rows [lo..hi]. Flat {number} targets carry `f` values per
	-- row (one per output), so their block is f times wider than the row slice.
	local function sliceTargets(targets: any, lo: number, hi: number, f: number): any
		if type(targets[1]) == "table" then
			return sliceRows(targets, lo, hi)
		end
		local size = (hi - lo + 1) * f
		local out: { any } = {}
		local base = (lo - 1) * f
		for i = 1, size do out[i] = targets[base + i] end
		return out
	end

	-- Loss: row-count-weighted mean across shards.
	local function poolLoss(pool: any, model: any, rows: { { number } }, targets: any, lossName: string, opts: any?): number
		local o = opts or {}
		local desc = descOf(model)
		desc.act = o.act or desc.act
		local outF = desc.sizes[#desc.sizes] or 1
		local n = #rows
		local ranges = rangesFor(n, pool.workers)
		local jobs: { any } = {}
		for si = 1, #ranges do
			local lo, hi = ranges[si][1], ranges[si][2]
			jobs[si] = {
				op = "loss",
				layerSizes = desc.sizes,
				act = desc.act,
				weightsStr = desc.weightsStr,
				rows = sliceRows(rows, lo, hi),
				targets = sliceTargets(targets, lo, hi, outF),
				lossName = lossName,
			}
		end
		local results = pool:dispatch(jobs)
		local totalLoss = 0
		local totalRows = 0
		for i = 1, #results do
			local cnt = ranges[i][2] - ranges[i][1] + 1
			totalLoss += results[i].loss * cnt
			totalRows += cnt
		end
		return if totalRows > 0 then totalLoss / totalRows else 0
	end

	-- Backward: returns mean loss AND writes summed gradients into the master
	-- model's params so the caller can optimizer:step() right after.
	local function poolBackward(pool: any, model: any, rows: { { number } }, targets: any, lossName: string, opts: any?): number
		local o = opts or {}
		local desc = descOf(model)
		desc.act = o.act or desc.act
		local params = Module.collectParams(model)
		local outF = desc.sizes[#desc.sizes] or 1
		local n = #rows
		local ranges = rangesFor(n, pool.workers)
		local jobs: { any } = {}
		for si = 1, #ranges do
			local lo, hi = ranges[si][1], ranges[si][2]
			jobs[si] = {
				op = "backward",
				layerSizes = desc.sizes,
				act = desc.act,
				weightsStr = desc.weightsStr,
				rows = sliceRows(rows, lo, hi),
				targets = sliceTargets(targets, lo, hi, outF),
				lossName = lossName,
			}
		end
		local results = pool:dispatch(jobs)
		local totalLoss = 0
		local totalRows = 0
		local sums: { { number } } = {}
		for i = 1, #results do
			local cnt = ranges[i][2] - ranges[i][1] + 1
			totalLoss += results[i].loss * cnt
			totalRows += cnt
			local grads = results[i].grads
			for g = 1, #grads do
				local target = sums[g]
				if not target then
					target = table.create(#grads[g], 0)
					sums[g] = target
				end
				for e = 1, #grads[g] do target[e] += grads[g][e] end
			end
		end
		-- write summed grads into the master model (accumulate semantics). The
		-- workers produced raw row-sums, so divide by the total row count to
		-- match the full-batch mean-loss gradient.
		for g = 1, #params do
			local summed = sums[g] or table.create(#params[g].data, 0)
			local garr = params[g].grad
			if not garr then
				garr = table.create(#params[g].data, 0)
				params[g].grad = garr
			end
			for e = 1, #garr do garr[e] = (if totalRows > 0 then summed[e] / totalRows else 0) end
		end
		return if totalRows > 0 then totalLoss / totalRows else 0
	end

	-- Score a GA genome population (lower loss = better). All genomes must
	-- share one architecture; each is scored on the FULL row batch. Returns
	-- one loss per genome in input order.
	local function poolScoreGenomes(pool: any, genomes: { any }, rows: { { number } }, targets: any, lossName: string, opts: any?): { number }
		local o = opts or {}
		local type0 = genomes[1].act or o.act or "tanh"
		local gens: { any } = {}
		local sizes = genomes[1].layerSizes
		for i = 1, #genomes do
			gens[i] = { act = genomes[i].act or type0, str = Parallel.genomeToWeightsString(genomes[i]) }
		end
		local ranges = rangesFor(#gens, math.min(pool.workers, #gens))
		local jobs: { any } = {}
		for si = 1, #ranges do
			local gl = ranges[si]
			local chunk: { string } = {}
			for k = gl[1], gl[2] do chunk[k - gl[1] + 1] = gens[k].str end
			jobs[si] = {
				op = "genome",
				layerSizes = sizes,
				act = type0,
				weightsStr = gens[1].str,
				rows = rows,
				targets = targets,
				lossName = lossName,
				genomes = chunk,
			}
		end
		local results = pool:dispatch(jobs)
		local scores: { number } = {}
		for si = 1, #ranges do
			local gl = ranges[si]
			for k = gl[1], gl[2] do scores[k] = results[si].scores[k - gl[1] + 1] end
		end
		return scores
	end

	--- One-shot API: attach the op facade to a live model using a default pool.
	--- Call handle:forward/loss/backward/scoreGenomes, then :shutdown().
	function Parallel.attach(model: any, opts: any?): any
		local o = opts or {}
		local pool = Parallel.newPool({
			workers = o.workers or Parallel._workers,
			mode = o.mode or Parallel.effective(),
			parent = o.parent or Parallel._parent,
			modulePath = o.modulePath,
		})
		return {
			mode = pool.mode,
			workers = pool.workers,
			model = model,
			_pool = pool,
			forward = function(self: any, rows: { { number } }, opts: any?)
				return poolForward(self._pool, self.model, rows, opts)
			end,
			loss = function(self: any, rows: { { number } }, targets: any, lossName: string, opts: any?)
				return poolLoss(self._pool, self.model, rows, targets, lossName, opts)
			end,
			backward = function(self: any, rows: { { number } }, targets: any, lossName: string, opts: any?)
				return poolBackward(self._pool, self.model, rows, targets, lossName, opts)
			end,
			scoreGenomes = function(self: any, genomes: { any }, rows: { { number } }, targets: any, lossName: string, opts: any?)
				return poolScoreGenomes(self._pool, genomes, rows, targets, lossName, opts)
			end,
			shutdown = function(self: any)
				self._pool:shutdown()
			end,
		}
	end

	return Parallel
end
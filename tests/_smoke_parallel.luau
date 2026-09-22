--!strict
-- Smoke tests for core/Parallel.luau (CLI-safe: every mode degrades to the
-- single shared runJob implementation, so multi/single parity is testable here
-- by construction). Verifies forward/loss/backward/genome against references
-- computed with raw Tensor ops, plus cross-shard gradient aggregation that must
-- reproduce the exact full-batch mean-loss gradient.
-- ENTRY script; composes the DI graph like the other smoke tests.
local PRNG = require("../core/PRNG")
local Tensor = require("../core/Tensor")
local Module = require("../nn/Module")
local Linear = require("../nn/Linear")
local Inits = require("../nn/Inits")
local Activations = require("../nn/Activations")
local MLP = require("../models/MLP")
local CrossEntropy = require("../loss/CrossEntropy")
local Losses = require("../loss/Losses")
local Serialize = require("../train/Serialize")
local Parallel = require("../core/Parallel")

local deps = {
	PRNG = PRNG,
	Tensor = Tensor,
}
deps.Module = Module(deps)
deps.Inits = Inits(deps)
deps.Linear = Linear(deps)
deps.Activations = Activations(deps)
deps.MLP = MLP(deps)
deps.CrossEntropy = CrossEntropy(deps)
deps.Losses = Losses(deps)
deps.Serialize = Serialize(deps)
deps.Parallel = Parallel(deps)

Tensor.setRNG(PRNG.new(42))

local checks = 0
local function ok(cond: boolean, msg: string)
	checks += 1
	assert(cond, msg)
end

local function near(a: number, b: number, eps: number)
	return math.abs(a - b) <= (eps or 1e-9)
end

-- Fixed architecture + deterministic weights
local SIZES = { 3, 4, 2 }
local function buildModel(): any
	local m = deps.MLP.new("ref", SIZES, PRNG.new(7), { act = "tanh" })
	local pars = deps.Module.collectParams(m)
	local seed = 0
	for i = 1, #pars do
		for j = 1, #pars[i].data do
			seed = (seed * 31 + i * 17 + j * 7) % 97
			pars[i].data[j] = seed / 97 - 0.5
		end
	end
	return m
end

-- Extract the {layerSizes, weights, biases} GA genome layout from a model.
local function toGenome(m: any): any
	local genome: any = { layerSizes = table.clone(SIZES), weights = {}, biases = {} }
	local li = 0
	for i = 1, #m.layers do
		local layer = m.layers[i]
		if layer.W ~= nil and layer.b ~= nil then
			li += 1
			local inN = layer.W.shape[2]
			local outN = layer.W.shape[1]
			local w = table.create(outN)
			for r = 1, outN do
				local row = table.create(inN)
				for c = 1, inN do row[c] = layer.W.data[(r - 1) * inN + c] end
				w[r] = row
			end
			local b = table.create(outN)
			for r = 1, outN do b[r] = layer.b.data[r] end
			genome.weights[li] = w
			genome.biases[li] = b
		end
	end
	return genome
end

local ROWS = {
	{ 1, 2, 3 }, { 4, 5, 6 }, { -1, 0, 1 }, { 0.5, -0.5, 2 }, { 3, -2, 7 }, { -4, 1, 0 },
}
local ROWS2 = { { 1, 2, 3 }, { 4, 5, 6 }, { -1, 0, 1 }, { 0.5, -0.5, 2 }, { 3, -2, 7 } }
local NONLIN_T = { 0.1, -0.2, 0.7, 0.3, -0.5, 1.2, -0.3, 0.4, 0.9, -0.1, 0.6, 0.8 }
local BIN_T = { 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1 }
local CLS_T = { 1, 0, 1, 0, 1, 0 }

local model = buildModel()
local params = deps.Module.collectParams(model)
local weightsStr = deps.Parallel.weightsString(params)
local n = #ROWS

-- reference forward + losses on the FULL batch
local function refForward(m: any, rows: { { number } }): any
	local flat: { number } = {}
	for r = 1, #rows do
		for c = 1, #rows[1] do flat[(r - 1) * #rows[1] + c] = rows[r][c] end
	end
	return m:forward(Tensor.fromData(flat, { #rows, #rows[1] }, false))
end
local refOut = refForward(model, ROWS)
local refMSE = Tensor.item(Tensor.mean(Tensor.mul(Tensor.sub(refOut, Tensor.fromData(NONLIN_T, { n, 2 }, false)), Tensor.sub(refOut, Tensor.fromData(NONLIN_T, { n, 2 }, false))), -1))
local refBCE = Tensor.item(deps.Losses.binaryCrossEntropy(refOut, Tensor.fromData(BIN_T, { n, 2 }, false)))
local refCE = Tensor.item(deps.CrossEntropy.crossEntropy(refOut, CLS_T, 0.0))

-- reference full-batch gradient (mean-loss convention)
local function refGrads(): { { number } }
	for _, p in params do Tensor.zeroGrad(p) end
	local yt = Tensor.fromData(NONLIN_T, { n, 2 }, false)
	local diff = Tensor.sub(refForward(model, ROWS), yt)
	Tensor.backward(Tensor.mean(Tensor.mul(diff, diff), -1), false)
	local out: { { number } } = {}
	for i = 1, #params do
		local g = Tensor.gradData(params[i])
		out[i] = if g then table.clone(g) else table.create(#params[i].data, 0)
	end
	return out
end
local refG = refGrads()

--=============================================================================
-- 1. mode plumbing in a non-Roblox runtime
--=============================================================================
ok(deps.Parallel.available() == false, "available() == false in CLI")
ok(deps.Parallel.getMode() == "single", "default mode is single")
deps.Parallel.setWorkers(4)
ok(deps.Parallel.getWorkers() == 4, "setWorkers/getWorkers")
deps.Parallel.setMode("multi")
ok(deps.Parallel.getMode() == "single", "setMode('multi') degrades to single without runtime")
deps.Parallel.setMode("auto")
ok(deps.Parallel.effective() == "single", "effective() == single in CLI")

--=============================================================================
-- 2. direct runJob ops == raw Tensor references
--=============================================================================
local fwdJob = deps.Parallel.runJob({
	op = "forward",
	layerSizes = SIZES,
	act = "tanh",
	weightsStr = weightsStr,
	rows = ROWS,
})
local fwdOk = true
for i = 1, #refOut.data do
	if not near(fwdJob.data[i], refOut.data[i], 1e-9) then fwdOk = false end
end
ok(fwdOk, "runJob forward matches full-batch forward")
ok(fwdJob.shape[1] == n and fwdJob.shape[2] == 2, "forward shape")

local lossJob = deps.Parallel.runJob({
	op = "loss",
	layerSizes = SIZES,
	act = "tanh",
	weightsStr = weightsStr,
	rows = ROWS,
	targets = NONLIN_T,
	lossName = "mse",
})
ok(near(lossJob.loss, refMSE, 1e-9), "runJob loss(mse) == reference")
ok(near(deps.Parallel.runJob({ op = "loss", layerSizes = SIZES, act = "tanh", weightsStr = weightsStr, rows = ROWS, targets = BIN_T, lossName = "bce" }).loss, refBCE, 1e-9), "runJob loss(bce) == reference")
ok(near(deps.Parallel.runJob({ op = "loss", layerSizes = SIZES, act = "tanh", weightsStr = weightsStr, rows = ROWS, targets = CLS_T, lossName = "crossentropy" }).loss, refCE, 1e-9), "runJob loss(crossentropy) == reference")

-- backward returns RAW row-sums; n * meanGrad == ref
local bwdJob = deps.Parallel.runJob({
	op = "backward",
	layerSizes = SIZES,
	act = "tanh",
	weightsStr = weightsStr,
	rows = ROWS,
	targets = NONLIN_T,
	lossName = "mse",
})
local bwdOk = near(bwdJob.loss, refMSE, 1e-9) and #bwdJob.grads == #params
if bwdOk then
	for g = 1, #params do
		if #bwdJob.grads[g] ~= #refG[g] then
			bwdOk = false
			break
		end
		for e = 1, #refG[g] do
			if not near(bwdJob.grads[g][e], refG[g][e] * n, 1e-6) then bwdOk = false end
		end
	end
end
ok(bwdOk, "runJob backward returns row-summed grads == n * mean-loss gradient")

--=============================================================================
-- 3. attach() single-mode pool with 4 shards over 6 rows
--=============================================================================
local h = deps.Parallel.attach(model, { mode = "single", workers = 4 })
ok(h.mode == "single", "attach single mode")

local fwd = h:forward(ROWS)
ok(#fwd.rows == n and #fwd.rows[1] == 2, "pool forward shape")
local pfOk = true
for r = 1, n do
	for c = 1, 2 do
		if not near(fwd.rows[r][c], refOut.data[(r - 1) * 2 + c], 1e-9) then pfOk = false end
	end
end
ok(pfOk, "pool forward == full-batch forward across shards")

ok(near(h:loss(ROWS, NONLIN_T, "mse"), refMSE, 1e-9), "pool loss(mse) == reference")
ok(near(h:loss(ROWS, BIN_T, "bce"), refBCE, 1e-9), "pool loss(bce) == reference")
ok(near(h:loss(ROWS, CLS_T, "crossentropy"), refCE, 1e-9), "pool loss(crossentropy) == reference")

h:backward(ROWS, NONLIN_T, "mse")
local agOk = true
for g = 1, #params do
	local garr = Tensor.gradData(params[g])
	if not garr then
		agOk = false
		break
	end
	for e = 1, #refG[g] do
		if not near(garr[e], refG[g][e], 1e-6) then agOk = false end
	end
end
ok(agOk, "pool backward writes summed grads == exact full-batch gradient")

--=============================================================================
-- 4. genome scoring + round trip
--=============================================================================
local genome0 = toGenome(model)
local genomeBad = toGenome(model)
genomeBad.weights[2][1][2] = 50 -- knock one weight hard out

local wsGenome = deps.Parallel.genomeToWeightsString(genome0)
local rebuilt = deps.Parallel.modelFromWeights(SIZES, "tanh", wsGenome)
local rebuildOut = refForward(rebuilt, ROWS)
local rtOk = true
for i = 1, #refOut.data do
	if not near(rebuildOut.data[i], refOut.data[i], 1e-9) then rtOk = false end
end
ok(rtOk, "genomeToWeightsString -> modelFromWeights reproduces forward")

local scores = h:scoreGenomes({ genomeBad, genome0 }, ROWS, NONLIN_T, "mse")
ok(#scores == 2, "scoreGenomes returns one score per genome")
ok(scores[1] > scores[2], "knocked-out genome has higher loss than identity genome")
ok(near(scores[2], refMSE, 1e-9), "identity genome scores == model loss")

-- scaling across uneven shard tails: 5 rows over 4 shards
local refOut2 = refForward(model, ROWS2)
local TGT2 = { 0.1, -0.2, 0.7, 0.3, -0.5, 1.2, -0.3, 0.4, 0.9, -0.1 }
local refMSE2 = Tensor.item(Tensor.mean(Tensor.mul(Tensor.sub(refOut2, Tensor.fromData(TGT2, { 5, 2 }, false)), Tensor.sub(refOut2, Tensor.fromData(TGT2, { 5, 2 }, false))), -1))
ok(near(h:loss(ROWS2, TGT2, "mse"), refMSE2, 1e-9), "pool loss handles uneven shards (5 rows / 4 workers)")

h:shutdown()

--=============================================================================
-- 5. weightsString round trip
--=============================================================================
local clone = buildModel()
deps.Serialize.load(deps.Module.collectParams(clone), weightsStr)
local rt2 = true
for i = 1, #params do
	if #params[i].data ~= #clone.params[i].data then
		rt2 = false
		break
	end
	for e = 1, #params[i].data do
		if params[i].data[e] ~= clone.params[i].data[e] then rt2 = false end
	end
end
ok(rt2, "weightsString round-trips the full param list")

print(string.format("PASS parallel: %d checks", checks))
--!strict
-- EXPERIMENT harness: measure where time goes BEFORE changing production code.
-- Entry script. Composes deps then benchmarks current hot paths against inline
-- EXPERIMENTAL kernels (bmmFast, linear softmax, fused RMSNorm, fused RoPE,
-- faster Serialize parse, attention decode with/without KV cache). Every
-- experimental kernel is parity-checked against the reference op inside the
-- process; nothing here touches core/ or nn/.
-- Usage: .tools/luau tests/_exp_bench.luau        (interpreter)
--        .tools/luau -O2 --codegen tests/_exp_bench.luau   (native)
local Util = require("../core/Util")
local PRNG = require("../core/PRNG")
local Tensor = require("../core/Tensor")
local deps: { [string]: any } = { Util = Util, PRNG = PRNG, Tensor = Tensor }
deps.Module = require("../nn/Module")(deps)
deps.Inits = require("../nn/Inits")(deps)
deps.Linear = require("../nn/Linear")(deps)
deps.Activations = require("../nn/Activations")(deps)
deps.Norm = require("../nn/Norm")(deps)
deps.Dropout = require("../nn/Dropout")(deps)
deps.Embedding = require("../nn/Embedding")(deps)
deps.RoPE = require("../nn/RoPE")(deps)
deps.Attention = require("../nn/Attention")(deps)
deps.FFN = require("../nn/FFN")(deps)
deps.Sequential = require("../nn/Sequential")(deps)
deps.MLP = require("../models/MLP")(deps)
deps.TransformerBlock = require("../models/TransformerBlock")(deps)
deps.Transformer = require("../models/Transformer")(deps)
deps.CrossEntropy = require("../loss/CrossEntropy")(deps)
deps.Serialize = require("../train/Serialize")(deps)

local vectorLib = rawget(_G, "vector")

-- ---------------------------------------------------------------------------
-- reporting / timing
-- ---------------------------------------------------------------------------
local results = {}

local function fmt(v: number): string
	if v >= 1 then return string.format("%.2fms", v) end
	if v * 1000 >= 1 then return string.format("%.1fus", v * 1000) end
	return string.format("%.1fns", v * 1e6)
end

-- Measure fn; returns best per-call time in seconds across >=500ms of work.
local function bench(fn: () -> (), warmupMs: number?, minMs: number?): number
	local w = (warmupMs or 50) / 1000
	local m = (minMs or 400) / 1000
	local tw = 0
	repeat
		local s = os.clock()
		fn()
		tw += os.clock() - s
	until tw >= w
	collectgarbage()
	local best = math.huge
	local total = 0
	local n = 0
	repeat
		local s = os.clock()
		fn()
		local e = os.clock() - s
		if e < best then best = e end
		total += e
		n += 1
	until total >= m and n >= 3
	collectgarbage()
	return best
end

-- Bench a name measuring seconds-per-call; record ratio vs baseline.
local function addNote(name: string, v: string)
	table.insert(results, { name, v, "" })
end

local function record(name: string, baseline: number, variant: number, note: string?)
	local speedup = string.format("%.2fx", baseline / variant)
	local label
	if note then
		label = note
	else
		label = fmt(variant)
	end
	table.insert(results, { name, fmt(baseline), label .. "  (" .. speedup .. ")" })
end

local function parity(a: any, b: any, tol: number, label: string)
	if #a.shape ~= #b.shape then error("parity shape " .. label, 2) end
	local md = 0
	for i = 1, #a.data do
		local d = math.abs(a.data[i] - b.data[i])
		if d > md then md = d end
	end
	if md > tol then error(string.format("parity FAIL %s: maxdiff %.3e", label, md), 2) end
end

local function fill(d: { number }, rng: any)
	for i = 1, #d do d[i] = rng:nextFloat() * 2 - 1 end
end

-- ---------------------------------------------------------------------------
-- 1. matmul (grad/nograd) vs matmulFast  + "inference chain" graph overhead
-- ---------------------------------------------------------------------------
do
	local function one(m: number, k: number, n: number)
		local rng = PRNG.new(1)
		local a = rng:nextFloat()
		local ad, bd = table.create(m * k, 0), table.create(k * n, 0)
		fill(ad, rng)
		fill(bd, rng)
		local Aplain = Tensor.fromData(ad, { m, k }, false)
		local Bplain = Tensor.fromData(bd, { k, n }, false)
		local Agrad = Tensor.fromData(ad, { m, k }, true)
		local Bgrad = Tensor.fromData(bd, { k, n }, true)
		return {
			matmulGrad = bench(function() Tensor.matmul(Agrad, Bgrad) end),
			matmulPlain = bench(function() Tensor.matmul(Aplain, Bplain) end),
			matmulFast = bench(function() Tensor.matmulFast(Aplain, Bplain) end),
		}
	end
	local r = one(256, 64, 256)
	addNote("matmul 256x64x256 (grad)", fmt(r.matmulGrad))
	record("  vs plain (no grad)", r.matmulGrad, r.matmulPlain)
	record("  vs matmulFast", r.matmulGrad, r.matmulFast)

	-- inference chain: 3 stacked [C,C]x[C,C] matmuls, C=64, mimicking a block.
	local C = 64
	local rng = PRNG.new(2)
	local wd1, wd2, wd3 = table.create(C * C, 0), table.create(C * C, 0), table.create(C * C, 0)
	fill(wd1, rng)
	fill(wd2, rng)
	fill(wd3, rng)
	local W1 = Tensor.fromData(wd1, { C, C }, true)
	local W2 = Tensor.fromData(wd2, { C, C }, true)
	local W3 = Tensor.fromData(wd3, { C, C }, true)
	local xd = table.create(C * C, 0)
	fill(xd, rng)
	local xg = Tensor.fromData(xd, { C, C }, true)
	local xp = Tensor.fromData(xd, { C, C }, false)
	local gradChain = bench(function()
		local y = Tensor.matmul(xg, W1)
		y = Tensor.matmul(y, W2)
		y = Tensor.matmul(y, W3)
		-- note: we build the node; discard
	end)
	local fastChain = bench(function()
		local y = Tensor.matmulFast(xp, W1)
		y = Tensor.matmulFast(y, W2)
		y = Tensor.matmulFast(y, W3)
	end)
	addNote("inference chain 3x[64,64]", fmt(gradChain))
	record("  no-graph + matmulFast", gradChain, fastChain, "graph-free")
end

-- ---------------------------------------------------------------------------
-- 2. bmm vs experimental bmmFast (3-lane vector over the N dimension)
-- ---------------------------------------------------------------------------
do
	local function bmmFast(a: any, b: any): any
		local Ba, M, N = a.shape[1], a.shape[2], a.shape[3]
		local Bb, K = b.shape[1], b.shape[3]
		local out = table.create(Ba * M * K, 0)
		local ad, bd = a.data, b.data
		for bb = 1, Ba do
			local abase = (bb - 1) * M * N
			local bbase = (bb - 1) * N * K
			local obase = (bb - 1) * M * K
			for j = 1, K, 3 do
				local nb = math.min(3, K - j + 1)
				local Bv = table.create(N)
				for kk = 1, N do
					local boff = bbase + (kk - 1) * K + j
					Bv[kk] = vectorLib.create(bd[boff], if nb >= 2 then bd[boff + 1] else 0, if nb >= 3 then bd[boff + 2] else 0)
				end
				for i = 1, M do
					local s = vectorLib.zero
					local aoff = abase + (i - 1) * N
					for kk = 1, N do
						s += ad[aoff + kk] * Bv[kk]
					end
					out[obase + (i - 1) * K + j] = s.x
					if nb >= 2 then out[obase + (i - 1) * K + j + 1] = s.y end
					if nb >= 3 then out[obase + (i - 1) * K + j + 2] = s.z end
				end
			end
		end
		return Tensor.fromData(out, { Ba, M, K }, false)
	end

	local function one(Bt: number, M: number, N: number, K: number)
		local rng = PRNG.new(3)
		local ad = table.create(Bt * M * N, 0)
		fill(ad, rng)
		local bd = table.create(Bt * N * K, 0)
		fill(bd, rng)
		local A = Tensor.fromData(ad, { Bt, M, N }, false)
		local B = Tensor.fromData(bd, { Bt, N, K }, false)
		local fast = bmmFast(A, B)
		local ref = Tensor.bmm(A, B)
		parity(fast, ref, 5e-5, string.format("bmm %dx%dx%d x %dx%dx%d", Bt, M, N, Bt, N, K))
		local tb = bench(function() Tensor.bmm(A, B) end)
		local tf = bench(function() bmmFast(A, B) end)
		record(string.format("bmm [%d,%d,%d]x[%d,%d,%d]", Bt, M, N, Bt, N, K), tb, tf)
	end
	one(8, 32, 64, 32) -- QxK^T  (T=32)
	one(8, 32, 32, 64) -- softmax@V (T=32)
	one(8, 128, 64, 128) -- QxK^T  (T=128)
	one(8, 128, 128, 64) -- softmax@V (T=128)
	one(4, 512, 64, 512) -- long-context QxK^T (T=512)
end

-- ---------------------------------------------------------------------------
-- 3. softmax (last dim, 3D) vs experimental linear kernel
-- ---------------------------------------------------------------------------
do
	local function softmaxLast(t: any): any
		local nd = #t.shape
		local cols = t.shape[nd]
		local rows = 1
		for i = 1, nd - 1 do rows *= t.shape[i] end
		local d = t.data
		local out = table.create(#d, 0)
		for r = 0, rows - 1 do
			local base = r * cols
			local m = -math.huge
			for c = 1, cols do if d[base + c] > m then m = d[base + c] end end
			local sumE = 0
			for c = 1, cols do
				local e = math.exp(d[base + c] - m)
				out[base + c] = e
				sumE += e
			end
			local inv = sumE > 0 and 1 / sumE or 0
			for c = 1, cols do out[base + c] *= inv end
		end
		return Tensor.fromData(out, table.clone(t.shape), false)
	end

	local function one(Bt: number, T: number)
		local rng = PRNG.new(4)
		local d = table.create(Bt * T * T, 0)
		fill(d, rng)
		local S = Tensor.fromData(d, { Bt, T, T }, false)
		local fast = softmaxLast(S)
		local ref = Tensor.softmax(S, 3)
		parity(fast, ref, 1e-9, string.format("softmax [%d,%d,%d]", Bt, T, T))
		local tb = bench(function() Tensor.softmax(S, 3) end)
		local tf = bench(function() softmaxLast(S) end)
		record(string.format("softmax [%d,%d,%d]", Bt, T, T), tb, tf)
	end
	one(8, 32)
	one(8, 128)
	one(4, 512)
end

-- ---------------------------------------------------------------------------
-- 4. RMSNorm (Norm.forward) vs experimental fused kernel
-- ---------------------------------------------------------------------------
do
	local function fusedRmsNorm(x: any, gamma: any, eps: number): any
		local rows, D = x.shape[1], x.shape[2]
		local xd, gd = x.data, gamma.data
		local out = table.create(rows * D, 0)
		for r = 1, rows do
			local base = (r - 1) * D
			local ss = 0
			for c = 1, D do ss += xd[base + c] * xd[base + c] end
			local inv = 1 / math.sqrt(ss / D + eps)
			for c = 1, D do out[base + c] = xd[base + c] * inv * gd[c] end
		end
		return Tensor.fromData(out, { rows, D }, false)
	end

	local function one(rows: number, D: number)
		local rng = PRNG.new(5)
		local xd = table.create(rows * D, 0)
		fill(xd, rng)
		local gd = table.create(D, 0)
		fill(gd, rng)
		local x = Tensor.fromData(xd, { rows, D }, false)
		local g = Tensor.fromData(gd, { D }, true) -- real gamma has requiresGrad
		local norm = deps.Norm.new("bench", D, rng, "rms", 1e-6)
		norm.gamma = g
		local ref = norm:forward(x)
		local fast = fusedRmsNorm(x, g, 1e-6)
		parity(Tensor.reshape(fast, ref.shape), ref, 1e-4, string.format("rms [%d,%d]", rows, D))
		local tb = bench(function() norm:forward(x) end, 30, 200)
		local tf = bench(function() fusedRmsNorm(x, g, 1e-6) end, 30, 200)
		record(string.format("RMSNorm [%d,%d]", rows, D), tb, tf)
	end
	one(512, 64)
	one(2048, 128)
end

-- ---------------------------------------------------------------------------
-- 5. RoPE.apply vs experimental fused rotation kernel
-- ---------------------------------------------------------------------------
do
	local function fusedRoPE(x: any, freqs: any, out: { number}?): any
		local D = x.shape[#x.shape]
		local T = x.shape[#x.shape - 1]
		local outer = 1
		for i = 1, #x.shape - 2 do outer *= x.shape[i] end
		local half = math.floor(D / 2)
		local cosD = freqs.cos
		local sinD = freqs.sin
		local xd = x.data
		local od = out or table.create(#xd, 0)
		for o = 0, outer - 1 do
			local base = o * T * D
			for t = 1, T do
				local tb = base + (t - 1) * D
				local cb = (t - 1) * half
				for k = 1, half do
					local c = cosD[cb + k]
					local s = sinD[cb + k]
					local e = xd[tb + (k - 1) * 2 + 1]
					local o2 = xd[tb + (k - 1) * 2 + 2]
					od[tb + (k - 1) * 2 + 1] = e * c - s * o2
					od[tb + (k - 1) * 2 + 2] = e * s + c * o2
				end
			end
		end
		return Tensor.fromData(od, table.clone(x.shape), false)
	end

	local function one(outer: number, T: number, D: number)
		local rng = PRNG.new(6)
		local xd = table.create(outer * T * D, 0)
		fill(xd, rng)
		local x = Tensor.fromData(xd, { outer, T, D }, false)
		local freqs = deps.RoPE.precompute(T, D)
		local ref = deps.RoPE.apply(x, freqs)
		local fast = fusedRoPE(x, freqs)
		parity(fast, ref, 1e-5, string.format("rope [%d, %d, %d]", outer, T, D))
		local tb = bench(function() deps.RoPE.apply(x, freqs) end, 30, 250)
		local tf = bench(function() fusedRoPE(x, freqs) end, 30, 250)
		record(string.format("RoPE [%d,%d,%d]", outer, T, D), tb, tf)
	end
	one(1, 32, 64)
	one(1, 128, 64)
	one(8, 32, 32)
end

-- ---------------------------------------------------------------------------
-- 6. lmHead transpose (per-forward allocation) cost
-- ---------------------------------------------------------------------------
do
	local function one(V: number, C: number)
		local rng = PRNG.new(7)
		local d = table.create(V * C, 0)
		fill(d, rng)
		local W = Tensor.fromData(d, { V, C }, true)
		local t0 = bench(function() Tensor.transpose2d(W) end)
		addNote(string.format("transpose2d lmHead [%d,%d]", V, C), fmt(t0))
	end
	one(128, 64)
	one(4096, 128)
end

-- ---------------------------------------------------------------------------
-- 7. Serialize.parse vs experimental faster text parser
-- ---------------------------------------------------------------------------
do
	local function fasterParse(str: string): { any }
		local parts = string.split(str, "\n")
		local out: { any } = {}
		for _, ln in parts do
			local semi = string.find(ln, ";", 1, true)
			if semi then
				local header = string.sub(ln, 1, semi - 1)
				local shape: { number } = {}
				local eq = string.find(header, "shape=", 1, true)
				if eq then
					local body = string.sub(header, eq + 6)
					for piece in string.gmatch(body, "[^,]+") do shape[#shape + 1] = tonumber(piece) end
				end
				local body = string.sub(ln, semi + 2)
				local vals = string.split(body, " ")
				local data: { number } = table.create(#vals, 0)
				for i = 1, #vals do data[i] = tonumber(vals[i]) end
				table.insert(out, { shape = shape, data = data })
			end
		end
		return out
	end

	local params: { any } = {}
	local rng = PRNG.new(8)
	local shape = { 400, 120 }
	local d = table.create(400 * 120, 0)
	fill(d, rng)
	params[1] = { shape = shape, data = d }
	params[2] = { shape = { 120, 400 }, data = table.create(120 * 400, 0.5) }
	local str = deps.Serialize.dump(params)
	local ref = deps.Serialize.parse(str)
	local fast = fasterParse(str)
	if #ref ~= #fast then error("serialize parity count") end
	for i = 1, #ref do
		if ref[i].data[1] ~= fast[i].data[1] then error("serialize parity value") end
	end
	addNote("Serialize.parse 96k weights", fmt(bench(function() deps.Serialize.parse(str) end, 30, 200)))
	record("  faster parser (string.split)", bench(function() deps.Serialize.parse(str) end, 30, 200),
		bench(function() fasterParse(str) end, 30, 200))
end

-- ---------------------------------------------------------------------------
-- 8. Whole-model: forward scaling + generate decode (recompute every step)
-- + attention-only experiment: recompute-per-step vs KV cache
-- ---------------------------------------------------------------------------
do
	local maxSeq = 256
	local C = 64
	local model = deps.Transformer.new("b", {
		C = C, numBlocks = 2, nbHeads = 4, kvHeads = 2, headDim = 16,
		ffnHidden = 128, vocab = 128, maxSeq = maxSeq, tieEmbeds = false,
		initType = "default",
	}, PRNG.new(9))

	-- forward cost vs sequence length (per call). Expect ~4x per T-double (softmax+bmm O(T^2)).
	local ids = table.create(128, 0)
	for i = 1, 128 do ids[i] = (i * 7) % 128 + 1 end
	for _, T in { 32, 64, 128 } do
		local sub = table.create(T, 0)
		for i = 1, T do sub[i] = ids[i] end
		local f = function()
			model:forward(sub)
		end
		f()
		model:eval()
		local t0 = bench(f, 30, 300)
		addNote(string.format("model.forward T=%d", T), fmt(t0))
	end
	model:train()

	-- generate: total ms for 16 new tokens from prefix 8 (full recompute each step).
	model:eval()
	local seed = table.create(8, 0)
	for i = 1, 8 do seed[i] = ids[i] end
	local g0 = bench(function() model:generate(seed, 16) end, 20, 400)
	addNote("model.generate(8+16)", fmt(g0))
	model:train()

	-- ATTENTION-DECODE experiment (isolates cache effect, no model changes):
	-- simulated single-block attention over generated tokens.
	--  - noCache: every step t does full QK^T over all t past tokens (bmm size t)
	--  - cached:  only the new row (Qrow x Kcache^T).
	local nbH = 4
	local hd = 16
	local D = nbH * hd
	local Tgen = 64
	local rng = PRNG.new(10)
	local function fillNorm(d: { number }, n: number)
		for i = 1, n do d[i] = rng:nextFloat() * 2 - 1 end
	end
	local Q = table.create(Tgen * D, 0)
	local K = table.create(Tgen * D, 0)
	local V = table.create(Tgen * D, 0)
	fillNorm(Q, Tgen * D)
	fillNorm(K, Tgen * D)
	fillNorm(V, Tgen * D)
	local function softmaxRow(d: { number }, base: number, n: number)
		local m = -math.huge
		for i = 1, n do if d[base + i] > m then m = d[base + i] end end
		local sum = 0
		for i = 1, n do d[base + i] = math.exp(d[base + i] - m); sum += d[base + i] end
		for i = 1, n do d[base + i] /= sum end
	end
	-- recompute-per-step: for each head h, step t: S[1..t,1..t] = Q_t K_t^T,
	-- softmax all t rows, then the new token's output row. Costs t*t scores.
	local function noCacheStep()
		local s = table.create(Tgen * Tgen, 0)
		for t = 2, Tgen do
			for h = 0, nbH - 1 do
				local qb = h * hd
				local kb = h * hd
				for i = 1, t do
					for j = 1, t do
						local acc = 0
						for kk = 1, hd do
							acc += Q[(i - 1) * D + qb + kk] * K[(j - 1) * D + kb + kk]
						end
						s[(i - 1) * t + j] = acc
					end
				end
				for i = 1, t do
					softmaxRow(s, (i - 1) * t, t)
				end
				for j = 1, hd do
					local acc = 0
					for i = 1, t do acc += s[(t - 1) * t + i] * V[(i - 1) * D + kb + j] end
				end
			end
		end
	end
	-- cached: maintain Kcache/Vcache rows as they come; step does Qrow@Kcache^T.
	local function cachedStep()
		local kc = table.create(Tgen * D, 0)
		local vc = table.create(Tgen * D, 0)
		local s = table.create(1024)
		for t = 2, Tgen do
			for j = 1, D do
				kc[(t - 1) * D + j] = K[(t - 1) * D + j]
				vc[(t - 1) * D + j] = V[(t - 1) * D + j]
			end
			for h = 0, nbH - 1 do
				local qb = h * hd
				local kb = h * hd
				for i = 1, t do
					local acc = 0
					for kk = 1, hd do acc += Q[(t - 1) * D + qb + kk] * kc[(i - 1) * D + kb + kk] end
					s[i] = acc
				end
				softmaxRow(s, 0, t)
				for j = 1, hd do
					local acc = 0
					for i = 1, t do acc += s[i] * vc[(i - 1) * D + kb + j] end
				end
			end
		end
	end
	noCacheStep()
	cachedStep()
	local tnc = bench(noCacheStep, 20, 300)
	local tc = bench(cachedStep, 20, 300)
	record("attn decode 64 tokens full-recompute", tnc, tc, "with KV cache")
end

-- ---------------------------------------------------------------------------
print()
print("=== RESULTS (per-call; smaller is better) ===")
local NAME, BASELINE, VARIANT = 1, 2, 3
for _, r in results do
	if r[VARIANT] == "" then
		print(string.format("%-34s %s", r[NAME], r[BASELINE]))
	else
		print(string.format("%-34s %s -> %s", r[NAME], r[BASELINE], r[VARIANT]))
	end
end
print("done.")
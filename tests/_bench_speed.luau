local Util = require("../core/Util")
local PRNG = require("../core/PRNG")
local Tensor = require("../core/Tensor")
local Module = require("../nn/Module")
local Inits = require("../nn/Inits")
local deps = { Util = Util, PRNG = PRNG, Tensor = Tensor }
deps.Module = Module(deps)
deps.Inits = Inits(deps)
for _, name in { "Activations", "Norm", "Dropout", "Embedding", "RoPE", "Attention", "FFN", "Sequential", "Linear", "MLP", "TransformerBlock", "Transformer", "CrossEntropy" } do
	local path = if name == "CrossEntropy" then "../loss/CrossEntropy" elseif name == "MLP" or name == "TransformerBlock" or name == "Transformer" then "../models/" .. name else "../nn/" .. name
	deps[name] = require(path)(deps)
end

local rng = PRNG.new(42)
local function fill(n, fn)
	local t = table.create(n)
	for i = 1, n do t[i] = fn(i) end
	return t
end

local function bench(label, f, iters)
	f()
	local t0 = os.clock()
	for _ = 1, iters do f() end
	local dt = os.clock() - t0
	print(string.format("%-40s %12.1f us/op", label, dt / iters * 1e6))
	return dt / iters * 1e6
end

local results = {}
local function ab(a, b, iters)
	local ta = bench(a[1], a[2], iters)
	local tb = bench(b[1], b[2], iters)
	print(string.format("  -> %s/op                      %10.1f x", if tb <= ta then "faster" else "slower", math.max(ta, tb) / math.min(ta, tb)))
	table.insert(results, { name = a[1], d = ta, n = tb })
end

--> old composite RMSNorm (what every forward did before): x*x -> mean -> +eps -> ^-0.5 -> *x -> *gamma
local function rmsComposite(x, gamma, eps)
	local nd = #x.shape
	local gs = table.create(nd)
	local rd = table.create(nd)
	for i = 1, nd do
		gs[i] = 1
		rd[i] = x.shape[i]
	end
	gs[nd] = x.shape[nd]
	rd[nd] = 1
	local gammaR = Tensor.reshape(gamma, gs)
	local x2 = Tensor.mul(x, x)
	local ms = Tensor.mean(x2, nd)
	local rms = Tensor.reshape(ms, rd)
	local inv = Tensor.pow(Tensor.addScalar(rms, eps), -0.5)
	return Tensor.mul(Tensor.mul(x, inv), gammaR)
end

do
	local n, m, k = 256, 128, 720
	local aT = Tensor.fromData(fill(n * m, function(i) return ((i * 37) % 91) / 47 end), { n, m }, true)
	local bT = Tensor.fromData(fill(m * k, function(i) return ((i * 53) % 89) / 41 end), { m, k }, true)
	local aF = Tensor.fromData(fill(n * m, function(i) return ((i * 37) % 91) / 47 end), { n, m }, false)
	local bF = Tensor.fromData(fill(m * k, function(i) return ((i * 53) % 89) / 41 end), { m, k }, false)
	ab(
		{ "matmul AUTODIFF graph forward (train)", function() local o = Tensor.matmul(aT, bT) end },
		{ "matmul auto-fast SIMD (no grad)", function() local o = Tensor.matmul(aF, bF) end },
		60
	)
end

do
	local B, T, hd = 8, 128, 64
	local x = Tensor.fromData(fill(B * T * hd, function(i) return ((i % 29) - 14) / 9 end), { B, T, hd }, false)
	local gamma = Tensor.fromData(fill(hd, function(i) return 1 + (i % 5) / 9 end), { hd }, false)
	local eps = 1e-6
	ab(
		{ "rmsNorm COMPOSITE (old: mul/mean/pow/...)", function() rmsComposite(x, gamma, eps) end },
		{ "rmsNorm FUSED (Tensor.rmsNorm)", function() Tensor.rmsNorm(x, gamma, eps) end },
		400
	)
end

-- old composite softmax over the last dim: max-shift, exp, sum, divide (each a
-- full tensor op with temporaries: max => reshape to [.., 1], sub/exp/sum,
-- reshape back, div against the [.., 1] view — exactly the pre-fusion path).
local function softmaxComposite(x, nd)
	local keep = {} for i = 1, nd - 1 do keep[i] = x.shape[i] end
	keep[nd] = 1
	local mx = Tensor.reshape(Tensor.max(x, nd), keep)
	local e = Tensor.exp(Tensor.sub(x, mx))
	local sm = Tensor.reshape(Tensor.sum(e, nd), keep)
	return Tensor.div(e, sm)
end
do
	local B, T, D = 8, 128, 128
	local x = Tensor.fromData(fill(B * T * D, function(i) return ((i % 17) - 8) / 7 end), { B, T, D }, false)
	ab(
		{ "softmax COMPOSITE (old: max/sub/exp/sum/div)", function() softmaxComposite(x, 3) end },
		{ "softmax last-dim FUSED (Tensor.softmax)", function() Tensor.softmax(x, 3) end },
		200
	)
end

print("\n=== end-to-end (native, real model) ===")
local m2 = deps.Transformer.new("m", { C = 128, numBlocks = 4, nbHeads = 4, kvHeads = 1, headDim = 32, ffnHidden = 256, vocab = 2048, maxSeq = 128, tieEmbeds = true }, PRNG.new(9))
local ids = {}
for i = 1, 16 do ids[i] = (i * 7) % 2048 + 1 end
m2:eval()
ab(
	{ "forward TRAIN mode (graph + bmm)", function() m2:train() return m2.forward(m2, ids) end },
	{ "forward EVAL mode (fused + SIMD)", function() m2:eval() return m2.forward(m2, ids) end },
	5
)
ab(
	{ "generate recompute (full forward per token)", function() m2.generate(m2, ids, 8, { temperature = 1.0, cache = false }) end },
	{ "generate with KV cache", function() m2.generate(m2, ids, 8, { temperature = 1.0 }) end },
	3
)
print("done.")
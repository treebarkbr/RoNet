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
local cfg = {
	C = 128, numBlocks = 4, nbHeads = 4, kvHeads = 1, headDim = 32,
	ffnHidden = 256, vocab = 2048, maxSeq = 128, tieEmbeds = true,
	dropoutP = 0.0, initType = "default",
}
local m = deps.Transformer.new("bench", cfg, rng)
local ids = {}
for i = 1, 16 do ids[i] = (i * 7) % cfg.vocab + 1 end
m:eval()

local function bench(label, f, iters)
	local t0 = os.clock()
	for _ = 1, iters do local ok, e = pcall(f); if not ok then print(label .. " ERROR " .. tostring(e)); return end end
	local dt = os.clock() - t0
	print(string.format("%-38s %10.1f us/op", label, dt / iters * 1e6))
end

bench("forward eval T=16 (fused+SIMD)", function() m.forward(m, ids) end, 10)

m:train()
bench("forward train T=16 (graph+bmm)", function() m.forward(m, ids) end, 10)
m:eval()

bench("generate cached 16+8", function() m.generate(m, ids, 8, { temperature = 1.0 }) end, 5)
bench("generate recompute 16+8", function() m.generate(m, ids, 8, { temperature = 1.0, cache = false }) end, 3)
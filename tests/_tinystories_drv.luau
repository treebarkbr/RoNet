--!strict
-- Shared driver for the TinyStories llama2.c import experiment. Assembles the
-- DI graph, requires a generated weights module, prints the cfg, a LOGITS line
-- (for tools/pt2ronet_check.py parity) and a KV-cache generation run.
return function(weightsPath: string, seed: { number }, nTokens: number)
	local Util = require("../core/Util")
	local PRNG = require("../core/PRNG")
	local Tensor = require("../core/Tensor")
	local Module = require("../nn/Module")
	local Inits = require("../nn/Inits")
	local Norm = require("../nn/Norm")
	local Dropout = require("../nn/Dropout")
	local Embedding = require("../nn/Embedding")
	local RoPE = require("../nn/RoPE")
	local Attention = require("../nn/Attention")
	local FFN = require("../nn/FFN")
	local TransformerBlock = require("../models/TransformerBlock")
	local Transformer = require("../models/Transformer")
	local Serialize = require("../train/Serialize")

	local deps = { Util = Util, PRNG = PRNG, Tensor = Tensor }
	deps.Module = Module(deps)
	deps.Inits = Inits(deps)
	deps.Norm = Norm(deps)
	deps.Dropout = Dropout(deps)
	deps.Embedding = Embedding(deps)
	deps.RoPE = RoPE(deps)
	deps.Attention = Attention(deps)
	deps.FFN = FFN(deps)
	deps.TransformerBlock = TransformerBlock(deps)
	deps.Transformer = Transformer(deps)
	deps.Serialize = Serialize(deps)

	Tensor.setRNG(PRNG.new(7))

	local t0 = os.clock()
	local W = require(weightsPath)
	local model = deps.Transformer.new("ts", W.cfg, PRNG.new(1))
	deps.Serialize.load(deps.Module.collectParams(model), W.weights)
	print(string.format("LOADED %.2fs", os.clock() - t0))

	local keys = { "C", "numBlocks", "nbHeads", "kvHeads", "headDim", "ffnHidden",
		"vocab", "maxSeq", "tieEmbeds", "dropoutP", "ropeBase", "rmsEps" }
	local cparts = {}
	for _, k in keys do
		local v = W.cfg[k]
		cparts[#cparts + 1] = k .. "=" .. (if type(v) == "number" then string.format("%g", v) else tostring(v))
	end
	print("CFG " .. table.concat(cparts, " "))

	model:eval()

	-- Parity probe: ids {1,2,3,4}; print only the LAST token's logits row so the
	-- LOGITS line stays bounded (V numbers, matches tools/pt2ronet_check.py).
	local par = { 1, 2, 3, 4 }
	local logits = model:forward(par)
	local V = W.cfg.vocab
	local base = (#par - 1) * V
	local parts = table.create(V)
	for v = 1, V do
		parts[v] = string.format("%.9g", logits.data[base + v])
	end
	print("LOGITS " .. table.concat(parts, " "))

	local tg = os.clock()
	local out = model:generate(seed, nTokens, { temperature = 0.8, topK = 50, cache = true })
	local dt = os.clock() - tg
	print("GEN " .. table.concat(out, " "))
	print(string.format("GEN_TOKENS %d GEN_TIME %.2fs %.2f tok/s", #out, dt, #out / dt))
end
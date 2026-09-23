--!strict
-- PyTorch import semantic check: forward the imported tiny checkpoint and print
-- the logits so tools/pt2ronet_check.py can compare them against an independent
-- numpy reference implementation of the same checkpoint. NOT part of run_all.sh.
-- ENTRY script; composes the DI graph (mirrors RoNet.luau).
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

local deps = {
	Util = Util,
	PRNG = PRNG,
	Tensor = Tensor,
}
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

local W = require("./fixtures/pt_tiny.weights")
local model = deps.Transformer.new("pt_check", W.cfg, PRNG.new(1))
deps.Serialize.load(deps.Module.collectParams(model), W.weights)

local ids = { 1, 2, 3, 4, 5, 6, 7, 8 }
local logits = model:forward(ids)
local parts = table.create(#logits.data)
for i = 1, #logits.data do
	parts[i] = string.format("%.9g", logits.data[i])
end
print("LOGITS " .. table.concat(parts, " "))
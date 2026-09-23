--!strict
-- PyTorch import smoke: load a Llama-style safetensors checkpoint converted by
-- tools/pt2ronet.py into a RoNet Transformer, then verify the load writes every
-- param exactly (numeric round-trip through Serialize) and the model runs.
-- ENTRY script; composes the DI graph the same way as RoNet.luau.
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

local ModuleM = deps.Module
local TransformerM = deps.Transformer
local SerializeM = deps.Serialize

local function check(cond: boolean, msg: string)
	if not cond then error("FAIL: " .. msg) end
end

local total = 0
local function ok(msg: string)
	total += 1
	print("ok:   " .. msg)
end

-- Fixture generated from tests/fixtures/pt_tiny (random Llama-style checkpoint,
-- L=2, C=16, 4 heads / 2 kv heads (GQA), dFFN=32, vocab 33, RoPE base 10000).
local W = require("./fixtures/pt_tiny.weights")

local model = TransformerM.new("pt", W.cfg, PRNG.new(1))
local params = ModuleM.collectParams(model)
ok("built transformer with " .. #params .. " params from imported cfg")

-- Config fidelity: norm eps and RoPE base must match the source checkpoint.
check(model.rmsEps == W.cfg.rmsEps, "rmsEps threaded through model")
check(W.cfg.rmsEps == 1e-5, "fixture rmsEps is 1e-5 (vs RoNet default 1e-6)")

-- Tied embeddings must be a single shared tensor after import.
check(model.lmHeadW == model.emb.table, "tied lm-head shares the embedding table")

-- Load imported weights, then re-dump and compare element-for-element.
SerializeM.load(params, W.weights)
local reDump = SerializeM.parse(SerializeM.dump(params))
local orig = SerializeM.parse(W.weights)
check(#reDump == #orig, "same param count after dump round-trip")
local totalVals = 0
for i = 1, #orig do
	local a = reDump[i]
	local b = orig[i]
	check(#a.shape == #b.shape, "shape rank matches for param " .. i)
	for d = 1, #a.shape do
		check(a.shape[d] == b.shape[d], "shape dim " .. d .. " matches for param " .. i)
	end
	check(#a.data == #b.data, "value count matches for param " .. i)
	for j = 1, #a.data do
		check(a.data[j] == b.data[j], "value " .. j .. " of param " .. i .. " exact after load->dump")
	end
	totalVals += #a.data
end
ok("all " .. totalVals .. " imported weights round-trip through Serialize exactly")

-- Forward (exercises GQA kv-expansion and RoPE) must give finite logits.
local logits = model:forward({ 1, 2, 3, 4, 5, 6, 7, 8 })
check(#logits.shape == 2 and logits.shape[1] == 8 and logits.shape[2] == W.cfg.vocab,
	"forward output shape is [T, vocab]")
for _, v in logits.data do
	check(v == v and v > -1e30, "logits finite")
end
ok("forward produces finite [8, " .. W.cfg.vocab .. "] logits (GQA + RoPE)")

-- Batched forward must agree with the single-sequence forward on the same ids.
local idsT = Tensor.fromData({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 1, 8 }, false)
local batch = model:forwardBatch(idsT)
check(#batch.data == #logits.data, "batch and single-seq logits same size")
for v = 1, #logits.data do
	check(batch.data[v] == logits.data[v], "logit " .. v .. " matches between batch and single-seq")
end
ok("forwardBatch agrees with forward on the same sequence")

print("PASS _smoke_pt (" .. total .. " checks)")
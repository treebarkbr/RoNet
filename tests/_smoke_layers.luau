--!strict
-- Layer/model smoke suite. This file is an ENTRY script: it may require() any
-- leaf module (resolution from this file's dir via ../). Composes the DI graph
-- exactly as init.luau will.
local Util = require("../core/Util")
local PRNG = require("../core/PRNG")
local Tensor = require("../core/Tensor")
local Module = require("../nn/Module")
local Inits = require("../nn/Inits")
local Linear = require("../nn/Linear")
local Activations = require("../nn/Activations")
local Norm = require("../nn/Norm")
local Dropout = require("../nn/Dropout")
local Embedding = require("../nn/Embedding")
local RoPE = require("../nn/RoPE")
local Attention = require("../nn/Attention")
local FFN = require("../nn/FFN")
local Sequential = require("../nn/Sequential")
local MLP = require("../models/MLP")
local TransformerBlock = require("../models/TransformerBlock")
local Transformer = require("../models/Transformer")
local CrossEntropy = require("../loss/CrossEntropy")

local deps = {
	Util = Util,
	PRNG = PRNG,
	Tensor = Tensor,
}
deps.Module = Module(deps)
deps.Inits = Inits(deps)
deps.Linear = Linear(deps)
deps.Activations = Activations(deps)
deps.Norm = Norm(deps)
deps.Dropout = Dropout(deps)
deps.Embedding = Embedding(deps)
deps.RoPE = RoPE(deps)
deps.Attention = Attention(deps)
deps.FFN = FFN(deps)
deps.Sequential = Sequential(deps)
deps.MLP = MLP(deps)
deps.TransformerBlock = TransformerBlock(deps)
deps.Transformer = Transformer(deps)
deps.CrossEntropy = CrossEntropy(deps)

local rng = PRNG.new(1337)
Tensor.setRNG(rng)

local failures = 0
local function check(label: string, cond: boolean)
	if not cond then
		failures += 1
		print("FAIL: " .. label)
	else
		print("ok:   " .. label)
	end
end

do -- XOR via MLP + hand-rolled MSE + manual SGD
	local model = deps.MLP.new("xor", { 2, 16, 16, 1 }, rng, { act = "relu" })
	local X = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
	local Y = { 0, 1, 1, 0 }
	local lr = 0.1
	local prevLoss = math.huge
	for step = 1, 400 do
		local total = 0
		for i = 1, 4 do
			local inp = Tensor.fromData(X[i], { 1, 2 }, false)
			local pred = model.forward(model, inp)
			local diff = Tensor.sub(pred, Tensor.scalar(Y[i]))
			local loss = Tensor.mean(Tensor.mul(diff, diff), -1)
			Tensor.backward(loss, false)
			total += loss.data[1]
			for _, p in model.params do
				local g = p.grad
				if g then
					for k = 1, #p.data do p.data[k] -= lr * g[k] end
					p.grad = nil
				end
			end
		end
		prevLoss = total / 4
	end
	check("MLP loss decreased (xor)", prevLoss < 0.2)
	-- evaluate
	local okVals = 0
	for i = 1, 4 do
		local inp = Tensor.fromData(X[i], { 1, 2 }, false)
		local pred = model.forward(model, inp)
		local got = if pred.data[1] > 0.5 then 1 else 0
		if got == Y[i] then okVals += 1 end
	end
	check("MLP classified XOR correctly", okVals == 4)
end

do -- tiny transformer: forward, loss, backward, finite-diff gradient sanity
	local cfg = {
		C = 8, numBlocks = 2, nbHeads = 1, kvHeads = 1, headDim = 8,
		ffnHidden = 16, vocab = 16, maxSeq = 16, tieEmbeds = true,
		dropoutP = 0.0, initType = "default",
	}
	local tmodel = deps.Transformer.new("tiny", cfg, rng)
	local ids = { 3, 7, 1, 9, 12, 4, 6, 8, 2, 5, 11, 10 }
	local logits = tmodel.forward(tmodel, ids)
	check("logits shape", logits.shape[1] == 12 and logits.shape[2] == 16)

	local ce = deps.CrossEntropy.crossEntropy(logits, { 7, 1, 9, 12, 4, 6, 8, 2, 5, 11, 10, 3 })
	Tensor.backward(ce, false)

	local p0 = tmodel.lmHeadW
	check("lmHeadW has grad", p0.grad ~= nil and #p0.grad == #p0.data)

	-- finite difference on one weight element
	local wi = 5
	local eps = 1e-5
	local g0 = p0.grad[wi]
	local base = ce.data[1]
	local d0 = p0.data[wi]
	p0.data[wi] = d0 + eps
	local logits2 = tmodel.forward(tmodel, ids)
	local ce2 = deps.CrossEntropy.crossEntropy(logits2, { 7, 1, 9, 12, 4, 6, 8, 2, 5, 11, 10, 3 })
	local fd = (ce2.data[1] - base) / eps
	p0.data[wi] = d0
	check("gradient agreement (lmHeadW)", math.abs(fd - g0) < 1e-2 * math.abs(g0) + 1e-4)

	-- all params gathered
	local allParams = deps.Module.collectParams(tmodel)
	check("params collected", #allParams > 0)
end

do -- KV-cache generate matches the recompute path, and eval detaches grads
	local cfg = {
		C = 8, numBlocks = 2, nbHeads = 2, kvHeads = 1, headDim = 8,
		ffnHidden = 16, vocab = 16, maxSeq = 16, tieEmbeds = true,
		dropoutP = 0.0, initType = "rezero",
	}
	local tmodel = deps.Transformer.new("kc", cfg, rng)
	local seed = { 3, 7, 1, 9 }
	-- topK=1 argmax: deterministic, so tokens must match exactly between paths.
	local cached = tmodel.generate(tmodel, seed, 6, { topK = 1 })
	local recalc = tmodel.generate(tmodel, seed, 6, { topK = 1, cache = false })
	check("cached generate == recompute", table.concat(cached, ",") == table.concat(recalc, ","))

	local lv = tmodel.forward(tmodel, seed) -- training mode, grads on
	tmodel:eval()
	check("params detached in eval", not tmodel.lmHeadW.requiresGrad)
	local lt = tmodel.forward(tmodel, seed) -- eval mode, SIMD fast path
	tmodel:train()
	check("params restored on train", tmodel.lmHeadW.requiresGrad)
	local maxd = 0
	for i = 1, #lv.data do
		local d = math.abs(lv.data[i] - lt.data[i])
		if d > maxd then maxd = d end
	end
	check("eval logits agree with train logits", maxd < 1e-4)
end

do -- ReZero variant runs
	local cfg = {
		C = 8, numBlocks = 2, nbHeads = 1, kvHeads = 1, headDim = 8,
		ffnHidden = 16, vocab = 16, maxSeq = 16, tieEmbeds = true,
		dropoutP = 0.0, initType = "rezero",
	}
	local tmodel = deps.Transformer.new("rz", cfg, rng)
	local logits = tmodel.forward(tmodel, { 3, 7, 1, 9 })
	check("rezero logits shape", logits.shape[1] == 4 and logits.shape[2] == 16)
end

if failures == 0 then
	print("ALL LAYER SMOKE TESTS PASSED")
else
	error(string.format("%d LAYER SMOKE TESTS FAILED", failures))
end
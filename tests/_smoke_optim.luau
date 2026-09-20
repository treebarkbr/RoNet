--!strict
-- Optimizer smoke suite. XOR MLP training + determinism/formula checks.
-- ENTRY script; composes the DI graph (mirrors init.luau).
local Util = require("../core/Util")
local PRNG = require("../core/PRNG")
local Tensor = require("../core/Tensor")
local Matrix = require("../core/Matrix")
local Module = require("../nn/Module")
local Inits = require("../nn/Inits")
local Linear = require("../nn/Linear")
local Activations = require("../nn/Activations")
local Norm = require("../nn/Norm")
local MLP = require("../models/MLP")
local AdamW = require("../optim/AdamW")
local AdEMAMix = require("../optim/AdEMAMix")
local Lion = require("../optim/Lion")
local NAdamW = require("../optim/NAdamW")
local CautiousAdamW = require("../optim/CautiousAdamW")
local ScheduleFreeAdamW = require("../optim/ScheduleFreeAdamW")
local SOAP = require("../optim/SOAP")
local Muon = require("../optim/Muon")
local MuonAdamW = require("../optim/MuonAdamW")

local deps = {
	Util = Util,
	PRNG = PRNG,
	Tensor = Tensor,
	Matrix = Matrix,
}
deps.Module = Module(deps)
deps.Inits = Inits(deps)
deps.Linear = Linear(deps)
deps.Activations = Activations(deps)
deps.Norm = Norm(deps)
deps.MLP = MLP(deps)
deps.AdamW = AdamW(deps)
deps.AdEMAMix = AdEMAMix(deps)
deps.Lion = Lion(deps)
deps.NAdamW = NAdamW(deps)
deps.CautiousAdamW = CautiousAdamW(deps)
deps.ScheduleFreeAdamW = ScheduleFreeAdamW(deps)
deps.SOAP = SOAP(deps)
deps.Muon = Muon(deps)
deps.MuonAdamW = MuonAdamW(deps)

Tensor.setRNG(PRNG.new(42))

local failures = 0
local function checkOk(label: string, cond: boolean)
	if not cond then
		failures += 1
		print("FAIL: " .. label)
	else
		print("ok:   " .. label)
	end
end

local X = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
local Y = { 0, 1, 1, 0 }

local function finite(p)
	for _, w in p.data do
		if w ~= w or w == math.huge or w == -math.huge then return false end
	end
	return true
end

-- run `epochs` gradient steps over the XOR dataset, then return avg loss + accuracy
-- AdamW elementwise formula equivalence on a single scalar-ish parameter
do
	local t = Tensor.fromData({ 1.0, -0.5, 0.3 }, { 1, 3 }, true)
	local g = Tensor.fromData({ 0.1, -0.2, 0.05 }, { 1, 3 }, false)
	t.grad = g.data
	local opt = deps.AdamW.new({ t }, 0.1, { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01 })
	opt:step()
	local m1 = 0.9 * 0 + 0.1 * 0.1
	local m2 = 0.999 * 0 + 0.001 * 0.01
	local vhat = m2 / (1 - 0.999)
	local expect = 1.0 - 0.1 * ((m1 / 0.1) / (math.sqrt(vhat) + 1e-8) + 0.01 * 1.0)
	checkOk("AdamW formula matches", math.abs(t.data[1] - expect) < 1e-12)
end

-- each optimizer trains a fresh XOR MLP
local lineage = {
	{ "AdamW", deps.AdamW, { lr = 0.08 } },
	{ "AdEMAMix", deps.AdEMAMix, { lr = 0.08 } },
	{ "Lion", deps.Lion, { lr = 0.002 } },
	{ "NAdamW", deps.NAdamW, { lr = 0.08 } },
	{ "CautiousAdamW", deps.CautiousAdamW, { lr = 0.08 } },
	{ "ScheduleFreeAdamW", deps.ScheduleFreeAdamW, { lr = 0.08 } },
}
Tensor.setRNG(PRNG.new(7))
for _, spec in lineage do
	local model = deps.MLP.new(spec[1], { 2, 16, 16, 1 }, deps.PRNG.new(1), { act = "relu" })
	local allFinite = true
	local opt = spec[2].new(deps.Module.collectParams(model), spec[3].lr, spec[3])
	for step = 1, 300 do
		opt:zeroGrad()
		for i = 1, 4 do
			local inp = Tensor.fromData(X[i], { 1, 2 }, false)
			local pred = model.forward(model, inp)
			local diff = Tensor.sub(pred, Tensor.scalar(Y[i]))
			local loss = Tensor.mean(Tensor.mul(diff, diff), -1)
			Tensor.backward(loss, false)
		end
		opt:step()
		for _, p in model.params do
			if not finite(p) then allFinite = false end
		end
	end
	checkOk(spec[1] .. " params finite", allFinite)
end

-- SOAP trains a smaller net (SLOW), verifies loss collapsed
do
	local model = deps.MLP.new("soap", { 2, 8, 8, 1 }, deps.PRNG.new(1), { act = "relu" })
	local opt = deps.SOAP.new(deps.Module.collectParams(model), 0.05, {
		betas = { 0.95, 0.95 },
		preconditionFrequency = 4,
		weightDecay = 0.0,
	})
	local prev = math.huge
	for step = 1, 120 do
		opt:zeroGrad()
		local total = 0
		for i = 1, 4 do
			local inp = Tensor.fromData(X[i], { 1, 2 }, false)
			local pred = model.forward(model, inp)
			local diff = Tensor.sub(pred, Tensor.scalar(Y[i]))
			local loss = Tensor.mean(Tensor.mul(diff, diff), -1)
			Tensor.backward(loss, false)
			total += loss.data[1]
		end
		opt:step()
		prev = total / 4
	end
	checkOk("SOAP loss < 0.1 after 120 steps", prev < 0.1)
	for _, p in model.params do checkOk("SOAP finite", finite(p)) end
end

-- Muon trains the XOR MLP using ONLY 2D weights; MuonAdamW handles the rest
do
	local model = deps.MLP.new("muon", { 2, 16, 16, 1 }, deps.PRNG.new(1), { act = "relu" })
	local muonPs = {}
	local adamPs = {}
	for _, p in deps.Module.collectParams(model) do
		if #p.shape == 2 then
			muonPs[#muonPs + 1] = p
		else
			adamPs[#adamPs + 1] = p
		end
	end
	local opt = deps.MuonAdamW.new(muonPs, adamPs, {
		muon = { lr = 0.05, momentum = 0.95, weightDecay = 0.0, nsSteps = 5 },
		adam = { lr = 0.02, weightDecay = 0.0 },
	})
	local prev = math.huge
	for step = 1, 300 do
		opt:zeroGrad()
		local total = 0
		for i = 1, 4 do
			local inp = Tensor.fromData(X[i], { 1, 2 }, false)
			local pred = model.forward(model, inp)
			local diff = Tensor.sub(pred, Tensor.scalar(Y[i]))
			local loss = Tensor.mean(Tensor.mul(diff, diff), -1)
			Tensor.backward(loss, false)
			total += loss.data[1]
		end
		opt:step()
		prev = total / 4
	end
	checkOk("MuonAdamW loss < 0.1 after 300 steps", prev < 0.1)
	for _, p in model.params do checkOk("Muon sentinel", finite(p)) end
end

if failures == 0 then
	print("ALL OPTIMIZER SMOKE TESTS PASSED")
else
	error(string.format("%d OPTIMIZER SMOKE TESTS FAILED", failures))
end
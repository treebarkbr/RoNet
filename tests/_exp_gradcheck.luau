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
local MLP = require("../models/MLP")
local TransformerBlock = require("../models/TransformerBlock")
local Transformer = require("../models/Transformer")
local CrossEntropy = require("../loss/CrossEntropy")

deps = {
	Util = Util,
	PRNG = PRNG,
	Tensor = Tensor,
}
deps.Module = Module(deps)
deps.Inits = Inits(deps)
for _, name in { "Activations", "Norm", "Dropout", "Embedding", "RoPE", "Attention", "FFN", "Sequential", "Linear", "MLP", "TransformerBlock", "Transformer", "CrossEntropy" } do
	local path = if name == "CrossEntropy" then "../loss/CrossEntropy"
		elseif name == "MLP" or name == "TransformerBlock" or name == "Transformer" then "../models/" .. name
		else "../nn/" .. name
	deps[name] = require(path)(deps)
end
local Rng = PRNG
local rng = PRNG.new(123)
local Tensor = Tensor
local rng = Rng.new(123)

local function fill(t, fn)
	for i = 1, #t.data do t.data[i] = fn(i) end
end

local function fdGrad(x, lossFn, scale)
	-- central difference numeric grad of lossFn(x) w.r.t. x.data[d], scale ~ |x| element
	local g = table.create(#x.data)
	for d = 1, #x.data do
		local h = 1e-4 * math.max(math.abs(x.data[d]), 0.01)
		local old = x.data[d]
		x.data[d] = old + h
		local l1 = lossFn()
		x.data[d] = old - h
		local l0 = lossFn()
		x.data[d] = old
		g[d] = (l1 - l0) / (2 * h)
	end
	return g
end

local function maxDiff(a, b)
	local m = 0
	for i = 1, math.min(#a, #b) do
		local d = math.abs(a[i] - b[i])
		if d > m then m = d end
	end
	return m
end

local function report(name, a, b)
	local m = maxDiff(a, b)
	print(string.format("%-22s max|diff|=%.3e %s", name, m, if m < 3e-4 then "OK" else "BAD"))
end

do -- rmsNorm grads
	local x = Tensor.fromData({ 1.2, -0.7, 0.4, 2.3, -1.1, 0.9, 0.1, 1.7 }, { 2, 4 }, true)
	local gamma = Tensor.fromData({ 1.0, 0.8, 1.2, 0.5 }, { 4 }, true)
	local eps = 1e-5
	local out = Tensor.rmsNorm(x, gamma, eps)
	local loss = Tensor.mean(Tensor.mul(out, out), -1)
	Tensor.backward(loss, false)
	local gx = table.clone(x.grad)
	local ggamma = table.clone(gamma.grad)
	local lossFn1 = function()
		local o = Tensor.rmsNorm(x, gamma, eps)
		local l = Tensor.mean(Tensor.mul(o, o), -1)
		return l.data[1]
	end
	local fx = fdGrad(x, lossFn1)
	local fgamma = fdGrad(gamma, lossFn1)
	report("rms dx", gx, fx)
	report("rms dgamma", ggamma, fgamma)
end

do -- layerNorm grads
	local x = Tensor.fromData({ 0.5, 1.5, -2.1, 3.3, -0.2, 1.1, 0.0, 2.9 }, { 2, 4 }, true)
	local gamma = Tensor.fromData({ 1.1, 0.7, 1.3, 0.9 }, { 4 }, true)
	local beta = Tensor.fromData({ 0.1, -0.2, 0.3, 0.0 }, { 4 }, true)
	local eps = 1e-5
	local out = Tensor.layerNorm(x, gamma, beta, eps)
	local loss = Tensor.mean(Tensor.mul(out, out), -1)
	Tensor.backward(loss, false)
	local gx = table.clone(x.grad)
	local ggamma = table.clone(gamma.grad)
	local gbeta = table.clone(beta.grad)
	local lossFn2 = function()
		local o = Tensor.layerNorm(x, gamma, beta, eps)
		local l = Tensor.mean(Tensor.mul(o, o), -1)
		return l.data[1]
	end
	report("layer dx", gx, fdGrad(x, lossFn2))
	report("layer dgamma", ggamma, fdGrad(gamma, lossFn2))
	report("layer dbeta", gbeta, fdGrad(beta, lossFn2))
end

do -- rotary grads
	local nd, T, hd = 2, 3, 8
	local size = nd * T * hd
	local x = Tensor.fromData(table.create(size, 0), { nd, T, hd }, true)
	local rng2 = PRNG.new(7)
	for i = 1, size do x.data[i] = rng2:nextFloat() * 2 - 1 end
	local half = hd // 2
	local seq = 4
	local cos = table.create(seq * half)
	local sin = table.create(seq * half)
	for p = 0, seq - 1 do
		for k = 1, half do
			local th = p / (10000 ^ ((2 * (k - 1)) / hd))
			cos[p * half + k] = math.cos(th)
			sin[p * half + k] = math.sin(th)
		end
	end
	local cosT = Tensor.fromData(cos, { seq, half }, false)
	local sinT = Tensor.fromData(sin, { seq, half }, false)
	local out = Tensor.rotary(x, cosT, sinT)
	local loss = Tensor.mean(Tensor.mul(out, out), -1)
	Tensor.backward(loss, false)
	local gx = table.clone(x.grad)
	local lossFn3 = function()
		local o = Tensor.rotary(x, cosT, sinT)
		local l = Tensor.mean(Tensor.mul(o, o), -1)
		return l.data[1]
	end
	report("rotary dx", gx, fdGrad(x, lossFn3))
end

do -- softmax (fast last-dim path) grads
	local x = Tensor.fromData({ 0.3, 1.2, -0.5, 0.8, 2.1, -1.4, 0.0, 0.7 }, { 2, 4 }, true)
	local out = Tensor.softmax(x, 2)
	local loss = Tensor.mean(Tensor.mul(out, out), -1)
	Tensor.backward(loss, false)
	local gx = table.clone(x.grad)
	local lossFn4 = function()
		local o = Tensor.softmax(x, 2)
		local l = Tensor.mean(Tensor.mul(o, o), -1)
		return l.data[1]
	end
	report("softmax2d dx", gx, fdGrad(x, lossFn4))
end

do -- rmspow softmax fast path grads
	local x = Tensor.fromData({ 0.3, 1.2, -0.5, 0.8, 2.1, -1.4, 0.0, 0.7 }, { 4, 2 }, true)
	local out = Tensor.softmax(x, 2)
	local loss = Tensor.mean(Tensor.mul(out, out), -1)
	Tensor.backward(loss, false)
	local gx = table.clone(x.grad)
	local lossFn5 = function()
		local o = Tensor.softmax(x, 2)
		local l = Tensor.mean(Tensor.mul(o, o), -1)
		return l.data[1]
	end
	report("softmax last-d dx", gx, fdGrad(x, lossFn5))
end
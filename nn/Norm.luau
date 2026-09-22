--!strict
-- Normalization layers. All operate over the LAST dim.
-- RMSNorm (Zhang & Sennrich 2019) and LayerNorm.
return function(deps: any)
	local Tensor = deps.Tensor
	local Inits = deps.Inits

	local function appendDim(shape: { number }, v: number): { number }
		local s = table.create(#shape + 1)
		for i = 1, #shape do s[i] = shape[i] end
		s[#shape + 1] = v
		return s
	end

	-- Ones-shaped gamma/beta broadcasting target: same ndims as x, ones except last.
	local function broadcastShape(shape: { number }): { number }
		local s = table.create(#shape)
		for i = 1, #shape do s[i] = 1 end
		s[#shape] = shape[#shape]
		return s
	end

	local Norm = {}
	Norm.__index = Norm

	-- op: "rms" | "layer"; eps default 1e-6 (rms) / 1e-5 (layer).
	function Norm.new(name: string, ndim: number, rng: any, op: string?, eps: number?)
		local self = setmetatable({ name = name, n = ndim, op = op or "rms", eps = eps or 1e-6 }, Norm)
		self.gamma = Tensor.fromData(Inits.ones({ ndim }), { ndim }, true)
		self.gamma.requiresGrad = true
		if self.op == "layer" then
			self.beta = Tensor.fromData(Inits.zeros({ ndim }), { ndim }, true)
			self.beta.requiresGrad = true
		end
		self.params = if self.beta then { self.gamma, self.beta } else { self.gamma }
		return self
	end

	-- Normalize x over the last dim (RMSNorm by default; LayerNorm if op is "layer").
	function Norm.forward(self: any, x: any): any
		local nd = #x.shape
		local lastDim = nd
		local xShape = Tensor.shapeOf(x)
		local gammaR = Tensor.reshape(self.gamma, broadcastShape(xShape))
		if self.op == "rms" then
			local x2 = Tensor.mul(x, x)
			local ms = Tensor.mean(x2, lastDim)
			local rms = Tensor.reshape(ms, appendDim(Tensor.shapeOf(ms), 1))
			local inv = Tensor.pow(Tensor.addScalar(rms, self.eps), -0.5)
			return Tensor.mul(Tensor.mul(x, inv), gammaR)
		end
		-- layer norm
		local mu = Tensor.mean(x, lastDim)
		local muB = Tensor.reshape(mu, appendDim(Tensor.shapeOf(mu), 1))
		local res = Tensor.sub(x, muB)
		local x2 = Tensor.mul(res, res)
		local varr = Tensor.mean(x2, lastDim)
		local iv = Tensor.pow(Tensor.addScalar(Tensor.reshape(varr, appendDim(Tensor.shapeOf(varr), 1)), self.eps), -0.5)
		local out = Tensor.mul(Tensor.mul(res, iv), gammaR)
		if self.beta then
			out = Tensor.add(out, Tensor.reshape(self.beta, broadcastShape(xShape)))
		end
		return out
	end

	return Norm
end
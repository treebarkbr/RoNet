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
		if self.op == "rms" then
			return Tensor.rmsNorm(x, self.gamma, self.eps)
		end
		return Tensor.layerNorm(x, self.gamma, self.beta, self.eps)
	end

	return Norm
end
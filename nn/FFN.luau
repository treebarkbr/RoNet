--!strict
-- SwiGLU feed-forward block: FFN(x) = W3(silu(W1 x) .* (W2 x)).
return function(deps: any)
	local Tensor = deps.Tensor
	local Inits = deps.Inits

	local FFN = {}
	FFN.__index = FFN

	function FFN.new(name: string, C: number, hidden: number, rng: any)
		local self = setmetatable({ name = name, C = C, hidden = hidden }, FFN)
		self.W1 = Tensor.fromData(Inits.glorotUniform({ C, hidden }, rng), { C, hidden }, true)
		self.W2 = Tensor.fromData(Inits.glorotUniform({ C, hidden }, rng), { C, hidden }, true)
		self.W3 = Tensor.fromData(Inits.kaimingUniform({ hidden, C }, rng), { hidden, C }, true)
		self.W1.requiresGrad = true
		self.W2.requiresGrad = true
		self.W3.requiresGrad = true
		self.params = { self.W1, self.W2, self.W3 }
		return self
	end

	function FFN.forward(self: any, x: any): any
		local nd = #x.shape
		local worked = x
		if nd > 2 then
			local n = 1
			for i = 1, nd - 1 do n *= x.shape[i] end
			worked = Tensor.reshape(x, { n, x.shape[nd] })
		end
		local g = Tensor.matmul(worked, self.W1) -- [N, C] @ [C, hidden]
		g = Tensor.silu(g)
		local u = Tensor.matmul(worked, self.W2)
		local h = Tensor.mul(g, u)
		local out = Tensor.matmul(h, self.W3) -- [N, hidden] @ [hidden, C]
		if nd > 2 then
			local outShape = table.create(nd)
			for i = 1, nd - 1 do outShape[i] = x.shape[i] end
			outShape[nd] = self.C
			out = Tensor.reshape(out, outShape)
		end
		return out
	end

	return FFN
end
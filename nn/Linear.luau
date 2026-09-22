--!strict
-- Fully-connected layer. W: [out, in], b: [out].
-- forward(x): y = x @ W^T + b   (x: [..., in] -> [..., out])
return function(deps: any)
	local Tensor = deps.Tensor
	local Inits = deps.Inits

	local Linear = {}
	Linear.__index = Linear

	-- Create a fully-connected layer with Glorot-initialized W and zero b.
	function Linear.new(name: string, fanIn: number, fanOut: number, rng: any)
		local self = setmetatable({ name = name, input = fanIn, out = fanOut }, Linear)
		self.W = Tensor.fromData(Inits.glorotUniform({ fanOut, fanIn }, rng), { fanOut, fanIn }, true)
		self.W.requiresGrad = true
		self.b = Tensor.fromData(Inits.zeros({ fanOut }), { fanOut }, true)
		self.b.requiresGrad = true
		self.params = { self.W, self.b }
		return self
	end

	-- Forward pass: y = x @ W^T + b. Reshapes automatically for >2-D leading dims.
	function Linear.forward(self: any, x: any): any
		local nd = #x.shape
		local worked = x
		if nd > 2 then
			local n = 1
			for i = 1, nd - 1 do n *= x.shape[i] end
			worked = Tensor.reshape(x, { n, x.shape[nd] })
		end
		local out = Tensor.matmul(worked, Tensor.transpose2d(self.W)) -- [N, in] @ [in, out]
		out = Tensor.add(out, self.b)
		if nd > 2 then
			local outShape = table.create(nd)
			for i = 1, nd - 1 do outShape[i] = x.shape[i] end
			outShape[nd] = self.out
			out = Tensor.reshape(out, outShape)
		end
		return out
	end

	return Linear
end
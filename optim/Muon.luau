--!strict
-- Muon (Jordan & Keller, Dec 2024; Moonlight: arXiv:2502.16982). SGD-momentum
-- post-processed by Newton-Schulz orthogonalization, scaled by
-- max(1, rows/cols)^0.5. Use ONLY for 2D hidden matrices; feed biases, scalar
-- gains, embeddings and the LM head to a companion AdamW (see MuonAdamW).
-- Requires deps.Matrix (newtonSchulz).
-- Class: new(params: {Tensor}, lr: number, opts: {momentum?, weightDecay?, nsSteps?, nesterov?})
return function(deps: any): any
	local Matrix = deps.Matrix
	local Optimizer = {}
	Optimizer.__index = Optimizer

	function Optimizer.new(params: { any }, lr: number, opts: any?): any
		local o = opts or {}
		local self = setmetatable({
			params = params,
			lr = lr,
			momentum = o.momentum or 0.95,
			wd = o.weightDecay or 0.0,
			nsSteps = o.nsSteps or 5,
			nesterov = if o.nesterov == nil then true else o.nesterov,
			buf = {},
		}, Optimizer)
		for _, p in params do
			self.buf[p] = table.create(#p.data, 0)
		end
		return self
	end

	function Optimizer.step(self)
		for _, p in self.params do
			local ndim = #p.shape
			if ndim ~= 2 then continue end
			local g = p.grad
			if not g then continue end
			local m = p.shape[1]
			local n = p.shape[2]
			if #g ~= m * n then continue end
			local buf = self.buf[p]
			for i = 1, m * n do
				buf[i] = self.momentum * buf[i] + (1 - self.momentum) * g[i]
			end
			local update = Matrix.copy(buf)
			if self.nesterov then
				for i = 1, m * n do
					update[i] = g[i] + self.momentum * (buf[i] - g[i])
				end
			end
			local orth = Matrix.newtonSchulz(update, m, n, self.nsSteps)
			local scale = math.max(1, m / n) ^ 0.5
			local d = p.data
			for i = 1, m * n do
				d[i] = d[i] * (1 - self.lr * self.wd) - self.lr * scale * orth[i]
			end
		end
	end

	function Optimizer.zeroGrad(self)
		for _, p in self.params do
			if p.grad then
				for i = 1, #p.grad do p.grad[i] = 0 end
			end
		end
	end

	function Optimizer.setLr(self, lr: number)
		self.lr = lr
	end

	function Optimizer.getLr(self): number
		return self.lr
	end

	function Optimizer.paramCount(self): number
		return #self.params
	end

	function Optimizer.stateDict(self): any
		return { buf = self.buf }
	end

	function Optimizer.loadStateDict(self, st: any)
		self.buf = st.buf
	end

	return Optimizer
end
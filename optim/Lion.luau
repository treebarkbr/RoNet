--!strict
-- Lion (Chen et al., 2023; arXiv:2302.06675). Memory-efficient: sign update via
-- interleaved momentum. update = sign(b1*m + (1-b1)*g).
-- Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, weightDecay?})
return function(deps: any): any
	local Optimizer = {}
	Optimizer.__index = Optimizer

	function Optimizer.new(params: { any }, lr: number, opts: any?): any
		local o = opts or {}
		local self = setmetatable({
			params = params,
			lr = lr,
			b1 = o.beta1 or 0.9,
			b2 = o.beta2 or 0.99,
			wd = o.weightDecay or 0.0,
			t = 0,
			m = {},
		}, Optimizer)
		for _, p in params do
			self.m[p] = table.create(#p.data, 0)
		end
		return self
	end

	function Optimizer.step(self)
		local lr, b1, b2, wd = self.lr, self.b1, self.b2, self.wd
		for _, p in self.params do
			local g = p.grad
			if g then
				local d = p.data
				local m = self.m[p]
				for i = 1, #d do
					local c = b1 * m[i] + (1 - b1) * g[i]
					local sgn = 0
					if c > 0 then sgn = 1 elseif c < 0 then sgn = -1 end
					d[i] = d[i] - lr * (sgn + wd * d[i])
					m[i] = b2 * m[i] + (1 - b2) * g[i]
				end
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
		return { m = self.m }
	end

	function Optimizer.loadStateDict(self, st: any)
		self.m = st.m
	end

	return Optimizer
end
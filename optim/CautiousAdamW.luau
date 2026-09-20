--!strict
-- Cautious AdamW (Liu et al., 2024; arXiv:2411.16085). Elementwise mask on the
-- first moment: keep only components aligned with the raw gradient (mhat*g > 0),
-- scale the rest down by alpha. update = alpha*masked_mhat/(sqrt(vhat)+eps).
-- Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, alpha?, eps?, weightDecay?})
return function(deps: any): any
	local Optimizer = {}
	Optimizer.__index = Optimizer

	function Optimizer.new(params: { any }, lr: number, opts: any?): any
		local o = opts or {}
		local self = setmetatable({
			params = params,
			lr = lr,
			b1 = o.beta1 or 0.9,
			b2 = o.beta2 or 0.999,
			alpha = o.alpha or 1.0,
			eps = o.eps or 1e-8,
			wd = o.weightDecay or 0.01,
			t = 0,
			m = {},
			v = {},
		}, Optimizer)
		for _, p in params do
			self.m[p] = table.create(#p.data, 0)
			self.v[p] = table.create(#p.data, 0)
		end
		return self
	end

	function Optimizer.step(self)
		self.t += 1
		local t = self.t
		local bc1 = 1 - self.b1 ^ t
		local bc2 = 1 - self.b2 ^ t
		local lr, b1, b2, eps, wd, alpha = self.lr, self.b1, self.b2, self.eps, self.wd, self.alpha
		for _, p in self.params do
			local g = p.grad
			if g then
				local d = p.data
				local m = self.m[p]
				local v = self.v[p]
				for i = 1, #d do
					local gi = g[i]
					local mi = b1 * m[i] + (1 - b1) * gi
					m[i] = mi
					local vi = b2 * v[i] + (1 - b2) * gi * gi
					v[i] = vi
					local mhat = mi / bc1
					local vhat = vi / bc2
					local mask = 0
					if mhat * gi > 0 then mask = 1 end
					local cm = mask * mhat + (1 - mask) * alpha * mhat
					d[i] = d[i] - lr * (cm / (math.sqrt(vhat) + eps) + wd * d[i])
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
		return { t = self.t, m = self.m, v = self.v }
	end

	function Optimizer.loadStateDict(self, st: any)
		self.t = st.t
		self.m = st.m
		self.v = st.v
	end

	return Optimizer
end
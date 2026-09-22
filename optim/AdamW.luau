--!strict
-- AdamW (Loshchilov & Hutter, ICLR 2019; arXiv:1711.05101).
-- Decoupled weight decay. Bias-corrected m and v per step.
-- Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, eps?, weightDecay?})
-- API: step(), zeroGrad(), setLr(lr), getLr(), stateDict(), loadStateDict(state), paramCount()
return function(deps: any): any
	local Optimizer = {}
	Optimizer.__index = Optimizer

	-- Create the optimizer over `params` (Module.collectParams(model)) at a base learning rate.
	function Optimizer.new(params: { any }, lr: number, opts: any?): any
		local o = opts or {}
		local self = setmetatable({
			params = params,
			lr = lr,
			b1 = o.beta1 or 0.9,
			b2 = o.beta2 or 0.999,
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

	-- Apply one update from the current parameter gradients (call Tensor.backward() first).
	function Optimizer.step(self)
		self.t += 1
		local s1 = 1 - self.b1 ^ self.t
		local s2 = 1 - self.b2 ^ self.t
		local lr = self.lr
		local b1, b2, eps, wd = self.b1, self.b2, self.eps, self.wd
		for _, p in self.params do
			local g = p.grad
			if g then
				local d = p.data
				local m = self.m[p]
				local v = self.v[p]
				for i = 1, #d do
					local gi = g[i]
					m[i] = b1 * m[i] + (1 - b1) * gi
					local vi = b2 * v[i] + (1 - b2) * gi * gi
					v[i] = vi
					d[i] = d[i] - lr * (m[i] / s1 / (math.sqrt(vi / s2) + eps) + wd * d[i])
				end
			end
		end
	end

	-- Zero every parameter gradient buffer.
	function Optimizer.zeroGrad(self)
		for _, p in self.params do
			if p.grad then
				for i = 1, #p.grad do p.grad[i] = 0 end
			end
		end
	end

	-- Set the current learning rate.
	function Optimizer.setLr(self, lr: number)
		self.lr = lr
	end

	-- Return the current learning rate.
	function Optimizer.getLr(self): number
		return self.lr
	end

	-- Number of tracked parameters.
	function Optimizer.paramCount(self): number
		return #self.params
	end

	-- Export optimizer state (moments, step counter) for checkpointing.
	function Optimizer.stateDict(self): any
		return { t = self.t, m = self.m, v = self.v }
	end

	-- Restore optimizer state from a stateDict.
	function Optimizer.loadStateDict(self, st: any)
		self.t = st.t
		self.m = st.m
		self.v = st.v
	end

	return Optimizer
end
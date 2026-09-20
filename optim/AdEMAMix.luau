--!strict
-- AdEMAMix (Pagliardini et al., 2024; arXiv:2409.03137).
-- Three moments: m1 (fast), m2 (slow, squared), m3 (very slow, same LR-scaled
-- by alpha). Optional beta3 temporal ramp toward a final beta3 over t_final
-- steps: beta3(t) = 1 - 1/(1 + alpha_s * t/t_final) with the paper's defaults
-- T_final = 10*k steps, s = 20 (alpha_s = s/log(t));
-- Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, beta3?, alpha?, tFinal?, alphaS?, eps?, weightDecay?})
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
			b3 = o.beta3 or 0.9999,
			alpha = o.alpha or 10,
			alphaS = o.alphaS or 20,
			tFinal = o.tFinal or 0,
			eps = o.eps or 1e-8,
			wd = o.weightDecay or 0.01,
			t = 0,
			m1 = {},
			m2 = {},
			m3 = {},
			v = {},
		}, Optimizer)
		for _, p in params do
			local n = #p.data
			self.m1[p] = table.create(n, 0)
			self.m2[p] = table.create(n, 0)
			self.m3[p] = table.create(n, 0)
			self.v[p] = table.create(n, 0)
		end
		return self
	end

	function Optimizer.beta3(self, t: number): number
		if self.tFinal > 0 and self.alphaS > 0 then
			local tau = 1 - 1 / (1 + self.alphaS * t / self.tFinal)
			return 1 - 1 / (1 + tau)
		end
		return self.b3
	end

	function Optimizer.step(self)
		self.t += 1
		local t = self.t
		local s1 = 1 - self.b1 ^ t
		local s2 = 1 - self.b2 ^ t
		local b3 = self:beta3(t)
		local lr, b1, b2, eps, wd, alpha = self.lr, self.b1, self.b2, self.eps, self.wd, self.alpha
		for _, p in self.params do
			local g = p.grad
			if g then
				local d = p.data
				local m1 = self.m1[p]
				local m3 = self.m3[p]
				local v = self.v[p]
				for i = 1, #d do
					local gi = g[i]
					local m1i = b1 * m1[i] + (1 - b1) * gi
					m1[i] = m1i
					local vi = b2 * v[i] + (1 - b2) * gi * gi
					v[i] = vi
					local m3i = b3 * m3[i] + (1 - b3) * gi
					m3[i] = m3i
					local mhat = m1i / s1 + alpha * m3i
					d[i] = d[i] - lr * (mhat / (math.sqrt(vi / s2) + eps) + wd * d[i])
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
		return { t = self.t, a = self.alpha, m1 = self.m1, m2 = self.m2, m3 = self.m3, v = self.v }
	end

	function Optimizer.loadStateDict(self, st: any)
		self.t = st.t
		self.alpha = st.a
		self.m1 = st.m1
		self.m2 = st.m2
		self.m3 = st.m3
		self.v = st.v
	end

	return Optimizer
end
--!strict
-- AdEMAMix (Pagliardini, Ablin & Grangier, 2024; arXiv:2409.03137).
-- Two gradient EMAs: m1 (fast, beta1) and m2 (slow, beta3), plus the second
-- moment v. The update mixes bias-corrected m1 with alpha * m2 (no bias
-- correction on m2, per the paper). Optional warmup schedulers over tFinal
-- steps, faithful to Section 3 / Algorithm 1 with T_alpha = T_beta3 = tFinal:
--   alpha(t) = min(t * alpha / tFinal, alpha)
--   beta3(t) = min(exp(ln(betaStart) * ln(beta3) /
--               ((1 - t/tFinal) * ln(beta3) + (t/tFinal) * ln(betaStart))), beta3)
-- with betaStart defaulting to beta1; keep tFinal = 0 to disable scheduling.
-- Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, beta3?, alpha?, tFinal?, betaStart?, eps?, weightDecay?})
return function(deps: any): any
	local Optimizer = {}
	Optimizer.__index = Optimizer

	local math_min = math.min
	local math_exp = math.exp
	local math_log = math.log

	function Optimizer.new(params: { any }, lr: number, opts: any?): any
		local o = opts or {}
		local self = setmetatable({
			params = params,
			lr = lr,
			b1 = o.beta1 or 0.9,
			b2 = o.beta2 or 0.999,
			b3 = o.beta3 or 0.9999,
			alpha = o.alpha or 10,
			tFinal = o.tFinal or 0,
			betaStart = o.betaStart or (o.beta1 or 0.9),
			eps = o.eps or 1e-8,
			wd = o.weightDecay or 0.01,
			t = 0,
			m1 = {},
			m2 = {},
			v = {},
		}, Optimizer)
		for _, p in params do
			local n = #p.data
			self.m1[p] = table.create(n, 0)
			self.m2[p] = table.create(n, 0)
			self.v[p] = table.create(n, 0)
		end
		return self
	end

	-- alpha(t): linear warmup from 0 to the final alpha over tFinal steps.
	function Optimizer.alphaAt(self: any, t: number): number
		if self.tFinal > 0 then
			return math_min(t * self.alpha / self.tFinal, self.alpha)
		end
		return self.alpha
	end

	-- beta3(t): schedules the EMA half-life linearly (paper App. A.1), from
	-- betaStart at t=0 up to beta3 at t=tFinal.
	function Optimizer.beta3(self: any, t: number): number
		if self.tFinal <= 0 then
			return self.b3
		end
		local u = math_min(t / self.tFinal, 1.0)
		local lnStart = math_log(self.betaStart)
		local lnEnd = math_log(self.b3)
		local denom = (1 - u) * lnEnd + u * lnStart
		return math_min(math_exp(lnStart * lnEnd / denom), self.b3)
	end

	function Optimizer.step(self)
		self.t += 1
		local t = self.t
		local s1 = 1 - self.b1 ^ t
		local s2 = 1 - self.b2 ^ t
		local a = self:alphaAt(t)
		local b3 = self:beta3(t)
		local lr, b1, b2, eps, wd = self.lr, self.b1, self.b2, self.eps, self.wd
		for _, p in self.params do
			local g = p.grad
			if g then
				local d = p.data
				local m1 = self.m1[p]
				local m2 = self.m2[p]
				local v = self.v[p]
				for i = 1, #d do
					local gi = g[i]
					local m1i = b1 * m1[i] + (1 - b1) * gi
					m1[i] = m1i
					local vi = b2 * v[i] + (1 - b2) * gi * gi
					v[i] = vi
					local m2i = b3 * m2[i] + (1 - b3) * gi
					m2[i] = m2i
					local mhat = m1i / s1 + a * m2i
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
		return { t = self.t, a = self.alpha, m1 = self.m1, m2 = self.m2, v = self.v }
	end

	function Optimizer.loadStateDict(self, st: any)
		self.t = st.t
		self.alpha = st.a
		self.m1 = st.m1
		self.m2 = st.m2
		self.v = st.v
	end

	return Optimizer
end
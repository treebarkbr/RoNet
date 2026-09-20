--!strict
-- Schedule-Free AdamW (Defazio & Mishchenko, 2024; arXiv:2405.15682).
-- No learning-rate schedule needed: the iterate is blended toward the exponential
-- average z, so lr can stay constant. Must call train()/eval() around phases.
-- Faithful to the official facebookresearch/schedule_free implementation.
-- Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, eps?, weightDecay?,
--        warmupSteps?, r?, weightLrPower?, innerMomentum?})
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
			eps = o.eps or 1e-8,
			wd = o.weightDecay or 0.0,
			warmupSteps = o.warmupSteps or 0,
			r = o.r or 0.0,
			weightLrPower = o.weightLrPower or 1.5,
			innerMomentum = o.innerMomentum or 0.0,
			k = 0,
			weightSum = 0.0,
			lrMax = -1.0,
			trainMode = true,
			z = {},
			v = {},
			expAvg = {},
		}, Optimizer)
		self.bc1 = 1.0
		for _, p in params do
			self.z[p] = table.clone(p.data)
			self.v[p] = table.create(#p.data, 0)
			self.expAvg[p] = nil
			if self.innerMomentum ~= 0 then
				self.expAvg[p] = table.create(#p.data, 0)
			end
		end
		return self
	end

	function Optimizer.train(self)
		if not self.trainMode then
			local b1 = self.b1
			for _, p in self.params do
				local z = self.z[p]
				local d = p.data
				for i = 1, #d do
					d[i] = d[i] + (1 - b1) * (z[i] - d[i])
				end
			end
			self.trainMode = true
		end
	end

	function Optimizer.eval(self)
		if self.trainMode then
			local inv = 1 - 1 / self.b1
			for _, p in self.params do
				local z = self.z[p]
				local d = p.data
				for i = 1, #d do
					d[i] = d[i] + inv * (z[i] - d[i])
				end
			end
			self.trainMode = false
		end
	end

	function Optimizer.step(self)
		if not self.trainMode then
			error("ScheduleFreeAdamW.step called while in eval mode; call train() first")
		end
		local k = self.k
		local sched = 1.0
		if self.warmupSteps > 0 and k < self.warmupSteps then
			sched = (k + 1) / self.warmupSteps
		end
		local bc2 = 1 - self.b2 ^ (k + 1)
		if self.innerMomentum ~= 0 then
			self.bc1 = 1 - self.innerMomentum ^ (k + 1)
		end
		local lr = self.lr * sched
		self.lrMax = math.max(lr, self.lrMax)
		local weight = (k + 1) ^ self.r * self.lrMax ^ self.weightLrPower
		self.weightSum += weight
		local ckp1 = weight / self.weightSum
		local b1, b2, eps, wd = self.b1, self.b2, self.eps, self.wd
		local im = self.innerMomentum
		local gradFactor = lr * (b1 * (1 - ckp1) - 1)
		for _, p in self.params do
			local g = p.grad
			if g then
				local d = p.data
				local z = self.z[p]
				local v = self.v[p]
				local ea = self.expAvg[p]
				for i = 1, #d do
					local gi = g[i]
					local vi = b2 * v[i] + (1 - b2) * gi * gi
					v[i] = vi
					local gn = gi / (math.sqrt(vi / bc2) + eps)
					if im ~= 0 then
						ea[i] = im * ea[i] + (1 - im) * gi
						gn = ea[i] / self.bc1 / (math.sqrt(vi / bc2) + eps)
					end
					if wd ~= 0 then
						gn += wd * d[i]
					end
					local y = d[i] + ckp1 * (z[i] - d[i])
					d[i] = y + gradFactor * gn
					z[i] -= lr * gn
				end
			end
		end
		self.k = k + 1
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
		return { k = self.k, w = self.weightSum, lm = self.lrMax, z = self.z, v = self.v }
	end

	function Optimizer.loadStateDict(self, st: any)
		self.k = st.k
		self.weightSum = st.w
		self.lrMax = st.lm
		self.z = st.z
		self.v = st.v
	end

	return Optimizer
end
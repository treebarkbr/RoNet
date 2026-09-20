--!strict
-- Learning-rate schedules. Each schedule is a pure function
-- `(step: number) -> factor` returning the multiplier on the base lr for that
-- step (step counts optimizer steps, starting at 1). Factory helpers:
--   Scheduler.constant()
--   Scheduler.warmupLinear(warmup, total, minFactor?)
--   Scheduler.warmupCosine(warmup, total, minFactor?)
--   Scheduler.linear(total, minFactor?)            (decay from 1 to minFactor)
--   Scheduler.warmupLinearCosine(cooldown, warmup, total, minFactor?)
--   Scheduler.forOptimizer(optimizer, schedule)    (held schedule, updates lr)
-- Written for speed: pure arithmetic, no per-step allocations beyond one number.
return function(deps: any): any
	local pi = math.pi
	local cos = math.cos

	local Scheduler = {}

	local function clamp01(v: number): number
		if v < 0 then return 0 end
		if v > 1 then return 1 end
		return v
	end

	function Scheduler.constant(): any
		return function(_step: number): number
			return 1
		end
	end

	-- Linear warmup for [1, warmup], then linear decay 1 -> minFactor over
	-- [warmup, total], clipped at minFactor (cooldown = total - warmup).
	function Scheduler.warmupLinear(warmup: number, total: number, minFactor: number?): any
		local cooldown = math.max(1, total - warmup)
		local mf = minFactor or 0.0
		return function(step: number): number
			if step < warmup then
				return step / warmup
			end
			local f = 1 - (step - warmup) / cooldown
			return math.max(mf, f)
		end
	end

	-- Linear warmup then cosine decay with a floor of minFactor.
	function Scheduler.warmupCosine(warmup: number, total: number, minFactor: number?): any
		local cooldown = math.max(1, total - warmup)
		local mf = minFactor or 0.1
		local amp = 1 - mf
		return function(step: number): number
			if step < warmup then
				return step / warmup
			end
			local t = clamp01((step - warmup) / cooldown)
			return mf + amp * 0.5 * (1 + cos(pi * t))
		end
	end

	-- Pure cosine decay from 1 to minFactor over total steps (no warmup).
	function Scheduler.cosine(total: number, minFactor: number?): any
		local mf = minFactor or 0.1
		local amp = 1 - mf
		return function(step: number): number
			local t = clamp01(step / math.max(1, total))
			return mf + amp * 0.5 * (1 + cos(pi * t))
		end
	end

	-- Inverse-square-root decay after optional warmup (transformer-style).
	-- factor = sqrt(warmup) / sqrt(step); = 1 at the warmup end, then decays.
	function Scheduler.invSqrt(warmup: number): any
		local warmRt = math.sqrt(warmup)
		return function(step: number): number
			local t = math.max(step, 1)
			if step < warmup then
				return t / warmup
			end
			return warmRt / math.sqrt(t)
		end
	end

	-- Attach a schedule to an optimizer: returns { step(), setBase(lr), getLr() }.
	-- step() applies schedule(stepIdx) to the base lr (captured at attach time,
	-- refreshable via setBase). Scheduled lr available as handle.lr.
	function Scheduler.forOptimizer(opt: any, schedule: any): any
		local h: any = { base = opt:getLr(), t = 0, lr = opt:getLr() }
		function h.setBase(self: any, lr: number)
			self.base = lr
		end
		h.t = 0
		function h.step(self: any): number
			self.t += 1
			self.lr = self.base * schedule(self.t)
			opt:setLr(self.lr)
			return self.lr
		end
		return h
	end

	return Scheduler
end
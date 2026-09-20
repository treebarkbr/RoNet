--!strict
-- Exponential moving average of parameter snapshots (Polyak averaging).
--   EMA.new(params, decay, opts:{warmupSteps?}) 
--   ema:update()    blend buffers toward current params (with warmup-scaled decay)
--   ema:copyBack()  overwrite model params with the averaged buffers
--   ema:stateDict() / ema:loadStateDict()   (for checkpointing)
-- Written for speed: single numeric pass per param, no allocations per update.
return function(deps: any): any
	local EMA = {}
	EMA.__index = EMA

	function EMA.new(params: { any }, decay: number, opts: any?): any
		local o = opts or {}
		local self = setmetatable({
			params = params,
			maxDecay = decay,
			warmup = o.warmupSteps or 0,
			step = 0,
			buf = {},
		}, EMA)
		for _, p in params do
			self.buf[p] = table.clone(p.data)
		end
		return self
	end

	-- Effective decay: ramps from ~1 towards maxDecay during warmup, so early
	-- snapshots are averaged broadly (hana-rhea recipe: min(decay, (1+t)/(t+2))).
	function EMA._decay(self: any): number
		local t = self.step
		if self.warmup > 0 and t < self.warmup then
			return math.min(self.maxDecay, (1 + t) / (t + 2))
		end
		return self.maxDecay
	end

	function EMA.update(self)
		self.step += 1
		local d = 1 - self:_decay()
		for _, p in self.params do
			local b = self.buf[p]
			local data = p.data
			local n = #data
			for i = 1, n do
				b[i] = b[i] + (data[i] - b[i]) * d
			end
		end
	end

	function EMA.copyBack(self)
		for _, p in self.params do
			local b = self.buf[p]
			local data = p.data
			for i = 1, #data do
				data[i] = b[i]
			end
		end
	end

	function EMA.stateDict(self): any
		local out: { any } = {}
		for i = 1, #self.params do
			out[i] = { shape = table.clone(self.params[i].shape), data = table.clone(self.buf[self.params[i]]) }
		end
		return { step = self.step, buffers = out }
	end

	function EMA.loadStateDict(self, st: any)
		self.step = st.step
		for i = 1, #self.params do
			local p = self.params[i]
			local b = self.buf[p]
			local src = st.buffers[i].data
			for j = 1, #b do b[j] = src[j] end
		end
	end

	return EMA
end
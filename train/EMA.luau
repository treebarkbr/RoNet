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

	-- Create an EMA tracker over params at the given max decay (Polyak averaging).
	-- opts.warmupSteps ramps the effective decay during the first steps.
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

	-- Blend the buffers one step toward the current param values.
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

	-- Overwrite the model params with the averaged buffers.
	function EMA.copyBack(self)
		for _, p in self.params do
			local b = self.buf[p]
			local data = p.data
			for i = 1, #data do
				data[i] = b[i]
			end
		end
	end

	-- Export averaged buffers (step + per-param {shape, data}) for checkpointing.
	function EMA.stateDict(self): any
		local out: { any } = {}
		for i = 1, #self.params do
			out[i] = { shape = table.clone(self.params[i].shape), data = table.clone(self.buf[self.params[i]]) }
		end
		return { step = self.step, buffers = out }
	end

	-- Restore averaged buffers from a stateDict produced by ema:stateDict().
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
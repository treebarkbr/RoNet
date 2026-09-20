--!strict
-- Composite optimizer following the modded-nanogpt / Moonlight production
-- recipe: 2D hidden matrices are optimized by Muon (Newton-Schulz removed
-- gradient post-processing) and every other parameter (scalars, biases,
-- embeddings, LM head) by AdamW at its own lr. Two-optimizer step keeps
-- gradients valid for both.
-- Class: new(muonParams: {Tensor}, adamParams: {Tensor}, opts:
--   {muon: {lr?, momentum?, weightDecay?, nsSteps?, nesterov?},
--    adam: {lr?, beta1?, beta2?, eps?, weightDecay?}})
return function(deps: any): any
	local Muon = deps.Muon
	local AdamW = deps.AdamW
	local Opt = {}
	Opt.__index = Opt

	function Opt.new(muonParams: { any }, adamParams: { any }, opts: any?): any
		local o = opts or {}
		local mo = o.muon or {}
		local ao = o.adam or {}
		local self = setmetatable({
			muon = Muon.new(muonParams, mo.lr or 0.05, {
				momentum = mo.momentum,
				weightDecay = mo.weightDecay,
				nsSteps = mo.nsSteps,
				nesterov = mo.nesterov,
			}),
			adam = AdamW.new(adamParams, ao.lr or 3e-4, {
				beta1 = ao.beta1,
				beta2 = ao.beta2,
				eps = ao.eps,
				weightDecay = ao.weightDecay,
			}),
		}, Opt)
		return self
	end

	function Opt.step(self)
		self.muon:step()
		self.adam:step()
	end

	function Opt.zeroGrad(self)
		self.muon:zeroGrad()
		self.adam:zeroGrad()
	end

	function Opt.setLr(self, lr: number)
		self.muon:setLr(lr)
		self.adam:setLr(lr)
	end

	function Opt.getLr(self): number
		return self.muon:getLr()
	end

	function Opt.paramCount(self): number
		return self.muon:paramCount() + self.adam:paramCount()
	end

	function Opt.stateDict(self): any
		return { muon = self.muon:stateDict(), adam = self.adam:stateDict() }
	end

	function Opt.loadStateDict(self, st: any)
		self.muon:loadStateDict(st.muon)
		self.adam:loadStateDict(st.adam)
	end

	return Opt
end
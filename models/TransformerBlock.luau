--!strict
-- Pre-norm transformer block: h = x + SA(LN(x)); h = h + FFN(LN(h)).
-- Optional residual scaling: initType "rezero" (alpha=0) or "deepnorm"
-- (alpha = (2L)^(1/4) as in DeepNorm, 2203.00555).
return function(deps: any)
	local Tensor = deps.Tensor
	local Norm = deps.Norm
	local Attention = deps.Attention
	local FFN = deps.FFN

	local Block = {}
	Block.__index = Block

	function Block.new(name: string, C: number, nbHeads: number, kvHeads: number, headDim: number, ffnHidden: number, rng: any, opts: any?)
		local o = opts or {}
		local self: any = setmetatable({
			name = name,
			C = C,
			layers = {},
			initType = o.initType or "default", -- "default" | "rezero" | "deepnorm"
			L = o.numBlocks or 1,
			params = {},
		}, Block)

		self.norm1 = Norm.new(name .. "/norm1", C, rng, "rms", 1e-6)
		self.attn = Attention.new(name .. "/attn", C, nbHeads, kvHeads or nbHeads, headDim, rng, o.attnOpts)
		self.norm2 = Norm.new(name .. "/norm2", C, rng, "rms", 1e-6)
		self.ffn = FFN.new(name .. "/ffn", C, ffnHidden, rng)

		self.layers = { self.norm1, self.attn, self.norm2, self.ffn }

		if self.initType == "rezero" then
			-- ReZero (2003.04887): gating initialized to 0.
			self.alpha1 = Tensor.fromData({ 0 }, { 1 }, true)
			self.alpha2 = Tensor.fromData({ 0 }, { 1 }, true)
			self.alpha1.requiresGrad = true
			self.alpha2.requiresGrad = true
			table.insert(self.params, self.alpha1)
			table.insert(self.params, self.alpha2)
		end
		for _, l in self.layers do
			if l.params then
				for _, t in l.params do table.insert(self.params, t) end
			end
		end
		return self
	end

	function Block.residualScale(self: any): number
		if self.initType == "deepnorm" then
			return (2 * self.L) ^ 0.25
		end
		return 1.0
	end

	local function onesLike(h: any): { number }
		local s = table.create(#h.shape)
		for i = 1, #h.shape do s[i] = 1 end
		return s
	end

	function Block.forward(self: any, x: any, freqs: any?): any
		local a1 = self.attn.forward(self.attn, self.norm1.forward(self.norm1, x), freqs)
		local h
		if self.initType == "rezero" then
			local alpha1 = Tensor.reshape(self.alpha1, onesLike(x))
			h = Tensor.add(x, Tensor.mul(a1, alpha1))
		else
			h = Tensor.add(x, Tensor.mulScalar(a1, self.residualScale(self)))
		end
		local a2 = self.ffn.forward(self.ffn, self.norm2.forward(self.norm2, h))
		if self.initType == "rezero" then
			local alpha2 = Tensor.reshape(self.alpha2, onesLike(h))
			h = Tensor.add(h, Tensor.mul(a2, alpha2))
		else
			h = Tensor.add(h, Tensor.mulScalar(a2, self.residualScale(self)))
		end
		return h
	end

	return Block
end
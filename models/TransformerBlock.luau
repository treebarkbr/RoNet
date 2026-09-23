--!strict
-- Pre-norm transformer block: h = x + SA(LN(x)); h = h + FFN(LN(h)).
-- Optional residual scaling: initType "rezero" (alpha=0, 2003.04887, gates the
-- branch) or "deepnorm" (DeepNorm 2203.00555, alpha = (2L)^(1/4) scales the
-- identity path: h = alpha*x + G(LN(x))).
return function(deps: any)
	local Tensor = deps.Tensor
	local Norm = deps.Norm
	local Attention = deps.Attention
	local FFN = deps.FFN

	local Block = {}
	Block.__index = Block

	-- Build a pre-norm transformer block. opts.initType: "default", "rezero"
	-- (2003.04887, branch gated by a learnable alpha starting at 0) or "deepnorm"
	-- (2203.00555, identity path scaled by (2L)^(1/4), opts.numBlocks = L).
	function Block.new(name: string, C: number, nbHeads: number, kvHeads: number, headDim: number, ffnHidden: number, rng: any, opts: any?)
		local o = opts or {}
		local rmsEps = o.rmsEps or 1e-6
		local self: any = setmetatable({
			name = name,
			C = C,
			layers = {},
			initType = o.initType or "default", -- "default" | "rezero" | "deepnorm"
			L = o.numBlocks or 1,
			rmsEps = rmsEps,
			params = {},
		}, Block)

		self.norm1 = Norm.new(name .. "/norm1", C, rng, "rms", rmsEps)
		self.attn = Attention.new(name .. "/attn", C, nbHeads, kvHeads or nbHeads, headDim, rng, o.attnOpts)
		self.norm2 = Norm.new(name .. "/norm2", C, rng, "rms", rmsEps)
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

	-- Residual-path scale factor (1.0 default/rezero, (2L)^(1/4) deepnorm).
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

	-- h = x + SA(LN(x)); h = h + FFN(LN(h)); freqs passed through to attention.
	function Block.forward(self: any, x: any, freqs: any?): any
		local a1 = self.attn.forward(self.attn, self.norm1.forward(self.norm1, x), freqs)
		local s = self.residualScale(self)
		local h
		if self.initType == "rezero" then
			-- ReZero (2003.04887): gate the branch (alpha initialized to 0).
			local alpha1 = Tensor.reshape(self.alpha1, onesLike(x))
			h = Tensor.add(x, Tensor.mul(a1, alpha1))
		elseif self.initType == "deepnorm" then
			-- DeepNorm (2203.00555): scale the identity path, h = alpha*x + G(LN(x)).
			h = Tensor.add(Tensor.mulScalar(x, s), a1)
		else
			h = Tensor.add(x, a1)
		end
		local a2 = self.ffn.forward(self.ffn, self.norm2.forward(self.norm2, h))
		if self.initType == "rezero" then
			local alpha2 = Tensor.reshape(self.alpha2, onesLike(h))
			h = Tensor.add(h, Tensor.mul(a2, alpha2))
		elseif self.initType == "deepnorm" then
			h = Tensor.add(Tensor.mulScalar(h, s), a2)
		else
			h = Tensor.add(h, a2)
		end
		return h
	end

	-- Incremental single-token decode with a KV cache. Mirrors forward for a
	-- [1, 1, C] token at global position pos (1-based); cache = { attn = {...} },
	-- matching the shape passed by Transformer.generate. Requires the model
	-- params to be detached (eval) so Attention.step runs on raw arrays.
	function Block.step(self: any, x: any, freqs: any, cache: any, pos: number): any
		local a1 = self.attn.step(self.attn, self.norm1.forward(self.norm1, x), freqs, cache.attn, pos)
		local s = self.residualScale(self)
		local h
		if self.initType == "rezero" then
			local alpha1 = Tensor.reshape(self.alpha1, onesLike(x))
			h = Tensor.add(x, Tensor.mul(a1, alpha1))
		elseif self.initType == "deepnorm" then
			h = Tensor.add(Tensor.mulScalar(x, s), a1)
		else
			h = Tensor.add(x, a1)
		end
		local a2 = self.ffn.forward(self.ffn, self.norm2.forward(self.norm2, h))
		if self.initType == "rezero" then
			local alpha2 = Tensor.reshape(self.alpha2, onesLike(h))
			h = Tensor.add(h, Tensor.mul(a2, alpha2))
		elseif self.initType == "deepnorm" then
			h = Tensor.add(Tensor.mulScalar(h, s), a2)
		else
			h = Tensor.add(h, a2)
		end
		return h
	end

	return Block
end
--!strict
-- Decoder-only Transformer (GPT-style). Config fields:
-- { C, numBlocks, nbHeads, kvHeads, headDim, ffnHidden, vocab, maxSeq, tieEmbeds, dropoutP, initType }
-- forward(ids: {number}) -> logits [N, vocab] consistent with the last block.
return function(deps: any)
	local Tensor = deps.Tensor
	local Embedding = deps.Embedding
	local Norm = deps.Norm
	local RoPE = deps.RoPE
	local Block = deps.TransformerBlock
	local Inits = deps.Inits
	local Dropout = deps.Dropout

	local Transformer = {}
	Transformer.__index = Transformer

	-- Build a GPT-style decoder-only transformer. cfg fields:
	-- { C, numBlocks, nbHeads, kvHeads, headDim, ffnHidden, vocab, maxSeq,
	--   tieEmbeds?, dropoutP?, initType?, ropeBase? }.
	function Transformer.new(name: string, cfg: any, rng: any)
		local self: any = setmetatable({
			name = name,
			C = cfg.C,
			numBlocks = cfg.numBlocks,
			vocab = cfg.vocab,
			tieEmbeds = if cfg.tieEmbeds == nil then true else cfg.tieEmbeds,
			initType = cfg.initType or "default",
			params = {},
			blocks = {},
		}, Transformer)

		self.emb = Embedding.new(name .. "/emb", self.vocab, self.C, rng)
		self.drop = Dropout.new(name .. "/drop", cfg.dropoutP or 0.0, rng)
		for bi = 1, self.numBlocks do
			local block = Block.new(name .. "/b" .. bi, self.C, cfg.nbHeads,
				cfg.kvHeads or cfg.nbHeads, cfg.headDim, cfg.ffnHidden, rng,
				{ initType = self.initType, numBlocks = self.numBlocks })
			self.blocks[bi] = block
			for _, t in block.params do table.insert(self.params, t) end
		end
		self.normF = Norm.new(name .. "/normF", self.C, rng, "rms", 1e-6)

		if self.tieEmbeds then
			-- shared weight object: dedup happens on collection
			self.lmHeadW = self.emb.table
			table.insert(self.params, self.lmHeadW)
		else
			self.lmHeadW = Tensor.fromData(Inits.glorotUniform({ self.vocab, self.C }, rng), { self.vocab, self.C }, true)
			self.lmHeadW.requiresGrad = true
			table.insert(self.params, self.lmHeadW)
		end
		for _, t in self.normF.params do table.insert(self.params, t) end
		for _, t in self.emb.params do table.insert(self.params, t) end

		self.freqs = RoPE.precompute(cfg.maxSeq, cfg.headDim, cfg.ropeBase)
		self.rng = rng
		return self
	end

	-- Single-sequence forward: ids {number} (1-based token ids) -> logits [T, vocab].
	function Transformer.forward(self: any, ids: { number }): any
		local x = self.emb.forward(self.emb, ids) -- [N, C]
		x = Tensor.reshape(x, { 1, #ids, self.C })
		if self.drop.training and self.drop.p > 0 then
			x = self.drop.forward(self.drop, x)
		end
		for _, block in self.blocks do
			x = block.forward(block, x, self.freqs)
		end
		x = self.normF.forward(self.normF, x) -- [1, N, C]
		x = Tensor.reshape(x, { #ids, self.C })
		local logits = Tensor.matmul(x, Tensor.transpose2d(self.lmHeadW)) -- [N, vocab]
		return logits
	end

	-- Batched sequence forward: idsT is a [B, T] int tensor -> logits [B, T, vocab].
	function Transformer.forwardBatch(self: any, idsT: any): any
		local B, T = idsT.shape[1], idsT.shape[2]
		local x = Tensor.gatherRows(self.emb.table, idsT.data) -- [B*T, C]
		x = Tensor.reshape(x, { B, T, self.C })
		if self.drop.training and self.drop.p > 0 then
			x = self.drop.forward(self.drop, x)
		end
		for _, block in self.blocks do
			x = block.forward(block, x, self.freqs)
		end
		x = self.normF.forward(self.normF, x) -- [B, T, C]
		x = Tensor.reshape(x, { B * T, self.C })
		local logits = Tensor.matmul(x, Tensor.transpose2d(self.lmHeadW)) -- [B*T, vocab]
		return Tensor.reshape(logits, { B, T, self.vocab })
	end

	-- Switch to eval mode (disables dropout); returns self for chaining.
	function Transformer.eval(self): any
		self.drop.training = false
		return self
	end

	-- Switch to train mode (enables dropout); returns self for chaining.
	function Transformer.train(self): any
		self.drop.training = true
		return self
	end

	-- Autoregressive sampling. seed: {number} prompt ids. Appends nTokens new ids.
	-- opts: { temperature?, topK? }. Uses the model's rng for non-greedy picks.
	function Transformer.generate(self: any, seed: { number }, nTokens: number, opts: any?): { number }
		local o = opts or {}
		local temp = o.temperature or 1.0
		local topK = o.topK
		local V = self.vocab
		local ids = table.clone(seed)
		local rng = self.rng
		local needsEval = self.drop.training
		if needsEval then self:eval() end
		for _ = 1, nTokens do
			local logits = self:forward(ids) -- [T, V]
			local Tn = #ids
			local base = (Tn - 1) * V
			local probs: { number } = table.create(V)
			local mx = -math.huge
			for v = 1, V do
				local s = logits.data[base + v] / temp
				probs[v] = s
				if s > mx then mx = s end
			end
			local sum = 0
			for v = 1, V do
				local e = math.exp(probs[v] - mx)
				probs[v] = e
				sum += e
			end
			for v = 1, V do probs[v] = probs[v] / sum end
			if topK and topK < V then
				local copy = table.clone(probs)
				local chosen: { number } = {}
				for _ = 1, topK do
					local best = 1
					for v = 2, V do
						if copy[v] > copy[best] then best = v end
					end
					table.insert(chosen, best)
					copy[best] = -1
				end
				local ksum = 0
				for i = 1, #chosen do ksum += probs[chosen[i]] end
				local rebuild: { number } = table.create(V)
				for v = 1, V do rebuild[v] = 0 end
				for i = 1, #chosen do rebuild[chosen[i]] = probs[chosen[i]] / ksum end
				probs = rebuild
			end
			local r = rng and rng:nextFloat() or math.random()
			local acc = 0
			local pick = V
			for v = 1, V do
				acc += probs[v]
				if r <= acc then
					pick = v
					break
				end
			end
			table.insert(ids, pick)
		end
		if needsEval then self:train() end
		return ids
	end

	return Transformer
end
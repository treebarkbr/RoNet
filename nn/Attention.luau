--!strict
-- Multi-head causal attention with RoPE and optional GQA (grouped query heads:
-- kv_heads < nb_heads shares K/V across groups; requires nb_heads % kv_heads == 0).
-- Layout: [B, T, C]. W shapes: Wq [C, nb*hd], Wk/Wv [C, kv*hd], Wo [nb*hd, C].
return function(deps: any)
	local Tensor = deps.Tensor
	local Inits = deps.Inits
	local RoPE = deps.RoPE

	local MaskCache = {}

	local function makeCausalMask(T: number): any
		local m = MaskCache[T]
		if m then return m end
		local data = table.create(T * T)
		for i = 1, T do
			for j = 1, T do
				data[(i - 1) * T + j] = if j <= i then 0.0 else -1e30
			end
		end
		m = Tensor.fromData(data, { T, T }, false)
		MaskCache[T] = m
		return m
	end

	local Attention = {}
	Attention.__index = Attention

	function Attention.new(name: string, C: number, nbHeads: number, kvHeads: number, headDim: number, rng: any, opts: any?)
		local o = opts or {}
		local self = setmetatable({
			name = name,
			C = C,
			nb = nbHeads,
			kv = kvHeads or nbHeads,
			hd = headDim,
			causal = if o.causal == nil then true else o.causal,
			useRope = if o.rope == nil then true else o.rope,
			rng = rng,
		}, Attention)
		self.Wq = Tensor.fromData(Inits.glorotUniform({ C, nbHeads * headDim }, rng), { C, nbHeads * headDim }, true)
		self.Wk = Tensor.fromData(Inits.glorotUniform({ C, self.kv * headDim }, rng), { C, self.kv * headDim }, true)
		self.Wv = Tensor.fromData(Inits.glorotUniform({ C, self.kv * headDim }, rng), { C, self.kv * headDim }, true)
		self.Wo = Tensor.fromData(Inits.glorotUniform({ nbHeads * headDim, C }, rng), { nbHeads * headDim, C }, true)
		self.Wq.requiresGrad = true
		self.Wk.requiresGrad = true
		self.Wv.requiresGrad = true
		self.Wo.requiresGrad = true
		self.params = { self.Wq, self.Wk, self.Wv, self.Wo }
		return self
	end

	local function expandKV(K4: any, kvHeads: number, nbHeads: number): any
		if kvHeads >= nbHeads then return K4 end
		-- K4: [B, kv, T, hd] -> [B, nb, T, hd] by repeating each kv head groupSize times.
		local group = nbHeads // kvHeads
		local pieces = {}
		for h = 1, kvHeads do
			local slice = Tensor.sliceDim(K4, 2, h) -- [B, T, hd]
			local sh = Tensor.shapeOf(slice) -- [B, T, hd]
			local sh2 = { sh[1], 1, sh[2], sh[3] }
			local rs = Tensor.reshape(slice, sh2) -- [B, 1, T, hd]
			local reps = {}
			for g = 1, group do reps[g] = rs end
			pieces[h] = Tensor.cat(reps, 2)
		end
		return Tensor.cat(pieces, 2)
	end

	function Attention.forward(self: any, x: any, freqs: any?): any
		local B, T = x.shape[1], x.shape[2]
		local hd = self.hd
		local D = self.C
		local xf = Tensor.reshape(x, { B * T, D })

		local Q = Tensor.matmul(xf, self.Wq) -- [B*T, nb*hd]
		local K = Tensor.matmul(xf, self.Wk) -- [B*T, kv*hd]
		local V = Tensor.matmul(xf, self.Wv)
		Q = Tensor.transpose(Tensor.reshape(Q, { B, T, self.nb, hd }), 2, 3) -- [B, nb, T, hd]
		K = Tensor.transpose(Tensor.reshape(K, { B, T, self.kv, hd }), 2, 3)
		V = Tensor.transpose(Tensor.reshape(V, { B, T, self.kv, hd }), 2, 3)

		if self.useRope and freqs and T <= freqs.cosT.shape[1] then
			Q = RoPE.apply(Q, freqs)
			K = RoPE.apply(K, freqs)
		end

		K = expandKV(K, self.kv, self.nb)
		V = expandKV(V, self.kv, self.nb)

		local Qf = Tensor.reshape(Q, { B * self.nb, T, hd })
		local Kf = Tensor.reshape(K, { B * self.nb, T, hd })
		local Vf = Tensor.reshape(V, { B * self.nb, T, hd })

		local S = Tensor.bmm(Qf, Tensor.transpose(Kf, 2, 3)) -- [B*nb, T, hd] x [B*nb, hd, T] -> [B*nb, T, T]
		S = Tensor.mulScalar(S, 1 / math.sqrt(hd))
		if self.causal and T > 1 then
			local m = makeCausalMask(T)
			S = Tensor.add(S, m)
		end
		local P = Tensor.softmax(S, 3)
		local O = Tensor.bmm(P, Vf) -- [B*nb, T, T] x [B*nb, T, hd] -> [B*nb, T, hd]

		O = Tensor.transpose(Tensor.reshape(O, { B, self.nb, T, hd }), 2, 3) -- [B, T, nb, hd]
		O = Tensor.reshape(O, { B * T, self.nb * hd })
		local Out = Tensor.matmul(O, self.Wo) -- [B*T, C]
		Out = Tensor.reshape(Out, { B, T, D })
		return Out
	end

	return Attention
end
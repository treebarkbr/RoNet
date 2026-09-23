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

	-- Create a multi-head attention module. opts.causal (default true) and
	-- opts.rope (default true) toggle masking and rotary embeddings.
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

	-- Attention over x: [B, T, C] -> [B, T, C]. Pass the RoPE.precompute result
	-- as freqs when the module has rope enabled.
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

		if self.useRope then
			if not freqs then
				error("Attention '" .. self.name .. "': RoPE enabled but no freq tables were passed", 2)
			end
			Q = RoPE.apply(Q, freqs)
			K = RoPE.apply(K, freqs)
		end

		K = expandKV(K, self.kv, self.nb)
		V = expandKV(V, self.kv, self.nb)

		local Qf = Tensor.reshape(Q, { B * self.nb, T, hd })
		local Kf = Tensor.reshape(K, { B * self.nb, T, hd })
		local Vf = Tensor.reshape(V, { B * self.nb, T, hd })

		local S: any
		if not (Qf.requiresGrad or Kf.requiresGrad) then
			-- inference: read K as-is, no transposed copy, SIMD lanes over hd
			S = Tensor.bmmNT(Qf, Kf)
		else
			S = Tensor.bmm(Qf, Tensor.transpose(Kf, 2, 3)) -- [B*nb, T, hd] x [B*nb, hd, T] -> [B*nb, T, T]
		end
		S = Tensor.mulScalar(S, 1 / math.sqrt(hd))
		if self.causal and T > 1 then
			local m = makeCausalMask(T)
			S = Tensor.add(S, m)
		end
		local P = Tensor.softmax(S, 3)
		local O: any
		if not (P.requiresGrad or Vf.requiresGrad) then
			O = Tensor.bmmFast(P, Vf) -- forward-only SIMD scores @ values
		else
			O = Tensor.bmm(P, Vf) -- [B*nb, T, T] x [B*nb, T, hd] -> [B*nb, T, hd]
		end

		O = Tensor.transpose(Tensor.reshape(O, { B, self.nb, T, hd }), 2, 3) -- [B, T, nb, hd]
		O = Tensor.reshape(O, { B * T, self.nb * hd })
		local Out = Tensor.matmul(O, self.Wo) -- [B*T, C]
		Out = Tensor.reshape(Out, { B, T, D })
		return Out
	end

	-- Incremental decode of one token at global position pos (1-based) with a KV
		-- cache: x [1, C] is the current token's hidden state. freqs has flat
		-- cos/sin tables, cache = { k = {number}, v = {number}, len } per block
		-- (row-major [len, kv*hd]). Assumes eval mode (no autograd, raw arrays);
		-- the freshly-built Q/K/V tensors are mutated in place for the rotation.
		function Attention.step(self: any, x: any, freqs: any, cache: any, pos: number): any
			local hd = self.hd
			local nb = self.nb
			local kv = self.kv
			local group = nb // kv
			local xf = Tensor.reshape(x, { 1, self.C })
			local qd = Tensor.matmul(xf, self.Wq).data
			local kd = Tensor.matmul(xf, self.Wk).data
			local vd = Tensor.matmul(xf, self.Wv).data

			local half = hd // 2
			if self.useRope then
				local c = freqs.cos
				local s = freqs.sin
				local cbase = (pos - 1) * half
				for h = 1, nb do
					local b = (h - 1) * hd
					for k = 1, half do
						local qe = qd[b + 2 * k - 1]
						local qo = qd[b + 2 * k]
						local cs = c[cbase + k]
						local ss = s[cbase + k]
						qd[b + 2 * k - 1] = qe * cs - qo * ss
						qd[b + 2 * k] = qe * ss + qo * cs
					end
				end
				for g = 1, kv do
					local b = (g - 1) * hd
					for k = 1, half do
						local ke = kd[b + 2 * k - 1]
						local ko = kd[b + 2 * k]
						local cs = c[cbase + k]
						local ss = s[cbase + k]
						kd[b + 2 * k - 1] = ke * cs - ko * ss
						kd[b + 2 * k] = ke * ss + ko * cs
					end
				end
			end

			local kvhd = kv * hd
			local kc, vc = cache.k, cache.v
			local len = cache.len
			local off = len * kvhd
			for j = 1, kvhd do
				kc[off + j] = kd[j]
				vc[off + j] = vd[j]
			end
			len += 1
			cache.len = len

			local invSqrt = 1 / math.sqrt(hd)
			local out = table.create(nb * hd, 0)
			for h = 1, nb do
				local g = ((h - 1) // group) + 1
				local qb = (h - 1) * hd
				local kb = (g - 1) * hd
				local scores = table.create(len)
				for i = 1, len do
					local so = (i - 1) * kvhd + kb
					local acc = 0
					for j = 1, hd do
						acc += qd[qb + j] * kc[so + j]
					end
					scores[i] = acc * invSqrt
				end
				local m = scores[1]
				for i = 2, len do
					if scores[i] > m then m = scores[i] end
				end
				local sum = 0
				for i = 1, len do
					local e = math.exp(scores[i] - m)
					scores[i] = e
					sum += e
				end
				local inv = 1 / sum
				for i = 1, len do scores[i] = scores[i] * inv end
				for j = 1, hd do
					local acc = 0
					for i = 1, len do
						acc += scores[i] * vc[(i - 1) * kvhd + kb + j]
					end
					out[qb + j] = acc
				end
			end

			local o = Tensor.fromData(out, { 1, nb * hd }, false)
			local Out = Tensor.matmul(o, self.Wo) -- [1, C]
			return Tensor.reshape(Out, { 1, 1, self.C })
		end

	return Attention
end
--!strict
-- Rotary position embeddings (RoPE, Su et al. 2022). Precompute freqs from
-- theta_i = base^(-2i/d). Apply rotation to the trailing dim in twos.
return function(deps: any)
	local Tensor = deps.Tensor

	local RoPE = {}

	-- Returns { cos: {number}[T][half], sin: {number}[T][half] } flat [T*half] each.
	function RoPE.precompute(seqLen: number, headDim: number, base: number?)
		local b = base or 10000.0
		local half = headDim / 2
		local freqs = table.create(seqLen * half)
		for p = 0, seqLen - 1 do
			for j = 0, half - 1 do
				freqs[p * half + j + 1] = p / (b ^ ((2 * j) / headDim))
			end
		end
		local cos = table.create(seqLen * half)
		local sin = table.create(seqLen * half)
		for i = 1, seqLen * half do
			cos[i] = math.cos(freqs[i])
			sin[i] = math.sin(freqs[i])
		end
		return { cos = cos, sin = sin, cosT = Tensor.fromData(cos, { seqLen, half }, false), sinT = Tensor.fromData(sin, { seqLen, half }, false) }
	end

	-- x: [..., T, D], freqs must have seqLen >= T. Rotates every (2k, 2k+1) pair.
	function RoPE.apply(x: any, freqs: any): any
		local xshape = Tensor.shapeOf(x)
		local nd = #xshape
		local Tsize = xshape[nd - 1]
		local D = xshape[nd]
		local half = math.floor(D / 2)
		local fC = freqs.cosT
		local fS = freqs.sinT
		if Tsize > fC.shape[1] then
			error("RoPE: sequence length exceeds precomputed freqs", 2)
		end
		local function trim(t: any): any
			if Tsize < t.shape[1] then
				return Tensor.sliceRange(t, 1, 1, Tsize)
			end
			return t
		end
		fC = trim(fC)
		fS = trim(fS)
		if half > fC.shape[2] then half = fC.shape[2] end

		-- Split trailing dim into pairs: [..., T, half, 2]
		local keep = table.create(nd - 1)
		for i = 1, nd - 1 do keep[i] = xshape[i] end
		local pairedShape = table.create(#keep + 2)
		for i = 1, #keep do pairedShape[i] = keep[i] end
		pairedShape[#keep + 1] = half
		pairedShape[#keep + 2] = 2
		local paired = Tensor.reshape(x, pairedShape)

		local even = Tensor.selectLastDim(paired, 1) -- [..., T, half]
		local odd = Tensor.selectLastDim(paired, 2)

		-- broadcast [T, half] against [..., T, half]: right-aligned ok
		local rotEven = Tensor.sub(Tensor.mul(even, fC), Tensor.mul(odd, fS))
		local rotOdd = Tensor.add(Tensor.mul(even, fS), Tensor.mul(odd, fC))

		-- stack back into [..., T, half, 2] then restore shape
		local stack = RoPE.stackLast(rotEven, rotOdd)
		return Tensor.reshape(stack, xshape)
	end

	-- stack [..., h] and [..., h] along a new trailing size-2 dim -> [..., h, 2].
	local function stackLast(a: any, b: any): any
		local ash = Tensor.shapeOf(a)
		local keep = table.create(#ash + 1)
		for i = 1, #ash do keep[i] = ash[i] end
		keep[#ash + 1] = 1
		local ra = Tensor.reshape(a, keep)
		local rb = Tensor.reshape(b, keep)
		-- both [..., h, 1]; cat along last dim (dim index = #keep)
		local catDim = #keep
		return Tensor.cat({ ra, rb }, catDim)
	end
	RoPE.stackLast = stackLast

	return RoPE
end
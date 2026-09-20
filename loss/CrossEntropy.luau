--!strict
-- Numerically stable mean cross-entropy. logits: [N, vocab]; targets: N token ids (1-based).
-- Returns a scalar Tensor (shape {}), ready for backward(). Optional label smoothing.
return function(deps: any)
	local Tensor = deps.Tensor

	local function crossEntropy(logits: any, targets: { number }, smooth: number?): any
		local N = logits.shape[1]
		local V = logits.shape[2]
		local s = smooth or 0.0

		local mx = Tensor.max(logits, 2) -- [N]
		local shifted = Tensor.sub(logits, Tensor.reshape(mx, { N, 1 }))
		local exps = Tensor.exp(shifted)
		local lse = Tensor.add(Tensor.log(Tensor.sum(exps, 2)), mx) -- [N]

		local picked = Tensor.takeLast(logits, targets) -- [N]
		local lseP = Tensor.sub(lse, picked)

		if s > 0 then
			-- label smoothing: (1-s)*(-log p_t) + s * KL(U || p) over the vocab
			local uniform = -math.log(V)
			local lseU = Tensor.sub(lse, Tensor.scalar(uniform))
			local loss = Tensor.add(Tensor.mulScalar(lseP, 1 - s), Tensor.mulScalar(lseU, s))
			return Tensor.mean(loss, -1)
		end
		return Tensor.mean(lseP, -1)
	end

	return { crossEntropy = crossEntropy }
end
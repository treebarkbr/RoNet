--!strict
-- Additional losses built from Tensor primitives (Tensor is passed via deps).
-- All return 0-dim scalar Tensors ready for backward().
--   crossEntropy(logits, targets, smooth?)     numeric-stable, mean over rows
--   logSoftmax(logits) -> [N, V]
--   klDiv(logitsP, logitsQ)                    mean row-wise KL(P || Q)
--   jsDiv(logitsP, logitsQ)?                   (identity-aggregated P & Q means)
--   focalLoss(logits, targets, gamma?, alpha?, smooth?)
--   binaryCrossEntropy(logits, targets01)      (logits = unscaled scores)
return function(deps: any)
	local Tensor = deps.Tensor

	-- log-softmax of [N, V] rows, numerically stable (subtract row max).
	local function logSoftmax(logits: any): any
		local N = logits.shape[1]
		local mx = Tensor.max(logits, 2) -- [N]
		local shifted = Tensor.sub(logits, Tensor.reshape(mx, { N, 1 }))
		local lse = Tensor.add(Tensor.log(Tensor.sum(Tensor.exp(shifted), 2)), mx) -- [N]
		return Tensor.sub(logits, Tensor.reshape(lse, { N, 1 }))
	end

	-- KL divergence, mean over the N rows (rows need not sum to 1).
	local function klDiv(logitsP: any, logitsQ: any): any
		local lp = logSoftmax(logitsP)
		local lq = logSoftmax(logitsQ)
		local p = Tensor.exp(lp)
		local perRow = Tensor.sum(Tensor.mul(p, Tensor.sub(lp, lq)), 2)
		return Tensor.mean(perRow, -1)
	end

	local function onesLike(t: any): any
		return Tensor.ones(t.shape)
	end

	-- Jensen-Shannon divergence (symmetrized, in [0, 1] loge units to fade apart).
	local function jsDiv(logitsP: any, logitsQ: any): any
		local lp = logSoftmax(logitsP)
		local lq = logSoftmax(logitsQ)
		local p = Tensor.exp(lp)
		local q = Tensor.exp(lq)
		local m = Tensor.mulScalar(Tensor.add(p, q), 0.5)
		local lm = Tensor.log(m)
		local perRow = Tensor.add(Tensor.sum(Tensor.mul(p, Tensor.sub(lp, lm)), 2),
			Tensor.sum(Tensor.mul(q, Tensor.sub(lq, lm)), 2))
		return Tensor.mulScalar(Tensor.mean(perRow, -1), 0.5)
	end

	-- Focal loss on softmax logits (Lin et al., 2017; gamma default 2). alpha
	-- is a per-target weight (multi-class alpha table or scalar); omitted = none.
	local function focalLoss(logits: any, targets: { number }, gamma: number?, alpha: any?): any
		local g = gamma or 2.0
		local lp = logSoftmax(logits)
		local lpAt = Tensor.takeLast(lp, targets) -- [N] log-softmax at the true class
		local pt = Tensor.exp(lpAt) -- [N] predicted prob at true class
		local negLoss = Tensor.mulScalar(lpAt, -1) -- -log p_t
		local oneMinus = Tensor.sub(onesLike(pt), pt) -- (1 - p_t)
		local w = Tensor.pow(oneMinus, g) -- (1-p_t)^g, the focal factor
		-- weight factors: alpha * (1-p_t)^g  on -log p_t
		if alpha then
			local a: any = alpha
			if type(a) == "number" then
				local av: { number } = table.create(#targets)
				for i = 1, #targets do av[i] = alpha end
				a = Tensor.fromData(av, { #targets }, false)
			end
			local aw = Tensor.mul(Tensor.takeLast(a, targets), w)
			return Tensor.mean(Tensor.mul(aw, negLoss), -1)
		end
		return Tensor.mean(Tensor.mul(w, negLoss), -1)
	end

	-- Binary cross-entropy from raw logits (targets in {0, 1}); mean over all
	-- elements. Numerically stable via clipping sigmoid probability.
	local function binaryCrossEntropy(logits: any, targets: any): any
		local p = Tensor.clip(Tensor.sigmoid(logits), 1e-7, 1 - 1e-7) -- same shape as logits
		local lp = Tensor.log(p)
		local lq = Tensor.log(Tensor.sub(onesLike(p), p))
		local per = Tensor.add(Tensor.mul(targets, lp), Tensor.mul(Tensor.sub(onesLike(p), targets), lq))
		return Tensor.mean(Tensor.neg(per), -1)
	end

	-- Language-model next-token loss over a batch of sequences.
-- logits: [B, T, V] from a batched model forward; inputs: [B, T] int tensor.
-- Predicts token t+1 from position t for t = 1..T-1 (mean per position).
-- Returns a 0-dim scalar ready for backward().
	local function lmCrossEntropy(logits: any, inputs: any, smooth: number?): any
		local B = logits.shape[1]
		local T = logits.shape[2]
		local V = logits.shape[3]
		if T < 2 then error("lmCrossEntropy: sequence length must be >= 2", 2) end
		local nPos = B * (T - 1)
		-- select prediction rows: every position except the last token in each seq
		local predIdx: { number } = table.create(nPos)
		local targets: { number } = table.create(nPos)
		local k = 0
		for b = 1, B do
			local rowBase = (b - 1) * T
			for t = 1, T - 1 do
				k += 1
				predIdx[k] = rowBase + t
				targets[k] = inputs.data[rowBase + t + 1]
			end
		end
		local flat = Tensor.reshape(logits, { B * T, V })
		local chosen = Tensor.gatherRows(flat, predIdx) -- [nPos, V]
		return deps.CrossEntropy.crossEntropy(chosen, targets, smooth)
	end

	return {
		crossEntropy = deps.CrossEntropy and deps.CrossEntropy.crossEntropy or nil,
		logSoftmax = logSoftmax,
		klDiv = klDiv,
		jsDiv = jsDiv,
		focalLoss = focalLoss,
		binaryCrossEntropy = binaryCrossEntropy,
		lmCrossEntropy = lmCrossEntropy,
	}
end
--!strict
-- SOAP (Vyas et al., 2014/2024; arXiv:2409.11321). Shampoo-style update with
-- Adam-in-eigenbasis. Eigenspaces recomputed from GG = grad grad' (Shampoo
-- preconditioners) every precondition_frequency steps; first step builds the
-- preconditioner and is skipped. 2D parameters are preconditioned; 1D fall back
-- to Adam. Requires deps.Matrix (eigh).
-- Class: new(params: {Tensor}, lr: number, opts: {betas?, eps?, weightDecay?,
--        preconditionFrequency?, shampooBeta?, maxPrecondDim?, precondition1d?})
return function(deps: any): any
	local Matrix = deps.Matrix
	local Optimizer = {}
	Optimizer.__index = Optimizer

	local function eighDesc(mat: { number }, n: number): { number }
		local _, Q = Matrix.eigh(mat, n)
		local q = Matrix.copy(Q)
		local half = math.floor(n / 2)
		for j = 1, half do
			local o1 = (j - 1) * n
			local o2 = (n - j) * n
			for i = 1, n do
				q[o1 + i], q[o2 + i] = q[o2 + i], q[o1 + i]
			end
		end
		return q
	end

	local function projectG(G: { number }, m: number, n: number, Q0: { number }, Q1: { number }): { number }
		local q0t = Matrix.trans(Q0, m, m)
		local t1 = Matrix.mm(q0t, G, m, m, m, n)
		return Matrix.mm(t1, Q1, m, n, n, n)
	end

	local function projectBack(G: { number }, m: number, n: number, Q0: { number }, Q1: { number }): { number }
		local t1 = Matrix.mm(Q0, G, m, m, m, n)
		return Matrix.mm(t1, Matrix.trans(Q1, n, n), m, n, n, n)
	end

	local function transposeFlat(G: { number }, m: number, n: number): { number }
		return Matrix.trans(G, m, n)
	end

	-- Create the optimizer over `params` (Module.collectParams(model)) at a base learning rate.
	function Optimizer.new(params: { any }, lr: number, opts: any?): any
		local o = opts or {}
		local betas = o.betas or { 0.95, 0.95 }
		local self = setmetatable({
			params = params,
			lr = lr,
			b1 = betas[1],
			b2 = betas[2],
			shampooBeta = if o.shampooBeta and o.shampooBeta >= 0 then o.shampooBeta else betas[2],
			eps = o.eps or 1e-8,
			wd = o.weightDecay or 0.01,
			freq = o.preconditionFrequency or 10,
			maxPrecondDim = o.maxPrecondDim or 10000,
			precond1d = if o.precondition1d then true else false,
			state = {},
		}, Optimizer)
		return self
	end

	-- Lazily build Shampoo preconditioner state (gg0/gg1, moments, eigenspaces) for a 2D param.
	function Optimizer._buildInitial(self, p: any, st: any)
		local d = p.data
		local m = p.shape[1]
		local n = p.shape[2]
		st.gg0 = Matrix.zeros(m * m)
		st.gg1 = Matrix.zeros(n * n)
		st.expAvg = table.create(#d, 0)
		st.expAvgSq = table.create(#d, 0)
		self:_updateGG(p, st)
		st.q0 = eighDesc(st.gg0, m)
		st.q1 = eighDesc(st.gg1, n)
	end

	-- Update the Shampoo preconditioner accumulators (gg0/gg1) from the current grad.
	function Optimizer._updateGG(self, p: any, st: any)
		local m = p.shape[1]
		local n = p.shape[2]
		local g = p.grad
		if #g < m * n then return end
		local s = self.shampooBeta
		local gg0 = Matrix.copy(st.gg0)
		local gg1 = Matrix.copy(st.gg1)
		local ggt = Matrix.mm(g, transposeFlat(g, m, n), m, n, n, m)
		local gtg = Matrix.mm(transposeFlat(g, m, n), g, n, m, m, n)
		for i = 1, m * m do st.gg0[i] = s * gg0[i] + (1 - s) * ggt[i] end
		for i = 1, n * n do st.gg1[i] = s * gg1[i] + (1 - s) * gtg[i] end
	end

	-- Apply one update from the current parameter gradients (call Tensor.backward() first).
	function Optimizer.step(self)
		local lr, b1, b2, eps, wd = self.lr, self.b1, self.b2, self.eps, self.wd
		for _, p in self.params do
			local g = p.grad
			if not g then continue end
			local ndim = #p.shape
			local st = self.state[p]
			if not st then
				st = { step = 0 }
				self.state[p] = st
			end
			if ndim == 2 and p.shape[1] <= self.maxPrecondDim and p.shape[2] <= self.maxPrecondDim then
				local m = p.shape[1]
				local n = p.shape[2]
				if not st.gg0 then
					st.step = 0
					self:_buildInitial(p, st)
					continue
				end
				local proj = projectG(g, m, n, st.q0, st.q1)
				local ea = st.expAvg
				local es = st.expAvgSq
				for i = 1, #ea do
					ea[i] = b1 * ea[i] + (1 - b1) * proj[i]
					es[i] = b2 * es[i] + (1 - b2) * proj[i] * proj[i]
				end
				st.step += 1
				local bc1 = 1 - b1 ^ st.step
				local bc2 = 1 - b2 ^ st.step
				local stepSize = lr * math.sqrt(bc2) / bc1
				local update = table.create(m * n)
				for i = 1, m * n do update[i] = ea[i] / (math.sqrt(es[i]) + eps) end
				local normG = projectBack(update, m, n, st.q0, st.q1)
				local d = p.data
				for i = 1, m * n do
					d[i] -= stepSize * normG[i]
					d[i] -= lr * wd * d[i]
				end
				local eaOrig = projectBack(ea, m, n, st.q0, st.q1)
				for i = 1, m * n do ea[i] = eaOrig[i] end
				self:_updateGG(p, st)
				if st.step % self.freq == 0 then
					st.q0 = eighDesc(st.gg0, m)
					st.q1 = eighDesc(st.gg1, n)
				end
				local np = projectG(ea, m, n, st.q0, st.q1)
				for i = 1, m * n do
					ea[i] = np[i]
				end
			else
				local ea = st.expAvg or table.create(#p.data, 0)
				local es = st.expAvgSq or table.create(#p.data, 0)
				st.expAvg = ea
				st.expAvgSq = es
				st.step += 1
				local bc1 = 1 - b1 ^ st.step
				local bc2 = 1 - b2 ^ st.step
				local d = p.data
				for i = 1, #d do
					local gi = g[i]
					ea[i] = b1 * ea[i] + (1 - b1) * gi
					es[i] = b2 * es[i] + (1 - b2) * gi * gi
					local mh = ea[i] / bc1
					local vh = es[i] / bc2
					d[i] = d[i] - lr * (mh / (math.sqrt(vh) + eps) + wd * d[i])
				end
			end
		end
	end

	-- Zero every parameter gradient buffer.
	function Optimizer.zeroGrad(self)
		for _, p in self.params do
			if p.grad then
				for i = 1, #p.grad do p.grad[i] = 0 end
			end
		end
	end

	-- Set the current learning rate.
	function Optimizer.setLr(self, lr: number)
		self.lr = lr
	end

	-- Return the current learning rate.
	function Optimizer.getLr(self): number
		return self.lr
	end

	-- Number of tracked parameters.
	function Optimizer.paramCount(self): number
		return #self.params
	end

	-- Export optimizer state (moments, step counter) for checkpointing.
	function Optimizer.stateDict(self): any
		return { s = self.state }
	end

	-- Restore optimizer state from a stateDict.
	function Optimizer.loadStateDict(self, st: any)
		self.state = st.s
	end

	return Optimizer
end
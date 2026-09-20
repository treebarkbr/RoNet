--!strict
-- Tiny training loop engine. Batches are built as real leading-dim tensors so a
-- whole minibatch is ONE forward, ONE backward, ONE optimizer step (single
-- autograd graph — the efficient path now that backward() accumulates per-call).
--   t = Trainer.new({ model, optimizer, schedule?, lossFn, data, epochs,
--                     batchSize?, evalFn?, evalEvery?, onEpochEnd? })
--   t:fit() -> { lossHistory: {number}, evals: {number} }
-- lossFn(model, xs, ys) -> 0-dim scalar Tensor ready for backward()
--   - xs: batched input tensor, ys: targets array {number}
-- Trainer.batchTensor(rows, featureShape) builds the input tensor.
return function(deps: any)
	local Tensor = deps.Tensor

	local Trainer = {}
	Trainer.__index = Trainer

	-- Build a single tensor from rows of equal-length numeric arrays.
	function Trainer.batchTensor(rows: { { number } }, featureShape: { number }): any
		local nRows = #rows
		local per = 1
		for d = 1, #featureShape do per *= featureShape[d] end
		local flat = table.create(nRows * per)
		for r = 1, nRows do
			local row = rows[r]
			local base = (r - 1) * per
			for i = 1, per do
				flat[base + i] = row[i]
			end
		end
		local shape = table.create(1 + #featureShape)
		shape[1] = nRows
		for d = 1, #featureShape do shape[d + 1] = featureShape[d] end
		return Tensor.fromData(flat, shape, false)
	end

	function Trainer.new(cfg: any): any
		local feature = cfg.featureShape or { #cfg.X[1] }
		local self = setmetatable({
			model = cfg.model,
			optimizer = cfg.optimizer,
			schedule = cfg.schedule,
			lossFn = cfg.lossFn,
			X = cfg.X,
			Y = cfg.Y,
			epochs = cfg.epochs or 1,
			batchSize = cfg.batchSize or 32,
			evalFn = cfg.evalFn,
			evalEvery = cfg.evalEvery or 1,
			onEpochEnd = cfg.onEpochEnd,
			feature = feature,
			lossHistory = {},
			evals = {},
			lastLoss = math.huge,
		}, Trainer)
		return self
	end

	function Trainer.fit(self): any
		local N = #self.X
		local bs = self.batchSize
		local lossFn = self.lossFn
		local X = self.X
		local model = self.model
		local opt = self.optimizer
		local sched = self.schedule
		local feature = self.feature

		for epoch = 1, self.epochs do
			local epochSum = 0
			local count = 0
			for start = 1, N, bs do
				local stop = math.min(start + bs - 1, N)
				local rows = table.create(stop - start + 1)
				local ys = table.create(stop - start + 1)
				for r = start, stop do
					rows[r - start + 1] = X[r]
					ys[r - start + 1] = self.Y[r]
				end
				local xs = Trainer.batchTensor(rows, feature)
				opt:zeroGrad()
				local loss = lossFn(model, xs, ys)
				Tensor.backward(loss, false)
				opt:step()
				if sched then sched:step() end
				epochSum += loss.data[1]
				count += 1
			end
			self.lastLoss = epochSum / math.max(1, count)
			table.insert(self.lossHistory, self.lastLoss)
			if self.evalFn and (epoch % self.evalEvery == 0 or epoch == self.epochs) then
				local v = self.evalFn(model)
				table.insert(self.evals, v)
			end
			if self.onEpochEnd then self.onEpochEnd(epoch, self.lastLoss) end
		end
		return self.lossHistory, self.evals
	end

	-- Run a gradient-free evaluation: total loss over data + evalFn value.
	function Trainer.evaluate(self, batchSize: number?): (number, number?)
		local N = #self.X
		local bs = batchSize or self.batchSize
		local lossFn = self.lossFn
		local total = 0
		local count = 0
		local feature = self.feature
		for start = 1, N, bs do
			local stop = math.min(start + bs - 1, N)
			local rows = table.create(stop - start + 1)
			local ys = table.create(stop - start + 1)
			for r = start, stop do
				rows[r - start + 1] = self.X[r]
				ys[r - start + 1] = self.Y[r]
			end
			local xs = Trainer.batchTensor(rows, feature)
			local loss = lossFn(self.model, xs, ys)
			total += loss.data[1]
			count += 1
		end
		self.lastLoss = total / math.max(1, count)
		if self.evalFn then
			return self.lastLoss, self.evalFn(self.model)
		end
		return self.lastLoss, nil
	end

	return Trainer
end
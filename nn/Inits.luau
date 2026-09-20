--!strict
-- Parameter initializers. Each returns a flat row-major data array for a shape.
-- Independent of the autograd engine; operates on raw Lua arrays.
return function(deps: any)
	local Inits = {}

	local function zerosData(n: number): { number }
	local d = table.create(n, 0)
	return d
end

	local function fillData(n: number, v: number): { number }
		local d = table.create(n, v)
		return d
	end

	function Inits.zeros(shape: { number }): { number }
		local n = 1
		for _, s in shape do n *= s end
		return zerosData(n)
	end

	function Inits.ones(shape: { number }): { number }
		local n = 1
		for _, s in shape do n *= s end
		return fillData(n, 1)
	end

	function Inits.normal(shape: { number }, rng: any, mean: number?, std: number?): { number }
		local n = 1
		for _, s in shape do n *= s end
		local m = mean or 0
		local sd = std or 0.02
		local d = table.create(n)
		for i = 1, n do
			d[i] = m + sd * rng.nextGaussian()
		end
		return d
	end

	function Inits.uniform(shape: { number }, rng: any, lo: number?, hi: number?): { number }
		local n = 1
		for _, s in shape do n *= s end
		local loV = lo or -0.05
		local hiV = hi or 0.05
		local span = hiV - loV
		local d = table.create(n)
		for i = 1, n do
			d[i] = loV + span * rng.nextFloat()
		end
		return d
	end

	-- Xavier/Glorot uniform: +-sqrt(6 / (fanIn + fanOut)).
	function Inits.glorotUniform(shape: { number }, rng: any): { number }
		local n = 1
		for _, s in shape do n *= s end
		local fanIn = shape[#shape]
		local fanOut = n / fanIn
		local a = math.sqrt(6 / (fanIn + fanOut))
		local d = table.create(n)
		for i = 1, n do
			d[i] = (rng.nextFloat() * 2 - 1) * a
		end
		return d
	end

	-- Kaiming uniform for ReLU: +-sqrt(6 / fanIn).
	function Inits.kaimingUniform(shape: { number }, rng: any): { number }
		local n = 1
		for _, s in shape do n *= s end
		local fanIn = shape[#shape]
		local a = math.sqrt(6 / fanIn)
		local d = table.create(n)
		for i = 1, n do
			d[i] = (rng.nextFloat() * 2 - 1) * a
		end
		return d
	end

	-- Small-initialized weights (pre-norm transformer convention; scale for our
	-- depths so activations stay ~unit magnitude under residual streams).
	function Inits.smallUniform(shape: { number }, rng: any, scale: number?): { number }
		local n = 1
		for _, s in shape do n *= s end
		local fanIn = shape[#shape]
		local a = (scale or 1) * math.sqrt(2 / fanIn)
		local d = table.create(n)
		for i = 1, n do
			d[i] = (rng.nextFloat() * 2 - 1) * a
		end
		return d
	end

	return Inits
end
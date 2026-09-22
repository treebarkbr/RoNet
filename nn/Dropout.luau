--!strict
-- Dropout (inverted dropout: mask = (u > p)/(1-p) while training, identity when eval).
return function(deps: any)
	local Tensor = deps.Tensor

	local Dropout = {}
	Dropout.__index = Dropout

	-- Create a dropout module (p default 0). train()/eval() control whether the
	-- mask is applied.
	function Dropout.new(name: string, p: number?, rng: any?)
		return setmetatable({ name = name, p = p or 0.0, rng = rng, training = true, params = {} }, Dropout)
	end

	-- Inverted dropout when training (mask ~ Bernoulli(keep) scaled by 1/keep);
	-- identity when p <= 0 or not in train mode.
	function Dropout.forward(self: any, x: any): any
		if not self.training or self.p <= 0 then return x end
		local keep = 1 - self.p
		local u = Tensor.rand(Tensor.shapeOf(x), self.rng)
		local mask = Tensor.gt(u, Tensor.scalar(self.p))
		local scaled = Tensor.mulScalar(mask, 1 / keep)
		return Tensor.mul(x, scaled)
	end

	return Dropout
end
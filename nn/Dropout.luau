--!strict
-- Dropout (inverted dropout: mask = (u > p)/(1-p) while training, identity when eval).
return function(deps: any)
	local Tensor = deps.Tensor

	local Dropout = {}
	Dropout.__index = Dropout

	function Dropout.new(name: string, p: number?, rng: any?)
		return setmetatable({ name = name, p = p or 0.0, rng = rng, training = true, params = {} }, Dropout)
	end

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
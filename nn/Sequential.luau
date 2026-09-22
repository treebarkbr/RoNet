--!strict
-- Sequential container: chains sub-layers. Sub-layers are instantiated by the
-- builder (called with rng) and each must respond to forward(x).
return function(deps: any)
	local Sequential = {}

	-- Build a chain of layers. `build(rng)` returns the sub-layer list; every
	-- sub-layer must respond to forward(x).
	function Sequential.new(name: string, build: (any) -> { any }, rng: any)
		local layers = build(rng)
		local params = {}
		for _, layer in layers do
			if layer._params then
				for _, p in layer._params do
					table.insert(params, p.tensor)
				end
			elseif layer.params then
				for _, t in layer.params do
					table.insert(params, t)
				end
			end
		end
		return setmetatable({ name = name, layers = layers, params = params }, Sequential)
	end

	-- Run the input through every layer in order.
	function Sequential.forward(self: any, x: any): any
		local out = x
		for _, layer in self.layers do
			out = layer.forward(layer, out)
		end
		return out
	end

	return Sequential
end
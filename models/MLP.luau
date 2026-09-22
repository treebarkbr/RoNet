--!strict
-- Simple MLP for regression/classification tests and small models.
return function(deps: any)
	local Linear = deps.Linear
	local Activations = deps.Activations

	local MLP = {}
	MLP.__index = MLP

	-- Build an MLP with sizes = list of per-layer widths (no. of params: sizes),
	-- e.g. {2, 16, 1}. opts.act is "relu" (default), "tanh", "silu", "gelu" or
	-- "sigmoid" and applies between every pair of dense layers.
	function MLP.new(name: string, sizes: { number }, rng: any, opts: any?)
		local o = opts or {}
		local actName = o.act or "relu"
		local self: any = setmetatable({ name = name, layers = {}, params = {} }, MLP)
		for i = 1, #sizes - 1 do
			local lin = Linear.new(name .. "/fc" .. i, sizes[i], sizes[i + 1], rng)
			table.insert(self.layers, lin)
			for _, t in lin.params do table.insert(self.params, t) end
			if i < #sizes - 1 then
				local act = { name = name .. "/act" .. i, params = {},
					forward = function(_, x) return Activations[actName](x) end }
				table.insert(self.layers, act)
			end
		end
		return self
	end

	-- Run the input through the layer chain in order.
	function MLP.forward(self: any, x: any): any
		local out = x
		for _, layer in self.layers do
			out = layer.forward(layer, out)
		end
		return out
	end

	-- Flip train/eval mode on all sub-layers that track `training` (dropout).
	function MLP.setTrain(self: any, train: boolean)
		for _, layer in self.layers do
			if layer.training ~= nil then layer.training = train end
		end
	end

	return MLP
end
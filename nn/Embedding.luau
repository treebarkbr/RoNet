--!strict
-- Token embedding table. indices: flat list of 1-based token ids [N] ->
-- returns [N, dim]. Gradients scatter back into the embedding table.
return function(deps: any)
	local Tensor = deps.Tensor
	local Inits = deps.Inits

	local Embedding = {}
	Embedding.__index = Embedding

	-- Create a [vocab, dim] embedding table, N(0, 0.02) initialized.
	function Embedding.new(name: string, vocab: number, dim: number, rng: any)
		local self = setmetatable({ name = name, vocab = vocab, dim = dim }, Embedding)
		self.table = Tensor.fromData(Inits.normal({ vocab, dim }, rng, 0, 0.02), { vocab, dim }, true)
		self.table.requiresGrad = true
		self.params = { self.table }
		return self
	end

	-- Gather rows for a flat list of 1-based token ids; gradients scatter back.
	function Embedding.forward(self: any, ids: { number }): any
		return Tensor.gatherRows(self.table, ids)
	end

	return Embedding
end
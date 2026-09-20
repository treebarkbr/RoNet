--!strict
-- Container helpers for learnable modules.
-- Every RoNetV4.1 module is a dependency-free factory: return function(deps) ... end
-- with deps = { Util, PRNG, Tensor }. No module may require() another module;
-- only the root entry (init.luau) composes the graph.
return function(deps: any): {
	new: (name: string?) -> any,
	collectParams: (root: any) -> { any },
}
	local Module = {}
	Module.__index = Module

	function Module.new(name: string?): any
		local self = setmetatable({
			name = name or "Module",
			_params = {},
			_children = {},
		}, Module)
		return self
	end

	function Module.addParam(self: any, name: string, tensor: any): any
		table.insert(self._params, { name = name, tensor = tensor })
		return tensor
	end

	function Module.addChild(self: any, name: string, child: any): any
		self._children[name] = child
		return child
	end

	-- Recursively collect every unique parameter tensor (deduped by reference so
	-- tied embeddings are not counted twice). Handles `params` arrays (layers)
	-- and `layers`/`_children` table traversal (containers).
	function Module.collectParams(root: any): { any }
		local seen: {any} = {}
		local out: {any} = {}
		local function visit(node: any)
			if node == nil then return end
			if node.params then
				for _, t in node.params do
					if not seen[t] then
						seen[t] = true
						table.insert(out, t)
					end
				end
			end
			if node.layers then
				for _, c in node.layers do
					visit(c)
				end
			end
			if node._children then
				for _, c in node._children do
					visit(c)
				end
			end
		end
		visit(root)
		return out
	end

	return Module
end
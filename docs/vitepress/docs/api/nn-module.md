# Module

Source: `nn/Module.luau`

Container helpers for learnable modules.  Every RoNetV4.1 module is a dependency-free factory: return function(deps) ... end  with deps = { Util, PRNG, Tensor }. No module may require() another module;  only the root entry (init.luau) composes the graph.

## Methods

#### `new(name: string?) -> any`

Create a bare container module (name optional). Add params/children with addParam/addChild.

#### `addParam(: any, name: string, tensor: any) -> any`

Register a named parameter tensor; returns the tensor.

#### `addChild(: any, name: string, child: any) -> any`

Register a named child module; returns the child.

#### `collectParams(root: any) -> { any }`

Recursively collect every unique parameter tensor (deduped by reference so tied embeddings are not counted twice). Handles `params` arrays (layers) and `layers`/`_children` table traversal (containers).


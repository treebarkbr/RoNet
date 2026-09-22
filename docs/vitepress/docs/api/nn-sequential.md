# Sequential

Source: `nn/Sequential.luau`

Sequential container: chains sub-layers. Sub-layers are instantiated by the  builder (called with rng) and each must respond to forward(x).

## Methods

#### `new(name: string, build: (any)`

Build a chain of layers. `build(rng)` returns the sub-layer list; every sub-layer must respond to forward(x).

#### `forward(: any, x: any) -> any`

Run the input through every layer in order.


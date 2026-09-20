# Sequential

Source: `nn/Sequential.luau`

Sequential container: chains sub-layers. Sub-layers are instantiated by the  builder (called with rng) and each must respond to forward(x).

## Methods

#### `new(name: string, build: (any)`

#### `forward(: any, x: any) -> any`


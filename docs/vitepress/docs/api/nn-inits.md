# Inits

Source: `nn/Inits.luau`

Parameter initializers. Each returns a flat row-major data array for a shape.  Independent of the autograd engine; operates on raw Lua arrays.

## Methods

#### `zeros(shape: { number }) -> { number }`

#### `ones(shape: { number }) -> { number }`

#### `normal(shape: { number }, rng: any, mean: number?, std: number?) -> { number }`

#### `uniform(shape: { number }, rng: any, lo: number?, hi: number?) -> { number }`

#### `glorotUniform(shape: { number }, rng: any) -> { number }`

Xavier/Glorot uniform: +-sqrt(6 / (fanIn + fanOut)).

#### `kaimingUniform(shape: { number }, rng: any) -> { number }`

Kaiming uniform for ReLU: +-sqrt(6 / fanIn).

#### `smallUniform(shape: { number }, rng: any, scale: number?) -> { number }`

Small-initialized weights (pre-norm transformer convention; scale for our depths so activations stay ~unit magnitude under residual streams).


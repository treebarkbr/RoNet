# Inits

Source: `nn/Inits.luau`

Parameter initializers. Each returns a flat row-major data array for a shape.  Independent of the autograd engine; operates on raw Lua arrays.

## Methods

#### `zeros(shape: { number }) -> { number }`

Zeroed data array for a shape.

#### `ones(shape: { number }) -> { number }`

All-ones data array for a shape.

#### `normal(shape: { number }, rng: any, mean: number?, std: number?) -> { number }`

Gaussian draws, N(mean or 0, std or 0.02).

#### `uniform(shape: { number }, rng: any, lo: number?, hi: number?) -> { number }`

Uniform draws over [lo or -0.05, hi or 0.05).

#### `glorotUniform(shape: { number }, rng: any) -> { number }`

Xavier/Glorot uniform: +-sqrt(6 / (fanIn + fanOut)).

#### `kaimingUniform(shape: { number }, rng: any) -> { number }`

Kaiming uniform for ReLU: +-sqrt(6 / fanIn).

#### `smallUniform(shape: { number }, rng: any, scale: number?) -> { number }`

Small-initialized weights (pre-norm transformer convention; scale for our depths so activations stay ~unit magnitude under residual streams).


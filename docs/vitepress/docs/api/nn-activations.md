# Activations

Source: `nn/Activations.luau`

Activation functions (functional, no params).

## Methods

#### `relu(x: any) -> any`

ReLU (max(x, 0)).

#### `silu(x: any) -> any`

SiLU / Swish.

#### `gelu(x: any) -> any`

GELU (tanh approximation).

#### `tanh(x: any) -> any`

Hyperbolic tangent.

#### `sigmoid(x: any) -> any`

Numerically stable sigmoid.

#### `softmax(x: any, dim: number?) -> any`

Softmax over the last dim by default (dim may be given as positive index).


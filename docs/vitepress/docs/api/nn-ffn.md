# FFN

Source: `nn/FFN.luau`

SwiGLU feed-forward block: FFN(x) = W3(silu(W1 x) .* (W2 x)).

## Methods

#### `new(name: string, C: number, hidden: number, rng: any)`

Create a SwiGLU FFN block [C -&gt; hidden -&gt; C].

#### `forward(: any, x: any) -> any`

SwiGLU forward: W3(silu(W1 x) .* (W2 x)); flattens &gt;2-D leading dims.


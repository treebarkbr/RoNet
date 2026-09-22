# Linear

Source: `nn/Linear.luau`

Fully-connected layer. W: [out, in], b: [out].  forward(x): y = x @ W^T + b   (x: [..., in] -&gt; [..., out])

## Methods

#### `new(name: string, fanIn: number, fanOut: number, rng: any)`

Create a fully-connected layer with Glorot-initialized W and zero b.

#### `forward(: any, x: any) -> any`

Forward pass: y = x @ W^T + b. Reshapes automatically for &gt;2-D leading dims.


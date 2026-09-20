# Linear

Source: `nn/Linear.luau`

Fully-connected layer. W: [out, in], b: [out].  forward(x): y = x @ W^T + b   (x: [..., in] -&gt; [..., out])

## Methods

#### `new(name: string, fanIn: number, fanOut: number, rng: any)`

#### `forward(: any, x: any) -> any`


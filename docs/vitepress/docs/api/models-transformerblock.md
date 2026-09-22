# TransformerBlock

Source: `models/TransformerBlock.luau`

Pre-norm transformer block: h = x + SA(LN(x)); h = h + FFN(LN(h)).  Optional residual scaling: initType "rezero" (alpha=0, 2003.04887, gates the  branch) or "deepnorm" (DeepNorm 2203.00555, alpha = (2L)^(1/4) scales the  identity path: h = alpha*x + G(LN(x))).

## Methods

#### `new(name: string, C: number, nbHeads: number, kvHeads: number, headDim: number, ffnHidden: number, rng: any, opts: any?)`

Build a pre-norm transformer block. opts.initType: "default", "rezero" (2003.04887, branch gated by a learnable alpha starting at 0) or "deepnorm" (2203.00555, identity path scaled by (2L)^(1/4), opts.numBlocks = L).

#### `residualScale(: any) -> number`

Residual-path scale factor (1.0 default/rezero, (2L)^(1/4) deepnorm).

#### `forward(: any, x: any, freqs: any?) -> any`

h = x + SA(LN(x)); h = h + FFN(LN(h)); freqs passed through to attention.


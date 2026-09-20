# TransformerBlock

Source: `models/TransformerBlock.luau`

Pre-norm transformer block: h = x + SA(LN(x)); h = h + FFN(LN(h)).  Optional residual scaling: initType "rezero" (alpha=0) or "deepnorm"  (alpha = (2L)^(1/4) as in DeepNorm, 2203.00555).

## Methods

#### `new(name: string, C: number, nbHeads: number, kvHeads: number, headDim: number, ffnHidden: number, rng: any, opts: any?)`

#### `residualScale(: any) -> number`

#### `forward(: any, x: any, freqs: any?) -> any`


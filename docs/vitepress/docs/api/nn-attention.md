# Attention

Source: `nn/Attention.luau`

Multi-head causal attention with RoPE and optional GQA (grouped query heads:  kv_heads &lt; nb_heads shares K/V across groups; requires nb_heads % kv_heads == 0).  Layout: [B, T, C]. W shapes: Wq [C, nb*hd], Wk/Wv [C, kv*hd], Wo [nb*hd, C].

## Methods

#### `new(name: string, C: number, nbHeads: number, kvHeads: number, headDim: number, rng: any, opts: any?)`

#### `forward(: any, x: any, freqs: any?) -> any`


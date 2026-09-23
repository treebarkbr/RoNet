# Attention

Source: `nn/Attention.luau`

Multi-head causal attention with RoPE and optional GQA (grouped query heads:  kv_heads &lt; nb_heads shares K/V across groups; requires nb_heads % kv_heads == 0).  Layout: [B, T, C]. W shapes: Wq [C, nb*hd], Wk/Wv [C, kv*hd], Wo [nb*hd, C].

## Methods

#### `new(name: string, C: number, nbHeads: number, kvHeads: number, headDim: number, rng: any, opts: any?)`

Create a multi-head attention module. opts.causal (default true) and opts.rope (default true) toggle masking and rotary embeddings.

#### `forward(: any, x: any, freqs: any?) -> any`

Attention over x: [B, T, C] -&gt; [B, T, C]. Pass the RoPE.precompute result as freqs when the module has rope enabled.

#### `step(: any, x: any, freqs: any, cache: any, pos: number) -> any`

Incremental decode of one token at global position pos (1-based) with a KV cache: x [1, C] is the current token's hidden state. freqs has flat cos/sin tables, cache = { k = {number}, v = {number}, len } per block (row-major [len, kv*hd]). Assumes eval mode (no autograd, raw arrays); the freshly-built Q/K/V tensors are mutated in place for the rotation.


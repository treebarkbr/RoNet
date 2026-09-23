# RoPE

Source: `nn/RoPE.luau`

Rotary position embeddings (RoPE, Su et al. 2022). Precompute freqs from  theta_i = base^(-2i/d). Apply rotation to the trailing dim in twos.

## Methods

#### `precompute(seqLen: number, headDim: number, base: number?)`

Returns { cos: {number}[T][half], sin: {number}[T][half] } flat [T*half] each.

#### `apply(x: any, freqs: any) -> any`

x: [..., T, D], freqs must have seqLen &gt;= T. Rotates every (2k, 2k+1) pair in one fused Tensor.rotary kernel (single pass, no [.., T, half] temporaries).


# Embedding

Source: `nn/Embedding.luau`

Token embedding table. indices: flat list of 1-based token ids [N] -&gt;  returns [N, dim]. Gradients scatter back into the embedding table.

## Methods

#### `new(name: string, vocab: number, dim: number, rng: any)`

#### `forward(: any, ids: { number }) -> any`


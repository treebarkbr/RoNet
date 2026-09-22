# Embedding

Source: `nn/Embedding.luau`

Token embedding table. indices: flat list of 1-based token ids [N] -&gt;  returns [N, dim]. Gradients scatter back into the embedding table.

## Methods

#### `new(name: string, vocab: number, dim: number, rng: any)`

Create a [vocab, dim] embedding table, N(0, 0.02) initialized.

#### `forward(: any, ids: { number }) -> any`

Gather rows for a flat list of 1-based token ids; gradients scatter back.


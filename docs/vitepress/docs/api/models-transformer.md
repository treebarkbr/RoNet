# Transformer

Source: `models/Transformer.luau`

Decoder-only Transformer (GPT-style). Config fields:  { C, numBlocks, nbHeads, kvHeads, headDim, ffnHidden, vocab, maxSeq, tieEmbeds, dropoutP, initType }  forward(ids: {number}) -&gt; logits [N, vocab] consistent with the last block.

## Methods

#### `new(name: string, cfg: any, rng: any)`

#### `forward(: any, ids: { number }) -> any`

#### `forwardBatch(: any, idsT: any) -> any`

Batched sequence forward: idsT is a [B, T] int tensor -&gt; logits [B, T, vocab].

#### `eval() -> any`

#### `train() -> any`

#### `generate(: any, seed: { number }, nTokens: number, opts: any?) -> { number }`

Autoregressive sampling. seed: {number} prompt ids. Appends nTokens new ids. opts: { temperature?, topK? }. Uses the model's rng for non-greedy picks.


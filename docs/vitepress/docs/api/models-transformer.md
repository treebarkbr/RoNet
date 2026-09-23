# Transformer

Source: `models/Transformer.luau`

Decoder-only Transformer (GPT-style). Config fields:  { C, numBlocks, nbHeads, kvHeads, headDim, ffnHidden, vocab, maxSeq, tieEmbeds, dropoutP, initType, ropeBase?, rmsEps? }  forward(ids: {number}) -&gt; logits [N, vocab] consistent with the last block.

## Methods

#### `new(name: string, cfg: any, rng: any)`

Build a GPT-style decoder-only transformer. cfg fields: { C, numBlocks, nbHeads, kvHeads, headDim, ffnHidden, vocab, maxSeq,   tieEmbeds?, dropoutP?, initType?, ropeBase?, rmsEps? }.

#### `forward(: any, ids: { number }) -> any`

Single-sequence forward: ids {number} (1-based token ids) -&gt; logits [T, vocab].

#### `forwardBatch(: any, idsT: any) -> any`

Batched sequence forward: idsT is a [B, T] int tensor -&gt; logits [B, T, vocab].

#### `eval() -> any`

Switch to eval mode (disables dropout) and detaches parameters so forward builds no autograd graph (matmul auto-routes to the SIMD kernel); returns self for chaining. train() restores the stored requiresGrad flags.

#### `train() -> any`

Switch to train mode (enables dropout); returns self for chaining.

#### `generate(: any, seed: { number }, nTokens: number, opts: any?) -> { number }`

Autoregressive sampling. seed: {number} prompt ids. Appends nTokens new ids. opts: { temperature?, topK?, cache? }. Uses the model's rng for non-greedy picks. cache defaults to true (KV-cache decode); set cache=false to replay a full forward per token (useful for parity checks).


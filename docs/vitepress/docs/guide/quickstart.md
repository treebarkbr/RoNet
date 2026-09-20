# Quick start

The library has no dependencies beyond a `luau` interpreter. Tests and example
scripts call `require` at the top level (entry scripts only), and every module
is a factory that expects a `deps` table. The two examples below are entry
scripts.

## A feed-forward classifier

```lua
local deps = require("../init")                -- composes the dependency graph
local MLP = deps.MLP
local AdamW = deps.AdamW
local Trainer = deps.Trainer
local Losses = deps.Losses
local PRNG = deps.PRNG
local Tensor = deps.Tensor
local Module = deps.Module

local model = MLP.new("net", { 2, 12, 1 }, PRNG.new(1), { act = "tanh" })
local opt = AdamW.new(Module.collectParams(model), 0.05)

Trainer.new({
  model = model,
  optimizer = opt,
  lossFn = function(m, xs, ys)                 -- xs: [B, 2] input tensor
    local out = m:forward(xs)                  -- out: [B, 1]
    return Losses.binaryCrossEntropy(out, Tensor.fromData(ys, { #ys, 1 }, false))
  end,
  X = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } },
  Y = { 1, 0, 0, 0 },
  epochs = 20,
  batchSize = 4,
  featureShape = { 2 },
}):fit()
```

## A tiny language model

```lua
local deps = require("../init")
local BPE = deps.BPE
local Transformer = deps.Transformer

local tok = BPE.train({ "fox hex yap zoo " }, 64) -- vocab = 256 + merges
local model = Transformer.new("lm", {
  C = 16, numBlocks = 2, nbHeads = 2, headDim = 8,
  ffnHidden = 32, vocab = tok:vocabSize(), maxSeq = 256,
}, deps.PRNG.new(3))

-- train with Trainer using deps.Losses.lmCrossEntropy(logits, idsTensor)...

local ids = tok:encode("fox ")
local out = model:generate(ids, 16, { temperature = 0.6, topK = 20 })
print(tok:decode(out))
```

## Conventions to know

- Gradients accumulate. Call `optimizer:zeroGrad()` before each batch; the
  Trainer does this for you.
- Batching builds one real leading dimension. `Transformer:forwardBatch(ids)`
  takes a `[B, T]` int tensor and returns `[B, T, vocab]` logits; use
  `Losses.lmCrossEntropy(logits, ids)` for the shifted next-token loss.
- `Module.collectParams(root)` deduplicates shared tensors by reference, so tied
  embeddings count once.
- `Serialize.dump(params)` returns a plain string; tokenizers persist via
  `tok:state()` and `BPE.fromState(st)`.
- `Transformer:eval()` disables dropout for `generate()`; `Transformer:train()`
  restores it.
- Every optimizer exposes `new`, `step`, `zeroGrad`, `setLr/getLr`, and
  `stateDict/loadStateDict`. Schedules attach with
  `Scheduler.forOptimizer(opt, schedule)`, stepped once per batch.

## Regenerating the API pages

```sh
node docs/gen.mjs
```

writes `docs/vitepress/docs/api/*.md` from the doc comments and method
signatures in the source.
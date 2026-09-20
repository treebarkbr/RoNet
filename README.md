# RoNetV4.1

A small neural network library for Luau, written as strict modules. It has an
autograd engine, tensor and matrix kernels, transformer building blocks, a set
of optimizers, a training loop, and a byte-level BPE tokenizer. It targets the
Luau command line interpreter used by Roblox developers, so every module keeps a
`--!strict` header and follows one dependency rule: only entry scripts call
`require`, and every module is a factory that takes a `deps` table.

The whole codebase runs under the stock Luau CLI with no external packages.

## Layout

```
core/       Util, PRNG, Tensor (autograd engine), Matrix (2D linear algebra)
nn/         Module container, Linear, Inits, Activations, Norm, Dropout,
            Embedding, RoPE, Attention (causal, GQA), FFN (SwiGLU), Sequential
models/     MLP, TransformerBlock (pre-norm, ReZero/DeepNorm options), Transformer
loss/       CrossEntropy (with label smoothing), extra losses (KL, JS, focal, BCE,
            LM next-token loss)
optim/      AdamW, AdEMAMix, Lion, NAdamW, CautiousAdamW, ScheduleFreeAdamW,
            SOAP, Muon, MuonAdamW
train/      Scheduler, EMA, Serialize (weight checkpoints), Trainer (batched fit)
data/       BPE (byte-level tokenizer)
tests/      entry scripts for each layer, plus tests/run_all.sh
.tools/     luau and luau-analyze binaries used by the test suite
```

## Requirements

- A Luau CLI binary (`luau`) and optionally `luau-analyze`. Copies ship in
  `.tools/`. To use them: `export LUAU=$(pwd)/.tools/luau` (the test runner
  honors the `LUAU` variable). Newer Luau releases work too; only the runner
  needs a `luau` executable.

## Running the tests

```
bash tests/run_all.sh                  # uses `luau` from PATH
LUAU=/path/to/luau bash tests/run_all.sh
```

Each test runs in its own process (the CLI requires are entry-only). The suite
covers the tensor engine with finite differences, every layer, every optimizer,
the training stack (schedulers, EMA, serialization, losses), the BPE tokenizer,
and one end-to-end language model fit followed by generation and checkpoint
round-trips.

## Quick start

### A feed-forward classifier

```lua
local deps = require("../init")            -- entry script
local MLP = deps.MLP
local AdamW = deps.AdamW
local Trainer = deps.Trainer
local Losses = deps.Losses

local model = MLP.new("net", { 2, 12, 1 }, deps.PRNG.new(1), { act = "tanh" })
local opt = AdamW.new(deps.Module.collectParams(model), 0.05)

trainer = Trainer.new({
  model = model,
  optimizer = opt,
  lossFn = function(m, xs, ys)                      -- xs: [B, 2] input tensor
    local out = m:forward(xs)                       -- out: [B, 1]
    return Losses.binaryCrossEntropy(out, deps.Tensor.fromData(ys, { #ys, 1 }, false))
  end,
  X = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } },   -- rows are plain arrays
  Y = { 1, 0, 0, 0 },
  epochs = 20,
  batchSize = 4,
  featureShape = { 2 },
})
local history = trainer:fit()
```

### A tiny language model

```lua
local deps = require("../init")
local BPE = deps.BPE
local Transformer = deps.Transformer

local tok = BPE.train({ "fox hex yap zoo " }, 64)   -- tokenizer, vocab = 256 + merges
local model = Transformer.new("lm", {
  C = 16, numBlocks = 2, nbHeads = 2, headDim = 8,
  ffnHidden = 32, vocab = tok:vocabSize(), maxSeq = 256,
}, deps.PRNG.new(3))

-- train with Trainer using deps.Losses.lmCrossEntropy(logits, ids), then:
local ids = tok:encode("fox ")
local out = model:generate(ids, 16, { temperature = 0.6, topK = 20 })
print(tok:decode(out))
```

## Conventions and usage notes

- **Gradient accumulation.** `Tensor.backward(loss)` accumulates into existing
  gradient buffers. Zero before each batch with `optimizer:zeroGrad()`, the same
  contract as PyTorch. The Trainer does this for you.
- **Batching.** `Trainer.batchTensor(rows, featureShape)` builds one real leading
  dimension so a minibatch is one forward, one backward, one optimizer step.
  `Transformer:forwardBatch(ids)` takes a `[B, T]` int tensor and returns
  `[B, T, vocab]` logits; sequence models the shifted next-token loss is
  `Losses.lmCrossEntropy(logits, ids)`.
- **Batched training is required for gradients** because parameters accumulate
  grads per sample; the Trainer is the supported loop.
- **Shared weights.** `Module.collectParams(root)` walks `params`/`layers` and
  deduplicates by reference, so tied embeddings count once.
- **Checkpoints.** `Serialize.dump(params)` returns a plain string;
  `Serialize.load(params, str)` writes it back into an existing list.
  Tokenizer state persists via `tok:state()` and `BPE.fromState(st)`.
- **Dropout and EVAL.** `Transformer:eval()` toggles dropout off for
  `generate()`, `Transformer:train()` restores it.
- **Optimizer API.** Every optimizer exposes `new(params, lr, opts)`, `step`,
  `zeroGrad`, `setLr/getLr`, `paramCount`, and `stateDict/loadStateDict`.
  Schedules attach with `Scheduler.forOptimizer(opt, schedule)` and are stepped
  once per batch.

## Research references

The implementation follows published designs where cited in file headers:
RMSNorm (Zhang and Sennrich 2019), RoPE (Su et al. 2022), grouped query attention
(Ainslie et al. 2023), ReZero (Bachlechner et al. 2020), DeepNorm (Wang et al.
2022), SwiGLU (Shazeer 2020), AdEMAMix (Ravfogel et al. 2024), Cautious AdamW,
Schedule-Free (Defazio et al. 2024), SOAP (Vyas et al. 2024), and Muon (Jordan
et al. 2024). Gradient accumulation follows the standard autograd convention.

## Limitations

- The engine is CPU only and fine for small experiments and Roblox-server
  workloads. Memory grows with graph depth on long sequences.
- `Matrix.eigh`, used by SOAP and Muon, is a Jacobi solver with a hardcoded
  iteration cap; it suits C up to a few hundred.
- There is no CUDA or GPU path and no mixed precision.
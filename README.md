<p align="center">
  <img src="docs/vitepress/docs/public/logo.png" width="160" alt="RoNet logo"/>
</p>

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
core/       Util, PRNG, Tensor (autograd engine), Matrix (2D linear algebra),
            Parallel (Actor-pool sharding for forward/loss/backward/scoreGenomes)
nn/         Module container, Linear, Inits, Activations, Norm, Dropout,
            Embedding, RoPE, Attention (causal, GQA), FFN (SwiGLU), Sequential
models/     MLP, TransformerBlock (pre-norm, ReZero/DeepNorm options), Transformer
loss/       CrossEntropy (with label smoothing), extra losses (KL, JS, focal, BCE,
            LM next-token loss)
optim/      AdamW, AdEMAMix, Lion, NAdamW, CautiousAdamW, ScheduleFreeAdamW,
            SOAP, Muon, MuonAdamW
train/      Scheduler, EMA, Serialize (weight checkpoints), Trainer (batched fit)
data/       BPE (byte-level tokenizer)
tools/      pt2ronet.py (HF safetensors -> RoNet import), pt2ronet_check.py
            (independent numpy cross-check of an imported model)
tests/      entry scripts for each layer, plus tests/run_all.sh
.tools/     luau and luau-analyze binaries used by the test suite
```

## Roblox port

A prebuilt Roblox Model lives at `builds/RoNet.rbxm`. Insert it anywhere under
`ReplicatedStorage` and `require(ReplicatedStorage.RoNet)` returns the fully
wired `deps` table (the root ModuleScript wires the graph with
`require(script.core.Util)`-style requires). The Model mirrors the repo Layout;
all modules are byte-for-byte the same source. Server Scripts can train with the
Trainer; clients can use the same module for inference. `RoNet.Parallel` can
shard `forward`/`loss`/`backward`/`scoreGenomes` across Roblox **Actor** sub-VMs
in multi-mode (`Parallel.setMode("multi")`), with single-mode fallback for the
CLI. Two runtime differences from the CLI: Roblox scripts do not expose Luau's
`vector`, so `matmulFast` falls back to the scalar `Tensor.matmul` (identical
values), and the test suite / `.tools/` binaries are not shipped. The artifact
is built with `build/rbxm/` (a small rbx-dom Rust tool; no Studio needed), not
hand-exported. Full details: [docs guide](/guide/roblox).

## Import a Hugging Face checkpoint

`tools/pt2ronet.py` converts a Hugging Face Llama-style `model.safetensors` plus
`config.json` (Llama 2/3, Mistral, and other GPT-NeoX-layout transformers) into
a standalone Luau module holding the exact RoNet Transformer `cfg` and a
`Serialize` weight string:

```sh
python3 tools/pt2ronet.py path/model.safetensors --config path/config.json --out weights.luau
```

Load it into a RoNet Transformer (the emitted module drops straight into the
`Serialize.load` contract):

```lua
local W = require("path.to.weights")
local model = deps.Transformer.new("pt", W.cfg, deps.PRNG.new(1))
deps.Serialize.load(deps.Module.collectParams(model), W.weights)
```

The converter applies RoNet's `[in, out]` matrix convention (every matmul
weight is transposed versus PyTorch), reproduces the `Module.collectParams`
parameter order, maps RoPE base and RMS epsilon from the source config, and
honors tied word embeddings. `tools/pt2ronet_check.py` cross-checks a converted
model against an independent numpy forward pass, and a tiny random checkpoint
(`tests/fixtures/pt_tiny`) runs the same path as a suite test
(`tests/_smoke_pt.luau`). Requirements: python3 + numpy. Full reference and a
config mapping table: [PyTorch import guide](/guide/ptimport).

Set expectations honestly: RoNet is a single-process Luau library with no GPU
path, so 2B/4B/7B+ checkpoints are not a realistic target. A 7B model is tens of
gigabytes of weights on disk, millions of weights in one source file, and far
beyond reachable memory once that source, the forward graph, and the logits all
live in one interpreter. The converter is aimed at small and tiny models (a few
million parameters), distilled checkpoints, and studying the internals of a big
model layer by layer. For real serving of large models, use the original
PyTorch implementation.

## Requirements

- A Luau CLI binary (`luau`) and optionally `luau-analyze`/`luau-compile`.
  Copies ship in `.tools/` (0.73x, native codegen enabled). To use them:
  `export LUAU=$(pwd)/.tools/luau` (the test runner honors the `LUAU`
  variable). Newer Luau releases work too; keep the trio version-matched
  (bytecode format changes between releases).
- The entry script is `RoNet.luau`, not `init.luau`: the stock Luau CLI cannot
  `require` a module literally named `init`, so the DI wiring avoids that name.

## Running the tests

```
bash tests/run_all.sh                  # uses `luau` from PATH
LUAU=/path/to/luau bash tests/run_all.sh
LUAU_OPTS="-O2 --codegen" bash tests/run_all.sh   # interpreted O2
bash tests/native.sh                   # full suite under native codegen (O2 + --codegen)
```

Each test runs in its own process (the CLI requires are entry-only). The suite
covers the tensor engine with finite differences, every layer, every optimizer,
the training stack (schedulers, EMA, serialization, losses), the BPE tokenizer,
one end-to-end language model fit followed by generation and checkpoint
round-trips, and a PyTorch checkpoint import smoke test that loads a converted
Llama-style fixture and round-trips every weight through Serialize.

`-O2` optimizes the bytecode; `--codegen` makes the VM translate hot functions
to native x64/aarch64 at load time (typical wall-clock win: ~1.5-2x on this
suite, ~4x on the matmul kernel). The native run is checked by CI via
`tests/native.sh`.

## Big tables and speed

RoNet tensors are plain dense `{number}` arrays (integer keys 1..n), which is
exactly the layout Luau treats best: a contiguous array-part with unboxed-ish
number elements and O(1) reads, no hash nodes, no sparse arrays. Allocation goes
through `table.create(n[, fill])` so big buffers are preallocated instead of
grown one element at a time. Things worth knowing when you push table sizes up:

- Keep numeric tensors dense and 1-based; avoid holes and avoid mixing
  string/number keys into the same table (that forces the array part into the
  slower hash part).
- Allocate with `table.create(n, 0)` before bulk-writing; repeat lengths with a
  `local len = #t` and reuse the local instead of re-indexing `#t` in a loop.
- For pure reads, `for v in t` is the fastest iteration form (indexed loops
  re-cost each read); index loops are still the right call when a kernel writes.
- RoNet exposes `Tensor.matmulFast(a, b)` for inference: a forward-only SIMD
  matmul that packs B's column blocks into Luau `vector`s (3 lanes), so the K
  loop becomes scalar * vector ops — single SSE/AVX instructions under
  `--codegen`. Measured at 512x512x512: ~3x faster than the scalar kernel in
  the interpreter, ~1.75x faster natively. It produces no autograd graph, so use
  it only where gradients are not needed. (`vector` and `buffer` are both
  available in the stock Luau CLI. Roblox scripts expose `buffer` but not Luau's
  `vector`, so there `matmulFast` silently falls back to the scalar kernel with
  identical results.)
- **Inference takes the fast path automatically.** `Transformer:eval()` detaches
  the parameters, so `forward`/`forwardBatch` build no autograd graph and
  `Tensor.matmul` auto-routes to the SIMD kernel; Attention reads the transposed
  K directly (`bmmNT`) and uses forward-only batched matmuls. Fused
  `Tensor.rmsNorm`/`layerNorm`/`rotary` and a linear last-dim `softmax` are used
  in both modes. `train()` restores the stored `requiresGrad` flags.

## Quick start

### A feed-forward classifier

```lua
local deps = require("../RoNet")            -- entry script
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
local deps = require("../RoNet")
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
- **Dropout, EVAL and KV cache.** `Transformer:eval()` toggles dropout off and
  detaches gradients (see the inference note above); `Transformer:train()`
  restores both. `generate` runs the decode phase with an incremental KV cache
  per block (raw K/V append, no repeated attention over the prompt), giving
  per-token cost independent of prompt length; pass `{ cache = false }` to
  replay a full forward per token instead (used by the parity tests).
- **Optimizer API.** Every optimizer exposes `new(params, lr, opts)`, `step`,
  `zeroGrad`, `setLr/getLr`, `paramCount`, and `stateDict/loadStateDict`.
  Schedules attach with `Scheduler.forOptimizer(opt, schedule)` and are stepped
  once per batch.
- **Parallel training.** `RoNet.Parallel` shards `forward`/`loss`/`backward`/
  `scoreGenomes` across Roblox Actor sub-VMs. `Parallel.setMode("multi")`
  rebuilds the batch across a worker pool and returns numbers identical to one
  full-batch call (workers scale losses by shard row count; the pool divides
  summed grads by total rows). `attach(model)` returns a handle with
  `h:loss`/`h:backward`/`h:scoreGenomes`/`h:shutdown`; `backward` writes summed
  gradients into the model so `optimizer:step()` works right after. `"single"`
  (default) and `"auto"` run the same jobs in-process and are the only modes the
  CLI can use. Multi-mode needs Play mode.

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
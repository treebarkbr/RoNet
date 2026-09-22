# Roblox port

RoNet ships as a prebuilt Roblox Model, `builds/RoNet.rbxm`. Drag it into the
Studio **Explorer** anywhere under `ReplicatedStorage` and you get the full
library as ordinary ModuleScripts — same API, same `--!strict` modules, nothing
extra to install.

## Structure

The Model inserts one **ModuleScript** named `RoNet` plus child Folders:

```
ReplicatedStorage/
└── RoNet (ModuleScript — the DI root)
    ├── core/    Util, PRNG, Tensor, Matrix, Parallel
    ├── nn/      Module, Inits, Linear, Activations, Norm, Dropout,
    │            Embedding, RoPE, Attention, FFN, Sequential
    ├── models/  MLP, TransformerBlock, Transformer
    ├── loss/    CrossEntropy, Losses
    ├── optim/   AdamW, AdEMAMix, Lion, NAdamW, CautiousAdamW,
    │            ScheduleFreeAdamW, SOAP, Muon, MuonAdamW
    ├── train/   Scheduler, EMA, Serialize, Trainer
    └── data/    BPE
```

The root is a single ModuleScript that wires the dependency graph, so:

```lua
local RoNet = require(ReplicatedStorage.RoNet)
```

returns the full `deps` table (`RoNet.Util`, `RoNet.Tensor`, `RoNet.MLP`,
`RoNet.Trainer`, …). Because every ModuleScript can `require()` in Roblox, no
factory indirection is lost — all modules stay identical to the CLI source.

## Usage

A minimal classifier, identical to the [interpreter quick start](/guide/quickstart):

```lua
-- ServerScriptService/Train.lua
local RoNet = require(ReplicatedStorage.RoNet)

local model = RoNet.MLP.new("net", { 2, 12, 1 }, RoNet.PRNG.new(1), { act = "tanh" })
local opt = RoNet.AdamW.new(RoNet.Module.collectParams(model), 0.05)

local trainer = RoNet.Trainer.new({
	model = model,
	optimizer = opt,
	lossFn = function(m, xs, ys)
		local out = m:forward(xs)
		return RoNet.Losses.binaryCrossEntropy(out, RoNet.Tensor.fromData(ys, { #ys, 1 }, false))
	end,
	X = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } },
	Y = { 1, 0, 0, 0 },
	epochs = 20,
	batchSize = 4,
	featureShape = { 2 },
})
trainer:fit()
```

The autograd engine, optimizers, Trainer, serialization, and BPE tokenizer all
run in the server runtime. Keep the library in `ReplicatedStorage` so server
Scripts can train; clients can `require()` the same module for inference or to
drive `Transformer:generate()`.

## Parallel (multi-threaded) training

`RoNet.Parallel` offloads the heavy ops — `forward`, `loss`, `backward`, and
`scoreGenomes` — onto Roblox **Actor** sub-VMs. One mode selection governs the
whole library:

```lua
local Parallel = RoNet.Parallel

Parallel.setMode("auto")          -- "single" | "multi" | "auto"
Parallel.setWorkers(4)            -- multi-mode shards the batch across N Actors
```

- `setMode("single")` runs every job in-process, sharded the same way the multi
  pool would shard it, so results are bit-identical to the multi path. This is
  the default and the only mode the CLI build can use.
- `setMode("multi")` builds an Actor pool: a Folder (`RoNet_WorkerPool`) plus
  one Script-backed Actor per worker, placed under `ReplicatedStorage` (move it
  with `Parallel.setParent(instance)`). Jobs are sent by value with
  `Actor:SendMessage`; results come back over a pool-owned `BindableEvent`.
  Requires **Play mode** (Actors do not run in Edit) — get the mode with
  `Parallel.available()`.
- `setMode("auto")` resolves to `multi` on a Roblox runtime and falls back to
  `single` under the CLI.

Attach a model to get a pooled handle:

```lua
-- ServerScriptService/TrainPar.lua
local RoNet = require(ReplicatedStorage.RoNet)

local model = RoNet.MLP.new("net", { 2, 12, 1 }, RoNet.PRNG.new(1), { act = "tanh" })
local opt = RoNet.AdamW.new(RoNet.Module.collectParams(model), 0.05)

local h = RoNet.Parallel.attach(model)          -- opts: { mode, workers, ... }

local loss = h:loss(rows, targets, "mse")       -- rows: {{number}}, targets: flat {number} (f per row) or rows-of-{number}
h:backward(rows, targets, "mse")                -- writes summed grads into `model`, so optimizer:step() works next
local scores = h:scoreGenomes(genomes, rows)    -- per-genome loss on the full batch
h:shutdown()                                    -- disconnects pool state, destroys the worker Folder
```

- Pooled ops return the **same numbers as a single full-batch call**: each
  worker scales its shard’s loss by its row count before `backward`, and the
  pool divides the summed gradients by the total row count.
- Handle ops accept any lossName the engine does: `mse`, `bce`, `crossentropy`.
- The parallel path is pure compute — no `require()`, no Instance mutation
  inside worker jobs — so shards are deterministic.

## Differences from the CLI build

- The stock Luau CLI exposes the `vector` and `buffer` libraries. **Roblox
  scripts do not expose Luau's `vector`** (only `buffer`), so
  `Tensor.matmulFast` automatically falls back to the scalar `Tensor.matmul`
  kernel and returns the exact same values. Under the CLI it still uses the
  SIMD fast path.
- Tests, `tests/`, and the `.tools/` Luau binaries are not shipped in the Model.
- Roblox supports per-module compilation hints (`--!native` / `--!optimize`),
  which the CLI's `-O2 --codegen` flags mirror at load time.

## Rebuilding the Model

`builds/RoNet.rbxm` is generated, not hand-assembled. `build/rbxm/` is a small
Rust tool ([rbx-dom](https://github.com/rojo-rbx/rbx-dom): `rbx_dom_weak` +
`rbx_binary`) that walks the repo exactly as the Model is laid out — the root
ModuleScript is `RoNet.luau` with each `require("./dir/Mod")` rewritten to
`require(script.dir.Mod)`, one Folder per source directory, and one
ModuleScript per `.luau` (files starting with `_` are skipped). To refresh the
artifact after changing source:

```sh
cargo run --release --manifest-path build/rbxm/Cargo.toml
```

It writes `builds/RoNet.rbxm` (LZ4-compressed binary, the same format Studio
exports). Inspect a file with `cargo run --release --manifest-path
build/rbxm/Cargo.toml -- dump builds/RoNet.rbxm`. No Studio session is needed
to rebuild the port.
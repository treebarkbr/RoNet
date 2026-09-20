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
    ├── core/    Util, PRNG, Tensor, Matrix
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

The tree matches the repo `Layout` exactly. To refresh `builds/RoNet.rbxm`
after changing source: create/update the ModuleScripts under
`ReplicatedStorage.RoNet` from the repo files (the root is `RoNet.luau`, whose
`require("./dir/Mod")` lines are written as `require(script.dir.Mod)`), then
right-click **Export Selection… → RoNet.rbxm**.
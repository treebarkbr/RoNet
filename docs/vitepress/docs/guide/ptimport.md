# Import a Hugging Face checkpoint

RoNet's `Transformer` is a GPT-NeoX-layout decoder: RMSNorm + RoPE + grouped
query attention + SwiGLU, which is the same architecture family as Llama 2,
Llama 3, and Mistral. `tools/pt2ronet.py` turns a Hugging Face
`model.safetensors` and its `config.json` into a self-contained Luau module,
so a pretrained model can run in RoNet without reimplementing anything.

```sh
python3 tools/pt2ronet.py model.safetensors --config config.json --out weights.luau
```

What you get is a `--!strict` module:

```lua
return {
    cfg = { C = ..., numBlocks = ..., tieEmbeds = true, ropeBase = ..., rmsEps = ..., ... },
    weights = [[
shape=32000,4096; 0.123 0.456 ...
...
    ]],
}
```

The `weights` string is exactly the `Serialize.dump` format, so it loads
directly into an existing parameter list:

```lua
local deps = require("../RoNet")
local W = require("path.to.weights")

local model = deps.Transformer.new("pt", W.cfg, deps.PRNG.new(1))
deps.Serialize.load(deps.Module.collectParams(model), W.weights)

local logits = model:forward({ 1, 2, 3 })   -- [T, vocab]
```

## Why a Python converter

The stock Luau CLI has no filesystem or socket access, so a checkpoint cannot
be read at `require` time. The bridge splits the work: Python does the file I/O
and applies RoNet's conventions, and the Luau side only ever sees the plain
string format `Serialize` already speaks. This keeps the libraries used by
training (numpy, safetensors parsing) out of the RoNet runtime.

Requirement: python3 with numpy. The converter parses `safetensors` directly
from its header/metadata layout (no `safetensors` package needed).

## CLI reference

```
usage: pt2ronet.py [-h] [--config CONFIG] [--out OUT] [--max-seq MAX_SEQ]
                   [--layers LAYERS] [--vocab VOCAB]
                   safetensors
```

| Argument | Default | Meaning |
|---|---|---|
| `safetensors` (positional) | - | Path to `model.safetensors`. |
| `--config` | sibling `config.json` | Path to the model config. |
| `--out` | stdout | Output `.luau` path. |
| `--max-seq` | derived from config, else 2048 | Context length for the precomputed RoPE tables. |
| `--layers` | all blocks | Keep only the first N decoder blocks (genuine weight slice). |
| `--vocab` | config vocab | Slice the embedding / lm-head rows to `[0, V)` (genuine weight slice). |

Both HF `LlamaForCausalLM` (`model.layers.N....`) and llama2.c
(`layers.N.attention_norm` / `attention.w{q,k,v,o}` / `feed_forward.w{1,2,3}`,
tied to `output.weight`) checkpoint namings are recognized automatically. Verify
an import from either family against an independent numpy forward pass with:

```sh
python3 tools/pt2ronet_check.py \
  --safetensors model.safetensors --config config.json \
  --layers N --vocab V --entry tests/_smoke_pt.luau
```

## Config mapping

Every field of the emitted `cfg` comes from the source `config.json`:

| RoNet `cfg` | HF `config.json` | Notes |
|---|---|---|
| `C` | `hidden_size` | Model width. |
| `numBlocks` | `num_hidden_layers` | Number of decoder blocks. |
| `nbHeads` | `num_attention_heads` | Query heads. |
| `kvHeads` | `num_key_value_heads` | Key/value heads; GQA when smaller than `nbHeads`. |
| `headDim` | `head_dim` | Falls back to `hidden_size / num_attention_heads`; must be even for RoPE. |
| `ffnHidden` | `intermediate_size` | SwiGLU hidden width. |
| `vocab` | `vocab_size` | Vocabulary size. |
| `maxSeq` | `original_max_position_embeddings` or `max_position_embeddings` | Falls back to 2048; `--max-seq` wins. |
| `tieEmbeds` | `tie_word_embeddings` | Defaults to `true`. |
| `ropeBase` | `rope_theta` | RoPE base; defaults to 10000. |
| `rmsEps` | `rms_norm_eps` | RMS normalization epsilon. |
| `dropoutP` | - | Forced to `0.0` so inference matches the source checkpoint. |
| `initType` | - | Defaults to `"default"`; weights are overwritten by the load anyway. |

## Parameter order and orientation

RoNet stores every matmul weight in the `[in, out]` convention, which is the
transpose of PyTorch's `[out, in]`. The converter transposes every linear
weight (`q/k/v/o_proj`, `gate/up/down_proj`) and leaves norm and embedding
tables untouched. It writes the parameters in exactly the order
`Module.collectParams` walks them, so the `Serialize.load` call above maps
everything positionally:

Per block, in order: `input_layernorm` (RMS weight), `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `post_attention_layernorm`, `gate_proj`, `up_proj`,
`down_proj`. Then the lm-head, the final norm, and the embedding table. When
`tie_word_embeddings` is set, the lm-head is the embedding table, so it is
written once and the trailing embedding line is omitted, matching how
`collectParams` deduplicates shared weights.

## Notes and gotchas

- **RoPE base.** Llama 2 uses `rope_theta = 10000`, which is RoNet's default.
  Llama 3 uses `500000`; the config ships that value so `ropeBase` is picked up
  automatically.
- **RMS epsilon.** Hugging Face Llama configs use `rms_norm_eps = 1e-5` while
  RoNet's default epsilon is `1e-6`. `rmsEps` threads the source value into the
  block norms, the final norm, and any model built from the emitted `cfg`.
- **GQA.** `num_key_value_heads < num_attention_heads` maps to RoNet's
  `kvHeads`; the attention kernel shares each key/value head across its query
  group, which is exactly how grouped query attention is defined.
- **Long-context scaling.** RoPE scaling (`rope_scaling`) and rolling/ALiBi-style
  norm variants are not applied. A sequence longer than `--max-seq` errors at
  forward time because the RoPE tables are precomputed up front.
- **Byte-level ids.** The converter maps weights as-is. RoNet token ids are
  1-based; Hugging Face conventions are 0-based, so shift your tokenizer's
  vocab when packing prompts (the known/unused-token positions differ).

## Realistic scale

To be clear about what this is for: RoNet runs in one Luau process on a CPU,
with no GPU path. A 2B/4B/7B+ checkpoint is not something it will ever serve,
and pretending otherwise wastes a lot of RAM and disk:

- A 7B-parameter model is roughly 28 GB of float32 weights before you touch
  activations. Hugging Face ships those as sharded `safetensors` files; a
  single `model.safetensors` of that size cannot even be opened by this
  converter, whose `Serialize` string embeds every weight.
- Even if the weights could be read, the emitted module would be a source file
  with hundreds of millions of numbers, and the interpreter would then hold the
  weight string, the parsed weight tensors, the forward graph, and the logits
  in the same address space.

Treat the importer as a tool for the scale the library actually runs: models of
a few million parameters, checkpoints distilled or pruned down to that size,
and studying the internals of a bigger model one block at a time. If your
workload genuinely needs a large Llama/Mistral model, run it in PyTorch with
the intended serving stack, not here.

## Verification

The import path is checked two ways, both independent of the RoNet runtime:

- `python3 tools/pt2ronet_check.py` reimplements the whole forward pass
  (RMSNorm, RoPE, GQA attention, SwiGLU logits) in numpy over the source
  checkpoint, runs `tests/_exp_pt_import.luau` in the Luau CLI, and compares
  the logits. It passes when the worst relative deviation stays under `1e-3`;
  on the bundled fixture it measures roughly `5e-9`.
- `tests/_smoke_pt.luau` (part of `tests/run_all.sh`) imports the converted
  fixture, round-trips every weight through Serialize exactly, checks tied
  embeddings share one tensor, and runs both single-sequence and batched
  forwards.

```sh
python3 tools/pt2ronet_check.py --luau .tools/luau
LUAU=$PWD/.tools/luau bash tests/run_all.sh
```

## The bundled fixture

`tests/fixtures/pt_tiny` is a deliberately small random Llama-style checkpoint
used by the tests above: hidden size 16, 2 layers, 4 query heads with 2
key/value heads (GQA), head dim 4, SwiGLU hidden 32, vocab 33, `rms_norm_eps`
`1e-5`, `rope_theta` `10000`, tied embeddings. It was generated with numpy
(`default_rng(7)`) and written as a conformant `safetensors` file;
`tests/fixtures/pt_tiny.weights.luau` is the reference output of the converter
that the Luau tests load.
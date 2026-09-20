# RoNetV4.1

A modular neural network library for [Luau](https://luau.org), written as strict
(`--!strict`) modules. It includes an autograd engine, tensor and matrix
kernels, transformer building blocks, a set of optimizers, a training loop, and
a byte-level BPE tokenizer. It runs entirely on the stock Luau command line
interpreter with no external packages.

## What you get

- `core` autograd engine with a Tensor type and a finite-difference-verified
  set of operators
- `nn` layers: Linear, Activations, Norm, Dropout, Embedding, RoPE, Attention
  (causal, grouped-query), FFN (SwiGLU), Sequential
- `models`: MLP, TransformerBlock (pre-norm with ReZero/DeepNorm options), and a
  Transformer with batched forward, shifted next-token loss, and autoregressive
  generation
- `loss`: stable cross-entropy with label smoothing, plus KL, JS, focal, BCE,
  and LM next-token losses
- `optim`: AdamW, AdEMAMix, Lion, NAdamW, CautiousAdamW, ScheduleFreeAdamW,
  SOAP, Muon, MuonAdamW
- `train`: LR schedulers, EMA weight averaging, checkpoint serialization, and a
  batched Trainer loop
- `data`: a deterministic byte-level BPE tokenizer with state persistence

## Exploring the docs

- **[Quick start](/guide/quickstart)** walks through a small classifier and a
  tiny language model.
- **[Roblox port](/guide/roblox)** explains the prebuilt `builds/RoNet.rbxm`
  Model and how to use the library inside a Studio place.
- **[API reference](/api/)** documents every module. The pages are generated
  from the doc comments in the source by `node docs/gen.mjs`.

## Running the tests

```sh
bash tests/run_all.sh                   # uses `luau` from PATH
LUAU=/path/to/luau bash tests/run_all.sh
```

The suite covers the tensor engine with finite differences, every layer, every
optimizer, the training stack, the BPE tokenizer, and one end-to-end language
model fit followed by generation and checkpoint round-trips.
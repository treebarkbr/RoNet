# Pretrained TinyStories example

The repo ships a runnable example, `examples/tinystories.luau`, that loads a
real TinyStories-trained transformer (~15M params, the Karpathy
`stories15M` lineage) through `tools/pt2ronet.py` and streams tokens in real
time with a tokens/second report.

```
Once upon a time, there was a little girl named Lily. She lived in a small
house with her mother. One day, her mother wanted to cook dinner for the
family. But her mother said, "Lily, we have to wait for the food to be ready."
```

## Run it

The weight fixtures are gitignored — the full 15M model serializes to ~200 MB
of text in `Serialize`'s format — so generating them is part of the setup.
Everything runs once:

```sh
# 1. Grab the checkpoint (llama2.c format, so it fits RoNet's Llama-family
#    architecture with tied embeddings).
mkdir -p /tmp/tinystories
curl -L -o /tmp/tinystories/model.safetensors \
  https://huggingface.co/0rn0/llama2-15m-tinystories/resolve/main/model.safetensors
curl -L -o /tmp/tinystories/config.json \
  https://huggingface.co/0rn0/llama2-15m-tinystories/resolve/main/config.json
curl -L -o /tmp/tinystories/tokenizer.model \
  https://huggingface.co/0rn0/llama2-15m-tinystories/resolve/main/tokenizer.model

# 2. Convert the safetensors into a Luau weight module.
python3 tools/pt2ronet.py /tmp/tinystories/model.safetensors \
  --config /tmp/tinystories/config.json --out tests/fixtures/tinystories15m.weights

# 3. Stream: each sampled token prints with a running clock, then a STATS
#    line reports tokens/s. Pipe through ts_decode.py for readable text.
.tools/luau -O2 --codegen examples/tinystories.luau \
  | python3 tools/ts_decode.py /tmp/tinystories/tokenizer.model
```

Output looks like:

```
loading weights ...
loaded ts (15.19M floats, 56 tensors) in 20.6s
generating +40 tokens (temp=0.80, topK=50) ...
tok   1/40  (+11.46s)  id 3119
tok   2/40  (+11.99s)  id 2463
...
IDS 9039 2502 264 932 ... 74611 323
STATS generated=40 tokens in 33.7s = 1.19 tokens/s (full sequence incl. seed: 64)
```

The token ids are RoNet's 1-based ids; `tools/ts_decode.py` subtracts 1 and
decodes with the checkpoint's SentencePiece model so ids can round-trip through
a text pipe.

## Faster slice + faster demo

The converter can slice genuine weights for a portable smoke-run fixture (2 of
6 blocks, first 2048 vocab rows), which loads in ~3s and generates at ~11 tok/s
instead of ~1.2 tok/s — at the cost of coherence (most of the model's story
vocabulary and layers are gone, so text degenerates into repetition):

```sh
python3 tools/pt2ronet.py /tmp/tinystories/model.safetensors \
  --config /tmp/tinystories/config.json \
  --layers 2 --vocab 2048 --out tests/fixtures/tinystories15m.demo.weights
```

Point `WEIGHTS_PATH` in `examples/tinystories.luau` at
`tinystories15m.demo.weights` to use it.

## Architectures

`tools/pt2ronet.py` understands both checkpoint families:

| Family | Tensor names | Embeddings |
|---|---|---|
| HF `LlamaForCausalLM` | `model.layers.N.{input_layernorm,self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj}` | `model.embed_tokens.weight`, optional `lm_head.weight` |
| llama2.c | `layers.N.{attention_norm,attention.w{q,k,v,o},ffn_norm,feed_forward.w{1,2,3}}` | tied `output.weight` (no separate table) |

Both paths transpose every projection `[out,in] -> [in,out]`, emit weights in
the exact `Module.collectParams` order, and derive `ffnHidden` from the gate
matrix when the config omits it. Exactness is guarded by
`tools/pt2ronet_check.py`, which reimplements the forward in numpy and compares
to RoNet's logits (imports agree to ~1e-5 relative).

## Streaming API

`Transformer.generate` accepts `onToken(rid, step)` in `opts`, called the
moment a token is sampled inside the KV-cache decode loop:

```lua
model:generate(seed, 40, {
  temperature = 0.8, topK = 50, cache = true,
  onToken = function(rid, step)
    print("tok " .. step .. "/40  id " .. rid)
  end,
})
```
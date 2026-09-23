#!/usr/bin/env python3
"""Convert a Llama-style safetensors checkpoint into a RoNet weight module.

Reads a `model.safetensors` + `config.json`, applies RoNet's parameter ordering
and [in, out] matrix convention (every linear weight is transposed vs PyTorch),
and emits a self-contained Luau module:

    return {
        cfg = { C = ..., numBlocks = ..., ... },   -- exact RoNet Transformer cfg
        weights = [[                             -- Serialize.dump format
shape=...; ...
        ]],
    }

Usage:
    python3 tools/pt2ronet.py model.safetensors \\
        [--config config.json] [--out weights.luau] [--max-seq 4096] \\
        [--layers N] [--vocab V]

Architectures (auto-detected from config.json):
    llama   HF LlamaForCausalLM naming (embed_tokens/layers.N/self_attn/mlp/...)
    llama2c llama2.c naming (attention_norm/ffn_norm/feed_forward.w1..w3) --
            the layout used by the TinyStories-15M / stories15M checkpoints.
            A missing tok_embeddings with only `output.weight` is handled as
            tied embeddings (the shared table is emitted once).

--layers N keeps only the first N transformer blocks; --vocab V slices the
embedding/lm-head rows to [0, V). Both keep genuine weights for a portable
demo. On the RoNet side:

    local W = require(path.to.weights)
    local model = deps.Transformer.new("pt", W.cfg, rng)
    deps.Serialize.load(deps.Module.collectParams(model), W.weights)

Requires numpy. Parses safetensors directly (no safetensors package needed).
"""

import argparse
import json
import sys

import numpy as np

PT = "<f4"  # safetensors float32 little-endian


def read_safetensors(path):
    with open(path, "rb") as f:
        header_len = np.frombuffer(f.read(8), dtype="<u8")[0]
        header = json.loads(f.read(int(header_len)))
        data = f.read()
    return header, data


def load_tensors(header, data):
    out = {}
    for key, meta in header.items():
        if not isinstance(meta, dict) or "shape" not in meta:
            continue
        count = 1
        for d in meta["shape"]:
            count *= d
        begin, _end = meta["data_offsets"]
        out[key] = np.frombuffer(data, dtype=PT, count=count, offset=begin).reshape(meta["shape"])
    return out


def roNet_cfg(cfg, max_seq_arg):
    if "hidden_size" in cfg:
        # --- HF LlamaForCausalLM ---
        nb = int(cfg["num_attention_heads"])
        C = int(cfg["hidden_size"])
        kv = int(cfg.get("num_key_value_heads", nb))
        hd = int(cfg.get("head_dim", C // nb))
        max_seq = max_seq_arg if max_seq_arg is not None else int(
            cfg.get("original_max_position_embeddings", cfg.get("max_position_embeddings", 2048))
        )
        return {
            "arch": "llama",
            "C": C,
            "numBlocks": int(cfg["num_hidden_layers"]),
            "nbHeads": nb,
            "kvHeads": kv,
            "headDim": hd,
            "ffnHidden": int(cfg["intermediate_size"]),
            "vocab": int(cfg["vocab_size"]),
            "maxSeq": max_seq,
            "tieEmbeds": bool(cfg.get("tie_word_embeddings", True)),
            "ropeBase": float(cfg.get("rope_theta", 10000.0)),
            "rmsEps": float(cfg.get("rms_norm_eps", 1e-6)),
        }
    # --- llama2.c --- (karpathy TinyStories-15M family: dim/n_layers/n_heads)
    nb = int(cfg["n_heads"])
    C = int(cfg["dim"])
    kv = int(cfg.get("n_kv_heads", nb))
    hd = C // nb
    return {
        "arch": "llama2c",
        "C": C,
        "numBlocks": int(cfg["n_layers"]),
        "nbHeads": nb,
        "kvHeads": kv,
        "headDim": hd,
        "ffnHidden": -1,  # filled from w1 rows after loading
        "vocab": int(cfg["vocab_size"]),
        "maxSeq": max_seq_arg if max_seq_arg is not None else int(cfg.get("max_seq_len", 256)),
        "tieEmbeds": True,  # llama2.c packs a single shared vocab table
        "ropeBase": float(cfg.get("rope_theta", 10000.0)),
        "rmsEps": float(cfg.get("norm_eps", 1e-5)),
    }


def serialize_line(arr):
    # float32 has at most ~9 significant decimal digits; %.9g round-trips an
    # f32 through float64 exactly and keeps the generated weight text compact.
    vals = " ".join(format(float(v), ".9g") for v in arr.ravel())
    if arr.ndim == 1:
        return f"shape={arr.shape[0]}; {vals}"
    return f"shape={arr.shape[0]},{arr.shape[1]}; {vals}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("safetensors", help="path to model.safetensors")
    ap.add_argument(
        "--config", default=None, help="path to config.json (default: sibling of the safetensors)"
    )
    ap.add_argument("--out", default=None, help="output .luau path (default: stdout)")
    ap.add_argument(
        "--max-seq", type=int, default=None, help="context length for RoPE tables (default: config or 2048)"
    )
    ap.add_argument(
        "--layers", type=int, default=None, help="keep only the first N transformer blocks"
    )
    ap.add_argument(
        "--vocab", type=int, default=None, help="slice embedding/lm-head rows to [0, V)"
    )
    args = ap.parse_args()

    cfg_path = args.config or (args.safetensors.rsplit("/", 1)[0] + "/config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    rc = roNet_cfg(cfg, args.max_seq)
    tie = rc["tieEmbeds"]

    header, data = read_safetensors(args.safetensors)
    tensors = load_tensors(header, data)

    if rc["arch"] == "llama":
        TPL = {
            "input_layernorm": "model.layers.%d.input_layernorm.weight",
            "post_attention_layernorm": "model.layers.%d.post_attention_layernorm.weight",
            "q_proj": "model.layers.%d.self_attn.q_proj.weight",
            "k_proj": "model.layers.%d.self_attn.k_proj.weight",
            "v_proj": "model.layers.%d.self_attn.v_proj.weight",
            "o_proj": "model.layers.%d.self_attn.o_proj.weight",
            "gate_proj": "model.layers.%d.mlp.gate_proj.weight",
            "up_proj": "model.layers.%d.mlp.up_proj.weight",
            "down_proj": "model.layers.%d.mlp.down_proj.weight",
        }
        final_norm = "model.norm.weight"
        embed_key = "model.embed_tokens.weight"
        lm_key = "lm_head.weight"
    else:
        TPL = {
            "input_layernorm": "layers.%d.attention_norm.weight",
            "post_attention_layernorm": "layers.%d.ffn_norm.weight",
            "q_proj": "layers.%d.attention.wq.weight",
            "k_proj": "layers.%d.attention.wk.weight",
            "v_proj": "layers.%d.attention.wv.weight",
            "o_proj": "layers.%d.attention.wo.weight",
            "gate_proj": "layers.%d.feed_forward.w1.weight",
            "up_proj": "layers.%d.feed_forward.w3.weight",
            "down_proj": "layers.%d.feed_forward.w2.weight",
        }
        final_norm = "norm.weight"
        embed_key = "tok_embeddings.weight"
        lm_key = "output.weight"

    C, kv, nb, hd = rc["C"], rc["kvHeads"], rc["nbHeads"], rc["headDim"]
    vocab = rc["vocab"]
    ffn = rc["ffnHidden"]
    if ffn < 0:
        ffn = tensors[TPL["gate_proj"] % 0].shape[0]  # llama2.c: w1 rows
        rc["ffnHidden"] = ffn

    # Optional genuine-weight slicing for a portable demo.
    n_blocks = args.layers if args.layers is not None else rc["numBlocks"]
    if n_blocks > rc["numBlocks"] or n_blocks <= 0:
        sys.exit(f"--layers {n_blocks} out of range [1, {rc['numBlocks']}]")
    rc["numBlocks"] = n_blocks
    if args.vocab is not None:
        if args.vocab > vocab or args.vocab <= 0:
            sys.exit(f"--vocab {args.vocab} out of range [1, {vocab}]")
        vocab = args.vocab
        rc["vocab"] = vocab

    expect = {embed_key: [vocab, C], final_norm: [C]}
    for i in range(n_blocks):
        expect[TPL["input_layernorm"] % i] = [C]
        expect[TPL["post_attention_layernorm"] % i] = [C]
        expect[TPL["q_proj"] % i] = [hd * nb, C]
        expect[TPL["k_proj"] % i] = [hd * kv, C]
        expect[TPL["v_proj"] % i] = [hd * kv, C]
        expect[TPL["o_proj"] % i] = [C, hd * nb]
        expect[TPL["gate_proj"] % i] = [ffn, C]
        expect[TPL["up_proj"] % i] = [ffn, C]
        expect[TPL["down_proj"] % i] = [C, ffn]

    missing = [k for k in expect if k not in tensors]
    # Tied checkpoints may ship the shared table under the lm-head name only
    # (llama2.c `output.weight`: no tok_embeddings tensor at all), or exclude
    # lm_head.weight because it is the embedding table.
    shared_from_lm = tie and lm_key in tensors and embed_key not in tensors
    if tie:
        missing = [k for k in missing if k not in (embed_key, lm_key)]
    elif shared_from_lm or lm_key not in expect:
        missing = [k for k in missing if k != lm_key]
    if missing:
        sys.exit("missing tensors: " + ", ".join(sorted(missing)))
    for k, shp in expect.items():
        t = tensors.get(k)
        if t is not None and list(t.shape) != shp:
            sys.exit(f"shape mismatch {k}: {list(t.shape)} != {shp}")

    if tie:
        lm = tensors.get(lm_key, tensors.get(embed_key))
    else:
        lm = tensors.get(lm_key)
        if lm is None:
            sys.exit("embeddings are untied but lm_head.weight missing")
    if lm is None:
        sys.exit("no lm_head or embedding table found")

    table = lm[:vocab]

    # RoNet param order (matches Module.collectParams): per block the 9 tensors,
    # then the lm-head, final norm, embedding table. When tied, the lm-head IS
    # the embedding table, so it is emitted once (deduped), matching collectParams.
    lines = []
    for i in range(n_blocks):
        for part, do_t in (
            ("input_layernorm", False),
            ("q_proj", True), ("k_proj", True), ("v_proj", True), ("o_proj", True),
            ("post_attention_layernorm", False),
            ("gate_proj", True), ("up_proj", True), ("down_proj", True),
        ):
            arr = tensors[TPL[part] % i]
            if do_t:
                arr = arr.T
            lines.append(serialize_line(arr))

    lines.append(serialize_line(table))                      # lm-head (tied = embedding)
    lines.append(serialize_line(tensors[final_norm]))
    if not tie:
        lines.append(serialize_line(tensors[embed_key][:vocab]))

    cfg_lines = ["\tcfg = {"]
    for k in ("C", "numBlocks", "nbHeads", "kvHeads", "headDim", "ffnHidden", "vocab", "maxSeq"):
        cfg_lines.append(f"\t\t{k} = {rc[k]},")
    cfg_lines.append("\t\ttieEmbeds = " + repr(rc["tieEmbeds"]).lower() + ",")
    cfg_lines.append("\t\tdropoutP = 0.0,")
    cfg_lines.append(f"\t\tropeBase = {rc['ropeBase']!r},")
    cfg_lines.append(f"\t\trmsEps = {rc['rmsEps']!r},")
    cfg_lines.append("\t},")

    text = (
        "--!strict\n"
        "-- Auto-generated by tools/pt2ronet.py -- do not edit by hand.\n"
        f"-- Source: {args.safetensors}\n"
        f"-- Config: {cfg_path}\n"
        "return {\n"
        + "\n".join(cfg_lines)
        + "\n\tweights = [[\n"
        + "\n".join(lines)
        + "\n\t]],\n"
        "}\n"
    )

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
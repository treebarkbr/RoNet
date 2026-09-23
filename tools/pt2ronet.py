#!/usr/bin/env python3
"""Convert an HF Llama-style safetensors checkpoint into a RoNet weight module.

Reads a `model.safetensors` + `config.json`, applies RoNet's parameter ordering
and [in, out] matrix convention (every linear weight is transposed vs PyTorch),
and emits a self-contained Luau module:

    return {
        cfg = { C = ..., numBlocks = ..., ... },   -- exact RoNet Transformer cfg
        weights = [[                                 -- Serialize.dump format
shape=...; ...
        ]],
    }

Usage:
    python3 tools/pt2ronet.py model.safetensors \\
        [--config config.json] [--out weights.luau] [--max-seq 4096]

On the RoNet side:

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
        if not isinstance(meta, dict):
            continue
        count = 1
        for d in meta["shape"]:
            count *= d
        begin, _end = meta["data_offsets"]
        out[key] = np.frombuffer(data, dtype=PT, count=count, offset=begin).reshape(meta["shape"])
    return out


def roNet_cfg(cfg, max_seq_arg):
    nb = int(cfg["num_attention_heads"])
    C = int(cfg["hidden_size"])
    kv = int(cfg.get("num_key_value_heads", nb))
    hd = int(cfg.get("head_dim", C // nb))
    max_seq = (
        max_seq_arg
        if max_seq_arg is not None
        else int(cfg.get("original_max_position_embeddings", cfg.get("max_position_embeddings", 2048)))
    )
    return {
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


def serialize_line(arr):
    if arr.ndim == 1:
        head = f"shape={arr.shape[0]}; "
    else:
        head = f"shape={arr.shape[0]},{arr.shape[1]}; "
    return head + " ".join(repr(float(v)) for v in arr.ravel())


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
    args = ap.parse_args()

    cfg_path = args.config or (args.safetensors.rsplit("/", 1)[0] + "/config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    rc = roNet_cfg(cfg, args.max_seq)
    tie = rc["tieEmbeds"]

    header, data = read_safetensors(args.safetensors)
    tensors = load_tensors(header, data)

    C, kv, nb, hd, ffn, vocab = rc["C"], rc["kvHeads"], rc["nbHeads"], rc["headDim"], rc["ffnHidden"], rc["vocab"]
    expect = {"model.embed_tokens.weight": [vocab, C], "model.norm.weight": [C]}
    for i in range(rc["numBlocks"]):
        expect[f"model.layers.{i}.input_layernorm.weight"] = [C]
        expect[f"model.layers.{i}.post_attention_layernorm.weight"] = [C]
        expect[f"model.layers.{i}.self_attn.q_proj.weight"] = [hd * nb, C]
        expect[f"model.layers.{i}.self_attn.k_proj.weight"] = [hd * kv, C]
        expect[f"model.layers.{i}.self_attn.v_proj.weight"] = [hd * kv, C]
        expect[f"model.layers.{i}.self_attn.o_proj.weight"] = [C, hd * nb]
        expect[f"model.layers.{i}.mlp.gate_proj.weight"] = [ffn, C]
        expect[f"model.layers.{i}.mlp.up_proj.weight"] = [ffn, C]
        expect[f"model.layers.{i}.mlp.down_proj.weight"] = [C, ffn]

    missing = [k for k in expect if k not in tensors]
    if tie:
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        sys.exit("missing tensors: " + ", ".join(sorted(missing)))
    for k, shp in expect.items():
        if k in tensors and list(tensors[k].shape) != shp:
            sys.exit(f"shape mismatch {k}: {list(tensors[k].shape)} != {shp}")

    lm = tensors.get("lm_head.weight", tensors["model.embed_tokens.weight"]) if tie else tensors.get("lm_head.weight")
    if lm is None:
        sys.exit("tie_word_embeddings=false but lm_head.weight missing")

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
    # RoNet param order (matches Module.collectParams): per block the 9 tensors,
    # then the lm-head, final norm, embedding table. When tied, the lm-head IS
    # the embedding table, so it is emitted once (deduped), matching collectParams.
    lines = []
    for i in range(rc["numBlocks"]):
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

    lines.append(serialize_line(lm))                      # lm-head (tied = embedding)
    lines.append(serialize_line(tensors["model.norm.weight"]))
    if not tie:
        lines.append(serialize_line(tensors["model.embed_tokens.weight"]))

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
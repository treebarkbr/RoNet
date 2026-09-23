#!/usr/bin/env python3
"""Cross-check a pt2ronet.py import against an independent numpy reference.

Runs the given Luau entry script (which prints a LOGITS line = the last token's
logits [V] from RoNet's forward on a fixed prompt), reimplements the same
forward pass in numpy (RMSNorm + RoPE + GQA attention + SwiGLU), and reports the
worst-case deviation. Supports both HF LlamaForCausalLM and llama2.c checkpoint
naming, and the sliced imports produced by `--layers` / `--vocab`.

Usage:
    python3 tools/pt2ronet_check.py
      [--safetensors path] [--config path] [--entry tests/_exp_....luau]
      [--layers N] [--vocab V] [--prompt ronet,ids] [--luau .tools/luau]
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

# llama2.c-style tensor names (stories15M lineage). HF names are formed by the
# generic mapping further down.
WQ = "layers.{}.attention.wq.weight"
WK = "layers.{}.attention.wk.weight"
WV = "layers.{}.attention.wv.weight"
WO = "layers.{}.attention.wo.weight"
W1 = "layers.{}.feed_forward.w1.weight"
W2 = "layers.{}.feed_forward.w2.weight"
W3 = "layers.{}.feed_forward.w3.weight"
AN = "layers.{}.attention_norm.weight"
FN = "layers.{}.ffn_norm.weight"
NORM = "norm.weight"
OUT = "output.weight"


def load_tensor(path, key):
    with open(path, "rb") as f:
        hlen = np.frombuffer(f.read(8), dtype="<u8")[0]
        header = json.loads(f.read(int(hlen)))
        data = f.read()
    m = header[key]
    begin, _end = m["data_offsets"]
    count = 1
    for d in m["shape"]:
        count *= d
    return np.frombuffer(data, dtype="<f4", count=count, offset=begin).reshape(m["shape"]).astype(np.float64)


def reference_logits(safetensors, cfg, layers, vocab, prompt):
    T_ = lambda k: load_tensor(safetensors, k)
    C = cfg.get("dim", cfg.get("hidden_size"))
    nb = cfg.get("n_heads", cfg.get("num_attention_heads"))
    kv = cfg.get("n_kv_heads", cfg.get("num_key_value_heads", nb))
    hd = cfg.get("head_dim", C // nb)
    eps = cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-5))
    base = cfg.get("rope_theta", 10000.0)

    is_llama2c = False
    try:
        T_(WQ.format(0))
        is_llama2c = True
    except KeyError:
        is_llama2c = False

    if is_llama2c:
        F = {  # llama2.c (stories15M lineage)
            "q": WQ, "k": WK, "v": WV, "o": WO,
            "g": W1, "u": W3, "d": W2,  # gate/up/down
            "an": AN, "fn": FN, "norm": NORM,
        }
        emb_key, lm_key = OUT, OUT
    else:  # HF LlamaForCausalLM
        F = {
            "q": "model.layers.{}.self_attn.q_proj.weight",
            "k": "model.layers.{}.self_attn.k_proj.weight",
            "v": "model.layers.{}.self_attn.v_proj.weight",
            "o": "model.layers.{}.self_attn.o_proj.weight",
            "g": "model.layers.{}.mlp.gate_proj.weight",
            "u": "model.layers.{}.mlp.up_proj.weight",
            "d": "model.layers.{}.mlp.down_proj.weight",
            "an": "model.layers.{}.input_layernorm.weight",
            "fn": "model.layers.{}.post_attention_layernorm.weight",
            "norm": "model.norm.weight",
        }
        emb_key, lm_key = "model.embed_tokens.weight", "lm_head.weight"
        try:
            T_(lm_key)
        except KeyError:
            lm_key = emb_key  # tied HF checkpoints may omit lm_head.weight

    ids = np.array(prompt)
    Tseq = len(ids)

    def rmsnorm(x, g):
        ms = np.mean(x * x, axis=-1, keepdims=True)
        return x * (ms + eps) ** -0.5 * g

    def rope(a, pos):
        shp = a.shape
        half = hd // 2
        freqs = pos[:, None] / (base ** (2 * np.arange(half) / hd))
        cs, sn = np.cos(freqs), np.sin(freqs)
        a = a.reshape(*shp[:-1], half, 2)
        even, odd = a[..., 0], a[..., 1]
        return np.stack([even * cs - odd * sn, even * sn + odd * cs], axis=-1).reshape(*shp[:-1], hd)

    def silu(z):
        return z * (1 / (1 + np.exp(-z)))

    x = T_(emb_key)[ids - 1]  # RoNet ids are 1-based
    for i in range(layers):
        Wq = T_(F["q"].format(i)).T
        Wk = T_(F["k"].format(i)).T
        Wv = T_(F["v"].format(i)).T
        Wo = T_(F["o"].format(i)).T
        W1_ = T_(F["g"].format(i)).T
        W2_ = T_(F["d"].format(i)).T
        W3_ = T_(F["u"].format(i)).T

        xn = rmsnorm(x, T_(F["an"].format(i)))
        Q = rope((xn @ Wq).reshape(Tseq, nb, hd).transpose(1, 0, 2), np.arange(Tseq))
        K = rope((xn @ Wk).reshape(Tseq, kv, hd).transpose(1, 0, 2), np.arange(Tseq))
        Vx = (xn @ Wv).reshape(Tseq, kv, hd).transpose(1, 0, 2)
        K = np.repeat(K, nb // kv, axis=0)
        Vx = np.repeat(Vx, nb // kv, axis=0)
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)
        S = S + np.where(np.arange(Tseq)[None, :] > np.arange(Tseq)[:, None], -1e30, 0.0)
        e = np.exp(S - S.max(axis=-1, keepdims=True))
        P = e / e.sum(axis=-1, keepdims=True)
        x = x + (P @ Vx).transpose(1, 0, 2).reshape(Tseq, nb * hd) @ Wo

        hx = rmsnorm(x, T_(F["fn"].format(i)))
        x = x + (silu(hx @ W1_) * (hx @ W3_)) @ W2_

    x = rmsnorm(x, T_(F["norm"]))
    logits = x @ T_(lm_key).T  # [Tseq, vocab]
    return logits[-1][:vocab]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--luau", default=os.path.join(".tools", "luau"))
    ap.add_argument("--safetensors", default=os.path.join(FIXTURE_DEFAULT, "model.safetensors"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--entry", default="tests/_exp_pt_import.luau")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--prompt", default="1,2,3,4,5,6,7,8")
    args = ap.parse_args()

    cfg_path = args.config or (os.path.dirname(args.safetensors) + "/config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    layers = args.layers if args.layers is not None else cfg.get("n_layers", cfg.get("num_hidden_layers"))
    vocab = args.vocab if args.vocab is not None else cfg.get("vocab_size")
    prompt = [int(v) for v in args.prompt.split(",")]

    ref = reference_logits(args.safetensors, cfg, layers, vocab, prompt)

    proc = subprocess.run([args.luau, args.entry], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit("luau failed:\n" + proc.stderr)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("LOGITS ")), None)
    if line is None:
        sys.exit("no LOGITS line in output:\n" + proc.stdout[-2000:])
    got = np.array([float(v) for v in line[len("LOGITS "):].split()], dtype=np.float64)
    if got.size != ref.size:
        sys.exit(f"size mismatch: got {got.size} logits, reference {ref.size}")

    diff = np.abs(got - ref)
    scale = np.maximum(1.0, np.abs(ref))
    worst = float(diff.max())
    worst_rel = float((diff / scale).max())
    print(f"numpy reference: {ref.shape}, {ref.size} logits (last token)")
    print(f"max abs  deviation: {worst:.3e}")
    print(f"max rel  deviation: {worst_rel:.3e}")
    if worst_rel < 1e-3:
        print("OK: RoNet logits match the independent numpy reference")
        return 0
    print("MISMATCH: RoNet forward disagrees with the numpy reference")
    return 1


FIXTURE_DEFAULT = "tests/fixtures/pt_tiny"

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Cross-check a pt2ronet.py import against an independent numpy reference.

Runs tests/_exp_pt_import.luau in the Luau CLI to get the RoNet transformer's
logits for a fixed prompt (ids 1..8) on tests/fixtures/pt_tiny, reimplements the
same forward pass in numpy (RMSNorm + RoPE + GQA attention + SwiGLU), and
reports the worst-case deviation. The transpose/ordering conventions of the
importer are correct iff the two agree to float32-scale tolerance.

Usage:
    python3 tools/pt2ronet_check.py [--luau .tools/luau]
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

FIXTURE = "tests/fixtures/pt_tiny"
PROMPT = list(range(1, 9))  # token ids 1..8


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


def reference_logits(safetensors):
    T_ = lambda k: load_tensor(safetensors, k)
    C, nb, kv, hd = 16, 4, 2, 4
    eps, base = 1e-5, 10000.0
    ids = np.array(PROMPT)
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

    x = T_("model.embed_tokens.weight")[ids - 1]  # RoNet ids are 1-based
    for i in range(2):
        Wq = T_(f"model.layers.{i}.self_attn.q_proj.weight").T
        Wk = T_(f"model.layers.{i}.self_attn.k_proj.weight").T
        Wv = T_(f"model.layers.{i}.self_attn.v_proj.weight").T
        Wo = T_(f"model.layers.{i}.self_attn.o_proj.weight").T
        W1 = T_(f"model.layers.{i}.mlp.gate_proj.weight").T
        W2 = T_(f"model.layers.{i}.mlp.up_proj.weight").T
        W3 = T_(f"model.layers.{i}.mlp.down_proj.weight").T

        xn = rmsnorm(x, T_(f"model.layers.{i}.input_layernorm.weight"))
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

        hx = rmsnorm(x, T_(f"model.layers.{i}.post_attention_layernorm.weight"))
        x = x + (silu(hx @ W1) * (hx @ W2)) @ W3

    x = rmsnorm(x, T_("model.norm.weight"))
    return x @ T_("model.embed_tokens.weight").T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--luau", default=os.path.join(".tools", "luau"))
    ap.add_argument("--fixture", default=FIXTURE)
    args = ap.parse_args()

    ref = reference_logits(os.path.join(args.fixture, "model.safetensors"))

    proc = subprocess.run(
        [args.luau, "tests/_exp_pt_import.luau"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit("luau failed:\n" + proc.stderr)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("LOGITS ")), None)
    if line is None:
        sys.exit("no LOGITS line in output:\n" + proc.stdout)
    got = np.array([float(v) for v in line[len("LOGITS "):].split()], dtype=np.float64).reshape(ref.shape)

    diff = np.abs(got - ref)
    scale = np.maximum(1.0, np.abs(ref))
    worst = float(diff.max())
    worst_rel = float((diff / scale).max())
    n = int(diff.size)
    print(f"numpy reference shape: {ref.shape}, {n} logits")
    print(f"max abs  deviation: {worst:.3e}")
    print(f"max rel  deviation: {worst_rel:.3e}")
    if worst_rel < 1e-3:
        print("OK: RoNet logits match the independent numpy reference")
        return 0
    print("MISMATCH: RoNet forward disagrees with the numpy reference")
    return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Decode the IDS line emitted by examples/tinystories.luau to readable text.

Standalone decoder so the Luau example never depends on a Python tokenizer.
Reads whitespace-separated token ids (RoNet's 1-based ids; the "IDS " prefix, if
present, is ignored) from stdin and decodes with a SentencePiece tokenizer.model.

Usage:
    ... | python3 tools/ts_decode.py /path/to/tokenizer.model
    python3 tools/ts_decode.py /path/to/tokenizer.model ids.txt
"""

import sys

try:
    import sentencepiece as spm
except ImportError:
    sys.exit("require: pip install sentencepiece (or use `pipx run`)")

def read_ids(path):
    parts = []
    for line in open(path) if path else sys.stdin:
        if line.startswith("IDS "):
            line = line[len("IDS "):]
        parts.extend(line.split())
    return [int(v) for v in parts]

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_path = sys.argv[1]
    ids = read_ids(sys.argv[2] if len(sys.argv) > 2 else None)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    spm_ids = [i - 1 for i in ids]  # RoNet ids are 1-based
    print(sp.DecodeIds(spm_ids))

if __name__ == "__main__":
    main()
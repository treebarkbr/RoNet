#!/usr/bin/env bash
# Run the full test suite under native code generation with full optimization:
#   -O2            bytecode optimization level 2
#   --codegen      JIT-style native codegen in the VM (x64/aarch64)
# Requires the 0.73x Luau from .tools/ (older builds lack --codegen-cold).
set -u
cd "$(dirname "$0")/.."
export LUAU="${LUAU:-$PWD/.tools/luau}"
export LUAU_OPTS="-O2 --codegen"
bash tests/run_all.sh
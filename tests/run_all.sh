#!/usr/bin/env bash
# Run every entry test in this repo. Each test is a standalone entry script
# (the Luau CLI only allows requires from the entry file), so each runs in its
# own luau process. Set LUAU=/path/to/luau to override the binary, and
# LUAU_OPTS for extra CLI flags (e.g. LUAU_OPTS="-O2 --codegen" for native).
set -u
cd "$(dirname "$0")/.."
LUAU_CMD="${LUAU:-luau}"
FAILED=0
# shellcheck disable=SC2086 # deliberate word-splitting of LUAU_OPTS
for t in RoNet.luau core/_smoke.luau tests/_smoke_layers.luau tests/_smoke_optim.luau tests/_smoke_train.luau tests/_smoke_bpe.luau tests/_smoke_lm.luau; do
	if [ ! -f "$t" ]; then
		continue
	fi
	printf '== %s ==\n' "$t"
	if timeout 300 "$LUAU_CMD" ${LUAU_OPTS:-} "$t"; then
		echo "PASS $t"
	else
		echo "FAIL $t"
		FAILED=1
	fi
done
exit $FAILED
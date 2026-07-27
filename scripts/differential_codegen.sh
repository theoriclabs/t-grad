#!/usr/bin/env bash
# Behavioural equivalence between the captured tinygrad kernels and the
# Lean-generated ones.
#
# This is the successor to L12's source-byte-equality predicate. That
# predicate asserts `renderKernel` output == the captured `.msl` bytes,
# but the former per-shape declaration table was a transcription OF
# those captures, so it was a round trip: it proved a transpiler and a
# renderer are mutual inverses. It says nothing about whether a
# *generated* kernel computes the right thing, and it necessarily dies
# the moment kernels stop being transcribed.
#
# What this checks instead: both kernels execute on one pair of seeded
# bf16 inputs and their output buffers must be bit-identical, while
# their *sources* must differ. Bit-identity against tinygrad's actual
# kernel is stronger than the existing `np.allclose`-vs-numpy check,
# which compares against a reference that could share a tiling bug with
# the code under test.
#
# Usage:
#   bash scripts/differential_codegen.sh            # all sentinels
#   bash scripts/differential_codegen.sh 64x64x64   # one shape
#
# Temp files go in a run-scoped mktemp dir on purpose. The gate scripts
# use 141 distinct hardcoded /tmp/tgrad_* paths, which is why two
# concurrent gate runs clobber each other; this one is safe to run
# alongside anything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TGRAD_CLI="${TGRAD_CLI:-$REPO_ROOT/.lake/build/bin/tgrad-cli}"
SEED="${TGRAD_DIFF_SEED:-42}"

if [[ ! -x "$TGRAD_CLI" ]]; then
  echo "  ✗ tgrad-cli not built at $TGRAD_CLI" >&2
  exit 1
fi

TMPDIR_RUN="$(mktemp -d -t tgrad_diffcodegen)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

if [[ $# -gt 0 ]]; then
  SHAPES=("$@")
else
  SHAPES=(
    64x64x64
    1024x1024x1024 1024x1024x2048 1024x1024x4096 1024x1024x8192
    2048x1024x1024 2048x2048x2048
    4096x1024x1024 4096x4096x4096
    8192x1024x1024 8192x8192x8192
  )
fi

n_ok=0
n_fail=0

for shape in "${SHAPES[@]}"; do
  out="$TMPDIR_RUN/$shape.txt"
  if ! "$TGRAD_CLI" matmul-differential --shape "$shape" --seed "$SEED" >"$out" 2>&1; then
    echo "  ✗ $shape — differential reported divergence or failed to run"
    sed 's/^/      /' "$out"
    n_fail=$((n_fail + 1))
    continue
  fi
  identical="$(grep -o 'diff_bit_identical: [01]' "$out" | awk '{print $2}')"
  src_equal="$(grep -o 'diff_sources_byte_equal: [01]' "$out" | awk '{print $2}')"
  compared="$(grep -o 'diff_bytes_compared: [0-9]*' "$out" | awk '{print $2}')"

  if [[ "$identical" != "1" ]]; then
    echo "  ✗ $shape — outputs are not bit-identical"
    sed 's/^/      /' "$out"
    n_fail=$((n_fail + 1))
    continue
  fi
  # Anti-cheat. If the generated source were byte-equal to the capture,
  # the transcription would have been quietly re-vendored and this
  # check would be measuring nothing.
  if [[ "$src_equal" != "0" ]]; then
    echo "  ✗ $shape — generated source is byte-equal to the capture; that is replay, not generation"
    n_fail=$((n_fail + 1))
    continue
  fi
  echo "  ✓ $shape — bit-identical over ${compared} bytes, sources differ"
  n_ok=$((n_ok + 1))
done

echo
echo "differential_codegen: ${n_ok} bit-identical, ${n_fail} failed"
if [[ "$n_fail" -ne 0 ]]; then
  exit 1
fi

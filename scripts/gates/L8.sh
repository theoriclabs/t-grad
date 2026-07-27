#!/usr/bin/env bash
# Gate L8 — algebraic MSL emit (kernel decl → MSL bytes).
#
# Per P6 (GOAL_NEXT.md §G8 + §G7 fall-back): L3's capture-lookup is
# augmented with a real algebraic renderer (Tgrad.Renderer.Metal.renderKernel)
# that walks a `KernelDecl` and produces MSL bytes. L8.a scope: simple
# kernels (copy_kernel today); the bf16 WMMA matmul kernels stay
# capture-lookup at L3 (their port to algebraic emit is L8.b — needs
# a much larger WMMA/simdgroup/LOOP-unroll renderer).
#
# Per §6 rule 1: scope-narrow ≠ correctness-narrow. The kernels covered
# at L8.a MUST byte-match a committed fixture AND compile-and-run.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : `Tgrad.Renderer.Metal.KernelDecl` + `renderKernel` exist;
#               `copyKernelDecl` exists; fixture file exists
#   - Layer C1: tgrad-cli render-metal-algebraic copy_kernel byte-matches
#               fixtures/codegen/copy_kernel.msl
#   - Layer C2: the emitted MSL compiles via Tgrad's Metal runtime
#               (ffi-compile-smoke returns fn_count: 1)
#   - Layer D : negative test — unknown kernel name rejected
#   - Layer E : evidence file
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L8_EMIT="$(tgrad_run_path L8_emit.msl)"
L8_COMPILE="$(tgrad_run_path L8_compile.txt)"
L8_NEG="$(tgrad_run_path L8_negative.txt)"

echo "[L8] algebraic MSL emit (kernel decl → bytes)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_symbols=(
  "structure KernelDecl"
  "structure BufferArg"
  "structure AttrArg"
  "inductive KernelArg"
  "inductive Stmt"
  "def renderKernel"
  "def copyKernelDecl"
)
for sym in "${required_symbols[@]}"; do
  if ! grep -qE "^${sym}\b" "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
    echo "  ✗ Tgrad/Renderer/Metal.lean missing '${sym}'"; exit 1
  fi
done
echo "  ✓ all ${#required_symbols[@]} required algebraic-emit decls in Renderer/Metal.lean"

required_fixtures=(
  fixtures/codegen/copy_kernel.msl
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} required fixtures present"

# ─── LAYER C1: byte-match emit vs fixture ─────────────────────────────
EMIT_OUT="$L8_EMIT"
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" render-metal-algebraic copy_kernel) \
    >"$EMIT_OUT" 2>&1 || {
  echo "  ✗ tgrad-cli render-metal-algebraic copy_kernel failed"
  cat "$EMIT_OUT"; exit 1
}
if ! diff -q "$EMIT_OUT" "$TGRAD_DIR/fixtures/codegen/copy_kernel.msl" >/dev/null; then
  echo "  ✗ algebraic emit does NOT byte-match captured fixture"
  diff "$EMIT_OUT" "$TGRAD_DIR/fixtures/codegen/copy_kernel.msl" | head -20
  exit 1
fi
echo "  ✓ Tgrad.Renderer.Metal.renderKernel(copyKernelDecl) byte-matches fixture"

# ─── LAYER C2: emitted MSL compiles via Tgrad runtime ─────────────────
COMPILE_OUT="$L8_COMPILE"
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" ffi-compile-smoke "$EMIT_OUT") \
    >"$COMPILE_OUT" 2>&1 || {
  echo "  ✗ ffi-compile-smoke rejected the algebraically-emitted MSL"
  cat "$COMPILE_OUT"; exit 1
}
grep -q "fn_count: 1" "$COMPILE_OUT" || {
  echo "  ✗ emitted MSL did not surface as a 1-kernel library"
  cat "$COMPILE_OUT"; exit 1
}
echo "  ✓ emitted MSL compiles via Tgrad.Runtime.Metal (fn_count: 1)"

# ─── LAYER D: negative test ───────────────────────────────────────────
set +e
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" render-metal-algebraic unknown_kernel) \
    >"$L8_NEG" 2>&1
neg_rc=$?
set -e
if [[ "$neg_rc" -eq 0 ]]; then
  echo "  ✗ render-metal-algebraic unknown_kernel returned 0 — should reject"
  cat "$L8_NEG"; exit 1
fi
grep -q "unknown kernel" "$L8_NEG" || {
  echo "  ✗ render-metal-algebraic unknown_kernel did not surface 'unknown kernel' error"
  cat "$L8_NEG"; exit 1
}
echo "  ✓ negative test correctly rejected (unknown kernel → nonzero exit)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
emit_hash="$(shasum -a 256 "$EMIT_OUT" | awk '{print $1}')"
fixture_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/codegen/copy_kernel.msl" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L8.json" <<EOF
{
  "gate": "L8",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L8.a — algebraic emit for simple Metal kernels (copy_kernel); WMMA matmul stays L3-captured",
  "hashes": {
    "emit_output_sha256":   "$emit_hash",
    "fixture_sha256":       "$fixture_hash"
  }
}
EOF
check_evidence_for L8 || exit 1
check_falsifiability_verified L8 || exit 1
echo "  ✓ L8 algebraic-emit gate green (evidence recorded)"

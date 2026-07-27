#!/usr/bin/env bash
# Gate L6 — Python authoring layer (real FFI: @[export] + ctypes + dylib).
#
# Per P4 (GOAL_NEXT.md §G_L6_b): L6.a's subprocess shell-out is replaced
# with a real bidirectional FFI bridge. libtgrad.dylib exports eight
# Python-callable symbols (tgrad_init, tgrad_handle_inc/dec,
# tgrad_tensor_alloc / tgrad_tensor_free / tgrad_tensor_write_bytes /
# tgrad_tensor_read_bytes / tgrad_matmul_64x64) that translate ctypes
# args into Lean's IO / ByteArray calling convention and dispatch the
# @[export] entries in Tgrad/PythonFFI.lean.
#
# Predicates per §G_L6_b:
#   - Layer A : universal preflight
#   - Layer B : Required modules / fixtures / built libtgrad.dylib /
#               required FFI symbols present
#   - Layer C : Python bench round-trips the captured tinygrad bytes
#               via ctypes (byte-match; same fixture as L5.b)
#   - Layer D1: anti-subprocess hard check — grep rejects subprocess.run
#   - Layer D2: negative test — unsupported shape raises NotInLeanScope
#   - Layer E : evidence file
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L6_DYLIB="$(tgrad_run_path L6_dylib.log)"
L6_BENCH="$(tgrad_run_path L6_bench.txt)"
L6_NEG="$(tgrad_run_path L6_negative.txt)"

echo "[L6] Python authoring layer (real FFI)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/PythonFFI.lean
  c/tgrad_python.c
  python/tgrad.py
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# The dylib must be buildable from the current source.
ensure_dylib "$L6_DYLIB" || exit 1
DYLIB="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
[[ -f "$DYLIB" ]] || { echo "  ✗ libtgrad.dylib not produced at $DYLIB"; exit 1; }
echo "  ✓ libtgrad.dylib built ($(stat -f%z "$DYLIB" 2>/dev/null || stat -c%s "$DYLIB") bytes)"

# Required Python-facing C symbols.
required_syms=(
  _tgrad_init
  _tgrad_tensor_alloc
  _tgrad_tensor_free
  _tgrad_tensor_write_bytes
  _tgrad_tensor_read_bytes
  _tgrad_matmul_64x64
  _tgrad_handle_inc
  _tgrad_handle_dec
)
for sym in "${required_syms[@]}"; do
  if ! nm -gU "$DYLIB" 2>/dev/null | awk '{print $3}' | grep -qx "$sym"; then
    echo "  ✗ libtgrad.dylib missing symbol: $sym"; exit 1
  fi
done
echo "  ✓ all ${#required_syms[@]} required FFI symbols present in libtgrad.dylib"

# Required @[export] declarations in PythonFFI.lean. The compiled
# symbols live in libtgrad_Tgrad.dylib (loaded as a dependency of
# libtgrad.dylib); requiring them at the source level is the right
# predicate — it's robust to dyld's two-tier symbol model.
required_lean_exports=(
  tgrad_tensor_alloc_lean
  tgrad_tensor_free_lean
  tgrad_tensor_write_bytes_lean
  tgrad_tensor_read_bytes_lean
  tgrad_matmul_64x64_lean
)
for sym in "${required_lean_exports[@]}"; do
  if ! grep -qE "^@\[export $sym\]" "$TGRAD_DIR/Tgrad/PythonFFI.lean"; then
    echo "  ✗ Tgrad/PythonFFI.lean missing @[export $sym]"; exit 1
  fi
done
echo "  ✓ all ${#required_lean_exports[@]} required @[export] declarations in PythonFFI.lean"
# Confirm the symbols are also resolvable from libtgrad.dylib (via its
# load-dep on libtgrad_Tgrad.dylib).
NMOUT="$(nm -m "$DYLIB" 2>/dev/null | grep -E ' _tgrad_(tensor|matmul)_.*_lean' || true)"
[[ -n "$NMOUT" ]] || { echo "  ✗ Lean @[export] symbols not resolvable from libtgrad.dylib"; exit 1; }
echo "  ✓ Lean @[export] symbols are visible to dyld via libtgrad.dylib"

# Pipeline fixtures (shared with L5.b's matmul-verify; the Python bench
# byte-matches against the SAME expected output).
required_fixtures=(
  fixtures/pipeline/matmul_64x64_bf16_seed42_a.bin
  fixtures/pipeline/matmul_64x64_bf16_seed42_b.bin
  fixtures/pipeline/matmul_64x64_bf16_seed42_expected.bin
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing pipeline fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} pipeline fixtures present"

# ─── LAYER C: behavioural — Python bench byte-matches ─────────────────
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench --shape 64x64x64 --dtype bf16) \
    >"$L6_BENCH" 2>&1 || {
  echo "  ✗ python3 python/tgrad.py bench failed"
  cat "$L6_BENCH"; exit 1
}
grep -q "py_byte_match: true"   "$L6_BENCH" || {
  echo "  ✗ Python bench did NOT byte-match the captured tinygrad output"
  cat "$L6_BENCH"; exit 1
}
grep -q "py_pipeline_ok: true"  "$L6_BENCH" || {
  echo "  ✗ Python bench missing py_pipeline_ok: true"
  cat "$L6_BENCH"; exit 1
}
echo "  ✓ Python bench byte-matches captured tinygrad output (8192 bytes via ctypes)"

# ─── LAYER D1: anti-subprocess hard check ─────────────────────────────
# Per §6 rule 2: subprocess.run / Popen / call / check_* must NOT appear
# in non-comment code paths under python/. (The 05_opaque_handle
# template explicitly uses ctypes, not subprocess.)
sub_hits="$(grep -nE '^[^#]*subprocess\.(run|Popen|call|check_)' "$TGRAD_DIR/python/tgrad.py" || true)"
if [[ -n "$sub_hits" ]]; then
  echo "  ✗ python/tgrad.py uses subprocess at runtime (forbidden per §6 rule 2):"
  echo "$sub_hits" | sed 's/^/      /'
  exit 1
fi
# Belt + braces: also forbid `import subprocess` at all.
if grep -qE '^[[:space:]]*(import[[:space:]]+subprocess|from[[:space:]]+subprocess)' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ python/tgrad.py imports subprocess (forbidden per §6 rule 2)"
  exit 1
fi
echo "  ✓ no subprocess usage in python/tgrad.py (anti-subprocess hard check)"

# ─── LAYER D2: negative test — out-of-scope shape rejected cleanly ────
set +e
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench --shape 7x9x11 --dtype bf16) \
    >"$L6_NEG" 2>&1
neg_rc=$?
set -e
if [[ "$neg_rc" -eq 0 ]]; then
  echo "  ✗ bench --shape 7x9x11 returned 0 — should reject as NotInLeanScope"
  cat "$L6_NEG"; exit 1
fi
grep -q "NotInLeanScope" "$L6_NEG" || {
  echo "  ✗ bench --shape 7x9x11 did not raise NotInLeanScope"
  cat "$L6_NEG"; exit 1
}
echo "  ✓ negative test correctly rejected (out-of-scope shape → NotInLeanScope)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
bench_hash="$(shasum -a 256 "$L6_BENCH" | awk '{print $1}')"
dylib_hash="$(shasum -a 256 "$DYLIB" | awk '{print $1}')"
expected_sha="$(shasum -a 256 "$TGRAD_DIR/fixtures/pipeline/matmul_64x64_bf16_seed42_expected.bin" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L6.json" <<EOF
{
  "gate": "L6",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L6.b — real FFI via @[export] + libtgrad.dylib + ctypes (no subprocess)",
  "hashes": {
    "python_bench_output_sha256":  "$bench_hash",
    "libtgrad_dylib_sha256":       "$dylib_hash",
    "expected_bytes_sha256":       "$expected_sha"
  }
}
EOF
check_evidence_for L6 || exit 1
check_falsifiability_verified L6 || exit 1
echo "  ✓ L6 Python-FFI gate green (evidence recorded)"

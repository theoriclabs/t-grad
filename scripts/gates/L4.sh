#!/usr/bin/env bash
# Gate L4 — FFI runtime (REQUIRES METAL HARDWARE).
#
# Verifies the C bridge to Metal (alloc / compile / dispatch / I/O)
# lifted from theograd_phases/{c, 13, 14}. Halts per GOAL.md §6.3
# when Metal is unavailable.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L4_AVAIL="$(tgrad_run_path L4_available.txt)"
L4_ALLOC="$(tgrad_run_path L4_alloc.txt)"
L4_COMPILE="$(tgrad_run_path L4_compile.txt)"
L4_DISPATCH="$(tgrad_run_path L4_dispatch.txt)"

echo "[L4] FFI runtime"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/Runtime/MetalDevice.lean
  Tgrad/Runtime/MetalAllocator.lean
  Tgrad/Runtime/MetalProgram.lean
  Tgrad/Runtime/Buffer.lean
  Tgrad/Runtime/Cache.lean
  c/metal_alloc.m
  c/metal_alloc_lean.c
  c/Makefile
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules / C bridge files present"

required_fixtures=(
  fixtures/codegen/matmul_64x64.msl
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing required fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} required fixtures present"

# ─── LAYER C: behavioural cross-validation ────────────────────────────

# Sub-predicate 4a: metalAvailable returns 1 on this host.
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" ffi-available) \
    >"$L4_AVAIL" 2>&1 || {
  echo "  ✗ tgrad-cli ffi-available failed"; cat "$L4_AVAIL"; exit 1
}
if ! grep -q "metal_available: 1" "$L4_AVAIL"; then
  echo "  ✗ Metal device probe returned 0 — halt per §6.3"
  cat "$L4_AVAIL"; exit 1
fi
echo "  ✓ Tgrad.Runtime.MetalDevice.metalAvailable returns 1"

# Sub-predicate 4b: ffi-alloc-cycle — alloc 1024 → free → realloc 1024;
# verify the second alloc returns the SAME pointer (LRU hit). Drops
# the LRU first so prior runs don't interfere.
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" ffi-alloc-cycle) \
    >"$L4_ALLOC" 2>&1 || {
  echo "  ✗ tgrad-cli ffi-alloc-cycle failed"; cat "$L4_ALLOC"; exit 1
}
if ! grep -q "lru_hit: true" "$L4_ALLOC"; then
  echo "  ✗ alloc-cycle didn't observe LRU hit — alloc/free/realloc not pointer-stable"
  cat "$L4_ALLOC"; exit 1
fi
echo "  ✓ Tgrad.Runtime.MetalAllocator alloc/free/realloc cycle hits LRU"

# Sub-predicate 4c: ffi-compile-smoke — compile the captured 64x64 MSL;
# library must report exactly 1 kernel function.
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" ffi-compile-smoke \
    "$TGRAD_DIR/fixtures/codegen/matmul_64x64.msl") \
    >"$L4_COMPILE" 2>&1 || {
  echo "  ✗ tgrad-cli ffi-compile-smoke failed"; cat "$L4_COMPILE"; exit 1
}
if ! grep -q "fn_count: 1" "$L4_COMPILE"; then
  echo "  ✗ compile-smoke expected fn_count: 1"; cat "$L4_COMPILE"; exit 1
fi
echo "  ✓ Tgrad.Runtime.MetalProgram compiles the 64x64 MSL (1 kernel)"

# Sub-predicate 4d: ffi-dispatch-copy — 16-element copy kernel
# round-trip. Verifies dispatch + buffer I/O bit-perfect.
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" ffi-dispatch-copy) \
    >"$L4_DISPATCH" 2>&1 || {
  echo "  ✗ tgrad-cli ffi-dispatch-copy failed"; cat "$L4_DISPATCH"; exit 1
}
if ! grep -q "bit_perfect: true" "$L4_DISPATCH"; then
  echo "  ✗ dispatch-copy round-trip not bit-perfect"; cat "$L4_DISPATCH"; exit 1
fi
echo "  ✓ Tgrad.Runtime.MetalProgram dispatches a 16-element copy kernel bit-perfect"

# ─── LAYER D: negative test ───────────────────────────────────────────
# Passing a NULL pointer to dispatch must surface as a non-zero exit
# or an exception, NOT a silent success.
if (cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" ffi-dispatch-null) >/dev/null 2>&1; then
  echo "  ✗ dispatch with NULL ptr returned 0 — should fail"
  exit 1
fi
echo "  ✓ negative test correctly rejected (NULL dispatch returns nonzero)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
avail_hash="$(shasum -a 256 "$L4_AVAIL" | awk '{print $1}')"
alloc_hash="$(shasum -a 256 "$L4_ALLOC" | awk '{print $1}')"
compile_hash="$(shasum -a 256 "$L4_COMPILE" | awk '{print $1}')"
disp_hash="$(shasum -a 256 "$L4_DISPATCH" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L4.json" <<EOF
{
  "gate": "L4",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "hashes": {
    "metal_available_sha256":  "$avail_hash",
    "alloc_cycle_sha256":  "$alloc_hash",
    "compile_smoke_sha256":  "$compile_hash",
    "dispatch_copy_sha256":  "$disp_hash"
  }
}
EOF
check_evidence_for L4 || exit 1
check_falsifiability_verified L4 || exit 1
echo "  ✓ L4 FFI-runtime gate green (evidence recorded)"

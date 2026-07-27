#!/usr/bin/env bash
# Gate L13.B — scalar matmul for below-TC-tile shapes (5 manifest entries:
# `(4, 4, 4)`, `(8, 8, 4)`, `(4, 32, 4)`, `(6, 32, 6)`, `(4, 4, 32)`).
#
# Per `GOAL_NEXT.md §8.RESUME` (sub-gate decomposition) and
# `GOAL_L13_B.md`. NO fall-back: 4/5 correct is L13.B RED. Inherits
# L13.A's `pickDispatchPlan` (extended with the below-TC-tile branch)
# and the L12 grammar (extended with `Stmt.declFloat`).
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : structural — `Stmt.declFloat` constructor +
#               `scalarMatmulKernelDecl` def + below-TC-tile branch in
#               `pickDispatchPlan` + dylib exports `_tgrad_matmul_small`
#               + Python wires `_lib.tgrad_matmul_small` + manifest
#               has the 5 below-TC-tile entries
#   - Layer C : behavioural — `bench-small` produces 5 JSONL rows,
#               all `correct: true`
#   - Layer D : anti-cheat
#       D1: `scalarMatmulKernelDecl` is pure (no IO)
#       D2: canonical numpy reference line present in bench harness
#           (`ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)`)
#       D3: L11 + L12 + L13_A still pass (refactor didn't break things)
#       D4: tgrad.py routes `(M, K, N) in _SMALL_TRIPLE_SET` through
#           `_lib.tgrad_matmul_small` (not the sentinel path)
#   - Layer E : evidence
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L13B_DYLIB="$(tgrad_run_path L13B_dylib.log)"
L13B_BENCH="$(tgrad_run_path L13B_bench.jsonl)"
L13B_LOG="$(tgrad_run_path L13B_bench.txt)"

echo "[L13_B] scalar matmul for below-TC-tile (5 shapes)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/Codegen/Opt/Heuristic.lean
  Tgrad/Renderer/Metal.lean
  Tgrad/Renderer/MatmulScalar.lean
  Tgrad/PythonFFI.lean
  python/tgrad.py
  python/tgrad_bench.py
  fixtures/bench/general_shape_manifest.json
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# Stmt.declFloat constructor in Metal.lean.
if ! grep -qE '\| declFloat' "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
  echo "  ✗ Renderer/Metal.lean missing Stmt.declFloat constructor"
  exit 1
fi
echo "  ✓ Stmt.declFloat constructor present"

# scalarMatmulKernelDecl in MatmulScalar.lean.
if ! grep -qE '^def scalarMatmulKernelDecl' \
       "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean"; then
  echo "  ✗ Renderer/MatmulScalar.lean missing scalarMatmulKernelDecl"
  exit 1
fi
echo "  ✓ scalarMatmulKernelDecl declared"

# Below-TC-tile branch in pickDispatchPlan.
if ! grep -qE 'M < 8 ∨ K < 8 ∨ N < 8' \
       "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean"; then
  echo "  ✗ Heuristic.lean missing the 'M < 8 ∨ K < 8 ∨ N < 8' branch"
  exit 1
fi
echo "  ✓ pickDispatchPlan has the below-TC-tile branch"

# Manifest has 5 below-TC-tile entries.
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
n_small="$("$PY" -c 'import json,sys
d = json.load(open(sys.argv[1]))
print(sum(1 for p in d if p.get("bucket")=="below_tc_tile"))' \
            "$TGRAD_DIR/fixtures/bench/general_shape_manifest.json")"
[[ "$n_small" -eq 5 ]] || {
  echo "  ✗ general_shape_manifest.json has $n_small below_tc_tile entries (need 5)"
  exit 1
}
echo "  ✓ manifest has exactly 5 below_tc_tile entries"

# Rebuild dylib (capturing the L13.B Lean changes).
ensure_dylib "$L13B_DYLIB" || exit 1
DYLIB="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
for sym in _tgrad_matmul _tgrad_matmul_alg _tgrad_matmul_small; do
  if ! nm -gU "$DYLIB" 2>/dev/null | awk '{print $3}' | grep -qx "$sym"; then
    echo "  ✗ libtgrad.dylib missing symbol: $sym"; exit 1
  fi
done
echo "  ✓ libtgrad.dylib exports _tgrad_matmul + _tgrad_matmul_alg + _tgrad_matmul_small"

# @[export tgrad_matmul_small_lean] in PythonFFI.lean.
if ! grep -qE '^@\[export tgrad_matmul_small_lean\]' \
       "$TGRAD_DIR/Tgrad/PythonFFI.lean"; then
  echo "  ✗ Tgrad/PythonFFI.lean missing @[export tgrad_matmul_small_lean]"
  exit 1
fi
echo "  ✓ @[export tgrad_matmul_small_lean] declaration present"

# ─── LAYER D1: scalarMatmulKernelDecl is pure (no IO in signature) ───
SIG=$(awk '/^def scalarMatmulKernelDecl/,/:= /' \
        "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean")
if echo "$SIG" | grep -qE '\bIO\b'; then
  echo "  ✗ scalarMatmulKernelDecl signature contains IO (must be pure)"
  exit 1
fi
echo "  ✓ scalarMatmulKernelDecl is pure (no IO in signature)"

# ─── LAYER D2: canonical numpy reference line in bench harness ───────
if ! grep -qF 'ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)' \
       "$TGRAD_DIR/python/tgrad_bench.py"; then
  echo "  ✗ tgrad_bench.py missing the canonical numpy reference line"
  exit 1
fi
echo "  ✓ tgrad_bench.py has canonical numpy reference (anti-self-comparison)"

# ─── LAYER D4: Python routes _SMALL_TRIPLE_SET through tgrad_matmul_small ──
if ! grep -qE '_SMALL_TRIPLE_SET' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py missing _SMALL_TRIPLE_SET (below-TC-tile routing)"
  exit 1
fi
if ! grep -qE '_lib\.tgrad_matmul_small' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py missing _lib.tgrad_matmul_small binding"
  exit 1
fi
echo "  ✓ tgrad.py wires _SMALL_TRIPLE_SET → _lib.tgrad_matmul_small"

# ─── LAYER C: bench-small sweep over the 5 below-TC-tile shapes ──────
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-small \
    --output "$L13B_BENCH") \
    >"$L13B_LOG" 2>&1 || {
  echo "  ✗ python bench-small failed"
  tail -20 "$L13B_LOG" | sed 's/^/      /'
  exit 1
}
n_rows="$(wc -l < "$L13B_BENCH" | awk '{print $1}')"
[[ "$n_rows" -eq 5 ]] || {
  echo "  ✗ bench-small produced $n_rows rows (need 5)"; exit 1
}
echo "  ✓ bench-small JSONL has exactly 5 rows"

STATS_JSON="$("$PY" - "$L13B_BENCH" <<'PYSTATS'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
n_correct = sum(1 for r in rows if r["correct"])
print(json.dumps({
    "n_correct": n_correct,
    "failed": [{"shape": r["shape"], "dist": r["dist"], "max_diff": r["max_abs_diff"]}
               for r in rows if not r["correct"]],
}))
PYSTATS
)"
N_CORRECT="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["n_correct"])')"
echo "  bench-small: correct=$N_CORRECT/5"
if [[ "$N_CORRECT" -ne 5 ]]; then
  echo "  ✗ L13.B RED: only $N_CORRECT/5 below-TC-tile shapes correct"
  echo "$STATS_JSON" | "$PY" -m json.tool 2>&1 | head -10 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ all 5 below-TC-tile shapes pass correctness"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
bench_hash="$(shasum -a 256 "$L13B_BENCH" | awk '{print $1}')"
manifest_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/bench/general_shape_manifest.json" | awk '{print $1}')"
scalar_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L13_B.json" <<EOF
{
  "gate": "L13_B",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L13.B — scalar matmul for below-TC-tile shapes; correctness only",
  "below_tc_tile_total":   5,
  "below_tc_tile_correct": $N_CORRECT,
  "hashes": {
    "bench_jsonl_sha256":     "$bench_hash",
    "manifest_sha256":        "$manifest_hash",
    "matmul_scalar_sha256":   "$scalar_hash"
  }
}
EOF
check_evidence_for L13_B || exit 1
check_falsifiability_verified L13_B || exit 1
echo "  ✓ L13.B scalar-matmul gate green (5/5 below-TC-tile correct)"

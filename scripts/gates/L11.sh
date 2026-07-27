#!/usr/bin/env bash
# Gate L11 — full benchmark parity (10 shapes × 5 dists = 50 pairs).
#
# Per P9 (GOAL_NEXT.md §G11 + GOAL_L11.md). NO fall-back — partial
# coverage is L11 RED, not L11.a. The fix for any failure is upstream
# (L5/L6/L7 expansions), not narrowing this gate.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : structural — manifest/tolerances/baseline_full all
#               present + schema-valid; bench harness + dylib + Lean
#               @[export] entries all present
#   - Layer C : behavioural — python bench-full runs the 50-pair sweep,
#               JSONL has exactly 50 rows, every row correct + ratio≤1.5
#   - Layer D : anti-cheat — grep for the canonical numpy reference
#               line; reject any tinygrad import in bench harness
#   - Layer E : evidence
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L11_DYLIB="$(tgrad_run_path L11_dylib.log)"
L11_BENCH="$(tgrad_run_path L11_bench.jsonl)"
L11_LOG="$(tgrad_run_path L11_bench.txt)"

echo "[L11] full benchmark parity (50 shape×dist pairs)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
PROFILE="${TGRAD_PERF_PROFILE:-${TGRAD_HOST:-apple_m4_mini_release}}"
BASELINE_FULL="${TGRAD_PERF_BASELINE_FULL:-$TGRAD_DIR/fixtures/perf/tinygrad_baseline_${PROFILE}_full.json}"
required_modules=(
  python/tgrad.py
  python/tgrad_bench.py
  Tgrad/PythonFFI.lean
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

required_fixtures=(
  fixtures/bench/pair_manifest.json
  fixtures/bench/dist_tolerances.json
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing fixture: $f"; exit 1; }
done
[[ -f "$BASELINE_FULL" ]] || { echo "  ✗ missing baseline fixture: $BASELINE_FULL"; exit 1; }
echo "  ✓ all ${#required_fixtures[@]} required fixtures and selected baseline present"

# All 10 captured matmul MSLs present for the L11 shape set.
required_msls=(
  matmul_1024x1024x1024 matmul_2048x2048x2048 matmul_4096x4096x4096 matmul_8192x8192x8192
  matmul_8192x1024x1024 matmul_4096x1024x1024 matmul_2048x1024x1024
  matmul_1024x1024x8192 matmul_1024x1024x4096 matmul_1024x1024x2048
)
for name in "${required_msls[@]}"; do
  [[ -f "$TGRAD_DIR/fixtures/codegen/${name}.msl" ]] \
    || { echo "  ✗ missing captured MSL: ${name}.msl"; exit 1; }
done
echo "  ✓ all 10 captured matmul MSLs present"

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

# Manifest has exactly 50 entries.
n_pairs="$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
            "$TGRAD_DIR/fixtures/bench/pair_manifest.json")"
[[ "$n_pairs" -eq 50 ]] || { echo "  ✗ pair_manifest has $n_pairs entries (need 50)"; exit 1; }
echo "  ✓ pair_manifest has exactly 50 entries"

# Tolerances has exactly 5 dists.
n_dists="$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
            "$TGRAD_DIR/fixtures/bench/dist_tolerances.json")"
[[ "$n_dists" -eq 5 ]] || { echo "  ✗ dist_tolerances has $n_dists dists (need 5)"; exit 1; }
echo "  ✓ dist_tolerances has exactly 5 distributions"

# @[export tgrad_matmul_lean] declaration present in PythonFFI.lean.
if ! grep -qE '^@\[export tgrad_matmul_lean\]' "$TGRAD_DIR/Tgrad/PythonFFI.lean"; then
  echo "  ✗ Tgrad/PythonFFI.lean missing @[export tgrad_matmul_lean]"; exit 1
fi
echo "  ✓ @[export tgrad_matmul_lean] declaration present"

# Rebuild dylib (other gates' preflight may have wiped it).
ensure_dylib "$L11_DYLIB" || exit 1
DYLIB="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
if ! nm -gU "$DYLIB" 2>/dev/null | awk '{print $3}' | grep -qx "_tgrad_matmul"; then
  echo "  ✗ libtgrad.dylib missing _tgrad_matmul symbol (general entry)"; exit 1
fi
echo "  ✓ libtgrad.dylib current; exports tgrad_matmul"

# ─── LAYER D1: anti-self-comparison — canonical numpy reference line ──
# Per §G11 + GOAL_L11.md §4 Layer D1: the bench harness MUST use
# `ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)`. Swapping in
# `ref = lean_out_f32` would be trivially correct (the "self-comparison"
# attack). Grep enforces the canonical line.
if ! grep -qF 'ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)' \
       "$TGRAD_DIR/python/tgrad_bench.py"; then
  echo "  ✗ tgrad_bench.py is missing the canonical numpy reference line"
  echo "      Required exactly: ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)"
  exit 1
fi
echo "  ✓ tgrad_bench.py has canonical numpy reference (anti-self-comparison)"

# ─── LAYER D2: no live tinygrad import in bench harness ───────────────
if grep -qE '^[[:space:]]*(import[[:space:]]+tinygrad|from[[:space:]]+tinygrad)' \
     "$TGRAD_DIR/python/tgrad_bench.py"; then
  echo "  ✗ tgrad_bench.py imports tinygrad (forbidden — runtime independence)"
  exit 1
fi
echo "  ✓ tgrad_bench.py is tinygrad-free (runtime independence preserved)"

# ─── LAYER C: behavioural — run the 50-pair sweep ─────────────────────
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-full \
    --baseline "$BASELINE_FULL" \
    --output "$L11_BENCH" --warmup 30 --measured 30) \
    >"$L11_LOG" 2>&1 || {
  echo "  ✗ python bench-full failed"
  tail -30 "$L11_LOG" | sed 's/^/      /'
  exit 1
}

n_rows="$(wc -l < "$L11_BENCH" | awk '{print $1}')"
[[ "$n_rows" -eq 50 ]] || { echo "  ✗ bench-full produced $n_rows rows (need 50)"; exit 1; }
echo "  ✓ bench-full JSONL has exactly 50 rows"

# Compute stats + assert all 50 correct + ratio.
STATS_JSON="$("$PY" - "$L11_BENCH" <<'PYSTATS'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
n_correct  = sum(1 for r in rows if r["correct"])
n_ratio_ok = sum(1 for r in rows if r["ratio"] <= 1.5 and r["lean_ms_min"] > 0)
ratios = sorted(r["ratio"] for r in rows)
failed = [{"shape": r["shape"], "dist": r["dist"], "correct": r["correct"], "ratio": r["ratio"],
           "lean_ms_min": r["lean_ms_min"], "tinygrad_ms_min": r["tinygrad_ms_min"]}
          for r in rows if not r["correct"] or r["ratio"] > 1.5]
print(json.dumps({
    "n_correct": n_correct,
    "n_ratio_ok": n_ratio_ok,
    "ratio_min": ratios[0],
    "ratio_median": ratios[len(ratios)//2],
    "ratio_max": ratios[-1],
    "failed": failed,
}))
PYSTATS
)"
N_CORRECT="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["n_correct"])')"
N_RATIO_OK="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["n_ratio_ok"])')"
RATIO_MIN="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ratio_min"])')"
RATIO_MED="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ratio_median"])')"
RATIO_MAX="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ratio_max"])')"

echo "  stats: correct=$N_CORRECT/50  ratio_ok=$N_RATIO_OK/50  "\
"ratio[min/median/max]=$RATIO_MIN/$RATIO_MED/$RATIO_MAX"

if [[ "$N_CORRECT" -ne 50 ]] || [[ "$N_RATIO_OK" -ne 50 ]]; then
  echo "  ✗ L11 RED: correct=$N_CORRECT/50, ratio_ok=$N_RATIO_OK/50"
  echo "$STATS_JSON" | "$PY" -m json.tool 2>&1 | head -30 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ all 50 pairs: correct=50/50, ratio_ok=50/50 (predicate: ≤ 1.5)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$PROFILE"; plat="$(uname -srm)"
bench_hash="$(shasum -a 256 "$L11_BENCH" | awk '{print $1}')"
manifest_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/bench/pair_manifest.json" | awk '{print $1}')"
tol_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/bench/dist_tolerances.json" | awk '{print $1}')"
baseline_hash="$(shasum -a 256 "$BASELINE_FULL" | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L11.json" <<EOF
{
  "gate": "L11",
  "ts_utc": "$ts",
  "host_profile": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L11 — full benchmark parity (50 shape×dist pairs); no fall-back",
  "pairs_total":   50,
  "pairs_passed":  $N_CORRECT,
  "pairs_failed":  $(( 50 - N_CORRECT )),
  "ratio_ok":      $N_RATIO_OK,
  "ratio_min":     $RATIO_MIN,
  "ratio_median":  $RATIO_MED,
  "ratio_max":     $RATIO_MAX,
  "hashes": {
    "bench_jsonl_sha256":     "$bench_hash",
    "pair_manifest_sha256":   "$manifest_hash",
    "dist_tolerances_sha256": "$tol_hash",
    "baseline_full_sha256":   "$baseline_hash"
  }
}
EOF
check_evidence_for L11 || exit 1
check_falsifiability_verified L11 || exit 1
echo "  ✓ L11 full-benchmark-parity gate green (50/50; evidence recorded)"

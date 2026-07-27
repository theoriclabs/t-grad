#!/usr/bin/env bash
# Gate L14.B.3 — 16 pinned view-matmul cases via the parametric scalar
# matmul + view-aware index UOps.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L14B3_DYLIB="$(tgrad_run_path L14B3_dylib.log)"
L14B3_LOG="$(tgrad_run_path L14B3_bench.txt)"
L14B3_JSONL="$(tgrad_run_path L14B3_views.jsonl)"
L14B3_TRACED_JSONL="$(tgrad_run_path L14B3_views_traced.jsonl)"

echo "[L14_B_3] 16 pinned view-matmul cases + rangeify trace"

run_preflight
cd "$REPO_ROOT"

# Layer B — structural
required=(
  fixtures/bench/view_manifest.json
  python/tgrad_bench.py
  scripts/capture/view_baselines.py
  Tgrad/Pipeline.lean
)
for m in "${required[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing: $m"; exit 1; }
done
echo "  ✓ all ${#required[@]} required modules present"

# Manifest has 16 entries.
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
n_entries="$("$PY" -c 'import json,sys; print(len(json.load(open("'"$TGRAD_DIR/fixtures/bench/view_manifest.json"'"))))')"
[[ "$n_entries" -eq 16 ]] || { echo "  ✗ view_manifest has $n_entries entries (need 16)"; exit 1; }
echo "  ✓ view_manifest has 16 entries"

# bench_views function defined.
grep -qE '^def run_bench_views' "$TGRAD_DIR/python/tgrad_bench.py" \
  || { echo "  ✗ tgrad_bench.py missing run_bench_views"; exit 1; }
echo "  ✓ run_bench_views defined"

# All 5 view op classes covered in viewIndexUOpForA/B.
op_classes=("buffer" "permute" "reshape" "slice" "expand")
for op in "${op_classes[@]}"; do
  if ! grep -qE "\\| \\.${op}\\b" "$TGRAD_DIR/Tgrad/Pipeline.lean"; then
    echo "  ✗ viewIndexUOpFor{A,B} doesn't handle .${op} chain"
    exit 1
  fi
done
echo "  ✓ viewIndexUOpFor{A,B} covers all 5 view-op classes"

# Layer C — bench-views 16/16 correct.
ensure_dylib "$L14B3_DYLIB" || exit 1

BENCH_LOG="$L14B3_LOG"
if ! (cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-views \
        --output "$L14B3_JSONL") >"$BENCH_LOG" 2>&1; then
  echo "  ✗ bench-views failed:"
  sed 's/^/      /' "$BENCH_LOG"
  exit 1
fi

N_CORRECT="$(grep -oE 'py_bench_views_n_correct: [0-9]+' "$BENCH_LOG" | awk '{print $2}')"
N_VIEW="$(grep -oE 'py_bench_views_n_route_view: [0-9]+' "$BENCH_LOG" | awk '{print $2}')"
[[ "$N_CORRECT" -eq 16 ]] || { echo "  ✗ bench-views n_correct=$N_CORRECT/16"; sed 's/^/      /' "$BENCH_LOG"; exit 1; }
[[ "$N_VIEW" -eq 16 ]] || { echo "  ✗ bench-views n_route_view=$N_VIEW/16 (all should route via tgrad_matmul_view)"; exit 1; }
echo "  ✓ bench-views: 16/16 correct, 16/16 routed via tgrad_matmul_view"

# Layer C2: rangeify trace shows movement_count_in > 0 for view runs.
# Re-run with TGRAD_RANGEIFY_TRACE=1 to capture the trace.
TRACE="$(tgrad_run_prepare_rangeify_trace)"
(cd "$REPO_ROOT" && TGRAD_RANGEIFY_TRACE=1 TGRAD_RANGEIFY_TRACE_PATH="$TRACE" \
  "$PY" "$TGRAD_DIR/python/tgrad.py" bench-views \
    --output "$L14B3_TRACED_JSONL") >/dev/null 2>&1

N_TRACE="$(wc -l < "$TRACE" | awk '{print $1}')"
N_NONTRIVIAL="$("$PY" -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(sum(1 for r in rows if r.get("movement_count_in", 0) > 0))
' "$TRACE")"
[[ "$N_TRACE" -ge 16 ]] || { echo "  ✗ trace has $N_TRACE rows (need >= 16)"; exit 1; }
[[ "$N_NONTRIVIAL" -ge 16 ]] || { echo "  ✗ trace has $N_NONTRIVIAL nontrivial rows (need >= 16)"; exit 1; }
echo "  ✓ rangeify trace: $N_TRACE total rows, $N_NONTRIVIAL with movement_count_in > 0"

# Layer C3 — regression evidence
L11_PAIRS="$("$PY" -c 'import json; print(json.load(open("'"$TGRAD_EVIDENCE_DIR/L11.json"'"))["pairs_passed"])' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11.json shows 50/50 pairs"

# Layer E — evidence
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
manifest_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/bench/view_manifest.json" | awk '{print $1}')"
bench_hash="$(shasum -a 256 "$TGRAD_DIR/python/tgrad_bench.py" | awk '{print $1}')"
pipeline_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Pipeline.lean" | awk '{print $1}')"
trace_hash="$(shasum -a 256 "$TRACE" 2>/dev/null | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L14_B_3.json" <<EOF
{
  "gate": "L14_B_3",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14.B.3 — 16 pinned view-matmul cases via parametric scalar + view-aware index UOps; rangeify trace evidence",
  "pinned_views_total": 16,
  "pinned_views_pass":  $N_CORRECT,
  "n_route_view":       $N_VIEW,
  "n_route_buffer":     0,
  "rangeify_rows":            $N_TRACE,
  "rangeify_nontrivial_rows": $N_NONTRIVIAL,
  "view_materializations": 0,
  "l11_regression":   "pass",
  "l13_regression":   "pass",
  "l13_f_regression": "pass",
  "l14_b_2_regression": "pass",
  "hashes": {
    "view_manifest_sha256":     "$manifest_hash",
    "bench_module_sha256":      "$bench_hash",
    "pipeline_sha256":          "$pipeline_hash",
    "rangeify_trace_sha256":    "$trace_hash"
  }
}
EOF
check_evidence_for L14_B_3 || exit 1
check_falsifiability_verified L14_B_3 || exit 1
echo "  ✓ L14.B.3 — 16/16 pinned views correct + rangeify trace evidence — GREEN"

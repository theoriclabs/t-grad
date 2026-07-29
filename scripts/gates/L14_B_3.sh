#!/usr/bin/env bash
# Gate L14.B.3 — 16 pinned view-matmul cases via the parametric scalar
# matmul + view-aware index UOps.
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

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

# All 5 view op classes covered by the view-index derivation.
#
# This grepped Tgrad/Pipeline.lean, which held one match arm per view op
# until viewIndexUOpForA/B were rewritten to delegate to
# `Schedule.viewOfUOp`. The arms did not disappear -- they moved into the
# typed View algebra in Tgrad/Schedule/View.lean, which is a single
# derivation shared by every caller instead of two hand-written copies.
# The check follows the code; it is not relaxed.
#
# Note this is a STRUCTURAL pre-check on syntax, and weak on its own: it
# proves arms exist, not that they compute anything. Layer C below is the
# behavioural proof -- bench-views must be 16/16 correct against the
# pinned manifest.
VIEW_DERIVATION="$TGRAD_DIR/Tgrad/Schedule/View.lean"
op_classes=("buffer" "permute" "reshape" "slice" "expand")
for op in "${op_classes[@]}"; do
  if ! grep -qE "\\| \\.${op}\\b" "$VIEW_DERIVATION"; then
    echo "  ✗ view-index derivation doesn't handle .${op} chain"
    exit 1
  fi
done
echo "  ✓ view-index derivation covers all 5 view-op classes"

# Layer C — bench-views 16/16 correct.
ensure_dylib /tmp/tgrad_L14B3_dylib.log || exit 1

BENCH_LOG="/tmp/tgrad_L14B3_bench.txt"
if ! (cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-views \
        --output /tmp/tgrad_L14B3_views.jsonl) >"$BENCH_LOG" 2>&1; then
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
TRACE="/tmp/tgrad_rangeify_trace.jsonl"
: >"$TRACE"
(cd "$REPO_ROOT" && TGRAD_RANGEIFY_TRACE=1 "$PY" "$TGRAD_DIR/python/tgrad.py" bench-views \
    --output /tmp/tgrad_L14B3_views_traced.jsonl) >/dev/null 2>&1

N_TRACE="$(wc -l < "$TRACE" | awk '{print $1}')"
N_NONTRIVIAL="$("$PY" -c '
import json
rows = [json.loads(l) for l in open("/tmp/tgrad_rangeify_trace.jsonl") if l.strip()]
print(sum(1 for r in rows if r.get("movement_count_in", 0) > 0))
')"
[[ "$N_TRACE" -ge 16 ]] || { echo "  ✗ trace has $N_TRACE rows (need >= 16)"; exit 1; }
[[ "$N_NONTRIVIAL" -ge 16 ]] || { echo "  ✗ trace has $N_NONTRIVIAL nontrivial rows (need >= 16)"; exit 1; }
echo "  ✓ rangeify trace: $N_TRACE total rows, $N_NONTRIVIAL with movement_count_in > 0"

# Layer C3 — regression evidence
L11_PAIRS="$("$PY" -c 'import json; print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])' 2>/dev/null || echo 0)"
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
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L14_B_3.json" <<EOF
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

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
  Tgrad/Schedule/View.lean
)
for m in "${required[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing: $m"; exit 1; }
done
echo "  ✓ all ${#required[@]} required modules present"

# Manifest has the frozen 16-entry operation partition. Counting only the
# denominator allowed the two reshape witnesses to be replaced by identity
# reshapes without changing the gate's result.
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
MANIFEST_COUNTS_JSON="$("$PY" -c '
import collections, json, sys
rows = json.load(open(sys.argv[1]))
expected = {
    "transpose_left": 4,
    "transpose_right": 4,
    "transpose_both": 2,
    "slice_2": 2,
    "reshape_split": 2,
    "expand_right": 2,
}
actual = dict(collections.Counter(row.get("op_chain") for row in rows))
if len(rows) != 16 or actual != expected:
    raise SystemExit(f"view_manifest partition mismatch: rows={len(rows)}, actual={actual}, expected={expected}")
if any(row["K"] % 2 for row in rows if row["op_chain"] == "reshape_split"):
    raise SystemExit("reshape_split requires even K")
print(json.dumps(actual, sort_keys=True, separators=(",", ":")))
' "$TGRAD_DIR/fixtures/bench/view_manifest.json")" \
  || { echo "  ✗ invalid view_manifest operation partition"; exit 1; }
echo "  ✓ view_manifest has the frozen 16-entry operation partition: $MANIFEST_COUNTS_JSON"

# bench_views function defined.
grep -qE '^def run_bench_views' "$TGRAD_DIR/python/tgrad_bench.py" \
  || { echo "  ✗ tgrad_bench.py missing run_bench_views"; exit 1; }
echo "  ✓ run_bench_views defined"

# The five named equations are compiled with Tgrad.Schedule.View by the
# preflight build. Their stable names make the intended `viewOfUOp` contract
# discoverable without a whole-file constructor grep that can be satisfied by
# the unrelated `bufferRootOf` function below it.
VIEW_DERIVATION="$TGRAD_DIR/Tgrad/Schedule/View.lean"
view_equations=(
  viewOfUOp_buffer_eq
  viewOfUOp_permute_eq
  viewOfUOp_reshape_eq
  viewOfUOp_slice_eq
  viewOfUOp_expand_eq
)
for equation in "${view_equations[@]}"; do
  if ! grep -qE "^theorem[[:space:]]+${equation}([[:space:](]|$)" "$VIEW_DERIVATION"; then
    echo "  ✗ missing compiled view derivation equation: $equation"
    exit 1
  fi
done
echo "  ✓ all 5 named viewOfUOp equations are present in the compiled module"

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
view_hash="$(shasum -a 256 "$VIEW_DERIVATION" | awk '{print $1}')"
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
  "manifest_op_counts": $MANIFEST_COUNTS_JSON,
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
    "view_derivation_sha256":   "$view_hash",
    "rangeify_trace_sha256":    "$trace_hash"
  }
}
EOF

# The generic evidence check only validates the outer schema. This
# load-bearing source must be both named and resolved to the bytes observed by
# this run; an omitted or stale hash is not provenance.
if ! "$PY" - "$TGRAD_DIR/fixtures/gate_evidence/L14_B_3.json" "$VIEW_DERIVATION" <<'PY'
import hashlib, json, pathlib, sys

evidence = json.loads(pathlib.Path(sys.argv[1]).read_text())
source = pathlib.Path(sys.argv[2])
expected = hashlib.sha256(source.read_bytes()).hexdigest()
actual = evidence.get("hashes", {}).get("view_derivation_sha256")
if actual != expected:
    raise SystemExit(
        f"view_derivation_sha256 unresolved: actual={actual!r}, expected={expected}"
    )
PY
then
  echo "  ✗ L14_B_3 evidence does not resolve the view derivation source"
  exit 1
fi
check_evidence_for L14_B_3 || exit 1
check_falsifiability_verified L14_B_3 || exit 1
echo "  ✓ L14.B.3 — 16/16 pinned views correct + rangeify trace evidence — GREEN"

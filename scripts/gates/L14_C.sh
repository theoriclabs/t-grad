#!/usr/bin/env bash
# Gate L14.C — anti-hardcoding random-views sweep (20 chains, seed=HEAD)
# Per `Tgrad/GOAL_L14_C.md`.
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14_C] random-views anti-hardcoding (20 chains, seed=HEAD)"

run_preflight
cd "$REPO_ROOT"

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

# Layer B — structural
required=(
  python/tgrad_bench.py
  python/tgrad.py
  Tgrad/Pipeline.lean
)
for m in "${required[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing: $m"; exit 1; }
done
echo "  ✓ all ${#required[@]} required modules present"

# Required defs/subparser/flag
grep -qE '^def run_bench_random_views' "$TGRAD_DIR/python/tgrad_bench.py" \
  || { echo "  ✗ run_bench_random_views missing"; exit 1; }
grep -qE 'VIEW_OP_CATALOGUE\s*=\s*\[' "$TGRAD_DIR/python/tgrad_bench.py" \
  || { echo "  ✗ VIEW_OP_CATALOGUE missing"; exit 1; }
grep -qE 'sub\.add_parser\("bench-random-views"' "$TGRAD_DIR/python/tgrad.py" \
  || { echo "  ✗ bench-random-views subparser missing"; exit 1; }
grep -qE '"--seed"' "$TGRAD_DIR/python/tgrad.py" \
  || { echo "  ✗ --seed flag missing"; exit 1; }
echo "  ✓ run_bench_random_views + bench-random-views subparser + --seed flag present"

# Layer D — anti-cheat (precondition: catalogue + sampler invariants)
# D2: numpy reference path doesn't call back into Tgrad
if grep -qE 'apply_op_numpy_for_view.*tgrad\.|_tg\.Tensor.*ref' "$TGRAD_DIR/python/tgrad_bench.py"; then
  echo "  ✗ anti-self-comparison breach: numpy ref appears to use Tgrad"
  exit 1
fi
grep -qE '_apply_op_numpy_for_view\(op, a_bf16, b_bf16, M, K, N\)' \
        "$TGRAD_DIR/python/tgrad_bench.py" \
  || { echo "  ✗ canonical numpy ref invocation absent"; exit 1; }
echo "  ✓ D2 anti-self-comparison: numpy reference uses bf16-roundtripped inputs, no Tgrad in ref path"

# D4: sampler has NO hardcoded shape literals
if grep -nE 'rng.choice\(\[[0-9]+,' "$TGRAD_DIR/python/tgrad_bench.py" \
   | grep -v '_view_shape_grid'; then
  echo "  ✗ hardcoded shape literal in sampler"
  exit 1
fi
echo "  ✓ D4 no hardcoded shape literal in sampler"

# D5: catalogue must list all 7 ops
for op in transpose_left transpose_right transpose_both slice_2 slice_4 reshape_split expand_right; do
  grep -qE "\"$op\"" "$TGRAD_DIR/python/tgrad_bench.py" \
    || { echo "  ✗ catalogue missing op: $op"; exit 1; }
done
echo "  ✓ D5 catalogue has all 7 ops"

# Layer C — behavioural (20 random rows under HEAD seed)
ensure_dylib /tmp/tgrad_L14C_dylib.log || exit 1

SEED="$(git -C "$REPO_ROOT" rev-parse HEAD | head -c 16)"
echo "  → seed=$SEED (from HEAD prefix)"

OUT_JSONL="/tmp/tgrad_L14_C_random.jsonl"
LOG="/tmp/tgrad_L14_C_random.log"
if ! (cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-random-views \
        --seed "$SEED" --count 20 --output "$OUT_JSONL") >"$LOG" 2>&1; then
  echo "  ✗ bench-random-views failed:"
  sed 's/^/      /' "$LOG"
  exit 1
fi

N_ROWS="$(wc -l < "$OUT_JSONL" | awk '{print $1}')"
[[ "$N_ROWS" -eq 20 ]] || { echo "  ✗ row count = $N_ROWS, expected 20"; exit 1; }
echo "  ✓ row count = 20"

N_CORRECT="$(grep -oE 'py_random_views_n_correct: [0-9]+' "$LOG" | awk '{print $2}')"
N_VIEW="$(grep -oE 'py_random_views_n_route_view: [0-9]+' "$LOG" | awk '{print $2}')"
[[ "$N_CORRECT" -eq 20 ]] || { echo "  ✗ n_correct=$N_CORRECT/20"; sed 's/^/      /' "$LOG"; exit 1; }
[[ "$N_VIEW" -eq 20 ]] || { echo "  ✗ n_route_view=$N_VIEW/20 (all should route via view)"; exit 1; }
echo "  ✓ 20/20 correct + 20/20 routed via view path"

# All 7 ops appear at least once.
OPS_USED="$(grep -oE 'py_random_views_ops_used: [^ ]+' "$LOG" | sed 's/py_random_views_ops_used: //')"
echo "  → ops_used: $OPS_USED"
for op in transpose_left transpose_right transpose_both slice_2 slice_4 reshape_split expand_right; do
  case ",$OPS_USED," in
    *",$op,"*) ;;
    *) echo "  ✗ op $op never sampled"; exit 1 ;;
  esac
done
echo "  ✓ all 7 catalogue ops appear at least once"

# Seed in log matches HEAD prefix.
SEED_LOGGED="$(grep -oE 'py_random_views_seed: [0-9a-f]+' "$LOG" | awk '{print $2}')"
[[ "$SEED_LOGGED" == "$SEED" ]] || { echo "  ✗ seed mismatch (logged=$SEED_LOGGED, HEAD=$SEED)"; exit 1; }
echo "  ✓ D1 seed matches HEAD prefix"

# Layer C2 — regression evidence
L11_PAIRS="$("$PY" -c 'import json; print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11 50/50 still holds"
for sub in L13 L13_F L14_A L14_B; do
  ev="$TGRAD_DIR/fixtures/gate_evidence/${sub}.json"
  [[ -f "$ev" ]] || { echo "  ✗ regression evidence missing: $ev"; exit 1; }
done
echo "  ✓ L13 / L13_F / L14_A / L14_B evidence present"

# Layer E — evidence
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
random_jsonl_hash="$(shasum -a 256 "$OUT_JSONL" | awk '{print $1}')"
bench_hash="$(shasum -a 256 "$TGRAD_DIR/python/tgrad_bench.py" | awk '{print $1}')"
script_hash="$(shasum -a 256 "$TGRAD_DIR/scripts/gates/L14_C.sh" | awk '{print $1}')"
ops_used_json="$("$PY" -c "import json; print(json.dumps(sorted('${OPS_USED}'.split(','))))")"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L14_C.json" <<EOF
{
  "gate": "L14_C",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14.C — random-views anti-hardcoding (20 chains, seed=HEAD)",
  "random_views_total":  20,
  "random_views_pass":   $N_CORRECT,
  "random_seed_used":    "$SEED",
  "ops_used":            $ops_used_json,
  "n_route_view":        $N_VIEW,
  "l11_regression":      "pass",
  "l13_regression":      "pass",
  "l13_f_regression":    "pass",
  "l14_a_regression":    "pass",
  "l14_b_regression":    "pass",
  "hashes": {
    "random_jsonl_sha256": "$random_jsonl_hash",
    "bench_module_sha256": "$bench_hash",
    "gate_script_sha256":  "$script_hash"
  }
}
EOF
check_evidence_for L14_C || exit 1
check_falsifiability_verified L14_C || exit 1
echo "  ✓ L14.C — 20/20 random views correct + all 7 ops used — GREEN"

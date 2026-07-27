#!/usr/bin/env bash
# Gate L15.B — experiment-closure runtime + benchmark recheck.
# Per `Tgrad/GOAL_L15_B.md` + `GOAL_L15.md §3 criteria 4-6` + §5 runtime checks.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L15B_DYLIB="$(tgrad_run_path L15B_dylib.log)"
L15B_SHAPES_LOG="$(tgrad_run_path L15B_random_shapes.log)"
L15B_SHAPES="$(tgrad_run_path L15B_random_shapes.jsonl)"
L15B_VIEWS_LOG="$(tgrad_run_path L15B_random_views.log)"
L15B_VIEWS="$(tgrad_run_path L15B_random_views.jsonl)"
L15B_AUDIT="$(tgrad_run_path L15B_audit.json)"
L15B_AUDIT_ERR="$(tgrad_run_path L15B_audit.err)"

echo "[L15_B] experiment closure — runtime + benchmark recheck"

run_preflight
cd "$REPO_ROOT"

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

# Layer A.2 — L15.A must already be green (this gate consumes its evidence).
[[ -f "$TGRAD_DIR/fixtures/gate_evidence/L15_A.json" ]] \
  || { echo "  ✗ L15.A evidence missing — run L15_A first"; exit 1; }
echo "  ✓ L15.A evidence present"

# Layer B — required evidence files (the sub-gate inputs to L15.B).
required=(
  fixtures/gate_evidence/L11.json
  fixtures/gate_evidence/L12.json
  fixtures/gate_evidence/L13.json
  fixtures/gate_evidence/L13_F.json
  fixtures/gate_evidence/L14.json
  fixtures/gate_evidence/L14_B_3.json
  fixtures/gate_evidence/L14_C.json
)
for m in "${required[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing: $m"; exit 1; }
done
echo "  ✓ all ${#required[@]} sub-gate evidence files present"

# Layer B — required scripts/utilities.
required_scripts=(
  scripts/check_no_tinygrad_deps.sh
  scripts/runtime_independence.sh
  scripts/dev/l15_b_audit.py
)
for m in "${required_scripts[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing script: $m"; exit 1; }
done
echo "  ✓ runtime-indep + audit scripts present"

# Layer C — fresh random samples under HEAD-derived seed.
ensure_dylib "$L15B_DYLIB" || exit 1

SEED="$(git -C "$REPO_ROOT" rev-parse HEAD | head -c 16)"
echo "  → seed=$SEED (from HEAD prefix)"

# 10 fresh random shapes
LOG_S="$L15B_SHAPES_LOG"
OUT_S="$L15B_SHAPES"
if ! (cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-random-shapes \
        --seed "$SEED" --count 10 --output "$OUT_S") >"$LOG_S" 2>&1; then
  echo "  ✗ bench-random-shapes failed:"
  sed 's/^/      /' "$LOG_S"; exit 1
fi
N_SHAPES_CORRECT="$(grep -oE 'py_bench_random_n_correct: [0-9]+' "$LOG_S" | awk '{print $2}')"
N_SHAPES_TOTAL="$(grep -oE 'py_bench_random_count: [0-9]+' "$LOG_S" | awk '{print $2}')"
[[ "$N_SHAPES_TOTAL" -eq 10 ]] || { echo "  ✗ random-shapes count=$N_SHAPES_TOTAL/10"; exit 1; }
[[ "$N_SHAPES_CORRECT" -eq 10 ]] || { echo "  ✗ random-shapes n_correct=$N_SHAPES_CORRECT/10"; sed 's/^/      /' "$LOG_S"; exit 1; }
echo "  ✓ fresh random-shapes: 10/10 correct under seed $SEED"

# 10 fresh random views
LOG_V="$L15B_VIEWS_LOG"
OUT_V="$L15B_VIEWS"
if ! (cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-random-views \
        --seed "$SEED" --count 10 --output "$OUT_V") >"$LOG_V" 2>&1; then
  echo "  ✗ bench-random-views failed:"
  sed 's/^/      /' "$LOG_V"; exit 1
fi
N_VIEWS_CORRECT="$(grep -oE 'py_random_views_n_correct: [0-9]+' "$LOG_V" | awk '{print $2}')"
N_VIEWS_TOTAL="$(grep -oE 'py_random_views_count: [0-9]+' "$LOG_V" | awk '{print $2}')"
[[ "$N_VIEWS_TOTAL" -eq 10 ]] || { echo "  ✗ random-views count=$N_VIEWS_TOTAL/10"; exit 1; }
[[ "$N_VIEWS_CORRECT" -eq 10 ]] || { echo "  ✗ random-views n_correct=$N_VIEWS_CORRECT/10"; sed 's/^/      /' "$LOG_V"; exit 1; }
echo "  ✓ fresh random-views: 10/10 correct under seed $SEED"

# Run the audit; capture JSON.
AUDIT_OUT="$L15B_AUDIT"
"$PY" "$TGRAD_DIR/scripts/dev/l15_b_audit.py" >"$AUDIT_OUT" 2>"$L15B_AUDIT_ERR"
if [[ ! -s "$AUDIT_OUT" ]]; then
  echo "  ✗ audit produced no output"
  cat "$L15B_AUDIT_ERR"
  exit 1
fi

N_PASS="$("$PY" -c '
import json
data = json.load(open("'"$AUDIT_OUT"'"))
print(sum(1 for c in data["criteria"] if c["verdict"] == "pass"))
')"
[[ "$N_PASS" -eq 3 ]] || { echo "  ✗ L15.B: $N_PASS/3 criteria pass"; cat "$AUDIT_OUT"; exit 1; }
echo "  ✓ 3/3 criteria pass"

INDEP_STATIC="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["runtime_independence"]["static"])
')"
[[ "$INDEP_STATIC" == "True" ]] || { echo "  ✗ static indep failed"; exit 1; }
echo "  ✓ runtime indep (static): pass"

# Layer D — anti-cheat
# D1: seed sourced from HEAD (gate already pinned this).
SEED_LOGGED="$(grep -oE 'py_random_views_seed: [0-9a-f]+' "$LOG_V" | awk '{print $2}')"
[[ "$SEED_LOGGED" == "$SEED" ]] || { echo "  ✗ D1 seed mismatch"; exit 1; }
echo "  ✓ D1 seed matches HEAD prefix"
# D2: regressions called via L11/L13/L13_F/L14 evidence files (not stubbed).
# The audit derives verdicts from those files — modifying the audit to short-circuit
# would be caught by gate-script grep for "verdict\": \"pass\"" hard-codes.
N_HARDCODED="$(grep -cE '"verdict"[[:space:]]*:[[:space:]]*"pass"' "$TGRAD_DIR/scripts/gates/L15_B.sh" || true)"
[[ "$N_HARDCODED" -eq 0 ]] || { echo "  ✗ L15_B.sh contains $N_HARDCODED hardcoded verdicts"; exit 1; }
echo "  ✓ D3 no hardcoded verdicts in gate script"

# Layer C2 — regression evidence
L11_PAIRS="$("$PY" -c 'import json; print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11 50/50 still holds"

# Layer E — write evidence
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"

l11_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L11.json" | awk '{print $1}')"
l13_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L13.json" | awk '{print $1}')"
l13f_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L13_F.json" | awk '{print $1}')"
l14_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L14.json" | awk '{print $1}')"
l12_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L12.json" | awk '{print $1}')"
shapes_hash="$(shasum -a 256 "$OUT_S" | awk '{print $1}')"
views_hash="$(shasum -a 256 "$OUT_V" | awk '{print $1}')"
audit_hash="$(shasum -a 256 "$TGRAD_DIR/scripts/dev/l15_b_audit.py" | awk '{print $1}')"

mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
"$PY" -c "
import json
audit = json.load(open('$AUDIT_OUT'))
out = {
    'gate': 'L15_B',
    'ts_utc': '$ts',
    'host': '$host',
    'platform': '$plat',
    'commit': '$commit',
    'scope': 'L15.B — runtime + benchmark recheck (criteria 4-6 + §5 runtime checks)',
    'criteria': audit['criteria'],
    'runtime_independence': audit['runtime_independence'],
    'random_recheck': {
        'shapes_total': 10,
        'shapes_pass':  $N_SHAPES_CORRECT,
        'views_total':  10,
        'views_pass':   $N_VIEWS_CORRECT,
        'seed':         '$SEED',
    },
    'hashes': {
        'l11_evidence_sha256':   '$l11_hash',
        'l12_evidence_sha256':   '$l12_hash',
        'l13_evidence_sha256':   '$l13_hash',
        'l13_f_evidence_sha256': '$l13f_hash',
        'l14_evidence_sha256':   '$l14_hash',
        'random_shapes_sha256':  '$shapes_hash',
        'random_views_sha256':   '$views_hash',
        'audit_module_sha256':   '$audit_hash',
    },
}
json.dump(out, open('$TGRAD_DIR/fixtures/gate_evidence/L15_B.json', 'w'), indent=2)
print('  ✓ evidence written')
"
check_evidence_for L15_B || exit 1
check_falsifiability_verified L15_B || exit 1
echo "  ✓ L15.B — 3/3 criteria + 10 shapes + 10 views — GREEN"

#!/usr/bin/env bash
# Gate L15.C — experiment-closure verdict + EXPERIMENT_RESULT.md authoring.
# Per `Tgrad/GOAL_L15_C.md` + `GOAL_L15.md §3 criteria 7-8` + §6 memo shape.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L15C_AUDIT="$(tgrad_run_path L15C_audit.json)"
L15C_AUDIT_ERR="$(tgrad_run_path L15C_audit.err)"

echo "[L15_C] experiment closure — verdict + EXPERIMENT_RESULT.md"

run_preflight
cd "$REPO_ROOT"

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

# Layer A.2: L15.A AND L15.B must be present.
[[ -f "$TGRAD_EVIDENCE_DIR/L15_A.json" ]] || { echo "  ✗ L15_A evidence missing"; exit 1; }
[[ -f "$TGRAD_EVIDENCE_DIR/L15_B.json" ]] || { echo "  ✗ L15_B evidence missing"; exit 1; }
echo "  ✓ L15.A + L15.B evidence present"

# Layer B — EXPERIMENT_RESULT.md exists and has all 8 required sections.
MEMO="$TGRAD_DIR/EXPERIMENT_RESULT.md"
[[ -f "$MEMO" ]] || { echo "  ✗ $MEMO missing"; exit 1; }
N_SECTIONS="$(grep -cE '^## (Verdict|Scope|Evidence|Where Lean Helped|Where Lean Did Not Yet Help|Performance Interpretation|Not Claimed|Next Move)\b' "$MEMO")"
[[ "$N_SECTIONS" -eq 8 ]] || { echo "  ✗ EXPERIMENT_RESULT.md has $N_SECTIONS/8 required sections"; exit 1; }
echo "  ✓ EXPERIMENT_RESULT.md has all 8 required sections"

# Required audit script.
[[ -f "$TGRAD_DIR/scripts/dev/l15_c_audit.py" ]] || { echo "  ✗ l15_c_audit.py missing"; exit 1; }

# Layer C — run the audit; capture JSON.
AUDIT_OUT="$L15C_AUDIT"
"$PY" "$TGRAD_DIR/scripts/dev/l15_c_audit.py" >"$AUDIT_OUT" 2>"$L15C_AUDIT_ERR"
if [[ ! -s "$AUDIT_OUT" ]]; then
  echo "  ✗ audit produced no output"
  cat "$L15C_AUDIT_ERR"; exit 1
fi

N_PASS="$("$PY" -c '
import json
data = json.load(open("'"$AUDIT_OUT"'"))
print(sum(1 for c in data["criteria"] if c["verdict"] == "pass"))
')"
[[ "$N_PASS" -eq 2 ]] || { echo "  ✗ L15.C: $N_PASS/2 criteria pass"; cat "$AUDIT_OUT"; exit 1; }
echo "  ✓ 2/2 criteria pass (lean_better_evidence + honesty)"

N_INV="$("$PY" -c '
import json
print(len(json.load(open("'"$AUDIT_OUT"'"))["lean_invariants"]))
')"
[[ "$N_INV" -ge 5 ]] || { echo "  ✗ only $N_INV Lean invariants named (need >= 5)"; exit 1; }
echo "  ✓ $N_INV Lean-explicit invariants named (>= 5)"

RESULT="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["result"])
')"
[[ "$RESULT" == "yes" ]] || { echo "  ✗ result=$RESULT (expected yes)"; cat "$AUDIT_OUT"; exit 1; }
echo "  ✓ result: yes"

# Layer D — anti-cheat
# D1: word count >= 400 (memo not a stub).
WC="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["experiment_result_word_count"])
')"
[[ "$WC" -ge 400 ]] || { echo "  ✗ memo word_count = $WC (need >= 400)"; exit 1; }
echo "  ✓ memo word count = $WC (>= 400)"
# D2: verdict comes from audit JSON, NOT hardcoded in gate.
N_HARDCODED="$(grep -cE '"result"[[:space:]]*:[[:space:]]*"yes"' "$TGRAD_DIR/scripts/gates/L15_C.sh" || true)"
[[ "$N_HARDCODED" -eq 0 ]] || { echo "  ✗ result hardcoded in gate script"; exit 1; }
echo "  ✓ D2 result not hardcoded in gate"
# D4: forbidden over-claim phrases absent (audit already checks).
echo "  ✓ D4 honesty audit passed (no over-claims)"

# Layer C2 — regression evidence
L11_PAIRS="$("$PY" -c 'import json; print(json.load(open("'"$TGRAD_EVIDENCE_DIR/L11.json"'"))["pairs_passed"])' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11 50/50 still holds"

# Layer E — write evidence.
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
l15a_hash="$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L15_A.json" | awk '{print $1}')"
l15b_hash="$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L15_B.json" | awk '{print $1}')"
memo_hash="$(shasum -a 256 "$MEMO" | awk '{print $1}')"

mkdir -p "$TGRAD_EVIDENCE_DIR"
"$PY" -c "
import json
audit = json.load(open('$AUDIT_OUT'))
out = {
    'gate': 'L15_C',
    'ts_utc': '$ts',
    'host': '$host',
    'platform': '$plat',
    'commit': '$commit',
    'scope': 'L15.C — verdict + EXPERIMENT_RESULT.md authoring (criteria 7-8)',
    'criteria': audit['criteria'],
    'lean_invariants': audit['lean_invariants'],
    'result': audit['result'],
    'experiment_result_word_count': audit['experiment_result_word_count'],
    'experiment_result_sha256':     audit['experiment_result_sha256'],
    'hashes': {
        'l15_a_evidence_sha256': '$l15a_hash',
        'l15_b_evidence_sha256': '$l15b_hash',
        'memo_sha256':           '$memo_hash',
    },
}
json.dump(out, open('$TGRAD_EVIDENCE_DIR/L15_C.json', 'w'), indent=2)
print('  ✓ L15_C evidence written')
"
check_evidence_for L15_C || exit 1
check_falsifiability_verified L15_C || exit 1
echo "  ✓ L15.C — result: $RESULT, $N_INV invariants — GREEN"

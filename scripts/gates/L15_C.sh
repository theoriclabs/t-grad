#!/usr/bin/env bash
# Gate L15.C — experiment-closure verdict + EXPERIMENT_RESULT.md authoring.
# Per `Tgrad/GOAL_L15_C.md` + `GOAL_L15.md §3 criteria 7-8` + §6 memo shape.
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L15_C] experiment closure — verdict + EXPERIMENT_RESULT.md"

run_preflight
cd "$REPO_ROOT"

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

# Observer self-test runs before consuming gate evidence or the real memo.
[[ -f "$TGRAD_DIR/scripts/dev/l15_c_audit.py" ]] \
  || { echo "  ✗ l15_c_audit.py missing"; exit 1; }
SELFTEST_LOG="$(mktemp "${TMPDIR:-/tmp}/tgrad_L15_C_selftest.XXXXXX")"
trap 'rm -f "$SELFTEST_LOG"' EXIT
if ! "$PY" "$TGRAD_DIR/scripts/dev/l15_c_audit.py" --self-test \
        >"$SELFTEST_LOG" 2>&1; then
  echo "  ✗ L15.C observer self-test failed:"
  sed 's/^/      /' "$SELFTEST_LOG"
  exit 1
fi
grep -qF 'l15_c_audit_self_test: pass' "$SELFTEST_LOG" \
  || { echo "  ✗ L15.C observer self-test did not report pass"; exit 1; }
grep -qF 'l15_c_self_test_promoted_result: inconclusive' "$SELFTEST_LOG" \
  || { echo "  ✗ L15.C self-test lost promoted-result identity"; exit 1; }
grep -qF 'l15_c_self_test_evidence_kind: historical' "$SELFTEST_LOG" \
  || { echo "  ✗ L15.C self-test lost evidence provenance"; exit 1; }
echo "  ✓ observer self-test: scoped result parsing + historical coherence"

# Layer A.2: L15.A AND L15.B must be present.
[[ -f "$TGRAD_DIR/fixtures/gate_evidence/L15_A.json" ]] || { echo "  ✗ L15_A evidence missing"; exit 1; }
[[ -f "$TGRAD_DIR/fixtures/gate_evidence/L15_B.json" ]] || { echo "  ✗ L15_B evidence missing"; exit 1; }
echo "  ✓ L15.A + L15.B evidence present"

# Layer B — the memo exists. Exact section/declaration/coherence structure is
# parsed by the observer below, including the honest historical-evidence form.
MEMO="$TGRAD_DIR/EXPERIMENT_RESULT.md"
[[ -f "$MEMO" ]] || { echo "  ✗ $MEMO missing"; exit 1; }

# Layer C — run the audit; capture JSON.
AUDIT_OUT="$(mktemp "${TMPDIR:-/tmp}/tgrad_L15_C_audit.XXXXXX")"
AUDIT_ERR="$(mktemp "${TMPDIR:-/tmp}/tgrad_L15_C_audit_err.XXXXXX")"
trap 'rm -f "$SELFTEST_LOG" "$AUDIT_OUT" "$AUDIT_ERR"' EXIT
"$PY" "$TGRAD_DIR/scripts/dev/l15_c_audit.py" >"$AUDIT_OUT" 2>"$AUDIT_ERR"
if [[ ! -s "$AUDIT_OUT" ]]; then
  echo "  ✗ audit produced no output"
  cat "$AUDIT_ERR"; exit 1
fi

MEMO_VALID="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["memo_contract"]["valid"])
')"
if [[ "$MEMO_VALID" != "True" ]]; then
  echo "  ✗ EXPERIMENT_RESULT.md contract is incoherent:"
  "$PY" -c '
import json
d=json.load(open("'"$AUDIT_OUT"'"))["memo_contract"]
print("\n".join(f"      {e}" for e in d["errors"]))
'
  exit 1
fi
echo "  ✓ memo shape, promoted declaration, and evidence provenance are coherent"

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

COMPUTED_ANSWER="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["computed_answer"])
')"
PROMOTED_RESULT="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["promoted_result"])
')"
EVIDENCE_KIND="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["memo_contract"]["evidence_kind"])
')"
[[ "$COMPUTED_ANSWER" =~ ^(yes|no|inconclusive)$ ]] \
  || { echo "  ✗ malformed computed answer=$COMPUTED_ANSWER"; exit 1; }
[[ "$PROMOTED_RESULT" =~ ^(yes|no|inconclusive)$ ]] \
  || { echo "  ✗ malformed promoted result=$PROMOTED_RESULT"; exit 1; }
echo "  ✓ computed answer: $COMPUTED_ANSWER"
echo "  ✓ promoted result: $PROMOTED_RESULT (evidence kind: $EVIDENCE_KIND)"

# Layer D — anti-cheat
# D1: word count >= 400 (memo not a stub).
WC="$("$PY" -c '
import json
print(json.load(open("'"$AUDIT_OUT"'"))["experiment_result_word_count"])
')"
[[ "$WC" -ge 400 ]] || { echo "  ✗ memo word_count = $WC (need >= 400)"; exit 1; }
echo "  ✓ memo word count = $WC (>= 400)"
# D2: promoted result comes from audit JSON, not a gate literal.
N_HARDCODED="$(grep -cE '"promoted_result"[[:space:]]*:[[:space:]]*"(yes|no|inconclusive)"' "$TGRAD_DIR/scripts/gates/L15_C.sh" || true)"
[[ "$N_HARDCODED" -eq 0 ]] || { echo "  ✗ promoted result hardcoded in gate script"; exit 1; }
echo "  ✓ D2 promoted result not hardcoded in gate"
# D4: forbidden over-claim phrases absent (audit already checks).
echo "  ✓ D4 honesty audit passed (no over-claims)"

# Layer C2 — regression evidence
L11_PAIRS="$("$PY" -c 'import json; print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11 50/50 still holds"

# Layer E — write evidence.
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
l15a_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L15_A.json" | awk '{print $1}')"
l15b_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L15_B.json" | awk '{print $1}')"
memo_hash="$(shasum -a 256 "$MEMO" | awk '{print $1}')"

mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
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
    'memo_contract': audit['memo_contract'],
    'computed_answer': audit['computed_answer'],
    'promoted_result': audit['promoted_result'],
    'experiment_result_word_count': audit['experiment_result_word_count'],
    'experiment_result_sha256':     audit['experiment_result_sha256'],
    'hashes': {
        'l15_a_evidence_sha256': '$l15a_hash',
        'l15_b_evidence_sha256': '$l15b_hash',
        'memo_sha256':           '$memo_hash',
    },
}
json.dump(out, open('$TGRAD_DIR/fixtures/gate_evidence/L15_C.json', 'w'), indent=2)
print('  ✓ L15_C evidence written')
"
# The evidence must propagate both identities exactly; substituting the
# computed answer for the promoted result is a gate failure, not a promotion.
"$PY" -c '
import json
audit=json.load(open("'"$AUDIT_OUT"'"))
evidence=json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L15_C.json"'"))
assert evidence["computed_answer"] == audit["computed_answer"]
assert evidence["promoted_result"] == audit["promoted_result"]
' || { echo "  ✗ L15.C result identities were not propagated exactly"; exit 1; }
echo "  ✓ computed and promoted result identities propagated exactly"
check_evidence_for L15_C || exit 1
check_falsifiability_verified L15_C || exit 1
echo "  ✓ L15.C — coherent closure; promoted result: $PROMOTED_RESULT, computed answer: $COMPUTED_ANSWER, $N_INV invariants — GREEN"

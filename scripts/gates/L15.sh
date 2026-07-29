#!/usr/bin/env bash
# Gate L15 umbrella — verifies L15.A + L15.B + L15.C and propagates the
# computed answer and memo-promoted result as distinct identities.
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L15] umbrella — experiment closure audit"

run_preflight

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

for sub in L15_A L15_B L15_C; do
  ev="$TGRAD_DIR/fixtures/gate_evidence/${sub}.json"
  [[ -f "$ev" ]] || { echo "  ✗ missing sub-gate evidence: $ev"; exit 1; }
  echo "  ✓ $sub evidence present"
done

# The umbrella records the memo promotion exactly. Green means coherent
# closure, not that the promoted result is necessarily `yes`.
PROMOTED_RESULT="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L15_C.json"'"))["promoted_result"])
')"
COMPUTED_ANSWER="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L15_C.json"'"))["computed_answer"])
')"
MEMO_VALID="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L15_C.json"'"))["memo_contract"]["valid"])
')"
[[ "$PROMOTED_RESULT" =~ ^(yes|no|inconclusive)$ ]] \
  || { echo "  ✗ malformed L15.C promoted result = $PROMOTED_RESULT"; exit 1; }
[[ "$COMPUTED_ANSWER" =~ ^(yes|no|inconclusive)$ ]] \
  || { echo "  ✗ malformed L15.C computed answer = $COMPUTED_ANSWER"; exit 1; }
[[ "$MEMO_VALID" == "True" ]] \
  || { echo "  ✗ L15.C memo contract is not coherent"; exit 1; }
echo "  ✓ L15.C promoted result: $PROMOTED_RESULT"
echo "  ✓ L15.C computed answer: $COMPUTED_ANSWER"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
sha_a="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L15_A.json" | awk '{print $1}')"
sha_b="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L15_B.json" | awk '{print $1}')"
sha_c="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L15_C.json" | awk '{print $1}')"
memo_hash="$(shasum -a 256 "$TGRAD_DIR/EXPERIMENT_RESULT.md" | awk '{print $1}')"

mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L15.json" <<EOF
{
  "gate": "L15",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L15 umbrella — coherent experiment closure (3 sub-gates + result identities)",
  "computed_answer": "$COMPUTED_ANSWER",
  "promoted_result": "$PROMOTED_RESULT",
  "sub_gates_green": ["L15_A", "L15_B", "L15_C"],
  "hashes": {
    "L15_A_evidence_sha256": "$sha_a",
    "L15_B_evidence_sha256": "$sha_b",
    "L15_C_evidence_sha256": "$sha_c",
    "memo_sha256":           "$memo_hash"
  }
}
EOF
# Reject a roll-up that substitutes the computed answer for the memo's
# promoted result, even when both values are individually enumerated.
"$PY" -c '
import json
c=json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L15_C.json"'"))
u=json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L15.json"'"))
assert u["computed_answer"] == c["computed_answer"]
assert u["promoted_result"] == c["promoted_result"]
' || { echo "  ✗ L15 umbrella changed computed/promoted result identity"; exit 1; }
echo "  ✓ result identities propagated exactly"
check_evidence_for L15 || exit 1
check_falsifiability_verified L15 || exit 1
echo "  ✓ L15 umbrella — coherent closure, promoted result: $PROMOTED_RESULT, computed answer: $COMPUTED_ANSWER — GREEN"

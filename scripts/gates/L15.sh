#!/usr/bin/env bash
# Gate L15 umbrella — verifies L15.A + L15.B + L15.C; rolls up evidence
# into L15.json with the final `result` field copied from L15_C.json.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
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

# The L15 umbrella's `result` field is the L15.C audit's result.
RESULT="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L15_C.json"'"))["result"])
')"
[[ "$RESULT" == "yes" ]] || { echo "  ✗ L15.C result = $RESULT (umbrella refuses to flip unless yes)"; exit 1; }
echo "  ✓ L15.C result: yes"

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
  "scope": "L15 umbrella — experiment-closure audit (3 sub-gates + final verdict)",
  "result": "$RESULT",
  "sub_gates_green": ["L15_A", "L15_B", "L15_C"],
  "hashes": {
    "L15_A_evidence_sha256": "$sha_a",
    "L15_B_evidence_sha256": "$sha_b",
    "L15_C_evidence_sha256": "$sha_c",
    "memo_sha256":           "$memo_hash"
  }
}
EOF
check_evidence_for L15 || exit 1
check_falsifiability_verified L15 || exit 1
echo "  ✓ L15 umbrella — result: $RESULT — GREEN"

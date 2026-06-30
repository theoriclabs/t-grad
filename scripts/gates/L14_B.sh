#!/usr/bin/env bash
# Gate L14.B umbrella — verifies L14.B.1 + L14.B.2 + L14.B.3 evidence
# is intact; rolls up sha256s into L14_B.json.
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14_B] umbrella — view methods + index-UOp codegen + 16 pinned views"

run_preflight

for sub in L14_B_1 L14_B_2 L14_B_3; do
  ev="$TGRAD_DIR/fixtures/gate_evidence/${sub}.json"
  [[ -f "$ev" ]] || { echo "  ✗ missing sub-gate evidence: $ev"; exit 1; }
  echo "  ✓ $sub evidence present"
done

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
sha_1="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L14_B_1.json" | awk '{print $1}')"
sha_2="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L14_B_2.json" | awk '{print $1}')"
sha_3="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L14_B_3.json" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L14_B.json" <<EOF
{
  "gate": "L14_B",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14.B umbrella — view methods (5 movement ctors) + index-UOp-driven matmul codegen + 16 pinned view-matmul cases passing",
  "sub_gates_green": ["L14_B_1", "L14_B_2", "L14_B_3"],
  "hashes": {
    "L14_B_1_evidence_sha256": "$sha_1",
    "L14_B_2_evidence_sha256": "$sha_2",
    "L14_B_3_evidence_sha256": "$sha_3"
  }
}
EOF
check_evidence_for L14_B || exit 1
check_falsifiability_verified L14_B || exit 1
echo "  ✓ L14.B umbrella gate green (3/3 sub-gates green)"

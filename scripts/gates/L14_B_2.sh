#!/usr/bin/env bash
# Gate L14.B.2 umbrella — verifies L14.B.2.a + L14.B.2.b + L14.B.2.c
# all have valid evidence files; rolls up sha256s into L14_B_2.json.
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14_B_2] umbrella — index-UOp-driven matmul codegen + rangeify wiring"

run_preflight

for sub in L14_B_2_a L14_B_2_b L14_B_2_c; do
  ev="$TGRAD_DIR/fixtures/gate_evidence/${sub}.json"
  [[ -f "$ev" ]] || { echo "  ✗ missing sub-gate evidence: $ev"; exit 1; }
  echo "  ✓ $sub evidence present"
done

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
sha_a="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L14_B_2_a.json" | awk '{print $1}')"
sha_b="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L14_B_2_b.json" | awk '{print $1}')"
sha_c="$(shasum -a 256 "$TGRAD_DIR/fixtures/gate_evidence/L14_B_2_c.json" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L14_B_2.json" <<EOF
{
  "gate": "L14_B_2",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14.B.2 umbrella — index-UOp-driven matmul codegen + Schedule.Rangeify wired into Pipeline.realize",
  "sub_gates_green": ["L14_B_2_a", "L14_B_2_b", "L14_B_2_c"],
  "hashes": {
    "L14_B_2_a_evidence_sha256": "$sha_a",
    "L14_B_2_b_evidence_sha256": "$sha_b",
    "L14_B_2_c_evidence_sha256": "$sha_c"
  }
}
EOF
check_evidence_for L14_B_2 || exit 1
check_falsifiability_verified L14_B_2 || exit 1
echo "  ✓ L14.B.2 umbrella gate green (3/3 sub-gates green)"

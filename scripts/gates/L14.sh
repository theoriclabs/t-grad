#!/usr/bin/env bash
# Gate L14 umbrella — verifies L14.A + L14.B + L14.C; rolls up
# sha256s into L14.json.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14] umbrella — matmul through movement-op views"

run_preflight

for sub in L14_A L14_B L14_C; do
  ev="$TGRAD_EVIDENCE_DIR/${sub}.json"
  [[ -f "$ev" ]] || { echo "  ✗ missing sub-gate evidence: $ev"; exit 1; }
  echo "  ✓ $sub evidence present"
done

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
sha_a="$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L14_A.json" | awk '{print $1}')"
sha_b="$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L14_B.json" | awk '{print $1}')"
sha_c="$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L14_C.json" | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L14.json" <<EOF
{
  "gate": "L14",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14 umbrella — matmul through movement-op views (Tensor.uop refactor + view methods + 16 pinned + 20 random)",
  "sub_gates_green": ["L14_A", "L14_B", "L14_C"],
  "hashes": {
    "L14_A_evidence_sha256": "$sha_a",
    "L14_B_evidence_sha256": "$sha_b",
    "L14_C_evidence_sha256": "$sha_c"
  }
}
EOF
check_evidence_for L14 || exit 1
check_falsifiability_verified L14 || exit 1
echo "  ✓ L14 umbrella gate green (3/3 sub-gates green)"

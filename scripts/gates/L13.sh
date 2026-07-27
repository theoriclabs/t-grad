#!/usr/bin/env bash
# Gate L13 — umbrella for the 5 sub-gates (L13.A..L13.E).
#
# Per `GOAL_NEXT.md §8.RESUME` (sub-gate decomposition): "After
# L13.A + L13.B + L13.C + L13.D + L13.E are all green, add `L13`
# to `GREEN_GATES` (the umbrella entry; the sub-gates stay in the
# array as well, so removing them trips `check_no_gate_regression`)."
#
# This umbrella verifies that all 5 sub-gate evidence files exist
# and have the expected scope marker. It does NOT re-run the
# sub-gates (each is invoked separately as part of the full sweep
# via GREEN_GATES).
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L13] umbrella — verifying L13.A..L13.E sub-gates are green"

run_preflight

# Each sub-gate's evidence file must exist + carry the expected gate field.
SUBGATES=("L13_A" "L13_B" "L13_C" "L13_D" "L13_E")
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
for sg in "${SUBGATES[@]}"; do
  file="$TGRAD_EVIDENCE_DIR/$sg.json"
  if [[ ! -f "$file" ]]; then
    echo "  ✗ sub-gate $sg evidence missing: $file"
    exit 1
  fi
  recorded_gate=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('gate',''))" "$file")
  if [[ "$recorded_gate" != "$sg" ]]; then
    echo "  ✗ sub-gate $sg evidence has wrong 'gate' field: $recorded_gate"
    exit 1
  fi
  echo "  ✓ $sg evidence present + valid"
done

# Each sub-gate must also be in GREEN_GATES (the ratchet).
GREEN_LINE=$(grep '^GREEN_GATES=' "$TGRAD_DIR/scripts/gate.sh")
for sg in "${SUBGATES[@]}"; do
  if ! echo "$GREEN_LINE" | grep -qE "\b$sg\b"; then
    echo "  ✗ $sg not in GREEN_GATES (ratchet inconsistent)"
    echo "    $GREEN_LINE"
    exit 1
  fi
done
echo "  ✓ all 5 sub-gates in GREEN_GATES"

# Evidence
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
mkdir -p "$TGRAD_EVIDENCE_DIR"
# Sub-gate evidence sha256s (pin the entire L13 ecosystem in one umbrella).
HASH_A=$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L13_A.json" | awk '{print $1}')
HASH_B=$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L13_B.json" | awk '{print $1}')
HASH_C=$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L13_C.json" | awk '{print $1}')
HASH_D=$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L13_D.json" | awk '{print $1}')
HASH_E=$(shasum -a 256 "$TGRAD_EVIDENCE_DIR/L13_E.json" | awk '{print $1}')
cat >"$TGRAD_EVIDENCE_DIR/L13.json" <<EOF
{
  "gate": "L13",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L13 umbrella — verifies L13.A + L13.B + L13.C + L13.D + L13.E all green",
  "sub_gates_total":  5,
  "sub_gates_green":  5,
  "sub_gates": ["L13_A", "L13_B", "L13_C", "L13_D", "L13_E"],
  "hashes": {
    "L13_A_evidence_sha256": "$HASH_A",
    "L13_B_evidence_sha256": "$HASH_B",
    "L13_C_evidence_sha256": "$HASH_C",
    "L13_D_evidence_sha256": "$HASH_D",
    "L13_E_evidence_sha256": "$HASH_E"
  }
}
EOF
check_evidence_for L13 || exit 1
check_falsifiability_verified L13 || exit 1
echo "  ✓ L13 umbrella green (A..E all green)"

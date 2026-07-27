#!/usr/bin/env bash
# Gate L0 — scaffold.
#
# Done when: the lakefile, three module stubs (Dtype, Shape, UOp), and
# the two executables build and run cleanly. No correctness claims yet.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L0_TEST_LOG="$(tgrad_run_path L0_tests.log)"

echo "[L0] scaffold"

# Universal checks first.
run_preflight

# Lake build + the Lean test exe.
./.lake/build/bin/tgrad-tests >"$L0_TEST_LOG" 2>&1 || {
  echo "  ✗ tgrad-tests exited nonzero"; cat "$L0_TEST_LOG"; exit 1
}

# Specific predicate: scaffold layer prints its OK line.
grep -qF 'scaffold layer ✓' "$L0_TEST_LOG" || {
  echo "  ✗ tgrad-tests did not print the scaffold OK line"; cat "$L0_TEST_LOG"; exit 1
}

# Write evidence.
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"
plat="$(uname -srm)"
build_hash="$(shasum -a 256 .lake/build/bin/tgrad-tests 2>/dev/null | awk '{print $1}' || echo unknown)"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L0.json" <<EOF
{
  "gate": "L0",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "hashes": { "tgrad_tests_binary": "$build_hash" }
}
EOF

check_evidence_for L0 || exit 1
check_falsifiability_verified L0 || exit 1
echo "  ✓ L0 scaffold gate green"

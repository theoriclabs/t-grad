#!/usr/bin/env bash
# Gate L9 — extended symbolic_simple rule library.
#
# Per P7 (GOAL_NEXT.md §G9 + §G7 fall-back): L1.a's 16-rule subset is
# extended to 22 rules in L9.a, with new rules for commutative duals
# (0+x, 1*x), idempotents (x|x), and subtraction identities (x-x,
# x-0, x|0). Full 62-rule port (incl. bool, where, cast, threefry,
# propagate_invalid) is L9.b expansion work.
#
# Per §6 rule 1: the scope-narrow correctness requirement is that
# (a) the L1 47-node DAG still reduces byte-equally (regression);
# (b) a new L9 fixture exercising the 6 new rules reduces byte-equally
# to its captured expected output (new behavioural).
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : Symbolic.lean has ≥ 22 rules; L9 input + expected
#               fixtures present
#   - Layer C1: L1 47-node DAG byte-equal to dag_out_expected.json
#               (regression — the L1 16 rules still work)
#   - Layer C2: L9 9-node DAG byte-equal to dag_out_l9_expected.json
#               (new — the 6 new rules collapse the chain to x)
#   - Layer D : negative test — reduce-symbolic-dag rejects malformed
#               JSON path with nonzero exit
#   - Layer E : evidence file
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L9_BASE="$(tgrad_run_path L9_base.json)"
L9_NEW="$(tgrad_run_path L9_new.json)"
L9_NEG="$(tgrad_run_path L9_negative.txt)"

echo "[L9] extended symbolic_simple (L1.a 16 + L9.a 6 = 22 rules)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
# Count rule entries in ruleSet — must be ≥ 22.
RULE_COUNT="$(grep -cE '^\s*\{ pat :=' "$TGRAD_DIR/Tgrad/Rules/Symbolic.lean" || true)"
if [[ "$RULE_COUNT" -lt 22 ]]; then
  echo "  ✗ Symbolic.lean has only $RULE_COUNT rules; L9.a requires ≥ 22"
  exit 1
fi
echo "  ✓ Symbolic.lean has $RULE_COUNT rules (≥ 22 required for L9.a)"

required_fixtures=(
  fixtures/symbolic/dag_in.json
  fixtures/symbolic/dag_out_expected.json
  fixtures/symbolic/dag_in_l9.json
  fixtures/symbolic/dag_out_l9_expected.json
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} required fixtures present"

# ─── LAYER C1: L1 regression ──────────────────────────────────────────
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" \
    reduce-symbolic-dag fixtures/symbolic/dag_in.json) \
    >"$L9_BASE" 2>&1 || {
  echo "  ✗ reduce-symbolic-dag on L1 47-node fixture failed"
  cat "$L9_BASE"; exit 1
}
if ! diff -q "$L9_BASE" "$TGRAD_DIR/fixtures/symbolic/dag_out_expected.json" >/dev/null; then
  echo "  ✗ L1 47-node DAG reduces DIFFERENTLY after L9 rule additions (regression)"
  diff "$L9_BASE" "$TGRAD_DIR/fixtures/symbolic/dag_out_expected.json" | head -20
  exit 1
fi
echo "  ✓ L1 47-node DAG still reduces byte-equally (no regression from new rules)"

# ─── LAYER C2: L9 new-rule fixture ────────────────────────────────────
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" \
    reduce-symbolic-dag fixtures/symbolic/dag_in_l9.json) \
    >"$L9_NEW" 2>&1 || {
  echo "  ✗ reduce-symbolic-dag on L9 fixture failed"
  cat "$L9_NEW"; exit 1
}
if ! diff -q "$L9_NEW" "$TGRAD_DIR/fixtures/symbolic/dag_out_l9_expected.json" >/dev/null; then
  echo "  ✗ L9 fixture does NOT reduce to expected (the 6 new rules' chain)"
  diff "$L9_NEW" "$TGRAD_DIR/fixtures/symbolic/dag_out_l9_expected.json" | head -20
  exit 1
fi
echo "  ✓ L9 fixture (6 new rules: 0+x, 1*x, x-x, x-0, x|0, x|x) reduces correctly"

# ─── LAYER D: negative test — bogus path ──────────────────────────────
set +e
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" \
    reduce-symbolic-dag /nonexistent/path.json) >"$L9_NEG" 2>&1
neg_rc=$?
set -e
if [[ "$neg_rc" -eq 0 ]]; then
  echo "  ✗ reduce-symbolic-dag on bogus path returned 0 — should reject"
  exit 1
fi
echo "  ✓ negative test correctly rejected (bogus path → nonzero exit)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
l1_hash="$(shasum -a 256 "$L9_BASE" | awk '{print $1}')"
l9_hash="$(shasum -a 256 "$L9_NEW" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L9.json" <<EOF
{
  "gate": "L9",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L9.a — 22-rule extended symbolic_simple subset (L1's 16 + 6 new); full 62-rule port is L9.b",
  "rule_count": $RULE_COUNT,
  "hashes": {
    "l1_47node_reduce_sha256":   "$l1_hash",
    "l9_new_reduce_sha256":      "$l9_hash"
  }
}
EOF
check_evidence_for L9 || exit 1
check_falsifiability_verified L9 || exit 1
echo "  ✓ L9 extended-symbolic gate green (evidence recorded)"

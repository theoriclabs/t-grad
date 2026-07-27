#!/usr/bin/env bash
# Gate L10 — rangeify movement-op coverage (extended).
#
# Per P7 (GOAL_NEXT.md §G10 + §G7 fall-back): L2's 2-op rangeify
# (RESHAPE 4 → PERMUTE [2,2]) is extended with a new test fixture
# covering a different shape pair (RESHAPE 6 → PERMUTE [3,2]). The
# full pm_mops port (INDEX + AFTER + SHAPED_WMMA + ctx-carrying
# rules) requires UOp surface additions tracked as L10.b expansion.
#
# Per §6 rule 1: L10.a scope-narrows COVERAGE (1 additional fixture
# vs L2's 1 fixture), but the fixture covered byte-matches against
# captured Tgrad output. Each new fixture's byte-equal output is a
# real correctness guarantee.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : L2 base fixtures present; L10 fixtures present
#   - Layer C1: L2 fixture still reduces byte-equally (regression)
#   - Layer C2: L10 fixture reduces byte-equally to captured expected
#   - Layer D : negative test — bogus path rejected
#   - Layer E : evidence file
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L10_BASE="$(tgrad_run_path L10_base.json)"
L10_NEW="$(tgrad_run_path L10_new.json)"
L10_NEG="$(tgrad_run_path L10_negative.txt)"

echo "[L10] rangeify coverage extension"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_fixtures=(
  fixtures/schedule/rangeify_input.json
  fixtures/schedule/rangeify_expected.json
  fixtures/schedule/rangeify_input_l10.json
  fixtures/schedule/rangeify_expected_l10.json
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} required fixtures present"

# ─── LAYER C1: L2 fixture regression ──────────────────────────────────
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" \
    rangeify fixtures/schedule/rangeify_input.json) \
    >"$L10_BASE" 2>&1 || {
  echo "  ✗ rangeify on L2 fixture failed"
  cat "$L10_BASE"; exit 1
}
if ! diff -q "$L10_BASE" "$TGRAD_DIR/fixtures/schedule/rangeify_expected.json" >/dev/null; then
  echo "  ✗ L2 rangeify fixture diverges (regression)"
  diff "$L10_BASE" "$TGRAD_DIR/fixtures/schedule/rangeify_expected.json" | head -20
  exit 1
fi
echo "  ✓ L2 rangeify fixture still byte-matches (no regression)"

# ─── LAYER C2: L10 fixture behavioural ────────────────────────────────
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" \
    rangeify fixtures/schedule/rangeify_input_l10.json) \
    >"$L10_NEW" 2>&1 || {
  echo "  ✗ rangeify on L10 fixture failed"
  cat "$L10_NEW"; exit 1
}
if ! diff -q "$L10_NEW" "$TGRAD_DIR/fixtures/schedule/rangeify_expected_l10.json" >/dev/null; then
  echo "  ✗ L10 rangeify fixture does NOT byte-match captured expected"
  diff "$L10_NEW" "$TGRAD_DIR/fixtures/schedule/rangeify_expected_l10.json" | head -20
  exit 1
fi
echo "  ✓ L10 rangeify fixture byte-matches captured expected (RESHAPE 6 → PERMUTE [3,2])"

# ─── LAYER D: negative test ───────────────────────────────────────────
set +e
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" \
    rangeify /nonexistent/path.json) >"$L10_NEG" 2>&1
neg_rc=$?
set -e
if [[ "$neg_rc" -eq 0 ]]; then
  echo "  ✗ rangeify on bogus path returned 0 — should reject"
  exit 1
fi
echo "  ✓ negative test correctly rejected (bogus path → nonzero exit)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
l2_hash="$(shasum -a 256 "$L10_BASE" | awk '{print $1}')"
l10_hash="$(shasum -a 256 "$L10_NEW" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L10.json" <<EOF
{
  "gate": "L10",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L10.a — rangeify coverage extended (1 new fixture); full pm_mops + INDEX/AFTER/SHAPED_WMMA + ctx-carrying rules is L10.b",
  "hashes": {
    "l2_rangeify_sha256":   "$l2_hash",
    "l10_rangeify_sha256":  "$l10_hash"
  }
}
EOF
check_evidence_for L10 || exit 1
check_falsifiability_verified L10 || exit 1
echo "  ✓ L10 rangeify-coverage gate green (evidence recorded)"

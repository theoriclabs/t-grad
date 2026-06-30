#!/usr/bin/env bash
# Gate L<n> — <one-line description>.
#
# Copy this file to scripts/gates/L<n>.sh and fill in the predicates
# per GOAL.md §G<n>. The five sections below are mandatory for every
# gate; do not remove any (the strong-done framework — README §11 —
# depends on all five layers).
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

GATE_NAME="L<n>"     # TODO: replace with actual gate name
GATE_OUTCOME="<outcome>"  # TODO: one-line description

echo "[$GATE_NAME] $GATE_OUTCOME"

# ─── LAYER A: universal preflight ─────────────────────────────────────
# Rejects sorry/axiom/unsafe/stale-cache/ratchet-shrink/new-warnings.
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
# Lists required modules / theorems / fixtures. Every entry MUST exist.
required_modules=(
  # TODO: list each Tgrad/<path>.lean the gate requires
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

required_theorems=(
  # TODO: list each "Tgrad/<path>.lean:<theorem_name>"
)
for entry in "${required_theorems[@]}"; do
  file="${entry%:*}"; thm="${entry##*:}"
  if ! grep -qE "^theorem[[:space:]]+$thm\b" "$REPO_ROOT/$file"; then
    echo "  ✗ missing theorem: $thm in $file"
    exit 1
  fi
done
echo "  ✓ all ${#required_theorems[@]} required theorems declared (preflight already rejected sorry/axiom)"

required_fixtures=(
  # TODO: list each fixtures/<path>.json the gate consumes
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing required fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} required fixtures present"

# ─── LAYER C: behavioural cross-validation ────────────────────────────
# Each predicate: tgrad-cli emit-X > tmp; diff tmp against captured.
# Agent cannot fake — captured fixtures came from tinygrad.
#
# Pattern (repeat per predicate):
#   ./.lake/build/bin/tgrad-cli <subcommand> [args] >/tmp/tgrad_${GATE_NAME}_X.json 2>&1 || {
#     echo "  ✗ tgrad-cli <subcommand> failed"; cat /tmp/tgrad_${GATE_NAME}_X.json; exit 1
#   }
#   if ! diff -q /tmp/tgrad_${GATE_NAME}_X.json "$TGRAD_DIR/fixtures/<path>.json" >/dev/null; then
#     echo "  ✗ <name> disagrees with captured fixture"
#     diff /tmp/tgrad_${GATE_NAME}_X.json "$TGRAD_DIR/fixtures/<path>.json" | head -20
#     exit 1
#   fi
#   echo "  ✓ <name> matches captured fixture"
#
# TODO: instantiate one block per behavioural predicate in §G<n>.

# ─── LAYER D: negative tests ──────────────────────────────────────────
# Type system must reject deliberately-broken code. Write a small
# .lean snippet to /tmp/, try to compile, expect failure.
#
# Pattern:
#   cat >/tmp/tgrad_neg_${GATE_NAME}.lean <<'EOF'
#   import Tgrad
#   open Tgrad
#   -- deliberately broken construction
#   def bad := ...
#   EOF
#   if (cd "$TGRAD_DIR" && lake env lean /tmp/tgrad_neg_${GATE_NAME}.lean) >/tmp/neg.log 2>&1; then
#     echo "  ✗ negative test compiled — type system isn't rejecting bad input"
#     exit 1
#   fi
#   echo "  ✓ negative test correctly rejected"
#
# TODO: instantiate at least one block per gate.

# ─── LAYER E: evidence ────────────────────────────────────────────────
# Write an evidence file with hashes of every behavioural output.
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"

# TODO: replace with one shasum per behavioural-predicate output.
# Example:
#   x_hash="$(shasum -a 256 /tmp/tgrad_${GATE_NAME}_X.json | awk '{print $1}')"

mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/${GATE_NAME}.json" <<EOF
{
  "gate": "$GATE_NAME",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "hashes": {
    "TODO": "fill with shasums of behavioural outputs"
  }
}
EOF

check_evidence_for "$GATE_NAME" || exit 1
check_falsifiability_verified "$GATE_NAME" || exit 1
echo "  ✓ $GATE_NAME $GATE_OUTCOME gate green (evidence recorded)"

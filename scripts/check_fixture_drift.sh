#!/usr/bin/env bash
# Tgrad fixture-drift watch (informational).
#
# Tgrad's fixtures are *copies* of phase fixtures (Rule 6: lift, don't
# import). When a phase regenerates its fixture (e.g. tinygrad's
# upstream changes drove a `make capture`), Tgrad's copy goes stale
# silently — the ratchet only protects against *internal* regression,
# not drift between Tgrad and the phase that fed it.
#
# This script byte-diffs each fixtures/<module>/<file>.json
# against its `theograd_phases/.../fixtures/<file>.json` counterpart
# and reports the drift. **Exit 0 even if drift exists** — drift is a
# signal for the agent to consider (e.g. "tinygrad updated, do we
# adopt the new capture?"), not a hard fail.
#
# Usage:
#   bash scripts/check_fixture_drift.sh            # summary
#   bash scripts/check_fixture_drift.sh --strict   # exit 1 on any drift
#   bash scripts/check_fixture_drift.sh --diff     # print full diffs

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="summary"
case "${1:-}" in
  --strict) MODE="strict" ;;
  --diff)   MODE="diff"   ;;
  "")       MODE="summary" ;;
  *) echo "usage: $0 [--strict|--diff]"; exit 2 ;;
esac

# Tgrad fixture path → source phase fixture path. Add a row when L<n>
# lifts a new fixture into Tgrad.
declare -a TGRAD_FIXTURES PHASE_SOURCES
TGRAD_FIXTURES=(
  "fixtures/dtype/lub_table.json"
  "fixtures/dtype/can_lossless_cast_table.json"
  "fixtures/shape/shape_table.json"
  "fixtures/shape/movement_table.json"
  "fixtures/symbolic/dag_in.json"
  "fixtures/symbolic/dag_out_expected.json"
)
PHASE_SOURCES=(
  "theograd_phases/17_dtype_system/fixtures/lub_table.json"
  "theograd_phases/17_dtype_system/fixtures/can_lossless_cast_table.json"
  "theograd_phases/18_shape_arithmetic/fixtures/shape_table.json"
  "theograd_phases/18_shape_arithmetic/fixtures/movement_table.json"
  "theograd_phases/03_pattern_inventory/fixtures/dag_in.json"
  "theograd_phases/03_pattern_inventory/fixtures/dag_out_expected.json"
)

drift_count=0
missing_count=0
not_yet_lifted=0
total=${#TGRAD_FIXTURES[@]}

for i in "${!TGRAD_FIXTURES[@]}"; do
  tgf="${REPO_ROOT}/${TGRAD_FIXTURES[$i]}"
  phf="${REPO_ROOT}/${PHASE_SOURCES[$i]}"
  if [[ ! -f "$tgf" ]]; then
    not_yet_lifted=$((not_yet_lifted + 1))
    [[ "$MODE" == "diff" ]] && echo "  — not yet lifted: ${TGRAD_FIXTURES[$i]}"
    continue
  fi
  if [[ ! -f "$phf" ]]; then
    missing_count=$((missing_count + 1))
    echo "  ✗ source phase fixture missing: ${PHASE_SOURCES[$i]}"
    continue
  fi
  if ! diff -q "$tgf" "$phf" >/dev/null 2>&1; then
    drift_count=$((drift_count + 1))
    echo "  ⚠ drift: ${TGRAD_FIXTURES[$i]}"
    echo "         vs ${PHASE_SOURCES[$i]}"
    if [[ "$MODE" == "diff" ]]; then
      diff "$tgf" "$phf" | sed 's/^/      /' | head -10
    fi
  else
    [[ "$MODE" == "diff" ]] && echo "  ✓ ${TGRAD_FIXTURES[$i]} matches phase source"
  fi
done

echo
echo "fixture-drift summary:"
echo "  total tracked:      $total"
echo "  lifted to Tgrad:    $((total - not_yet_lifted))"
echo "  in drift:           $drift_count"
echo "  source missing:     $missing_count"
echo "  not yet lifted:     $not_yet_lifted"

if [[ "$MODE" == "strict" && ($drift_count -gt 0 || $missing_count -gt 0) ]]; then
  echo
  echo "✗ --strict mode: exiting 1 due to $drift_count drift(s) + $missing_count missing source(s)"
  exit 1
fi
exit 0

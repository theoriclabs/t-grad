#!/usr/bin/env bash
# Tgrad canonical gate runner.
#
# Usage:
#   bash scripts/gate.sh           — run every currently-green gate
#                                          in order; fail on first red.
#                                          This is the regression sweep.
#   bash scripts/gate.sh L<n>      — run gate L<n> only.
#   bash scripts/gate.sh --single L<n>
#                                       — same as above, explicit single-gate
#                                          spelling for agents.
#   bash scripts/gate.sh --list    — list known gates + status.
#
# Each gate is a script under scripts/gates/L<n>.sh that:
#   1. sources scripts/lib/checks.sh (universal preflight)
#   2. verifies the gate's own predicates
#   3. writes evidence to fixtures/gate_evidence/L<n>.json
#
# Anti-shortcut design — see README.md. Key points:
#   - GREEN_GATES below is the ratchet. It only grows. Removing entries
#     trips check_no_gate_regression in checks.sh.
#   - Each gate's done predicate is the gate runner's verification of
#     the work, NOT the agent's report. The agent cannot fake a gate
#     by printing tagged lines.
#   - Universal preflight rejects `sorry`, `axiom`, `unsafe`, and
#     stale build artifacts.

set -euo pipefail

# --- ratchet ------------------------------------------------------------
# Append a gate here only after its scripts/gates/L<n>.sh passes for
# real. Removing an entry trips a regression check and fails the run.
#
# HISTORICAL ONE-SHOT CUT (P2, 2026-05-14): L6 was temporarily removed
# from this list during the pre-v1.0.0 ladder. P4 redid L6 as real FFI
# (@[export] + libtgrad.dylib + ctypes; subprocess explicitly forbidden)
# and re-added it on 2026-05-14. That was the ONLY authorized
# exception to the "ratchet only grows" rule; check_no_gate_regression's
# baseline is now (L0..L6) and further cuts are not permitted.
GREEN_GATES=(L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13_A L13_B L13_C L13_D L13_E L13 L13_F L13_F_STRICT_A L13_F_STRICT_B L13_F_STRICT_C L14_A L14_B_1 L14_B_2_a L14_B_2_b L14_B_2_c L14_B_2 L14_B_3 L14_B L14_C L14 L15_A L15_B L15_C L15)
# -----------------------------------------------------------------------

ALL_GATES=(L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13_A L13_B L13_C L13_D L13_E L13 L13_F L13_F_STRICT_A L13_F_STRICT_B L13_F_STRICT_C L14_A L14_B_1 L14_B_2_a L14_B_2_b L14_B_2_c L14_B_2 L14_B_3 L14_B L14_C L14 L15_A L15_B L15_C L15)
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TGRAD_DIR="$REPO_ROOT"
export TGRAD_BENCH_MODE=full
source "$TGRAD_DIR/scripts/lib/run_context.sh"
tgrad_run_context_init
cd "$REPO_ROOT"

run_gate() {
  local gate="$1"
  local script="$TGRAD_DIR/scripts/gates/$gate.sh"
  if [[ ! -f "$script" ]]; then
    echo "[$gate] not yet implemented — no script at $script"
    return 1
  fi
  bash "$script"
}

list_gates() {
  echo "Tgrad gates:"
  for g in "${ALL_GATES[@]}"; do
    local marker=" "
    for green in "${GREEN_GATES[@]}"; do
      [[ "$g" == "$green" ]] && marker="✓"
    done
    local script="$TGRAD_DIR/scripts/gates/$g.sh"
    local existence=""
    [[ -f "$script" ]] && existence=" (script: ✓)" || existence=" (script: —)"
    echo "  [$marker] $g$existence"
  done
  echo
  echo "Active (next-to-flip) gate: see README.md"
  echo
  echo "ratchet — GREEN_GATES = ${GREEN_GATES[*]}"
  echo "         entries can only be added (after the gate passes), never removed."
}

ratchet_check() {
  # Source lib/checks.sh so we can call check_no_gate_regression directly.
  # This runs EVEN WHEN GREEN_GATES is empty — previously the per-gate
  # for-loop would silently skip the regression check on an emptied
  # ratchet (the exact sabotage `L0_falsifiability.md` claims to catch).
  source "$TGRAD_DIR/scripts/lib/checks.sh"
  if [[ "${#GREEN_GATES[@]}" -eq 0 ]]; then
    echo "✗ ratchet check: GREEN_GATES is empty — at minimum L0 should be present"
    echo "  (the ratchet only grows; removing entries is a regression)"
    return 1
  fi
  check_no_gate_regression || return 1
}

main() {
  case "${1:-}" in
    --list|list) list_gates; return 0 ;;
    "")
      ratchet_check || { echo "✗ ratchet regression detected — aborting"; return 1; }
      run_global_preflight || { echo "✗ global preflight failed — aborting"; return 1; }
      echo "═══ Tgrad regression sweep — running ${#GREEN_GATES[@]} green gate(s) ═══"
      for g in "${GREEN_GATES[@]}"; do
        echo ""
        echo "──────── $g ────────"
        run_gate "$g" || { echo ""; echo "✗ $g RED — aborting"; return 1; }
      done
      echo ""
      echo "✓ all ${#GREEN_GATES[@]} green gate(s) still green"
      return 0
      ;;
    --single)
      if [[ -z "${2:-}" ]]; then
        echo "missing gate after --single"
        return 2
      fi
      ratchet_check || { echo "✗ ratchet regression detected — aborting"; return 1; }
      run_global_preflight || { echo "✗ global preflight failed — aborting"; return 1; }
      echo "──────── $2 (single-gate run) ────────"
      run_gate "$2"
      ;;
    L0|L1|L2|L3|L4|L5|L6|L7|L8|L9|L10|L11|L12|L13|L14|L15|\
    L13_A|L13_B|L13_C|L13_D|L13_E|L13_F|L13_F_STRICT_A|L13_F_STRICT_B|L13_F_STRICT_C|L14_A|L14_B_1|L14_B_2_a|L14_B_2_b|L14_B_2_c|L14_B_2|L14_B_3|L14_B|L14_C|L14|L15_A|L15_B|L15_C|L15)
      ratchet_check || { echo "✗ ratchet regression detected — aborting"; return 1; }
      run_global_preflight || { echo "✗ global preflight failed — aborting"; return 1; }
      echo "──────── $1 (single-gate run) ────────"
      run_gate "$1"
      ;;
    *)
      echo "unknown gate: $1"
      echo "valid: L0..L15 + sub-gates L13_A..L13_F, L13_F_STRICT_A..B, L14_A..L14_C, L15_A..L15_C  (or --list)"
      return 2
      ;;
  esac
}

main "$@"

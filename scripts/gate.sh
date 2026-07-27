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
#   bash scripts/gate.sh --regenerate-evidence /absolute/candidate/path \
#     /absolute/prepared-runtime-certificate.json
#                                      — run all gates serially, continue after
#                                         red gates, and retain an attributable
#                                         candidate outside the repository.
#                                         This never edits committed evidence.
#   bash scripts/gate.sh --verify-evidence
#                                      — strictly audit the committed snapshot.
#
# Each gate is a script under scripts/gates/L<n>.sh that:
#   1. sources scripts/lib/checks.sh (universal preflight)
#   2. verifies the gate's own predicates
#   3. writes evidence to the inherited run-owned TGRAD_EVIDENCE_DIR.
# Publication to fixtures/gate_evidence is a separate complete-green action.
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
  case "$gate" in
    L0|L1|L2|L3|L4|L5|L6|L7|L8|L9|L10|L11|L12|L13|L14|L15|\
    L13_A|L13_B|L13_C|L13_D|L13_E|L13_F|L13_F_STRICT_A|L13_F_STRICT_B|L13_F_STRICT_C|L14_A|L14_B_1|L14_B_2_a|L14_B_2_b|L14_B_2_c|L14_B_2|L14_B_3|L14_B|L14_C|L15_A|L15_B|L15_C) ;;
    *) echo "unknown gate: $gate"; return 2 ;;
  esac
  local script="$TGRAD_DIR/scripts/gates/$gate.sh"
  if [[ ! -f "$script" ]]; then
    echo "[$gate] not yet implemented — no script at $script"
    return 1
  fi
  bash "$script"
}

record_candidate_outcome() {
  local candidate="$1" gate="$2" status="$3" returncode="$4" log="$5"
  local blocked_by="${6:-}"
  local args=(
    record
    --candidate "$candidate"
    --gate "$gate"
    --status "$status"
    --returncode "$returncode"
    --log "$log"
  )
  if [[ -n "$blocked_by" ]]; then
    args+=(--blocked-by "$blocked_by")
  fi
  python3 "$TGRAD_DIR/scripts/evidence/candidate.py" "${args[@]}"
}

regenerate_evidence() {
  local candidate="${1:-}"
  local performance_certificate="${2:-${TGRAD_PERFORMANCE_CERTIFICATE:-}}"
  if [[ -z "$candidate" ]]; then
    echo "missing absolute candidate path after --regenerate-evidence"
    return 2
  fi
  case "$candidate" in
    /*) ;;
    *) echo "candidate path must be absolute: $candidate"; return 2 ;;
  esac
  local candidate_parent candidate_name
  candidate_parent="${candidate%/*}"
  candidate_name="${candidate##*/}"
  [[ -n "$candidate_parent" && -n "$candidate_name" ]] || {
    echo "candidate path must name a child of an existing directory: $candidate"
    return 2
  }
  candidate_parent="$(cd -P -- "$candidate_parent" 2>/dev/null && pwd -P)" || {
    echo "candidate parent does not exist: ${candidate%/*}"
    return 2
  }
  candidate="$candidate_parent/$candidate_name"

  # The run root is part of the retained artifact closure.  Gate scripts may
  # reuse filenames, so candidate.py snapshots each version by content hash.
  export TGRAD_KEEP_RUN_DIR=1
  export TGRAD_EVIDENCE_DIR="$candidate/evidence"
  local init_args=(
    init
    --candidate "$candidate"
    --run-root "$TGRAD_RUN_DIR"
    --gates "${ALL_GATES[@]}"
    --green-gates "${GREEN_GATES[@]}"
  )
  if [[ -n "$performance_certificate" ]]; then
    init_args+=(--performance-certificate "$performance_certificate")
  fi
  python3 "$TGRAD_DIR/scripts/evidence/candidate.py" "${init_args[@]}"
  tgrad_run_context_init

  local preflight_log="$candidate/logs/PREFLIGHT.log"
  local preflight_rc
  python3 "$TGRAD_DIR/scripts/evidence/candidate.py" begin \
    --candidate "$candidate" --producer PREFLIGHT
  set +e
  {
    ratchet_check && run_global_preflight
  } >"$preflight_log" 2>&1
  preflight_rc=$?
  set -e
  python3 "$TGRAD_DIR/scripts/evidence/candidate.py" capture \
    --candidate "$candidate" --producer PREFLIGHT
  cat "$preflight_log"

  if [[ "$preflight_rc" -ne 0 ]]; then
    echo "✗ preflight RED — recording every gate as blocked"
    local blocked_gate
    for blocked_gate in "${ALL_GATES[@]}"; do
      record_candidate_outcome \
        "$candidate" "$blocked_gate" blocked "$preflight_rc" "$preflight_log" \
        "PREFLIGHT:red"
    done
  else
    echo "═══ attributable evidence sweep — ${#ALL_GATES[@]} serial gate(s) ═══"
    local gate log rc status
    for gate in "${ALL_GATES[@]}"; do
      log="$candidate/logs/$gate.log"
      echo ""
      echo "──────── $gate ────────"
      local blockers
      blockers="$(python3 "$TGRAD_DIR/scripts/evidence/candidate.py" blockers \
        --candidate "$candidate" --gate "$gate")"
      if [[ -n "$blockers" ]]; then
        {
          echo "$gate blocked by reviewed release dependencies:"
          echo "$blockers"
        } >"$log"
        cat "$log"
        record_candidate_outcome "$candidate" "$gate" blocked 125 "$log" "$blockers"
        continue
      fi
      python3 "$TGRAD_DIR/scripts/evidence/candidate.py" begin \
        --candidate "$candidate" --producer "$gate"
      set +e
      run_gate "$gate" >"$log" 2>&1
      rc=$?
      set -e
      if [[ "$rc" -eq 0 ]]; then status=pass; else status=red; fi
      cat "$log"
      record_candidate_outcome "$candidate" "$gate" "$status" "$rc" "$log"
    done
  fi

  # Finalize is intentionally nonzero for a retained red candidate.  The
  # caller gets diagnostics and a complete manifest, never a partial publish.
  local finalize_rc
  set +e
  python3 "$TGRAD_DIR/scripts/evidence/candidate.py" finalize --candidate "$candidate"
  finalize_rc=$?
  set -e
  if [[ "$finalize_rc" -ne 0 ]]; then
    echo "✗ candidate retained but is not promotable: $candidate"
    return "$finalize_rc"
  fi
  echo "✓ complete-green candidate retained at $candidate"
  echo "  promotion remains an explicit reviewed operation"
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
    --verify-evidence)
      python3 "$TGRAD_DIR/scripts/dev/evidence_provenance_audit.py" \
        --strict --evidence-dir "$TGRAD_DIR/fixtures/gate_evidence"
      ;;
    --regenerate-evidence)
      regenerate_evidence "${2:-}" "${3:-}"
      ;;
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

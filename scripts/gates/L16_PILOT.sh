#!/usr/bin/env bash
# Gate L16_PILOT — re-observe the pilot helper-import evidence.
#
# --check-generated only proves committed Lean matches committed JSON.
# This gate also runs the full --check (live re-observation vs committed
# evidence). It is intentionally NOT in GREEN_GATES while the product /
# environment has drifted past the promoted observation at b1df552: a
# green claim would require a separate promotion judgement.
#
# Predicates:
#   A  universal preflight
#   B  required observer + fixture + generated Lean present
#   D  gate script wires repo-venv python + both check modes (anti-cheat)
#   C  --check-generated AND full --check against committed evidence
#   E  evidence (only reached when --check is green)
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

GATE_NAME="L16_PILOT"
echo "[$GATE_NAME] pilot helper-import re-observation (--check-generated + --check)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight
cd "$REPO_ROOT"

OBSERVER="$TGRAD_DIR/scripts/spec/observe_pilot.py"
EVIDENCE_JSON="$TGRAD_DIR/fixtures/requirements/pilot_helpers_b1df552.json"
EVIDENCE_LEAN="$TGRAD_DIR/Tgrad/Evidence/PilotGenerated.lean"
GATE_SCRIPT="$TGRAD_DIR/scripts/gates/${GATE_NAME}.sh"

# ─── LAYER B: structural ──────────────────────────────────────────────
required_modules=(
  scripts/spec/observe_pilot.py
  fixtures/requirements/pilot_helpers_b1df552.json
  Tgrad/Evidence/PilotGenerated.lean
  python/tgrad.py
  scripts/parity/shim/tinygrad/__init__.py
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# ─── LAYER D: anti-cheat — this gate must wire both checks + venv py ──
# Before resolving the interpreter so a sabotaged pin is caught here
# rather than by a confusing path-not-found later. Grep executable
# lines only (ignore comments) so doc chatter cannot satisfy the predicate.
# Pin needle is split so a bulk replace of ".venv/bin/python" cannot
# rewrite this check into a tautology.
code_lines() { grep -E '^[[:space:]]*[^#[:space:]]' "$GATE_SCRIPT" || true; }
if ! code_lines | grep -qE -- '--check-generated'; then
  echo "  ✗ ${GATE_NAME}.sh does not invoke --check-generated"
  exit 1
fi
# Require a full --check invocation that is not only --check-generated.
if ! code_lines | grep -E -- '--check\b' | grep -v -- '--check-generated' >/dev/null; then
  echo "  ✗ ${GATE_NAME}.sh does not invoke full --check (only --check-generated is insufficient)"
  exit 1
fi
_pin_a='.venv/bin'
_pin_b='/python'
if ! code_lines | grep -qF "${_pin_a}${_pin_b}"; then
  echo "  ✗ ${GATE_NAME}.sh does not pin ${_pin_a}${_pin_b}"
  exit 1
fi
echo "  ✓ gate script wires ${_pin_a}${_pin_b} + --check-generated + --check"

# Repo venv python EXPLICITLY — never bare `python3` on PATH.
PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "  ✗ repo venv interpreter missing or not executable: $PY"
  echo "      L16_PILOT requires .venv/bin/python (do not fall back to PATH python3)"
  exit 1
fi
echo "  ✓ using repo venv interpreter: $PY"

# Runtime artifact required by the live observer.
ensure_dylib /tmp/tgrad_${GATE_NAME}_dylib.log || exit 1
echo "  ✓ libtgrad.dylib present for live observation"

# ─── LAYER C: behavioural ─────────────────────────────────────────────
GEN_LOG="/tmp/tgrad_${GATE_NAME}_check_generated.log"
if ! "$PY" "$OBSERVER" --python "$PY" --check-generated \
     --output "$EVIDENCE_JSON" --lean-output "$EVIDENCE_LEAN" \
     >"$GEN_LOG" 2>&1; then
  echo "  ✗ observe_pilot.py --check-generated failed"
  sed 's/^/      /' "$GEN_LOG"
  exit 1
fi
echo "  ✓ --check-generated: committed Lean matches committed JSON"
sed 's/^/      /' "$GEN_LOG"

CHECK_LOG="/tmp/tgrad_${GATE_NAME}_check.log"
set +e
"$PY" "$OBSERVER" --python "$PY" --check \
  --output "$EVIDENCE_JSON" --lean-output "$EVIDENCE_LEAN" \
  >"$CHECK_LOG" 2>&1
CHECK_RC=$?
set -e
if [[ "$CHECK_RC" -ne 0 ]]; then
  echo "  ✗ observe_pilot.py --check failed (exit $CHECK_RC)"
  echo "      Live re-observation does not match committed evidence."
  echo "      This is an honest red: do not regenerate evidence from this gate."
  echo "      Promote a fresh observation only via an explicit human judgement."
  sed 's/^/      /' "$CHECK_LOG"
  exit 1
fi
echo "  ✓ --check: live observation matches committed evidence"
sed 's/^/      /' "$CHECK_LOG"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
observer_hash="$(shasum -a 256 "$OBSERVER" | awk '{print $1}')"
json_hash="$(shasum -a 256 "$EVIDENCE_JSON" | awk '{print $1}')"
lean_hash="$(shasum -a 256 "$EVIDENCE_LEAN" | awk '{print $1}')"
check_hash="$(shasum -a 256 "$CHECK_LOG" | awk '{print $1}')"

mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/${GATE_NAME}.json" <<EOF
{
  "gate": "$GATE_NAME",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L16_PILOT — live re-observation of pilot helper-import evidence",
  "hashes": {
    "observe_pilot_sha256": "$observer_hash",
    "pilot_helpers_json_sha256": "$json_hash",
    "pilot_generated_lean_sha256": "$lean_hash",
    "check_log_sha256": "$check_hash"
  }
}
EOF

check_evidence_for "$GATE_NAME" || exit 1
check_falsifiability_verified "$GATE_NAME" || exit 1
echo "  ✓ $GATE_NAME pilot re-observation gate green (evidence recorded)"

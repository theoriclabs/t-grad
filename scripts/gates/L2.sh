#!/usr/bin/env bash
# Gate L2 — schedule layer (rangeify + memory planning + scheduler).
#
# Verifies the lifts from theograd_phases/04 (rangeify), 05 (memory
# planning), and 06 (scheduler) into Tgrad's unified namespace.
# Behavioural checks: tgrad-cli rangeify / mem-plan / schedule each
# byte-diff vs a captured fixture from the corresponding source phase.
#
# Anti-shortcut design: each behavioural predicate cross-validates
# against captured tinygrad output; the agent can't fake the matches
# because the fixtures came from running tinygrad upstream.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L2_RANGEIFY="$(tgrad_run_path L2_rangeify.json)"
L2_MEM="$(tgrad_run_path L2_mem.json)"
L2_SCHED="$(tgrad_run_path L2_sched.json)"
L2_NEG_LEAN="$(tgrad_run_path L2_negative.lean)"
L2_NEG_LOG="$(tgrad_run_path L2_negative.log)"

echo "[L2] schedule layer"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/Schedule/Rangeify.lean
  Tgrad/Schedule/Indexing.lean
  Tgrad/Schedule/Memory.lean
  Tgrad/Schedule/Linear.lean
  Tgrad/Schedule/Item.lean
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

required_theorems=(
  "Tgrad/Schedule/Item.lean:allSinksHaveName"
)
for entry in "${required_theorems[@]}"; do
  file="${entry%:*}"; thm="${entry##*:}"
  if ! grep -qE "^theorem[[:space:]]+$thm\b" "$REPO_ROOT/$file"; then
    echo "  ✗ missing theorem: $thm in $file"
    exit 1
  fi
done
echo "  ✓ all ${#required_theorems[@]} required theorems declared (preflight rejected sorry/axiom)"

required_fixtures=(
  fixtures/schedule/rangeify_input.json
  fixtures/schedule/rangeify_expected.json
  fixtures/schedule/intervals.json
  fixtures/schedule/assignment_expected.json
  fixtures/schedule/detailed_schedule.json
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing required fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} required fixtures present"

# ─── LAYER C: behavioural cross-validation ────────────────────────────

# Sub-predicate 4a: rangeify — apply the captured RESHAPE+PERMUTE chain
# backward; emit the LOAD-side index UOp records; compare byte-for-byte.
./.lake/build/bin/tgrad-cli rangeify "$TGRAD_DIR/fixtures/schedule/rangeify_input.json" \
    >"$L2_RANGEIFY" 2>&1 || {
  echo "  ✗ tgrad-cli rangeify failed"; cat "$L2_RANGEIFY"; exit 1
}
if ! diff -q "$L2_RANGEIFY" "$TGRAD_DIR/fixtures/schedule/rangeify_expected.json" >/dev/null; then
  echo "  ✗ Tgrad.Schedule.Rangeify disagrees with captured chain output"
  diff "$L2_RANGEIFY" "$TGRAD_DIR/fixtures/schedule/rangeify_expected.json" | head -20
  exit 1
fi
echo "  ✓ Tgrad.Schedule.Rangeify matches captured 2-op chain output"

# Sub-predicate 4b: memory planning — greedy interval coloring on the
# 5-buffer fixture; assignment table must byte-match.
./.lake/build/bin/tgrad-cli mem-plan "$TGRAD_DIR/fixtures/schedule/intervals.json" \
    >"$L2_MEM" 2>&1 || {
  echo "  ✗ tgrad-cli mem-plan failed"; cat "$L2_MEM"; exit 1
}
if ! diff -q "$L2_MEM" "$TGRAD_DIR/fixtures/schedule/assignment_expected.json" >/dev/null; then
  echo "  ✗ Tgrad.Schedule.Memory disagrees with captured assignment"
  diff "$L2_MEM" "$TGRAD_DIR/fixtures/schedule/assignment_expected.json" | head -20
  exit 1
fi
echo "  ✓ Tgrad.Schedule.Memory matches captured 5-buffer assignment"

# Sub-predicate 4c: scheduler — DetailedSchedule round-trip (read +
# re-emit). The fixture is both the input and the expected output —
# this verifies the typed DetailedSchedule data model + JSON emit
# preserves shape byte-for-byte.
./.lake/build/bin/tgrad-cli schedule "$TGRAD_DIR/fixtures/schedule/detailed_schedule.json" \
    >"$L2_SCHED" 2>&1 || {
  echo "  ✗ tgrad-cli schedule failed"; cat "$L2_SCHED"; exit 1
}
if ! diff -q "$L2_SCHED" "$TGRAD_DIR/fixtures/schedule/detailed_schedule.json" >/dev/null; then
  echo "  ✗ Tgrad.Schedule.Linear disagrees with captured detailed schedule"
  diff "$L2_SCHED" "$TGRAD_DIR/fixtures/schedule/detailed_schedule.json" | head -20
  exit 1
fi
echo "  ✓ Tgrad.Schedule.Linear matches captured detailed schedule"

# ─── LAYER D: negative tests ──────────────────────────────────────────
# The typed ScheduleItem sum forbids constructing a `.sink` without a
# function name. This snippet attempts exactly that — must fail to compile.
cat >"$L2_NEG_LEAN" <<'EOF'
import Tgrad
open Tgrad
-- SinkItem requires `functionName : String`. A SinkItem without one
-- is a type error; constructing .sink with a CopyItem-shaped payload
-- (no functionName) is also a type error.
def bad : ScheduleItem := .sink { bufferCount := 4, totalSrcCount := 4 }
EOF
if (cd "$TGRAD_DIR" && lake env lean "$L2_NEG_LEAN") >"$L2_NEG_LOG" 2>&1; then
  echo "  ✗ negative test compiled — type system isn't enforcing SinkItem.functionName"
  cat "$L2_NEG_LOG"
  exit 1
fi
echo "  ✓ negative test correctly rejected (SinkItem requires functionName)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
rangeify_hash="$(shasum -a 256 "$L2_RANGEIFY" | awk '{print $1}')"
mem_hash="$(shasum -a 256 "$L2_MEM" | awk '{print $1}')"
sched_hash="$(shasum -a 256 "$L2_SCHED" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L2.json" <<EOF
{
  "gate": "L2",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "hashes": {
    "rangeify_sha256":  "$rangeify_hash",
    "mem_plan_sha256":  "$mem_hash",
    "schedule_sha256":  "$sched_hash"
  }
}
EOF
check_evidence_for L2 || exit 1
check_falsifiability_verified L2 || exit 1
echo "  ✓ L2 schedule-layer gate green (evidence recorded)"

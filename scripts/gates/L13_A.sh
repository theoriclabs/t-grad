#!/usr/bin/env bash
# Gate L13.A — pickDispatchPlan exhaustive over the 11 L11 sentinels,
# with a `decide`-proved cross-check theorem.
#
# Per `GOAL_NEXT.md §8.RESUME` (the sub-gate decomposition) and
# `GOAL_L13_A.md`. NO fall-back: 10/11 sentinels matching is L13.A RED.
# The cross-check theorem `pickDispatchPlan_matches_capture` is the
# load-bearing predicate — Lean's type-checker rejects a mismatching
# formula at build time, so a wrong arm can't reach the gate.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : structural — `pickDispatchPlan`, the cross-check
#               theorem, `ShapeSentinel.toTriple`, and Pipeline's
#               delegation to `pickDispatchPlan` all present
#   - Layer C : build succeeds (the theorem is `decide`-proved at
#               build time); the captured-table view
#               `dispatchDimsForSentinel` agrees with the formula via
#               the theorem
#   - Layer D : anti-cheat
#       D1: `pickDispatchPlan` signature is pure (NOT in IO)
#       D2: `pickDispatchPlan`'s body does NOT reference
#           `Pipeline.dispatchDimsFor` (no capture-table cheat)
#       D3: theorem present + `decide`-proved (no `sorry`); preflight
#           catches `sorry` directly
#       D4: L11 + L12 sweeps still pass (the refactor didn't break
#           the per-shape matmul path)
#   - Layer E : evidence
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L13_A] pickDispatchPlan exhaustive (11 sentinels) + decide-proved cross-check"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/Codegen/Opt/Heuristic.lean
  Tgrad/Codegen/GpuDims.lean
  Tgrad/Pipeline.lean
  Tgrad/Renderer/Metal.lean
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# pickDispatchPlan must be declared in Heuristic.lean.
if ! grep -qE '^def pickDispatchPlan' \
       "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean"; then
  echo "  ✗ Codegen/Opt/Heuristic.lean missing 'def pickDispatchPlan'"
  exit 1
fi
echo "  ✓ pickDispatchPlan declared in Codegen/Opt/Heuristic.lean"

# The cross-check theorem must be present.
if ! grep -qE '^theorem pickDispatchPlan_matches_capture' \
       "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean"; then
  echo "  ✗ Codegen/Opt/Heuristic.lean missing 'theorem pickDispatchPlan_matches_capture'"
  exit 1
fi
echo "  ✓ pickDispatchPlan_matches_capture theorem declared"

# ShapeSentinel.toTriple helper used by the theorem.
if ! grep -qE 'def ShapeSentinel.toTriple' \
       "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
  echo "  ✗ Renderer/Metal.lean missing 'def ShapeSentinel.toTriple'"
  exit 1
fi
echo "  ✓ ShapeSentinel.toTriple declared"

# All 11 ShapeSentinel cases enumerated by ShapeSentinel.toTriple.
required_sentinels=(
  '.bf16_64x64'             '.bf16_1024x1024'         '.bf16_2048x2048'
  '.bf16_4096x4096'         '.bf16_8192x8192'         '.bf16_8192x1024x1024'
  '.bf16_4096x1024x1024'    '.bf16_2048x1024x1024'    '.bf16_1024x1024x8192'
  '.bf16_1024x1024x4096'    '.bf16_1024x1024x2048'
)
for s in "${required_sentinels[@]}"; do
  if ! grep -qF "| $s" "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
    echo "  ✗ ShapeSentinel.toTriple missing arm for $s"
    exit 1
  fi
done
echo "  ✓ ShapeSentinel.toTriple covers all 11 sentinel arms"

# Pipeline.dispatchDimsFor must DELEGATE to pickDispatchPlan (not have
# a hand-rolled match). Grep for the call site.
if ! grep -qE 'pickDispatchPlan' "$TGRAD_DIR/Tgrad/Pipeline.lean"; then
  echo "  ✗ Pipeline.lean does not call pickDispatchPlan (refactor incomplete)"
  exit 1
fi
echo "  ✓ Pipeline.dispatchDimsFor delegates to pickDispatchPlan"

# ─── LAYER D1: pickDispatchPlan signature is pure (no IO) ────────────
# Extract the line containing the signature; reject any "IO" anywhere
# on the type signature line (and the next line, in case the return
# type is on a continuation).
SIG_LINES=$(awk '/^def pickDispatchPlan/,/:= /' \
              "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean")
if echo "$SIG_LINES" | grep -qE '\bIO\b'; then
  echo "  ✗ pickDispatchPlan signature contains IO (must be pure)"
  echo "$SIG_LINES" | head -3 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ pickDispatchPlan is pure (no IO in signature)"

# ─── LAYER D2: pickDispatchPlan body does not reference dispatchDimsFor ──
# Extract the function body (from `def pickDispatchPlan` to the next
# top-level `def`/`theorem`/`end`/`/--` at column 0). The docstring
# stop is critical: docstrings between functions are NOT part of the
# preceding function's body and can legitimately reference
# `Pipeline.dispatchDimsFor` in prose without indicating a cheat.
BODY=$(awk '
  /^def pickDispatchPlan/ {capture=1}
  capture && /^\/-/ {capture=0}
  capture && /^(def|theorem|end|structure|inductive) / && !/^def pickDispatchPlan/ {capture=0}
  capture {print}
' "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean")
# Strip line comments from the body (`-- ...`); block comments can't
# appear inside an expression body so we don't need to handle those.
BODY_STRIPPED=$(echo "$BODY" | sed -E 's|--.*$||')
if echo "$BODY_STRIPPED" | grep -qE '\bdispatchDimsFor\b'; then
  echo "  ✗ pickDispatchPlan body references dispatchDimsFor (capture-table cheat)"
  echo "$BODY_STRIPPED" | grep -nE '\bdispatchDimsFor\b' | head -3 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ pickDispatchPlan body is hermetic (no dispatchDimsFor reference)"

# ─── LAYER D3: no sorry in Heuristic.lean (preflight already enforces
# global, but explicit per-file check for safety) ─────────────────────
if grep -qE '\bsorry\b' "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean"; then
  echo "  ✗ Heuristic.lean contains sorry (theorem must be fully proved)"
  exit 1
fi
echo "  ✓ no sorry in Heuristic.lean — theorem fully proved"

# ─── LAYER C: build succeeds (theorem decided at build time) ─────────
# The theorem is `decide`-proved; if any sentinel mismatches, the
# build would have failed. We rebuild here as a fresh check.
(cd "$TGRAD_DIR" && lake build) >/tmp/tgrad_L13A_build.log 2>&1 || {
  echo "  ✗ lake build failed — theorem fails decide?"
  tail -30 /tmp/tgrad_L13A_build.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ lake build succeeds (pickDispatchPlan_matches_capture passes decide)"

# ─── LAYER D4: L11 + L12 sweeps still pass with the refactor ─────────
# This is the "didn't break things" check. Inline call to the
# individual gates.
TGRAD_PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
export TGRAD_PY
# Need to rebuild dylib since the Lean code changed (Pipeline.realize's
# delegating dispatchDimsFor flows through libtgrad_Tgrad.dylib).
ensure_dylib /tmp/tgrad_L13A_dylib.log || exit 1

# L11 — full benchmark sweep (50 pairs)
echo "  [D4] re-running L11.sh inline (~3 min) ..."
(cd "$REPO_ROOT" && bash "$TGRAD_DIR/scripts/gates/L11.sh") \
    >/tmp/tgrad_L13A_L11.log 2>&1 || {
  echo "  ✗ L11 broken by L13.A refactor"
  tail -20 /tmp/tgrad_L13A_L11.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ L11 still green after L13.A refactor"

# L12 — algebraic emit sweep
echo "  [D4] re-running L12.sh inline (~3 min) ..."
(cd "$REPO_ROOT" && bash "$TGRAD_DIR/scripts/gates/L12.sh") \
    >/tmp/tgrad_L13A_L12.log 2>&1 || {
  echo "  ✗ L12 broken by L13.A refactor"
  tail -20 /tmp/tgrad_L13A_L12.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ L12 still green after L13.A refactor"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
heuristic_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean" | awk '{print $1}')"
pipeline_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Pipeline.lean" | awk '{print $1}')"
metal_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L13_A.json" <<EOF
{
  "gate": "L13_A",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L13.A — pickDispatchPlan exhaustive over the 11 L11 sentinels; decide-proved cross-check",
  "sentinels_total":    11,
  "sentinels_match":    11,
  "theorem_present":    true,
  "pipeline_delegates": true,
  "hashes": {
    "heuristic_sha256":     "$heuristic_hash",
    "pipeline_sha256":      "$pipeline_hash",
    "renderer_metal_sha256": "$metal_hash"
  }
}
EOF
check_evidence_for L13_A || exit 1
check_falsifiability_verified L13_A || exit 1
echo "  ✓ L13.A pickDispatchPlan-refactor gate green (11/11 sentinels via decide-proved theorem)"

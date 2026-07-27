#!/usr/bin/env bash
# Gate L3 — codegen + renderer.
#
# Verifies the lifts from theograd_phases/{01, 07, 08, 09, 10} into
# Tgrad's unified namespace. L3 stays at capture-and-replay for the
# matmul MSL (algebraic emit is L8 per the brief).
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L3_LIN="$(tgrad_run_path L3_linearize.json)"
L3_TC="$(tgrad_run_path L3_tc.txt)"
L3_TC_NEG="$(tgrad_run_path L3_tc_negative.txt)"
L3_MSL="$(tgrad_run_path L3_matmul.msl)"

echo "[L3] codegen + renderer"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/Codegen/Linearize.lean
  Tgrad/Codegen/Simplify.lean
  Tgrad/Codegen/GpuDims.lean
  Tgrad/Codegen/Opt/Apply.lean
  Tgrad/Codegen/Opt/Heuristic.lean
  Tgrad/Codegen/Opt/Tc.lean
  Tgrad/Codegen/Opt/IsTcEligible.lean
  Tgrad/Renderer/Base.lean
  Tgrad/Renderer/CStyle.lean
  Tgrad/Renderer/Metal.lean
  Tgrad/Renderer/WmmaArgs.lean
  Tgrad/Renderer/CodeForOp.lean
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

required_theorems=(
  "Tgrad/Codegen/Opt/IsTcEligible.lean:tc_eligible_64x64_bf16_f32"
  "Tgrad/Codegen/Opt/IsTcEligible.lean:tc_ineligible_4x4_bf16_f32"
  "Tgrad/Codegen/Opt/IsTcEligible.lean:tc_ineligible_int32"
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
  fixtures/codegen/linearize_in.json
  fixtures/codegen/linearize_expected.json
  fixtures/codegen/matmul_64x64.msl
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing required fixture: $f"; exit 1; }
done
echo "  ✓ all ${#required_fixtures[@]} required fixtures present"

# ─── LAYER C: behavioural cross-validation ────────────────────────────

# Sub-predicate 4a: linearize — DFS post-order toposort on phase-07's
# 8-UOp sub-DAG.
./.lake/build/bin/tgrad-cli linearize "$TGRAD_DIR/fixtures/codegen/linearize_in.json" \
    >"$L3_LIN" 2>&1 || {
  echo "  ✗ tgrad-cli linearize failed"; cat "$L3_LIN"; exit 1
}
if ! diff -q "$L3_LIN" "$TGRAD_DIR/fixtures/codegen/linearize_expected.json" >/dev/null; then
  echo "  ✗ Tgrad.Codegen.Linearize disagrees with captured output"
  diff "$L3_LIN" "$TGRAD_DIR/fixtures/codegen/linearize_expected.json" | head -20
  exit 1
fi
echo "  ✓ Tgrad.Codegen.Linearize matches captured post-order"

# Sub-predicate 4b: apply-opt-tc — captured TC opt rewrite returns the
# pinned sha (0x820a2f5e) for the bf16 64x64 sentinel; `null` for any
# other input.
./.lake/build/bin/tgrad-cli apply-opt-tc bf16_64x64 \
    >"$L3_TC" 2>&1 || {
  echo "  ✗ tgrad-cli apply-opt-tc bf16_64x64 failed"; cat "$L3_TC"; exit 1
}
expected_tc="0x820a2f5e"
if ! grep -q "$expected_tc" "$L3_TC"; then
  echo "  ✗ apply-opt-tc bf16_64x64 didn't return expected sha $expected_tc"
  cat "$L3_TC"; exit 1
fi
echo "  ✓ Tgrad.Codegen.Opt.Apply returns captured sha for bf16 64x64"

./.lake/build/bin/tgrad-cli apply-opt-tc fp32_4x4 \
    >"$L3_TC_NEG" 2>&1 || {
  echo "  ✗ tgrad-cli apply-opt-tc fp32_4x4 failed"; cat "$L3_TC_NEG"; exit 1
}
if ! grep -q "null" "$L3_TC_NEG"; then
  echo "  ✗ apply-opt-tc fp32_4x4 should return null (unsupported shape/dtype)"
  cat "$L3_TC_NEG"; exit 1
fi
echo "  ✓ Tgrad.Codegen.Opt.Apply returns null for non-captured input"

# Sub-predicate 4c: render-metal — capture lookup for bf16 64×64.
# renderMetal resolves fixture paths from REPO_ROOT, so cd there.
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" render-metal bf16_64x64) \
    >"$L3_MSL" 2>&1 || {
  echo "  ✗ tgrad-cli render-metal failed"; cat "$L3_MSL"; exit 1
}
if ! diff -q "$L3_MSL" "$TGRAD_DIR/fixtures/codegen/matmul_64x64.msl" >/dev/null; then
  echo "  ✗ Tgrad.Renderer.Metal output disagrees with captured 64×64 MSL"
  diff "$L3_MSL" "$TGRAD_DIR/fixtures/codegen/matmul_64x64.msl" | head -20
  exit 1
fi
echo "  ✓ Tgrad.Renderer.Metal returns captured 64×64 MSL byte-equal"

# ─── LAYER D: negative test ───────────────────────────────────────────
# Asking render-metal for a non-captured shape must return a non-zero
# exit code rather than emitting an unspecified MSL.
if (cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" render-metal completely_unknown_shape) >/dev/null 2>&1; then
  echo "  ✗ render-metal for unknown shape returned 0 — should fail"
  exit 1
fi
echo "  ✓ negative test correctly rejected (render-metal exits nonzero for unknown shapes)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
lin_hash="$(shasum -a 256 "$L3_LIN" | awk '{print $1}')"
tc_hash="$(shasum -a 256 "$L3_TC" | awk '{print $1}')"
msl_hash="$(shasum -a 256 "$L3_MSL" | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L3.json" <<EOF
{
  "gate": "L3",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "hashes": {
    "linearize_sha256":  "$lin_hash",
    "apply_opt_tc_sha256":  "$tc_hash",
    "render_metal_sha256":  "$msl_hash"
  }
}
EOF
check_evidence_for L3 || exit 1
check_falsifiability_verified L3 || exit 1
echo "  ✓ L3 codegen-renderer gate green (evidence recorded)"

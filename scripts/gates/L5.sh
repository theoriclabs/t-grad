#!/usr/bin/env bash
# Gate L5 — end-to-end pipeline composition + numerical correctness.
#
# L5.a (original): Pipeline.realize composes L2+L3+L4 into a single Lean
# function that exercises the captured 64×64 bf16 matmul kernel end-to-
# end on Metal (compile + alloc + dispatch with rc=0).
#
# L5.b (this gate's strengthened layer per P3 / §G_L5_b): adds a real
# numerical-correctness predicate. tgrad-cli matmul-verify writes the
# captured tinygrad input bytes into Metal buffers, dispatches
# Pipeline.realize, reads the output back, and byte-compares against
# the captured tinygrad output (3 × 8192-byte .bin fixtures, seed=42).
# Per GOAL_NEXT.md §6 rule 1, scope-narrow ≠ correctness-narrow:
# this single shape MUST byte-match, not just run.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L5_MATMUL="$(tgrad_run_path L5_matmul.txt)"
L5_VERIFY="$(tgrad_run_path L5_verify.txt)"

echo "[L5] pipeline composition"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/Tensor.lean
  Tgrad/Pipeline.lean
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# Verify Pipeline.realize is a real function (not a sentinel chain).
if ! grep -qE "^def Pipeline\.realize\b|^def realize\b" "$TGRAD_DIR/Tgrad/Pipeline.lean"; then
  echo "  ✗ Tgrad/Pipeline.lean: no real `Pipeline.realize` definition"
  exit 1
fi
echo "  ✓ Tgrad.Pipeline.realize is a real function (composes the stages)"

# ─── LAYER C: behavioural — end-to-end pipeline runs ──────────────────
(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" matmul --shape 64x64x64 --dtype bf16) \
    >"$L5_MATMUL" 2>&1 || {
  echo "  ✗ tgrad-cli matmul --shape 64x64x64 --dtype bf16 failed"
  cat "$L5_MATMUL"; exit 1
}
# Verify the pipeline reported success and ran each stage.
for token in "compile_ok: 1" "alloc_ok: 1" "dispatch_rc: 0" "pipeline_ok: 1"; do
  if ! grep -q "$token" "$L5_MATMUL"; then
    echo "  ✗ Pipeline.realize output missing required token: '$token'"
    cat "$L5_MATMUL"; exit 1
  fi
done
echo "  ✓ Tgrad.Pipeline.realize 64×64 bf16 matmul completed end-to-end (compile + alloc + dispatch)"

# ─── LAYER C2: L5.b — byte-match vs captured tinygrad output ──────────
# Per §G_L5_b: required input/output fixtures must be present.
required_fixtures=(
  fixtures/pipeline/matmul_64x64_bf16_seed42_a.bin
  fixtures/pipeline/matmul_64x64_bf16_seed42_b.bin
  fixtures/pipeline/matmul_64x64_bf16_seed42_expected.bin
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing pipeline fixture: $f"; exit 1; }
  sz=$(stat -f%z "$REPO_ROOT/$f" 2>/dev/null || stat -c%s "$REPO_ROOT/$f")
  [[ "$sz" -eq 8192 ]] || { echo "  ✗ $f size=$sz expected 8192"; exit 1; }
done
echo "  ✓ all 3 pipeline fixtures present (8192 bytes each)"

(cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" matmul-verify --shape 64x64x64 --seed 42) \
    >"$L5_VERIFY" 2>&1 || {
  echo "  ✗ tgrad-cli matmul-verify failed"
  cat "$L5_VERIFY"; exit 1
}
grep -q "matmul_verify_ok: 1" "$L5_VERIFY" || {
  echo "  ✗ matmul-verify did NOT byte-match the captured tinygrad output"
  cat "$L5_VERIFY"; exit 1
}
echo "  ✓ Pipeline.realize output byte-matches captured tinygrad output (8192 bytes)"

# Sanity-check the fixture itself hasn't drifted (if someone mutates it
# without re-capturing, the predicate should surface that).
expected_sha="$(shasum -a 256 "$TGRAD_DIR/fixtures/pipeline/matmul_64x64_bf16_seed42_expected.bin" | awk '{print $1}')"

# ─── LAYER D: negative test ───────────────────────────────────────────
# Out-of-scope shape must surface a structured NotInLeanScope, not a
# silent zero return.
if (cd "$REPO_ROOT" && "$TGRAD_DIR/.lake/build/bin/tgrad-cli" matmul --shape 7x9x11 --dtype bf16) \
    >/dev/null 2>&1; then
  echo "  ✗ matmul --shape 7x9x11 returned 0 — should fail (NotInLeanScope)"
  exit 1
fi
echo "  ✓ negative test correctly rejected (out-of-scope shape → NotInLeanScope)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
matmul_hash="$(shasum -a 256 "$L5_MATMUL" | awk '{print $1}')"
verify_hash="$(shasum -a 256 "$L5_VERIFY" | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L5.json" <<EOF
{
  "gate": "L5",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "hashes": {
    "matmul_64x64_pipeline_sha256":  "$matmul_hash",
    "matmul_verify_output_sha256":   "$verify_hash",
    "expected_bytes_sha256":         "$expected_sha"
  }
}
EOF
check_evidence_for L5 || exit 1
check_falsifiability_verified L5 || exit 1
echo "  ✓ L5 pipeline gate green (evidence recorded)"

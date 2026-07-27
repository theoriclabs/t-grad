#!/usr/bin/env bash
# Gate L13.F.STRICT.A — Stmt grammar for threadgroup memory + manual WMMA loads.
#
# Predicates:
#   - Layer A: universal preflight.
#   - Layer B: five new Stmt constructors, render cases, and synthetic
#              kernel declaration present.
#   - Layer C: synthetic_tg_kernel renders byte-equal to the committed
#              fixture and compiles via ffi-compile-smoke.
#   - Layer C2: L12 and L13_F stay green.
#   - Layer D: renderKernel stays pure; synthetic decl uses all five
#              new constructors.
#   - Layer E: evidence file.
set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
cd "$REPO_ROOT"
source "$TGRAD_DIR/scripts/lib/checks.sh"
STRICT_A_EMIT="$(tgrad_run_path L13F_STRICT_A_synthetic_tg_kernel.msl)"
STRICT_A_RENDER_ERR="$(tgrad_run_path L13F_STRICT_A_render.err)"
STRICT_A_COMPILE="$(tgrad_run_path L13F_STRICT_A_compile.txt)"
STRICT_A_L13F="$(tgrad_run_path L13F_STRICT_A_L13_F.log)"
STRICT_A_L12="$(tgrad_run_path L13F_STRICT_A_L12.log)"

echo "[L13_F_STRICT_A] Stmt grammar for tg memory + manual WMMA loads"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

METAL="$TGRAD_DIR/Tgrad/Renderer/Metal.lean"
FIXTURE="$TGRAD_DIR/fixtures/codegen/synthetic_tg_kernel.msl"
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"
ctors=(threadgroupDecl threadgroupBarrier threadgroupLoad threadgroupStore perThreadWmmaLoad)

# ─── LAYER B: structural ──────────────────────────────────────────────
[[ -f "$METAL" ]] || { echo "  ✗ missing renderer: $METAL"; exit 1; }
[[ -x "$TGRAD_CLI" ]] || { echo "  ✗ missing tgrad-cli at $TGRAD_CLI"; exit 1; }

ctor_count="$(grep -cE '^[[:space:]]*\|[[:space:]]+(threadgroupDecl|threadgroupBarrier|threadgroupLoad|threadgroupStore|perThreadWmmaLoad)\b' "$METAL")"
if [[ "$ctor_count" -ne 5 ]]; then
  echo "  ✗ expected 5 L13.F.STRICT.A Stmt ctors, found $ctor_count"
  exit 1
fi
echo "  ✓ all 5 new Stmt constructors present"

render_block="$(awk '/^partial def render \(indent : String\)/{flag=1} flag{print} /^end Stmt/{if(flag){exit}}' "$METAL")"
for ctor in "${ctors[@]}"; do
  if ! grep -qE "^[[:space:]]*\\|[[:space:]]*\\.${ctor}\\b" <<<"$render_block"; then
    echo "  ✗ Stmt.render missing case for .$ctor"
    exit 1
  fi
done
echo "  ✓ Stmt.render has cases for all 5 new ctors"

if ! grep -qE '^def synthetic_tg_kernel : KernelDecl' "$METAL"; then
  echo "  ✗ Renderer/Metal.lean missing synthetic_tg_kernel"
  exit 1
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "  ✗ missing synthetic fixture: $FIXTURE"
  exit 1
fi
echo "  ✓ synthetic_tg_kernel decl + fixture present"

# ─── LAYER D1/D3: anti-cheat structural checks ────────────────────────
if ! grep -qE '^def renderKernel \(k : KernelDecl\) : String' "$METAL"; then
  echo "  ✗ renderKernel signature is not pure `KernelDecl → String`"
  exit 1
fi
echo "  ✓ renderKernel stays pure (KernelDecl → String)"

synthetic_block="$(awk '/^def synthetic_tg_kernel : KernelDecl/{flag=1} flag{print} flag && /trailingNewline := true/{exit}' "$METAL")"
for ctor in "${ctors[@]}"; do
  if ! grep -qE "\\.${ctor}\\b" <<<"$synthetic_block"; then
    echo "  ✗ synthetic_tg_kernel does not exercise .$ctor"
    exit 1
  fi
done
synthetic_ctor_count="$(grep -oE '\.(threadgroupDecl|threadgroupBarrier|threadgroupLoad|threadgroupStore|perThreadWmmaLoad)\b' <<<"$synthetic_block" | wc -l | awk '{print $1}')"
if [[ "$synthetic_ctor_count" -ne 5 ]]; then
  echo "  ✗ synthetic_tg_kernel should exercise exactly 5 new ctor uses, found $synthetic_ctor_count"
  exit 1
fi
echo "  ✓ synthetic_tg_kernel exercises all 5 new ctors exactly once"

# ─── LAYER C: render + byte-equal + compile smoke ─────────────────────
EMIT_OUT="$STRICT_A_EMIT"
"$TGRAD_CLI" render-metal-algebraic synthetic_tg_kernel >"$EMIT_OUT" 2>"$STRICT_A_RENDER_ERR" || {
  echo "  ✗ render-metal-algebraic synthetic_tg_kernel failed"
  cat "$STRICT_A_RENDER_ERR"
  exit 1
}
if ! cmp -s "$EMIT_OUT" "$FIXTURE"; then
  echo "  ✗ synthetic_tg_kernel render drifted from committed fixture"
  diff "$EMIT_OUT" "$FIXTURE" | head -40 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ synthetic_tg_kernel render byte-equals fixture"

COMPILE_OUT="$STRICT_A_COMPILE"
"$TGRAD_CLI" ffi-compile-smoke "$EMIT_OUT" >"$COMPILE_OUT" 2>&1 || {
  echo "  ✗ ffi-compile-smoke rejected synthetic_tg_kernel"
  cat "$COMPILE_OUT"
  exit 1
}
fn_count="$(awk -F': ' '/^fn_count/{print $2}' "$COMPILE_OUT")"
if [[ "$fn_count" != "1" ]]; then
  echo "  ✗ synthetic_tg_kernel fn_count=$fn_count (expected 1)"
  cat "$COMPILE_OUT"
  exit 1
fi
echo "  ✓ synthetic-tg-kernel: fn_count: 1 (ffi-compile-smoke)"

# ─── LAYER C2: regression checks ──────────────────────────────────────
run_gate_isolated "$TGRAD_DIR/scripts/gates/L13_F.sh" >"$STRICT_A_L13F" 2>&1 || {
  echo "  ✗ L13_F regression failed"
  tail -40 "$STRICT_A_L13F" | sed 's/^/      /'
  exit 1
}
echo "  ✓ L13_F regression gate still green under §1.RELAX"

run_gate_isolated "$TGRAD_DIR/scripts/gates/L12.sh" >"$STRICT_A_L12" 2>&1 || {
  echo "  ✗ L12 regression failed"
  tail -40 "$STRICT_A_L12" | sed 's/^/      /'
  exit 1
}
echo "  ✓ L12 regression gate still green"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
renderer_hash="$(shasum -a 256 "$METAL" | awk '{print $1}')"
fixture_hash="$(shasum -a 256 "$FIXTURE" | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L13_F_STRICT_A.json" <<EOF
{
  "gate": "L13_F_STRICT_A",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L13.F.STRICT.A — Stmt grammar for tg memory + barriers + per-thread WMMA loads",
  "new_ctors": ["threadgroupDecl", "threadgroupBarrier", "threadgroupLoad",
                "threadgroupStore", "perThreadWmmaLoad"],
  "synthetic_kernel_compiled": true,
  "l13_f_regression": "pass",
  "l12_regression": "pass",
  "hashes": {
    "metal_renderer_sha256": "$renderer_hash",
    "synthetic_kernel_msl_sha256": "$fixture_hash"
  }
}
EOF
check_evidence_for L13_F_STRICT_A || exit 1
check_falsifiability_verified L13_F_STRICT_A || exit 1
echo "  ✓ L13.F.STRICT.A gate green (synthetic tg kernel + regressions)"

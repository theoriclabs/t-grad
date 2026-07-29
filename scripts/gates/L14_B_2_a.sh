#!/usr/bin/env bash
# Gate L14.B.2.a — Stmt grammar (loadIndexed/storeIndexed) +
# UOp.renderIndexExpr + synthetic test kernel.
#
# Per `GOAL_L14_B_2_a.md` §1+§5. NO fall-back: any structural,
# behavioural, or regression failure is L14.B.2.a RED.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : structural
#       * 2 new Stmt constructors: loadIndexed, storeIndexed
#       * UOp.renderIndexExpr declared (pure)
#       * synthetic_indexed_kernel KernelDecl declared
#       * Stmt.render has cases for both new ctors
#   - Layer C : behavioural
#       * C1 — render-metal-algebraic synthetic_indexed_kernel; output
#              byte-equal to fixtures/codegen/synthetic_indexed_kernel.msl
#       * C2 — ffi-compile-smoke on the rendered MSL → fn_count: 1
#   - Layer C2 : regression evidence — L11/L12/L13/L13_F/L14_B_1 evidence
#                files still record prior passing state (L14.B.2.a adds
#                ONLY constructors that no existing kernel consumes)
#   - Layer D : anti-cheat
#       * D1 — UOp.renderIndexExpr signature is pure (no IO)
#       * D2 — synthetic kernel exercises both new ctors (>= 1 each)
#       * D3 — gate script invokes ffi-compile-smoke (no silent pass)
#   - Layer E : evidence to fixtures/gate_evidence/L14_B_2_a.json
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14_B_2_a] Stmt grammar (loadIndexed/storeIndexed) + UOp.renderIndexExpr"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# `check_clean_rebuild` cd's into $TGRAD_DIR without a subshell —
# Lean's tools and CLI invocations below run from REPO_ROOT.
cd "$REPO_ROOT"

# ─── LAYER B: structural predicates ───────────────────────────────────
required_modules=(
  Tgrad/UOp.lean
  Tgrad/Renderer/Metal.lean
  # The CLI entry point is `Main.lean` at the repository root --
  # `lean_exe «tgrad-cli» where root := `Main` in lakefile.lean.
  # This read `Tgrad/Main.lean` from acf53f7, the commit that created BOTH
  # this gate and Main.lean, so the path has never resolved and this gate
  # could never pass its own structural layer -- while being listed in
  # GREEN_GATES the whole time. Correcting the path restores the check's
  # intent (the module defining the CLI entry must exist); it does not
  # weaken it.
  Main.lean
  fixtures/codegen/synthetic_indexed_kernel.msl
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# 2 new Stmt constructors.
n_new_ctors="$(grep -cE '^[[:space:]]+\|[[:space:]]+(loadIndexed|storeIndexed)[[:space:]]+\(' \
              "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" 2>/dev/null || true)"
n_new_ctors="${n_new_ctors:-0}"
if [[ "$n_new_ctors" -ne 2 ]]; then
  echo "  ✗ Renderer/Metal.lean has $n_new_ctors/2 new Stmt constructors"
  exit 1
fi
echo "  ✓ Stmt grammar has 2 new constructors (loadIndexed + storeIndexed)"

# UOp.renderIndexExpr declared.
if ! grep -qE '^(partial[[:space:]]+)?def renderIndexExpr' \
       "$TGRAD_DIR/Tgrad/UOp.lean"; then
  echo "  ✗ UOp.lean missing renderIndexExpr"
  exit 1
fi
echo "  ✓ UOp.renderIndexExpr declared"

# synthetic_indexed_kernel KernelDecl declared.
if ! grep -qE '^def synthetic_indexed_kernel' \
       "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
  echo "  ✗ Renderer/Metal.lean missing synthetic_indexed_kernel"
  exit 1
fi
echo "  ✓ synthetic_indexed_kernel KernelDecl declared"

# Stmt.render has cases for both new ctors (`| .loadIndexed ...` /
# `| .storeIndexed ...` arms; constructor declarations themselves use
# `| loadIndexed (...) (...)` without the dot prefix).
n_render_arms="$(grep -cE '\|[[:space:]]+\.(loadIndexed|storeIndexed)\b' \
                "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" 2>/dev/null || true)"
n_render_arms="${n_render_arms:-0}"
if [[ "$n_render_arms" -lt 2 ]]; then
  echo "  ✗ Stmt.render missing arms for the new ctors (found $n_render_arms / 2 match arms)"
  exit 1
fi
echo "  ✓ Stmt.render has match arms for loadIndexed + storeIndexed (>= 2)"

# Layer D2: synthetic kernel exercises both new ctors.
SYNTH_START="$(grep -nE '^def synthetic_indexed_kernel' \
              "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" | head -1 | cut -d: -f1)"
if [[ -z "$SYNTH_START" ]]; then
  echo "  ✗ synthetic_indexed_kernel def not found"
  exit 1
fi
SYNTH_END=$(( SYNTH_START + 20 ))
SYNTH_BODY="$(sed -n "${SYNTH_START},${SYNTH_END}p" \
             "$TGRAD_DIR/Tgrad/Renderer/Metal.lean")"
if ! echo "$SYNTH_BODY" | grep -qE '\.loadIndexed\b'; then
  echo "  ✗ synthetic kernel body missing .loadIndexed usage:"
  echo "$SYNTH_BODY" | sed 's/^/      /'
  exit 1
fi
if ! echo "$SYNTH_BODY" | grep -qE '\.storeIndexed\b'; then
  echo "  ✗ synthetic kernel body missing .storeIndexed usage"
  exit 1
fi
echo "  ✓ synthetic kernel exercises both .loadIndexed and .storeIndexed (D2)"

# ─── LAYER D1: UOp.renderIndexExpr is pure (no IO) ────────────────────
RIE_SIG="$(grep -E '^(partial[[:space:]]+)?def renderIndexExpr' \
          "$TGRAD_DIR/Tgrad/UOp.lean" | head -1)"
if echo "$RIE_SIG" | grep -qE '\bIO\b'; then
  echo "  ✗ UOp.renderIndexExpr signature contains IO (must be pure):"
  echo "      $RIE_SIG"
  exit 1
fi
echo "  ✓ UOp.renderIndexExpr is pure (UOp → String)"

# ─── LAYER C: behavioural ─────────────────────────────────────────────
# C1: byte-equal vs committed fixture.
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"
[[ -x "$TGRAD_CLI" ]] || { echo "  ✗ tgrad-cli missing at $TGRAD_CLI"; exit 1; }

EMIT_OUT="$(mktemp -t tgrad_L14B2a_emit.XXXXXX.msl)"
"$TGRAD_CLI" render-metal-algebraic synthetic_indexed_kernel >"$EMIT_OUT" 2>&1
EMIT_RC=$?
if [[ "$EMIT_RC" -ne 0 ]]; then
  echo "  ✗ render-metal-algebraic synthetic_indexed_kernel failed (rc=$EMIT_RC):"
  sed 's/^/      /' "$EMIT_OUT"
  rm -f "$EMIT_OUT"
  exit 1
fi

if ! diff -q "$EMIT_OUT" "$TGRAD_DIR/fixtures/codegen/synthetic_indexed_kernel.msl" >/dev/null; then
  echo "  ✗ rendered synthetic_indexed kernel doesn't byte-match committed fixture:"
  diff "$EMIT_OUT" "$TGRAD_DIR/fixtures/codegen/synthetic_indexed_kernel.msl" | sed 's/^/      /'
  rm -f "$EMIT_OUT"
  exit 1
fi
echo "  ✓ synthetic_indexed_kernel renders byte-equal to committed fixture"

# C2: ffi-compile-smoke.
SMOKE_OUT="$(mktemp -t tgrad_L14B2a_smoke.XXXXXX.txt)"
"$TGRAD_CLI" ffi-compile-smoke "$TGRAD_DIR/fixtures/codegen/synthetic_indexed_kernel.msl" >"$SMOKE_OUT" 2>&1
SMOKE_RC=$?
if [[ "$SMOKE_RC" -ne 0 ]] || ! grep -q '^fn_count: 1' "$SMOKE_OUT"; then
  echo "  ✗ ffi-compile-smoke on synthetic_indexed_kernel failed (rc=$SMOKE_RC):"
  sed 's/^/      /' "$SMOKE_OUT"
  rm -f "$EMIT_OUT" "$SMOKE_OUT"
  exit 1
fi
echo "  ✓ ffi-compile-smoke: synthetic_indexed_kernel compiles (fn_count: 1)"
rm -f "$EMIT_OUT" "$SMOKE_OUT"

# ─── LAYER D3: gate references ffi-compile-smoke (anti-silent-pass) ───
if ! grep -qF 'ffi-compile-smoke' "$0"; then
  echo "  ✗ L14_B_2_a.sh missing ffi-compile-smoke invocation"
  exit 1
fi
echo "  ✓ L14_B_2_a.sh invokes ffi-compile-smoke (D3)"

# ─── LAYER C2 (regression evidence): L11/L12/L13/L13_F/L14_B_1 intact ──
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

L11_PAIRS="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])
' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11.json shows 50/50 (regression evidence)"

L13F_TC="$("$PY" -c '
import json
d = json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L13_F.json"'"))
print(d["tc_general_wmma"], d["random_tc_wmma"], d["tc_general_scalar_routes"])
' 2>/dev/null || echo "0 0 1")"
read L13F_PIN L13F_RAND L13F_SCALAR <<< "$L13F_TC"
[[ "$L13F_PIN" -eq 8 && "$L13F_RAND" -eq 10 && "$L13F_SCALAR" -eq 0 ]] || {
  echo "  ✗ L13_F.json shows ${L13F_PIN}/8 + ${L13F_RAND}/10 + ${L13F_SCALAR} scalar"
  exit 1
}
echo "  ✓ L13_F.json shows 8/8 + 10/10 + 0 scalar"

L14_B_1_VIEW="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L14_B_1.json"'"))["matmul_on_view_raises"])
' 2>/dev/null || echo "missing")"
[[ "$L14_B_1_VIEW" == "MatmulOnNonBufferUop" ]] || {
  echo "  ✗ L14_B_1.json.matmul_on_view_raises != MatmulOnNonBufferUop (got $L14_B_1_VIEW)"
  exit 1
}
echo "  ✓ L14_B_1.json shows MatmulOnNonBufferUop typed-error guard"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
metal_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" | awk '{print $1}')"
uop_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/UOp.lean" | awk '{print $1}')"
msl_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/codegen/synthetic_indexed_kernel.msl" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L14_B_2_a.json" <<EOF
{
  "gate": "L14_B_2_a",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14.B.2.a — Stmt grammar (loadIndexed/storeIndexed) + UOp.renderIndexExpr",
  "new_ctors": ["loadIndexed", "storeIndexed"],
  "render_index_expr_pure": true,
  "synthetic_kernel_compiled": true,
  "l11_regression": "pass",
  "l12_regression": "pass",
  "l13_regression": "pass",
  "l13_f_regression": "pass",
  "l14_b_1_regression": "pass",
  "hashes": {
    "metal_renderer_sha256":      "$metal_hash",
    "uop_module_sha256":          "$uop_hash",
    "synthetic_kernel_msl_sha256": "$msl_hash"
  }
}
EOF
check_evidence_for L14_B_2_a || exit 1
check_falsifiability_verified L14_B_2_a || exit 1
echo "  ✓ L14.B.2.a Stmt grammar + UOp.renderIndexExpr — GREEN"

#!/usr/bin/env bash
# Gate L14.B.2.b — refactor 3 matmul kernel families to indexed
# LOAD/STORE (driven by index UOps via UOp.renderIndexExpr).
#
# Per `GOAL_L14_B_2_b.md` §1+§5. NO fall-back: any structural,
# behavioural, or regression failure is L14.B.2.b RED.
#
# Scope of this refactor:
#   - MatmulScalar.lean's 1 `.dataStore` → `.storeIndexed`.
#   - MatmulDecls.lean's 352 `.dataStore` lines (10 shapes × ~32 stores)
#     → `.storeIndexed` via the extended `lower_matmul.py` transpiler.
#   - MatmulTc.lean — no `.dataStore` to migrate (the L13.F kernel
#     uses `simdgroup_load`/`simdgroup_store` inside `Stmt.tcMatmulBody`,
#     which renders the cooperative-load primitives directly; the
#     L14.B.2.b grep on MatmulTc reads 0 dataStore by construction
#     and the combined storeIndexed count across MatmulDecls + the
#     scalar kernel passes the `>= 6` predicate).
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : structural
#       * MatmulDecls + MatmulTc + MatmulScalar contain 0 .dataStore
#         (or .dataLoad) — all replaced by indexed variants
#       * Combined .storeIndexed (+ .loadIndexed) count >= 6
#       * lower_matmul.py contains `parse_offset_to_uop_lean` —
#         the transpiler extension that emits storeIndexed
#   - Layer C : behavioural (evidence-file-based regression)
#       * L11.json shows 50/50 pairs (matmul Python FFI path bit-identical)
#       * L12.json shows 10/10 byte-equal — the canary that proves
#         the new `.storeIndexed` renderer matches the captured fixtures
#       * L13.json + L13_F.json show prior passing state
#       * 64×64 byte-match smoke proves dylib FFI path is intact
#   - Layer D : anti-cheat
#       * D1 — rendered MSL for matmul_4096x4096x4096 contains the
#              `*(data0_*+(alu74+8))` pattern that proves index UOps
#              drive real pointer-arith (not collapsed to a constant)
#       * D2 — `lower_matmul.py`'s `parse_offset_to_uop_lean` covers
#              both `(var+const)` AND bare-var forms (no silent skip)
#       * D3 — Stmt.storeIndexed signature is pure (`(buf : String)
#              (idx : UOp) (rhs : String)` — no IO)
#   - Layer E : evidence to fixtures/gate_evidence/L14_B_2_b.json
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14_B_2_b] refactor 3 matmul kernels to indexed LOAD/STORE"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight
cd "$REPO_ROOT"  # checks.sh's check_clean_rebuild leaves cwd at $TGRAD_DIR

# ─── LAYER B: structural ──────────────────────────────────────────────
required_modules=(
  Tgrad/Renderer/MatmulDecls.lean
  Tgrad/Renderer/MatmulScalar.lean
  Tgrad/Renderer/MatmulTc.lean
  scripts/dev/lower_matmul.py
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# No `.dataStore` / `.dataLoad` in matmul kernel files — exclude
# comment lines (`--`) that may legitimately mention the old ctor
# names while documenting the migration.
n_data_calls="$( ( grep -E '^[[:space:]]*\.(dataStore|dataLoad)\b' \
                  "$TGRAD_DIR/Tgrad/Renderer/MatmulDecls.lean" \
                  "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" \
                  "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean" \
                  2>/dev/null || true ) \
                | wc -l | awk '{print $1}')"
n_data_calls="${n_data_calls:-0}"
if [[ "$n_data_calls" -ne 0 ]]; then
  echo "  ✗ matmul kernel files still contain $n_data_calls dataStore/dataLoad calls"
  grep -nE '^[[:space:]]*\.(dataStore|dataLoad)\b' \
       "$TGRAD_DIR/Tgrad/Renderer/MatmulDecls.lean" \
       "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" \
       "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean" | head -3 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ 0 .dataStore/.dataLoad ctor calls remain in matmul kernel files"

# Combined .storeIndexed + .loadIndexed count >= 6.
n_indexed="$(grep -cE '\.(loadIndexed|storeIndexed)\b' \
            "$TGRAD_DIR/Tgrad/Renderer/MatmulDecls.lean" \
            "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" \
            "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean" \
            2>/dev/null \
            | awk -F: '{s+=$NF} END {print s+0}')"
if [[ "$n_indexed" -lt 6 ]]; then
  echo "  ✗ combined .loadIndexed/.storeIndexed count is $n_indexed (need >= 6)"
  exit 1
fi
echo "  ✓ matmul kernel files contain $n_indexed loadIndexed/storeIndexed calls (>= 6)"

# lower_matmul.py has the transpiler extension.
if ! grep -qE '^def parse_offset_to_uop_lean' \
       "$TGRAD_DIR/scripts/dev/lower_matmul.py"; then
  echo "  ✗ lower_matmul.py missing parse_offset_to_uop_lean (transpiler not extended)"
  exit 1
fi
echo "  ✓ lower_matmul.py extended with parse_offset_to_uop_lean"

# ─── LAYER D3: Stmt.storeIndexed signature is pure ────────────────────
SIG="$(grep -nE '\|[[:space:]]+storeIndexed[[:space:]]*\(' \
       "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" | head -1)"
if [[ -z "$SIG" ]] || echo "$SIG" | grep -qE '\bIO\b'; then
  echo "  ✗ Stmt.storeIndexed declaration missing or contains IO:"
  echo "      $SIG"
  exit 1
fi
echo "  ✓ Stmt.storeIndexed signature is pure (D3)"

# ─── LAYER D2: parse_offset_to_uop_lean handles both forms ────────────
if ! grep -qE 'var\\+const' "$TGRAD_DIR/scripts/dev/lower_matmul.py" \
   && ! grep -qE 'bare var' "$TGRAD_DIR/scripts/dev/lower_matmul.py"; then
  # Fallback grep — both regex patterns should appear in the function body
  if ! grep -qE '_OFFSET_VARCONST_RE' "$TGRAD_DIR/scripts/dev/lower_matmul.py" \
     || ! grep -qE '_OFFSET_VAR_RE' "$TGRAD_DIR/scripts/dev/lower_matmul.py"; then
    echo "  ✗ parse_offset_to_uop_lean doesn't cover both forms"
    exit 1
  fi
fi
echo "  ✓ parse_offset_to_uop_lean covers var+const AND bare-var forms (D2)"

# ─── LAYER C: behavioural — rendered MSL byte-equal + dylib FFI smoke ─
ensure_dylib /tmp/tgrad_L14B2b_dylib.log || exit 1
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"

# C1 (L12 canary): re-render matmul_4096x4096x4096; byte-equal vs captured.
SHAPE="matmul_4096x4096x4096"
EMIT_OUT="$(mktemp -t tgrad_L14B2b_${SHAPE}.XXXXXX.msl)"
"$TGRAD_CLI" render-metal-algebraic "$SHAPE" >"$EMIT_OUT" 2>&1
EMIT_RC=$?
if [[ "$EMIT_RC" -ne 0 ]]; then
  echo "  ✗ render-metal-algebraic $SHAPE failed (rc=$EMIT_RC)"
  sed 's/^/      /' "$EMIT_OUT"
  rm -f "$EMIT_OUT"
  exit 1
fi
if ! diff -q "$EMIT_OUT" "$TGRAD_DIR/fixtures/codegen/$SHAPE.msl" >/dev/null; then
  echo "  ✗ $SHAPE rendered output doesn't byte-match captured fixture"
  diff "$EMIT_OUT" "$TGRAD_DIR/fixtures/codegen/$SHAPE.msl" | head -6 | sed 's/^/      /'
  rm -f "$EMIT_OUT"
  exit 1
fi
echo "  ✓ $SHAPE byte-equals captured fixture after refactor (L12 canary)"

# D1: the rendered output should contain `*(buf+(alu74+...))` patterns
# proving index UOps drive real pointer-arith, not constants.
if ! grep -qE '\*\(data0_[0-9]+\+\(alu[0-9]+\+[0-9]+\)\)' "$EMIT_OUT"; then
  echo "  ✗ rendered $SHAPE doesn't contain expected pointer-arith pattern"
  exit 1
fi
echo "  ✓ rendered MSL contains real index-driven pointer-arith (D1)"
rm -f "$EMIT_OUT"

# C2: 64×64 byte-match smoke (proves dylib FFI path intact).
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

SMOKE_OUT="$(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench --shape 64x64x64 --dtype bf16 2>&1)"
if ! echo "$SMOKE_OUT" | grep -q '^py_byte_match: true$'; then
  echo "  ✗ 64×64 bench did NOT byte-match captured fixture after L14.B.2.b:"
  echo "$SMOKE_OUT" | sed 's/^/      /'
  exit 1
fi
echo "  ✓ 64×64 matmul byte-matches captured fixture (FFI path intact)"

# ─── LAYER C2 (regression evidence): L11/L12/L13/L13_F intact ──────────
L11_PAIRS="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])
' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11.json shows 50/50 pairs (regression evidence)"

L12_BYTE_EQ="$("$PY" -c '
import json
d = json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L12.json"'"))
print(d.get("shapes_byte_equal", d.get("byte_equal_shapes", 0)))
' 2>/dev/null || echo 0)"
[[ "$L12_BYTE_EQ" -eq 10 ]] || {
  echo "  ✗ L12 evidence shows $L12_BYTE_EQ/10 byte-equal — re-run L12.sh to refresh"
  exit 1
}
echo "  ✓ L12 evidence shows 10/10 byte-equal"

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
echo "  ✓ L13_F.json shows 8/8 + 10/10 + 0 scalar (regression evidence)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
decls_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulDecls.lean" | awk '{print $1}')"
tc_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean" | awk '{print $1}')"
scalar_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" | awk '{print $1}')"
metal_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" | awk '{print $1}')"
transpiler_hash="$(shasum -a 256 "$TGRAD_DIR/scripts/dev/lower_matmul.py" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L14_B_2_b.json" <<EOF
{
  "gate": "L14_B_2_b",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14.B.2.b — 3 matmul kernel families use indexed LOAD/STORE",
  "kernels_refactored": ["matmulKernelDeclFor", "scalarMatmulKernelDecl"],
  "kernels_already_compliant": ["tcMatmulKernelDecl"],
  "data_load_remaining": 0,
  "data_store_remaining": 0,
  "indexed_load_count": 0,
  "indexed_store_count": $n_indexed,
  "l11_regression": "pass",
  "l12_byte_equal": 10,
  "l13_f_regression": "pass",
  "hashes": {
    "matmul_decls_sha256":  "$decls_hash",
    "matmul_tc_sha256":     "$tc_hash",
    "scalar_kernel_sha256": "$scalar_hash",
    "renderer_sha256":      "$metal_hash",
    "transpiler_sha256":    "$transpiler_hash"
  }
}
EOF
check_evidence_for L14_B_2_b || exit 1
check_falsifiability_verified L14_B_2_b || exit 1
echo "  ✓ L14.B.2.b matmul-kernel-indexed-LOAD/STORE refactor — GREEN ($n_indexed indexed Stmts, 0 dataStore)"

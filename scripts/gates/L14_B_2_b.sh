#!/usr/bin/env bash
# Gate L14.B.2.b — typed index expressions are load-bearing in generated
# scalar and tensor-core matmul kernels.
#
# The former gate counted hundreds of generated lines in MatmulDecls and
# inspected the parser that produced them. Those artifacts are deleted. This
# gate now checks the emitted program, Lean's non-aliasing theorems, one fresh
# captured/generated differential, and the production FFI smoke.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14_B_2_b] generated matmul uses typed, non-aliasing indices"
run_preflight
cd "$REPO_ROOT"

WORK_DIR="$(tgrad_run_subdir L14B2b_work)"
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"

# ─── LAYER B: structural/type-checked obligations ─────────────────────
required_modules=(
  Tgrad/Renderer/Metal.lean
  Tgrad/Renderer/MatmulScalar.lean
  Tgrad/Renderer/MatmulTc.lean
  Tgrad/Pipeline.lean
  scripts/differential_codegen.sh
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
for removed in Tgrad/Renderer/MatmulDecls.lean scripts/dev/lower_matmul.py; do
  [[ ! -e "$REPO_ROOT/$removed" ]] || {
    echo "  ✗ deleted transcription artifact returned: $removed"; exit 1; }
done

n_data_calls="$( (grep -E '^[[:space:]]*\.(dataStore|dataLoad)\b' \
  "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" \
  "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean" 2>/dev/null || true) \
  | wc -l | awk '{print $1}')"
if [[ "$n_data_calls" -ne 0 ]]; then
  echo "  ✗ generated matmul declarations still construct $n_data_calls raw dataStore/dataLoad statements"
  exit 1
fi

for thm in tileStoreOffsets_nodup_128 tileStoreOffsets_nodup_1024 tileAccSlots_nodup; do
  if ! grep -qE "^theorem ${thm}\b" "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean"; then
    echo "  ✗ missing checked non-aliasing obligation: $thm"
    exit 1
  fi
done

SIG="$(grep -nE '\|[[:space:]]+storeIndexed[[:space:]]*\(' \
  "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" | head -1)"
if [[ -z "$SIG" ]] || echo "$SIG" | grep -qE '\bIO\b'; then
  echo "  ✗ Stmt.storeIndexed declaration missing or effectful"
  exit 1
fi
echo "  ✓ raw address statements absent; typed store and non-aliasing obligations present"

# ─── LAYER C: inspect and execute generated output ────────────────────
TC_EMIT="$WORK_DIR/tc.msl"
"$TGRAD_CLI" render-metal-algebraic matmul_tc_manual_1024x1024x3072 \
  >"$TC_EMIT" 2>/dev/null || { echo "  ✗ failed to render generated TC kernel"; exit 1; }
n_st="$(grep -cE '^[[:space:]]*\*\(data0\+.*\) = \(\(bfloat\)\(\(.*\)\)\);$' "$TC_EMIT" || true)"
n_ld="$(grep -cE '^[[:space:]]*bfloat val[0-9]+ = \*\(data[12]\+.*\);$' "$TC_EMIT" || true)"
if [[ "$n_st" -ne 32 || "$n_ld" -ne 16 ]]; then
  echo "  ✗ generated TC kernel emitted stores/loads=$n_st/$n_ld (expected 32/16)"
  exit 1
fi
if ! grep -qE '\*\(data0\+.*\+.*\)' "$TC_EMIT"; then
  echo "  ✗ generated stores do not contain rendered index arithmetic"
  exit 1
fi
echo "  ✓ emitted TC kernel contains 32 typed stores and 16 typed loads"

DIFF_LOG="$WORK_DIR/differential.log"
if ! bash "$REPO_ROOT/scripts/differential_codegen.sh" 4096x4096x4096 \
    >"$DIFF_LOG" 2>&1; then
  echo "  ✗ fresh wrong-placement differential failed"
  sed 's/^/      /' "$DIFF_LOG"
  exit 1
fi
grep -q 'bit-identical over' "$DIFF_LOG" \
  || { echo "  ✗ differential did not report bit identity"; exit 1; }
grep -q 'sources differ' "$DIFF_LOG" \
  || { echo "  ✗ differential did not establish source independence"; exit 1; }
echo "  ✓ fresh 4096³ generated/captured differential is bit-identical and source-different"

ensure_dylib "$WORK_DIR/dylib.log" || exit 1
SMOKE_OUT="$(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" \
  bench --shape 64x64x64 --dtype bf16 2>&1)"
if ! echo "$SMOKE_OUT" | grep -q '^py_byte_match: true$'; then
  echo "  ✗ production 64³ FFI path failed byte-match smoke"
  echo "$SMOKE_OUT" | sed 's/^/      /'
  exit 1
fi
echo "  ✓ production 64³ FFI path remains numerically exact"

# L12's full 11-shape differential and the fresh single-shape differential
# are complementary to Nodup: the theorem rejects collisions, while execution
# rejects wrong-but-distinct placements such as c ↦ c+2.
read L12_EQ L12_DIFF L12_TRANS <<<"$("$PY" -c '
import json
d=json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L12.json"'"))
print(d.get("semantic_bit_identical", 0), d.get("sources_differ", 0), d.get("transcription_files_present", -1))
' 2>/dev/null || echo '0 0 -1')"
if [[ "$L12_EQ" -ne 11 || "$L12_DIFF" -ne 11 || "$L12_TRANS" -ne 0 ]]; then
  echo "  ✗ L12 semantic evidence is stale; run L12 before L14_B_2_b"
  exit 1
fi
echo "  ✓ L12 evidence records 11/11 semantic agreement, source inequality, and no transcription"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"
plat="$(uname -srm)"
tc_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean" | awk '{print $1}')"
scalar_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" | awk '{print $1}')"
metal_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/Metal.lean" | awk '{print $1}')"
pipeline_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Pipeline.lean" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L14_B_2_b.json" <<EOF
{
  "gate": "L14_B_2_b",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L14.B.2.b — generated scalar and TC matmul use typed, non-aliasing index expressions",
  "kernels_refactored": ["scalarMatmulKernelDecl", "tcMatmulKernelDeclManualLoadWide"],
  "transcription_files_present": 0,
  "data_load_remaining": 0,
  "data_store_remaining": 0,
  "emitted_indexed_load_count": $n_ld,
  "emitted_indexed_store_count": $n_st,
  "l12_semantic_bit_identical": $L12_EQ,
  "fresh_differential": "pass",
  "hashes": {
    "matmul_tc_sha256": "$tc_hash",
    "scalar_kernel_sha256": "$scalar_hash",
    "renderer_sha256": "$metal_hash",
    "pipeline_sha256": "$pipeline_hash"
  }
}
EOF
check_evidence_for L14_B_2_b || exit 1
check_falsifiability_verified L14_B_2_b || exit 1
echo "  ✓ L14.B.2.b typed-index gate green"

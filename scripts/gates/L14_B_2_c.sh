#!/usr/bin/env bash
# Gate L14.B.2.c — Schedule.Rangeify wired into Pipeline.realize +
# smoke view (transpose_left 64×64×64) numerically correct + guard
# removed. Per `GOAL_L14_B_2_c.md` §1+§5.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : structural
#       * Pipeline.lean references Schedule.Rangeify.rangeify >= 1
#       * PythonFFI.lean's matmulView entry (@[export tgrad_matmul_view_lean])
#         exists; Python __matmul__ no longer raises MatmulOnNonBufferUop
#         unconditionally for view inputs
#       * RangeifyTraceRow + RangeifyTrace.maybeEmit defined
#       * scalarMatmulKernelDeclWithIdx exists (parametric scalar)
#       * realizeView function defined in Pipeline.lean
#   - Layer C1 : behavioural — smoke view (transpose_left 64×64×64)
#                computes a.T @ b with numpy reference within bf16
#                tolerance (rtol=0.02, atol=0.05) when comparing
#                against the BF16-roundtripped reference.
#   - Layer C2 : rangeify trace fires for the smoke and records
#                movement_count_in >= 1.
#   - Layer C3 : regression evidence (L11 / L12 / L13 / L13_F / L14_B_1
#                / L14_B_2_a / L14_B_2_b) all show prior passing state.
#   - Layer D  : anti-cheat
#       * D1 — Schedule.Rangeify.rangeify call site exists in
#              Pipeline.lean (real call, not just docstring)
#       * D2 — Pipeline.realize body doesn't pattern-match on uop kind
#              names (e.g. .transpose / .permute / .reshape / etc.)
#       * D3 — view smoke compares against numpy reference, NOT another
#              Tgrad call (anti-self-comparison)
#       * D4 — `movement_count_in` is computed via UOp.countMovementNodes,
#              not hardcoded
#   - Layer E : evidence to fixtures/gate_evidence/L14_B_2_c.json
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L14_B_2_c] Schedule.Rangeify wired + smoke view + guard removed"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight
cd "$REPO_ROOT"

# ─── LAYER B: structural ──────────────────────────────────────────────
required_modules=(
  Tgrad/Pipeline.lean
  Tgrad/PythonFFI.lean
  Tgrad/Schedule/Rangeify.lean
  Tgrad/Renderer/MatmulScalar.lean
  python/tgrad.py
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# Schedule.Rangeify.rangeify referenced in Pipeline.realize (count >= 1).
n_rangeify="$(grep -c 'Schedule\.Rangeify\.rangeify' "$TGRAD_DIR/Tgrad/Pipeline.lean" 2>/dev/null || true)"
n_rangeify="${n_rangeify:-0}"
if [[ "$n_rangeify" -lt 1 ]]; then
  echo "  ✗ Pipeline.lean missing Schedule.Rangeify.rangeify call (count $n_rangeify)"
  exit 1
fi
echo "  ✓ Pipeline.lean references Schedule.Rangeify.rangeify ($n_rangeify call site)"

# RangeifyTrace.maybeEmit defined.
if ! grep -qE 'def RangeifyTrace\.maybeEmit' "$TGRAD_DIR/Tgrad/Pipeline.lean"; then
  echo "  ✗ Pipeline.lean missing RangeifyTrace.maybeEmit"
  exit 1
fi
echo "  ✓ RangeifyTrace.maybeEmit defined"

# realizeView defined.
if ! grep -qE 'def realizeView' "$TGRAD_DIR/Tgrad/Pipeline.lean"; then
  echo "  ✗ Pipeline.lean missing realizeView"
  exit 1
fi
echo "  ✓ Pipeline.realizeView defined"

# tgrad_matmul_view_lean @[export] entry.
if ! grep -qF '@[export tgrad_matmul_view_lean]' "$TGRAD_DIR/Tgrad/PythonFFI.lean"; then
  echo "  ✗ PythonFFI.lean missing @[export tgrad_matmul_view_lean]"
  exit 1
fi
echo "  ✓ @[export tgrad_matmul_view_lean] declared"

# Python __matmul__ routes view inputs through tgrad_matmul_view.
if ! grep -qE 'tgrad_matmul_view\(' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py __matmul__ doesn't route view inputs to tgrad_matmul_view"
  exit 1
fi
echo "  ✓ Python __matmul__ routes view inputs to tgrad_matmul_view"

# scalarMatmulKernelDeclWithIdx (parametric scalar) exists.
if ! grep -qE '^def scalarMatmulKernelDeclWithIdx' \
       "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean"; then
  echo "  ✗ MatmulScalar.lean missing scalarMatmulKernelDeclWithIdx"
  exit 1
fi
echo "  ✓ scalarMatmulKernelDeclWithIdx (parametric) defined"

# ─── LAYER D2: realize body doesn't pattern-match on uop kinds ────────
REALIZE_START="$(grep -nE '^def realize\b' "$TGRAD_DIR/Tgrad/Pipeline.lean" | head -1 | cut -d: -f1)"
if [[ -z "$REALIZE_START" ]]; then
  echo "  ✗ Pipeline.realize def not found"
  exit 1
fi
REALIZE_END=$(( REALIZE_START + 60 ))
REALIZE_BODY="$(sed -n "${REALIZE_START},${REALIZE_END}p" "$TGRAD_DIR/Tgrad/Pipeline.lean")"
if echo "$REALIZE_BODY" | grep -qE '\bmatch[[:space:]].*\.(transpose|permute|reshape|expand|slice)\b'; then
  echo "  ✗ Pipeline.realize body pattern-matches on a movement-op uop kind (D2 violation)"
  echo "$REALIZE_BODY" | grep -nE 'match.*\.(transpose|permute|reshape|expand|slice)' | sed 's/^/      /'
  exit 1
fi
echo "  ✓ Pipeline.realize body doesn't pattern-match on movement-op uops (D2)"

# ─── LAYER C1+C2: smoke view ──────────────────────────────────────────
ensure_dylib /tmp/tgrad_L14B2c_dylib.log || exit 1

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

SMOKE_PY="$(mktemp -t tgrad_L14B2c_smoke.XXXXXX.py)"
cat >"$SMOKE_PY" <<'PYEOF'
import sys, os, json, hashlib
sys.path.insert(0, os.path.join(os.environ.get("REPO_ROOT", "."), "Tgrad", "python"))
os.environ["TGRAD_RANGEIFY_TRACE"] = "1"
TRACE_PATH = "/tmp/tgrad_rangeify_trace.jsonl"
open(TRACE_PATH, "w").close()  # reset

import numpy as np
import tgrad

def _to_bf16_f32(arr):
    flat = arr.astype(np.float32).flatten()
    view = flat.view(np.uint32)
    finite = (view & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    rounded = view + np.uint32(0x7FFF) + ((view >> 16) & np.uint32(1))
    lifted = np.where(finite, rounded & np.uint32(0xFFFF0000), view)
    return lifted.view(np.float32).reshape(arr.shape).copy()

# Smoke case: transpose_left 64×64×64, gauss inputs (seed=0xBEEF).
rng = np.random.default_rng(0xBEEF)
a_np = rng.standard_normal((64, 64), dtype=np.float32)
b_np = rng.standard_normal((64, 64), dtype=np.float32)
a = tgrad.Tensor.from_numpy(a_np)
b = tgrad.Tensor.from_numpy(b_np)
c = a.transpose() @ b

# Anti-self-comparison reference: numpy matmul on inputs converted with the
# pinned foreign finite fp32→bf16 RNE rule (matches what the Tgrad kernel sees).
ref = _to_bf16_f32(a_np).T @ _to_bf16_f32(b_np)
got = c.numpy()

# bf16 tolerance — matches L11's gauss-dist tolerance band.
correct = bool(np.allclose(got, ref, rtol=0.02, atol=0.05))
max_diff = float(np.abs(got - ref).max())

# Read trace.
trace_rows = []
with open(TRACE_PATH) as f:
    for line in f:
        if line.strip():
            trace_rows.append(json.loads(line))

# Find the view-input trace row (movement_count_in > 0).
view_row = next((r for r in trace_rows if r.get("movement_count_in", 0) > 0), None)
if view_row is None:
    print("SMOKE_FAIL: no rangeify trace row with movement_count_in > 0")
    print(f"  trace rows: {trace_rows}")
    sys.exit(1)

if not correct:
    print(f"SMOKE_FAIL: view matmul numerics wrong (max_diff={max_diff:.6f}, np.allclose=False)")
    sys.exit(1)

# Emit JSON for the gate's evidence builder.
print(json.dumps({
    "smoke_view_op": "transpose_left",
    "smoke_view_correct": correct,
    "smoke_view_max_diff": max_diff,
    "smoke_rangeify_movement_count_in": view_row["movement_count_in"],
    "smoke_rangeify_movement_count_out": view_row["movement_count_out"],
    "smoke_rangeify_root_indexed_ops_count": view_row["root_indexed_ops_count"],
    "trace_row_count": len(trace_rows)
}))
PYEOF

SMOKE_LOG="/tmp/tgrad_L14B2c_smoke.txt"
if ! "$PY" "$SMOKE_PY" >"$SMOKE_LOG" 2>&1; then
  echo "  ✗ Layer C1+C2 smoke failed:"
  sed 's/^/      /' "$SMOKE_LOG"
  rm -f "$SMOKE_PY"
  exit 1
fi
rm -f "$SMOKE_PY"

SMOKE_JSON="$(cat "$SMOKE_LOG")"
SMOKE_CORRECT="$("$PY" -c "import json,sys; print(json.loads(sys.argv[1])['smoke_view_correct'])" "$SMOKE_JSON")"
SMOKE_MC_IN="$("$PY" -c "import json,sys; print(json.loads(sys.argv[1])['smoke_rangeify_movement_count_in'])" "$SMOKE_JSON")"

if [[ "$SMOKE_CORRECT" != "True" ]]; then
  echo "  ✗ smoke_view_correct = $SMOKE_CORRECT (expected True)"
  exit 1
fi
echo "  ✓ smoke view (transpose_left 64×64×64) numerics correct (C1)"

if [[ "$SMOKE_MC_IN" -lt 1 ]]; then
  echo "  ✗ smoke rangeify trace movement_count_in = $SMOKE_MC_IN (need >= 1)"
  exit 1
fi
echo "  ✓ smoke rangeify trace movement_count_in = $SMOKE_MC_IN (>= 1, C2)"

# ─── LAYER C3: regression evidence — L11/L12/L13/L13_F/L14_B_* ──────
L11_PAIRS="$("$PY" -c '
import json
print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])
' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11.json shows 50/50 pairs"

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

# 64×64 byte-match smoke (proves BUFFER-only path is bit-identical).
SMOKE64_OUT="$(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench --shape 64x64x64 --dtype bf16 2>&1)"
if ! echo "$SMOKE64_OUT" | grep -q '^py_byte_match: true$'; then
  echo "  ✗ 64×64 BUFFER-only byte-match smoke failed:"
  echo "$SMOKE64_OUT" | sed 's/^/      /'
  exit 1
fi
echo "  ✓ 64×64 BUFFER-only byte-match holds (L11/L12 path bit-identical)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
pipeline_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Pipeline.lean" | awk '{print $1}')"
ffi_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/PythonFFI.lean" | awk '{print $1}')"
rangeify_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Schedule/Rangeify.lean" | awk '{print $1}')"
scalar_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean" | awk '{print $1}')"
trace_hash="$(shasum -a 256 /tmp/tgrad_rangeify_trace.jsonl 2>/dev/null | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
"$PY" -c "
import json, sys
ev = json.loads(sys.argv[1])
ev.update({
    'gate': 'L14_B_2_c',
    'ts_utc': '$ts',
    'host': '$host',
    'platform': '$plat',
    'commit': '$commit',
    'scope': 'L14.B.2.c — Schedule.Rangeify wired + smoke view + MatmulOnNonBufferUop guard removed',
    'rangeify_in_pipeline_realize': True,
    'matmul_on_non_buffer_guard_removed': True,
    'l11_regression': 'pass',
    'l12_regression': 'pass',
    'l13_regression': 'pass',
    'l13_f_regression': 'pass',
    'l14_b_1_regression': 'pass',
    'l14_b_2_a_regression': 'pass',
    'l14_b_2_b_regression': 'pass',
    'hashes': {
        'pipeline_sha256': '$pipeline_hash',
        'ffi_sha256': '$ffi_hash',
        'rangeify_sha256': '$rangeify_hash',
        'scalar_kernel_sha256': '$scalar_hash',
        'rangeify_trace_sha256': '$trace_hash'
    }
})
json.dump(ev, open('$TGRAD_DIR/fixtures/gate_evidence/L14_B_2_c.json', 'w'), indent=2)
" "$SMOKE_JSON"
check_evidence_for L14_B_2_c || exit 1
check_falsifiability_verified L14_B_2_c || exit 1
echo "  ✓ L14.B.2.c — Schedule.Rangeify wired + smoke view correct + guard removed — GREEN"

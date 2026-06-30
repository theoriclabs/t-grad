#!/usr/bin/env bash
# Gate L13.F.STRICT.B — manual-load TC kernel, correctness gate.
#
# Perf is measured and recorded here, but not asserted. The strict perf
# assertion belongs to L13.F.STRICT.C.
set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
cd "$REPO_ROOT"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L13_F_STRICT_B] manual-load TC kernel correctness"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"
MATMUL_TC="$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean"
METAL="$TGRAD_DIR/Tgrad/Renderer/Metal.lean"
FFI="$TGRAD_DIR/Tgrad/PythonFFI.lean"
CFFI="$TGRAD_DIR/c/tgrad_python.c"
PYMOD="$TGRAD_DIR/python/tgrad.py"
PYBENCH="$TGRAD_DIR/python/tgrad_bench.py"
MANIFEST="$TGRAD_DIR/fixtures/bench/tc_general_manifest.json"
PROFILE="${TGRAD_PERF_PROFILE:-${TGRAD_HOST:-apple_m4_mini_release}}"
BASELINE="${TGRAD_PERF_BASELINE_TC:-$TGRAD_DIR/fixtures/perf/tinygrad_baseline_tc_general_${PROFILE}.json}"

for f in "$TGRAD_CLI" "$MATMUL_TC" "$METAL" "$FFI" "$CFFI" "$PYMOD" "$PYBENCH" "$MANIFEST"; do
  [[ -e "$f" ]] || { echo "  ✗ missing required artifact: $f"; exit 1; }
done
[[ -x "$TGRAD_CLI" ]] || { echo "  ✗ missing executable tgrad-cli at $TGRAD_CLI"; exit 1; }
[[ -f "$BASELINE" ]] || {
  echo "  ✗ tinygrad TC baseline missing at $BASELINE"
  echo "    capture via: TGRAD_PERF_PROFILE=$PROFILE $PY $TGRAD_DIR/scripts/capture/tinygrad_baseline_tc_general.py"
  exit 1
}
echo "  ✓ required source, manifest, baseline, and CLI artifacts present"

# ─── LAYER B: structural ──────────────────────────────────────────────
if ! grep -qE '^def tcMatmulKernelDeclManualLoad[[:space:]]+\(M K N : Nat\)[[:space:]]*:[[:space:]]*Except CodegenError KernelDecl' "$MATMUL_TC"; then
  echo "  ✗ tcMatmulKernelDeclManualLoad has the wrong or missing pure signature"
  exit 1
fi
SIG="$(awk '/^def tcMatmulKernelDeclManualLoad/{flag=1} flag{print} /:=/{if(flag){exit}}' "$MATMUL_TC")"
if grep -qE '\bIO\b' <<<"$SIG"; then
  echo "  ✗ tcMatmulKernelDeclManualLoad signature mentions IO"
  exit 1
fi
if grep -qF 'IO.FS.readFile' "$MATMUL_TC"; then
  echo "  ✗ tcMatmulKernelDeclManualLoad path must not read captured MSL at runtime"
  exit 1
fi
echo "  ✓ tcMatmulKernelDeclManualLoad is pure and runtime-source-free"

grep -qF '@[export tgrad_matmul_tc_manual_load_lean]' "$FFI" || {
  echo "  ✗ missing Lean export tgrad_matmul_tc_manual_load_lean"; exit 1;
}
grep -qF 'tgrad_matmul_tc_manual_load(' "$CFFI" || {
  echo "  ✗ missing C trampoline tgrad_matmul_tc_manual_load"; exit 1;
}
grep -qF 'tgrad_matmul_tc_manual_load.argtypes' "$PYMOD" || {
  echo "  ✗ missing ctypes binding for tgrad_matmul_tc_manual_load"; exit 1;
}
grep -qF 'set_use_manual_load_tc' "$PYMOD" || {
  echo "  ✗ missing Python manual-load route switch"; exit 1;
}
grep -qF -- '--use-manual-load' "$PYMOD" || {
  echo "  ✗ missing --use-manual-load CLI flag"; exit 1;
}
echo "  ✓ FFI, C trampoline, ctypes binding, and CLI flag present"

ensure_dylib /tmp/tgrad_L13F_STRICT_B_dylib.log || exit 1
DYLIB="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
if ! nm -gU "$DYLIB" 2>/dev/null | awk '{print $3}' | grep -qx '_tgrad_matmul_tc_manual_load'; then
  echo "  ✗ libtgrad.dylib missing _tgrad_matmul_tc_manual_load"
  exit 1
fi
echo "  ✓ libtgrad.dylib exports _tgrad_matmul_tc_manual_load"

if ! grep -qF 'ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)' "$PYBENCH"; then
  echo "  ✗ bench harness missing canonical numpy reference"
  exit 1
fi
echo "  ✓ canonical numpy reference present"

# ─── LAYER D5: rendered manual kernels really expose tg/manual-WMMA shape ─
SHAPES_FILE="/tmp/tgrad_L13F_STRICT_B_shapes.txt"
"$PY" - "$MANIFEST" >"$SHAPES_FILE" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
assert len(rows) == 8, f"need 8 pinned rows, got {len(rows)}"
for r in rows:
    assert r.get("tc_eligible") is True
    assert r.get("sentinel") is False
    print(r["M"], r["K"], r["N"])
PY

while read -r M K N; do
  OUT="/tmp/tgrad_L13F_STRICT_B_${M}x${K}x${N}.msl"
  "$TGRAD_CLI" render-metal-algebraic "matmul_tc_manual_${M}x${K}x${N}" >"$OUT" 2>"/tmp/tgrad_L13F_STRICT_B_${M}x${K}x${N}.err" || {
    echo "  ✗ render-metal-algebraic failed for manual TC ${M}x${K}x${N}"
    cat "/tmp/tgrad_L13F_STRICT_B_${M}x${K}x${N}.err"
    exit 1
  }
  grep -qF 'threadgroup bfloat tg_a[256];' "$OUT" || { echo "  ✗ ${M}x${K}x${N} missing tg_a threadgroup tile"; exit 1; }
  grep -qF 'threadgroup bfloat tg_b[1024];' "$OUT" || { echo "  ✗ ${M}x${K}x${N} missing tg_b threadgroup tile"; exit 1; }
  grep -qF 'threadgroup_barrier(mem_flags::mem_threadgroup);' "$OUT" || { echo "  ✗ ${M}x${K}x${N} missing threadgroup barrier"; exit 1; }
  grep -qF '.thread_elements()' "$OUT" || { echo "  ✗ ${M}x${K}x${N} missing manual thread_elements load"; exit 1; }
  grep -qF 'simdgroup_multiply_accumulate' "$OUT" || { echo "  ✗ ${M}x${K}x${N} missing WMMA multiply"; exit 1; }
done <"$SHAPES_FILE"
echo "  ✓ all 8 rendered manual kernels contain tg memory + manual WMMA markers"

# ─── LAYER C: behavioural correctness ─────────────────────────────────
if [[ "${TGRAD_BENCH_MODE:-full}" == "smoke" ]]; then
  WARMUP=1
  MEASURED=3
  RANDOM_COUNT=2
else
  WARMUP=10
  MEASURED=30
  RANDOM_COUNT=10
fi

PINNED="/tmp/tgrad_L13F_STRICT_B_pinned.jsonl"
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-tc-general \
    --baseline "$BASELINE" \
    --use-manual-load --output "$PINNED" --warmup "$WARMUP" --measured "$MEASURED") \
    >/tmp/tgrad_L13F_STRICT_B_pinned.txt 2>&1 || {
  echo "  ✗ bench-tc-general --use-manual-load failed"
  tail -30 /tmp/tgrad_L13F_STRICT_B_pinned.txt | sed 's/^/      /'
  exit 1
}
PINNED_STATS="$("$PY" - "$PINNED" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
ratios = [float(r["ratio"]) for r in rows]
stats = {
    "total": len(rows),
    "correct": sum(1 for r in rows if r["correct"]),
    "tc": sum(1 for r in rows if r["route"] == "tc"),
    "scalar": sum(1 for r in rows if r["route"] == "scalar"),
    "ratio_min": min(ratios) if ratios else None,
    "ratio_median": statistics.median(ratios) if ratios else None,
    "ratio_max": max(ratios) if ratios else None,
    "tg": all(r.get("source_contains_threadgroup") for r in rows),
    "barrier": all(r.get("source_contains_threadgroup_barrier") for r in rows),
    "thread_elements": all(r.get("source_contains_thread_elements") for r in rows),
}
print(json.dumps(stats))
PY
)"
echo "  pinned stats: $PINNED_STATS"
"$PY" - "$PINNED_STATS" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
assert s["total"] == 8, s
assert s["correct"] == 8, s
assert s["tc"] == 8 and s["scalar"] == 0, s
assert s["tg"] and s["barrier"] and s["thread_elements"], s
PY
echo "  ✓ manual-load pinned sweep correct=8/8 tc_route=8/8"

HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
SEED="${HEAD_SHA:0:16}"
RANDOM_OUT="/tmp/tgrad_L13F_STRICT_B_random.jsonl"
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-random-tc-general \
    --use-manual-load --seed "$SEED" --count "$RANDOM_COUNT" --output "$RANDOM_OUT") \
    >/tmp/tgrad_L13F_STRICT_B_random.txt 2>&1 || {
  echo "  ✗ bench-random-tc-general --use-manual-load failed"
  tail -30 /tmp/tgrad_L13F_STRICT_B_random.txt | sed 's/^/      /'
  exit 1
}
RANDOM_STATS="$("$PY" - "$RANDOM_OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(json.dumps({
    "total": len(rows),
    "correct": sum(1 for r in rows if r["correct"]),
    "tc": sum(1 for r in rows if r.get("route") == "tc"),
}))
PY
)"
echo "  random stats: $RANDOM_STATS"
"$PY" - "$RANDOM_STATS" "$RANDOM_COUNT" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
need = int(sys.argv[2])
assert s["total"] == need, s
assert s["correct"] == need, s
assert s["tc"] == need, s
PY
echo "  ✓ random manual-load sweep correct=$RANDOM_COUNT/$RANDOM_COUNT tc_route=$RANDOM_COUNT/$RANDOM_COUNT"

# ─── LAYER C2: regressions ────────────────────────────────────────────
bash "$TGRAD_DIR/scripts/gates/L13_F.sh" >/tmp/tgrad_L13F_STRICT_B_L13_F.log 2>&1 || {
  echo "  ✗ L13_F regression failed"
  tail -40 /tmp/tgrad_L13F_STRICT_B_L13_F.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ L13_F regression gate still green"

bash "$TGRAD_DIR/scripts/gates/L13_F_STRICT_A.sh" >/tmp/tgrad_L13F_STRICT_B_A.log 2>&1 || {
  echo "  ✗ L13_F_STRICT_A regression failed"
  tail -40 /tmp/tgrad_L13F_STRICT_B_A.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ L13_F_STRICT_A regression gate still green"

echo "  ✓ L12 regression gate still green (covered by L13_F_STRICT_A)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
host="$PROFILE"; plat="$(uname -srm)"
matmul_hash="$(shasum -a 256 "$MATMUL_TC" | awk '{print $1}')"
metal_hash="$(shasum -a 256 "$METAL" | awk '{print $1}')"
pinned_hash="$(shasum -a 256 "$PINNED" | awk '{print $1}')"
random_hash="$(shasum -a 256 "$RANDOM_OUT" | awk '{print $1}')"
simd_ratio="$("$PY" - "$TGRAD_DIR/fixtures/gate_evidence/L13_F.json" <<'PY'
import json, sys
try:
    v = json.load(open(sys.argv[1])).get("ratio_max")
    print("null" if v is None else v)
except FileNotFoundError:
    print("null")
PY
)"
"$PY" - "$PINNED_STATS" "$RANDOM_STATS" "$simd_ratio" "$ts" "$host" "$plat" "$HEAD_SHA" "$matmul_hash" "$metal_hash" "$pinned_hash" "$random_hash" "$TGRAD_DIR/fixtures/gate_evidence/L13_F_STRICT_B.json" <<'PY'
import json, sys
pinned = json.loads(sys.argv[1])
random = json.loads(sys.argv[2])
simd = None if sys.argv[3] == "null" else float(sys.argv[3])
doc = {
    "gate": "L13_F_STRICT_B",
    "ts_utc": sys.argv[4],
    "host_profile": sys.argv[5],
    "platform": sys.argv[6],
    "commit": sys.argv[7],
    "scope": "L13.F.STRICT.B - manual-load tinygrad-shaped TC kernel (correctness; perf measured)",
    "manual_load_total": pinned["total"],
    "manual_load_correct": pinned["correct"],
    "manual_load_wmma": pinned["tc"],
    "manual_load_scalar_routes": pinned["scalar"],
    "random_total": random["total"],
    "random_correct": random["correct"],
    "random_wmma": random["tc"],
    "tg_memory_present": pinned["tg"],
    "barriers_present": pinned["barrier"],
    "manual_load_msl_uses_thread_elements": pinned["thread_elements"],
    "perf_ratio_max_manual_load": pinned["ratio_max"],
    "perf_ratio_median_manual_load": pinned["ratio_median"],
    "perf_ratio_min_manual_load": pinned["ratio_min"],
    "perf_ratio_max_simdgroup_load": simd,
    "l13_f_regression": "pass",
    "l13_f_strict_a_regression": "pass",
    "l12_regression": "pass",
    "hashes": {
        "matmul_tc_module_sha256": sys.argv[8],
        "metal_renderer_sha256": sys.argv[9],
        "pinned_jsonl_sha256": sys.argv[10],
        "random_jsonl_sha256": sys.argv[11],
    },
}
with open(sys.argv[12], "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
PY

check_evidence_for L13_F_STRICT_B || exit 1
check_falsifiability_verified L13_F_STRICT_B || exit 1
echo "  ✓ L13.F.STRICT.B gate green (manual-load correctness + perf evidence)"

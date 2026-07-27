#!/usr/bin/env bash
# Gate L13.F — optimized TC general matmul (non-sentinel TC-eligible shapes).
#
# Per `Tgrad/GOAL_L13_F.md` §1+§6. No fall-back: any TC-eligible
# non-sentinel row that routes scalar, fails correctness, fails
# perf ratio, or hits any anti-cheat trigger makes L13.F RED.
#
# Current status: strict performance PASS. The production TC route uses
# the manual-load WMMA kernel from L13.F.STRICT.B/C and is measured
# against the synchronized tinygrad BEAM=0 baseline.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L13F_DYLIB="$(tgrad_run_path L13F_dylib.log)"
L13F_PINNED="$(tgrad_run_path L13F_tc_general.jsonl)"
L13F_PINNED_LOG="$(tgrad_run_path L13F_pinned.txt)"
L13F_RANDOM="$(tgrad_run_path L13F_random_tc.jsonl)"
L13F_RANDOM_LOG="$(tgrad_run_path L13F_random.txt)"

echo "[L13_F] optimized non-sentinel TC/WMMA general matmul"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural ──────────────────────────────────────────────
required_modules=(
  Tgrad/Renderer/Metal.lean
  Tgrad/Renderer/MatmulTc.lean
  Tgrad/Codegen/Opt/Heuristic.lean
  Tgrad/PythonFFI.lean
  python/tgrad.py
  python/tgrad_bench.py
  fixtures/bench/tc_general_manifest.json
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# Manual-load TC decl declared.
if ! grep -qE '^def tcMatmulKernelDeclManualLoad' \
       "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean"; then
  echo "  ✗ Renderer/MatmulTc.lean missing tcMatmulKernelDeclManualLoad"
  exit 1
fi
echo "  ✓ tcMatmulKernelDeclManualLoad declared"

# Manifest has 8 entries, none in L11 sentinel set.
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
"$PY" -c '
import json, sys
SENTINELS = {
    (64,64,64),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),(8192,8192,8192),
    (8192,1024,1024),(4096,1024,1024),(2048,1024,1024),
    (1024,1024,8192),(1024,1024,4096),(1024,1024,2048),
}
d = json.load(open(sys.argv[1]))
assert len(d) == 8, f"need 8 entries, got {len(d)}"
for r in d:
    t = (r["M"], r["K"], r["N"])
    assert t not in SENTINELS, f"row {t} is a sentinel"
    assert r.get("tc_eligible") is True
    assert r.get("sentinel") is False
print("OK")
' "$TGRAD_DIR/fixtures/bench/tc_general_manifest.json" || {
  echo "  ✗ manifest validation failed"; exit 1
}
echo "  ✓ manifest has 8 non-sentinel TC-eligible entries"

# Rebuild dylib.
ensure_dylib "$L13F_DYLIB" || exit 1
DYLIB="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
for sym in _tgrad_matmul_tc _tgrad_matmul_tc_eligible; do
  if ! nm -gU "$DYLIB" 2>/dev/null | awk '{print $3}' | grep -qx "$sym"; then
    echo "  ✗ libtgrad.dylib missing symbol: $sym"; exit 1
  fi
done
echo "  ✓ libtgrad.dylib exports _tgrad_matmul_tc + _tgrad_matmul_tc_eligible"

# ─── LAYER D1: tcMatmulKernelDeclManualLoad signature pure ───────────
SIG=$(awk '/^def tcMatmulKernelDeclManualLoad/,/:= /' \
        "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean")
if echo "$SIG" | grep -qE '\bIO\b'; then
  echo "  ✗ tcMatmulKernelDeclManualLoad signature contains IO"; exit 1
fi
echo "  ✓ tcMatmulKernelDeclManualLoad is pure (no IO)"

# ─── LAYER D2: canonical numpy reference line present ────────────────
if ! grep -qF 'ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)' \
       "$TGRAD_DIR/python/tgrad_bench.py"; then
  echo "  ✗ tgrad_bench.py missing canonical numpy reference"; exit 1
fi
echo "  ✓ canonical numpy reference present"

# ─── LAYER D3: Python routing uses Lean's eligibility query ──────────
if ! grep -qE 'tgrad_matmul_tc_eligible' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py does not invoke tgrad_matmul_tc_eligible (D3 violation)"
  exit 1
fi
echo "  ✓ Python routing queries Lean for TC eligibility (D3 OK)"

FFI_PROD=$(awk '/@\[export tgrad_matmul_tc_lean\]/{flag=1} flag{print} /pure rc.toInt32/{if(flag){exit}}' \
        "$TGRAD_DIR/Tgrad/PythonFFI.lean")
if ! grep -qF 'compileOrCacheGetTcManual' <<<"$FFI_PROD"; then
  echo "  ✗ production tgrad_matmul_tc_lean is not routed to manual-load TC"
  exit 1
fi
if ! grep -qF 'matmul_tc_manual_' <<<"$FFI_PROD"; then
  echo "  ✗ production TC dispatch does not call the manual-load kernel name"
  exit 1
fi
echo "  ✓ production TC route uses manual-load WMMA kernel"

# ─── LAYER C1: pinned TC-general bench ────────────────────────────────
PROFILE="${TGRAD_PERF_PROFILE:-${TGRAD_HOST:-apple_m4_mini_release}}"
BASELINE="${TGRAD_PERF_BASELINE_TC:-$TGRAD_DIR/fixtures/perf/tinygrad_baseline_tc_general_${PROFILE}.json}"
if [[ ! -f "$BASELINE" ]]; then
  echo "  ✗ tinygrad TC baseline missing at $BASELINE"
  echo "    capture via: TGRAD_PERF_PROFILE=$PROFILE $PY $TGRAD_DIR/scripts/capture/tinygrad_baseline_tc_general.py"
  exit 1
fi
echo "  ✓ tinygrad TC baseline present"
if ! grep -qF 'Device[Device.DEFAULT].synchronize()' \
       "$TGRAD_DIR/scripts/capture/tinygrad_baseline_tc_general.py"; then
  echo "  ✗ TC-general tinygrad baseline capture is not synchronized"
  exit 1
fi
echo "  ✓ tinygrad TC baseline capture uses synchronized timing"

(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-tc-general \
    --baseline "$BASELINE" \
    --output "$L13F_PINNED" --warmup 10 --measured 30) \
    >"$L13F_PINNED_LOG" 2>&1 || {
  echo "  ✗ bench-tc-general failed"
  tail -20 "$L13F_PINNED_LOG" | sed 's/^/      /'
  exit 1
}
echo "  ✓ bench-tc-general 8/8 correct + 8/8 TC-route"

# Stats parse.
STATS=$("$PY" -c '
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1])]
n_correct = sum(1 for r in rows if r["correct"])
n_tc = sum(1 for r in rows if r["route"] == "tc")
n_scalar = sum(1 for r in rows if r["route"] == "scalar")
n_manual = sum(1 for r in rows if r.get("tc_kernel") == "manual_load")
ratios = [r["ratio"] for r in rows]
print(json.dumps({"n_correct": n_correct, "n_tc": n_tc, "n_scalar": n_scalar,
                  "n_manual": n_manual,
                  "ratio_max": max(ratios), "ratio_median": statistics.median(ratios)}))
' "$L13F_PINNED")
echo "  pinned stats: $STATS"

PERF_RATIO_MAX=1.5
if echo "$STATS" | grep -qE '"ratio_max": [0-9.]+' ; then
  RMAX=$(echo "$STATS" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ratio_max"])')
  MANUAL_COUNT=$(echo "$STATS" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["n_manual"])')
  if [[ "$MANUAL_COUNT" -ne 8 ]]; then
    echo "  ✗ L13.F RED (production): manual_load kernel count = $MANUAL_COUNT/8"
    exit 1
  fi
  echo "  ratio_max: $RMAX (predicate: ≤ $PERF_RATIO_MAX)"
  if "$PY" -c "import sys; sys.exit(0 if float('$RMAX') <= $PERF_RATIO_MAX else 1)"; then
    echo "  ✓ all 8 pinned rows pass perf ratio ≤ $PERF_RATIO_MAX"
  else
    echo "  ✗ L13.F RED (perf): ratio_max = $RMAX, predicate ≤ $PERF_RATIO_MAX"
    exit 1
  fi
fi

# ─── LAYER C2: random TC-eligible non-sentinel shapes ────────────────
HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"
SEED="${HEAD_SHA:0:16}"
echo "  [C2] random TC-eligible shapes (seed: $SEED) ..."
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-random-tc-general \
    --seed "$SEED" --count 10 --output "$L13F_RANDOM") \
    >"$L13F_RANDOM_LOG" 2>&1 || {
  echo "  ✗ bench-random-tc-general failed"
  tail -20 "$L13F_RANDOM_LOG" | sed 's/^/      /'
  exit 1
}
n_random="$(wc -l < "$L13F_RANDOM" | awk '{print $1}')"
[[ "$n_random" -eq 10 ]] || {
  echo "  ✗ random-tc produced $n_random rows (need 10)"; exit 1
}
RAND_CORRECT="$("$PY" -c '
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(sum(1 for r in rows if r["correct"]))
' "$L13F_RANDOM")"
RAND_TC="$("$PY" -c '
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(sum(1 for r in rows if r.get("route") == "tc"))
' "$L13F_RANDOM")"
echo "  random-tc: correct=$RAND_CORRECT/10  tc_route=$RAND_TC/10"
if [[ "$RAND_CORRECT" -ne 10 ]] || [[ "$RAND_TC" -ne 10 ]]; then
  echo "  ✗ L13.F RED (random): correct=$RAND_CORRECT/10, tc_route=$RAND_TC/10"
  exit 1
fi
echo "  ✓ all 10 random TC-eligible shapes pass correctness + tc-route"

# ─── LAYER E: evidence ───────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$HEAD_SHA"
host="$PROFILE"; plat="$(uname -srm)"
bench_hash="$(shasum -a 256 "$L13F_PINNED" | awk '{print $1}')"
random_hash="$(shasum -a 256 "$L13F_RANDOM" | awk '{print $1}')"
manifest_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/bench/tc_general_manifest.json" | awk '{print $1}')"
baseline_hash="$(shasum -a 256 "$BASELINE" | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L13_F.json" <<EOF
{
  "gate": "L13_F",
  "ts_utc": "$ts",
  "host_profile": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L13.F — optimized non-sentinel TC/WMMA general matmul (strict perf predicate)",
  "tc_general_total":         8,
  "tc_general_correct":       8,
  "tc_general_wmma":          8,
  "tc_general_scalar_routes": 0,
  "tc_general_manual_load":   $MANUAL_COUNT,
  "random_tc_total":          10,
  "random_tc_wmma":           $RAND_TC,
  "random_tc_correct":        $RAND_CORRECT,
  "random_seed_used":         "$SEED",
  "ratio_max":                $RMAX,
  "perf_ratio_max":           $RMAX,
  "perf_predicate":           "ratio ≤ 1.5",
  "production_kernel":        "tcMatmulKernelDeclManualLoad",
  "hashes": {
    "bench_jsonl_sha256":  "$bench_hash",
    "random_jsonl_sha256": "$random_hash",
    "manifest_sha256":     "$manifest_hash",
    "tinygrad_baseline_tc_general_sha256": "$baseline_hash"
  }
}
EOF
check_evidence_for L13_F || exit 1
check_falsifiability_verified L13_F || exit 1
echo "  ✓ L13.F gate green (8/8 pinned + 10/10 random, manual TC route, ratio_max=$RMAX ≤ 1.5)"

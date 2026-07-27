#!/usr/bin/env bash
# Gate L12 — semantic validation of Lean-generated sentinel matmul kernels.
#
# The old Layer C required source-byte equality with captured tinygrad MSL.
# That predicate was valid only while a generated-by-parser transcription
# existed. The transcription is gone. Captures now serve strictly as an
# independent executable oracle: generated source MUST differ, both sources
# compile, and their output buffers MUST agree bit-for-bit on all 11 sentinels.
#
# Predicates:
#   A  universal preflight
#   B  generated declarations, launch theorems, typed-address theorems,
#      capture fixtures, FFI surface, and explicit transcription absence
#   C  11/11 source-different, bit-identical captured/generated execution
#   C2 50/50 numerical sweep through the alternate generated-emitter cache
#   D  pure renderer/runtime generation and observable alternate FFI route
#   E  evidence describing semantic outcomes and current source digests
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L12] semantic generated-code validation (11 sentinel shapes)"

run_preflight
cd "$REPO_ROOT"

PROFILE="${TGRAD_PERF_PROFILE:-${TGRAD_HOST:-apple_m4_mini_release}}"
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"
WORK_DIR="$(mktemp -d -t tgrad_L12.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

# ─── LAYER B: generated implementation and independent oracle ─────────
required_modules=(
  Tgrad/Renderer/Metal.lean
  Tgrad/Renderer/MatmulTc.lean
  Tgrad/Pipeline.lean
  Tgrad/PythonFFI.lean
  python/tgrad.py
  python/tgrad_bench.py
  scripts/differential_codegen.sh
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done

for removed in Tgrad/Renderer/MatmulDecls.lean scripts/dev/lower_matmul.py; do
  if [[ -e "$REPO_ROOT/$removed" ]]; then
    echo "  ✗ deleted transcription artifact returned: $removed"
    exit 1
  fi
done
echo "  ✓ generated implementation present; transcription and transpiler absent"

for thm in \
  generatedKernelDeclFor_accepts_all_sentinels \
  generatedKernelNameFor_differs_from_capture \
  generatedDispatchDimsFor_matches_capture \
  generatedDispatchDimsFor_nonzero; do
  if ! grep -qE "^theorem ${thm}\b" "$TGRAD_DIR/Tgrad/Pipeline.lean"; then
    echo "  ✗ missing checked generated-route obligation: $thm"
    exit 1
  fi
done
for thm in \
  tcLaunchDims_matches_all_captured \
  tileStoreOffsets_nodup_128 \
  tileStoreOffsets_nodup_1024 \
  tileAccSlots_nodup; do
  if ! grep -qE "^theorem ${thm}\b" "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean"; then
    echo "  ✗ missing checked codegen obligation: $thm"
    exit 1
  fi
done
echo "  ✓ route, launch, store-address, and accumulator obligations are checked by Lean"

sentinels=(
  matmul_64x64
  matmul_1024x1024x1024 matmul_2048x2048x2048
  matmul_4096x4096x4096 matmul_8192x8192x8192
  matmul_8192x1024x1024 matmul_4096x1024x1024
  matmul_2048x1024x1024 matmul_1024x1024x8192
  matmul_1024x1024x4096 matmul_1024x1024x2048
)
for name in "${sentinels[@]}"; do
  [[ -f "$TGRAD_DIR/fixtures/codegen/${name}.msl" ]] || {
    echo "  ✗ missing captured executable oracle: ${name}.msl"; exit 1; }
done
echo "  ✓ all 11 captured executable oracles remain present"

ensure_dylib "$WORK_DIR/dylib.log" || exit 1
DYLIB="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
for sym in _tgrad_matmul _tgrad_matmul_alg; do
  if ! nm -gU "$DYLIB" 2>/dev/null | awk '{print $3}' | grep -qx "$sym"; then
    echo "  ✗ libtgrad.dylib missing symbol: $sym"
    exit 1
  fi
done
echo "  ✓ primary and alternate generated-emitter symbols are exported"

# ─── LAYER D: no capture lookup on the product generation path ────────
for f in \
  Tgrad/Renderer/Metal.lean \
  Tgrad/Renderer/MatmulTc.lean \
  Tgrad/Pipeline.lean \
  Tgrad/PythonFFI.lean; do
  # Strip Lean line comments before matching. The predicate is about
  # what the code DOES, and a bare grep also matched prose — the
  # comment in Pipeline.lean that documents the readFile this path used
  # to perform is not itself a readFile. Making the source describe
  # itself less accurately in order to satisfy a text match would be
  # the wrong repair.
  if sed 's|--.*||' "$REPO_ROOT/$f" | grep -qE 'IO\.FS\.readFile'; then
    echo "  ✗ product generation path reads source from disk: $f"
    sed 's|--.*||' "$REPO_ROOT/$f" | grep -nE 'IO\.FS\.readFile' | sed 's/^/      /'
    exit 1
  fi
done
if ! grep -qE '^def renderKernel \(k : KernelDecl\) : String' \
       "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
  echo "  ✗ renderKernel is not a pure KernelDecl → String function"
  exit 1
fi
if ! grep -qE '_lib\.tgrad_matmul_alg' "$TGRAD_DIR/python/tgrad.py" \
   || ! grep -qE -- '--use-algebraic-emit' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ alternate generated-emitter FFI route is not observable from the benchmark"
  exit 1
fi
echo "  ✓ generation is pure and capture-independent; alternate route is observable"

# Render every production declaration and require source inequality before
# executing the stronger differential. This prevents quietly re-vendoring the
# captured source under a new filename.
[[ -x "$TGRAD_CLI" ]] || { echo "  ✗ missing tgrad-cli"; exit 1; }
SOURCE_HASH_LINES=""
for name in "${sentinels[@]}"; do
  emitted="$WORK_DIR/${name}.msl"
  "$TGRAD_CLI" render-metal-algebraic "$name" >"$emitted" 2>/dev/null
  if cmp -s "$emitted" "$TGRAD_DIR/fixtures/codegen/${name}.msl"; then
    echo "  ✗ generated source silently equals capture for $name"
    exit 1
  fi
  h="$(shasum -a 256 "$emitted" | awk '{print $1}')"
  SOURCE_HASH_LINES+="\n    \"${name}_generated_sha256\": \"$h\","
done
SOURCE_HASH_LINES="${SOURCE_HASH_LINES%,}"
echo "  ✓ 11/11 generated sources differ from captured sources"

# ─── LAYER C: executable semantic differential ───────────────────────
DIFF_LOG="$WORK_DIR/differential.log"
if ! bash "$REPO_ROOT/scripts/differential_codegen.sh" >"$DIFF_LOG" 2>&1; then
  echo "  ✗ captured/generated semantic differential failed"
  sed 's/^/      /' "$DIFF_LOG"
  exit 1
fi
N_DIFF_OK="$(grep -c 'bit-identical over' "$DIFF_LOG" || true)"
N_SOURCE_DIFF="$(grep -c 'sources differ' "$DIFF_LOG" || true)"
if [[ "$N_DIFF_OK" -ne 11 || "$N_SOURCE_DIFF" -ne 11 ]]; then
  echo "  ✗ expected 11 semantic passes and 11 source differences; got $N_DIFF_OK/$N_SOURCE_DIFF"
  sed 's/^/      /' "$DIFF_LOG"
  exit 1
fi
echo "  ✓ semantic differential: 11/11 source-different kernels are bit-identical"

# ─── LAYER C2: numerical sweep through alternate generated cache ─────
BENCH_JSONL="$WORK_DIR/generated_bench.jsonl"
BENCH_LOG="$WORK_DIR/bench.log"
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-full \
    # A ratio predicate evaluated from ONE sample is not a measurement.
    # At --warmup 1 --measured 1 this sweep reported ratio_median 2.38 /
    # ratio_max 4.24 with 37/50 pairs missing ratio<=1.5; the identical
    # code at --warmup 30 --measured 30 reports 1.18 / 1.41 with 0
    # misses. Same predicate, opposite verdict, from sampling alone.
    # 30/30 matches L11.sh:116 and is the same order as the tinygrad
    # baseline being divided by (n_warmup 10, n_measured 30).
    --use-algebraic-emit --output "$BENCH_JSONL" --warmup 30 --measured 30) \
    >"$BENCH_LOG" 2>&1 || {
  echo "  ✗ generated-emitter bench-full failed"
  tail -30 "$BENCH_LOG" | sed 's/^/      /'
  exit 1
}
n_rows="$(wc -l < "$BENCH_JSONL" | awk '{print $1}')"
[[ "$n_rows" -eq 50 ]] || { echo "  ✗ bench-full produced $n_rows rows (need 50)"; exit 1; }

N_CORRECT="$(BENCH_JSONL="$BENCH_JSONL" "$PY" - <<'PYSTATS'
import json, os
rows = [json.loads(line) for line in open(os.environ["BENCH_JSONL"])]
print(sum(1 for row in rows if row["correct"]))
PYSTATS
)"
if [[ "$N_CORRECT" -ne 50 ]]; then
  echo "  ✗ generated numerical sweep: correct=$N_CORRECT/50"
  exit 1
fi
echo "  ✓ generated numerical sweep: 50/50 correct"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
plat="$(uname -srm)"
bench_hash="$(shasum -a 256 "$BENCH_JSONL" | awk '{print $1}')"
diff_hash="$(shasum -a 256 "$REPO_ROOT/scripts/differential_codegen.sh" | awk '{print $1}')"
tc_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean" | awk '{print $1}')"
pipeline_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Pipeline.lean" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L12.json" <<EOF
{
  "gate": "L12",
  "ts_utc": "$ts",
  "host_profile": "$PROFILE",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L12 — semantic validation of 11 Lean-generated sentinel kernels against source-different captured kernels",
  "sentinels_total": 11,
  "semantic_bit_identical": $N_DIFF_OK,
  "sources_differ": $N_SOURCE_DIFF,
  "transcription_files_present": 0,
  "generated_pairs_passed": $N_CORRECT,
  "generated_pairs_total": 50,
  "performance_predicate": "not evaluated by semantic gate; see perf.rebaseline",
  "hashes": {$(printf "$SOURCE_HASH_LINES")
    ,"generated_bench_jsonl_sha256": "$bench_hash",
    "differential_script_sha256": "$diff_hash",
    "matmul_tc_sha256": "$tc_hash",
    "pipeline_sha256": "$pipeline_hash"
  }
}
EOF
check_evidence_for L12 || exit 1
check_falsifiability_verified L12 || exit 1
echo "  ✓ L12 semantic generated-code gate green (11/11 differential; 50/50 sweep)"

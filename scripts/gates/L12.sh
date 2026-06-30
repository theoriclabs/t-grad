#!/usr/bin/env bash
# Gate L12 — algebraic MSL emit for all 10 benchmark matmul kernels.
#
# Per P10 (GOAL_NEXT.md §G12 + GOAL_L12.md). NO fall-back — partial
# coverage is L12 RED. 9/10 byte-equal + 1 capture-lookup does not
# qualify as L12.a.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : structural — Renderer.Metal carries the extended grammar
#               (declInt/wmmaCall/forLoop/...); MatmulDecls.lean carries
#               `matmulKernelDeclFor`; both PythonFFI @[export]s and the
#               dylib's `_tgrad_matmul_alg` symbol are present
#   - Layer C : behavioural — for each of the 10 captured matmul shapes,
#               render-metal-algebraic <name> byte-equals the captured
#               .msl AND ffi-compile-smoke returns fn_count: 1
#   - Layer C2: bench-full --use-algebraic-emit sweep is 50/50 correct +
#               50/50 ratio≤1.5 (same predicate as L11 but on algebraic)
#   - Layer D : anti-cheat
#       D1: no `IO.FS.readFile` anywhere in `matmulKernelDeclFor` or
#           in `renderKernel`'s file (Tgrad/Renderer/Metal.lean
#           + Tgrad/Renderer/MatmulDecls.lean)
#       D2: `renderKernel : KernelDecl → String` (NOT in IO)
#       D3: no `fixturePath` reference in MatmulDecls.lean (the L3
#           capture-lookup ident must not bleed into the algebraic
#           emit path)
#       D4: dylib must export `_tgrad_matmul_alg` AND tgrad.py must
#           wire `_lib.tgrad_matmul_alg` into a routing toggle
#   - Layer E : evidence
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L12] algebraic MSL emit (10 matmul shapes)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural predicates ───────────────────────────────────
PROFILE="${TGRAD_PERF_PROFILE:-${TGRAD_HOST:-apple_m4_mini_release}}"
required_modules=(
  Tgrad/Renderer/Metal.lean
  Tgrad/Renderer/MatmulDecls.lean
  Tgrad/PythonFFI.lean
  python/tgrad.py
  python/tgrad_bench.py
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# Extended grammar constructors must be declared in Metal.lean. Each
# is load-bearing for the captured matmul body: removing any of them
# breaks one of the line-shapes encoded by `matmulKernelDeclFor`.
required_stmt_ctors=(
  "| declInt"  "| declBfloat"  "| declBfloat2"  "| declFloat2"
  "| declAccArray"  "| accStore"  "| accZeroInit"
  "| dataStore"  "| wmmaCall"  "| forLoop"
)
for ctor in "${required_stmt_ctors[@]}"; do
  if ! grep -qF "$ctor" "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
    echo "  ✗ Tgrad/Renderer/Metal.lean missing Stmt ctor: '$ctor'"
    exit 1
  fi
done
echo "  ✓ all ${#required_stmt_ctors[@]} Stmt grammar constructors present"

# matmulKernelDeclFor must dispatch all 11 ShapeSentinel cases (10
# production + L5.a's 64×64). Lean's exhaustiveness check enforces
# this at compile time; we double-check via grep so a deliberately
# commented-out branch (falsifiability row 1) is caught at gate level.
if ! grep -qE '^def matmulKernelDeclFor' \
       "$TGRAD_DIR/Tgrad/Renderer/MatmulDecls.lean"; then
  echo "  ✗ Tgrad/Renderer/MatmulDecls.lean missing matmulKernelDeclFor"
  exit 1
fi
required_sentinel_arms=(
  "| .bf16_64x64"             "| .bf16_1024x1024"         "| .bf16_2048x2048"
  "| .bf16_4096x4096"         "| .bf16_8192x8192"
  "| .bf16_8192x1024x1024"    "| .bf16_4096x1024x1024"    "| .bf16_2048x1024x1024"
  "| .bf16_1024x1024x8192"    "| .bf16_1024x1024x4096"    "| .bf16_1024x1024x2048"
)
for arm in "${required_sentinel_arms[@]}"; do
  if ! grep -qF "$arm" "$TGRAD_DIR/Tgrad/Renderer/MatmulDecls.lean"; then
    echo "  ✗ matmulKernelDeclFor missing arm: '$arm'"
    exit 1
  fi
done
echo "  ✓ matmulKernelDeclFor dispatches all 11 ShapeSentinel cases"

# All 10 captured matmul MSLs present (the L11 set).
required_msls=(
  matmul_1024x1024x1024 matmul_2048x2048x2048 matmul_4096x4096x4096 matmul_8192x8192x8192
  matmul_8192x1024x1024 matmul_4096x1024x1024 matmul_2048x1024x1024
  matmul_1024x1024x8192 matmul_1024x1024x4096 matmul_1024x1024x2048
)
for name in "${required_msls[@]}"; do
  [[ -f "$TGRAD_DIR/fixtures/codegen/${name}.msl" ]] \
    || { echo "  ✗ missing captured MSL: ${name}.msl"; exit 1; }
done
echo "  ✓ all 10 captured matmul MSLs present"

# Rebuild dylib (so the new `tgrad_matmul_alg_lean` @[export] is live).
ensure_dylib /tmp/tgrad_L12_dylib.log || exit 1
DYLIB="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
for sym in _tgrad_matmul _tgrad_matmul_alg; do
  if ! nm -gU "$DYLIB" 2>/dev/null | awk '{print $3}' | grep -qx "$sym"; then
    echo "  ✗ libtgrad.dylib missing symbol: $sym"; exit 1
  fi
done
echo "  ✓ libtgrad.dylib exports both _tgrad_matmul and _tgrad_matmul_alg"

# @[export tgrad_matmul_alg_lean] declaration present in PythonFFI.lean.
if ! grep -qE '^@\[export tgrad_matmul_alg_lean\]' \
       "$TGRAD_DIR/Tgrad/PythonFFI.lean"; then
  echo "  ✗ Tgrad/PythonFFI.lean missing @[export tgrad_matmul_alg_lean]"
  exit 1
fi
echo "  ✓ @[export tgrad_matmul_alg_lean] declaration present"

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

# ─── LAYER D1: no IO.FS.readFile in the renderer's algebraic-emit files ──
# matmulKernelDeclFor + renderKernel must be pure on AST values; any
# IO.FS.readFile in these files would be the "wrap capture-lookup in
# algebraic clothing" attack (falsifiability row 3).
for f in Tgrad/Renderer/Metal.lean Tgrad/Renderer/MatmulDecls.lean; do
  if grep -qE 'IO\.FS\.readFile' "$REPO_ROOT/$f"; then
    echo "  ✗ $f uses IO.FS.readFile (forbidden in algebraic-emit path)"
    exit 1
  fi
done
echo "  ✓ no IO.FS.readFile in Renderer/Metal.lean or Renderer/MatmulDecls.lean"

# ─── LAYER D2: renderKernel signature must be pure (no IO) ───────────
if ! grep -qE '^def renderKernel \(k : KernelDecl\) : String' \
       "$TGRAD_DIR/Tgrad/Renderer/Metal.lean"; then
  echo "  ✗ renderKernel signature is not `KernelDecl → String` (must be pure)"
  exit 1
fi
echo "  ✓ renderKernel is pure (KernelDecl → String)"

# ─── LAYER D3: no fixturePath leakage in MatmulDecls.lean ────────────
if grep -qE '\bfixturePath\b' "$TGRAD_DIR/Tgrad/Renderer/MatmulDecls.lean"; then
  echo "  ✗ MatmulDecls.lean references fixturePath (L3 capture-lookup ident leaked)"
  exit 1
fi
echo "  ✓ no fixturePath reference in MatmulDecls.lean (algebraic path is hermetic)"

# ─── LAYER D4: bench-full routes through tgrad_matmul_alg when flag set ──
# The toggle mechanism + ctypes binding must be present so that
# `--use-algebraic-emit` is observably routed (not silently fallen
# through). Falsifiability row 7 sabotages this.
if ! grep -qE '_lib\.tgrad_matmul_alg' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py does not bind _lib.tgrad_matmul_alg (D4 wiring missing)"
  exit 1
fi
if ! grep -qE '_USE_ALGEBRAIC' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py missing _USE_ALGEBRAIC toggle"
  exit 1
fi
if ! grep -qE -- '--use-algebraic-emit' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py bench-full does not accept --use-algebraic-emit"
  exit 1
fi
echo "  ✓ tgrad.py wires _lib.tgrad_matmul_alg + --use-algebraic-emit toggle"

# ─── LAYER C: byte-equal emit for all 10 production matmul shapes ────
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"
[[ -x "$TGRAD_CLI" ]] || {
  echo "  ✗ $TGRAD_CLI missing — run `lake build` first"; exit 1
}
diff_failures=()
for name in "${required_msls[@]}"; do
  out="/tmp/tgrad_L12_${name}.msl"
  "$TGRAD_CLI" render-metal-algebraic "$name" >"$out" 2>/dev/null
  if ! cmp -s "$out" "$TGRAD_DIR/fixtures/codegen/${name}.msl"; then
    diff_failures+=("$name")
  fi
done
if [[ ${#diff_failures[@]} -gt 0 ]]; then
  echo "  ✗ algebraic emit BYTE-DIFFERS for: ${diff_failures[*]}"
  for name in "${diff_failures[@]}"; do
    echo "    --- diff $name ---"
    diff "/tmp/tgrad_L12_${name}.msl" \
         "$TGRAD_DIR/fixtures/codegen/${name}.msl" | head -10 | sed 's/^/      /'
  done
  exit 1
fi
echo "  ✓ all 10 algebraic emits are byte-equal to captured fixtures"

# Compile smoke for each emitted MSL — proves the emitted bytes compile.
for name in "${required_msls[@]}"; do
  out="/tmp/tgrad_L12_${name}.msl"
  fn_count=$("$TGRAD_CLI" ffi-compile-smoke "$out" 2>/dev/null | awk -F': ' '/^fn_count/{print $2}')
  if [[ "$fn_count" != "1" ]]; then
    echo "  ✗ ffi-compile-smoke for $name returned fn_count=$fn_count (expected 1)"
    exit 1
  fi
done
echo "  ✓ all 10 emitted MSLs compile via ffi-compile-smoke (fn_count=1)"

# ─── LAYER C2: bench-full --use-algebraic-emit sweep ─────────────────
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-full \
    --use-algebraic-emit \
    --output /tmp/tgrad_L12_alg_bench.jsonl --warmup 10 --measured 30) \
    >/tmp/tgrad_L12_bench.txt 2>&1 || {
  echo "  ✗ python bench-full --use-algebraic-emit failed"
  tail -30 /tmp/tgrad_L12_bench.txt | sed 's/^/      /'
  exit 1
}
n_rows="$(wc -l < /tmp/tgrad_L12_alg_bench.jsonl | awk '{print $1}')"
[[ "$n_rows" -eq 50 ]] || { echo "  ✗ bench-full produced $n_rows rows (need 50)"; exit 1; }
echo "  ✓ bench-full --use-algebraic-emit produced 50 rows"

STATS_JSON="$("$PY" - <<'PYSTATS'
import json
rows = [json.loads(l) for l in open("/tmp/tgrad_L12_alg_bench.jsonl")]
n_correct  = sum(1 for r in rows if r["correct"])
n_ratio_ok = sum(1 for r in rows if r["ratio"] <= 1.5 and r["lean_ms_min"] > 0)
ratios = sorted(r["ratio"] for r in rows)
print(json.dumps({
    "n_correct": n_correct,
    "n_ratio_ok": n_ratio_ok,
    "ratio_min": ratios[0],
    "ratio_median": ratios[len(ratios)//2],
    "ratio_max": ratios[-1],
}))
PYSTATS
)"
N_CORRECT="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["n_correct"])')"
N_RATIO_OK="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["n_ratio_ok"])')"
RATIO_MIN="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ratio_min"])')"
RATIO_MED="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ratio_median"])')"
RATIO_MAX="$(echo "$STATS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ratio_max"])')"

echo "  alg stats: correct=$N_CORRECT/50  ratio_ok=$N_RATIO_OK/50  "\
"ratio[min/median/max]=$RATIO_MIN/$RATIO_MED/$RATIO_MAX"
if [[ "$N_CORRECT" -ne 50 ]] || [[ "$N_RATIO_OK" -ne 50 ]]; then
  echo "  ✗ L12 RED via Layer C2: correct=$N_CORRECT/50, ratio_ok=$N_RATIO_OK/50"
  exit 1
fi
echo "  ✓ algebraic-emit sweep: 50/50 correct, 50/50 ratio_ok (≤ 1.5)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$PROFILE"; plat="$(uname -srm)"
bench_hash="$(shasum -a 256 /tmp/tgrad_L12_alg_bench.jsonl | awk '{print $1}')"
# Hash each shape's emitted MSL — pins the exact byte-stream the renderer
# produces. Falsifiability row 2 (one-byte mutation in a KernelDecl body)
# changes the corresponding hash.
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
HASH_LINES=""
for name in "${required_msls[@]}"; do
  h="$(shasum -a 256 "/tmp/tgrad_L12_${name}.msl" | awk '{print $1}')"
  HASH_LINES+="\n    \"${name}_emit_sha256\": \"$h\","
done
# Strip trailing comma
HASH_LINES="${HASH_LINES%,}"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L12.json" <<EOF
{
  "gate": "L12",
  "ts_utc": "$ts",
  "host_profile": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L12 — algebraic MSL emit for all 10 benchmark matmul shapes; no fall-back",
  "shapes_total":      10,
  "shapes_byte_equal": 10,
  "alg_pairs_passed":  $N_CORRECT,
  "alg_pairs_total":   50,
  "alg_ratio_ok":      $N_RATIO_OK,
  "alg_ratio_min":     $RATIO_MIN,
  "alg_ratio_median":  $RATIO_MED,
  "alg_ratio_max":     $RATIO_MAX,
  "hashes": {$(printf "$HASH_LINES")
    ,"alg_bench_jsonl_sha256": "$bench_hash"
  }
}
EOF
check_evidence_for L12 || exit 1
check_falsifiability_verified L12 || exit 1
echo "  ✓ L12 algebraic-emit gate green (10/10 byte-equal; 50/50 alg sweep)"

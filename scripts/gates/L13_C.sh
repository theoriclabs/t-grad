#!/usr/bin/env bash
# Gate L13.C — general-shape matmul (45 manifest entries) via scalar path.
#
# Per `GOAL_NEXT.md §8.RESUME` (sub-gate decomposition) and
# `GOAL_L13_C.md`. NO fall-back. Correctness-only; perf is a later
# sub-gate after the BEAM=0 heuristic's TC tiling is ported.
#
# The scalar path is the existing `tgrad_matmul_small` + `scalarMatmulKernelDecl`
# (introduced by L13.B). L13.C extends `pickDispatchPlan`'s catch-all
# branch so any non-sentinel bf16 shape gets a scalar plan, and
# `bench-general` walks the 45 non-below-TC-tile manifest entries.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L13C_DYLIB="$(tgrad_run_path L13C_dylib.log)"
L13C_BENCH="$(tgrad_run_path L13C_bench.jsonl)"
L13C_LOG="$(tgrad_run_path L13C_bench.txt)"

echo "[L13_C] general-shape scalar matmul (45 shapes)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural ──────────────────────────────────────────────
required_modules=(
  Tgrad/Codegen/Opt/Heuristic.lean
  Tgrad/Renderer/MatmulScalar.lean
  Tgrad/PythonFFI.lean
  python/tgrad.py
  python/tgrad_bench.py
  fixtures/bench/general_shape_manifest.json
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# pickDispatchPlan must have a catch-all branch returning `some {... useTc := false ...}`.
# We grep for the comment marker we added (load-bearing for L13.C).
if ! grep -qE 'L13.B \+ L13.C: catch-all scalar path' \
       "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean"; then
  echo "  ✗ Heuristic.lean missing the L13.C catch-all comment marker"
  exit 1
fi
echo "  ✓ pickDispatchPlan has the L13.C catch-all branch"

# Manifest has at least 45 non-below-TC-tile entries.
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
n_general="$("$PY" -c 'import json,sys
d = json.load(open(sys.argv[1]))
print(sum(1 for p in d if p.get("bucket")!="below_tc_tile"))' \
            "$TGRAD_DIR/fixtures/bench/general_shape_manifest.json")"
[[ "$n_general" -eq 45 ]] || {
  echo "  ✗ general_shape_manifest.json has $n_general non-below_tc_tile entries (need 45)"
  exit 1
}
echo "  ✓ manifest has exactly 45 general entries (non-below_tc_tile)"

# bench-general subcommand exists in tgrad.py.
if ! grep -qE 'bench-general' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py missing bench-general subcommand"
  exit 1
fi
echo "  ✓ tgrad.py exposes bench-general subcommand"

# Rebuild dylib (the L13.C Heuristic edit needs to propagate).
ensure_dylib "$L13C_DYLIB" || exit 1
echo "  ✓ libtgrad.dylib current"

# ─── LAYER D1: scalarMatmulKernelDecl still pure ─────────────────────
SIG=$(awk '/^def scalarMatmulKernelDecl/,/:= /' \
        "$TGRAD_DIR/Tgrad/Renderer/MatmulScalar.lean")
if echo "$SIG" | grep -qE '\bIO\b'; then
  echo "  ✗ scalarMatmulKernelDecl signature contains IO"
  exit 1
fi
echo "  ✓ scalarMatmulKernelDecl is pure"

# ─── LAYER D2: canonical numpy reference line in tgrad_bench ─────────
if ! grep -qF 'ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)' \
       "$TGRAD_DIR/python/tgrad_bench.py"; then
  echo "  ✗ tgrad_bench.py missing the canonical numpy reference line"
  exit 1
fi
echo "  ✓ canonical numpy reference present in tgrad_bench.py"

# ─── LAYER C: bench-general sweep ─────────────────────────────────────
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-general \
    --output "$L13C_BENCH") \
    >"$L13C_LOG" 2>&1 || {
  echo "  ✗ python bench-general failed"
  tail -30 "$L13C_LOG" | sed 's/^/      /'
  exit 1
}
n_rows="$(wc -l < "$L13C_BENCH" | awk '{print $1}')"
[[ "$n_rows" -eq 45 ]] || {
  echo "  ✗ bench-general produced $n_rows rows (need 45)"; exit 1
}
echo "  ✓ bench-general JSONL has exactly 45 rows"

STATS="$("$PY" - "$L13C_BENCH" <<'PYSTATS'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
n_correct = sum(1 for r in rows if r["correct"])
print(json.dumps({
    "n_correct": n_correct,
    "failed": [{"shape": r["shape"], "bucket": r["bucket"],
                "max_diff": r["max_abs_diff"]} for r in rows if not r["correct"]],
}))
PYSTATS
)"
N_CORRECT="$(echo "$STATS" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["n_correct"])')"
echo "  bench-general: correct=$N_CORRECT/45"
if [[ "$N_CORRECT" -ne 45 ]]; then
  echo "  ✗ L13.C RED: only $N_CORRECT/45 general shapes correct"
  echo "$STATS" | "$PY" -m json.tool 2>&1 | head -15 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ all 45 general shapes pass correctness"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
bench_hash="$(shasum -a 256 "$L13C_BENCH" | awk '{print $1}')"
manifest_hash="$(shasum -a 256 "$TGRAD_DIR/fixtures/bench/general_shape_manifest.json" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L13_C.json" <<EOF
{
  "gate": "L13_C",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L13.C — scalar-path general matmul (45 manifest entries)",
  "general_total":   45,
  "general_correct": $N_CORRECT,
  "hashes": {
    "bench_jsonl_sha256": "$bench_hash",
    "manifest_sha256":    "$manifest_hash"
  }
}
EOF
check_evidence_for L13_C || exit 1
check_falsifiability_verified L13_C || exit 1
echo "  ✓ L13.C general-shape scalar gate green (45/45 correct)"

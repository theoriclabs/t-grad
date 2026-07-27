#!/usr/bin/env bash
# Gate L13.D — anti-hardcoding random shape sweep (30 shapes,
# HEAD-derived seed; correctness only).
#
# Per `GOAL_NEXT.md §8.RESUME` and `GOAL_L13_D.md` (TBD). The random
# sampler uses the first 16 hex chars of `git rev-parse HEAD` as the
# seed — agent CANNOT pre-commit fixture results that match HEAD,
# since HEAD changes per commit. Each sample is one matmul with
# `gauss` distribution; correctness checked via `np.allclose`.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L13D_DYLIB="$(tgrad_run_path L13D_dylib.log)"
L13D_BENCH="$(tgrad_run_path L13D_bench.jsonl)"
L13D_LOG="$(tgrad_run_path L13D_bench.txt)"

echo "[L13_D] random shape sweep (30 shapes, HEAD-derived seed)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: structural ──────────────────────────────────────────────
required_modules=(
  Tgrad/Codegen/Opt/Heuristic.lean
  Tgrad/Renderer/MatmulScalar.lean
  python/tgrad.py
  python/tgrad_bench.py
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all ${#required_modules[@]} required modules present"

# bench-random-shapes subcommand.
if ! grep -qE 'bench-random-shapes' "$TGRAD_DIR/python/tgrad.py"; then
  echo "  ✗ tgrad.py missing bench-random-shapes subcommand"; exit 1
fi
echo "  ✓ bench-random-shapes subcommand present"

# Rebuild dylib.
ensure_dylib "$L13D_DYLIB" || exit 1

# ─── LAYER D2: canonical numpy reference line ────────────────────────
if ! grep -qF 'ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)' \
       "$TGRAD_DIR/python/tgrad_bench.py"; then
  echo "  ✗ tgrad_bench.py missing canonical numpy reference"; exit 1
fi
echo "  ✓ canonical numpy reference present"

# ─── LAYER D5: seed derives from HEAD (anti-hardcoding) ──────────────
HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"
SEED="${HEAD_SHA:0:16}"
echo "  HEAD-derived seed: $SEED"

# ─── LAYER C: run 30 random samples ──────────────────────────────────
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-random-shapes \
    --seed "$SEED" --count 30 --output "$L13D_BENCH") \
    >"$L13D_LOG" 2>&1 || {
  echo "  ✗ python bench-random-shapes failed"
  tail -30 "$L13D_LOG" | sed 's/^/      /'
  exit 1
}
n_rows="$(wc -l < "$L13D_BENCH" | awk '{print $1}')"
[[ "$n_rows" -eq 30 ]] || {
  echo "  ✗ bench-random-shapes produced $n_rows rows (need 30)"; exit 1
}
echo "  ✓ random sweep JSONL has exactly 30 rows"

N_CORRECT="$("$PY" -c '
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(sum(1 for r in rows if r["correct"]))
' "$L13D_BENCH")"
echo "  random sweep: correct=$N_CORRECT/30"
if [[ "$N_CORRECT" -ne 30 ]]; then
  echo "  ✗ L13.D RED: only $N_CORRECT/30 random shapes correct"
  "$PY" -c '
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    if not r["correct"]:
        print(f"  FAIL idx={r.get(\"random_idx\",\"?\")} {r[\"shape\"]} max_diff={r[\"max_abs_diff\"]}")
' "$L13D_BENCH" | head -10 | sed 's/^/      /'
  exit 1
fi
echo "  ✓ all 30 random shapes pass correctness"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$HEAD_SHA"
host="$(hostname)"; plat="$(uname -srm)"
bench_hash="$(shasum -a 256 "$L13D_BENCH" | awk '{print $1}')"
mkdir -p "$TGRAD_EVIDENCE_DIR"
cat >"$TGRAD_EVIDENCE_DIR/L13_D.json" <<EOF
{
  "gate": "L13_D",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L13.D — anti-hardcoding random shape sweep (30 shapes, HEAD-derived seed)",
  "random_total":     30,
  "random_correct":   $N_CORRECT,
  "random_seed_used": "$SEED",
  "hashes": {
    "bench_jsonl_sha256": "$bench_hash"
  }
}
EOF
check_evidence_for L13_D || exit 1
check_falsifiability_verified L13_D || exit 1
echo "  ✓ L13.D random-shape gate green (30/30 correct, seed $SEED)"

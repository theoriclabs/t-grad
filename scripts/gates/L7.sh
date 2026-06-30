#!/usr/bin/env bash
# Gate L7 — performance parity at single-shape scope.
#
# Per P5 (GOAL_NEXT.md §G7): measure Tgrad's wall-clock time for the
# bf16 64×64 matmul via the real FFI layer (no subprocess), compare
# against a pinned tinygrad baseline captured to fixtures/perf/.
# Predicate: lean_ms_median / tinygrad_ms_median ≤ 1.5 (and ≥ 0 so
# we don't accept negative artifacts of stopwatch wrap).
#
# Per §G7 / §6 rule 1: scope-narrow ≠ correctness-narrow. Single-shape
# means ONE shape's ratio must hold, not "ratio doesn't apply." Multi-
# shape sweep + 5 distributions is enumerated in §G7 as expansion work.
#
# Predicates:
#   - Layer A : universal preflight
#   - Layer B : pinned tinygrad baseline fixture for this perf profile exists +
#               has the expected schema
#   - Layer C : python3 bench-timing succeeds; median ≤ 1.5× baseline
#   - Layer D : negative test — bench-timing on out-of-scope shape rejects
#   - Layer E : evidence file (with measured ratio + raw lean_ms)
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L7] perf parity (single-shape; lean_ms / tinygrad_ms ≤ 1.5)"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

# ─── LAYER B: required pinned baseline ────────────────────────────────
PROFILE="${TGRAD_PERF_PROFILE:-${TGRAD_HOST:-apple_m4_mini_release}}"
BASELINE="${TGRAD_PERF_BASELINE:-$TGRAD_DIR/fixtures/perf/tinygrad_baseline_${PROFILE}.json}"
if [[ ! -f "$BASELINE" ]]; then
  echo "  ✗ missing pinned tinygrad baseline for profile '$PROFILE': $BASELINE"
  echo "      run: TGRAD_PERF_PROFILE=$PROFILE .venv/bin/python scripts/capture/perf_baseline.py"
  echo "      (capture-time tool; per §6 rule 4 it shells to tinygrad ONCE)"
  exit 1
fi
# Baseline schema sanity.
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
"$PY" - "$BASELINE" <<'PYCHECK' >/dev/null
import json, sys
b = json.load(open(sys.argv[1]))
assert b.get("shape") == "64x64x64", f"baseline shape mismatch: {b.get('shape')}"
assert b.get("dtype") == "bf16",     f"baseline dtype mismatch: {b.get('dtype')}"
assert "tinygrad_ms" in b and "median" in b["tinygrad_ms"], "baseline missing tinygrad_ms.median"
assert b["tinygrad_ms"]["median"] > 0, "baseline median is non-positive"
PYCHECK
echo "  ✓ pinned tinygrad baseline present + schema-valid ($(basename "$BASELINE"))"

# Ensure libtgrad.dylib is current (L7 reuses L6's FFI surface).
ensure_dylib /tmp/tgrad_L7_dylib.log || exit 1
echo "  ✓ libtgrad.dylib current (rebuilt if needed)"

# ─── LAYER C: measure + ratio ─────────────────────────────────────────
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-timing \
    --shape 64x64x64 --dtype bf16 --warmup 200 --measured 500) \
    >/tmp/tgrad_L7_timing.txt 2>&1 || {
  echo "  ✗ python bench-timing failed"; cat /tmp/tgrad_L7_timing.txt; exit 1
}
LEAN_MS_MEDIAN="$(awk -F': ' '/py_lean_ms_median/ {print $2}' /tmp/tgrad_L7_timing.txt)"
TINY_MS_MEDIAN="$("$PY" -c "import json; print(json.load(open('$BASELINE'))['tinygrad_ms']['median'])")"
[[ -n "$LEAN_MS_MEDIAN" ]] || { echo "  ✗ couldn't parse py_lean_ms_median"; cat /tmp/tgrad_L7_timing.txt; exit 1; }

# Compute ratio + 1.5× check in Python (bash floats are awkward).
RATIO_AND_CHECK="$("$PY" - "$LEAN_MS_MEDIAN" "$TINY_MS_MEDIAN" <<'PYRATIO'
import sys
lean   = float(sys.argv[1])
tinyg  = float(sys.argv[2])
ratio  = lean / tinyg if tinyg > 0 else float("inf")
ok     = ratio <= 1.5 and lean > 0
print(f"{ratio:.4f} {'true' if ok else 'false'}")
PYRATIO
)"
RATIO="$(echo "$RATIO_AND_CHECK" | awk '{print $1}')"
OK="$(echo "$RATIO_AND_CHECK" | awk '{print $2}')"

echo "  lean_ms_median:     $LEAN_MS_MEDIAN ms"
echo "  tinygrad_ms_median: $TINY_MS_MEDIAN ms"
echo "  ratio:              $RATIO  (predicate: ≤ 1.5)"
if [[ "$OK" != "true" ]]; then
  if [[ "$(echo "$LEAN_MS_MEDIAN > 0" | awk '{print ($1>0)}')" != "1" ]]; then
    echo "  ✗ perf-parity predicate failed: lean_ms_median = $LEAN_MS_MEDIAN (must be > 0)"
    echo "      — the timing loop is reporting zero work; likely 'a @ b' was skipped"
  else
    echo "  ✗ perf-parity predicate failed: lean_ms/tinygrad_ms = $RATIO > 1.5"
    echo "      per §G7 fall-back: verify zero-copy / library cache / LRU before halting"
  fi
  exit 1
fi
echo "  ✓ Tgrad bf16 64×64 matmul within 1.5× of pinned tinygrad baseline"

# ─── LAYER D: negative — out-of-scope shape rejects ───────────────────
set +e
(cd "$REPO_ROOT" && "$PY" "$TGRAD_DIR/python/tgrad.py" bench-timing \
    --shape 7x9x11 --dtype bf16) >/tmp/tgrad_L7_neg.txt 2>&1
neg_rc=$?
set -e
if [[ "$neg_rc" -eq 0 ]]; then
  echo "  ✗ bench-timing --shape 7x9x11 returned 0 — should reject"
  cat /tmp/tgrad_L7_neg.txt; exit 1
fi
grep -q "NotInLeanScope" /tmp/tgrad_L7_neg.txt || {
  echo "  ✗ bench-timing --shape 7x9x11 did not raise NotInLeanScope"
  cat /tmp/tgrad_L7_neg.txt; exit 1
}
echo "  ✓ negative test correctly rejected (out-of-scope shape → NotInLeanScope)"

# ─── LAYER E: evidence ────────────────────────────────────────────────
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$PROFILE"; plat="$(uname -srm)"
timing_hash="$(shasum -a 256 /tmp/tgrad_L7_timing.txt | awk '{print $1}')"
baseline_hash="$(shasum -a 256 "$BASELINE" | awk '{print $1}')"
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L7.json" <<EOF
{
  "gate": "L7",
  "ts_utc": "$ts",
  "host_profile": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "scope": "L7.a — single-shape (bf16 64×64) perf parity via real FFI",
  "lean_ms_median":     $LEAN_MS_MEDIAN,
  "tinygrad_ms_median": $TINY_MS_MEDIAN,
  "ratio":              $RATIO,
  "predicate":          "ratio <= 1.5",
  "hashes": {
    "timing_output_sha256":  "$timing_hash",
    "baseline_sha256":       "$baseline_hash"
  }
}
EOF
check_evidence_for L7 || exit 1
check_falsifiability_verified L7 || exit 1
echo "  ✓ L7 perf-parity gate green (evidence recorded)"

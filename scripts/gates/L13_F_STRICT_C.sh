#!/usr/bin/env bash
# Gate L13.F.STRICT.C — perf parity flip.
#
# Production TC routing must use the manual-load WMMA kernel and L13_F
# must pass under the strict ratio <= 1.5 predicate. (The historical
# §1.RELAX authorisation lived in `Tgrad/GOAL_L13_F.md`; that doc was
# pruned at v1.0.0 along with the rest of the GOAL_*.md agent ladder.
# Strict-perf evidence now stands on L13_F.json's `perf_ratio_max`
# alone — see Tgrad/EXPERIMENT_RESULT.md for context.)
set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
cd "$REPO_ROOT"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L13_F_STRICT_C] perf parity flip"

# ─── LAYER A: universal preflight ─────────────────────────────────────
run_preflight

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
FFI="$TGRAD_DIR/Tgrad/PythonFFI.lean"
MATMUL_TC="$TGRAD_DIR/Tgrad/Renderer/MatmulTc.lean"
L13F_GATE="$TGRAD_DIR/scripts/gates/L13_F.sh"
BASELINE_CAPTURE="$TGRAD_DIR/scripts/capture/tinygrad_baseline_tc_general.py"
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"

for f in "$FFI" "$MATMUL_TC" "$L13F_GATE" "$BASELINE_CAPTURE"; do
  [[ -f "$f" ]] || { echo "  ✗ missing required artifact: $f"; exit 1; }
done
echo "  ✓ required source, gate, and baseline-capture files present"

# ─── LAYER B: structural ──────────────────────────────────────────────
FFI_PROD="$(awk '/@\[export tgrad_matmul_tc_lean\]/{flag=1} flag{print} /pure rc.toInt32/{if(flag){exit}}' "$FFI")"
if ! grep -qF 'compileOrCacheGetTcManual' <<<"$FFI_PROD"; then
  echo "  ✗ production tgrad_matmul_tc_lean does not use compileOrCacheGetTcManual"
  exit 1
fi
if ! grep -qF 'matmul_tc_manual_' <<<"$FFI_PROD"; then
  echo "  ✗ production tgrad_matmul_tc_lean does not dispatch the manual kernel name"
  exit 1
fi
if grep -qE 'tcMatmulKernelDecl[[:space:]]' "$FFI"; then
  echo "  ✗ PythonFFI still references the old simdgroup-load TC decl"
  exit 1
fi
echo "  ✓ production TC FFI is wired to manual-load kernel generation"

# NOTE: the historical §1.RELAX retraction check on GOAL_L13_F.md was
# removed at v1.0.0 alongside the agent-ladder docs. Strict-perf proof
# now relies on L13_F.json's `perf_ratio_max <= 1.5` assertion below.

if ! grep -qF 'Device[Device.DEFAULT].synchronize()' "$BASELINE_CAPTURE"; then
  echo "  ✗ tinygrad TC-general baseline capture is not synchronized"
  exit 1
fi
echo "  ✓ tinygrad TC-general baseline capture synchronizes measured work"

# ─── LAYER C: strict behavioural gate ─────────────────────────────────
bash "$L13F_GATE" >/tmp/tgrad_L13F_STRICT_C_L13_F.log 2>&1 || {
  echo "  ✗ strict L13_F regression failed"
  tail -60 /tmp/tgrad_L13F_STRICT_C_L13_F.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ L13_F strict gate passes"

L13F_EVID="$TGRAD_DIR/fixtures/gate_evidence/L13_F.json"
"$PY" - "$L13F_EVID" <<'PY'
import json, sys
p = sys.argv[1]
e = json.load(open(p))
assert e["perf_predicate"] == "ratio ≤ 1.5", e
assert float(e["perf_ratio_max"]) <= 1.5, e
assert e["production_kernel"] == "tcMatmulKernelDeclManualLoad", e
assert e["tc_general_correct"] == 8 and e["tc_general_wmma"] == 8, e
assert e["tc_general_scalar_routes"] == 0, e
assert e.get("tc_general_manual_load") == 8, e
assert e["random_tc_correct"] == 10 and e["random_tc_wmma"] == 10, e
PY
echo "  ✓ L13_F evidence records strict perf + manual production"

# ─── LAYER C2: focused regressions ────────────────────────────────────
bash "$TGRAD_DIR/scripts/gates/L13_F_STRICT_A.sh" >/tmp/tgrad_L13F_STRICT_C_A.log 2>&1 || {
  echo "  ✗ L13_F_STRICT_A regression failed"
  tail -40 /tmp/tgrad_L13F_STRICT_C_A.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ L13_F_STRICT_A regression gate still green"

bash "$TGRAD_DIR/scripts/gates/L13_F_STRICT_B.sh" >/tmp/tgrad_L13F_STRICT_C_B.log 2>&1 || {
  echo "  ✗ L13_F_STRICT_B regression failed"
  tail -60 /tmp/tgrad_L13F_STRICT_C_B.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ L13_F_STRICT_B regression gate still green"

# ─── LAYER D: anti-cheat ──────────────────────────────────────────────
grep -qF 'PERF_RATIO_MAX=1.5' "$L13F_GATE" || {
  echo "  ✗ L13_F.sh does not pin PERF_RATIO_MAX=1.5"; exit 1;
}
if grep -qF 'PERF_RATIO_MAX=12' "$L13F_GATE"; then
  echo "  ✗ L13_F.sh still contains relaxed PERF_RATIO_MAX=12"
  exit 1
fi
grep -qF 'bench-random-tc-general' "$L13F_GATE" || {
  echo "  ✗ L13_F.sh no longer runs random TC-general recheck"; exit 1;
}
grep -qF 'SEED="${HEAD_SHA:0:16}"' "$L13F_GATE" || {
  echo "  ✗ L13_F.sh random seed is not sourced from HEAD"; exit 1;
}
echo "  ✓ strict threshold and random recheck are structurally pinned"

[[ -x "$TGRAD_CLI" ]] || { echo "  ✗ missing executable tgrad-cli at $TGRAD_CLI"; exit 1; }
EMIT="/tmp/tgrad_L13F_STRICT_C_manual_prod.msl"
"$TGRAD_CLI" render-metal-algebraic matmul_tc_manual_1024x1024x3072 >"$EMIT" 2>/tmp/tgrad_L13F_STRICT_C_render.err || {
  echo "  ✗ render-metal-algebraic failed for production manual TC kernel"
  cat /tmp/tgrad_L13F_STRICT_C_render.err | sed 's/^/      /'
  exit 1
}
grep -qF 'threadgroup_barrier(mem_flags::mem_threadgroup);' "$EMIT" || {
  echo "  ✗ rendered production manual TC source lacks threadgroup barrier marker"; exit 1;
}
grep -qF '.thread_elements()' "$EMIT" || {
  echo "  ✗ rendered production manual TC source lacks manual thread_elements loads"; exit 1;
}
grep -qF 'simdgroup_multiply_accumulate' "$EMIT" || {
  echo "  ✗ rendered production manual TC source lacks WMMA multiply"; exit 1;
}
echo "  ✓ rendered production manual TC source exposes manual WMMA markers"

# ─── LAYER E: evidence ────────────────────────────────────────────────
mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
"$PY" - "$REPO_ROOT" "$TGRAD_DIR" "$L13F_EVID" "$TGRAD_DIR/fixtures/gate_evidence/L13_F_STRICT_C.json" <<'PY'
import datetime as dt, hashlib, json, os, platform, subprocess, sys
from pathlib import Path

repo = Path(sys.argv[1])
tgrad = Path(sys.argv[2])
l13f_evid = Path(sys.argv[3])
out_path = Path(sys.argv[4])
e = json.loads(l13f_evid.read_text())

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
l13f_sha = sha(l13f_evid)
out = {
    "gate": "L13_F_STRICT_C",
    "ts_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "host_profile": os.environ.get("TGRAD_PERF_PROFILE", os.environ.get("TGRAD_HOST", "apple_m4_mini_release")),
    "platform": platform.platform(),
    "commit": commit,
    "scope": "L13.F.STRICT.C — perf parity flip (manual-load kernel as production; §1.RELAX rescinded)",
    "perf_predicate": "ratio ≤ 1.5",
    "relax_retracted": True,
    "production_kernel": "tcMatmulKernelDeclManualLoad",
    "old_kernel_status": "not referenced by PythonFFI production routing",
    "pinned_total": e["tc_general_total"],
    "pinned_correct": e["tc_general_correct"],
    "pinned_manual_load": e["tc_general_manual_load"],
    "pinned_ratio_max": e["perf_ratio_max"],
    "random_total": e["random_tc_total"],
    "random_correct": e["random_tc_correct"],
    "random_wmma": e["random_tc_wmma"],
    "l13_f_evidence_sha256": l13f_sha,
    "hashes": {
        "ffi_module_sha256": sha(tgrad / "Tgrad/PythonFFI.lean"),
        "matmul_tc_module_sha256": sha(tgrad / "Tgrad/Renderer/MatmulTc.lean"),
        "l13_f_evidence_sha256": l13f_sha,
    },
}
out_path.write_text(json.dumps(out, indent=2) + "\n")
PY

check_evidence_for L13_F_STRICT_C || exit 1
check_falsifiability_verified L13_F_STRICT_C || exit 1
echo "  ✓ L13.F.STRICT.C gate green"

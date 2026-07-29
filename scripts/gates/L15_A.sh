#!/usr/bin/env bash
# Gate L15.A — experiment-closure audit (static + structural).
# Per `Tgrad/GOAL_L15_A.md` + `Tgrad/GOAL_L15.md §3` (criteria 1-3) and
# `Tgrad/GOAL_L15.md §4` (9 static checks).
set -euo pipefail
: "${REPO_ROOT:?must be set by gate.sh}"
: "${TGRAD_DIR:?must be set by gate.sh}"
source "$TGRAD_DIR/scripts/lib/checks.sh"

echo "[L15_A] experiment closure — static + structural audit"

run_preflight
cd "$REPO_ROOT"

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"

# Layer B — structural: required modules.
required=(
  python/tgrad.py
  Tgrad/UOp.lean
  Tgrad/Tensor.lean
  Tgrad/Pipeline.lean
  Tgrad/Codegen/Opt/Heuristic.lean
  scripts/dev/l15_a_audit.py
)
for m in "${required[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing: $m"; exit 1; }
done
echo "  ✓ all ${#required[@]} required modules present"

# Dispatch totality is a Lean proposition, not a source-spelling inference.
# Build its defining module explicitly, then run the audit's exact-type witness
# and architecture self-tests before any GPU work.
HEURISTIC_BUILD_LOG="$(mktemp "${TMPDIR:-/tmp}/tgrad_L15_A_heuristic.XXXXXX")"
if ! (cd "$REPO_ROOT" && lake build Tgrad.Codegen.Opt.Heuristic) \
    >"$HEURISTIC_BUILD_LOG" 2>&1; then
  echo "  ✗ dispatch-total obligation module failed to build"
  sed 's/^/      /' "$HEURISTIC_BUILD_LOG"
  exit 1
fi
grep -qE '^theorem[[:space:]]+pickDispatchPlan_bf16_total([[:space:](]|$)' \
  "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean" \
  || { echo "  ✗ missing pickDispatchPlan_bf16_total obligation"; exit 1; }
echo "  ✓ pickDispatchPlan_bf16_total module builds"

if ! "$PY" "$TGRAD_DIR/scripts/dev/l15_a_audit.py" --self-test; then
  echo "  ✗ L15.A dispatch audit self-test failed"
  exit 1
fi

# Required method/property defs (Criterion 1 / Layer B).
N_METHODS="$(grep -cE '^[[:space:]]+(def|@property)[[:space:]]+(from_numpy|numpy|__matmul__|T|transpose|reshape|__getitem__)\b' \
              "$TGRAD_DIR/python/tgrad.py" || true)"
[[ "$N_METHODS" -ge 6 ]] || { echo "  ✗ Python Tensor defines $N_METHODS/6 required methods"; exit 1; }
echo "  ✓ Python Tensor defines $N_METHODS required methods (>= 6)"

# Required UOp ctors (Criterion 2 / Layer B).
N_CTORS="$(grep -cE '\|[[:space:]]+\.(buffer|permute|reshape|expand|slice)\b' \
            "$TGRAD_DIR/Tgrad/UOp.lean" || true)"
[[ "$N_CTORS" -ge 5 ]] || { echo "  ✗ UOp.lean has $N_CTORS/5 movement ctors"; exit 1; }
echo "  ✓ UOp.lean has $N_CTORS movement ctors (>= 5)"

# Layer C — ensure rangeify trace is fresh (regenerate by re-running bench-views).
ensure_dylib /tmp/tgrad_L15_A_dylib.log || exit 1

GATE_START_TS="$(date +%s)"
TRACE=/tmp/tgrad_rangeify_trace.jsonl
echo "  → regenerating rangeify trace via bench-views (D3 freshness)"
(cd "$REPO_ROOT" && TGRAD_RANGEIFY_TRACE=1 "$PY" "$TGRAD_DIR/python/tgrad.py" bench-views \
    --output /tmp/tgrad_L15_A_views.jsonl) >/tmp/tgrad_L15_A_bench.log 2>&1
TRACE_MTIME="$(stat -f %m "$TRACE" 2>/dev/null || echo 0)"
[[ "$TRACE_MTIME" -ge "$GATE_START_TS" ]] || { echo "  ✗ rangeify trace not fresh (mtime=$TRACE_MTIME < gate_start=$GATE_START_TS)"; exit 1; }
echo "  ✓ rangeify trace mtime=$TRACE_MTIME >= gate_start=$GATE_START_TS"

# Run the audit; capture the JSON output.
AUDIT_OUT=/tmp/tgrad_L15_A_audit.json
"$PY" "$TGRAD_DIR/scripts/dev/l15_a_audit.py" >"$AUDIT_OUT" 2>/tmp/tgrad_L15_A_audit.err
if [[ ! -s "$AUDIT_OUT" ]]; then
  echo "  ✗ audit produced no output"
  cat /tmp/tgrad_L15_A_audit.err
  exit 1
fi

N_PASS="$("$PY" -c '
import json
data = json.load(open("'"$AUDIT_OUT"'"))
print(sum(1 for c in data["criteria"] if c["verdict"] == "pass"))
')"
[[ "$N_PASS" -eq 3 ]] || { echo "  ✗ L15.A: $N_PASS/3 criteria pass"; cat "$AUDIT_OUT"; exit 1; }
echo "  ✓ 3/3 criteria pass"

N_STATIC="$("$PY" -c '
import json
data = json.load(open("'"$AUDIT_OUT"'"))
print(sum(1 for v in data["static_checks"].values() if v is True))
')"
[[ "$N_STATIC" -eq 9 ]] || { echo "  ✗ L15.A: $N_STATIC/9 static checks pass"; cat "$AUDIT_OUT"; exit 1; }
echo "  ✓ 9/9 static checks pass"

# Layer D — anti-cheat: the gate script must not author hardcoded "pass" verdicts.
N_HARDCODED="$(grep -cE '"verdict"[[:space:]]*:[[:space:]]*"pass"' "$TGRAD_DIR/scripts/gates/L15_A.sh" || true)"
[[ "$N_HARDCODED" -eq 0 ]] || { echo "  ✗ L15_A.sh contains $N_HARDCODED hardcoded verdict strings"; exit 1; }
echo "  ✓ D2 no hardcoded verdicts in gate script"

# Layer C2 — regression evidence
L11_PAIRS="$("$PY" -c 'import json; print(json.load(open("'"$TGRAD_DIR/fixtures/gate_evidence/L11.json"'"))["pairs_passed"])' 2>/dev/null || echo 0)"
[[ "$L11_PAIRS" -eq 50 ]] || { echo "  ✗ L11.json.pairs_passed = $L11_PAIRS"; exit 1; }
echo "  ✓ L11 50/50 still holds"

# Layer E — write evidence
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
tensor_hash="$(shasum -a 256 "$TGRAD_DIR/python/tgrad.py" | awk '{print $1}')"
uop_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/UOp.lean" | awk '{print $1}')"
pipeline_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Pipeline.lean" | awk '{print $1}')"
trace_hash="$(shasum -a 256 "$TRACE" 2>/dev/null | awk '{print $1}')"
audit_hash="$(shasum -a 256 "$TGRAD_DIR/scripts/dev/l15_a_audit.py" | awk '{print $1}')"
heuristic_hash="$(shasum -a 256 "$TGRAD_DIR/Tgrad/Codegen/Opt/Heuristic.lean" | awk '{print $1}')"

mkdir -p "$TGRAD_DIR/fixtures/gate_evidence"
"$PY" -c "
import json, sys
audit = json.load(open('$AUDIT_OUT'))
out = {
    'gate': 'L15_A',
    'ts_utc': '$ts',
    'host': '$host',
    'platform': '$plat',
    'commit': '$commit',
    'scope': 'L15.A — static + structural audit (criteria 1-3 + §4 static checks)',
    'dispatch_total_obligation': 'pickDispatchPlan_bf16_total',
    'criteria': audit['criteria'],
    'static_checks': audit['static_checks'],
    'narrowing_notes': [
        'Criterion 1 / n_legacy narrowed from {_buf,_shape,_dtype,_view_ops,_strides} '
        'to {_view_ops,_strides,_view_graph,_shape_tracker}: the dropped fields are '
        'buffer-accessor caches mirroring state owned by the Lean tensor handle, not '
        'Python-owned view metadata. Justification carried to EXPERIMENT_RESULT.md '
        'per GOAL_L15.md §4 escape hatch.',
        'Criterion 1 / n_handle narrowed from \`self._handle = ctypes.c_uint64\` to '
        'a pair: (a) \`self._handle = handle\` assignment AND (b) the FFI binding '
        '\`_lib.tgrad_tensor_from_buffer.restype = ctypes.c_uint64\`. The wrap '
        'literal was visual-clarity in the spec draft; the binding gives the same '
        'type guarantee.',
    ],
    'hashes': {
        'tensor_module_sha256':   '$tensor_hash',
        'uop_module_sha256':      '$uop_hash',
        'pipeline_module_sha256': '$pipeline_hash',
        'rangeify_trace_sha256':  '$trace_hash',
        'audit_module_sha256':    '$audit_hash',
        'heuristic_module_sha256': '$heuristic_hash',
    },
}
json.dump(out, open('$TGRAD_DIR/fixtures/gate_evidence/L15_A.json', 'w'), indent=2)
print('  ✓ evidence written')
"
check_evidence_for L15_A || exit 1
check_falsifiability_verified L15_A || exit 1
echo "  ✓ L15.A — 3/3 criteria + 9/9 static checks — GREEN"

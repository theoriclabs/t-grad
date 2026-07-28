#!/usr/bin/env bash
# Fast, non-authoritative development checks for agent edit loops.
#
# This script is intentionally NOT a gate. It does not write
# fixtures/gate_evidence/*.json, does not verify falsifiability
# tables, and does not replace `bash scripts/gate.sh`.
#
# Usage:
#   bash scripts/devcheck.sh --all
#   bash scripts/devcheck.sh L13_F
#   bash scripts/devcheck.sh L14_B
set -euo pipefail

export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TGRAD_DIR="$REPO_ROOT"
export TGRAD_BENCH_MODE=smoke
export PYTHONPATH="$TGRAD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
DEV_SEED="${TGRAD_DEVCHECK_SEED:-decafbadc0ffee01}"
ALLOW_METAL_RUNTIME_SKIP="${TGRAD_ALLOW_METAL_RUNTIME_SKIP:-0}"
cd "$REPO_ROOT"

source "$TGRAD_DIR/scripts/lib/checks.sh"

gate="${1:-}"
if [[ -z "$gate" ]]; then
  echo "usage: bash scripts/devcheck.sh <gate|--all>"
  echo "example: bash scripts/devcheck.sh L14_B"
  exit 2
fi

PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
TGRAD_CLI="$TGRAD_DIR/.lake/build/bin/tgrad-cli"

has_subcommand() {
  local name="$1"
  grep -qE "['\"]$name['\"]|add_parser\\(['\"]$name['\"]" "$TGRAD_DIR/python/tgrad.py"
}

run_cmd() {
  echo "  $*" >&2
  "$@"
}

metal_runtime_unavailable_log() {
  local log="$1"
  grep -Eq \
    "tgrad_tensor_alloc|returned rc=-2|Pipeline\.realizeView failed" \
    "$log"
}

skip_metal_runtime_smoke() {
  local label="$1"
  local log="$2"
  if grep -q "tgrad_tensor_alloc" "$log"; then
    echo "  ⚠ $label skipped: Metal allocation unavailable in this environment"
    sed 's/^/      /' "$log"
    return 0
  fi
  if metal_runtime_unavailable_log "$log" && [[ "$ALLOW_METAL_RUNTIME_SKIP" == "1" ]]; then
    echo "  ⚠ $label skipped: Metal runtime compile/dispatch unavailable in this environment"
    sed 's/^/      /' "$log"
    return 0
  fi
  return 1
}

cheap_preflight() {
  echo "[devcheck] cheap preflight"
  check_no_sorry           || return 1
  check_no_axiom           || return 1
  check_no_unsafe          || return 1
  check_no_gate_regression || return 1
  check_shell_continuation || return 1
  check_real_chronology   || return 1
  check_architecture_boundary || return 1

  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/observe_pilot.py" --check-generated \
    || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/generate_broadcast_add_manifest.py" \
    --check || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/check_broadcast_add_trial_lock.py" \
    --lock "$TGRAD_DIR/fixtures/requirements/broadcast_add_trial_lock_v1.json" \
    || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/check_broadcast_add_trial_lock.py" \
    --lock "$TGRAD_DIR/fixtures/requirements/broadcast_add_trial_lock_v2.json" \
    || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/generate_broadcast_add_amendment_v2.py" \
    --check || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/generate_broadcast_add_amendment_v3.py" \
    --check || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/generate_broadcast_add_trial_lock_v3.py" \
    --check || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/check_broadcast_add_v4_tooling_amendment.py" \
    || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/check_broadcast_add_v5_relation_amendment.py" \
    || return 1
  run_cmd "$PY" "$TGRAD_DIR/scripts/spec/check_broadcast_add_v6_identity_split.py" \
    || return 1
  run_cmd "$PY" -m unittest \
    scripts.spec.test_broadcast_add_observer \
    scripts.spec.test_broadcast_add_trial_lock \
    scripts.spec.test_broadcast_add_amendment_v2 \
    scripts.spec.test_broadcast_add_amendment_v3 \
    scripts.spec.test_broadcast_add_trial_lock_v3 \
    scripts.spec.test_broadcast_add_v4_tooling_amendment \
    scripts.spec.test_broadcast_add_relation \
    scripts.spec.test_broadcast_add_manifest || return 1

  # The ranked-broadcast and int32-elementwise Metal regressions are
  # explicitly registered because unittest discovery is not used here.
  # A unique log avoids adding another process-global /tmp collision.
  local ranked_broadcast_log
  ranked_broadcast_log="$(mktemp "${TMPDIR:-/tmp}/tgrad_ranked_broadcast.XXXXXX")"
  if ! run_cmd "$PY" -m unittest \
      scripts.spec.test_ranked_broadcast \
      scripts.spec.test_int32_elementwise \
      >"$ranked_broadcast_log" 2>&1; then
    if skip_metal_runtime_smoke "ranked broadcast regression" "$ranked_broadcast_log"; then
      rm -f "$ranked_broadcast_log"
    else
      sed 's/^/      /' "$ranked_broadcast_log"
      rm -f "$ranked_broadcast_log"
      return 1
    fi
  else
    rm -f "$ranked_broadcast_log"
  fi

  if [[ -f "$TGRAD_DIR/c/Makefile" ]]; then
    make -C "$TGRAD_DIR/c" >/tmp/tgrad_devcheck_make.log 2>&1 || {
      echo "  ✗ make -C c failed"
      sed 's/^/      /' /tmp/tgrad_devcheck_make.log
      return 1
    }
  fi
  (cd "$TGRAD_DIR" && lake build tgrad-cli tgrad-tests TgradSpec) \
    >/tmp/tgrad_devcheck_lake.log 2>&1 || {
      echo "  ✗ lake build tgrad-cli tgrad-tests failed"
      sed 's/^/      /' /tmp/tgrad_devcheck_lake.log
      return 1
    }
  ensure_dylib /tmp/tgrad_devcheck_dylib.log || return 1
  echo "[devcheck] cheap preflight ✓"
}

smoke_basic_matmul() {
  if ! run_cmd "$PY" "$TGRAD_DIR/python/tgrad.py" bench --shape 64x64x64 --dtype bf16 \
      >/tmp/tgrad_devcheck_basic.txt 2>&1; then
    if skip_metal_runtime_smoke "runtime smoke" /tmp/tgrad_devcheck_basic.txt; then
      return 0
    fi
    cat /tmp/tgrad_devcheck_basic.txt
    return 1
  fi
  grep -q "py_byte_match: true" /tmp/tgrad_devcheck_basic.txt || {
    echo "  ✗ basic 64x64x64 bench did not byte-match"
    cat /tmp/tgrad_devcheck_basic.txt
    return 1
  }
}

smoke_timing() {
  if ! run_cmd "$PY" "$TGRAD_DIR/python/tgrad.py" bench-timing \
      --shape 64x64x64 --dtype bf16 --warmup 1 --measured 3 \
      >/tmp/tgrad_devcheck_timing.txt 2>&1; then
    if grep -q "tgrad_tensor_alloc" /tmp/tgrad_devcheck_timing.txt; then
      echo "  ⚠ timing smoke skipped: Metal allocation unavailable in this environment"
      cat /tmp/tgrad_devcheck_timing.txt | sed 's/^/      /'
      return 0
    fi
    cat /tmp/tgrad_devcheck_timing.txt
    return 1
  fi
  grep -q "py_lean_ms_median" /tmp/tgrad_devcheck_timing.txt || {
    echo "  ✗ timing smoke missing py_lean_ms_median"
    cat /tmp/tgrad_devcheck_timing.txt
    return 1
  }
}

smoke_render_algebraic() {
  [[ -x "$TGRAD_CLI" ]] || { echo "  ✗ missing tgrad-cli at $TGRAD_CLI"; return 1; }
  run_cmd "$TGRAD_CLI" render-metal-algebraic matmul_1024x1024x1024 \
    >/tmp/tgrad_devcheck_matmul.msl
  grep -q "simdgroup_multiply_accumulate" /tmp/tgrad_devcheck_matmul.msl || {
    echo "  ✗ algebraic render smoke did not emit simdgroup_multiply_accumulate"
    return 1
  }
}

smoke_random_shapes() {
  # Keep this smoke small and deterministic. The full random-shape
  # gate samples large shapes; devcheck only proves the Python/FFI path
  # is alive without creating large Metal buffers.
  smoke_basic_matmul
}

smoke_tc_general() {
  run_cmd "$PY" -c '
import tgrad
assert tgrad._lib.tgrad_matmul_tc_eligible(128, 128, 128) == 1
assert tgrad._lib.tgrad_matmul_tc_eligible(96, 128, 128) == 1
assert tgrad._lib.tgrad_matmul_tc_eligible(64, 64, 64) == 1
assert tgrad._lib.tgrad_matmul_tc_eligible(31, 64, 64) == 0
assert tgrad._lib.tgrad_matmul_tc_eligible(64, 7, 64) == 0
assert tgrad._lib.tgrad_matmul_tc_eligible(64, 64, 48) == 0
print("tc_eligibility_smoke: true")
' >/tmp/tgrad_devcheck_tc.txt
}

smoke_synthetic_tg_kernel() {
  [[ -x "$TGRAD_CLI" ]] || { echo "  ✗ missing tgrad-cli at $TGRAD_CLI"; return 1; }
  run_cmd "$TGRAD_CLI" render-metal-algebraic synthetic_tg_kernel \
    >/tmp/tgrad_devcheck_synthetic_tg_kernel.msl
  grep -qF "threadgroup_barrier(mem_flags::mem_threadgroup);" \
    /tmp/tgrad_devcheck_synthetic_tg_kernel.msl || {
    echo "  ✗ synthetic_tg_kernel missing threadgroup barrier"
    return 1
  }
  grep -qF "mat_a.thread_elements()[0]" \
    /tmp/tgrad_devcheck_synthetic_tg_kernel.msl || {
    echo "  ✗ synthetic_tg_kernel missing per-thread WMMA load"
    return 1
  }
  run_cmd "$TGRAD_CLI" ffi-compile-smoke /tmp/tgrad_devcheck_synthetic_tg_kernel.msl \
    >/tmp/tgrad_devcheck_synthetic_tg_kernel_compile.txt
  grep -q "fn_count: 1" /tmp/tgrad_devcheck_synthetic_tg_kernel_compile.txt || {
    echo "  ✗ synthetic_tg_kernel failed ffi-compile-smoke"
    cat /tmp/tgrad_devcheck_synthetic_tg_kernel_compile.txt
    return 1
  }
}

smoke_manual_tc_kernel() {
  [[ -x "$TGRAD_CLI" ]] || { echo "  ✗ missing tgrad-cli at $TGRAD_CLI"; return 1; }
  local out="/tmp/tgrad_devcheck_manual_tc_1024x1024x3072.msl"
  run_cmd "$TGRAD_CLI" render-metal-algebraic matmul_tc_manual_1024x1024x3072 \
    >"$out"
  grep -qF "threadgroup bfloat tg_a[256];" "$out" || {
    echo "  ✗ manual TC kernel missing tg_a threadgroup tile"
    return 1
  }
  grep -qF "threadgroup_barrier(mem_flags::mem_threadgroup);" "$out" || {
    echo "  ✗ manual TC kernel missing threadgroup barrier"
    return 1
  }
  grep -qF ".thread_elements()" "$out" || {
    echo "  ✗ manual TC kernel missing thread_elements path"
    return 1
  }
  run_cmd "$TGRAD_CLI" ffi-compile-smoke "$out" \
    >/tmp/tgrad_devcheck_manual_tc_compile.txt
  grep -q "fn_count: 1" /tmp/tgrad_devcheck_manual_tc_compile.txt || {
    echo "  ✗ manual TC kernel failed ffi-compile-smoke"
    cat /tmp/tgrad_devcheck_manual_tc_compile.txt
    return 1
  }
  run_cmd "$PY" "$TGRAD_DIR/python/tgrad.py" bench-tc-general \
    --use-manual-load --warmup 1 --measured 1 \
    --output /tmp/tgrad_devcheck_manual_tc.jsonl \
    >/tmp/tgrad_devcheck_manual_tc.txt
  grep -q "py_bench_tc_general_n_correct: 8" /tmp/tgrad_devcheck_manual_tc.txt || {
    echo "  ✗ manual TC pinned smoke did not report 8/8 correctness"
    cat /tmp/tgrad_devcheck_manual_tc.txt
    return 1
  }
}

smoke_views() {
  if has_subcommand bench-random-views; then
    if ! run_cmd "$PY" "$TGRAD_DIR/python/tgrad.py" bench-random-views \
        --seed "$DEV_SEED" --count 1 --output /tmp/tgrad_devcheck_random_views.jsonl \
        >/tmp/tgrad_devcheck_random_views.txt 2>&1; then
      if skip_metal_runtime_smoke "view smoke" /tmp/tgrad_devcheck_random_views.txt; then
        return 0
      fi
      cat /tmp/tgrad_devcheck_random_views.txt
      return 1
    fi
  else
    echo "  - bench-random-views unavailable; skipping view smoke"
  fi
}

cheap_preflight

echo "[devcheck] smoke for $gate"
case "$gate" in
  --all|all)
    run_cmd bash "$TGRAD_DIR/scripts/check_no_tinygrad_deps.sh"
    smoke_basic_matmul
    smoke_render_algebraic
    smoke_tc_general
    smoke_views
    ;;
  L0|L1|L2|L3|L4|L5)
    [[ -x "$TGRAD_CLI" ]] && run_cmd "$TGRAD_CLI" --help >/tmp/tgrad_devcheck_cli_help.txt || true
    ;;
  L6)
    smoke_basic_matmul
    ;;
  L7)
    smoke_timing
    ;;
  L8|L9|L10|L11|L12)
    smoke_basic_matmul
    smoke_render_algebraic
    ;;
  L13|L13_A|L13_B|L13_C|L13_D|L13_E)
    smoke_random_shapes
    ;;
  L13_F)
    smoke_tc_general
    ;;
  L13_F_STRICT_A)
    smoke_synthetic_tg_kernel
    ;;
  L13_F_STRICT_B)
    smoke_manual_tc_kernel
    ;;
  L13_F_STRICT_C)
    smoke_tc_general
    smoke_manual_tc_kernel
    ;;
  L14|L14_A|L14_B|L14_C)
    smoke_views
    ;;
  L15|L15_A|L15_B|L15_C)
    smoke_random_shapes
    smoke_views
    ;;
  *)
    echo "unknown gate for devcheck: $gate"
    echo "known gates follow scripts/gate.sh naming, e.g. L13_F or L14_B"
    exit 2
    ;;
esac

echo "[devcheck] $gate smoke ✓"
if [[ "$gate" == "--all" || "$gate" == "all" ]]; then
  echo "note: devcheck is non-authoritative. Flip gates with: bash scripts/gate.sh"
else
  echo "note: devcheck is non-authoritative. Flip gates with: bash scripts/gate.sh --single $gate"
fi

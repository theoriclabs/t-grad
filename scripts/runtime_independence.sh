#!/usr/bin/env bash
# Verify this standalone repo can build and run without a sibling
# tinygrad checkout.
#
# Load-bearing demonstration that Tgrad is a replacement, not a wrapper.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO_ROOT/scripts/lib/run_context.sh"
tgrad_run_context_init

sandbox="$(tgrad_run_subdir runtime_independence_sandbox)"
make_log="$(tgrad_run_path independence_make_c.log)"
lake_log="$(tgrad_run_path independence_lake.log)"
dylib_log="$(tgrad_run_path independence_dylib.log)"
gate_log="$(tgrad_run_path independence_gate.log)"
dylib2_log="$(tgrad_run_path independence_dylib2.log)"
static_log="$(tgrad_run_path independence_static.log)"
bench_log="$(tgrad_run_path independence_bench.log)"

echo "  [runtime_independence] sandbox: $sandbox"

mkdir -p "$sandbox/repo"
(cd "$REPO_ROOT" && tar \
  --exclude .git \
  --exclude .lake \
  --exclude c/build \
  --exclude __pycache__ \
  --exclude '*.egg-info' \
  --exclude dist \
  -cf - . 2>/dev/null) | (cd "$sandbox/repo" && tar xf -)
# Init a minimal .git so check_no_gate_regression has a HEAD to read
(cd "$sandbox/repo" && git init -q && git add . && git \
  -c user.email='sandbox@tgrad' -c user.name='sandbox' \
  commit -q -m 'sandbox snapshot' 2>/dev/null || true)

# Confirm tinygrad is NOT present in the sandbox.
if [[ -d "$sandbox/repo/tinygrad" ]]; then
  echo "  ✗ runtime_independence: sandbox accidentally contains tinygrad/"
  exit 1
fi
if [[ -d "$sandbox/repo/theograd_phases" ]]; then
  echo "  ✗ runtime_independence: sandbox accidentally contains theograd_phases/"
  exit 1
fi
echo "  ✓ sandbox has standalone Tgrad repo only (no sibling tinygrad/, theograd_phases/, Theograd/)"

# Build inside the sandbox.
# The Tgrad lakefile builds .o files from c/ via the Makefile then lake
# build does the rest. lean_path / lake_home should resolve normally
# since lean-toolchain points at the same installed toolchain.
(cd "$sandbox/repo/c" && make 2>&1) >"$make_log" 2>&1 || {
  echo "  ✗ runtime_independence: make -C c failed in sandbox"
  tail -30 "$make_log" | sed 's/^/      /'
  exit 1
}

(cd "$sandbox/repo" && lake build Tgrad:shared tgrad-cli tgrad-tests) >"$lake_log" 2>&1 || {
  echo "  ✗ runtime_independence: lake build failed in sandbox"
  tail -30 "$lake_log" | sed 's/^/      /'
  exit 1
}

(cd "$sandbox/repo/c" && make dylib) >"$dylib_log" 2>&1 || {
  echo "  ✗ runtime_independence: make -C c dylib failed in sandbox"
  tail -30 "$dylib_log" | sed 's/^/      /'
  exit 1
}
echo "  ✓ sandbox build succeeds (lake build + dylib)"

if [[ "${TGRAD_RUNTIME_FULL_GATE:-0}" == "1" ]]; then
  TGRAD_DIR="$sandbox/repo" \
  REPO_ROOT="$sandbox/repo" \
    bash "$sandbox/repo/scripts/gate.sh" >"$gate_log" 2>&1 || {
    echo "  ✗ runtime_independence: gate sweep failed in tinygrad-free sandbox"
    echo "      (full log: $gate_log)"
    tail -50 "$gate_log" | sed 's/^/        /'
    exit 1
  }
  echo "  ✓ gate sweep passes in sandbox"

  (cd "$sandbox/repo/c" && make dylib) >"$dylib2_log" 2>&1 || {
    echo "  ✗ runtime_independence: post-sweep dylib rebuild failed"
    tail -20 "$dylib2_log" | sed 's/^/        /'
    exit 1
  }
else
  TGRAD_DIR="$sandbox/repo" bash "$sandbox/repo/scripts/check_no_tinygrad_deps.sh" \
    >"$static_log" 2>&1 || {
    echo "  ✗ runtime_independence: static independence check failed"
    sed 's/^/        /' "$static_log"
    exit 1
  }
  echo "  ✓ static independence check passes in sandbox"
fi

# Confirm the libtgrad.dylib bench runs without tinygrad present.
PY="${TGRAD_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="python3"
(cd "$sandbox/repo" && \
  TGRAD_LIB="$sandbox/repo/.lake/build/lib/libtgrad.dylib" \
  "$PY" "$sandbox/repo/python/tgrad.py" bench --shape 64x64x64) \
    >"$bench_log" 2>&1 || {
  echo "  ✗ runtime_independence: python bench failed in tinygrad-free sandbox"
  tail -20 "$bench_log" | sed 's/^/        /'
  exit 1
}
grep -q "py_byte_match: true" "$bench_log" || {
  echo "  ✗ runtime_independence: sandbox python bench did NOT byte-match"
  tail -20 "$bench_log" | sed 's/^/        /'
  exit 1
}
echo "  ✓ standalone repo passes tinygrad-free sandbox build + python bench"

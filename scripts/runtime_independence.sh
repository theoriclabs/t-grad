#!/usr/bin/env bash
# Verify this standalone repo can build and run without a sibling
# tinygrad checkout.
#
# Load-bearing demonstration that Tgrad is a replacement, not a wrapper.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

sandbox="$(mktemp -d -t tgrad_indep_XXXXXX)"
trap 'rm -rf "$sandbox"' EXIT

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
(cd "$sandbox/repo/c" && make 2>&1) >/tmp/tgrad_indep_make_c.log 2>&1 || {
  echo "  ✗ runtime_independence: make -C c failed in sandbox"
  tail -30 /tmp/tgrad_indep_make_c.log | sed 's/^/      /'
  exit 1
}

(cd "$sandbox/repo" && lake build Tgrad:shared tgrad-cli tgrad-tests) >/tmp/tgrad_indep_lake.log 2>&1 || {
  echo "  ✗ runtime_independence: lake build failed in sandbox"
  tail -30 /tmp/tgrad_indep_lake.log | sed 's/^/      /'
  exit 1
}

(cd "$sandbox/repo/c" && make dylib) >/tmp/tgrad_indep_dylib.log 2>&1 || {
  echo "  ✗ runtime_independence: make -C c dylib failed in sandbox"
  tail -30 /tmp/tgrad_indep_dylib.log | sed 's/^/      /'
  exit 1
}
echo "  ✓ sandbox build succeeds (lake build + dylib)"

if [[ "${TGRAD_RUNTIME_FULL_GATE:-0}" == "1" ]]; then
  TGRAD_DIR="$sandbox/repo" \
  REPO_ROOT="$sandbox/repo" \
    bash "$sandbox/repo/scripts/gate.sh" >/tmp/tgrad_indep_gate.log 2>&1 || {
    echo "  ✗ runtime_independence: gate sweep failed in tinygrad-free sandbox"
    echo "      (full log: /tmp/tgrad_indep_gate.log)"
    tail -50 /tmp/tgrad_indep_gate.log | sed 's/^/        /'
    exit 1
  }
  echo "  ✓ gate sweep passes in sandbox"

  (cd "$sandbox/repo/c" && make dylib) >/tmp/tgrad_indep_dylib2.log 2>&1 || {
    echo "  ✗ runtime_independence: post-sweep dylib rebuild failed"
    tail -20 /tmp/tgrad_indep_dylib2.log | sed 's/^/        /'
    exit 1
  }
else
  TGRAD_DIR="$sandbox/repo" bash "$sandbox/repo/scripts/check_no_tinygrad_deps.sh" \
    >/tmp/tgrad_indep_static.log 2>&1 || {
    echo "  ✗ runtime_independence: static independence check failed"
    cat /tmp/tgrad_indep_static.log | sed 's/^/        /'
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
    >/tmp/tgrad_indep_bench.log 2>&1 || {
  echo "  ✗ runtime_independence: python bench failed in tinygrad-free sandbox"
  tail -20 /tmp/tgrad_indep_bench.log | sed 's/^/        /'
  exit 1
}
grep -q "py_byte_match: true" /tmp/tgrad_indep_bench.log || {
  echo "  ✗ runtime_independence: sandbox python bench did NOT byte-match"
  tail -20 /tmp/tgrad_indep_bench.log | sed 's/^/        /'
  exit 1
}
echo "  ✓ standalone repo passes tinygrad-free sandbox build + python bench"

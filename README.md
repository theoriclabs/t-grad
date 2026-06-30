# Tgrad

[![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)](pyproject.toml)
[![Lean 4](https://img.shields.io/badge/Lean-4-purple)](lean-toolchain)
[![Backend: Apple Silicon Metal](https://img.shields.io/badge/backend-Apple%20Silicon%20Metal-black)](#requirements)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Tgrad is a bounded Lean 4 + Metal demo runtime for bf16 2-D matmul on
Apple Silicon. Python is a thin authoring layer over a Lean-owned
runtime exposed through `ctypes`.

This is not a general tensor library and it is not a tinygrad
replacement. It is a compact experiment in writing a real numerical
runtime slice with Lean in the loop.

## Highlights

- Lean-owned runtime path for a bounded bf16 matmul slice.
- Python `Tensor` wrapper with familiar `Tensor.from_numpy(...)`, `@`,
  `.numpy()`, and `.to_bytes()` helpers.
- Apple Silicon Metal backend through a small C/Objective-C bridge.
- Tensor-core routes for TC-eligible shapes plus scalar fallback paths
  for the supported range.
- View-composed matmul tests for transpose, reshape, permute, expand,
  and slice cases covered by the release gates.
- Committed release evidence for correctness, runtime independence, and
  performance parity against captured tinygrad baselines.

## What Tgrad Is

Tgrad keeps the high-level authoring surface in Python while moving the
runtime loop, renderer, scheduler, dispatch decisions, and FFI exports
into Lean 4. The Metal bridge owns platform work such as buffer
allocation, shader compilation, and dispatch.

The project is intentionally small. The goal is to show that Lean 4 can
own meaningful systems code, with executable gates around the claims,
without pretending to implement all of tinygrad.

## Requirements

- macOS on Apple Silicon with Metal support.
- Xcode command line tools.
- Lean from `lean-toolchain` via elan/lake.
- Python 3.11 or newer.
- `numpy`.

Check that the macOS SDK is visible:

```sh
xcrun --sdk macosx --show-sdk-path
```

## Quickstart: Build

From this repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .

make -C c
lake build Tgrad:shared tgrad-cli tgrad-tests
make -C c dylib
```

## Quickstart: Run A Matmul

```sh
TGRAD_LIB="$PWD/.lake/build/lib/libtgrad.dylib" \
  .venv/bin/python -m tgrad bench --shape 64x64x64 --dtype bf16
```

Expected shape of the result:

```text
py_shape: 64x64x64
py_dtype: bf16
py_byte_match: true
py_pipeline_ok: true
```

## Python API

```python
import numpy as np
import tgrad

a = tgrad.Tensor.from_numpy(np.random.randn(64, 64).astype(np.float32))
b = tgrad.Tensor.from_numpy(np.random.randn(64, 64).astype(np.float32))
c = a @ b

print(c.numpy().shape)
```

Run with the dylib path in the environment:

```sh
TGRAD_LIB="$PWD/.lake/build/lib/libtgrad.dylib" .venv/bin/python example.py
```

## Verify

Fast local checks:

```sh
bash scripts/check_no_tinygrad_deps.sh
bash scripts/devcheck.sh --all
```

Release gates:

```sh
bash scripts/gate.sh --list
bash scripts/gate.sh L7
bash scripts/gate.sh
```

`scripts/gate.sh` runs the 37-gate ratchet and writes evidence JSON to
`fixtures/gate_evidence/`. Those evidence files are committed as a demo
snapshot; CI smoke checks avoid rewriting them.

## Performance Evidence

Performance numbers are profile-specific. The committed public profile
is `apple_m4_mini_release`; tinygrad baselines are dev-time fixtures
captured on Metal with `USE_TC=1`, `BEAM=0`, and `NOOPT=0`.

The checked-in gate-evidence snapshot records:

| Gate | Claim | Evidence |
|---|---|---|
| L7 | Single bf16 `64x64x64` timing | Tgrad median `0.2362 ms`, tinygrad median `0.731 ms`, ratio `0.3231` |
| L11 | Full 50 shape/distribution sweep | `50/50` correct, `50/50` within `ratio <= 1.5`, ratio min/median/max `0.3582 / 0.9354 / 1.284` |
| L12 | Algebraic MSL emit for captured shapes | `10/10` captured kernels byte-equal, algebraic sweep `50/50`, `alg_ratio_max = 1.23` |
| L13_F | TC-general manual-load WMMA route | `8/8` pinned rows correct and routed through WMMA, `10/10` random TC rows correct, `ratio_max = 0.8786` |

Timings are synchronized wall-clock measurements around dispatched Metal
work, not isolated GPU counter timings. Re-capture tinygrad baselines
and re-run the gates under the same `TGRAD_PERF_PROFILE` before
publishing machine-specific comparisons.

## Regenerating Baselines

To regenerate the single-shape, full-sweep, and TC-general baselines on
another machine:

```sh
TGRAD_PERF_PROFILE=my_machine .venv/bin/python scripts/capture/perf_baseline.py
TGRAD_PERF_PROFILE=my_machine .venv/bin/python scripts/capture/perf_baseline_full.py
TGRAD_PERF_PROFILE=my_machine .venv/bin/python scripts/capture/tinygrad_baseline_tc_general.py
```

For noisy or thermally sensitive Metal runs, prefer the resumable full
baseline capturer:

```sh
TGRAD_PERF_PROFILE=my_machine \
  .venv/bin/python scripts/capture/perf_baseline_full_stable.py \
  --passes 3 --warmup 10 --measured 30 --resume
```

It writes the normal gate-compatible
`fixtures/perf/tinygrad_baseline_<profile>_full.json` plus an audit JSONL
with per-pass samples, selected pass, host metadata, and cooldown
settings.

Then run gates with the same profile:

```sh
TGRAD_PERF_PROFILE=my_machine bash scripts/gate.sh L7
TGRAD_PERF_PROFILE=my_machine bash scripts/gate.sh L11
TGRAD_PERF_PROFILE=my_machine bash scripts/gate.sh L13_F
```

## Supported Slice

In scope:

- bf16 input/output with fp32 accumulation.
- Apple Silicon Metal backend.
- Contiguous 2-D matmul.
- Captured sentinel matmul kernels.
- Algebraic MSL emit for the captured matmul set.
- TC-general manual-load WMMA kernels.
- Scalar fallback paths for supported smaller or non-TC shapes.
- View-composed matmul cases covered by the committed transpose,
  reshape, permute, expand, and slice gates.

## Not Claimed

- Full tinygrad replacement.
- Autograd.
- Arbitrary dtype support.
- CUDA, ROCm, OpenCL, CPU, or non-Metal backends.
- Full BEAM search.
- Formal proof of equivalence to tinygrad.
- Arbitrary tinygrad movement semantics beyond the committed view gates.

## Repository Layout

- `Tgrad/`: Lean runtime, model, renderer, scheduler, optimizer, and FFI
  modules.
- `python/`: thin Python wrapper and benchmark harness.
- `c/`: Metal bridge and Python-facing dylib target.
- `fixtures/`: codegen, benchmark, perf, and gate evidence fixtures.
- `scripts/`: gate runner, smoke checks, capture scripts, and release
  validation helpers.
- `EXPERIMENT_RESULT.md`: release audit narrative and scope notes.

Runtime code must not import or shell out to tinygrad. The tinygrad
capture scripts under `scripts/capture/` are dev-time tools for
regenerating baseline fixtures.

## License And Attribution

Tgrad is released under the MIT license. See `LICENSE`.

This repository contains a bounded Lean/Metal demo derived from a
tinygrad study. See `NOTICE.md` for attribution and scope notes.

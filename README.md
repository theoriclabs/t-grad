# Tgrad

Tgrad is a Lean 4 demo runtime for a bounded bf16 Metal matmul path.
It keeps the runtime loop in Lean, exposes a small Python authoring
layer through `ctypes`, and uses a C/Objective-C bridge for Metal buffer
allocation, compilation, and dispatch.

This is a demo repository, not a general tensor library. The implemented
slice is intentionally narrow: bf16 input/output, fp32 accumulation,
Apple Silicon Metal, and matmul/view cases covered by the committed
gate evidence.

## Requirements

- macOS on Apple Silicon with Metal support
- Xcode command line tools (`xcrun --sdk macosx --show-sdk-path`)
- Lean from `lean-toolchain` via elan/lake
- Python 3.11 or newer
- `numpy`

## Quickstart

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .

make -C c
lake build Tgrad:shared tgrad-cli tgrad-tests
make -C c dylib

TGRAD_LIB="$PWD/.lake/build/lib/libtgrad.dylib" \
  .venv/bin/python -m tgrad bench --shape 64x64x64
```

Python usage:

```python
import numpy as np
import tgrad

a = tgrad.Tensor.from_numpy(np.random.randn(64, 64).astype(np.float32))
b = tgrad.Tensor.from_numpy(np.random.randn(64, 64).astype(np.float32))
c = a @ b
print(c.numpy().shape)
```

## What Is Included

- `Tgrad/`: Lean runtime, renderer, scheduler, optimizer, FFI exports,
  and model/control-plane modules.
- `python/`: thin Python wrapper and benchmark harness.
- `c/`: Metal bridge and Python-facing dylib target.
- `fixtures/`: codegen, benchmark, perf, and gate evidence fixtures.
- `scripts/`: gate runner, smoke checks, capture scripts, and release
  validation helpers.

Runtime code must not import or shell out to tinygrad. The tinygrad
capture scripts under `scripts/capture/` are dev-time tools for
regenerating committed baseline fixtures.

## Verify

Fast local smoke:

```sh
bash scripts/check_no_tinygrad_deps.sh
bash scripts/devcheck.sh --all
```

Authoritative release gates:

```sh
bash scripts/gate.sh --list
bash scripts/gate.sh L7
bash scripts/gate.sh
```

`scripts/gate.sh` runs the 37-gate ratchet and writes evidence JSON to
`fixtures/gate_evidence/`. Those evidence files are committed as a demo
snapshot; CI smoke checks avoid rewriting them.

## Performance Fixtures

The committed public perf profile is `apple_m4_mini_release`.

To regenerate the single-shape, full-sweep, and TC-general baselines on
another machine:

```sh
TGRAD_PERF_PROFILE=my_machine .venv/bin/python scripts/capture/perf_baseline.py
TGRAD_PERF_PROFILE=my_machine .venv/bin/python scripts/capture/perf_baseline_full.py
TGRAD_PERF_PROFILE=my_machine .venv/bin/python scripts/capture/tinygrad_baseline_tc_general.py
```

Then run gates with the same profile:

```sh
TGRAD_PERF_PROFILE=my_machine bash scripts/gate.sh L7
TGRAD_PERF_PROFILE=my_machine bash scripts/gate.sh L11
TGRAD_PERF_PROFILE=my_machine bash scripts/gate.sh L13_F
```

Checked-in release evidence records:

| Gate | Claim | Evidence |
|---|---|---|
| L7 | bf16 64x64x64 timing ratio <= 1.5 | `ratio = 0.3231` |
| L11 | 50 shape/distribution pairs correct and ratio <= 1.5 | `50/50`, `ratio_max = 1.284` |
| L12 | algebraic emit byte-equals captured MSL for 10 shapes | `10/10`, alg sweep `50/50`, `alg_ratio_max = 1.23` |
| L13_F | TC-general manual-load route correct and ratio <= 1.5 | `8/8`, `ratio_max = 0.8786` |

Timings are synchronized wall-clock measurements around dispatched Metal
work, not isolated GPU counter timings.

## Scope

In scope:

- bf16 matmul with fp32 accumulation
- Apple Silicon Metal backend
- tensor-core route for TC-eligible shapes
- scalar route for smaller/non-TC shapes in the supported range
- view composition through transpose, reshape, permute, expand, and slice

Not claimed:

- full tinygrad replacement
- autograd
- arbitrary dtype support
- non-Metal backends
- full BEAM search
- formal proof of equivalence to tinygrad

See `EXPERIMENT_RESULT.md` for the release audit narrative.

## License

Tgrad is released under the MIT license. See `LICENSE`.

This repository contains a bounded Lean/Metal demo derived from a
tinygrad study. See `NOTICE.md` for attribution and scope notes.

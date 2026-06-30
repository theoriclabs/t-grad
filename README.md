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

Latest local timing for the release-profile single-shape benchmark
(fresh non-sandboxed iTerm shell):

```sh
export TGRAD_LIB="$PWD/.lake/build/lib/libtgrad.dylib"
export PYTHONPATH=python

TGRAD_PERF_PROFILE=apple_m4_mini_release \
  ../.venv/bin/python python/tgrad.py bench-timing \
  --shape 64x64x64 --warmup 50 --measured 200
```

| Shape | Dtype | Tgrad median | tinygrad baseline median | Ratio | Speedup | Nominal throughput |
|---|---:|---:|---:|---:|---:|---:|
| 64x64x64 | bf16 | 0.2143 ms | 0.7165 ms | 0.2991 | 3.34x | 2.45 GFLOP/s |

This tiny shape is launch-overhead dominated, so the ratio is more useful
than the raw GFLOP/s number. The throughput column uses `2*M*N*K` floating
point operations divided by the measured median wall time.

Latest local multi-shape run from the same non-sandboxed iTerm shell:

```sh
export TGRAD_LIB="$PWD/.lake/build/lib/libtgrad.dylib"
export PYTHONPATH=python

TGRAD_PERF_PROFILE=apple_m4_mini_release \
  ../.venv/bin/python python/tgrad.py bench-tc-general \
  --use-manual-load --warmup 10 --measured 30

TGRAD_PERF_PROFILE=apple_m4_mini_release \
  ../.venv/bin/python python/tgrad.py bench-full \
  --warmup 30 --measured 30
```

TC-general manual-load sweep:

| Shape | Bucket | Tgrad median | tinygrad median | Ratio |
|---|---|---:|---:|---:|
| `1024x1024x3072` | known-M/K, new-N | 3.4695 ms | 8.2306 ms | 0.4215 |
| `1536x1024x1024` | new-M | 1.7106 ms | 5.7798 ms | 0.2960 |
| `1024x1536x1024` | new-K | 1.7417 ms | 3.1221 ms | 0.5579 |
| `1536x1536x1536` | all-new mid | 3.6855 ms | 2.8697 ms | 1.2843 |
| `2048x1536x3072` | mixed large | 7.4302 ms | 16.7248 ms | 0.4443 |
| `3072x1024x1536` | tall-ish | 3.9411 ms | 11.9354 ms | 0.3302 |
| `1024x2048x3072` | wide-ish | 5.2315 ms | 10.5227 ms | 0.4972 |
| `3072x2048x1024` | large tall | 5.0259 ms | 6.7595 ms | 0.7435 |

Summary: `8/8` correct, `8/8` tensor-core route, ratio median `0.4707`,
ratio max `1.2843`.

Full sweep over 50 shape/distribution pairs:

| Shape | Correct | Tgrad min range | Ratio min | Ratio median | Ratio max |
|---|---:|---:|---:|---:|---:|
| `1024x1024x1024` | 5/5 | 1.4321-1.5760 ms | 0.9199 | 0.9675 | 1.1696 |
| `2048x2048x2048` | 5/5 | 6.2951-8.4748 ms | 1.1778 | 1.3847 | 1.5879 |
| `4096x4096x4096` | 5/5 | 66.7433-76.0641 ms | 1.7650 | 1.8893 | 2.0229 |
| `8192x8192x8192` | 5/5 | 685.6828-885.5495 ms | 2.0098 | 2.2318 | 2.5987 |
| `8192x1024x1024` | 5/5 | 8.1208-9.0324 ms | 1.4474 | 1.6036 | 1.6373 |
| `4096x1024x1024` | 5/5 | 3.7548-4.5320 ms | 1.1806 | 1.2301 | 1.4717 |
| `2048x1024x1024` | 5/5 | 2.1854-2.3059 ms | 1.1409 | 1.1648 | 1.1947 |
| `1024x1024x8192` | 5/5 | 8.7118-8.9742 ms | 1.6352 | 1.6604 | 1.6937 |
| `1024x1024x4096` | 5/5 | 3.8555-4.5780 ms | 1.0851 | 1.3142 | 1.4913 |
| `1024x1024x2048` | 5/5 | 2.2606-2.4518 ms | 1.1786 | 1.2347 | 1.2691 |

Summary: `50/50` correct, `30/50` within the `ratio <= 1.5` performance
gate, ratio min `0.9199`, ratio median `1.4101`, ratio max `2.5987`.
The full-sweep ratios compare Tgrad min wall time to the checked-in
tinygrad min baseline. Rows above the ratio threshold are still
numerically correct; the CLI reports them as `PERF_MISS` rows so they are
clearly tuning targets rather than correctness failures.

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
with per-pass samples, selected pass, host metadata, and cooldown settings.

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

## See Also

- [Clockworks C Compiler](github.com/ClockworksCompute/ccc) a C Compiler written in Lean 4, which can detect Heartbleed class of bugs.

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

Read the background post: [We Rewrote tinygrad in Lean](https://theoric.com/blog/we-rewrote-tinygrad-in-lean/).

## Highlights

- Lean-owned runtime path for a bounded bf16 matmul slice.
- Python `Tensor` wrapper with familiar `Tensor.from_numpy(...)`, `@`,
  `.numpy()`, and `.to_bytes()` helpers.
- Apple Silicon Metal backend through a small C/Objective-C bridge.
- Tensor-core routes for TC-eligible shapes plus scalar fallback paths
  for the supported range.
- View-composed matmul tests for transpose, reshape, permute, expand,
  and slice cases covered by the release gates.
- Bounded view `.numpy()`/`.to_bytes()` materialization through a
  rangeified, bit-preserving Metal copy kernel.
- An execution differential for all 11 captured/generated sentinel kernels;
  sources intentionally differ and 240 MB of outputs match bit-for-bit.
- Production sentinel dispatch now uses those parametric generated kernels;
  the per-shape Lean transcription and its parser have been deleted.
- Captured MSL remains only as an independent executable oracle used by the
  semantic differential; it is neither imported nor read by product runtime.
- A separate checked specification for runtime capabilities, findings,
  growth cases, resource constraints, and repository evolution.
- Historical gate artifacts retained for audit; their performance and
  provenance claims are not treated as current promoted evidence.

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
lake build Tgrad:shared TgradSpec tgrad-spec tgrad-cli tgrad-tests
make -C c dylib
```

Inspect the checked product/work specification:

```sh
.lake/build/bin/tgrad-spec
```

`Tgrad` and `TgradSpec` are separate build roots. The former is the product
library linked into the runtime; the latter contains the stable ontology,
runtime-work inventory, evidence-bearing findings, growth cases, live resource
constraints, event-based evolution protocol, and executable work graph. The
specification is checked without becoming part of
`libtgrad_Tgrad.dylib`.

## Growing Tgrad

[Growing Tgrad](GROWING_TGRAD.md) separates repeatable work performed **by**
the codebase from repository-evolution work performed **on** it. The checked
model connects runtime observations to findings, growth cases, attempts,
immutable candidate trees, exact-tree checks, and promotion certificates.

The current `tgrad-spec` report also states the migration limit explicitly:
warp parameterization, the codegen differential, its additive L12 layer, the
provenance auditor, and view materialization have replayable promotion
histories. Older `Progress.complete` entries predate the protocol and are not
themselves promotion certificates.

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
print(a.T.numpy().shape)  # supported views materialize before host readback
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

Audit the committed evidence provenance separately:

```sh
python3 scripts/dev/evidence_provenance_audit.py
```

It currently exits nonzero by design: 37/37 files name an absent commit,
77/115 non-transient hashes are unresolved, 28 roll-ups disagree, and 27
files use a host key their own gate script does not emit. The auditor becomes
a fatal release predicate only after an owner-authorized serial regeneration.

Release gates:

```sh
bash scripts/gate.sh --list
bash scripts/gate.sh L7
bash scripts/gate.sh
```

`scripts/gate.sh` runs the historical 37-gate suite and writes evidence JSON to
`fixtures/gate_evidence/`. Run it only serially: many scripts share fixed
`/tmp/tgrad_*` paths and the performance cases share one GPU. The committed
JSON is an audit snapshot, not evidence for the current tree: it names a
commit absent from this repository and contains stale recorded hashes.

## Performance Evidence

Tgrad currently makes no promoted performance-parity claim. The historical
L7/L11/L12/L13_F ratios are not a valid kernel/runtime comparison:

- the Tgrad and tinygrad timed regions enclose different work;
- tinygrad was measured without `TinyJit` while Tgrad used cached,
  pre-selected kernels;
- the results compare live Tgrad measurements with frozen baselines;
- the committed evidence hashes do not match the committed baselines;
- the smallest reported sweep ratios are impossible for byte-identical
  kernels dispatched with identical geometry.

Sentinels now route through the parametric generator. The next admissible
experiment measures both runtimes in one session with symmetric boundaries,
retains raw distributions and provenance, and reports an honest regression if
that is the result. Timings remain serial because this machine has one GPU.

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
- Parametric TC matmul declarations are authoritative for all 11 sentinels and
  aligned general shapes. Captured MSL is an independent differential oracle,
  not a product build or dispatch input.
- TC-general manual-load WMMA kernels.
- Scalar fallback paths for supported smaller or non-TC shapes.
- View-composed matmul cases covered by transpose, reshape, permute, expand,
  and slice differential tests. Supported view `.numpy()`/`.to_bytes()` uses
  indexed materialization; unsupported non-contiguous reshape and empty views
  reject explicitly before Metal allocation/dispatch.

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

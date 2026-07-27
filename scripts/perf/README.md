# Paired runtime benchmark

`paired_runtime.py` is a diagnostic operator for comparing this Tgrad checkout
with a pinned upstream checkout under controlled, paired conditions. It records
measurements and provenance; it does not decide whether a revision passes or
fails.

## Pinned comparison checkout

Both subjects must be attributable. The default reference is deliberately
fixed:

- checkout: `/tmp/tgrad-upstream-19c4d736`
- commit: `19c4d736f2bc8e26d21f08b28ffd6298408da00f`
- tree: `855cca3b00c38841a6d3a043284f3a2ca696d4b0`

Prepare that checkout separately. The pinned source is known to import on Metal
with the Python 3.12 environment at
`/tmp/tgrad-upstream-py312/bin/python`. Select it explicitly while launching
the operator:

```sh
.venv/bin/python scripts/perf/paired_runtime.py \
  --python-executable /tmp/tgrad-upstream-py312/bin/python \
  --raw-output /tmp/tgrad-paired.raw.jsonl \
  --summary-output /tmp/tgrad-paired.summary.json
```

The launcher performs at most one `execve` before importing either runtime.
After that replacement, local Tgrad and pinned tinygrad are loaded and sampled
in the same Python process. It never starts an interpreter per sample.

The operator sets and records `DEV=METAL` for the run. That value is visible to
the measured implementations. If the invoking environment contains the
deprecated `METAL=1` selector, the operator records and removes it from the
benchmark environment: it is not inherited by either side of the pair.

The local Tgrad subject must also be a clean Git tree by default. A dirty or
unattributed local tree fails before either runtime is imported. The
`--allow-dirty-tgrad` and `--allow-unknown-tinygrad-revision` switches exist
only for conspicuously labelled diagnostics; their output is not promotable.
For a promotable run, the operator rebuilds `libtgrad.dylib` through the
repository build graph before importing either runtime, rejects an alternate
`TGRAD_LIB`, rechecks that the source revision stayed clean and unchanged, and
records the exact loaded path, byte size, SHA-256, source commit/tree, and
build commands. A diagnostic override may observe an existing binary but is
labelled as such.

Metal is not visible inside the project sandbox. Real sessions must therefore
run unsandboxed and serially: they use one GPU and process-global runtime state.
The command above is documentation, not a command run by the focused tests.

## Operator contract

The current checkout and pinned reference are loaded as a live pair in one
long-lived Python process. Each sample is therefore collected under the same
process-level conditions instead of starting a subprocess per sample. This
same-process design is part of the benchmark contract; there is no per-sample
subprocess isolation.

Correctness is checked before timing. Both sides receive the same deterministic
bfloat16 payloads, and their bf16 output bytes must match exactly. Each timed
session then prepares its real repeated-call route and byte-checks that route
again before collecting a sample. On tinygrad this second check is an actual
captured TinyJit replay, not the un-JIT reference call. The operator forces and
records `DEV=METAL`, `JIT=1`, `DEBUG=0`, `PROFILE=0`, `BEAM=0`, and
`GRAPH_ONE_KERNEL=0`, requires a non-null capture object and valid replay
count, verifies that replay retains the captured output buffer, and rejects a
disabled or failed capture before timing. On Tgrad it is
an explicit `PreparedMatmul` plan with one compiled route and one reusable
output allocation; correctness poisons that output first and asserts its
buffer identity is unchanged after execution. Before timing, both prepared
routes also execute a second deterministic same-shape input pair, must agree
bit-for-bit, must differ from the primary output, and then restore the primary
inputs. This catches a captured route that merely returns stale output.

Measurements are organized into sessions. Within each session, a seeded
schedule interleaves the two implementations in both `AB` and `BA` order. The
seed and realized ordering are recorded so the pairing can be audited and
reproduced. Pairing, interleaving, and keeping both implementations live in the
same process reduce sensitivity to drift; they do not eliminate system noise.

The operator writes raw, sample-level JSON Lines (`JSONL`), a derived summary,
and a completion manifest in one directory. The raw JSONL retains every
attempted observation, pairing and session
identifier, realized order, timing-boundary label, error, and raw duration. Its
`run_id` joins it to the summary, which records revision, environment,
toolchain, workload, boundary availability, and configuration provenance. The
summary records the raw byte hash and count. The completion manifest is written
last and cryptographically joins the hashes of both artifacts; a raw or summary
file without that marker is incomplete evidence. The derived statistics are a
convenience view and must not replace the raw stream.
Every invocation receives a fresh run-instance identifier and UTC capture
timestamp, and both participate in `run_id`; two repeated runs with the same
code and configuration therefore cannot alias as one evidence run.

## Timing boundaries

Every reported timing is labeled with the boundary it actually measures.
Boundaries are not interchangeable:

- The default Tgrad prepared-runtime route excludes route selection,
  rendering/compilation, and output allocation. It includes the Python
  prepared call, Lean plan lookup, FFI, Metal submission, and the wait that
  makes completion observable.
- The TinyJit replay boundary measures replay of an already prepared TinyJit
  program. It is not a measurement of initial capture or compilation.
- TinyJit's first and capture calls are retained as separate composite
  preparation observations. They include runtime and synchronization and are
  not mislabeled as isolated compiler timings. Tgrad plan creation is likewise
  retained as a separate composite preparation observation. Dispatch-only GPU
  timestamps remain unavailable because neither public route exposes the same
  portable counter boundary.
- An optional un-JIT measurement is a distinct diagnostic boundary. It is
  reported separately and must not be compared as though it were TinyJit
  replay or the synchronized Tgrad route.

If a boundary cannot be established faithfully for an implementation or
platform, the operator records that boundary as unavailable. It does not
silently substitute another boundary, fabricate a zero, or fold unlike
boundaries together.

The default pair is a symmetric prepared-runtime comparison, not a kernel
comparison: both sides reuse fixed-shape compiled state, resident inputs, and
output storage, and both return only after device completion. Every comparison carries the machine-readable field
`kernel_speed_claim_eligible`; it is `false` for every boundary currently
available. No downstream report may relabel these observations as kernel
speed.

## Statistical methodology

The summary reports paired statistics from the retained raw samples, including
variance information and bootstrap uncertainty. Bootstrap resampling gives
each logical session equal weight and resamples pairs within each selected
session. Read central estimates
together with dispersion, sample counts, availability, and the bootstrap
interval; a single ratio or minimum time is not a sufficient interpretation of
a noisy runtime comparison. The summary also retains absolute duration and
effective operational-rate quantiles for both sides so physical plausibility
can be checked without reverse-engineering a ratio. The rate is
`2*M*K*N / observed boundary time`; because the boundaries include host work,
runtime lookup/replay, FFI or Python dispatch, and synchronization, it is not
isolated kernel throughput.

The output also captures provenance needed to understand a run: current and
reference revisions, the pinned reference tree, dirty-state information when
available, configuration, relevant environment selection,
payload description, seeds, sessions, ordering, timing boundaries, and output
schema information.

A revision override exists only for diagnostics. Using it departs from the
default pinned comparison and must remain conspicuous in provenance; results
from an override are not a replacement for the default pinned run.

The operator intentionally defines no performance thresholds, regression
budgets, pass/fail verdicts, or frozen baseline numbers. Results describe the
recorded run on its recorded machine and software state. Policy decisions, if
any, belong outside this measurement tool.

Release policy is deliberately a second stage. `VARIANCE_MODEL.md` requires
independent calibration runs and independent evaluation runs; the paired
operator cannot certify itself. `repeatability_decision.json` currently says
`calibration_required`, so the evidence publisher rejects every purported
prepared-runtime certificate. After the owner serially records calibration,
derives a variance-sensitive rule from calibration only, and reviews that rule,
evaluation artifacts and their external certificate can be supplied to
`scripts/gate.sh --regenerate-evidence`. Candidate collection copies those
files into the release closure; they are not frozen baseline files read by the
timed path. A certificate only enumerates completed artifact sets. The release
validator re-parses their raw observations, re-derives the per-run hierarchy
and maximum upper interval, and applies the reviewed threshold; authored
`run_count`, `session_count`, and `passed` fields are not part of the schema.

## Focused tests

The focused tests validate the operator contract, scheduling, correctness
gating, serialization, statistics, provenance, and failure handling. They use
controlled test doubles and run no real performance timing.

Run them from the repository virtual environment:

```sh
PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python scripts/dev/test_paired_runtime.py
```

Passing these tests shows that the measurement machinery behaves as specified;
it does not produce or validate a real runtime result. A real timing run occurs
only when the benchmark operator itself is invoked against supported hardware.

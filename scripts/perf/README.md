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

Metal is not visible inside the project sandbox. Real sessions must therefore
run unsandboxed and serially: they use one GPU and process-global runtime state.
The command above is documentation, not a command run by the focused tests.

## Operator contract

The current checkout and pinned reference are loaded as a live pair in one
long-lived Python process. Each sample is therefore collected under the same
process-level conditions instead of starting a subprocess per sample. This
same-process design is part of the benchmark contract; there is no per-sample
subprocess isolation.

Correctness is checked before either output file is created. Both sides receive
the same deterministic bfloat16 payloads, and their bf16 output bytes must
match exactly. A correctness mismatch aborts with no timed evidence rather
than allowing an invalid timing into the statistics.

Measurements are organized into sessions. Within each session, a seeded
schedule interleaves the two implementations in both `AB` and `BA` order. The
seed and realized ordering are recorded so the pairing can be audited and
reproduced. Pairing, interleaving, and keeping both implementations live in the
same process reduce sensitivity to drift; they do not eliminate system noise.

The operator writes raw, sample-level JSON Lines (`JSONL`) and a derived
summary. The raw JSONL retains every attempted observation, pairing and session
identifier, realized order, timing-boundary label, error, and raw duration. Its
`run_id` joins it to the summary, which records revision, environment,
toolchain, workload, boundary availability, and configuration provenance. The
derived statistics are a convenience view and must not replace the raw stream.
Every invocation receives a fresh run-instance identifier and UTC capture
timestamp, and both participate in `run_id`; two repeated runs with the same
code and configuration therefore cannot alias as one evidence run.

## Timing boundaries

Every reported timing is labeled with the boundary it actually measures.
Boundaries are not interchangeable:

- The Tgrad synchronized route measures the named Tgrad execution route and
  includes the synchronization required to make completion observable at that
  boundary.
- The TinyJit replay boundary measures replay of an already prepared TinyJit
  program. It is not a measurement of initial capture or compilation.
- TinyJit's first and capture calls are retained as separate composite
  preparation observations. They include runtime and synchronization and are
  not mislabeled as isolated compiler timings. Tgrad's isolated compile and
  dispatch-only boundaries are explicitly unavailable where the public route
  cannot expose them.
- An optional un-JIT measurement is a distinct diagnostic boundary. It is
  reported separately and must not be compared as though it were TinyJit
  replay or the synchronized Tgrad route.

If a boundary cannot be established faithfully for an implementation or
platform, the operator records that boundary as unavailable. It does not
silently substitute another boundary, fabricate a zero, or fold unlike
boundaries together.

The default pair is an operational repeated-call comparison, not a kernel
comparison: Tgrad includes output allocation while tinygrad uses prepared
TinyJit replay. Every comparison carries the machine-readable field
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
TFLOP/s quantiles for both sides so physical plausibility can be checked
without reverse-engineering a ratio.

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

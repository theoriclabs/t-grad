# Plan: finish view correctness + real codegen

This document explains rationale and recovery strategy. It is **not the
current-state database**. The checked source of truth is
`Tgrad/Spec/Work.lean`, whose dependency, readiness, validation, and
parallel-frontier predicates are proved by `native_decide` in the separate
`TgradSpec` library.

Current state:

- **The immediate correctness tier landed in `e1d5760`.** Payload length is
  checked against shape, views retain their base tensor, and view readback
  fails explicitly instead of returning the wrong values.
- **View materialization landed in `e6241bd`.** Exact tree `790d413…`
  makes `.numpy()` and `.to_bytes()` rangeify and materialize supported views,
  and passed isolated build/API/safety/numerical/differential checks on base
  `bdc01b0`. Earlier `995eb7e…` and `ac393d2…` checks remain attached to
  abandoned stale-base attempts.
- **The semantic codegen harness landed in `a6d5958`.** All 11 generated
  sentinels differ textually from their captures but are bit-identical in
  execution across 240 MB of output.
- **Generated production routing landed in `fd945b1`.** All 11 sentinels now
  use the parametric WMMA generator with generated nonzero launch geometry.
- **The transcription deletion migration is complete in the current tree.**
  `MatmulDecls.lean` and `lower_matmul.py` are gone; L12 now requires source
  inequality plus 11/11 bit-identical execution, while the independent Nodup
  theorem remains a separate obligation.

Companion documents: `PLAN_TINYGRAD_COMPAT.md` (the longer arc),
`Tgrad/Ontology.lean` (stable product vocabulary), and `Tgrad/Spec/*`
(runtime work, mutable findings, growth cases, evolution events, live
conditions, and current work). `GROWING_TGRAD.md` records the full design and
the migration from mutable progress flags to exact-tree promotion evidence.

## 0. Shape of the codebase

The project now has two build roots with different responsibilities:

| Root | Owns | Must not own |
|---|---|---|
| `Tgrad` | product datatypes, compiler passes, renderer, runtime, FFI | roadmap state, maturity judgments, audit prose |
| `TgradSpec` | ontology pins, epistemic claims, architecture, findings, resource policies, work graph | runtime behavior or production dispatch |

Within `TgradSpec`, the separation is deliberate:

- `Ontology.lean` contains stable sorts and morphisms. It changes when the
  domain language changes, not whenever a bug is fixed.
- `Spec/Findings.lean` contains mutable claims with evidence and upgrade paths.
- `Spec/LiveConditions.lean` contains re-observable machine/repository limits,
  such as the single GPU and fixed `/tmp` namespace.
- `Spec/RuntimeWork.lean` describes repeatable product, verification, and
  specification work performed by the codebase.
- `Spec/Growth.lean` connects observations and findings to intended capability
  deltas, validators, acceptance, and rollback.
- `Spec/Evolution.lean` checks the intent → attempt → candidate → exact-tree
  checks → promotion event protocol.
- `Spec/Work.lean` is the current roadmap projection. It computes readiness,
  priority, file conflicts, and the authoring frontier, but its historical
  `Progress.complete` values are not durable promotion certificates.

This replaces the previous failure mode where fixed bugs remained as
constructors of an ontology and therefore continued to compile as if they
were current facts. The older `Tgrad/Model/*` modules are no longer imported by
the product or specification roots; they remain historical material until a
separate deletion change.

## 1. Why this ordering

The three safety fixes landed first because they were small, critical, and
host-local. View materialization then crossed Python, Lean FFI, the renderer,
the scheduler, and Metal while deliberately reusing the existing two-handle C
trampoline (`bHandle = 0` is the unary operation). Codegen can proceed beside
it only while their write sets are disjoint.

Codegen promotion is ordered by evidence, not by deletion: first build a
differential oracle, then broaden the generator, then type its stores, then
route production, then change the gates, and only then remove the
transcription. Performance is measured last because any earlier number still
measures the old implementation or perturbs the one-GPU verification queue.

## 2. Hard constraints on parallel execution

These are properties of this machine and this repo, verified, and they
bound everything below.

| Constraint | Evidence | Consequence |
|---|---|---|
| **One GPU** | `c/metal_alloc.m:26,167` — `g_device`, `g_queue` are process-global statics | All timing work is serial. Never run two benchmarks concurrently; a parallel agent doing an unrelated build will also perturb timings. |
| **No locks on device state** | `g_device`, `g_queue`, `g_lru`, `g_lru_count` have no mutex; `NSMutableDictionary` pipeline cache is not thread-safe | Correctness runs in *separate processes* are safe. Threads within one process are not. |
| **141 hardcoded `/tmp/tgrad_*` paths** | `grep -rhoE '/tmp/tgrad_[a-z_.]+' scripts/` → 141 distinct, only 7 files use `mktemp` | Two concurrent gate/devcheck runs clobber each other. **Parallel implementation is safe; parallel verification is not.** |
| **~239 MB `.lake` per worktree**, 3.5 GB free | `du -sh .lake` | Caps concurrent worktrees at ~3. Prefer shared-checkout + serialized builds over many worktrees. |
| **Gates write runtime evidence** | `scripts/gate.sh` writes gitignored `fixtures/gate_evidence/` | Evidence is per-sweep; never commit it. Umbrellas fail on clean checkouts until children run. Verification uses `tgrad-tests` + targeted Python checks when a full sweep is not appropriate. |

**Design rule that follows:** parallelism is applied to *authoring*,
not to *verification*. Agents write concurrently; a single integrator
builds and verifies serially. This is the opposite of the usual
default and is forced by the GPU and the `/tmp` collisions.

**Prerequisite W0** (unblocks parallel verification, optional): give
every gate script a run-scoped temp dir (`TGRAD_TMP="$(mktemp -d)"`)
instead of the 141 fixed paths. Mechanical, ~35 files, but it cannot
be verified without running the gates it edits — so it is only worth
doing if parallel verification actually becomes the bottleneck. Not on
the critical path.

## 3. Checked work graph

Single-writer file ownership is the organising principle: **parallel authors
must have disjoint `writes` sets.** Verification is independently constrained
by the resource policies in `Tgrad/Spec/LiveConditions.lean`.

Completed nodes:

- `foundation.renderer-runtime` (`f679bf7`)
- `foundation.view-algebra` (`f679bf7`)
- `correctness.buffer-shape` (`e1d5760`)
- `correctness.view-lifetime` (`e1d5760`)
- `correctness.view-readback-safety` (`e1d5760`)
- `view.materialize` (`e6241bd`)
- `codegen.typed-stores` (`1787c83`)
- `codegen.warp-parameter` (`75f856b`)
- `codegen.differential-harness` (`a6d5958`)
- `gates.semantic-codegen` (`aa67497`)
- `evidence.audit-tool` (`bdc01b0`)
- `codegen.route-sentinels` (`fd945b1`)

`codegen.warp-parameter` first widened generation without widening routing.
`fd945b1` then moved eligibility and dispatch together: production now routes
on `tcMatmulKernelDeclManualLoadWide`, and `tcLaunchDims` handles the collapsed
two-warp 64-wide tile without a zero grid dimension.

Two disjoint attempts were claimed:

1. `view.materialize`: indexed copy kernel plus Python/Lean readback route;
   exact candidate `790d413…` landed as `e6241bd` and its lease is released.
2. `codegen.differential-harness`: execute captured and generated kernels on
   identical inputs and preserve both outputs/source hashes on divergence;
   landed as `a6d5958` and its lease is released.

The frontier changed during planning: typed-store work initially occupied
`MatmulTc.lean`, `Metal.lean`, `UOp.lean`, and `L14_B_2_b.sh`, which correctly
excluded warp parameterization. Once that work landed as `1787c83`, the active
write set disappeared and the checked frontier widened back to three. The warp
worker then claimed and completed one slot; its lease closed. The differential
harness subsequently landed as `a6d5958`, making `codegen.route-sentinels`
dependency-ready. That route still writes `Pipeline.lean` and
`PythonFFI.lean`, so it remained excluded while materialization was active.
After exact-tree promotion at `e6241bd`, the computed safe frontier became
`codegen.route-sentinels`. After exact-tree promotion at `fd945b1`, it became
`codegen.delete-transcription`. This is the intended behavior of a live work
model.

The next dependency chain is:

```text
codegen.warp-parameter ---------\
codegen.typed-stores -----------+-> codegen.route-sentinels
codegen.differential-harness ---/
codegen.differential-harness -> gates.semantic-codegen [landed aa67497]
codegen.route-sentinels + gates.semantic-codegen
                                   -> codegen.delete-transcription [complete]
                                   -> perf.rebaseline [next]
                                   -> evidence.regenerate
                                   -> evidence.enforce-provenance
```

`codegen.delete-transcription` was proved ready by generated production
routing and the additive semantic differential, then completed by deleting the
two artifacts and retiring only the transcription-specific layer.

## 4. Parallel schedule

```text
landed      [view.materialize] e6241bd; write lease released
landed      [codegen.differential-harness] a6d5958
landed      [codegen.warp-parameter] 75f856b; renderer lease released
landed      [gates.semantic-codegen] aa67497; additive C3
landed      [evidence.audit-tool] bdc01b0; diagnostic, not fatal
landed      [codegen.route-sentinels] fd945b1; generated production route

integrate   one shared-build/test run at a time

next        performance rebaseline -> evidence regeneration -> fatal audit
            (exclusive GPU, clean tree, serial)
```

The deletion promotion is recorded, so the safe authoring frontier is
`perf.rebaseline`.
That does not authorize more worktrees blindly:
`LiveConditions.sourceTree` is tentative and must be re-probed against free
disk and current writers. Nor does it authorize parallel verification: route
and materialization checks require the Metal GPU, and Lean build artifacts are
shared. The differential harness itself now uses `mktemp`, but the historical
gates around it still contain fixed paths.

The highest-contention file is `Tgrad/PythonFFI.lean`. Materialization writes
it now; sentinel routing writes it later, after generator/store work. Encoding
that dependency avoids asking agents to coordinate overlapping edits by prose.

## 5. Verification protocol

Per workstream, before integration:

- **View materialization (passed on `790d413…`, landed as `e6241bd`):** contiguous control,
  transpose, multi-axis/partial/strided slice, reshape, expansion on both axes,
  chained movement, repeated readback, zero-size/invalid rejection, and
  temporary-parent lifetime matched numpy. Raw bf16 checks included NaNs,
  infinities, signed zero, and subnormals; the copy is `ushort`, not a numeric
  bf16 round-trip.
- **Generator/store/routing:** build `TgradSpec` and `Tgrad`, run
  `tgrad-tests`, then compare every sentinel numerically against both the
  captured kernel and numpy.
- **Differential harness (passed in `a6d5958`):** 11/11 sources differ and
  11/11 outputs are bit-identical. Changing one store from `c` to `c+2`
  preserved pairwise distinctness but produced 727,933 differing bytes. This
  proves `tileStoreOffsets_nodup` and execution differential are complementary:
  the theorem catches collisions; the harness catches wrong-but-distinct
  placement. The semantic gate must require both.
- **Integration:** one owner, serial, clean tree, `lake build TgradSpec
  Tgrad:shared tgrad-tests`, then `.lake/build/bin/tgrad-tests`, then targeted
  Python/Metal checks. Run the full gate sweep only after namespacing its
  temporary files or under an explicit single-run lock.

## 6. Scoping answers

Both open questions are now resolved.

### Coverage: 10 of 11, and the eleventh is cheap

The parametric generator emits **10 of the 11 sentinels**. Only
`64x64x64` fails, and only on `N`: `M` appears in the kernel body just
as a comment and a dispatch term, `K` only as `K/8` and the A-strides,
so `M ≥ 128` / `K ≥ 128` in the guard are over-strict relative to what
the body needs. The real constraint is `N`, which is hardcoded to a
128-wide block across 4 warps.

**Fix: parameterise the warp count**, `W = min(4, N/32)`, collapsing
the N grid dim when `N/(32W) == 1` — which is exactly what tinygrad
itself did for this shape. The captured kernel name confirms it:
`r_2_32_2_2_4_4_8` has seven components where the other ten have
eight (the N grid dim is gone) and `local.y = 2` instead of `4`.
Everything else — `upcast (2,4,4)`, 16 WMMAs, `acc0[32]`, the lane
bit-twiddles — is identical. Crucially this **preserves the existing
dispatch dims**, so `pickDispatchPlan_matches_capture` keeps
type-checking. Roughly 20 lines.

Rejected alternatives: predication wastes half the threadgroup and
changes the dispatch dims; the scalar fallback would put 4096
single-thread threadgroups on the shape L7 times and would likely turn
both L5 (bit-exactness) and L7 (ratio ≤ 1.5) red.

### Equivalence: promoted execution evidence

The review compared generated and captured kernels by substituting every
`alu` definition and evaluating each load/store as a concrete address,
resolving WMMA calls to operand-address tuples so variable renaming did not
matter: 384 index points per shape, 10 shapes, zero observed divergences.
That design input is now superseded by executable evidence. `a6d5958` executes
both routes and compares 240 MB over all 11 sentinels bit-for-bit;
`aa67497` makes it a mandatory additive L12 C3 layer while the old green
byte-equality layer still existed. The current deletion migration retired the
old layer and made the differential primary, so verification was strengthened
before migration pressure arrived.

Divergences found are all non-semantic: store emission order (all 32
addresses distinct, no aliasing), kernel and buffer names, and two
dead structures the generator emits *only to satisfy gate greps* —
`threadgroup bfloat tg_a[256]` / `tg_b[1024]`, never referenced, plus
`if (false) { threadgroup_barrier(...); }`. Those allocate 2.5 KB of
threadgroup memory per threadgroup and should be deleted before
benchmarking; they are already priced into today's numbers, so
removing them is upside-only.

### Perf: unknown until the symmetric rebaseline

The source-level hypothesis is favorable: the capture hoists 16 `alu`
temporaries and uses shifts, while the generator computes fewer expressions
whose power-of-two multipliers should lower to shifts. That is a compiler
hypothesis, not performance evidence.

The committed `L13_F.json` ratio cannot settle the question. Its tinygrad
baseline is noisy, its evidence hashes do not match the shipped tree, and the
two runtimes are not measured across a symmetric boundary. Therefore the
performance state is **unknown**, including for `8192³`. The first admissible
number is the serial, same-session, dispatch-boundary-matched run represented
by `perf.rebaseline`; a regression is an acceptable result and must be
reported rather than hidden behind the old fixture.

The regeneration attempt produced stronger, direct repeatability evidence.
On the same `e90607f` code, same GPU, frozen baseline, and `30/30`
configuration, three consecutive L11 runs missed 2/50, 25/50, and 10/50 rows
with `ratio_median` 1.186/1.502/1.300 and `ratio_max`
1.655/3.667/2.552. The twelve-fold swing in misses is larger than the effect
the fixed threshold is trying to detect. Separately, L12's generated path was
50/50 correct while its diagnostic changed from median/max 2.38/4.24 and 37
misses at `1/1` sampling to 1.18/1.41 and zero misses at `30/30`, without a
code change.

These are not the promised honest benchmark—the denominator remains frozen
and the runtime boundaries remain asymmetric—but they falsify the existing
pass/fail method empirically. L12 promotes correctness only;
`perf.rebaseline` must measure both sides live and interleaved, retain raw
paired distributions, estimate within-run and between-run variance, and
predeclare any decision rule from that variance. The honest current codegen
diagnostic is roughly 1.2–1.5x the frozen tinygrad baseline, noisy and
occasionally worse. The historical 0.9354 median measured replayed tinygrad
MSL and was never a generated-code result.

### `.numpy()` materialization: validated candidate, with bounded debt

`copy_kernel.msl` is `float`, rank-1, and carries its body as two
string literals with no index UOp — strides cannot enter it. The
synthetic indexed kernel has the right two statements but a hardcoded
index. Neither is reusable. The real template is
`scalarMatmulKernelDeclWithIdx` + `Pipeline.realizeView`, which
already proves the mechanism end to end.

Commit `e6241bd` adds the missing declaration and realization path without
a C edit: handle zero is impossible in the registry, so
`tgrad_matmul_view(viewHandle, 0)` is reserved as the unary materialization
operation. The copy consumes rangeify's index UOp, delinearizes one flat output
thread into canonical scheduler coordinates, copies through `ushort` so all
bf16 payload bits survive, validates source bounds/allocation/index width, and
returns a contiguous Tensor.

This is the first runtime operation whose addressing is governed directly by
the scheduler's rangeified output. Remaining debt is lifecycle and naming:
the overloaded matmul symbol should eventually become
`tgrad_tensor_materialize`, repeated readback recompiles and registers a fresh
temporary, and the append-only registry/cache policy needs its own work item.

### Two corrections to the original framing

- `check_fixture_drift.sh` is **not** in the blast radius; it tracks
  six dtype/shape/symbolic JSONs, no `.msl`.
- Before `1787c83`, `L14_B_2_b.sh` counted source occurrences and
  `MatmulDecls` supplied nearly all of them while `MatmulTc` supplied none.
  That pressure is gone: `tcManualLoadMatmulBody` now emits 32
  `Stmt.storeIndexed` and 16 typed loads, the gate checks emitted output, and
  `tileStoreOffsets_nodup` is a checked non-aliasing obligation. Deleting the
  transcription therefore no longer weakens the typed-addressing predicate.

### L12's current predicate

Byte-equality died with the transcription. Its successor is
**differential result equivalence**. In one process it compiles both
the captured MSL and the generated kernel, dispatches both over one pair of
random bf16 inputs, and asserts the output buffers are bit-identical.
That is strictly stronger than comparing source bytes, and stronger
than the existing `allclose`-vs-numpy check, because it compares
against tinygrad's actual kernel. Add the anti-cheat inverse too —
asserts the rendered source is *not* byte-equal to the capture, so the
transcription cannot be quietly re-vendored. The harness supports a
ULP-tolerance fallback because `metal_alloc.m:134` compiles with
`options:nil`, but all 11 current sentinels pass bit-identically.

### Evidence provenance: observable before enforceable

`bdc01b0` turns the manual provenance review into a calibrated auditor. Its
baseline found 37/37 files naming an absent commit. The partial serial
regeneration at `7c7dc0f` repaired two gate blockers and retained 11 files
genuinely produced by their current scripts at `e90607f`; the current audit
reports 26/37 absent commits, 76/115 unresolved non-transient hashes, 28
roll-up disagreements, and 17 writer-key mismatches. Synthetic evidence tied
to HEAD passes. The audit is deliberately diagnostic. Making a known-red
check fatal would create a blocker without repairing its subject.

The attempted sweep also falsified two assumptions about the gate harness
itself. `L14_B_1` had never been runnable in this repository layout: its
`Tgrad/fixtures/pipeline` path raised `FileNotFoundError` before the first
assertion, despite committed green evidence. `a62a784` repairs the path.
L12's anti-replay predicate then matched a Lean comment accurately describing
the deleted `readFile` behavior. `b56bed4` strips line comments before matching
and verifies that a real call is still rejected. The general rule is to fix a
text predicate's semantic approximation, not make source documentation less
accurate so the grep turns green.

The checked graph therefore separates three items:

```text
evidence.not-tracked-check [landed]
perf.rebaseline -> evidence.regenerate-runtime -> evidence.enforce-hash-integrity
```

Committed gate evidence was retired: `fixtures/gate_evidence/` is gitignored and
`check_gate_evidence_not_tracked` fails if it is re-tracked. Regeneration is an
owner-authorized serial GPU operation that must produce runtime evidence for all
37 gates from one measured tree. The `7c7dc0f` tree is recorded as
an abandoned candidate: its two gate repairs are retained, but
L11 stayed red and no partial snapshot was promoted. Only after the paired
performance method is stable and complete regeneration succeeds does hash
integrity enforcement become honest.

## 7. Risks

- **W7 deletes the only reference implementation.** Mitigate by
  keeping the captured `.msl` fixtures (they are already the
  differential reference) and only deleting the Lean transcription.
  Never delete both in one change.
- **The perf regression in W8 will be real and visible.** It should be
  published as such. It is the first honest performance number the
  project will have produced, and framing it as a regression against a
  dishonest baseline is the wrong framing — the old number was not a
  measurement of Tgrad's codegen.
- **Parallel agents cannot verify their own work** on this machine
  (§2). Any agent that reports "verified" after running benchmarks
  concurrently with another agent has produced noise.

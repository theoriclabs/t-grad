# Reaching tinygrad parity

`GROWING_TGRAD.md` specifies how a change moves safely from intent to
promotion. It is a transmission. This document supplies the parts it
does not have: **a destination, a way to measure distance to it, and
the order of travel.**

The gap is concrete. `Spec/Work.lean` carries `goalDistance` as a
hand-authored `Nat`. Nothing computes it, so nothing can contradict
it. A framework that can prove a change was legitimate but cannot say
whether it moved toward tinygrad will happily certify a thousand
legitimate changes that arrive nowhere.

## 1. Where we actually are

Stated plainly, because the roadmap only works if the origin is honest:

| | tinygrad | Tgrad |
|---|---|---|
| ops | ~50 | **1** (matmul) |
| dtypes | 14 | **1** (bf16) |
| backends | 6+ | **1** (Metal) |
| rank | n-D | **2-D** |
| autograd | yes | no |
| JIT / BEAM | yes | no |

Two structural facts matter more than the table:

- `Linearize` handles **5 of 21** `UOp` constructors.
- `graphRewriteBottomUp` — a real pattern matcher with 22 rules — has
  **one caller in the repository, a CLI subcommand.**

So there is no general lowering path. Matmul reaches the GPU through a
matmul-shaped generator. Everything else would need its own.

## 2. The destination, in three layers

"Parity" is not one property. Splitting it is what makes it checkable.

**Surface parity** — behaviour. tinygrad's own test suite passes
against Tgrad. Measured as a count, per file.

**Structural parity** — the architecture that makes the surface
reachable and maintainable:

- every semantic payload typed, not carried in `String`
- every morphism total or honestly partial — no `panic!`-to-`default`
  on a reachable path
- one spine: every op lowers through the same path
- backends are instances of a renderer interface, not forks

**Epistemic parity** — how the codebase knows:

- every capability claim traces to a foreign oracle (§4)
- every numeric threshold has characterised variance before it gates
- every check has been observed red

The review is the argument for including the last two. This project
reached a passing bf16 matmul with 37 green gates, and the benchmarked
path was replaying tinygrad's own captured kernel off disk. Surface
progress without structural and epistemic shape produces a demo that
collapses the moment you lean on it.

## 3. The metric: tinygrad's own tests

Parity should be measured by the suite tinygrad ships, run against
Tgrad through the FFI. That suite is:

| group | files | size | backend needed |
|---|---|---|---|
| `test/null` | 55 | 571 KB | none |
| `test/unit` | 44 | 376 KB | one |
| `test/backend` | 42 | 644 KB | each |

`test/backend/test_ops.py` alone is 196 KB of differential tests.

Why this rather than a hand-written checklist:

- **It is authored by someone else.** An agent cannot satisfy it
  without implementing the feature. This is the whole game (§4).
- **It is exhaustive in ways a checklist is not.** It encodes a decade
  of edge cases nobody on this project would think to write.
- **It is already partitioned along the axis that constrains us.**
  tinygrad's own README splits tests by backend requirement. The 55
  `null` files need no GPU, so they run in CI, run fast, and run in
  parallel — which matters because verification here is serial on one
  Metal device.
- **It directly encodes the stated design goal.** The project's claim
  is interface continuity with tinygrad. This measures exactly that.

The number to move: **files passing, and tests passing within them.**
Publish it. Let it be embarrassing at first.

There is an immediate, unusually cheap first target.
`test/null/test_graph_rewrite.py` is 23 KB testing precisely the
pattern-matching rewrite engine Tgrad already has and does not use.
That is a foreign oracle, available today, for a component that is
already written.

## 4. The law of foreign oracles

**A check whose expected output was produced by the same codebase that
produces the actual output is not evidence.**

This is the single strongest generalisation the review produced, and
it is empirical, not aesthetic. Classify every check by where its
expected value comes from:

- **Foreign** — tinygrad's tests, tinygrad's captured kernel output,
  PyTorch, numpy. Trustworthy.
- **Internal-differential** — two independently written paths in this
  repo compared on the same input (scalar vs WMMA). Weaker, but real:
  both must be wrong in the same way to pass.
- **Self-referential** — expected value derived from the
  implementation. Not evidence.

The session's record, unedited:

*Foreign checks that caught real defects:* the numpy view differential
found two silent wrong-answer bugs (a multi-axis slice reading the
wrong column, a broadcast on the wrong axis); the captured-kernel
differential caught two injected codegen faults that the build **and**
the `Nodup` theorem both passed.

*Self-referential checks that proved to be vacuous:* L12's byte
equality (a round trip against a transcription of the thing under
test); `_rendered_kernel_source()` building an f-string and grepping
it; `if (false) { threadgroup_barrier(...); }` emitted so a grep would
find it; a `.storeIndexed` line count that scored 48 generated
statements as 3 and would have scored the same code written longhand
as 48; L15 grading prose in a markdown file; `Tests.lean` asserting
nothing at all and exiting 0.

The split is clean. **Rule: no capability is marked complete on
self-referential evidence alone.** `Spec/Epistemic.lean` is where this
classification belongs.

## 5. The order of travel

The sequence is forced by one observation: there is no general path,
so every op added now costs a bespoke generator, and 50 of those is
the same architecture that produced the replay problem.

**Phase 0 — build the instrument.** The tinygrad-suite shim and its
score. Start with `test/null`: no GPU, so it runs in CI and in
parallel. Everything after this is measured by it, so it comes first.

**Phase 1 — one spine.** Make `a + b` reach the GPU through
`rangeify → schedule → linearize → render`, then re-express matmul
through the same path and delete the special case.

Doing elementwise `add` *after* matmul looks like going backwards. It
is not. Matmul is done *specially*; `add` is the simplest thing that
forces the general path to exist. **Exit criterion: `add` and `matmul`
produce kernels through the same code, and the elementwise slice of
`test_ops.py` passes.** After this, ops are table entries.

**Phase 2 — the movement algebra.** `View` with `mask`,
`ShapeTracker` as a list of views, `compose` / `simplify` / `invert`.
Pure, total, no GPU, theorem-dense — the best Lean showcase in the
codebase, and `test/null` and `test/unit` both carry oracles for it.

**Phase 3 — a second backend (CPU).** Deliberately early. It is the
forcing function that proves the renderer is an interface rather than
a Metal-shaped blob, and it makes the oracle runnable without the GPU
— which the metric in §3 depends on. The architecture requirement
falls out of the measurement requirement.

**Phase 4 — op breadth and dtypes.** Now cheap. `Dtype.lub` already
exists and is correct; it is simply dead. Oracles: `test_ops.py`,
`test_dtype*.py`.

**Phase 5 — autograd.** ~200 lines in tinygrad because it is a graph
transform. In Lean it is a fold over `UOp` — trivial if the spine is
real, impossible otherwise. Oracle: `test_gradient.py`.

**Phase 6 — JIT, BEAM, multi-device, further backends.**

## 6. What agents need in order to be the transformation

The existing protocol handles authorisation, leases, scope and
promotion, and it worked: two agents ran in parallel through this
whole session without one conflict, coordinating by explicit file
claims. Five additions, each earned:

1. **Every work item names its foreign oracle at intake.** An item
   that cannot name one is not admissible. This kills unfalsifiable
   work before it is authored rather than after it is reviewed.

2. **Distance is computed, never authored.** `goalDistance` becomes
   the count of failing oracle tests in the item's declared scope. An
   agent then cannot claim progress without moving a number it does
   not control.

3. **Falsification is an acceptance obligation.** A check counts only
   once it has been observed red. Applied five times this session, it
   caught three vacuous checks — including one where the theorem and
   the build both stayed green while the kernel was wrong.

4. **Numeric thresholds carry a variance obligation.** No threshold
   gates anything until its run-to-run spread is characterised. The
   perf predicate reported 2/50 failures, then 25/50, then 10/50 on
   consecutive runs of identical code; its variance exceeds the effect
   it purports to detect, so any single verdict from it is a coin flip.

5. **Parallelism applies to authoring, not verification.** One GPU;
   141 hardcoded `/tmp/tgrad_*` paths; a shared Lean build tree.
   Agents write concurrently, one integrator verifies serially.

## 7. What this plan deliberately does not do

**It does not gate on performance.** Not because performance does not
matter, but because a threshold whose measurement is not reproducible
manufactures false verdicts in both directions. Performance becomes a
reported distribution, measured with both sides in one run rather than
against a frozen baseline, and it gates nothing until §6.4 is
satisfied.

**It does not chase the 37-gate ratchet.** The gates keep real value
for properties tests cannot see — purity, absence of filesystem reads
on the product path, theorem existence. But they stop being the
measure of progress. A self-authored ratchet is a self-referential
oracle at the project scale, and §4 applies to it too.

## 8. First move

Build the shim and publish the first score, before writing any feature
code. It will be a low number. That is the point: it is the first
statement about this project's distance to tinygrad that was not
produced by this project.

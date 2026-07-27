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

## 1. Where we actually are — and what we do not yet know

Stated plainly, because the roadmap only works if the origin is honest:

| dimension | tinygrad target | Tgrad today |
|---|---|---|
| revision | **not pinned yet** | this repository tree |
| public Tensor surface | generated inventory still missing | bounded Python surface centred on bf16 matmul and movement views |
| UOp / Ops surface | generated inventory still missing | 21 local constructors; partial lowering |
| dtypes | generated inventory still missing | bf16 is load-bearing; parts of a wider lattice exist but are not on the path |
| backends | target profile not declared | Metal only |
| rank | n-D | 2-D product path; bounded movement algebra underneath |
| autograd | part of upstream contract | absent |
| JIT / optimization search | part of upstream contract | absent |

The missing numbers are deliberate. Current tinygrad documentation exposes a
broad and changing Tensor, dtype, UOp, runtime, JIT, multi-device, NN, and
state-management surface. A count copied into prose is stale as soon as
upstream moves. The first parity action is therefore to pin one official
revision and generate source, API, dtype, Ops, backend, and test manifests.
Until that exists, a percentage such as “7% compatible” has no denominator and
must not enter evidence.

Two structural facts matter more than the table:

- `Linearize` handles **5 of 21** `UOp` constructors.
- `graphRewriteBottomUp` — a real pattern matcher with 22 rules — has
  **one caller in the repository, a CLI subcommand.**

So there is no general lowering path. Matmul reaches the GPU through a
matmul-shaped generator. Everything else would need its own.

## 2. The destination, in three layers

"Parity" is not one property. Splitting it is what makes it checkable.

**Surface parity** — behaviour. Generated upstream requirements and applicable
upstream tests pass against Tgrad. Per-file/test counts are a useful progress
projection, but cannot replace mandatory requirement cells or hide exclusions.

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
Tgrad through the FFI. An unpinned inspection used while writing this plan
found the following shape. These counts are orientation, **not** the
compatibility denominator; the pinned manifest produced in Phase 0 replaces
them:

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
self-referential evidence alone.** `Spec/Parity.lean` records oracle origin,
subject/verifier identity, equivalence relation and validator calibration;
`Spec/Epistemic.lean` records certainty and the upgrade path.

## 5. The order of travel

The sequence is forced by one observation: there is no general path, so every
operation added now tends toward a bespoke generator—the architecture that
produced the replay problem.

**Phase 0 — build the instrument.** The tinygrad-suite shim and its
score. Start with `test/null`: no GPU, so it runs in CI and in
parallel. Everything after this is measured by it, so it comes first.

**Phase 1 — executable semantics.** Establish dtype/scalar meaning, current
movement-UOp/indexing semantics, the pinned UOp vocabulary, and independent
tensor/kernel evaluators. These are CPU-only and give every later pass a
meaning to preserve.

**Phase 2 — one spine and CPU.** Make `a + b` traverse
`rewrite → rangeify → schedule → lower → linearize → render`, add a simple CPU
target through the same interfaces, then re-express matmul through the common
path.

Doing elementwise `add` *after* matmul looks like going backwards. It
is not. Matmul is done *specially*; `add` is the simplest thing that
forces the general path to exist. **Exit criterion: `add` and `matmul`
produce kernels through the same staged compiler, and the imported elementwise
slice passes on CPU and Metal.** After this, operation breadth is a vertical
semantic slice, not a new pipeline.

**Phase 3 — safe runtime boundary.** Introduce session-owned generational
handles, a generated generic ABI, buffer/lifetime validation, explicit
diagnostics and structured traces before substantially widening Python/Metal
usage.

**Phase 4 — vertical operation families.** Elementwise and dtype promotion,
movement/indexing, reductions, matmul/convolution, random and effects. Every
packet crosses API, evaluator, compiler, CPU, Metal, numerical and upstream
tests together.

**Phase 5 — autograd, TinyJit and ecosystem.** These become graph/runtime
transformations over the common spine rather than parallel implementations.

**Phase 6 — multi-device, backend profiles, optimization and continuous
upstream intake.** Performance instrumentation exists early, but a performance
claim is qualified only after semantic closure and measured repeatability.

## 6. What agents need in order to be the transformation

The existing protocol handles authorisation, leases, scope and
promotion, and it worked: two agents ran in parallel through this
whole session without one conflict, coordinating by explicit file
claims. Five additions, each earned:

1. **Every new work item names an independent oracle relation at intake.**
   Usually this is an upstream test or live runtime; sometimes a mathematical
   law, sanitizer, or separately implemented evaluator is stronger. The key is
   that expected output does not depend on the path being changed.

2. **Distance is computed, never authored.** `goalDistance` becomes
   a generated vector of mandatory requirement cells, foreign tests,
   downstream gaps, evidence strength, risk, cost and scarce-resource demand.
   An agent cannot claim progress by editing a number it controls.

3. **Falsification calibrates validators.** Promoting evidence cites a
   versioned validator whose relevant mutations have been observed red.
   Applied five times this session, this
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

## 9. The system that grows Tgrad

The roadmap is not a list of features. It is a feedback system with two
changing inputs: Tgrad and tinygrad.

```text
 official tinygrad revision ──capture──> versioned contract
                                           │
 current Tgrad tree ───────observe─────────┤
                                           v
                                   coverage-cell gaps
                                           │ shape
                                           v
                                    executable work
                                           │ lease
                                           v
 agent + immutable base tree ───────> candidate tree
                                           │ falsify + validate
                                           v
                                  admissible evidence
                                           │ promote
                                           v
                                    new current tree
                                           │
                         upstream drift ────┴── repeat
```

This separates six things that the earlier process blurred:

1. **Target** — an immutable description of one tinygrad revision.
2. **Current state** — observations about one Tgrad tree, with epistemic state.
3. **Gap** — a difference between target and current state. It is not yet work.
4. **Work packet** — a transaction an agent can actually own and reverse.
5. **Evidence** — an observation tied to both revisions and an immutable
   scenario manifest.
6. **Promotion** — the atomic update that changes what the project claims.

The checked form is `Tgrad/Spec/Parity.lean`. `targetUpstream` and
`targetContract` are deliberately `unknown`: the plan does not fabricate the
pin it says is missing. `ProgramTemplate` is deliberately weaker than
`Work.WorkItem`: long-horizon intent is not mutation authority.

The loop has two modes:

- **Convergence mode:** keep the upstream pin fixed while Tgrad closes cells.
- **Tracking mode:** after a profile is conformant, ingest a newer upstream
  pin, compute drift, and close only the new gaps.

Mixing the modes makes distance meaningless. Moving the target while claiming
progress can hide regressions; freezing forever makes “tinygrad compatible” a
historical claim. A target change is therefore an explicit promotion of its
own, with before/after manifests and a newly computed gap set.

## 10. What “full parity” means

There is no context-free full parity. A claim has the form:

```text
Tgrad tree T is conformant to tinygrad revision U
for profile P over required domains D and dimensions K,
under scenario manifest S, with evidence E.
```

The project should publish several profiles rather than one inflated badge:

| profile | meaning |
|---|---|
| semantic core | dtype/scalar semantics, movement/view algebra, UOp, rewrites, scheduling, lowering |
| public API | user-visible Tensor, autograd, JIT, NN/state, and interop behaviour independent of backend breadth |
| Metal | public/core semantics on the Metal backend, including workloads |
| portable | the same compiler and API on CPU plus at least one accelerator backend |
| ecosystem | state loading, optimizers, training/inference examples, serialization and library-facing behaviour |
| all declared backends | every backend row the project explicitly claims, never hardware inferred from another row |

Each profile is a matrix, not a percentage:

```text
domain × dimension × dtype family × shape class × backend × mode
```

Dimensions are distinct because they fail independently:

- API: names, signatures, defaults, exceptions and return shapes.
- Semantic: graph meaning and observable values.
- Numerical: promotion, rounding, NaN/Inf, overflow and tolerances.
- Gradient: reverse-mode graph and accumulation behaviour.
- Compiler: all supported operations traverse the declared common spine.
- Runtime: allocation, ownership, synchronization, errors, caching and threads.
- Ecosystem: NN/state/JIT/workload compatibility.
- Backend: the same contract on each declared target.
- Performance: a distribution over a workload and host profile.

Compatibility requires the first eight for the declared profile. Performance
is a separate claim. This prevents both common category errors: calling a fast
kernel compatible, and calling a correct interpreter performant.

A cell is conformant only when all of these hold:

1. the upstream and Tgrad revisions are exact;
2. the scenario inventory is immutable and hashed;
3. the observed state is confirmed, not tentative;
4. at least one promoting oracle is foreign, mathematical, or genuinely
   independent;
5. the validator has been observed red under a relevant mutation;
6. exclusions and unavailable hardware are explicit;
7. evidence is fresh for the subject tree;
8. no mandatory child cell is merely averaged away by a percentage.

“Full” for profile `P` means every required cell is conformant. A single
drifted cell makes the profile drifted. Missing hardware makes a backend cell
unknown or blocked, never green.

### 10.1 Requirement families generated from upstream

The research pin observed the following families. The extractor must derive
the exact rows and selectors; this table defines the shape, not hand-authored
counts.

| family | contract to generate | primary upstream oracle families |
|---|---|---|
| package/API | top-level exports; Tensor names, signatures, defaults, special methods, containers, exceptions | `test/backend/test_tensor.py`, `test/test_tiny.py`, tensor data/IO/invalid/getitem units |
| dtype/scalar | concrete and weak dtypes, aliases, promotion/LUB, defaults, accumulation, cast/bitcast/truncation, BF16/FP8 storage | null/unit/backend dtype suites and transcendental tests |
| UOp/rewrite | Ops constructors and typed payloads, structural identity, shape/dtype inference, symbolic range, UPat/matcher/rewrite, stage validity | UOp, symbolic, pattern matcher, graph rewrite, spec and fuzz suites |
| movement/index | reshape, permute, expand, pad, shrink, flip, masks, negative axes/steps, symbolic shape, contiguous views, assignment/aliasing | indexing/getitem/setitem, movement operation tests and shape fuzzers |
| schedule/memory | callification, rangeification, realization, dependency/effect ordering, copies, buffer identity, memory lifetime/reuse | schedule, rangeify, indexing, memory-planner, schedule-cache and setitem-schedule tests |
| autograd/function | gradient/backward, accumulation, detach, repeated/higher-order behaviour, custom gradients, calls, parameters and multiple outputs | gradient, function, call/callify and training tests |
| codegen | decomposition, typed optimization actions, tensor cores/search, lowering, linearization, control flow, target legalization and renderer failures | linearizer, renderer, instruction-selection, kernel-opt, tensor-core and process-replay suites |
| runtime/device | device names, buffers/views/lifetime, alloc/copy/sync, compile/program/graph execution, interop, profiling and diagnostics | backend graph/interop/subbuffer/zero-copy/wait/profiler and device suites |
| TinyJit | capture/replay phases, input replacement, symbolic values, mutation/aliasing, random, graph batching, multi-device, lifetime and footguns | unit/backend JIT, JIT cases/footguns, symbolic JIT and graph tests |
| NN/state/models | layers, optimizers, state discovery/loading, serialization, archives/filesystems, ONNX and official model/train examples | backend NN/optimizer, tensor IO/state units, model and end-to-end tests |
| multi-device | tuple device identity, shard/replicate/gather, MULTI/MSELECT/MSTACK, axis propagation, collectives and JIT graphs | null/unit/backend multitensor and allreduce tests |
| backend profile | runtime module, renderer/compiler variant, architecture, OS/hardware, applicable CI selectors and compile-only rows | pinned upstream workflow matrix, not a made-up Cartesian matrix |
| performance | compile speed, driver/dispatch speed, model/scheduler speed and kernel/codegen speed | pinned speed workloads with live paired repeated sessions |

The candidate snapshot contained 82 upstream `Ops` constructors, 17 concrete
scalar dtypes plus weak types, and 16 runtime modules (eight highlighted public
compute runtimes). These numbers are useful extractor regression checks only
after Tgrad reproduces the pin; they are not prose-owned completion targets.

## 11. The ideal codebase shape

The ideal architecture is organised around meaning, transformation, target,
and evidence. It is not organised around the order features happened to be
ported.

```text
Tgrad/
  Contract/                 pinned upstream vocabulary and generated manifests
  Semantics/                DType, scalar values, promotion, pure Tensor meaning
  Shape/                    movement/view algebra, masks, symbolic/index algebra
  IR/                       UOp, Ops payloads, UPat, rewrites
  Evaluate/                 pure reference evaluator for supported IR
  Frontend/                 lazy Tensor graph construction and Python contracts
  Compiler/
    Schedule/               realization boundaries and schedule formation
    Lower/                  graph-to-kernel/index lowering
    Optimize/               typed Opts, memory planning, search
    Linearize/              structured program ordering and control flow
  Backend/
    Target.lean             target capabilities and renderer/runtime interfaces
    CPU/                    simple oracle-capable backend
    Metal/                  renderer, compiler bridge, allocator, queue
    <future backend>/       instance of the same contracts
  Autograd/                 graph transform over shared IR
  Jit/                      capture, replacement, replay, invalidation
  NN/                       layers, optimizers, state and serialization
  Verification/             properties and adapters; never imported by product
  Spec/                     epistemic state, findings, work, evolution, parity

python/
  tgrad/                    compatibility surface, generated signatures, adapters

fixtures/upstream/<rev>/    immutable generated manifests, exclusions and hashes
tests/
  unit/                     local semantic units
  property/                 laws and generated cases
  differential/             live foreign/reference comparisons
  conformance/              imported upstream-suite shards
  workload/                 end-to-end model/training/state cases
  performance/              paired raw observations and analysis
```

This is a target ownership map, not an instruction to perform a directory
rewrite first. Move code when a vertical slice needs the boundary; avoid a
large path-only refactor that produces no new conformance evidence.

### 11.1 The semantic nucleus

There must be one definition of dtype promotion, scalar operations, shape
indexing, and UOp meaning. Frontend, evaluator, optimizer and backends consume
it. Today semantic facts are split among Python, Lean, string payloads, and
Metal behaviour. That allows routes to disagree silently.

The pure reference evaluator is strategically important. It is not the fast
runtime. It provides:

- an executable meaning for UOps before a backend exists;
- a differential path independent of lowering and rendering;
- a place to state optimizer-preservation properties;
- cheap CPU-only validation for agents and CI;
- a way to distinguish semantic defects from backend defects.

### 11.2 One compiler spine

Every operation follows:

```text
Tensor API → lazy UOp graph → rewrite → schedule → lower/index
           → optimize → linearize → target renderer → runtime
```

Special kernels remain valuable, but they are optimization choices selected
behind this spine. They do not own separate public semantics, shape rules,
allocation policy, or evidence. A new operation is complete only when both the
reference evaluator and at least one compiled backend implement the same typed
node. A backend is complete only when it consumes the common lowered program.

### 11.3 Interfaces are capability contracts

The backend boundary must name:

- supported dtypes and vector/tensor-core forms;
- address spaces, indexing width and alignment;
- control flow, barriers and synchronization;
- launch geometry limits;
- allocation/import/export and ownership;
- compilation, cache identity, diagnostics and teardown;
- device enumeration, copy and multi-device primitives.

An unsupported capability returns an explicit error before dispatch. It never
falls through to `panic!`, an empty string, a zero grid, stale output, or a
silently slower route whose cost is unbounded.

### 11.4 Specification is out of the product dependency cone

The product cannot import roadmap, findings or evidence data. The spec may
inspect product types and generated manifests. Verification may execute the
product. Promotion may update the spec. The reverse dependencies are forbidden.
This keeps universal specification from becoming runtime metadata theatre.

### 11.5 One vocabulary, stage-certified graphs

tinygrad gets leverage from a unified UOp vocabulary. Lean can retain that
leverage while ruling out stage-invalid graphs. The target is an arena-backed,
hash-consed `RawGraph` with stable node identifiers and typed attributes,
wrapped by a stage certificate:

```lean
inductive Stage | tensor | scheduled | kernel | linear | target

structure Graph (stage : Stage) where
  raw   : RawGraph
  valid : ValidAt stage raw
```

This is preferable both to five unrelated ASTs and to one unrestricted
recursive UOp accepted everywhere. It preserves a common pattern/rewrite
vocabulary while making conversions explicit. Graph nodes do not contain raw
device pointers; semantic attributes do not escape through `String`; every
stage boundary returns structured diagnostics rather than panic-to-default.

Each pass has separately named mechanism, validation, semantic law and policy:

```text
run pass → validate output → compare semantics → record trace
          policy decides when to run; performance decides whether it helped
```

That separation prevents a heuristic from becoming semantics and prevents a
passing build from standing in for preservation.

### 11.6 Session-owned runtime and generated ABI

Python should own a session reference, opaque generational tensor handles,
Python protocol normalization, and exception translation. It should not own
Metal pointers, canonical graph metadata, route selection, TC eligibility,
output allocation or view lifetimes.

`Runtime.Session` owns the graph arena, tensor store, devices, buffers, caches,
events, diagnostics and teardown. Backend state is session-scoped,
synchronized, bounded and destructible. Cache identity is content plus target
and compiler configuration—not a reusable pointer address. A versioned
operation schema generates Lean exports, C declarations, Python bindings and
ABI conformance tests, replacing operation-specific FFI growth.

The target migration is therefore:

```text
Python Tensor(_session, _id)
  → generic tensor_apply(opcode, handles, typed attributes)
  → session-owned lazy graph
  → realize/copyout/release
```

This lifecycle work starts as soon as the CPU/common compiler interfaces are
stable. Broadening the public API on top of append-only registries, raw pointer
identity, missing bounds checks and process-global unsynchronized Metal state
would multiply the current safety debt.

## 12. The evidence lattice

Not all passing checks justify the same claim. Use this ascending lattice:

1. **Inventory:** symbol, signature, constructor and test manifests agree.
2. **Compile/exhaustiveness:** the mapping is total over the pinned vocabulary.
3. **Unit:** known examples exercise one implementation.
4. **Property:** laws cover generated values and structural cases.
5. **Metamorphic:** related inputs/operations preserve expected relations.
6. **Differential:** a live independent implementation agrees on the same input.
7. **Upstream suite:** the foreign compatibility contract passes unchanged or
   through a minimal, audited adapter.
8. **Backend matrix:** the contract holds on each actual declared target.
9. **Workload:** models, training steps, JIT and state round trips compose.
10. **Performance distribution:** paired repeated sessions support a scoped
    optimization statement.

Higher evidence does not erase lower invariants. The codegen differential and
`tileStoreOffsets_nodup` taught exactly this: the theorem detects collisions;
the differential detects wrong-but-distinct placement. Neither subsumes the
other.

Every evidence record carries:

- upstream revision;
- Tgrad subject tree;
- verifier tree and validator version;
- adapter hash and equivalence-relation version;
- scenario-manifest hash;
- oracle origin;
- command/environment/backend profile;
- raw artifact hash;
- exclusions, uncertainty and failure classification.

Falsification calibration belongs to the referenced validator. A normal
passing observation does not rerun every sabotage; it cites a validator whose
versioned falsifier manifest has already been observed red. If validator code,
adapter, relation or calibration changes, prior evidence is stale.

Generated scorecards are views over these records, never hand-authored truth.
Changing a test adapter, exclusion, tolerance, denominator, or expected result
invalidates affected evidence and is reviewed as a contract change.

Promotion obligations are generated from the affected requirement dimensions.
A certificate may reference checks satisfying those obligations; it may not
declare a smaller obligation set for itself. This closes the same laundering
hole at the work level that generated manifests close at the coverage level.

## 13. From a parity gap to agent-sized work

A gap is often too broad to implement. “Add reductions” is not an admissible
work item. The work shaper repeatedly splits until every packet has all of the
following:

| field | required answer |
|---|---|
| target | exact upstream revision, profile and coverage cells |
| base | immutable Tgrad tree |
| authority | owner goal/evidence repair/delegation and its finite effect budget |
| delta | one observable capability change |
| preconditions | promoted work/evidence this packet relies on |
| read/write/effect set | exact paths, subtrees and generated families; no hidden outputs |
| resources | build tree, tmp namespace, GPU, timing lane, evidence store |
| oracle | foreign or independent expected behaviour |
| validators | verifier tree, calibrated validator IDs, commands and expected artifacts—not “tests pass” |
| falsifiers | mutations expected to turn each promoting check red |
| recovery | revert, split, or re-observe action for each failure class |
| promotion | exact coverage/work/evolution changes after integration |

Use vertical slices for product breadth:

```text
Tensor.add(float32, broadcast shape)
  API → graph → evaluator → rewrite → schedule → CPU → Metal → gradient → tests
```

This is preferable to parallel horizontal projects named “frontend,”
“scheduler,” and “renderer,” which can each finish locally while no user
operation works. Horizontal packets are justified only for shared substrate
that multiple immediately following slices consume: pinned manifests, scalar
semantics, movement/view algebra, reference evaluator, target interface, test
harness.

### 13.1 Mechanical split rules

Split a template when any of these is true:

- it has more than one independently observable delta;
- implementation and oracle would be authored by the same agent;
- write sets overlap another ready packet unnecessarily;
- CPU and GPU validation have different failure/recovery modes;
- correctness and performance would be promoted together;
- unsupported cases cannot be enumerated;
- rollback cannot remove the change without unrelated loss;
- expected runtime exceeds one integration window;
- its acceptance phrase contains “and” joining independent claims.

Merge packets only when splitting would duplicate a semantic invariant or make
neither half executable. Shape semantics and movement lowering may share a
transaction; a Metal optimization and its paired benchmark harness should not.

### 13.2 Distance is computed from cells

Delete hand-authored `goalDistance`. Selection uses a vector:

```text
(mandatory cells closed,
 foreign tests newly passing,
 downstream gaps unblocked,
 evidence strength gained,
 operational risk reduced,
 estimated cost and scarce-resource demand)
```

Do not collapse this vector prematurely into one score. Lexicographic policy is
clearer: first preserve trust, then unblock the common spine, then close the
largest prerequisite cut, then prefer cheap foreign-test gains. A fast local
win does not outrank a foundational blocker merely because both become one
number.

Dependency edges also have kinds. `blocks` affects readiness; `related` adds
context; `parentChild` shapes reporting; `discoveredFrom` records provenance;
`invalidatesOnChange` follows read/interface dependencies. Only hard `blocks`
edges appear in the current template DAG, whose list order is checked as a
topological order. Treating every relation as “depends on” creates false
serialization; treating every relation as prose misses invalidation.

Readiness is not `Progress.complete` plus a caller-supplied list. A
`ReadyWitness` is derived from replayed promotion certificates, exact base
ancestry, the current upstream target, authority, calibrated validators and
fresh resource observations. `WorkPacket.structurallyExecutable` checks packet
shape; `ReadyWitness.supports` decides whether that packet may run now.

## 14. Agents as repository transformations

An agent attempt is best modelled as:

```text
attempt : BaseTree × WorkPacket × Observations
       → CandidateTree × ValidationBundle × AttemptReport
```

The function is not pure—the environment can fail—but its authority is finite.
It may write only the leased set and may consume only declared scarce resources.
Its report is not evidence; validators produce evidence.

### 14.1 Inner loop

1. Re-observe base tree, upstream pin, dependencies and live resources.
2. Claim the packet and its write/resource leases.
3. Run the exact baseline oracle; distinguish pre-existing red from regression.
4. Implement the smallest candidate that changes the expected delta.
5. Run cheap local units/properties.
6. Run the foreign differential or upstream shard.
7. Apply declared falsifiers and observe promoting checks red.
8. Restore candidate, validate exact artifacts and unsupported ledger.
9. Produce a candidate commit and structured report.
10. A trusted runner computes the actual Git effects and rejects differences
    outside the packet; candidate-reported effects are only a hint.
11. Integrator applies candidates in dependency order. Rebase, cherry-pick and
    merge produce a new candidate identity, so required checks rerun on the
    integrated tree. Evidence transport is a future optimization allowed only
    when the validator's complete input-closure digest is unchanged.
12. Promotion atomically updates capability, coverage, finding, work and
    evolution records—or updates none of them.

Generic retries are forbidden. Failure recovery depends on class:

- wrong semantics → minimize counterexample, repair semantic nucleus;
- oracle mismatch → audit adapter/target pin, never weaken expected output;
- merge conflict → rebase and re-observe dependencies;
- flaky environment → retain raw observations, classify indeterminate;
- resource contention → reschedule serially;
- packet too broad → split at an observable boundary;
- architecture breach → reject candidate even if surface tests pass.

### 14.2 Independence of implementation and judgment

The same candidate must not quietly change implementation, foreign-test
adapter, expected output, tolerance, exclusion and promotion rule together.
When a contract change is legitimate, split it:

1. contract/oracle transaction, reviewed and observed against old behaviour;
2. implementation transaction, judged by the already-promoted contract.

An agent may author both sequentially, but not collapse their authority into
one unreviewable green commit. Imported upstream tests remain immutable; the
adapter is intentionally thin and included in evidence identity.

### 14.3 Parallelism policy for this repository

| activity | parallel today? | condition |
|---|---|---|
| read-only discovery / gap analysis | yes | no generated or evidence writes |
| disjoint implementation authoring | yes, bounded | neither packet writes what the other reads or writes; disk/worktree capacity re-probed |
| shared `.lake` build | no | one writer because incremental artifacts are shared |
| gate/devcheck verification | no | 141 fixed `/tmp/tgrad_*` paths until namespaced |
| CPU/property verification | later | safe after temp namespaces and isolated build outputs |
| Metal correctness in separate processes | logically yes, operationally serial preferred | one GPU; avoid interference and memory pressure |
| performance | never parallel on this host | exclusive GPU and thermal lane |
| committed evidence integration | no | single integrator and clean measured tree |

Parallelism is a property of resources and writes, not an agent count. Three
agents editing disjoint files may be safe; two agents invoking the same gate are
not. The scheduler computes compatibility from leases and re-probes disk, build
processes, temp policy and GPU activity before launch.

## 15. The dependency program

`Tgrad.Spec.Parity.program` checks that these templates have unique identifiers,
known dependencies, ordered stages, foreign oracles, falsifiers and split
rules. The human-readable critical path is:

### Stage A — trust substrate

1. Pin upstream and generate manifests.
2. Import/adapter-run upstream null, unit and backend tests; publish the first
   honest score plus explicit exclusions.
3. Namespace temporary artifacts so CPU verification can parallelize.
4. Replace frozen performance denominators with live paired observations; keep
   verdict `indeterminate` until repeated-session variance supports a rule.
5. Finish coherent evidence regeneration and make provenance audit fatal only
   after every file is genuinely produced by the checked writers.

The first three can author in parallel after write leases are disjoint.
Evidence integration and GPU timing remain serial.

### Stage B — executable semantics

1. Total dtype/scalar semantics.
2. n-D movement/view semantics: reshape, permute, expand, pad, shrink, flip,
   masks, symbolic dimensions, aliasing and indexed UOps.
3. Pinned UOp/Ops schema with typed payloads.
4. Pure reference evaluator.

Current tinygrad expresses movement through UOps and lowers those through
indexing/rangeification; it no longer requires reproducing an older standalone
`ShapeTracker` class. Tgrad may retain an internal `View` representation if it
helps proofs and compilation, but parity is with observable movement and index
semantics, not with that historical class boundary.

This stage maximizes future leverage: it moves correctness off Metal, gives
rewrites a meaning to preserve, and lets agents validate before touching a
backend.

### Stage C — one compiler spine and CPU

1. Stabilize the minimum stage/pass/target/runtime contracts needed by one
   immediate vertical slice—not an abstract framework in isolation.
2. Make elementwise `add` traverse Tensor, rewrite, schedule, lower, linearize
   and the simplest CPU target, compared independently with tensor and kernel
   evaluators.
3. Put Metal behind the same contracts, then route generic matmul through the
   path; specialized WMMA becomes an optimizer choice.

This stage is complete only when Metal can be unavailable and the same semantic
tests still run on CPU. Adding more bespoke Metal generators before this point
increases distance from the target architecture even if an op count rises.

### Stage D — safe runtime foundation

1. Introduce session ownership, generational handles and a generic generated
   ABI.
2. Enforce buffer size/shape, view lifetime, release and thread contracts.
3. Make diagnostics, cache identity, byte budgets and teardown explicit.
4. Emit structured traces proving which graph, route, artifact, buffers and
   launch governed an execution.

This starts immediately after the common CPU/compiler interfaces stabilize.
Pure semantic slices can develop beside it, but broad Python/Metal promotion
waits for safe ownership. The reference evaluator and CPU backend remain
different implementations; their agreement is useful only if they do not
share the path under test.

### Stage E — vertical semantic slices

1. Elementwise ALU and broadcasting across dtype families.
2. Movement, masking, padding and indexing/gather/scatter.
3. Reductions and accumulation dtypes.
4. General matmul/convolution.
5. Random generation.
6. Assignment, in-place operations, aliasing and realization effects.

Each slice closes API, evaluator, compiler, CPU, Metal, numerical and upstream
test cells together. Unsupported combinations stay generated gaps.

### Stage F — training and JIT

1. Reverse-mode autograd as a graph transform.
2. TinyJit capture, replacement, replay and invalidation.

Correct lifetime precedes aggressive caching. JIT is measured against JIT on
both sides; non-JIT host overhead is a separate result.

### Stage G — ecosystem and scale

1. NN layers, optimizers, state traversal and serialization.
2. Pinned inference and multi-step training workloads.
3. Multi-device/sharding/collectives.
4. Backend profile matrix: CPU and Metal first, then each explicitly selected
   upstream runtime according to available hardware and strategic value.

“All backends” is not a prerequisite for useful Metal parity. It is a distinct
profile with explicit blocked cells where hardware is absent. Current tinygrad
also distinguishes public compute runtimes from auxiliary runtime modules and
uses a non-Cartesian CI matrix; Tgrad should capture that exact matrix rather
than inventing “every test on every backend.”

### Stage H — optimization and continuous parity

1. Typed optimization actions, memory planning and search/BEAM.
2. Per-backend workload tuning under unchanged semantic oracles.
3. Upstream manifest watcher that opens typed gaps for new/remapped/removed
   surface and tests.
4. Calibrated lag policy for advancing the pin.

Optimization comes late in the dependency program but begins locally once a
vertical slice is semantically conformant. It may run alongside breadth work as
long as it cannot weaken compatibility evidence.

## 16. Bootstrap queue

The first executable queue should be shaped from these three dependency-free
templates, not from feature requests:

### `parity.pin-upstream`

- Generate, do not hand-write: revision metadata, source tree hash, public
  modules/classes/functions/signatures/defaults, Ops constructors/payloads,
  dtypes/aliases/promotion facts, runtimes, upstream CI selectors, and complete
  test inventory.
- Store under `fixtures/upstream/<revision>/` with the capture tool and raw
  hashes.
- Produce an exclusion schema with owner, rationale and upgrade path; start
  with no silent exclusions.
- Falsify by changing one generated symbol/hash and proving drift turns red.

The research snapshot used for this plan observed official tinygrad package
version `0.13.0` at commit `19c4d736f2bc8e26d21f08b28ffd6298408da00f`
on 2026-07-27. That is a **candidate input**, not yet Tgrad's promoted target:
the repository must reproduce the capture and commit its manifests before the
pin becomes confirmed.

### `harness.namespace-temporaries`

- Replace fixed `/tmp/tgrad_*` paths with one run root created by `mktemp -d`.
- Pass the run root explicitly; trap cleanup; commit selected failed artifacts
  by copy, never by sharing a global name.
- Falsify with concurrent copies of representative non-GPU gates.
- Only after promotion may the scheduler parallelize those verification lanes.

### `harness.paired-performance`

- Run live Tgrad and pinned tinygrad in one session, interleaved and randomized.
- Match graph/JIT/compile/dispatch boundaries and report them separately.
- Persist raw paired samples, temperature/power when available, warmup policy,
  workload order and session identity.
- Estimate within-session and between-session variation before proposing a
  decision rule.
- Do not turn current L11 red into green by threshold adjustment.

After `parity.pin-upstream`, `parity.import-test-contract` and the semantic
foundation become shapeable. The first imported shard should exercise the
existing rewrite engine because it gives immediate foreign feedback on real
code; it is not allowed to become a detour from making that engine load-bearing.

## 17. Anti-patterns this program rejects

- **Latest-master parity without a pin:** there is no reproducible target.
- **Feature-count parity:** names do not establish semantics or composition.
- **Percent green without a denominator:** exclusions can manufacture progress.
- **Broad phases as work items:** nobody owns an observable transaction.
- **Horizontal teams forever:** components finish while no vertical operation
  works.
- **Porting implementation before importing its oracle:** green remains
  self-authored.
- **Test adapter as compatibility layer:** a thick shim can hide API mismatch.
- **Backend forks:** each op/backend pair becomes bespoke and growth turns
  quadratic.
- **Proof count as correctness:** theorems over authored tables can prove the
  table, not reality.
- **Foreign differential as the only invariant:** both coverage holes and local
  algebraic defects need properties/theorems too.
- **One performance ratio:** boundary, workload and variance disappear.
- **Agent report as evidence:** only observed artifacts promote claims.
- **Gate weakening under pressure:** split the claim or leave it red.
- **Specifying unstable implementation detail in Lean:** causes schema churn and
  teaches agents to game types.
- **Prose-only stable structure:** prevents mechanical gap generation.
- **One giant parity rewrite:** impossible to falsify, integrate or recover.

## 18. Completion and compounding

The codebase is not merely “grown” when a pinned profile turns green. It is
mechanically growable when:

1. a new upstream revision can be captured without editing the ontology;
2. its manifest diff generates typed unknowns/gaps;
3. ready gaps can be shaped into disjoint work packets mechanically;
4. agents can author candidates in parallel under leases;
5. validators independently produce evidence tied to exact trees;
6. one integrator can promote or reject atomically;
7. every promotion makes later work cheaper—more tests run without Metal, more
   operations share semantics, more backends reuse lowering, and more gap
   selection is computed;
8. ten times more agent compute improves coverage without redesigning the work
   system.

That is the target beyond parity: not one finished port, but a repository whose
representation of its destination, distance, work and evidence lets agents
keep converging as both codebases evolve.

## 19. Primary upstream references used to shape the contract

The capture tool, not this prose, becomes normative. These official sources
explain why the generated contract needs its current dimensions:

- [tinygrad repository and test organization](https://github.com/tinygrad/tinygrad)
- [developer layout: schedule → codegen/optimization → renderer → runtime](https://docs.tinygrad.org/developer/layout/)
- [current UOp vocabulary](https://docs.tinygrad.org/developer/uop/)
- [Tensor API](https://docs.tinygrad.org/tensor/)
- [dtype API and lattice](https://docs.tinygrad.org/dtypes/)
- [runtime/backend matrix](https://docs.tinygrad.org/runtime/)
- [performance dimensions](https://docs.tinygrad.org/developer/speed/)
- [pinned research snapshot](https://github.com/tinygrad/tinygrad/commit/19c4d736f2bc8e26d21f08b28ffd6298408da00f)

## 20. Oracle scope: public behavior, not tinygrad's representation

The pinned `test/null`, `test/unit`, and `test/backend` inventory is classified
per file in `fixtures/parity/oracle_classification.json`, regenerated by
`scripts/parity/classify_oracle.py`. `api_surface` files assert
representation-independent Tensor, dtype, or device behavior and form the
parity target. `infrastructure` covers tinygrad's own supporting tools and
formats, while `ambiguous` records files with too little evidence to classify
without guessing.

`internal_repr` is deliberately outside the compatibility denominator. Those
files constrain tinygrad-specific UOp identity and graph shape, codegen,
renderers, scheduling, optimization, or runtime layout. Tgrad intentionally
uses typed per-constructor payloads instead of tinygrad's `arg: Any`; requiring
tinygrad's internal representation would measure implementation cloning, not
observable compatibility. A mixed file with substantive internal assertions is
therefore classified `internal_repr` at this file-level metric, even if it also
contains numerical checks. Internal and infrastructure files are exclusions,
not parity failures, and classification is fixed from the pinned upstream
source before any Tgrad score is consulted.

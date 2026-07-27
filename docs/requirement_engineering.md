# Requirements engineering for the tinygrad-to-Lean rewrite

**Status:** design basis, one retrospective pilot rehearsal, and one canonical
diagnostic suite baseline, 2026-07-27

**Starting product baseline:** `c465f89` on `main`

**Current canonical suite subject:** `14ffa30` (clean source tree; runtime built
by the observer from that revision)

**Purpose:** define how Tgrad separates requirements, specification, implementation, evidence, and work before scaling the parity program.

The specification substrate is pointed in the right direction, but it starts one layer too late. It is primarily an ontology of the Tgrad machine and its development process. A tinygrad rewrite needs to begin with the problem world: Python programs, users, tensor values, errors, effects, devices, and the observations that must remain unchanged when tinygrad is replaced.

That is the central lesson from Michael Jackson: software has value only through effects in the world, so requirements must be stated in world terms—not in terms of the machine’s internal decomposition. [“The World and the Machine”](https://courses.cs.washington.edu/courses/csep503/19wi/schedule/papers/TheWorldAndTheMachine.pdf)

## What the first executed cycle established

The pilot is now more than a proposed ontology, but this first cycle is a
retrospective diagnostic rehearsal rather than the prospective validation
needed for scale. The compatibility-surface candidate already existed before
the work packet was derived, and the closure logic was generalized after the
post-change result. The assembled artifacts represent every stage of
`observe → derive → change → observe → re-derive`, but their chronology was not
that sequence. They therefore do not establish that a frozen model can predict
and close a newly authored change.

| Stage | Revision/artifact | Mechanically established result |
|---|---|---|
| Requirements kernel | `dbacfdd`–`c34a76e` | Three requirements are expressed over world domains, monitored/controlled phenomena, assumptions, profiles, and typed relation descriptors/observation dimensions without importing product modules. The descriptors do not yet denote executable relations. |
| Boundary and conformance | `60e9ef4` | Boundary specifications and candidate product mappings are distinct; mapped symbols are compile-time pinned, but mappings do not count as conformance. |
| Baseline observation | `f09b51e` | An earlier, weaker observer localized a `tinygrad.helpers` failure at the strict substitution boundary; broadcast-add and view-lifetime observations were classified as blocked rather than falsely failed. Because the current observer executes the real product and has stronger identity/calibration rules, this artifact is historical localization evidence rather than a same-protocol before/after measurement. |
| Derived work | `0031066` | The observed state instantiated the authored `WORK-PY-COMPAT-HELPERS` schema, located the likely change at the Python substitution boundary, required only CPU plus a Lean build, and predicted two model-level unblocks. The candidate already existed, so this is not a prospective prediction. |
| Candidate change | `6860375`, `b1df552` | The strict compatibility surface was widened. The pilot then exposed an unrelated eager NumPy dependency during package activation; making that dtype dependency lazy removed the environmental confound without weakening the no-fallback rule. |
| Current observation | [pilot_helpers_b1df552.json](../fixtures/requirements/pilot_helpers_b1df552.json) | The real `python/tgrad.py` and exact runtime artifact are loaded; the isolated helper-import scenario observes module ownership and three required names; package-path and missing-name mutants calibrate those two dimensions independently. Source, adapter, runtime, verifier, environment, and scenario identities are recorded. |
| Re-derived state | [PilotState.lean](../Tgrad/Growth/PilotState.lean), [Work.lean](../Tgrad/Growth/Work.lean) | The isolated helper observation is `passed_calibrated`; add and view move from model-level `blocked` to `unobserved`; the single helper packet is absent. This required a closure-logic repair, so derivation stability has not yet been demonstrated. Adequacy remains open and the target remains unpromoted. |

This is a useful positive result for the method: the result was consistent
with the known boundary and resource class, distinguished a product gap from downstream
unknowns, caught a hidden environment coupling, and forced evidence to become
more precise under adversarial review. Regeneration now removes the packet
without editing a status cell, but the derivation program itself changed; the
next cycle must freeze it before candidate authoring.

It is not yet evidence that Tgrad is tinygrad-compatible, nor that the method
is ready for 590 requirements. The canonical suite has now been observed, but
no suite case was prospectively mapped to an interpreted requirement; only two
pilot observation dimensions have reproduced fault injections, two pilot
behaviors remain unobserved by requirement-specific validators, all three
adequacy arguments remain open, and the upstream target remains an extracted
candidate. In the guide’s maturity vocabulary this is **instrumented**, not
yet **operationalized** or **compounding**.

## Canonical suite baseline: facts, not a parity verdict

The frozen 34-file `api_surface` contract has now been run on the pinned
upstream revision and on Tgrad through the strict no-fallback shim. The paired
bundle is [pair_877ed54ec823_c2c01285c788](../fixtures/parity/observations/pair_877ed54ec823_c2c01285c788/).
Promotion replays every raw pytest event stream and reapplies the upstream
oracle before accepting the bundle. [SuiteGenerated.lean](../Tgrad/Evidence/SuiteGenerated.lean)
imports the resulting execution facts into the checked specification root.

| Fact | Upstream | Tgrad |
|---|---:|---:|
| Contract files | 34 | 34 |
| File outcomes | 33 passed, 1 environment-unobserved | 16 blocked by product surface, 15 nonconforming, 1 collection error, 1 environment-unobserved, 1 upstream-unobserved |
| Raw passing executions | 1,145 | 17 |
| Raw failures / errors | 0 / 0 | 836 / 159 |
| Upstream-eligible cases | 1,145 | same oracle |
| Eligible cases matched by descriptor | — | 1,004 |
| Eligible cases passed | — | 17 |
| Eligible cases nonpassing / missing | — | 987 / 141 |
| Descriptor mismatches | — | 98 |

These numbers are diagnostic coordinates, not a compatibility percentage.
The unit of execution is a pytest case; the unit of requirement engineering is
a world-facing behavioral obligation. No exact case-to-requirement witness map
was frozen before this Tgrad result, so importing this bundle changes **zero**
requirement states. The first admissible bridge is:

```text
replay-validated suite facts
    → reviewed exact case/descriptor-to-requirement witnesses
    → requirement-specific observations and calibration
    → promotion decision
```

The replay machinery also falsified itself usefully. An initial promotion
attempt rejected diagnostics whose hashes depended on a temporary observer
root. Schema 8 makes normalization root-independent and a regression test now
requires two different snapshot roots to produce identical normalized
diagnostics. A second attempt found an undefined artifact-copy source before
repository mutation; atomic staging removed the partial bundle and the copy
closure is now tested byte-for-byte. These are positive method signals because
the evidence boundary rejected its own defects rather than accepting a green
story.

One provenance obligation remains open: generated-file checking currently
replays through the current verifier source and the local pinned checkout. Once
that verifier evolves, historical evidence needs its recorded verifier source
or an independently versioned replay executable; otherwise a valid old bundle
cannot be regenerated hermetically. Until that artifact is added, suite facts
may be checked on this baseline but must not be treated as a durable promoted
requirement claim.

## The missing specification kernel

The foundational relationships should be:

```text
D ∧ S ⊨ R        specification adequacy
M ⊨ S            implementation satisfaction
```

Where:

- `D` is indicative domain knowledge: facts and assumptions about Python,
  NumPy, Metal, bf16, hardware, and admitted user programs.
- `R` is optative: the effect required in the problem world.
- `S` is optative but restricted to phenomena shared by the complete Tgrad
  replacement and its environment.
- `M` is the complete replacement machine: compatibility shim, adapters, Lean
  implementation, runtime, and backend. Platform and adapter assumptions may
  be needed when establishing `M ⊨ S`.

Observed tinygrad behavior is evidence used to interpret the pinned public
contract. It is neither automatically an indicative domain fact nor, by
itself, the normative requirement.

Zave and Jackson distinguish indicative domain descriptions—what is true—from optative requirements—what we want to become true. They then require the specification and domain knowledge together to entail the requirement. [“Four Dark Corners of Requirements Engineering”](https://doi.org/10.1145/237432.237434)

The pre-pilot codebase had excellent fragments of this model but not the
relationship itself. The new `Tgrad/Requirements`, `Tgrad/Specification`,
`Tgrad/Conformance`, `Tgrad/Evidence`, and `Tgrad/Growth` pilot modules now
record the identities, dimensions, evidence obligations, and epistemic states
needed to formulate a narrow instance. They do not yet give `D`, `S`, or `R`
executable denotations or establish either entailment. The rest of the parity
model has not yet been migrated.

- [Ontology.lean](../Tgrad/Ontology.lean#L30) begins with `dtype`, `shape`, `uop`, `kernelDecl`, `metalSource`, and `buffer`. Those are machine/design concepts, not problem-world concepts.
- [Parity.lean](../Tgrad/Spec/Parity.lean#L198) calls an upstream symbol/test inventory a `Requirement`, but the record contains no behavioral predicate.
- [EvidenceRef](../Tgrad/Spec/Parity.lean#L107) is quite strong: it records subject, verifier, adapter, environment, relation, artifacts, and validator.
- What is missing is the typed connection from requirement to specification, and from specification to implementation evidence.

The older substrate therefore knows a great deal about evidence provenance and
work organization, while the pilot can state the obligation and track what is
missing but cannot yet discharge the central claim:

> Given these facts about Python and tinygrad, this boundary behavior from Tgrad is sufficient to produce this required user-visible result.

## The three shapes must remain separate

The project needs three linked graphs, not one giant roadmap.

| Shape | Organizing unit | Governing question |
|---|---|---|
| Functionality | Problem frames and observable trace relations | What must users and programs observe? |
| Codebase | Refinement stages and implementation components | How does Tgrad produce those observations? |
| Work | Unsatisfied obligations and transformations | What change would close a demonstrated gap? |

The essential linkage is:

```text
world requirement
    ↓ adequacy
machine-boundary specification
    ↓ conformance
implementation/refinement stages
    ↓ observation
revision-bound evidence
    ↓ derivation
current state and gaps
    ↓ planning
agent work packet
```

No edge should be skipped.

In particular:

- An upstream test is not a requirement. It is a witness or observer of one.
- A Tgrad file is not implementation evidence. It is a candidate location.
- A passing build is not semantic conformance.
- A commit is not completion.
- A work template is not an actual gap unless derived from the current requirement/evidence state.
- A codebase architecture is not a requirement. It is one proposed solution.

## The shape of functionality

“Tinygrad compatibility” is too broad to be one undifferentiated feature matrix. It should be decomposed into problem frames—subproblems with explicit world domains, shared phenomena, requirements, and assumptions.

The Gunter–Gunter–Jackson–Zave reference model emphasizes shared phenomena at the machine/environment interface. [“A Reference Model for Requirements and Specifications”](https://citeseerx.ist.psu.edu/document?doi=0b6396d9bbc1f1782f0634d622078da88ed808bc&repid=rep1&type=pdf)

For Tgrad, the initial frames should be:

| Frame | World domains | Shared phenomena |
|---|---|---|
| Python substitution | Python importer, client program, pinned public contract | Modules, names, attributes, signatures, exceptions |
| Tensor transformation | Tensor values, shapes, dtypes, mathematical operations | Calls, input values, output values, dtype/shape results |
| Views and workpieces | Base storage, views, aliases, mutations | Index maps, storage identity, lifetime, readback |
| Commanded execution | User program, lazy graph, runtime | Realization requests, completion, errors, diagnostics |
| Stateful behavior | Random state, assignment, autograd, JIT/cache | Seeds, mutations, graph state, cache effects |
| Backend connection | Metal/CPU/device environment | Allocation, dispatch, synchronization, device errors |
| Quality/performance | Workload and hardware profile | Latency distributions, memory use, throughput |

This immediately explains the historical “six operations but still 0/34”
observation in [PLAN_OP_BREADTH.md](../PLAN_OP_BREADTH.md#L183).

At the `fdc741d` measurement, twenty-nine files failed during collection because
`tinygrad.helpers` and related module phenomena were absent. That established
failure in the Python-substitution frame. It did **not** establish failure of
the newly implemented tensor operations, because those tests never reached
them. Their correct status was `unobserved behind prerequisite`, not `failed`,
`absent`, or `conformant`.

The current isolated observation establishes that the isolated three-name
helper-import prerequisite passes. The canonical rerun does not turn that fact
into broad substitution parity: it exposes a new, exact mixture of product
surface blocks, collection error, assertion failures, and environment-unobserved
cases. Those are new execution facts rather than retroactive claims about the
helper requirement or the six implemented compute operations.

That distinction is impossible with one flat `CoverageState`.

## Requirements must constrain behavior

The highest-level compatibility requirement should resemble:

```text
For every program P and environment e admitted by profile p:

  observations(run(P, Tgrad, e))
    ≈p
  observations(run(P, tinygrad@revision, e))
```

But this must be refined. Each requirement needs:

- The declared compatibility profile.
- Problem-world domains involved.
- Monitored and controlled phenomena.
- Preconditions and applicability.
- Required return-value relation.
- Shape and dtype relation.
- Exception relation.
- Side-effect, mutation, aliasing, and lifetime relation.
- Evaluation-order or laziness relation where observable.
- Acceptable numerical equivalence.
- Environmental assumptions.
- Normative source and provenance.
- Known ambiguity or undefined behavior.
- Applicable observers and required calibrations.

The current 590 rows should be renamed conceptually to something like `UpstreamContractCandidate` or `SurfaceInventoryEntry`. They answer:

> What might require interpretation?

They do not yet answer:

> What behavior is required?

A symbol such as `Tensor.sum` may generate several actual requirements:

- Import and attribute availability.
- Signature and accepted argument forms.
- Axis normalization.
- Shape of the result.
- Dtype selection and accumulation.
- Empty-axis behavior.
- Error behavior.
- View and alias interaction.
- Lazy graph construction.
- Numerical relation after realization.

Conversely, several upstream tests may witness the same requirement. Requirement count must not simply equal test count or symbol count.

## Use the four-variable model without moving the requirement boundary

Parnas and Madey’s four-variable model is especially useful here. It separates world variables from machine-interface variables. [“Functional Documents for Computer Systems”](https://doi.org/10.1016/0167-6423(95)96871-J)

The primary requirement boundary is:

```text
client/problem world ↔ complete Tgrad replacement
```

The complete replacement includes the Python shim, C/Lean adapters, Lean core,
runtime, and backend. At that boundary:

- `m`: monitored world phenomena—Python calls, tensors, seeds, device state.
- `c`: controlled world phenomena—returns, exceptions, tensor values, mutations, device effects.

The Python/C/Lean boundary is a nested refinement boundary inside that machine.
When the Lean core is treated as the nested machine:

- `i`: encoded Lean-core inputs—handles, op codes, shapes, bytes, graph nodes.
- `o`: Lean-core outputs—handles, buffers, result codes, diagnostics, traces.

Then define four relations:

```text
REQ(m, c)       required behavior of the complete replacement
IN(m, i)        shim/adapter encoding into the Lean core
SOFT(i, o)      behavior of the Lean core and runtime
OUT(o, c)       adapter decoding back into world-visible behavior
```

Correctness requires:

```text
IN ; SOFT ; OUT ⊆ REQ
```

under the declared domain assumptions.

This makes two things first-class that are currently easily mistaken for scaffolding:

1. The substitution shim is part of the compatibility machine, not merely a test convenience.
2. The Python/C/Lean FFI is a refinement boundary whose encoding and decoding can independently violate requirements.

A mathematically correct Lean reduction does not establish `Tensor.sum` compatibility if `IN` cannot import the right module or encode an axis. Likewise, a passing Python call does not establish Lean correctness if `OUT` silently reshapes or decodes a view incorrectly.

## The shape of the codebase

The ideal codebase should be treated as a refinement architecture, not as part of the requirements model:

```text
Python compatibility boundary
    ↓
abstract tensor/effect semantics
    ↓
typed graph semantics
    ↓
rewrite and schedule semantics
    ↓
kernel/program semantics
    ↓
backend runtime semantics
    ↓
device observations
```

Every arrow needs four things:

- A source semantics.
- A target semantics.
- A refinement relation.
- Evidence or a theorem that the transformation preserves the relevant observations.

This is stronger than the current `Morphism` record, which records sources, targets, and totality but no meaning-preservation relation.

For example:

```lean
structure RefinementClaim where
  sourceStage : StageId
  targetStage : StageId
  relation : RelationId
  assumptions : List AssumptionId
  obligations : List ObligationId
```

Optimizations then become refinements rather than “capabilities.” A rewrite is valid because it preserves tensor semantics under stated assumptions, not because the rewrite function exists and has tests.

The present `idealDependencies` in [Parity.lean](../Tgrad/Spec/Parity.lean#L387) is useful design documentation, but its theorem only establishes consistency of the authored architecture. It should be labelled a design-hypothesis check, not parity evidence.

## Ground truth is plural and revisioned

There should not be one mutable “current truth” table. Use distinct planes:

1. **Extracted upstream facts**
   Symbols, signatures, tests, backends, source locations at an exact revision.

2. **Interpreted contract decisions**
   Public/internal classification, applicable profile, normative source, ambiguity decisions.

3. **Product facts**
   Symbols, routes, operations, dtypes, backends and transitions mechanically extracted from an exact Tgrad tree.

4. **Observations**
   What a specific validator saw on a specific subject tree and environment.

5. **Promoted claims**
   Claims accepted because current, calibrated evidence satisfies their obligations.

6. **Working candidates**
   Branch or dirty-tree changes that have not been promoted and must never be reported as completed capability.

Across the starting baseline and current pilot subject, the planes read:

| Plane | Current observation | Correct status |
|---|---|---|
| Upstream extraction | Manifest exists for `19c4d736…` | Candidate target observed |
| Lean target | `targetUpstream` remains `.unknown` | Not promoted |
| Canonical product | `main` at `c465f89` | Starting product baseline |
| Pilot candidate | strict substitution change at `b1df552` | Observed product subject; repository merge and semantic promotion are separate facts |
| General operation spine | graph-indexed realization, broadcast pointwise operations, reductions, and fused reduce-of-elementwise matmul are merged | Implemented; each claim still depends on its own evidence |
| Specialized matmul route | retained alongside the general expression route | Optimization candidate, not the semantic definition of matmul |
| L12 performance predicate | its flags now execute correctly and the frozen-baseline predicate is red and non-repeatable | Open measurement-methodology failure, not a codegen verdict |
| API suite execution | paired schema-8 run at upstream `19c4d736…` and Tgrad `14ffa30` | Current diagnostic facts: 17/1,145 upstream-eligible cases passed, 987 nonpassing, 141 missing, 98 descriptor mismatches; no requirement parity inference |
| Pilot helper behavior | calibrated isolated-scenario pass at `b1df552` | The current derived model emits neither an implementation gap nor a failed-behavior gap for this isolated scenario; executable conformance semantics, adequacy, and target promotion remain open |

The fact that the manifest exists while [targetUpstream](../Tgrad/Spec/Parity.lean#L36) remains unknown is not necessarily wrong—capture and promotion should be separate—but the model needs an explicit `candidate → reviewed → promoted` transition.

Revision scoping also prevents old truths from lingering. The earlier statement that `rangeify` was the identity is no longer true: it is now a real partial transformation in [Rangeify.lean](../Tgrad/Schedule/Rangeify.lean#L84). Meanwhile [Findings.lean](../Tgrad/Spec/Findings.lean#L116) still says to revisit a reduction issue “before reduction becomes a supported runtime operation,” although reduction support has landed. This is exactly the stale-schema failure the Clockworks philosophy warns about.

## The shape of work

There are two distinct meanings of work:

### Work by the codebase

This is Tgrad’s operational transition system:

```text
ingest → represent → transform → schedule → lower → compile
       → allocate → dispatch → synchronize → decode
```

Each transition needs preconditions, postconditions, failure modes, resources, and observable effects. [RuntimeWork.lean](../Tgrad/Spec/RuntimeWork.lean#L129) begins this model, but its `implemented` helper promotes file paths directly to `.confirmed`. A path list proves neither implementation nor behavior, and the table has already fallen behind the product.

### Work on the codebase

This is an agent-mediated state transition:

```text
baseline + obligation
    → candidate transformation
    → calibrated observations
    → promotion decision
    → new baseline
```

An agent should not author the truth that its own work is complete. It should produce a candidate and evidence. Promotion derives completion.

The current authored `goalDistance : Nat` in [Work.lean](../Tgrad/Spec/Work.lean#L93) should disappear. Work should be generated from typed gaps:

- Unknown domain fact → discovery work.
- Uninterpreted upstream surface → requirements work.
- Missing boundary specification → specification work.
- Failed adequacy argument → modeling work.
- Failed implementation conformance → product work.
- Missing observer → verification work.
- Uncalibrated observer → falsification work.
- Environmental obstacle → infrastructure or profile decision.
- Stale evidence → regeneration work.

Van Lamsweerde’s goal-oriented work is useful here: refine goals, identify obstacles, and assign responsibility explicitly rather than treating every failure as an implementation task. [“Handling Obstacles in Goal-Oriented Requirements Engineering”](https://www.cs.ucf.edu/~turgut/heng_than.pdf)

A generated work packet should contain:

```text
obligation being closed
baseline subject revision
problem frame and affected phenomena
assumptions relied upon
candidate components, not mandatory files
independent oracle
validator and calibration requirements
authoring write/resource set
serial verification requirements
closure predicate
recovery action
```

The current [Growth/Work.lean](../Tgrad/Growth/Work.lean) is not yet that
generic compiler. It conditionally emits one helper-specific packet whose
components, resources, transitions, priority fields, and recovery text are
authored. Only its applicability and blocked-requirement count are derived.
The helper→add/view dependency is likewise an interpreted model decision, not a
discovered fact, until downstream observers confirm it. The pilot should call
this a conditional instantiation of an authored work schema.

The mechanism for growing the system then becomes:

```text
deriveGaps(target, promotedClaims)
  → selectWork(gaps, dependencies, risk, resources)
  → applyCandidate(baseline, work)
  → observe(candidate, validators)
  → promote or reject
  → recompute
```

That is much more mechanizable than maintaining a large authored roadmap and proving that its IDs are unique.

## Lean decomposition

The pilot introduced the first vertical slice beside the existing specs rather
than rewriting them: `Requirements/{World,Relation,Requirements,Pilot}`,
`Specification/{Boundary,Pilot}`, `Conformance/Claims`,
`Evidence/{Observations,PilotGenerated}`, and
`Growth/{Derived,PilotState,Work,PilotReport}`. These modules deliberately
follow the dependency direction shown earlier.

If the pilot passes the scale gate, extend that slice toward:

```text
Tgrad/Requirements/
  World.lean
  Phenomena.lean
  Assumptions.lean
  Goals.lean
  Requirements.lean
  ProblemFrames.lean
  Obstacles.lean

Tgrad/Specification/
  Boundary.lean
  TensorSemantics.lean
  Effects.lean
  Profiles.lean
  Adequacy.lean

Tgrad/Refinement/
  Graph.lean
  Schedule.lean
  Kernel.lean
  Backend.lean

Tgrad/Evidence/
  Observation.lean
  Validator.lean
  Calibration.lean
  Promotion.lean

Tgrad/Growth/
  Gap.lean
  WorkPacket.lean
  Selection.lean
  Evolution.lean
```

The first pilot should cover only three end-to-end requirements:

1. `import tinygrad.helpers` succeeds or fails according to the declared substitution profile.
2. Broadcasted `Tensor.add` preserves declared shape, dtype, value, error, and realization behavior.
3. A transposed/sliced view readback preserves values, shape, storage/lifetime rules, and exceptions.

Those three deliberately span lexical substitution, pure transformation, and workpiece/lifetime behavior. If the model handles them cleanly, it can scale. If it cannot, adding 590 rows will only produce schema theatre.

The next prospective cycle has now corrected one weakness in that original
decomposition. The broad retrospective add row is retained for historical
evidence, but prospective add work is split into legal same-dtype behavior,
dtype promotion, incompatible-shape exceptions, and realization idempotence.
The frozen manifest and work packet are linked from
[plan_2026-07-27.md](plan_2026-07-27.md#frozen-shape-of-the-next-prospective-packet).
No product candidate has been selected: the first predicted transformation is
observer/specification work, and its result may honestly be failed, calibrated
pass, blocked, or verifier error.

The legacy bf16 add requirement is explicitly retained as historical rather
than counted alongside the new requirements as if it had already been
superseded. The trial-definition stage predicts no observation change. The
observer stage has a distinct write/resource set and an uncertainty envelope;
its eventual outcome cannot derive conformance while adequacy remains open.

This is now a genuinely prospective definition rather than a reconstruction:
revision `0d5f1c1f8cab375f824cb5f1ae7c1865a1a5c078` freezes the requirement,
boundary, manifest, mutation policy, derivation rule, and packet before either
the observer or a product candidate exists. A separate machine-checked
[trial lock](../fixtures/requirements/broadcast_add_trial_lock_v1.json) binds
that Git tree, each critical file hash, the upstream pin, and the declared
execution environment. Its pending-artifact fields are part of the result:
they prevent an immutable definition from being rhetorically upgraded into an
executed trial. The next admissible transition is observer implementation and
mutation calibration; product authoring remains forbidden until a baseline
observation has been recorded against this lock.

The first frozen definition was in fact rejected before observation. Two
mutants were specified to prevent result construction while also preserving
the observed dtype of that result, an impossible validator obligation. No
observer workaround was accepted and no product probe was used to fit the
repair. The V2 amendment preserves V1 and records the sole correction: dtype
may be unobserved for those two result-preventing mutants. This is evidence
that definition review can stop invalid work before product authoring; it is
not evidence about Tgrad behavior or about requirement adequacy.
The repaired definition is independently frozen at
`9762bb722c9f76283cf62cb16e8df2f902dc92ba` and bound by the
[V2 trial lock](../fixtures/requirements/broadcast_add_trial_lock_v2.json).

The observer implementation follows the same rule. Its first draft mutated
already-produced dimension dictionaries and was rejected: a comparator that
fabricates its own faults has not demonstrated sensitivity to faulty behavior.
The accepted architecture launches eight isolated child probes whose faults
act before observation and whose literal streams, branch source,
configuration, executable-tree identity, and replay closure are all bound.
The first pinned-upstream execution then failed before a baseline could exist:
the frozen revision's Metal initialization receives `None` from `UTF8String`
and cannot realize/read back the legal inputs. The replayable diagnostic
artifact is stored under
`fixtures/requirements/observations/05a83ab54d49812233cc60f8a5d37e676009d2914c5a58d668e67c6ac83d5cee/`.
It is a diagnostic blocker with `baseline_eligible=false`; it is not silently
accepted as calibration, an observation pass, or evidence about Tgrad. This
correctly stops the baseline-first protocol before a TGrad run.

A controlled host probe subsequently falsified the original environmental
diagnosis. In the managed execution sandbox, `MTLCreateSystemDefaultDevice()`
returns nil; outside that sandbox, the same pinned tinygrad tree, CPython
launcher, macOS host, and M4 return `Apple M4`. The downstream `UTF8String`
exception was therefore a secondary symptom of a hidden execution-context
boundary, not evidence that the upstream revision was defective. The target
and semantic definition remain unchanged.

The first unsandboxed V2 observation then produced a second replayable
diagnostic, `cd5cab1b0b5e3630f18da2829e87e67b6634e4bd6cbe237733fc70e2dd180f34`.
Six of eight mutants calibrated. The other two were detected in every required
semantic dimension with the exact declared dimension footprint, but V2 also
required trace records for completed stages such as `invoke_add` to differ.
They correctly did not differ. V3 therefore amends only those two expected
trace footprints and moves them into a frozen verifier contract. It changes no
requirement, scenario, semantic dimension, mutant implementation, target, or
product file. The V2 diagnostic remains ineligible as a baseline.

The V3 implementation preflight then rejected V3 itself before another Metal
run: its generator checked the old observer hash against the mutable current
worktree, while the V3 packet explicitly required that observer to change.
V4 records the sole tooling correction prospectively: verify the old observer
against the Git object at the V3 definition revision. V3's trace correction,
all behavioral semantics, and the empty product write set remain inherited.

The committed V4 observer then produced the first admissible upstream baseline,
`84a58222575eab06ecc72889e1dbbe2a2084849356673a8d299c21ad2e41a844`.
All six scenarios reached their declared terminal behavior and all eight
isolated mutants were rejected. The artifact binds the pinned upstream tree,
V4 effective contract, V3 semantic lock, observer/probe Git identity,
environment, raw streams, normalized streams, protocols, and mutant branches.
It remains observation-only: adequacy is open and no conformance or parity
claim follows from an upstream baseline.

The first TGrad comparison attempt was then refused before TGrad loaded. The
baseline recorded the verifier repository revision/tree, and committing the
baseline necessarily changed those run-specific provenance fields while the
observer, probe, schema, and verifier files remained byte-identical. V5 freezes
the relation repair: retain Git revision/tree/hash in each artifact, but compare
executable verifier equivalence by observer hash, probe hash, schema, clean
state, and verifier-file hashes. The upstream baseline remains immutable.

Review refuted V5 before implementation: its relation code still lived in the
observer file whose whole-file hash it proposed comparing. Implementing V5
would therefore invalidate its own equality premise. V6 freezes an architectural
split instead. Subject-protocol equivalence is probe/schema/manifest/semantic-
lock identity; observer and Git identities remain per-run provenance; a new
pure relation module is independently hashed and mutation-calibrated.

The first admissible TGrad comparison is now recorded as
`a4fcdc9090a20ce6c01283d3795bb55e099a3fcd45e29f608f2e128182022bba`.
All six scenarios reached the strict TGrad substitution and then stopped at
`construct_left`: `tinygrad.Tensor` is correctly identical to `tgrad.Tensor`,
but Tgrad's `Tensor.__init__` still exposes its internal `(buf, size, shape,
dtype, ...)` protocol instead of tinygrad's public `Tensor(data, dtype=...)`
protocol. The five legal scenarios each have six downstream semantic
dimensions unobserved. The incompatible-shape scenario happened to share the
upstream terminal outcome `raised`, but differs in stage, exception class, and
message; this is not incompatibility-error conformance.

This observation changes the next work item without changing the requirement:
the immediate candidate is the public construction boundary. It should expose
downstream add observations for the already-supported float32 cases while
leaving unsupported int32 behavior explicit. It must not be described as a
broadcast-add implementation or as parity progress until a fresh run observes
those dimensions.

The resulting prospective packet is
[`WORK-PY-TENSOR-PUBLIC-CONSTRUCTOR-V1`](../fixtures/requirements/broadcast_add_constructor_candidate_v1.json).
It freezes a one-file product write set and a deliberately weak prediction:
four float32 scenarios should cross both construction stages, while the two
int32 scenarios remain explicit capability gaps. Its refusal to predict an
arithmetic result is part of the specification. If the next observation stops
at readback, reshape, addition, or realization, that is new information rather
than a failed promise about full broadcast parity.

Implementation preflight refuted V1's work shape before commit:
`scripts/parity/fused_matmul_differential.py` was a second direct consumer of
the internal raw-buffer constructor. V2 adds that migration file and requires
a repository-wide absence check for the old call form. The semantic contract
and expected observation transitions are unchanged. This is a concrete work-
prediction miss, and therefore evidence that dependency discovery must include
verification code—not only the product import graph.

The V2 implementation at `4a489e3` was then observed under the unchanged V6
protocol. Evidence
`8f6a7f9dc4bf105d23773e6fafc7dc145b8b3e7d8c5b6bf0a0cc8edfd9531de1`
matches the prospective partition exactly: all four float32 pairs construct,
and both int32 scenarios remain explicit unsupported-dtype failures. The
three legal float32 scenarios now stop at `snapshot_inputs` because the public
Tensor lacks `tolist`. The incompatible pair reaches `invoke_add` and raises
Tgrad's shape error. This is observability gain—construction changed from a
common blocker to a passed dimension—but still supplies no legal add result.

The next prospective packet,
[`WORK-PY-F32-VIEW-READBACK-V1`](../fixtures/requirements/broadcast_add_readback_candidate_v1.json),
owns both sides of the newly exposed boundary: the public `Tensor.tolist`
method and the Lean view materializer's element width. Its fault model includes
the tempting partial fix—adding `tolist` while leaving float32 reshape views
unmaterializable—and a 16-bit copy of 32-bit storage. The only predicted
observer transition is that three legal float32 scenarios reach `invoke_add`.

Evidence
`317fc368cfcda64ed0ee45eab33f49a985605ad75dc921021f4a7d86040572d1`
confirms that prediction. Same-shape and singleton-axis float32 addition now
reach result shape and dtype observation—both are `(2,3)`/float32—and stop at
`realize_1` because `Tensor.realize` is absent. Two-sided rank-3 broadcasting
reaches `invoke_add` and stops at the explicit rank-2 guard. These are now two
separable work fronts: realization identity for already materialized results,
and generalized right-aligned broadcasting. Keeping them separate preserves
fault localization and avoids crediting a convenience method with kernel work.

## How to know whether the method is working

Compilation is necessary but almost uninformative here. The method is working
only if it improves the truthfulness, diagnostic power, and economics of real
changes. Track these measures per closed work cycle:

| Measure | Definition | Pilot result | Scale gate |
|---|---|---|---|
| Grounded-promotion rate | Promoted claims carrying exact requirement, profile, assumptions, relation semantics, boundary specification, target, subject, runtime artifact, verifier, adapter, environment, scenario, and calibrated validator identities / all promoted claims | Not yet defined: 0 claims are promoted. The zero denominator must not be reported as 100%. | 100% once promotion begins |
| Fault detection | Injected faults rejected / injected faults attempted | 2/2 for the helper scenario: package-path contamination and missing public name; 2/6 on the pilot-wide gate | At least six faults across the three pilot frames before migration |
| Fault localization | Rejected faults assigned to the correct requirement/boundary without changing unrelated states | Package-path contamination localized to module resolution; removing `DEV` localized to public surface. Neither result establishes add or view semantics. | 100% for the declared pilot mutation matrix |
| Work prediction precision | Frozen, prospectively emitted packets whose component class, oracle, resources, and state transitions match a subsequently authored change | Not measured: the candidate predated the packet | No unexplained transition; revise the dependency model when a prediction misses |
| Observability gain | Requirements moving from blocked to a more informative executable state | Two model states moved from `blocked` to `unobserved`; 0/2 downstream observers have yet been executed | Every prerequisite packet must expose at least one downstream observer or justify why not |
| Manual-verdict count | Coverage, completion, priority, or promotion cells edited by an agent instead of derived | Zero status cells, but derivation code changed after the result; stability gate not met | Zero status edits and zero derivation changes within a frozen cycle |
| Change locality | Unrelated requirement axes changed / unrelated axes present | No unrelated requirement was promoted or failed | Zero unexplained churn |
| Reproducibility | Identical-input runs producing byte-identical evidence and status | Deterministic observer checks and repeated status hashes agree on the current host | Three clean reruns; distributions plus method identity for performance |
| Evidence-boundary self-falsification | Verifier defects detected before promotion / defects exercised | Six distinct definition/tooling defects were stopped before a product claim: hidden sandbox GPU access, two over-strong trace footprints, mutable-old-hash chronology, child active-manifest projection, self-invalidating full verifier identity, and observer/relation self-reference | Every discovered verifier defect becomes a replay or mutation regression |
| Cycle cost | Human decisions, agent time, wall time, scarce hardware time, and files touched to close one obligation | High: six protocol/tooling revisions were needed before one upstream and one TGrad observation became comparable. The payoff is exact localization at public construction rather than a false arithmetic verdict; the cost is not yet proven economical | Must become cheaper on the next three requirement cycles than raw-suite/manual-roadmap diagnosis |

Two metrics need special care.

First, a falling pass count can still represent information gain if failures
move from harness/collection errors to correctly localized semantic assertions.
The preferred progress order is:

```text
unknown → observable → correctly failing → conforming → promoted
```

Second, requirement count is not a progress denominator. Requirements split and
merge as interpretation improves. Report transitions and closed obligations,
not a flattering percentage over a mutable hand-written list.

Before scale, evidence must also bind hashes of the interpreted requirement,
profile, assumptions, relation semantics, and boundary specification—not only
their IDs. Every mandatory dimension needs an actual result and a relevant
fault calibration. The current pilot binds the exact `SpecId` during Lean
qualification and records per-dimension results, but semantic-definition hashes
remain an open design obligation.

The current observation also hashes the exact loaded dylib but does not yet
prove that those binary bytes were produced from the declared product source
revision. This is transparent artifact identity, not source-to-binary
provenance. Such provenance is therefore another explicit promotion blocker.

The method should be rejected or redesigned if repeated cycles exhibit any of
these symptoms:

- Work packets routinely point at the wrong boundary or resource class.
- A single change causes broad unexplained status churn.
- Mutants survive while ordinary examples pass.
- Environmental or verifier faults are reported as product failures.
- Agents must hand-edit completion or priority to make the account look right.
- Maintaining the model costs more than the diagnostic ambiguity it removes.
- More agent compute produces more prose and rows but not more calibrated
  observations, closed unknowns, or reusable validators.

## The strongest definition of “done”

A requirement is done for a Tgrad tree only when:

```text
target revision/profile is promoted
∧ requirement behavior is specified
∧ relevant domain assumptions are explicit
∧ boundary specification is adequate for the requirement
∧ implementation conforms to the boundary specification
∧ current evidence observes all mandatory dimensions
∧ validators are mutation-calibrated
∧ evidence names the exact requirement/specification semantics, subject,
  runtime artifact, verifier, adapter and environment
∧ no prerequisite or obstacle leaves the result merely unobserved
```

That is stricter than “test passed,” but it is also far more informative. It tells us whether the gap is in requirements, the adapter, semantics, implementation, evidence, or environment.

The deepest redesign is therefore not “add more facts to Lean.” It is:

> Make Lean hold the argument connecting world requirements to machine specifications, implementations, observations, and derived work.

That would turn Tgrad from a codebase with a formalized roadmap into a codebase that can mechanically explain what it is for, what it currently guarantees, what remains unknown, and which transformation most directly advances it.

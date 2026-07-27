# Requirements engineering for the tinygrad-to-Lean rewrite

**Status:** design basis, 2026-07-27

**Product baseline:** `c465f89` on `main`

**Purpose:** define how Tgrad separates requirements, specification, implementation, evidence, and work before scaling the parity program.

The specification substrate is pointed in the right direction, but it starts one layer too late. It is primarily an ontology of the Tgrad machine and its development process. A tinygrad rewrite needs to begin with the problem world: Python programs, users, tensor values, errors, effects, devices, and the observations that must remain unchanged when tinygrad is replaced.

That is the central lesson from Michael Jackson: software has value only through effects in the world, so requirements must be stated in world terms—not in terms of the machine’s internal decomposition. [“The World and the Machine”](https://courses.cs.washington.edu/courses/csep503/19wi/schedule/papers/TheWorldAndTheMachine.pdf)

## The missing specification kernel

The foundational relationships should be:

```text
D ∧ S ⇒ R        requirements adequacy
M ⇒ S            implementation conformance
```

Where:

- `D` is domain knowledge: facts and assumptions about Python, tinygrad, NumPy, Metal, bf16, hardware, and user programs.
- `R` is the requirement: the effect required in the problem world.
- `S` is the machine specification, stated only over phenomena shared between Tgrad and its environment.
- `M` is the actual Tgrad implementation.

Zave and Jackson distinguish indicative domain descriptions—what is true—from optative requirements—what we want to become true. They then require the specification and domain knowledge together to entail the requirement. [“Four Dark Corners of Requirements Engineering”](https://doi.org/10.1145/237432.237434)

Tgrad currently has excellent fragments of this model, but not the relationship itself.

- [Ontology.lean](../Tgrad/Ontology.lean#L30) begins with `dtype`, `shape`, `uop`, `kernelDecl`, `metalSource`, and `buffer`. Those are machine/design concepts, not problem-world concepts.
- [Parity.lean](../Tgrad/Spec/Parity.lean#L198) calls an upstream symbol/test inventory a `Requirement`, but the record contains no behavioral predicate.
- [EvidenceRef](../Tgrad/Spec/Parity.lean#L107) is quite strong: it records subject, verifier, adapter, environment, relation, artifacts, and validator.
- What is missing is the typed connection from requirement to specification, and from specification to implementation evidence.

The current substrate therefore knows a great deal about evidence provenance and work organization, but cannot yet express the central claim:

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
| Python substitution | Python importer, client program, tinygrad package | Modules, names, attributes, signatures, exceptions |
| Tensor transformation | Tensor values, shapes, dtypes, mathematical operations | Calls, input values, output values, dtype/shape results |
| Views and workpieces | Base storage, views, aliases, mutations | Index maps, storage identity, lifetime, readback |
| Commanded execution | User program, lazy graph, runtime | Realization requests, completion, errors, diagnostics |
| Stateful behavior | Random state, assignment, autograd, JIT/cache | Seeds, mutations, graph state, cache effects |
| Backend connection | Metal/CPU/device environment | Allocation, dispatch, synchronization, device errors |
| Quality/performance | Workload and hardware profile | Latency distributions, memory use, throughput |

This immediately explains the “six operations but still 0/34” observation in [PLAN_OP_BREADTH.md](../PLAN_OP_BREADTH.md#L183).

Twenty-nine files fail during collection because `tinygrad.helpers` and related module phenomena are absent. That establishes failure in the Python-substitution frame. It does **not** establish failure of the newly implemented tensor operations, because those tests never reach them. Their correct status is `unobserved behind prerequisite`, not `failed`, `absent`, or `conformant`.

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

## Use the four-variable model at the Python/Lean boundary

Parnas and Madey’s four-variable model is especially useful here. It separates world variables from machine-interface variables. [“Functional Documents for Computer Systems”](https://doi.org/10.1016/0167-6423(95)96871-J)

For Tgrad:

- `m`: monitored world phenomena—Python calls, tensors, seeds, device state.
- `c`: controlled world phenomena—returns, exceptions, tensor values, mutations, device effects.
- `i`: Lean-machine inputs—handles, op codes, shapes, bytes, graph nodes.
- `o`: Lean-machine outputs—handles, buffers, result codes, diagnostics, traces.

Then define four relations:

```text
REQ(m, c)       tinygrad-compatible required behavior
IN(m, i)        Python/C adapter encoding
SOFT(i, o)      behavior of the Lean implementation
OUT(o, c)       decoding back into Python-visible behavior
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

As of the baseline named at the top of this document, the planes read:

| Plane | Current observation | Correct status |
|---|---|---|
| Upstream extraction | Manifest exists for `19c4d736…` | Candidate target observed |
| Lean target | `targetUpstream` remains `.unknown` | Not promoted |
| Canonical product | `main` at `c465f89` | Product baseline |
| General operation spine | graph-indexed realization, broadcast pointwise operations, reductions, and fused reduce-of-elementwise matmul are merged | Implemented; each claim still depends on its own evidence |
| Specialized matmul route | retained alongside the general expression route | Optimization candidate, not the semantic definition of matmul |
| L12 performance predicate | its flags now execute correctly and the frozen-baseline predicate is red and non-repeatable | Open measurement-methodology failure, not a codegen verdict |
| API parity | 0/34, with 29 collection failures | Substitution surface absent; most operation semantics unobserved |

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

## A better Lean decomposition

I would introduce this beside the current specs rather than rewrite everything at once:

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
∧ evidence names the exact subject, verifier, adapter and environment
∧ no prerequisite or obstacle leaves the result merely unobserved
```

That is stricter than “test passed,” but it is also far more informative. It tells us whether the gap is in requirements, the adapter, semantics, implementation, evidence, or environment.

The deepest redesign is therefore not “add more facts to Lean.” It is:

> Make Lean hold the argument connecting world requirements to machine specifications, implementations, observations, and derived work.

That turns Tgrad from a codebase with a formalized roadmap into a codebase that can mechanically explain what it is for, what it currently guarantees, what remains unknown, and which transformation most directly advances it.

# Growing Tgrad: work by the codebase and work on the codebase

This document records the design learned from the deep review, the repair
work, and the “Lean as a Universal Specification Language” philosophy. Its
purpose is not to make the repository look complete. Its purpose is to make
the next increment of Tgrad easier to select, execute, observe, validate,
promote, and—when necessary—reverse.

The executable counterparts are:

- `Tgrad/Ontology.lean`: stable vocabulary of the product domain;
- `Tgrad/Spec/RuntimeWork.lean`: repeatable work performed **by** Tgrad;
- `Tgrad/Spec/Growth.lean`: the feedback bridge from observations to change;
- `Tgrad/Spec/Evolution.lean`: the event protocol for work performed **on**
  Tgrad;
- `Tgrad/Spec/Work.lean`: the current findings-to-work projection and picker;
- `Tgrad/Spec/LiveConditions.lean`: re-observable limits on concurrency.

Run `.lake/build/bin/tgrad-spec` to query the current model.

## 1. The two meanings of work

The word *work* was hiding two different ontologies.

### Work by the codebase

This is a repeatable capability of a revision. It consumes artifacts and
produces artifacts. It can be run more than once without changing the source
tree merely because it ran.

There are three realms:

1. **Product work** transforms tensors: ingest, represent views, normalize,
   select a route, lower, render, compile, allocate, dispatch, synchronize,
   and materialize.
2. **Verification work** transforms a candidate revision plus scenarios into
   observations and validation evidence: build checks, regressions,
   differential execution, performance experiments, and provenance checks.
3. **Specification work** transforms the checked schema into findings,
   frontiers, integrity results, and reports.

Verification and specification are reflexive work: the codebase performs
computation *about* itself. They still do not mutate the repository.

### Work on the codebase

This is repository evolution. It has a different lifecycle and different
authorities:

```text
intent
  -> leased attempt at a base revision
  -> immutable candidate tree with observed effects
  -> check runs tied to that exact tree
  -> promotion certificate
  -> new revision
```

An intent is not an attempt. An attempt is not a candidate. A candidate is not
evidence. Passing checks are not acceptance. A commit identifier by itself is
not a promotion certificate.

Conflating these facts caused several earlier pathologies: work was “complete”
because prose or a commit string said so; evidence named a tree that was not in
the repository; and a gate result survived after its inputs changed.

## 2. The hinge between them

The two work domains form one feedback system:

```text
revision
  -> runtime capability
  -> execution under a scenario
  -> observation
  -> evidence with provenance
  -> claim update or finding
  -> growth case
  -> evolution intent
  -> attempt
  -> candidate tree
  -> verification/specification work
  -> promotion certificate
  -> revision
```

Every arrow matters. In particular:

- a runtime event is not automatically an observation;
- an observation is not evidence until its method, scenario, subject tree,
  and artifact provenance are retained;
- evidence does not decide policy by itself;
- a finding needs an upgrade path;
- an attempted repair does not change the product capability until promoted;
- a check only speaks about the exact candidate tree it names.

`Growth.Case` represents the middle of this loop. It links observations,
findings, evolution work, intended capability deltas, validators, acceptance,
and rollback. `Evolution.applyEvent` represents the state-changing end.

## 3. Three state planes

The system is easier to reason about when state is divided into three planes.

### Product capability state

What can this revision do, over what domain, through which route, with which
resources and failure modes?

The current capability vocabulary is:

- `loadBearing`: this is the authoritative path;
- `bounded`: real, but only for an explicit domain;
- `referenceOnly`: retained as an oracle, not production behavior;
- `bypassed`: the abstraction exists but does not govern the result;
- `missing`: the capability is explicitly absent.

This distinction came directly from the renderer and scheduler findings. A
function can exist, compile, and even be exercised by a CLI while production
bypasses it. “It ran” is weaker than “its output governed execution.”

### Epistemic state

What do we know about the product state, and how could that knowledge improve?

Claims are `confirmed`, `tentative`, `unknown`, or `deferred`. Each state
contains evidence, a basis and upgrade method, a resolution method, or a
return condition. Unknowns are first-class work inputs, not embarrassing
holes to conceal.

### Evolution state

What change is authorized, who or what is attempting it, which tree resulted,
which checks ran on that tree, and whether it was accepted?

This plane is event-sourced. Derived dashboard states such as “in progress” or
“complete” should be projections of events, not mutable facts with no history.

## 4. What Tgrad actually does today

The runtime is not yet one uniform compiler path. It is a small collection of
routes that share some stages:

1. the sentinel route uses the parametric manual-load generator;
2. the aligned general tensor-core route uses that same generator;
3. the scalar fallback emits one-thread-per-output-element code;
4. view matmul uses view-derived index expressions and a scalar kernel;
5. view readback rangeifies a movement chain, renders a bit-preserving indexed
   copy, and returns a contiguous buffer for `.numpy()`/`.to_bytes()`;
6. CLI sentinel execution has its own bounded entry path.

Rangeification is real, the Lean renderer is load-bearing, and `fd945b1` makes
generated tensor-core declarations/names/geometry authoritative for every
sentinel. The per-shape declarations and parser have now been deleted. Captured
MSL survives only as an independent executable oracle: L12 requires all 11
generated sources to differ and all 11 output buffers to agree bit-for-bit.

The runtime inventory records, for each unit:

- realm, verb, inputs, and outputs;
- owning component;
- resources and scale drivers;
- implementation evidence and support state;
- observation method;
- failure surface.

This prevents a flat component list from hiding dynamic behavior. GPU dispatch
scales differently from graph normalization; buffer allocation fails
differently from a provenance check; and a bounded, checked-but-unpromoted
view-materialization candidate before promotion was not the same state as a
bypassed performance verifier; the promoted materializer is now explicitly
bounded rather than being upgraded to an unqualified capability.

View materialization exposed five reusable design rules for work **by** the
codebase:

1. **The semantic owner must supply the shape.** Python's cached view shape and
   `Tensor.shape` are useful API projections, but `Schedule.viewOfUOp` is the
   validator for collapsed shape/stride/offset semantics. The materialized
   result is sized from that validated view and checked back against Python.
2. **A pass is load-bearing only when its output governs execution.** The copy
   kernel consumes the index UOp produced by `rangeify`; recomputing the same
   formula beside the scheduler would leave the scheduler observational.
3. **Representation-preserving work is not numerical work.** Readback copies
   `ushort`, not `bfloat`, because the contract includes NaN payloads, signed
   zero, infinities, and subnormals—not merely approximately equal numbers.
4. **Invalid boundary states fail before resources are touched.** Empty views,
   unsupported chains, out-of-range source indices, undersized root buffers,
   signed-index overflow, and native-size truncation are rejected before Metal
   allocation or dispatch.
5. **A local capability can expose lifecycle debt.** Every readback currently
   compiles a library and registers a temporary Tensor in append-only caches.
   Correctness is bounded and real, but repeated-read efficiency and registry
   reclamation remain separate work rather than being hidden in the success
   claim.

The codegen, gate-repair, and provenance work added six rules for verification
work:

1. **Strengthen before removing.** L12 gained semantic C3 while its old
   byte-equality Layer C was still green. The later deletion retired only the
   transcription-specific layer; no migration pressure created a moment where
   the predicate had to be weakened just to proceed.
2. **Complementary checks are separate obligations.** `Nodup` proves store
   offsets do not collide. It does not prove they are correct: `c -> c+2` and
   `24*K+1 -> 24*K+2` both left the build green while execution diverged for
   all 11 kernels. Structural non-aliasing and behavioural placement therefore
   remain independently required.
3. **Auditing, repairing, regenerating, and enforcing are different work.**
   The provenance auditor established a 37/37 absent-commit baseline. The
   `7c7dc0f` attempt repaired two blockers and retained 11 files genuinely
   produced by their scripts, moving the current observation to 26/37 absent
   commits, 76/115 unresolved hashes, 28 bad roll-ups, and 17 writer
   mismatches. That is meaningful progress, but not a coherent snapshot.
   Enforcement becomes honest only after complete regeneration produces a
   passing subject; making a known-red auditor fatal earlier would add
   friction, not integrity.
4. **Semantic and performance gates must not hold each other hostage.** The
   generated route remained 50/50 correct while L12's frozen-baseline
   diagnostic changed from 37/50 misses at `1/1` to zero at `30/30`. L12
   certifies generated semantics; the separately scheduled `perf.rebaseline`
   owns performance under symmetric measurement.
5. **Code-inspection predicates must distinguish code from commentary.** L12's
   anti-replay grep turned red because an accurate Lean comment mentioned the
   `readFile` behavior that had been deleted. The repair strips line comments
   before matching and still catches a real call. Rewording accurate source
   documentation merely to placate a grep would optimize the codebase for its
   checker rather than strengthen the checker.
6. **Repeatability precedes thresholds.** Identical consecutive L11 `30/30`
   runs missed 2, 25, and 10 of 50 rows, with maxima 1.655, 3.667, and 2.552.
   The variance is larger than the effect the fixed 1.5 rule tries to detect.
   Future performance work must pair both live systems in one run, retain raw
   distributions, estimate within-run and between-run variance, and derive a
   decision rule before applying it. If those observations cannot support a
   stable rule, the result is indeterminate.

## 5. Capability, execution, observation, and evidence are different

Four terms need stable meanings:

- **Capability**: repeatable work a revision can perform.
- **Execution**: one invocation of a capability under concrete inputs and
  ambient conditions.
- **Observation**: selected facts extracted from an execution.
- **Evidence**: an observation packaged with subject identity, method,
  scenario, artifacts, and provenance so that a claim can rely on it.

The benchmark audit exposed the cost of collapsing these. A synchronized wall
clock number was a real observation, but it did not support a kernel-speed
claim because the two executions enclosed different work. A byte-equal kernel
was real evidence of textual identity, but not of independent generation. A
WMMA substring was an observation of text, not evidence that the corresponding
route governed execution.

The subject of evidence must be immutable. In the evolution protocol every
`CheckRun` names a `CandidateId` and its exact tree hash. Promotion rejects a
check whose tree differs from the candidate, even if the command and output
look plausible.

## 6. Growth is a vector, not a score

Repository size, gate count, and one “maturity” number are poor measures of
growth. Tgrad tracks intended deltas along independent aspects:

- supported domain;
- semantics;
- safety;
- architecture;
- performance;
- observability;
- provenance;
- maintainability.

Useful change kinds include adding a capability, widening a domain, repairing
semantics, strengthening safety, replacing a bypass, improving observability,
improving efficiency, deleting scaffolding, and upgrading evidence.

Consequences:

- replacing transcribed kernels with generated kernels is architectural and
  maintainability growth even if the first honest benchmark regresses;
- rejecting view readback was safety and semantic growth although it narrowed
  the successful API surface;
- deleting dead model scaffolding can be growth;
- discovering that a green performance claim is unsupported can be epistemic
  growth while decreasing the count of promoted claims;
- optimizing without preserving numerical and safety obligations is not
  growth of the whole system.

Growth therefore need not be monotone in every coordinate. Promotion records
the claimed delta and residual risks instead of manufacturing a universal
green status.

## 7. Validation obligations are orthogonal

The previous model treated validation as a ladder:

```text
syntactic < structural < semantic < differential < performance
```

That is wrong. Performance is not stronger than correctness. A build does not
establish provenance. Differential agreement does not establish safe
ownership. Human acceptance cannot substitute for numerical execution.

The checked model now uses a set of independent obligations:

- build;
- unit regression;
- API contract;
- safety;
- numerical;
- semantic;
- differential;
- performance;
- provenance;
- resource isolation;
- human review.

A promotion certificate lists the obligations required for that change and
the exact check runs that discharge them. `Evolution.applyEvent` rejects
obligation substitution. The work picker does not assign a larger priority
merely because a task mentions performance.

## 8. Work intent, attempt, candidate, check, and promotion

### Intent

An intent states the desired change, findings addressed, dependencies,
runtime scope, expected write set, resource needs, acceptance obligations,
and recovery plan. It is planning authority, not evidence that work began.

### Attempt

An attempt names its actor, base revision, authorized effect budget, and
resource lease. Starting an attempt can fail because its intent is unknown,
its lease is malformed, a resource is at capacity, or its effects overlap an
active attempt.

The lease uses a logical event epoch in the pure model. A real orchestrator
must additionally bind it to wall-clock/process liveness and renew or abandon
it explicitly.

### Candidate

A candidate is an immutable tree produced by an attempt. Its observed effects
must fit within the attempt’s authorized effects. The current model uses exact
declared scopes; a future integration wrapper should expand directory scopes
and compare them against the actual Git diff.

### Check run

A check run records validator capability, obligation, outcome, command, and
artifact digest. The validator must be reflexive work that is currently
`loadBearing` or `bounded`; a missing or bypassed validator cannot certify a
promotion merely because its name exists.

Failed and blocked checks are still useful observations. They should be
recorded rather than erased by a green-only evidence store.

### Promotion

A promotion certificate names a growth case, candidate, exact check runs,
required obligations, accepting actors, clean target revision, and residual
risks. Promotion fails when a required obligation is absent, any selected
check failed, the target tree differs, or the certificate is incomplete.

Rollback is part of the growth case because a failed rollout is normal
evolution state, not an exception to the model.

## 9. The shape of the codebase

Tgrad now has separate product and specification build roots.

```text
Tgrad product root
  stable domain types and executable implementation
        ^ compile-time symbol pins
        |
TgradSpec root
  epistemic claims
  architecture and runtime-work inventory
  findings and growth cases
  live resource conditions
  evolution protocol and work graph
```

The specification imports the product to pin real types and functions. The
product does not import roadmap state, audit prose, or maturity judgments.
This avoids making mutable planning data part of the runtime ABI.

Within the specification, separate by rate of change:

1. **Ontology** changes when the language of the domain changes.
2. **Architecture/runtime work** changes when computational structure changes.
3. **Findings** change when observations update claims.
4. **Growth cases and work intents** change when action becomes possible.
5. **Live conditions** are re-probed whenever concurrency/resource decisions
   are made.
6. **Evolution events** append as work executes.

The old `Tgrad/Model/*` approach mixed authored maturity judgments, roadmap
prose, and self-validating literals. It demonstrated schema shape but did not
ground most claims in runtime observations. It remains outside both roots and
should eventually be deleted in an isolated change.

## 10. Parallelism follows the shape of work

Authoring and verification have different conflict structures.

### Authoring

Independent write sets can be authored in parallel after dependency and
authority checks. The first durable attempt history is now complete: 64×64
tensor-core generator parameterization landed as `75f856b`, and its renderer
lease has been released.

Two disjoint attempts were then active:

- view materialization over `Pipeline.lean`, `PythonFFI.lean`, and
  `python/tgrad.py`;
- the captured/generated differential harness over `Main.lean` and its new
  comparison script.

The differential harness landed as `a6d5958`: all 11 generated sources differ
from their captures while all 11 executions are bit-identical over 240 MB of
output. Its lease is released and `verify.codegen-differential` is now
load-bearing. A falsification changed one store from `c` to `c+2`; the
pairwise-distinct theorem remained green while execution differed in 727,933
bytes. The theorem proves non-aliasing; the differential proves right
placement. Neither substitutes for the other.

The first view candidate (`995eb7e…`) passed checks, then became stale when
HEAD advanced. Its attempt was abandoned rather than laundering those checks
onto a new tree. That happened once more for `ac393d2…` after the semantic-gate
and evidence-audit commits. A final isolated worktree based at `bdc01b0`
produced tree `790d413…`; the full suite passed and that exact tree was committed
as `e6241bd`. The two stale candidates remain attached only to abandoned
attempts, making invalidated work visible instead of erasing it.

The view lease was released, then `codegen.route-sentinels` landed as
`fd945b1`. Its exact tree `8475550…` passed build, eligibility/safety,
64/96/128 production-route numerical checks, and the 11-sentinel semantic
differential. The route lease is released. The subsequent deletion removed the
per-shape declaration table and parser, migrated semantic gates, and leaves
`perf.rebaseline` as the next computed safe authoring frontier. Its exact-tree
deletion promotion is recorded separately; no codegen writer remains active.

The subsequent evidence-regeneration attempt is also in the live ledger, but
as an abandoned candidate rather than a promotion. Tree `5ccb293…` contains
two gate repairs and 11 reproducible evidence files. Its provenance check is
still red, and two performance-repeatability checks failed. This records an
important state that a mutable “done” flag cannot express: useful committed
effects survived, while the claimed all-gate regeneration did not complete.
The owner explicitly authorized this diagnostic sweep before
`perf.rebaseline`; that authorization allowed observation, not promotion past
an unsatisfied dependency.

The frontier is recomputed from dependencies, current progress, active
writers, live attempt events, and declared effect sets. It is not a standing
instruction to spawn three worktrees.

### Verification

Verification is serial on this repository and machine:

- 141 distinct fixed `/tmp/tgrad_*` paths make concurrent gate/devcheck runs
  clobber one another;
- the Lean build tree is shared;
- the machine has one Metal GPU;
- timing experiments interfere even across separate processes;
- evidence generation writes shared committed fixtures.

Separate processes avoid the process-global Metal state race, but they do not
make one GPU suitable for concurrent timing. Threads in one runtime remain
unsafe because the Metal globals, LRU, caches, and Lean registry are not
protected.

Each additional worktree costs roughly 239 MB of `.lake` artifacts, and free
disk was about 3.5 GB when observed. This is a live condition, not an ontology
fact, so it must be re-probed.

The resulting operating model is unusual but clear:

```text
parallelize investigation and disjoint authoring
serialize builds that share artifacts
serialize all gate/devcheck execution until temp paths are namespaced
serialize all GPU performance work
use one integrator to record candidate checks and promotion
```

## 11. What to mechanize—and what not to mechanize

The bitter lesson applies to repository evolution too. Encode the substrate
that makes better reasoning useful; do not freeze today’s reasoning policy.

Mechanize:

- identities and typed references;
- artifact flow;
- support states and explicit bounds;
- epistemic upgrade paths;
- dependency and effect conflicts;
- resource capacities and leases;
- exact candidate-tree provenance;
- independent validation obligations;
- promotion and rollback conditions;
- event replay and rejected transitions;
- queries over findings, gaps, and frontiers.

Do not hardcode as timeless truth:

- one priority heuristic;
- one repair strategy;
- one agent decomposition;
- one benchmark threshold;
- today’s disk capacity;
- the present list of findings as exhaustive;
- a fixed order of validation “strength”;
- model-generated claims about operational reality.

`priorityScore` is explicitly a bootstrap heuristic. Agents may reason more
deeply about expected information gain, user value, or risk, but the selected
work must still satisfy the checked dependency, authority, effect, resource,
and promotion substrate.

## 12. Failure patterns to keep visible

The review produced recurring patterns that should remain named:

- **Schema theatre:** rich records whose fields are authored prose and whose
  theorem only checks that the prose is non-empty.
- **Validation theatre:** grepping for strings, self-grepping gate scripts, or
  checking evidence-file existence while claiming runtime behavior.
- **Route theatre:** generating or tracing a route that does not govern the
  production result.
- **Provenance theatre:** writing hashes but never recomputing them; recording
  a commit without verifying it exists.
- **Differential theatre:** comparing an implementation with a transcription
  of the same fixture and presenting equality as independent agreement.
- **Benchmark boundary mismatch:** comparing different work while describing
  the result as kernel performance.
- **Green-only history:** retaining successful snapshots while discarding
  failures, blockers, and environment changes.
- **Mutable completion:** overwriting “planned” with “complete” without an
  attempt, candidate, checks, or acceptance event.
- **Ontology churn:** encoding temporary bugs or roadmap state as permanent
  constructors.
- **Policy petrification:** putting today’s prioritization or repair strategy
  into the stable substrate.

The answer is not “more fields.” Each field needs an authority, observation
method, update event, and consumer that changes behavior when the field
changes.

## 13. Current mechanized guarantees and limits

### Guarantees compiled today

- runtime-work IDs are unique;
- their artifact flow is closed over declared external artifacts;
- product, verification, and specification realms are all non-empty;
- bounded view materialization and bypassed performance/provenance validation
  are explicit;
- captured/generated execution differential is load-bearing and source
  inequality is part of its contract;
- growth cases refer to known runtime work, findings, and evolution work;
- every open finding enters a growth case;
- promoted growth cases refer only to completed roadmap items;
- work dependencies and runtime scopes are known and non-self-referential;
- planned work has non-empty validation obligations and discriminating prose;
- the current dependency-ready authoring frontier is computed;
- stale-tree check evidence is rejected;
- one validation obligation cannot substitute for another;
- promotions require passed checks on the exact candidate tree;
- promoted view tree `790d413…` has all five required obligations; stale
  `995eb7e…` and `ac393d2…` checks remain attached only to abandoned attempts;
- the provenance auditor is load-bearing and calibrated in both directions,
  while fatal enforcement remains explicitly bypassed until regeneration;
- performance-repeatability diagnosis is bounded and load-bearing, while
  performance validation itself remains bypassed;
- partial evidence tree `5ccb293…` has failed provenance/performance checks,
  is attached to an abandoned attempt, and has no promotion certificate.

### Honest limits

- the completed warp-parameter attempt is the first fully replayable promotion
  in the durable ledger; view materialization is the first history that also
  records abandoned stale candidates before exact-tree promotion;
- historical `Progress` values remain a migration projection because those
  commits predate the attempt/candidate/check protocol;
- path effects are declared exact scopes, not yet checked against a Git diff;
- leases use logical epochs and are not yet bound to process liveness;
- `applyEvent` does not yet receive roadmap progress, so it cannot distinguish
  an ordinary dependency-ready attempt from an owner-authorized diagnostic
  override; the checked projection records that this regeneration was not
  dependency-ready and was not promoted;
- observations and check artifacts are not yet persisted by a standard runner;
- the finding registry is scoped to the active correctness/codegen/evidence
  program, not the complete deep-audit backlog;
- several existing gate capabilities are correctly classified as bypassed;
- the model checks reference integrity, not the truth of arbitrary prose.

The spec represents the missing durable history as `unknown` with a concrete
upgrade path. Historical events should be backfilled only when exact trees and
artifacts can be recovered; invented completeness would be worse than an
explicit gap.

## 14. How a future change should flow

1. Re-probe live conditions: Git state, active writers, disk, build process,
   temp namespace, GPU activity, and evidence store.
2. Query open findings and growth cases.
3. Select an intent whose dependencies are complete and whose effect set does
   not overlap active attempts.
4. Start an attempt with base revision, actor, effect budget, and leases.
5. Author without silently widening scope. If scope must widen, record a new
   authorization event rather than editing history.
6. Produce an immutable candidate and compare the actual diff to the budget.
7. Determine required obligations from the growth delta—not from a global
   hierarchy.
8. Run checks serially where resources require it. Give every run a unique
   temp namespace and retain raw artifacts.
9. Record failed and blocked checks as well as passed checks.
10. Promote only with exact-tree passed runs covering all required
    obligations and named acceptance authority.
11. Update capability state, findings, and growth metrics from the promoted
    event.
12. Re-query the frontier. The next action should be a computed consequence
    of the new state.

## 15. The next mechanization increments

The substrate is useful now, but “mechanized growth” becomes operational only
when ordinary development commands emit its events. The next increments are:

1. **Persist an append-only evolution log.** Use a reviewable text format and
   replay it through `Evolution.applyEvent` during `tgrad-spec`.
2. **Add an attempt wrapper.** Capture base commit/tree, actor, leases, and
   authorized paths before an agent edits.
3. **Check actual effects.** Expand directory budgets, inspect `git diff`, and
   reject or request authority for out-of-budget changes.
4. **Type dependency overrides.** Require `attemptStarted` to prove roadmap
   readiness or carry an explicit owner authorization whose candidates cannot
   promote until the missing dependencies pass.
5. **Namespace verification.** Replace fixed `/tmp/tgrad_*` paths with one
   run-scoped directory and record the run ID.
6. **Standardize check artifacts.** Record candidate tree, command, environment
   fingerprint, raw output digest, validator version, start/end, and outcome.
7. **Build a promotion command.** It should query missing obligations, reject
   stale trees, and emit the certificate; it must not manufacture pass results.
8. **Project dashboards from events.** Replace hand-authored `Progress` values
   once enough new history exists.
9. **Broaden observation probes.** Instrument which scheduler output, renderer
   source, route, grid, and kernel actually govern each dispatch.
10. **Ingest the remaining audit backlog.** FFI ownership, registry/caches,
   threading, diagnostics, gate integrity, and API safety need normalized
   findings and growth cases.
11. **Close the codegen loop honestly.** Keep the landed differential oracle,
    widened generator, typed-store theorem, and semantic gate load-bearing;
    with routing and deletion complete, measure generated performance serially.

## 16. The criterion for a growing codebase

A codebase is mechanizing its growth when it can answer, from checked and
replayable state:

- What work can this revision perform?
- Which path actually governs each result?
- What is missing, bypassed, bounded, or reference-only?
- What observations support each claim, for which exact tree?
- Which findings follow, and how can their epistemic state improve?
- What changes are authorized and dependency-ready?
- Which can be authored concurrently, and which checks must serialize?
- What candidate did an attempt produce, and did it stay within scope?
- Which independent obligations remain before promotion?
- Who accepted the candidate, with what residual risks and rollback?
- How did promotion change the capability, epistemic, and evolution planes?
- What does the new state make ready next?

Tgrad cannot answer all of these operationally yet. It can now state the
questions in Lean, reject several invalid transitions, expose the current
frontier, and represent the remaining gaps without pretending they are
finished. That is the foundation on which the codebase can grow without its
specification drifting into a parallel fiction.

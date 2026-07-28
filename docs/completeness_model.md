# Mechanising the completeness claim

**Status:** accepted design, revised 2026-07-28 against integrated `main`.
This document refines the four-predicate model in
[`requirement_engineering.md`](requirement_engineering.md): catalog closure,
requirement discharge, profile completion, and evolutionary completion. It is
an architecture for making those claims hard to forge. It is not itself a
completion certificate.

## Current mechanical boundary

The first authenticated input boundary is now integrated:

- target candidate `tinygrad@19c4d736f2bc8e26d21f08b28ffd6298408da00f`,
  tree `855cca3b00c38841a6d3a043284f3a2ca696d4b0`;
- 1,562 foreign Git blobs authenticated from the pinned object graph;
- 331 upstream Python test files, with a distinct 138-file API-surface policy
  subset;
- 307 selected Tensor declarations, collapsing to 295 unique method names,
  47 methods declared directly on `Tensor`, and 5 unique properties;
- 82 `Ops` members, 52 dtype names, and 16 backend declarations;
- canonical closure SHA-256
  `ae93a447ecd98b7bcb9abd3c282e46c56a1cf313b13648253c276a91a5eb1c73`.

The target disposition remains `extracted_candidate`. The bundle deliberately
does **not** assert target promotion, catalog closure, interpretation of the
590 candidate requirement rows, runtime/build identity, scenario adequacy,
runtime parity, or profile completion. Those omissions are typed limits in
`Tgrad/Contract/SourceClosure.lean`, not prose caveats bolted on afterward.

This ordering matters. The rejected branch
`mechanics/synthetic-completion-v1` at `5c6b96e` predates the authenticated
foreign closure and must not be merged. A later synthetic campaign must consume
the promoted target and authenticated closure; otherwise it can demonstrate
only consistency among locally authored tokens.

## 1. The premise: certificates are not proofs

A `structure` carrying `inventoryHash : Hash` and `adequacy :
AcceptedAdequacy` is a record. Lean can check that the record is internally
consistent. Lean cannot check that the hash was computed rather than typed, or
that `.accepted` was earned rather than asserted. Making `complete` derived
rather than a field does not fix this: it relocates the forgery from the
conclusion to the premises.

This is not hypothetical. Repeated defects in this repository have had the
same shape—a check whose two sides came from the same place:

| defect | the two sides |
|---|---|
| pilot `.stale` unreachable | `context.subjectTree` and `observation.subjectTree` both came from generated pilot data |
| umbrella gates vacuous | `[[ -f evidence.json ]]` where the file was committed with the gate |
| L12 performance sweep | a comment asserted sampling flags the shell did not pass |
| old gate evidence | 26 of 37 files named a commit absent from the released history |
| first source-closure repair | generated output and an extractor source differed by one trailing byte |

The design rule for every claim-bearing field is therefore:

> **Adversarial distance.** A field is admissible only if the party that
> benefits from it being green is not the party that determines its value.

There are four admissible provenance classes:

1. **Imported:** recomputed from a pinned foreign source.
2. **Derived:** recomputed from imported inputs by an identified verifier.
3. **Calibrated:** supported by attacks that were executed and failed.
4. **Judgment:** accepted by named authority and forced to narrow the printed
   claim, with explicit invalidation conditions.

A field that is none of these is decoration. A locally authored digest token
is not imported merely because its Lean type is called `ContentDigest`.

## 2. The three shapes that must stay distinct

Completeness work fails when three different graphs are collapsed into one.

### Functionality shape

This is the problem-world and behavior graph: public spellings, admitted
programs, state transitions, values, errors, effects, performance boundaries,
backends, and compositions. It comes from the promoted upstream source closure
plus explicit profile judgments. It must not be shaped by what Tgrad currently
implements.

### Codebase shape

This is the replacement machine: Python authoring shim, Lean semantics and
graph, scheduling and rewriting, renderers, runtime, C/Metal boundary, and
backends. Product symbols and runtime stages ground requirements, but source
paths do not establish behavior. The architecture law remains: deleting
Python may delete the user syntax and marshalling, but must not delete the
semantics.

### Work shape

This is the causal dependency graph from current gaps to candidate
transformations. A requirement partition need not equal an implementation
packet: several requirements can share one cause, and one public method can
induce several requirements. Work is generated from gaps, dependencies,
resources, leases, and validators. It is not a feature list an agent marks
done.

The control loop connects these shapes without identifying them:

```text
pinned upstream source closure
  -> extracted inventory and accepted dispositions       functionality
  -> executable requirements, partitions and scenarios   functionality
  -> neutral observations and derived gaps
  -> dependency-shaped work packet                        work
  -> agent-produced candidate tree                         codebase
  -> independent validation and falsification
  -> exact-tree promotion and recomputation
```

## 3. Where the forgery surface actually is

The broadcast-add observer already covers eleven dimensions and has eight
calibration mutants. The largest remaining exposures are broader:

| # | possible forgery | required defence |
|---:|---|---|
| 1 | inventory omits or over-excludes | authenticated source closure plus total disposition ledger |
| 2 | scenarios dodge hard admitted programs | upstream-derived generators and explicit search records |
| 3 | fewer dimensions are declared than the requirement class needs | mandatory-dimension policy independent of the contribution |
| 4 | an agent types `adequacy := .accepted` | derive assurance from challenges or record explicit authority |
| 5 | an environment set is broader than what ran | exact environment and runtime attestations |
| 6 | ontology or verifier changes after seeing the result | full-history freeze integrity and separate contract transactions |
| 7 | a candidate selects an easier profile or target | owner-promoted target/profile identity outside product transaction |
| 8 | a generated and a hand-authored artifact agree on a lie | regenerate both from authenticated raw inputs and compare canonical closure |

The profile denominator is therefore not “the 590 rows” and not “the 138 test
files.” Those are candidate inventories. The denominator closes only after
every authenticated source item has one accepted disposition and all required
dispositions map to executable requirements.

## 4. Scenario closure is the weakest link

The likeliest false green is a requirement whose scenarios quietly omit cases
Tgrad cannot express. For example, `Tensor.realize()` returning `self` matches
an already-materialized identity scenario, but upstream `realize()` can force
an unrealized graph. Tgrad's inability to construct that input is a divergence
or blocker, never an exclusion.

> **Admission rule.** A program belongs to the profile because promoted
> upstream and accepted scope judgments admit it. Tgrad support has no vote in
> admission.

Interactions over 295 method names cannot be closed by an authored finite
list. The honest evidence is a reproducible search record:

```text
generator identity and grammar
promoted source closure and profile
seed derivation, seeds, partitions, and budget
subject/runtime/adapter/relation/environment identities
divergences and harness failures found
mutation calibration of the search and evaluator
```

Authored scenarios remain valuable regression pins. They are not the closure
argument. “No divergence in N programs from G at seeds S” is bounded evidence,
not a universal quantifier, and the rendered claim must say so.

## 5. Adequacy cannot be asserted

`D ∧ S ⊨ R` is semantic entailment over a real-world domain. It is not made
mechanical by adding an `.accepted` constructor. Use executable challenges:

```lean
structure AdequacyChallenge where
  requirement   : RequirementId
  dimension     : Dimension
  program       : ProgramRef
  satisfiesSpec : ObservedFact
  violatesReq   : ObservedFact
```

Then derive a graded state:

```text
no challenges                         -> open
any observed counterexample           -> refuted
every mandatory dimension attacked,
  no counterexample, policy permits it -> survivedSearch
accepted external rationale           -> acceptedBy(judgment)
machine-checked theorem                -> proved
```

An agent may add challenges, which can only make acceptance harder. An
unattacked mandatory dimension stays open. `survivedSearch` records the
generator, source closure, seeds, budget, partitions, and calibration; it must
never render as `proved`.

## 6. Prospectivity is a repository-history property—but not only that

The integrated checker
`scripts/contract/check_real_chronology.py` establishes two real facts for the
five declared broadcast-add cycles:

1. each freeze commit is an ancestor of its candidate commit; and
2. the judging closure is byte-identical at **every** commit on the
   freeze-to-candidate ancestry path.

Checking every intermediate commit is stronger than an endpoint diff: it
rejects moving the ruler and reverting it before the candidate. The older
`scripts/spec/check_cycle_prospectivity.py` from the unmerged design branch
checked only endpoint path differences, so it is intentionally not imported.

The real checker is nevertheless not a complete prospectivity certificate.
Its cycle registry is transcribed, and it does not yet bind all four events:

```text
calibrated observation_before
  -> frozen work packet
  -> implementation candidate
  -> clean-tree observation_after
```

The required next form combines both strengths:

- all four commits exist and are ordered by ancestry;
- every commit on freeze→observation-after preserves the generated judging
  input closure, including ontology, observer, relation, adapter, calibration,
  environment policy, and claim renderer;
- the cycle registry is derived from promotion/rejection/abandonment events,
  so an inconvenient cycle cannot be omitted;
- a `prospective` label additionally requires a blind/freeze protocol
  attestation. Git ancestry proves order and ruler stability, not that the
  candidate was unknown when the packet was authored.

## 7. No authored file may construct a completion claim

Generated modules already have a reproducibility pattern: generated Lean says
not to hand edit it, and a generator's `--check` mode rejects drift. Extend
that pattern:

1. claim artifacts are generated from raw imported/observed inputs;
2. certificate constructors are private or generated-only;
3. authored source is audited for conclusion fields and constructor use;
4. a generated claim is regenerated and compared canonically;
5. every check is falsified independently before it becomes a gate.

The current generated-claim guard and synthetic claim fixture establish useful
schema behavior. They do not authenticate the foreign source or runtime and
therefore do not establish a Tgrad compatibility result.

## 8. The claim prints its own scope

“Complete” must be unspeakable without indices. A claim renderer emits
something of this form, or emits blockers:

```text
Tgrad@<subject-tree>/<runtime> satisfies tinygrad@<target-tree>, profile <P>,
in environments {<E>}, with <n> exclusions and <m> tolerances,
under assurance policy <A> and search budget <N> at seeds <S>.
```

Exclusions, blocked hardware, tolerances, assurance grades, generator budgets,
and target age are inline. Widening a tolerance or narrowing a backend set
changes the claim identity and its sentence.

## 9. Profile completion reports a lattice, not a score

The useful output is a complete state vector and generated gaps:

```lean
structure ProfileCompletionCertificate where
  closure      : CatalogClosureCertificate
  required     : List RequirementId
  discharges   : List RequirementDischargeCertificate
  cycles       : List CycleCertificate
  -- derived:
  -- states    : Multiset RequirementState
  -- gaps      : List Gap
  -- complete  : Bool := gaps.isEmpty
```

No averaging: one drifted required row keeps the profile drifted. Progress is
still visible because the report prints counts by `open`, `blocked`,
`refuted`, `survivedSearch`, `acceptedBy`, and `proved`, plus the exact gaps.
A parity percentage may be a diagnostic projection; it is not the completion
predicate.

## 10. The scalable agent work unit

The unit is not “implement method X.” It is one interpretation contribution:

```text
nonempty upstream source-item set
<-> nonempty requirement-fragment set
-> executable denotation and partitions
-> upstream-derived generator and regression pins
-> mandatory dimensions, mutants, and adequacy challenges
-> observed current gap
-> dependency-shaped candidate packet
```

The contract contribution is admissible only if all of these are mechanical:

1. its denotation executes against pinned upstream without harness error;
2. upstream passes its own admitted scenarios;
3. every declared dimension has a mutant caught by that dimension alone;
4. every mandatory dimension for the requirement class is declared;
5. two runs produce the same observation identity;
6. the contribution touches no product or ontology code;
7. its source items are in the authenticated closure and its requirement
   references resolve;
8. removing the contribution creates a generated catalog gap.

An implementation candidate then has a separate acceptance transaction:

1. reproduce the frozen baseline on its exact base;
2. compute the actual Git effect set rather than trusting the packet;
3. execute the foreign and independent oracles;
4. rerun or cite identity-equal calibration attacks;
5. bind source tree, built runtime, adapter, relation, scenario, environment,
   verifier, and evidence identities;
6. promote exactly one candidate tree or retain a typed rejection.

This separation lets agents author in parallel while judgment remains serial.

## 11. Revised execution order

The dependency order after the source-closure audit is:

| order | packet | state and purpose |
|---:|---|---|
| 1 | assurance, completion schema, chronology schema | integrated substrate; no release claim |
| 2 | generated-claim guard | integrated synthetic regeneration guard |
| 3 | `contract.source-closure-v1` | integrated authenticated candidate closure |
| 4 | `contract.target-promotion-v1` | **next owner decision**; accepts target/closure only |
| 5 | `mechanics.synthetic-completion-v2` | rebuild the toy campaign on the authenticated promoted target; reject all declared mutants |
| 6 | authored/generated claim boundary guard | make conclusion-authoring mechanically impossible |
| 7 | contribution acceptance harness | enable parallel, independently judged interpretation work |
| 8 | disposition ledger and catalog closure | close the real denominator over authenticated items |
| 9 | executable trace AST and pilot imports | replace prose requirements with neutral denotations |
| 10 | scenario search and adequacy challenges | attack hard partitions independently of Tgrad support |
| 11 | runtime attestation and invalidation graph | bind source to dylib and make freshness causal |
| 12 | first real discharge and prospective agent transaction | prove the loop works end to end before scaling |
| 13 | full profile tracking and upstream advance | measure compatibility lag without ontology edits |

`target-promotion-v1` must not imply catalog closure. It says only: this is the
foreign revision and authenticated source boundary against which later
interpretation will be judged.

## 12. How we know the method is working

Track both assurance and economics:

| signal | working direction | stop/redesign signal |
|---|---|---|
| denominator integrity | 100% authenticated items have exactly one reviewed disposition | ordinary upstream additions require ontology edits or disappear silently |
| independent discharge | more required rows gain current evidence without edits to expected values | score moves mainly through shim, exclusion, or ruler changes |
| falsification yield | every new checker is shown red; discovered faults become permanent mutants | green checks have never rejected a realistic forgery |
| causal work prediction | frozen packets predict actual write sets and state transitions | repeated unexplained transitions or packet expansion after results |
| derivation stability | no judging-input change inside prospective cycles | observers or tolerances change to rescue product results |
| cycle economics | repeated requirement families get cheaper in wall time and human review | model upkeep costs more than reconstructing the answer manually |
| evolutionary tracking | a new upstream pin yields a deterministic diff, invalidations, and gaps | target advance requires hand-editing completion state |

The decisive falsifier remains a toy closed profile, but it must now sit on an
authenticated promoted target. The system should produce one valid bounded
claim and then reject mutations for an omitted source item, missing
disposition, stale subject tree, substituted relation, surviving validator
fault, blocked environment, target drift, hidden profile narrowing, mislabeled
bounded search, omitted cycle, mutation-and-reversion, and relocated judging
logic.

## 13. Honest limits

- The admitted Metal environment is currently narrow. Hardware not run is
  blocked, not implicitly supported.
- bf16 relations may require explicit numerical tolerances. A widened tolerance
  changes the profile identity.
- Adequacy can remain `survivedSearch` or `acceptedBy`; neither is a theorem.
- Search budgets are not exhaustiveness.
- Git authenticates object identity and ancestry under its hash/object model;
  it is not a foreign behavioral oracle and does not prove historical intent.
- CPython AST extraction is part of the trusted computing base. The 3.12,
  3.13, and 3.14 CI matrix checks cross-version stability; it does not prove a
  parser theorem.
- A promoted target is not a closed catalog. A closed catalog is not a
  discharged profile. A discharged profile at one target is not evolutionary
  completion.

Those limits are not reasons to weaken the method. They are indices the method
must force every human-facing claim to carry.

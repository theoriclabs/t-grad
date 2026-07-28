# Mechanising the completeness claim

**Status:** design, 2026-07-27. Responds to the four-predicate completeness
model. Assumes [`requirement_engineering.md`](requirement_engineering.md) and
[`growth_log_2026-07-27.md`](growth_log_2026-07-27.md).

The four predicates — catalog closure, requirement discharge, profile
completion, evolutionary completion — are the right decomposition and this
document does not relitigate them. It answers a narrower question: **what
makes each one impossible to forge**, given that the work will be done by
agents and must therefore be accepted mechanically rather than by review.

## 1. The premise: certificates are not proofs

A `structure` carrying `inventoryHash : Hash` and `adequacy :
AcceptedAdequacy` is a *record*. Lean can check it is internally consistent.
Lean cannot check that the hash was computed rather than typed, or that
`.accepted` was earned rather than asserted. Making `complete` derived rather
than a field does not fix this: it relocates the forgery from the conclusion
to the premises.

This is not hypothetical. Every defect found in this repository this week has
the same shape — a check whose two sides come from the same place:

| defect | the two sides |
|---|---|
| pilot `.stale` unreachable | `context.subjectTree` and `observation.subjectTree` both from `PilotGenerated` |
| umbrella gates vacuous | `[[ -f evidence.json ]]` where the file is committed to git |
| L12 perf sweep | comment asserted rigour the shell never applied |
| `gate_evidence` | 26/37 files certified a commit absent from the tree |

So the design rule for every certificate field is:

> **Adversarial distance.** A field is admissible only if the party that
> benefits from it being green is not the party that determines its value.

Three ways to obtain that distance, in descending order of strength:

1. **Recomputation from a foreign pinned source.** The value is a function of
   the upstream revision. Forging it requires forging upstream.
2. **Survived refutation.** The value records attacks that were executed and
   failed. Forging it requires the attacks to have run.
3. **Structural derivation with a scope-reducing cost.** A judgement is
   allowed, but it must narrow the printed claim.

A field that is none of these is decoration. Below, every proposed field is
labelled with which one it is.

## 2. Where the forgery surface actually is

Given what already exists — the broadcast-add observer declares 11 dimensions
including `realize_identity`, `repeated_readback`, `inputs_unchanged`, and
`exception_stage/class/message`, calibrated by 8 mutants — the per-requirement
observation machinery is mature. The exposure is elsewhere:

| # | forgery | addressed by the three proposed certificates? |
|---|---|---|
| 1 | inventory omits or over-excludes | yes — `CatalogClosureCertificate` |
| 2 | **scenarios dodge the hard cases** | **no** |
| 3 | fewer dimensions declared than the class requires | partly |
| 4 | `adequacy := .accepted` typed by an agent | no — relocated, not removed |
| 5 | environment set claimed broader than tested | yes |
| 6 | **ontology edited mid-cycle to make derivation agree** | **no** |

(2) and (6) are the two that matter most and neither is covered. (6) is not
speculative: `requirement_engineering.md` records that the first pilot cycle
"required a closure-logic repair, so derivation stability has not yet been
demonstrated."

## 3. Scenario closure is the weakest link

The likeliest way this model produces a false green is not a forged hash. It
is a requirement whose scenario set quietly omits the cases Tgrad cannot do.

A concrete instance exists today. `Tensor.realize()` returns `self`. The
observer declares `realize_identity` as a dimension and calibrates it with
`MUT-REALIZE-RETURNS-COPY`, so that dimension is honestly observed. But
upstream `realize()` *forces evaluation of an unrealized tensor*, and no
scenario covers that — because Tgrad has no laziness and therefore cannot
construct the input. The dimension is covered; the scenario space is not.

Under an exclusion mechanism that permits "out of scope: Tgrad is eager", this
becomes a permanent invisible hole. That is exactly backwards:

> **The admissibility rule.** A program admitted by the profile is admitted
> because *upstream* accepts it. If upstream expresses a program and Tgrad
> cannot, that is a **divergence**, recorded as failing or blocked. It is
> never an exclusion.

Mechanically: the scenario generator is driven by upstream and does not know
what Tgrad supports. This is the foreign-oracle law applied to scenario
*selection* rather than to expected values, and it is the single most
important correction in this document.

### Generated scenarios, not enumerated ones

Interactions cannot be enumerated — 297 tensor methods make the sequence space
infinite. So closure over scenarios cannot be a list. It is a **search
record**:

```text
generator G (grammar over admitted programs, pinned)
seed S, budget N
divergences found: k
```

"No divergence in N programs from G at seed S" is reproducible, falsifiable,
and honest. Authored scenarios remain as regression pins, but they are not the
closure argument — a pinned list of six cases cannot quantify over behaviour.

This is adversarial distance of type 2 (survived refutation).

## 4. Adequacy cannot be asserted

`D ∧ S ⊨ R` is semantic entailment over a real-world domain. It is not
mechanically provable, and any field an agent can set to `.accepted`
reintroduces `complete := true` one level down.

The mechanical substitute is refutation that failed:

```lean
/-- An adequacy challenge: a program the profile admits, which satisfies the
    boundary specification `S`. If it violates the world requirement `R`, then
    `S` was not adequate and the challenge refutes it. -/
structure AdequacyChallenge where
  requirement   : RequirementId
  dimension     : Dimension        -- which mandatory dimension it probes
  program       : ProgramRef       -- executable, not prose
  satisfiesSpec : Bool             -- observed, not asserted
  violatesReq   : Bool             -- observed, not asserted
```

Adequacy is then **derived**, never written:

```lean
def adequacyState (mandatory : List Dimension)
    (challenges : List AdequacyChallenge) : AdequacyState :=
  if challenges.isEmpty then .open
  else if challenges.any (·.violatesReq) then .refuted
  else if mandatory.all (fun d => challenges.any (·.dimension == d)) then .accepted
  else .open   -- a dimension nobody attacked is not evidence of adequacy
```

An agent may *add* challenges — that is useful work and it can only make
acceptance harder. It cannot assert acceptance. Note the final branch: an
uncovered mandatory dimension yields `.open`, not `.accepted`. Silence is not
a pass.

## 5. Prospectivity is a git-ancestry property

The tracking predicate — "the process repeats without redesigning the
ontology" — reads as a soft process claim. It is not. It is mechanically
checkable, and the check has already been run by hand against this
repository's real history.

Auditing the five broadcast-add cycles required exactly two operations:

```
git merge-base --is-ancestor <freeze> <implementation>   # for each cycle
git log --oneline -- <observer files>                    # none after first impl
```

All five cycles passed; all 12 observer commits preceded the first product
change. That ad-hoc audit should become the standing certificate:

```lean
structure CycleCertificate where
  cycle            : CycleId
  observationBefore : CommitRef    -- calibrated observation
  freeze           : CommitRef     -- work packet frozen
  implementation   : CommitRef     -- product change
  observationAfter : CommitRef     -- clean-tree re-observation
  frozenPaths      : List PathGlob -- observer + derivation + ontology
```

with validity **derived** from the repository, not recorded:

- `observationBefore` is an ancestor of `freeze`
- `freeze` is an ancestor of `implementation`
- `implementation` is an ancestor of `observationAfter`
- `git diff --name-only freeze..observationAfter ∩ frozenPaths = ∅`

The last conjunct is the one that matters. It makes the pilot's admitted
failure — generalising the closure logic after seeing the result — impossible
to hide, because the derivation code is inside `frozenPaths` and any commit
touching it during the cycle invalidates that cycle's prospectivity.

This is adversarial distance of type 1: git history is a foreign source
relative to the agent claiming the cycle was prospective.

## 6. No authored file may construct a certificate

The mechanism already exists and works: `PilotGenerated.lean` carries a
`GENERATED … do not hand edit` header, and `observe_pilot.py --check-generated`
fails if the file drifts from its JSON.

Extend it, do not invent something new:

1. Certificates may appear **only** in generated modules.
2. A repo check fails if a certificate constructor appears in an authored
   file — same family as `shell_continuation_audit.py` and
   `gate_evidence_not_tracked.py`.
3. Both run in devcheck cheap preflight.

That is the mechanical content of "no agent should set `complete := true`".

## 7. The claim prints its own scope

`Complete` must be unspeakable without its indices. The certificate's renderer
emits the sentence, and there is no other way to obtain the word:

```text
Tgrad@<tree> is complete for tinygrad@<rev>, profile <P>,
environments {<E>}, with <n> exclusions and <m> tolerances,
under search budget <N> at seed <S>.
```

Exclusions and tolerances are *printed inline*, so scope reduction cannot be
hidden. Widening a tolerance changes the sentence. This is adversarial
distance of type 3.

## 8. What the profile certificate reports

`ProfileComplete` as a Boolean is true-in-principle and useless-in-practice:
it will be false for years, so it carries no gradient and nobody will look at
it. Report the lattice and derive the Boolean, exactly as the pilot already
does with `states` + `gaps`:

```lean
structure ProfileCompletionCertificate where
  closure      : CatalogClosureCertificate
  required     : List RequirementId
  discharges   : List RequirementDischargeCertificate
  cycles       : List CycleCertificate     -- §5
  -- derived, never authored:
  --   states : Multiset RequirementState
  --   gaps   : List Gap
  --   complete : Bool   := gaps.isEmpty
```

No averaging, per the model: one drifted requirement drifts the profile. But
the *reported* artefact is the state multiset, so progress is visible without
the Boolean having to lie.

## 9. The agent work unit, and its acceptance test

This is the part that determines whether 590 items is tractable. If accepting
an agent's contribution requires human reading, review is the bottleneck and
the programme does not scale.

**Unit:** one inventory item → (requirement, denotation, generator, mutants,
challenges).

**Admissible iff all six hold, all mechanically:**

1. The denotation **executes against pinned upstream** without harness error.
2. The **upstream side passes** its own scenarios. If upstream fails, the
   requirement misreads upstream — this catches invented requirements.
3. Every declared dimension has ≥1 mutant caught by **that dimension alone**
   (independence — already the pilot's rule via `calibration_indeterminate`).
4. Every **mandatory dimension for the item's class** is declared.
5. **Determinism**: two runs produce an identical observation identity.
6. The contribution touches **no product code and no ontology code**
   (`git diff --name-only` ∩ those globs = ∅).

Gate 6 is the foreign-oracle law applied to authoring: an agent that can edit
the product while authoring its requirement can make the requirement match the
implementation. Gate 2 is what stops a requirement upstream does not have.

Gates 1–5 are already implemented in substance by `observe_broadcast_add.py`
for one requirement. The work is to generalise them into a harness that any
contribution is run through, not to invent them.

## 10. Sequencing

Cheapest and most-proven first. Each step is independently useful.

| # | step | why here |
|---|---|---|
| 1 | `CycleCertificate` + ancestry checker | check already proven by hand; pure git, no GPU, no upstream; locks the property the pilot admits it lacks |
| 2 | authored-vs-generated certificate guard | trivial; same family as two guards already shipped |
| 3 | contribution acceptance harness (§9) | generalises working code; unblocks parallel agent work |
| 4 | `CatalogClosureCertificate` over the 590 | mechanical shape, real labour, parallelisable once (3) exists |
| 5 | adversarial scenario generation (§3) | highest value, hardest; needs (3) |
| 6 | `ProfileCompletionCertificate` | derived; last by construction |

Steps 1 and 2 are pure infrastructure and depend on no open judgement. Step 4
depends on the profile's mandatory-dimension policy per item class. Step 5 is
the one that turns "we ran tests" into "we searched a space".

## 11. Honest limits

- **`E` is one machine.** Metal means the admitted environment set is `{this
  M4}` for the foreseeable future. That is a legitimate claim and a narrow
  one; §7 requires it to be printed rather than implied.
- **`≈P` is per-dimension with tolerances.** bf16 numerics cannot be
  bit-exact against a different backend. Tolerances belong to the profile,
  and widening one is a scope change that the rendered sentence must show.
- **Adequacy never becomes proof.** §4 buys survived refutation, not
  entailment. The model should not claim more.
- **Search budgets are not exhaustiveness.** §3 buys a reproducible search,
  not a quantifier. `complete` under a budget means "no divergence found at
  budget N", and the sentence says so.

None of these are reasons to weaken the model. They are reasons the rendered
claim must carry its indices.

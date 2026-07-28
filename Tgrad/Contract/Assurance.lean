import Tgrad.Contract.Identity

/-! # Tgrad.Contract.Assurance — graded assurance, provenance, and profile policy

This is the mechanical-completion assurance kernel (queue item
`mechanics.assurance-kernel-v1`), revised by `mechanics.completion-schema` so
identity carriers and profile policies are content-addressable.

It belongs to the `TgradSpec` root only. Product/runtime modules must not
import it.

The kernel deliberately separates:

* proof from bounded search from explicit judgment from unresolved states;
* claim-field provenance into exactly four auditable forms;
* profile acceptance into named predicates, not a total order over states.

An agent-authored Boolean conclusion is never stored as evidence. Policy
decisions are derived from an obligation's assurance state and the profile's
declared acceptance rule.
-/

namespace Tgrad.Contract

/-! ## Identity carriers for assurance references -/

structure BlockerRef where
  id : BlockerId
  kind : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure CounterexampleRef where
  id : CounterexampleId
  artifact : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Bound search evidence: generator, closure, seeds, budget and calibration
identity. Finite survival is not a proof. -/
structure SearchCertificateRef where
  id : SearchCertificateId
  generatorId : String
  sourceClosure : SourceClosureId
  seeds : ContentDigest
  budgetDesc : String
  partitions : ContentDigest
  divergenceCount : Nat
  calibration : CalibrationId
  deriving DecidableEq, BEq, Repr, Inhabited

structure ImportedSourceRef where
  id : ImportedSourceId
  sourceClosure : SourceClosureId
  deriving DecidableEq, BEq, Repr, Inhabited

structure DerivedComputationRef where
  id : DerivedComputationId
  verifier : ValidatorId
  inputClosure : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def BlockerRef.wellFormed (r : BlockerRef) : Bool :=
  r.id.wellFormed && !r.kind.trimAscii.isEmpty

def CounterexampleRef.wellFormed (r : CounterexampleRef) : Bool :=
  r.id.wellFormed && r.artifact.wellFormed

/-- Structural validity of a search-run record. A divergent run remains valid
evidence; it simply does not establish survival. -/
def SearchCertificateRef.wellFormed (r : SearchCertificateRef) : Bool :=
  r.id.wellFormed &&
  !r.generatorId.trimAscii.isEmpty &&
  r.sourceClosure.wellFormed &&
  r.seeds.wellFormed &&
  !r.budgetDesc.trimAscii.isEmpty &&
  r.partitions.wellFormed &&
  r.calibration.wellFormed

/-- Survival eligibility: structurally valid and zero recorded divergences. -/
def SearchCertificateRef.supportsSurvival (r : SearchCertificateRef) : Bool :=
  r.wellFormed && r.divergenceCount == 0

/-- Alias emphasizing the zero-divergence conjunct of survival eligibility. -/
def SearchCertificateRef.establishesNoDivergence (r : SearchCertificateRef) : Bool :=
  r.supportsSurvival

def ImportedSourceRef.wellFormed (r : ImportedSourceRef) : Bool :=
  r.id.wellFormed && r.sourceClosure.wellFormed

def DerivedComputationRef.wellFormed (r : DerivedComputationRef) : Bool :=
  r.id.wellFormed && r.verifier.wellFormed && r.inputClosure.wellFormed

/-! ## Assurance states

No unqualified `accepted : Bool`. Each non-open constructor carries a typed
reference to the fact that justifies that grade.
-/

inductive AssuranceState where
  | open
  | blocked (reference : BlockerRef)
  | refuted (counterexample : CounterexampleRef)
  | survivedSearch (certificate : SearchCertificateRef)
  | acceptedBy (judgment : JudgmentIdentity)
  | proved (proof : ProofIdentity)
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Constructor tag only. Used by acceptance predicates so policies name
admitted constructors without ranking blocked, refuted, judgment, search and
proof against each other. -/
inductive AssuranceConstructor where
  | open
  | blocked
  | refuted
  | survivedSearch
  | acceptedBy
  | proved
  deriving DecidableEq, BEq, Repr, Inhabited

def AssuranceState.constructor : AssuranceState → AssuranceConstructor
  | .open => .open
  | .blocked _ => .blocked
  | .refuted _ => .refuted
  | .survivedSearch _ => .survivedSearch
  | .acceptedBy _ => .acceptedBy
  | .proved _ => .proved

def AssuranceState.wellFormed : AssuranceState → Bool
  | .open => true
  | .blocked r => r.wellFormed
  | .refuted r => r.wellFormed
  | .survivedSearch r => r.supportsSurvival
  | .acceptedBy r => r.wellFormed
  | .proved r => r.wellFormed

/-! ## Claim-field provenance

Every claim-bearing field must be one of these four forms. There is no
unclassified constructor.
-/

inductive FieldProvenance where
  | imported (source : ImportedSourceRef)
  | derived (computation : DerivedComputationRef)
  | calibrated (campaign : CalibrationIdentity)
  | judgment (judgment : JudgmentIdentity)
  deriving DecidableEq, BEq, Repr, Inhabited

def FieldProvenance.wellFormed : FieldProvenance → Bool
  | .imported s => s.wellFormed
  | .derived c => c.wellFormed
  | .calibrated c => c.wellFormed
  | .judgment j => j.wellFormed

/-- Structural classifier: true exactly on the four explicit constructors. -/
def FieldProvenance.isClassified : FieldProvenance → Bool
  | .imported _ => true
  | .derived _ => true
  | .calibrated _ => true
  | .judgment _ => true

structure ClaimField where
  name : String
  provenance : FieldProvenance
  deriving DecidableEq, BEq, Repr, Inhabited

def ClaimField.wellFormed (field : ClaimField) : Bool :=
  !field.name.trimAscii.isEmpty && field.provenance.wellFormed

/-- Admissible only when classified *and* the embedded provenance reference is
well-formed. Constructor tags alone are not enough. -/
def ClaimField.hasAdmissibleProvenance (field : ClaimField) : Bool :=
  field.provenance.isClassified && field.wellFormed

/-! ## Obligation classes and versioned profile assurance policy

Acceptance rules are named predicates over constructor tags. A profile may
admit proof for one class and bounded search for another; that does not define
a lattice where search is “below” proof or judgment is “below” search.

Policies are a finite, total, nodup association list so they admit content
identity. There is no function field.
-/

inductive ObligationClass where
  | adequacy
  | catalogClosure
  | requirementDischarge
  | scenarioObservation
  | performanceQualification
  deriving DecidableEq, BEq, Repr, Inhabited

def allObligationClasses : List ObligationClass :=
  [.adequacy, .catalogClosure, .requirementDischarge, .scenarioObservation,
   .performanceQualification]

def ObligationClass.tag : ObligationClass → String
  | .adequacy => "adequacy"
  | .catalogClosure => "catalogClosure"
  | .requirementDischarge => "requirementDischarge"
  | .scenarioObservation => "scenarioObservation"
  | .performanceQualification => "performanceQualification"

def ObligationClass.isSemanticCompatibility : ObligationClass → Bool
  | .adequacy => true
  | .requirementDischarge => true
  | .scenarioObservation => true
  | .catalogClosure => false
  | .performanceQualification => false

/-- Explicit admission sets. Each constructor lists which assurance grades it
accepts; omitted constructors fail the rule. -/
inductive AcceptanceRule where
  /-- Only `proved` meets the rule. -/
  | requireProof
  /-- Only `survivedSearch` meets the rule. Proof is not implied. -/
  | acceptBoundedSearch
  /-- Only `acceptedBy` meets the rule. Proof is not implied. -/
  | acceptJudgment
  /-- Release-facing rule: proof, bounded search, or judgment. Open, blocked
  and refuted are excluded by omission, not by ranking. -/
  | releaseEligible
  deriving DecidableEq, BEq, Repr, Inhabited

def AcceptanceRule.tag : AcceptanceRule → String
  | .requireProof => "requireProof"
  | .acceptBoundedSearch => "acceptBoundedSearch"
  | .acceptJudgment => "acceptJudgment"
  | .releaseEligible => "releaseEligible"

def AcceptanceRule.admitsConstructor
    (rule : AcceptanceRule) (ctor : AssuranceConstructor) : Bool :=
  match rule with
  | .requireProof => ctor == .proved
  | .acceptBoundedSearch => ctor == .survivedSearch
  | .acceptJudgment => ctor == .acceptedBy
  | .releaseEligible =>
      ctor == .proved || ctor == .survivedSearch || ctor == .acceptedBy

def AcceptanceRule.admits (rule : AcceptanceRule) (state : AssuranceState) : Bool :=
  state.wellFormed && rule.admitsConstructor state.constructor

structure ObligationRule where
  obligationClass : ObligationClass
  rule : AcceptanceRule
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProfileAssurancePolicy where
  profileId : ProfileId
  version : Nat
  rules : List ObligationRule
  deriving DecidableEq, BEq, Repr, Inhabited

def ProfileAssurancePolicy.classes (policy : ProfileAssurancePolicy) : List ObligationClass :=
  policy.rules.map (·.obligationClass)

def ProfileAssurancePolicy.wellFormed (policy : ProfileAssurancePolicy) : Bool :=
  policy.profileId.wellFormed &&
  policy.version > 0 &&
  listNodup policy.classes &&
  listSetEq policy.classes allObligationClasses

def ProfileAssurancePolicy.ruleFor?
    (policy : ProfileAssurancePolicy) (cls : ObligationClass) : Option AcceptanceRule :=
  (policy.rules.find? (fun entry => entry.obligationClass == cls)).map (·.rule)

/-- Total lookup; well-formed policies always hit. Ill-formed policies yield none. -/
def ProfileAssurancePolicy.ruleFor
    (policy : ProfileAssurancePolicy) (cls : ObligationClass) : Option AcceptanceRule :=
  if policy.wellFormed then policy.ruleFor? cls else none

/-- Canonical content identity: profile, version, and sorted class→rule pairs. -/
def ProfileAssurancePolicy.contentId (policy : ProfileAssurancePolicy) : ContentDigest :=
  let sorted :=
    policy.rules.mergeSort (fun a b => a.obligationClass.tag ≤ b.obligationClass.tag)
  let body :=
    String.intercalate ";"
      (sorted.map fun entry => s!"{entry.obligationClass.tag}={entry.rule.tag}")
  digest s!"{policy.profileId.value}|v{policy.version}|{body}"

def uniformPolicy (profileId : ProfileId) (version : Nat) (rule : AcceptanceRule) :
    ProfileAssurancePolicy :=
  { profileId := profileId,
    version := version,
    rules := allObligationClasses.map fun cls =>
      { obligationClass := cls, rule := rule } }

/-- Derived policy decision. The verdict is computed from the retained policy,
class and state; it is not an independently authored evidence field. -/
inductive PolicyVerdict where
  | meets
  | fails
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Retains the actual policy inputs. Verdict is always recomputed from them;
there is no separately authorable rule or conclusion field. -/
structure PolicyDecision where
  policy : ProfileAssurancePolicy
  obligationClass : ObligationClass
  state : AssuranceState
  deriving DecidableEq, BEq, Repr, Inhabited

def PolicyDecision.rule? (decision : PolicyDecision) : Option AcceptanceRule :=
  decision.policy.ruleFor decision.obligationClass

def PolicyDecision.verdict (decision : PolicyDecision) : PolicyVerdict :=
  match decision.rule? with
  | some rule =>
      if decision.policy.wellFormed && rule.admits decision.state then .meets else .fails
  | none => .fails

def PolicyDecision.meets (decision : PolicyDecision) : Bool :=
  decision.verdict == .meets

def ProfileAssurancePolicy.decide
    (policy : ProfileAssurancePolicy)
    (obligationClass : ObligationClass)
    (state : AssuranceState) : PolicyDecision :=
  { policy := policy, obligationClass := obligationClass, state := state }

def ProfileAssurancePolicy.meets
    (policy : ProfileAssurancePolicy)
    (obligationClass : ObligationClass)
    (state : AssuranceState) : Bool :=
  (policy.decide obligationClass state).meets

/-! ## Kernel fixtures used by the executable checks -/

def sampleProof : ProofIdentity :=
  { id := { value := "proof.adeq.demo" }, theoremName := "D_and_S_entails_R" }

def sampleSearch : SearchCertificateRef :=
  { id := { value := "search.scenario.demo" },
    generatorId := "gen.scenario.v1",
    sourceClosure := { digest := digest "sc-hash" },
    seeds := digest "seeds-hash",
    budgetDesc := "seeds=8,cases<=256",
    partitions := digest "part-hash",
    divergenceCount := 0,
    calibration := { value := "cal.omit-ops-mixin" } }

def sampleJudgment : JudgmentIdentity :=
  { id := { value := "judgment.scope.demo" },
    authority := "owner",
    scope := digest "scope-hash",
    invalidation := digest "inv-hash" }

def sampleBlocker : BlockerRef :=
  { id := { value := "blocker.env.metal" }, kind := "environment" }

def sampleCounterexample : CounterexampleRef :=
  { id := { value := "cx.mutant.survived" }, artifact := digest "cx-hash" }

def sampleImported : ImportedSourceRef :=
  { id := { value := "import.upstream.closure" },
    sourceClosure := { digest := digest "src-hash" } }

def sampleDerived : DerivedComputationRef :=
  { id := { value := "derived.status" },
    verifier := { value := "verifier.pilot-status" },
    inputClosure := digest "in-hash" }

def sampleCalibration : CalibrationIdentity :=
  { id := { value := "cal.helpers" },
    campaign := digest "cal-hash",
    faultModel := "missing-public-name" }

/-- Proof-required profile: adequacy demands a proof reference. -/
def proofRequiredPolicy : ProfileAssurancePolicy :=
  uniformPolicy { value := "profile.proof-required.demo" } 1 .requireProof

/-- Bounded-search profile: scenario observation admits search certificates. -/
def boundedSearchPolicy : ProfileAssurancePolicy where
  profileId := { value := "profile.bounded-search.demo" }
  version := 1
  rules :=
    allObligationClasses.map fun cls =>
      { obligationClass := cls,
        rule :=
          if cls == .scenarioObservation then .acceptBoundedSearch
          else .requireProof }

/-- Release-eligible profile: positive grades only; unresolved states fail. -/
def releaseEligiblePolicy : ProfileAssurancePolicy :=
  uniformPolicy { value := "profile.release-eligible.demo" } 1 .releaseEligible

def claimFieldsDemo : List ClaimField :=
  [ { name := "targetRevision", provenance := .imported sampleImported },
    { name := "statusVector", provenance := .derived sampleDerived },
    { name := "mutantSensitivity", provenance := .calibrated sampleCalibration },
    { name := "profileScope", provenance := .judgment sampleJudgment } ]

/-! ## Executable semantic checks -/

theorem proof_satisfies_proof_required_policy :
    proofRequiredPolicy.meets .adequacy (.proved sampleProof) = true := by
  native_decide

theorem survivedSearch_does_not_satisfy_proof_required :
    proofRequiredPolicy.meets .adequacy (.survivedSearch sampleSearch) = false := by
  native_decide

theorem acceptedBy_does_not_satisfy_proof_required :
    proofRequiredPolicy.meets .adequacy (.acceptedBy sampleJudgment) = false := by
  native_decide

theorem survivedSearch_satisfies_bounded_search_policy :
    boundedSearchPolicy.meets .scenarioObservation (.survivedSearch sampleSearch) = true := by
  native_decide

theorem open_does_not_satisfy_release_eligible :
    releaseEligiblePolicy.meets .requirementDischarge .open = false := by
  native_decide

theorem blocked_does_not_satisfy_release_eligible :
    releaseEligiblePolicy.meets .requirementDischarge (.blocked sampleBlocker) = false := by
  native_decide

theorem refuted_does_not_satisfy_release_eligible :
    releaseEligiblePolicy.meets .requirementDischarge (.refuted sampleCounterexample) =
      false := by
  native_decide

theorem every_field_provenance_is_classified (p : FieldProvenance) :
    p.isClassified = true := by
  cases p <;> rfl

theorem demo_claim_fields_all_admissible :
    claimFieldsDemo.all ClaimField.hasAdmissibleProvenance = true := by
  native_decide

theorem demo_claim_fields_cover_all_four_provenance_forms :
    (claimFieldsDemo.any fun f => match f.provenance with | .imported _ => true | _ => false) &&
    (claimFieldsDemo.any fun f => match f.provenance with | .derived _ => true | _ => false) &&
    (claimFieldsDemo.any fun f => match f.provenance with | .calibrated _ => true | _ => false) &&
    (claimFieldsDemo.any fun f => match f.provenance with | .judgment _ => true | _ => false) =
      true := by
  native_decide

/-- Policy verdict is a function of retained policy/class/state, not a free Boolean. -/
example :
    (proofRequiredPolicy.decide .adequacy (.proved sampleProof)).verdict = .meets ∧
    (proofRequiredPolicy.decide .adequacy (.survivedSearch sampleSearch)).verdict = .fails := by
  native_decide

theorem proof_required_policy_is_content_identified :
    proofRequiredPolicy.wellFormed = true ∧
    proofRequiredPolicy.contentId =
      digest "profile.proof-required.demo|v1|adequacy=requireProof;catalogClosure=requireProof;performanceQualification=requireProof;requirementDischarge=requireProof;scenarioObservation=requireProof" := by
  native_decide

/-! ## Negative checks: malformed references cannot meet policy -/

def emptyProof : ProofIdentity := default

def divergentSearch : SearchCertificateRef :=
  { sampleSearch with divergenceCount := 1 }

def malformedImportedField : ClaimField :=
  { name := "targetRevision", provenance := .imported default }

def zeroVersionPolicy : ProfileAssurancePolicy :=
  uniformPolicy { value := "profile.zero-version.demo" } 0 .requireProof

theorem default_proof_does_not_satisfy_proof_required :
    proofRequiredPolicy.meets .adequacy (.proved emptyProof) = false := by
  native_decide

theorem divergent_search_is_structurally_well_formed :
    divergentSearch.wellFormed = true := by
  native_decide

theorem divergent_search_does_not_support_survival :
    divergentSearch.supportsSurvival = false := by
  native_decide

theorem divergent_survivedSearch_state_not_well_formed :
    AssuranceState.wellFormed (.survivedSearch divergentSearch) = false := by
  native_decide

theorem admits_rejects_malformed_and_divergent_states :
    (AcceptanceRule.requireProof.admits (.proved emptyProof) = false) &&
    (AcceptanceRule.acceptBoundedSearch.admits (.survivedSearch divergentSearch) =
      false) = true := by
  native_decide

theorem divergent_search_does_not_satisfy_bounded_search :
    boundedSearchPolicy.meets .scenarioObservation (.survivedSearch divergentSearch) =
      false := by
  native_decide

theorem malformed_provenance_is_not_admissible :
    ClaimField.hasAdmissibleProvenance malformedImportedField = false := by
  native_decide

theorem zero_version_policy_does_not_meet :
    zeroVersionPolicy.meets .adequacy (.proved sampleProof) = false := by
  native_decide

end Tgrad.Contract

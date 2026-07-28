/-! # Tgrad.Contract.Assurance — graded assurance, provenance, and profile policy

This is the mechanical-completion assurance kernel (queue item
`mechanics.assurance-kernel-v1`). It belongs to the `TgradSpec` root only.
Product/runtime modules must not import it.

The kernel deliberately separates:

* proof from bounded search from explicit judgment from unresolved states;
* claim-field provenance into exactly four auditable forms;
* profile acceptance into named predicates, not a total order over states.

An agent-authored Boolean conclusion is never stored as evidence. Policy
decisions are derived from an obligation's assurance state and the profile's
declared acceptance rule.
-/

namespace Tgrad.Contract

/-! ## Identity carriers for assurance references

Full contract identities land in later M0 modules. These carriers are enough
for the kernel to require typed references instead of bare Booleans.
-/

structure BlockerRef where
  id : String
  kind : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure CounterexampleRef where
  id : String
  artifactHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Bound search evidence: generator, closure, seeds, budget and calibration
identity. Finite survival is not a proof. -/
structure SearchCertificateRef where
  id : String
  generatorId : String
  sourceClosureHash : String
  seedsHash : String
  budgetDesc : String
  partitionHash : String
  divergenceCount : Nat
  calibrationId : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure JudgmentRef where
  id : String
  authority : String
  scopeHash : String
  invalidationHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProofRef where
  id : String
  theoremName : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ImportedSourceRef where
  id : String
  sourceClosureHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure DerivedComputationRef where
  id : String
  verifierId : String
  inputClosureHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure CalibrationCampaignRef where
  id : String
  faultModel : String
  campaignHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

def BlockerRef.wellFormed (r : BlockerRef) : Bool :=
  !r.id.trimAscii.isEmpty && !r.kind.trimAscii.isEmpty

def CounterexampleRef.wellFormed (r : CounterexampleRef) : Bool :=
  !r.id.trimAscii.isEmpty && !r.artifactHash.trimAscii.isEmpty

/-- Structural validity of a search-run record. A divergent run remains valid
evidence; it simply does not establish survival. -/
def SearchCertificateRef.wellFormed (r : SearchCertificateRef) : Bool :=
  !r.id.trimAscii.isEmpty &&
  !r.generatorId.trimAscii.isEmpty &&
  !r.sourceClosureHash.trimAscii.isEmpty &&
  !r.seedsHash.trimAscii.isEmpty &&
  !r.budgetDesc.trimAscii.isEmpty &&
  !r.partitionHash.trimAscii.isEmpty &&
  !r.calibrationId.trimAscii.isEmpty

/-- Survival eligibility: structurally valid and zero recorded divergences. -/
def SearchCertificateRef.supportsSurvival (r : SearchCertificateRef) : Bool :=
  r.wellFormed && r.divergenceCount == 0

/-- Alias emphasizing the zero-divergence conjunct of survival eligibility. -/
def SearchCertificateRef.establishesNoDivergence (r : SearchCertificateRef) : Bool :=
  r.supportsSurvival

def JudgmentRef.wellFormed (r : JudgmentRef) : Bool :=
  !r.id.trimAscii.isEmpty &&
  !r.authority.trimAscii.isEmpty &&
  !r.scopeHash.trimAscii.isEmpty &&
  !r.invalidationHash.trimAscii.isEmpty

def ProofRef.wellFormed (r : ProofRef) : Bool :=
  !r.id.trimAscii.isEmpty && !r.theoremName.trimAscii.isEmpty

def ImportedSourceRef.wellFormed (r : ImportedSourceRef) : Bool :=
  !r.id.trimAscii.isEmpty && !r.sourceClosureHash.trimAscii.isEmpty

def DerivedComputationRef.wellFormed (r : DerivedComputationRef) : Bool :=
  !r.id.trimAscii.isEmpty &&
  !r.verifierId.trimAscii.isEmpty &&
  !r.inputClosureHash.trimAscii.isEmpty

def CalibrationCampaignRef.wellFormed (r : CalibrationCampaignRef) : Bool :=
  !r.id.trimAscii.isEmpty &&
  !r.faultModel.trimAscii.isEmpty &&
  !r.campaignHash.trimAscii.isEmpty

/-! ## Assurance states

No unqualified `accepted : Bool`. Each non-open constructor carries a typed
reference to the fact that justifies that grade.
-/

inductive AssuranceState where
  | open
  | blocked (reference : BlockerRef)
  | refuted (counterexample : CounterexampleRef)
  | survivedSearch (certificate : SearchCertificateRef)
  | acceptedBy (judgment : JudgmentRef)
  | proved (proof : ProofRef)
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
  | calibrated (campaign : CalibrationCampaignRef)
  | judgment (judgment : JudgmentRef)
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
-/

inductive ObligationClass where
  | adequacy
  | catalogClosure
  | requirementDischarge
  | scenarioObservation
  | performanceQualification
  deriving DecidableEq, BEq, Repr, Inhabited

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

structure ProfileAssurancePolicy where
  profileId : String
  version : Nat
  ruleFor : ObligationClass → AcceptanceRule
  deriving Inhabited

def ProfileAssurancePolicy.wellFormed (policy : ProfileAssurancePolicy) : Bool :=
  !policy.profileId.trimAscii.isEmpty && policy.version > 0

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
  deriving Inhabited

def PolicyDecision.rule (decision : PolicyDecision) : AcceptanceRule :=
  decision.policy.ruleFor decision.obligationClass

def PolicyDecision.verdict (decision : PolicyDecision) : PolicyVerdict :=
  if decision.policy.wellFormed && decision.rule.admits decision.state then
    .meets
  else
    .fails

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

def sampleProof : ProofRef :=
  { id := "proof.adeq.demo", theoremName := "D_and_S_entails_R" }

def sampleSearch : SearchCertificateRef :=
  { id := "search.scenario.demo",
    generatorId := "gen.scenario.v1",
    sourceClosureHash := "sc-hash",
    seedsHash := "seeds-hash",
    budgetDesc := "seeds=8,cases<=256",
    partitionHash := "part-hash",
    divergenceCount := 0,
    calibrationId := "cal.omit-ops-mixin" }

def sampleJudgment : JudgmentRef :=
  { id := "judgment.scope.demo",
    authority := "owner",
    scopeHash := "scope-hash",
    invalidationHash := "inv-hash" }

def sampleBlocker : BlockerRef :=
  { id := "blocker.env.metal", kind := "environment" }

def sampleCounterexample : CounterexampleRef :=
  { id := "cx.mutant.survived", artifactHash := "cx-hash" }

def sampleImported : ImportedSourceRef :=
  { id := "import.upstream.closure", sourceClosureHash := "src-hash" }

def sampleDerived : DerivedComputationRef :=
  { id := "derived.status",
    verifierId := "verifier.pilot-status",
    inputClosureHash := "in-hash" }

def sampleCalibration : CalibrationCampaignRef :=
  { id := "cal.helpers",
    faultModel := "missing-public-name",
    campaignHash := "cal-hash" }

/-- Proof-required profile: adequacy demands a proof reference. -/
def proofRequiredPolicy : ProfileAssurancePolicy where
  profileId := "profile.proof-required.demo"
  version := 1
  ruleFor := fun
    | .adequacy => .requireProof
    | .catalogClosure => .requireProof
    | .requirementDischarge => .requireProof
    | .scenarioObservation => .requireProof
    | .performanceQualification => .requireProof

/-- Bounded-search profile: scenario observation admits search certificates. -/
def boundedSearchPolicy : ProfileAssurancePolicy where
  profileId := "profile.bounded-search.demo"
  version := 1
  ruleFor := fun
    | .scenarioObservation => .acceptBoundedSearch
    | .adequacy => .requireProof
    | .catalogClosure => .requireProof
    | .requirementDischarge => .requireProof
    | .performanceQualification => .requireProof

/-- Release-eligible profile: positive grades only; unresolved states fail. -/
def releaseEligiblePolicy : ProfileAssurancePolicy where
  profileId := "profile.release-eligible.demo"
  version := 1
  ruleFor := fun _ => .releaseEligible

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

/-! ## Negative checks: malformed references cannot meet policy -/

def emptyProof : ProofRef := default

def divergentSearch : SearchCertificateRef :=
  { sampleSearch with divergenceCount := 1 }

def malformedImportedField : ClaimField :=
  { name := "targetRevision", provenance := .imported default }

def zeroVersionPolicy : ProfileAssurancePolicy where
  profileId := "profile.zero-version.demo"
  version := 0
  ruleFor := fun _ => .requireProof

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

import Tgrad.Spec.RuntimeWork
import Tgrad.Spec.Findings

/-! # Tgrad.Spec.Growth — the bridge from execution to evolution

Runtime capabilities do work and produce observations. Observations upgrade or
create findings. Evolution work changes the repository. Verification and
specification capabilities then judge whether that change may be promoted.

This module types the bridge; it intentionally does not encode the reasoning
policy that diagnoses observations or invents repairs. Better agents and more
compute should improve those judgments without requiring a new substrate.
-/

namespace Tgrad.Spec.Growth

structure EvolutionWorkId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

def evolutionWorkId (value : String) : EvolutionWorkId := { value }

instance : ToString EvolutionWorkId where
  toString id := id.value

inductive ChangeKind where
  | addCapability
  | widenDomain
  | repairSemantics
  | strengthenSafety
  | replaceBypass
  | improveObservability
  | improveEfficiency
  | deleteScaffolding
  | upgradeEvidence
  deriving DecidableEq, BEq, Repr, Inhabited

inductive DeltaAspect where
  | supportedDomain
  | semantics
  | safety
  | architecture
  | performance
  | observability
  | provenance
  | maintainability
  deriving DecidableEq, BEq, Repr, Inhabited

structure ObservationSpec where
  runtimeWork : Runtime.WorkId
  scenario : String
  expected : String
  probe : String
  evidenceProduced : String
  deriving Repr, Inhabited

structure CapabilityDelta where
  runtimeWork : Runtime.WorkId
  kind : ChangeKind
  aspects : List DeltaAspect
  before : Runtime.CapabilityState
  after : Runtime.CapabilityState
  rationale : String
  deriving Repr, Inhabited

inductive Stage where
  | discovered
  | selected
  | implementing
  | validated
  | promoted
  | deferred
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Independent reasons a candidate may have to be checked. These are a set
of obligations, not a strength ladder: a fast wrong kernel does not satisfy a
numerical obligation, and a correct kernel does not establish provenance. -/
inductive ObligationKind where
  | build
  | unitRegression
  | apiContract
  | safety
  | numerical
  | semantic
  | differential
  | performance
  | provenance
  | resourceIsolation
  | humanReview
  deriving DecidableEq, BEq, Repr, Inhabited

structure Promotion where
  validators : List Runtime.WorkId
  obligations : List ObligationKind
  acceptance : String
  rollback : String
  deriving Repr, Inhabited

structure Case where
  id : String
  findingIds : List String
  observations : List ObservationSpec
  evolutionWork : List EvolutionWorkId
  deltas : List CapabilityDelta
  stage : Stage
  promotion : Promotion
  epistemic : Epistemic String
  deriving Repr, Inhabited

def Case.referencesKnownRuntimeWork (growthCase : Case) : Bool :=
  let observationIds := growthCase.observations.map (·.runtimeWork)
  let deltaIds := growthCase.deltas.map (·.runtimeWork)
  let validatorIds := growthCase.promotion.validators
  (observationIds ++ deltaIds ++ validatorIds).all Runtime.workIds.contains

def Case.referencesKnownFindings
    (knownFindingIds : List String) (growthCase : Case) : Bool :=
  growthCase.findingIds.all knownFindingIds.contains

def Case.referencesKnownEvolutionWork
    (knownEvolutionIds : List EvolutionWorkId) (growthCase : Case) : Bool :=
  growthCase.evolutionWork.all knownEvolutionIds.contains

def Case.hasClosedLoopShape (growthCase : Case) : Bool :=
  !growthCase.findingIds.isEmpty &&
  !growthCase.observations.isEmpty &&
  !growthCase.evolutionWork.isEmpty &&
  !growthCase.deltas.isEmpty &&
  !growthCase.promotion.validators.isEmpty &&
  !growthCase.promotion.obligations.isEmpty &&
  !growthCase.promotion.acceptance.isEmpty &&
  !growthCase.promotion.rollback.isEmpty &&
  growthCase.epistemic.hasUpgradePath

def validatorIsReflexiveWork (id : Runtime.WorkId) : Bool :=
  match Runtime.workUnitFor? id with
  | some unit => unit.realm == .verification || unit.realm == .specification
  | none => false

def Case.validatorsAreReflexiveWork (growthCase : Case) : Bool :=
  growthCase.promotion.validators.all validatorIsReflexiveWork

def Case.promotedEvolutionIsComplete
    (completedEvolutionIds : List EvolutionWorkId) (growthCase : Case) : Bool :=
  if growthCase.stage == .promoted then
    growthCase.evolutionWork.all completedEvolutionIds.contains
  else true

def Case.deltasAreSpecific (growthCase : Case) : Bool :=
  growthCase.deltas.all (fun delta =>
    !delta.aspects.isEmpty && !delta.rationale.isEmpty)

def knownFindingIds : List String := findings.map (·.id)

def casesWellFormed
    (knownEvolutionIds completedEvolutionIds : List EvolutionWorkId)
    (cases : List Case) : Bool :=
  let caseIds := cases.map (·.id)
  caseIds.eraseDups.length == caseIds.length &&
  cases.all (fun growthCase =>
    growthCase.referencesKnownRuntimeWork &&
    growthCase.referencesKnownFindings knownFindingIds &&
    growthCase.referencesKnownEvolutionWork knownEvolutionIds &&
    growthCase.hasClosedLoopShape &&
    growthCase.validatorsAreReflexiveWork &&
    growthCase.promotedEvolutionIsComplete completedEvolutionIds &&
    growthCase.deltasAreSpecific)

def openFindingsCovered (cases : List Case) : Bool :=
  let covered := (cases.flatMap (·.findingIds)).eraseDups
  (openFindings.map (·.id)).all covered.contains

/-- A growth case is a candidate queue, not an encoded reasoning policy. The
agent chooses among these typed candidates using current observations. -/
def candidateEvolutionWork (growthCase : Case) : List EvolutionWorkId :=
  growthCase.evolutionWork

structure Metrics where
  runtimeUnits : Nat
  loadBearingUnits : Nat
  boundedUnits : Nat
  missingOrBypassedUnits : Nat
  growthCases : Nat
  promotedCases : Nat
  openFindingsCovered : Nat
  openFindingsTotal : Nat
  deriving Repr, Inhabited

def metrics (cases : List Case) : Metrics :=
  { runtimeUnits := Runtime.workUnits.length,
    loadBearingUnits := (Runtime.unitsInState .loadBearing).length,
    boundedUnits := (Runtime.unitsInState .bounded).length,
    missingOrBypassedUnits := Runtime.missingOrBypassed.length,
    growthCases := cases.length,
    promotedCases := (cases.filter (fun c => c.stage == .promoted)).length,
    openFindingsCovered := (openFindings.filter (fun finding =>
      cases.any (fun c => c.findingIds.contains finding.id))).length,
    openFindingsTotal := openFindings.length }

end Tgrad.Spec.Growth

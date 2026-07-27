import Tgrad.Evidence.Observations
import Tgrad.Specification.Pilot

/-! # Tgrad.Growth.Derived — state and work derived from obligations

The axes remain separate so a prerequisite failure cannot masquerade as an
operation failure.  No field below is an authored "percent complete" or
`goalDistance`.
-/

namespace Tgrad.Growth

open Tgrad.Requirements
open Tgrad.Specification
open Tgrad.Spec
open Tgrad.Evidence
open Tgrad.Conformance

inductive InventoryState where
  | catalogued
  | interpreted
  deriving DecidableEq, BEq, Repr, Inhabited

inductive SpecificationState where
  | missing
  | structurallySpecified
  deriving DecidableEq, BEq, Repr, Inhabited

inductive AdequacyState where
  | open
  | tentative
  | accepted
  | refuted
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ImplementationState where
  | noCandidate
  | candidateMapped
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ObservationState where
  | unobserved
  | stale
  | blocked
  | failed
  | verifierError
  | passedUncalibrated
  | passedCalibrated
  deriving DecidableEq, BEq, Repr, Inhabited

inductive PromotionState where
  | targetUnpromoted
  | adequacyOpen
  | observationInsufficient
  | conformant
  deriving DecidableEq, BEq, Repr, Inhabited

structure RequirementState where
  requirement : RequirementId
  inventory : InventoryState
  specification : SpecificationState
  adequacy : AdequacyState
  implementation : ImplementationState
  observation : ObservationState
  promotion : PromotionState
  deriving DecidableEq, BEq, Repr, Inhabited

def deriveAdequacy (claim : AdequacyClaim) : AdequacyState :=
  match claim.result with
  | .confirmed .refuted _ => .refuted
  | .confirmed _ _ => .accepted
  | .tentative _ _ _ => .tentative
  | .unknown _ _ => .open
  | .deferred _ _ => .open

def deriveImplementation
    (requirement : RequirementId)
    (candidates : List CandidateMapping) : ImplementationState :=
  if candidates.any (fun candidate => candidate.requirement == requirement)
  then .candidateMapped
  else .noCandidate

def deriveObservation
    (context : PromotionContext) (requirement : Requirement)
    (validators : List ValidatorRef) (observations : List Observation) :
    ObservationState :=
  let relevant := observations.filter (fun observation =>
    observation.requirement == requirement.id)
  if relevant.isEmpty then .unobserved
  else
    let current := relevant.filter (fun observation => observation.currentIn context)
    if current.isEmpty then .stale
    else if current.any (fun observation => observation.outcome == .failed) then .failed
    else if current.any (fun observation => observation.outcome == .verifierError) then .verifierError
    else if current.any (fun observation => observation.outcome == .blocked) then .blocked
    else if current.any (fun observation =>
      observation.behaviorallyQualified context requirement validators) then .passedCalibrated
    else .passedUncalibrated

def derivePromotion
    (context : PromotionContext) (adequacy : AdequacyState)
    (observation : ObservationState) : PromotionState :=
  if context.target.disposition != .promoted then .targetUnpromoted
  else if adequacy != .accepted then .adequacyOpen
  else if observation == .passedCalibrated then .conformant
  else .observationInsufficient

def deriveRequirementState
    (context : PromotionContext) (requirement : Requirement)
    (specification : BoundarySpec) (adequacyClaim : AdequacyClaim)
    (candidates : List CandidateMapping) (validators : List ValidatorRef)
    (observations : List Observation) : RequirementState :=
  let specificationState :=
    if specification.structurallyCovers requirement
    then SpecificationState.structurallySpecified
    else SpecificationState.missing
  let adequacy := deriveAdequacy adequacyClaim
  let observation := deriveObservation context requirement validators observations
  { requirement := requirement.id
    inventory := .interpreted
    specification := specificationState
    adequacy
    implementation := deriveImplementation requirement.id candidates
    observation
    promotion := derivePromotion context adequacy observation }

inductive GapKind where
  | targetPromotion
  | specification
  | adequacy
  | implementation
  | observation
  | validator
  | failedBehavior
  | environment
  deriving DecidableEq, BEq, Repr, Inhabited

structure Gap where
  id : String
  requirement : RequirementId
  kind : GapKind
  description : String
  deriving DecidableEq, BEq, Repr, Inhabited

def gapsFor (state : RequirementState) : List Gap :=
  let reqPrefix := state.requirement.value
  let targetGap :=
    if state.promotion == .targetUnpromoted then
      [{ id := s!"GAP-TARGET-{reqPrefix}", requirement := state.requirement,
         kind := .targetPromotion,
         description := "The extracted upstream target has not been promoted." }]
    else []
  let specGap :=
    if state.specification == .missing then
      [{ id := s!"GAP-SPEC-{reqPrefix}", requirement := state.requirement,
         kind := .specification,
         description := "No boundary specification structurally covers the requirement." }]
    else []
  let adequacyGap :=
    if state.adequacy != .accepted then
      [{ id := s!"GAP-ADEQUACY-{reqPrefix}", requirement := state.requirement,
         kind := .adequacy,
         description := "The D ∧ S ⇒ R adequacy obligation is not accepted." }]
    else []
  let implementationGap :=
    if state.implementation == .noCandidate then
      [{ id := s!"GAP-IMPLEMENTATION-{reqPrefix}", requirement := state.requirement,
         kind := .implementation,
         description := "No product implementation candidate is mapped to the specification." }]
    else []
  let observationGap :=
    match state.observation with
    | .unobserved | .stale =>
        [{ id := s!"GAP-OBSERVATION-{reqPrefix}", requirement := state.requirement,
           kind := .observation,
           description := "No current observation covers the requirement." }]
    | .blocked =>
        [{ id := s!"GAP-ENVIRONMENT-{reqPrefix}", requirement := state.requirement,
           kind := .environment,
           description := "A prerequisite or environment condition blocks observation." }]
    | .failed =>
        [{ id := s!"GAP-BEHAVIOR-{reqPrefix}", requirement := state.requirement,
           kind := .failedBehavior,
           description := "A current observation reports non-conforming behavior." }]
    | .verifierError | .passedUncalibrated =>
        [{ id := s!"GAP-VALIDATOR-{reqPrefix}", requirement := state.requirement,
           kind := .validator,
           description := "The observer failed or has not established calibrated sensitivity." }]
    | .passedCalibrated => []
  targetGap ++ specGap ++ adequacyGap ++ implementationGap ++ observationGap

end Tgrad.Growth

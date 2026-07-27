import Tgrad.Spec.Parity

/-! # Tgrad.Spec.ParityCoverage — total coverage, never a fabricated score

`Parity.Contract.structurallyWellFormed` intentionally accepts partial working
contracts.  Publication needs a stronger law: exactly one cell for every
generated requirement, immutable identity joins, and no scalar percentage that
can hide missing or excluded rows.  This module owns that stronger boundary.
-/

namespace Tgrad.Spec.Parity

/-- This literal is the reviewed denominator for upstream revision
`19c4d736f2bc`.  A future extractor result cannot silently move it: changing
the target requires a separate promotion that updates this anchor. -/
def reviewedRequirementCount : Nat := 590

theorem generated_target_matches_reviewed_denominator :
    ParityTarget.requirementCount = reviewedRequirementCount := by
  native_decide

def Contract.requirementIds (contract : Contract) : List String :=
  contract.requirements.map (·.id)

def Contract.cellIds (contract : Contract) : List String :=
  contract.cells.map (·.requirementId)

/-- A matrix is total only when its cells are an exact one-to-one cover of the
requirement denominator.  Subsets and duplicate rows are rejected even if all
present cells happen to pass. -/
def Contract.hasTotalCoverageMatrix (contract : Contract) : Bool :=
  let requirementIds := contract.requirementIds
  let cellIds := contract.cellIds
  requirementIds.eraseDups.length == requirementIds.length &&
  cellIds.eraseDups.length == cellIds.length &&
  requirementIds.length == cellIds.length &&
  requirementIds.all cellIds.contains &&
  cellIds.all requirementIds.contains

def CoverageCell.confirmedState? (cell : CoverageCell) : Option CoverageState :=
  match cell.state with
  | .confirmed value _ => some value
  | _ => none

structure CoverageCounts where
  total : Nat
  unknown : Nat
  absent : Nat
  scaffold : Nat
  bounded : Nat
  conformant : Nat
  drifted : Nat
  blockedByHardware : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

def Contract.coverageCounts (contract : Contract) : CoverageCounts :=
  contract.cells.foldl (fun counts cell =>
    let counts := { counts with total := counts.total + 1 }
    match cell.confirmedState? with
    | none => { counts with unknown := counts.unknown + 1 }
    | some .absent => { counts with absent := counts.absent + 1 }
    | some .scaffold => { counts with scaffold := counts.scaffold + 1 }
    | some .bounded => { counts with bounded := counts.bounded + 1 }
    | some .conformant => { counts with conformant := counts.conformant + 1 }
    | some .drifted => { counts with drifted := counts.drifted + 1 }
    | some .blockedByHardware =>
        { counts with blockedByHardware := counts.blockedByHardware + 1 }) {}

def CoverageCounts.accountedFor (counts : CoverageCounts) : Bool :=
  counts.total ==
    counts.unknown + counts.absent + counts.scaffold + counts.bounded +
    counts.conformant + counts.drifted + counts.blockedByHardware

/-! ## Atomic observations

The 590 requirements are the public denominator.  A required row expands to
one obligation for every declared dimension and environment.  Agents author
observations, never row states; the row state is derived from the selected
atomic outcomes. -/

inductive ObservationOutcome where
  | pass
  | fail
  | error
  | blocked
  deriving DecidableEq, BEq, Repr, Inhabited

structure CoverageObligation where
  id : String
  requirementId : String
  requirementIdentity : String
  dimension : Dimension
  environmentId : String
  equivalenceRelation : String
  deriving DecidableEq, BEq, Repr, Inhabited

def CoverageObligation.wellFormed (obligation : CoverageObligation) : Bool :=
  !obligation.id.isEmpty &&
  !obligation.requirementId.isEmpty &&
  !obligation.requirementIdentity.isEmpty &&
  !obligation.environmentId.isEmpty &&
  !obligation.equivalenceRelation.isEmpty

structure CoverageObservation where
  id : String
  obligationId : String
  outcome : ObservationOutcome
  evidenceId : String
  diagnostic : String := ""
  deriving DecidableEq, BEq, Repr, Inhabited

def CoverageObservation.wellFormed
    (obligationIds evidenceIds : List String)
    (observation : CoverageObservation) : Bool :=
  !observation.id.isEmpty &&
  obligationIds.contains observation.obligationId &&
  evidenceIds.contains observation.evidenceId &&
  (observation.outcome == .pass || !observation.diagnostic.isEmpty)

inductive DerivedCoverageState where
  | unobserved
  | pass
  | fail
  | blocked
  deriving DecidableEq, BEq, Repr, Inhabited

def deriveCoverageState
    (obligations : List CoverageObligation)
    (observations : List CoverageObservation) : DerivedCoverageState :=
  if obligations.isEmpty then .unobserved
  else if obligations.any fun obligation =>
      !(observations.any fun observation => observation.obligationId == obligation.id)
    then .unobserved
  else if obligations.any fun obligation =>
      observations.any fun observation =>
        observation.obligationId == obligation.id &&
        (observation.outcome == .fail || observation.outcome == .error)
    then .fail
  else if obligations.any fun obligation =>
      observations.any fun observation =>
        observation.obligationId == obligation.id &&
        observation.outcome == .blocked
    then .blocked
  else .pass

def observationsSelectAtMostOnePerObligation
    (observations : List CoverageObservation) : Bool :=
  let ids := observations.map (·.obligationId)
  ids.eraseDups.length == ids.length

def atomicCoverageWellFormed
    (requirements : List Requirement)
    (obligations : List CoverageObligation)
    (observations : List CoverageObservation)
    (evidence : List EvidenceRef) : Bool :=
  let requirementIds := requirements.map (·.id)
  let obligationIds := obligations.map (·.id)
  let evidenceIds := evidence.map (·.id)
  requirementIds.length == reviewedRequirementCount &&
  requirementIds.eraseDups.length == requirementIds.length &&
  obligationIds.eraseDups.length == obligationIds.length &&
  observationsSelectAtMostOnePerObligation observations &&
  (obligations.all fun obligation =>
    obligation.wellFormed && requirementIds.contains obligation.requirementId) &&
  (observations.all fun observation =>
    observation.wellFormed obligationIds evidenceIds)

theorem no_observation_cannot_become_a_pass
    (obligations : List CoverageObligation) :
    deriveCoverageState obligations [] != .pass := by
  cases obligations <;> simp [deriveCoverageState]

/-- Claim readiness strengthens `Contract.complete` with a total denominator.
The only top-level result is a boolean universal claim; exact state counts are
available for diagnosis, but this API deliberately defines no percentage. -/
def Contract.coverageClaimReady (contract : Contract) : Bool :=
  contract.hasTotalCoverageMatrix && contract.complete

def targetSkeletonFor (subjectTree : String) (profile : Profile) : Contract :=
  contractSkeleton subjectTree profile

theorem generated_skeleton_is_an_exact_coverage_matrix
    (subjectTree : String) (profile : Profile) :
    (targetSkeletonFor subjectTree profile).hasTotalCoverageMatrix = true := by
  native_decide

theorem generated_skeleton_counts_every_requirement
    (subjectTree : String) (profile : Profile) :
    (targetSkeletonFor subjectTree profile).coverageCounts.total =
      reviewedRequirementCount := by
  native_decide

theorem generated_skeleton_counts_are_accounted_for
    (subjectTree : String) (profile : Profile) :
    (targetSkeletonFor subjectTree profile).coverageCounts.accountedFor = true := by
  native_decide

theorem importing_the_denominator_does_not_create_a_public_api_claim :
    (targetSkeletonFor "unobserved-subject-tree" .publicApi).coverageClaimReady = false := by
  native_decide

theorem target_contract_remains_epistemically_unknown :
    targetContract.isConfirmed = false := by
  native_decide

end Tgrad.Spec.Parity

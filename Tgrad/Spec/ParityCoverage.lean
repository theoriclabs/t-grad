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

/-! ## Diagnostic suite projections

A suite run and a promotable coverage observation are deliberately different
things.  The former can reveal where compatibility first fails even when the
run omitted the immutable subject tree, environment manifest, or raw
diagnostics required by `EvidenceRef`.  Keeping that distinction typed lets us
use an honest red result to shape work without laundering it into parity
evidence.

The projection below is over the same 590 generated rows as the contract.  It
does not add a second denominator and it does not define a percentage. -/

inductive SuiteFailureStage where
  | collection
  | execution
  deriving DecidableEq, BEq, Repr, Inhabited

inductive SuiteProjectionDisposition where
  | required
  | excluded
  | notApplicable
  deriving DecidableEq, BEq, Repr, Inhabited

inductive SuiteCellObservation where
  | unobserved
  | pass
  | failure (stage : SuiteFailureStage)
  deriving DecidableEq, BEq, Repr, Inhabited

structure DiagnosticCoverageCell where
  requirementId : String
  disposition : SuiteProjectionDisposition
  observation : SuiteCellObservation
  sourceArtifactHash : Option String := none
  rationale : String
  deriving DecidableEq, BEq, Repr, Inhabited

def DiagnosticCoverageCell.wellFormed (cell : DiagnosticCoverageCell) : Bool :=
  !cell.requirementId.isEmpty && !cell.rationale.isEmpty &&
  match cell.disposition, cell.observation, cell.sourceArtifactHash with
  | .required, .unobserved, none => true
  | .required, .pass, some hash => !hash.isEmpty
  | .required, .failure _, some hash => !hash.isEmpty
  | .excluded, .unobserved, none => true
  | .notApplicable, .unobserved, none => true
  | _, _, _ => false

theorem an_excluded_cell_cannot_be_observed_as_passing
    (requirementId rationale : String) (artifactHash : String) :
    (DiagnosticCoverageCell.wellFormed
      { requirementId,
        disposition := .excluded,
        observation := .pass,
        sourceArtifactHash := some artifactHash,
        rationale }) = false := by
  rfl

structure SuiteProjectionCounts where
  total : Nat
  required : Nat
  excluded : Nat
  notApplicable : Nat
  observed : Nat
  unobservedRequired : Nat
  passed : Nat
  collectionFailures : Nat
  executionFailures : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

def SuiteProjectionCounts.accountedFor (counts : SuiteProjectionCounts) : Bool :=
  counts.total == counts.required + counts.excluded + counts.notApplicable &&
  counts.required == counts.observed + counts.unobservedRequired &&
  counts.observed == counts.passed + counts.collectionFailures +
    counts.executionFailures

structure UpstreamSuiteCounts where
  files : Nat
  passedFiles : Nat
  collectionFailures : Nat
  executionFailures : Nat
  emptyFiles : Nat
  timedOutFiles : Nat
  passedTests : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

def UpstreamSuiteCounts.accountedFor (counts : UpstreamSuiteCounts) : Bool :=
  counts.files == counts.passedFiles + counts.collectionFailures +
    counts.executionFailures + counts.emptyFiles + counts.timedOutFiles

structure PublicOracleCounts where
  files : Nat
  passedFiles : Nat
  passedTests : Nat
  skippedTests : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

def PublicOracleCounts.allFilesPassed (counts : PublicOracleCounts) : Bool :=
  counts.files > 0 && counts.files == counts.passedFiles

structure DiagnosticSuiteProjection where
  upstreamRevision : String
  profile : Profile
  requirementCount : Nat
  requirementInventoryHash : String
  requirementIdsHash : String
  classificationHash : String
  sourceFixtureBundleHash : String
  adapterBundleHash : String
  resultBundleHash : String
  projectionContentHash : String
  referenceArtifactCommit : String
  subjectTree : Epistemic String
  verifierTree : Epistemic String
  environmentManifestHash : Epistemic String
  rawDiagnosticsBundleHash : Epistemic String
  equivalenceRelationRegistryHash : Epistemic String
  validatorCalibrationBundleHash : Epistemic String
  counts : SuiteProjectionCounts
  upstream : UpstreamSuiteCounts
  publicOracle : PublicOracleCounts
  deriving Repr, Inhabited

def DiagnosticSuiteProjection.wellFormed
    (projection : DiagnosticSuiteProjection) : Bool :=
  !projection.upstreamRevision.isEmpty &&
  !projection.requirementInventoryHash.isEmpty &&
  !projection.requirementIdsHash.isEmpty &&
  !projection.classificationHash.isEmpty &&
  !projection.sourceFixtureBundleHash.isEmpty &&
  !projection.adapterBundleHash.isEmpty &&
  !projection.resultBundleHash.isEmpty &&
  !projection.projectionContentHash.isEmpty &&
  !projection.referenceArtifactCommit.isEmpty &&
  projection.requirementCount == reviewedRequirementCount &&
  projection.counts.total == projection.requirementCount &&
  projection.counts.accountedFor &&
  projection.upstream.accountedFor &&
  projection.publicOracle.allFilesPassed

/-- The observed cells may be imported into a still-partial contract only when
the run records every identity needed to interpret those observations.  This
does not claim completeness or conformance; `Contract.coverageClaimReady`
remains the separate universal promotion predicate. -/
def DiagnosticSuiteProjection.canImportObservedCells
    (projection : DiagnosticSuiteProjection) : Bool :=
  projection.wellFormed &&
  projection.subjectTree.isConfirmed &&
  projection.verifierTree.isConfirmed &&
  projection.environmentManifestHash.isConfirmed &&
  projection.rawDiagnosticsBundleHash.isConfirmed &&
  projection.equivalenceRelationRegistryHash.isConfirmed &&
  projection.validatorCalibrationBundleHash.isConfirmed

/-- First foreign-suite observation of Tgrad's public surface.  The suite,
classification, and adapter bytes are preserved at `fdc741d`; the reviewed
requirement inventory and this projection were added later and bind those
bytes explicitly.  The runner did not record the product tree, environment, or
raw diagnostics it used, nor did it bind a verifier tree, relation registry,
or validator calibration.  The 34 observed rows are therefore real diagnostic
cells and deliberately not promoting evidence. -/
def firstPublicSurfaceProjection : DiagnosticSuiteProjection :=
  { upstreamRevision := "19c4d736f2bc8e26d21f08b28ffd6298408da00f",
    profile := .metal,
    requirementCount := 590,
    requirementInventoryHash :=
      "1843c762a3b16e72b351bfd4f1447b05644e06253b7ab20c0593e05fb28cda9b",
    requirementIdsHash :=
      "00d6421732a3df2f33f6d1520626c5084a8b66a91c81a2f4386fa82d8f5041c0",
    classificationHash :=
      "8281ef9c195b730ccd48d8af6600592c44d7abbe647a6103c4b9f54f1e2f2ba3",
    sourceFixtureBundleHash :=
      "38a2e21ea96d1f84a1e84e106c45a5cf8bf00f4aedec5bd858ec3916cf078076",
    adapterBundleHash :=
      "2a926a85a62b3563500872a2f43bdec6cde79a0d19143602f996a7734f14b685",
    resultBundleHash :=
      "7e5e587189d6b801e7ca31483219e0fdf89cace6e6cdf953616b9afd49b7ee95",
    projectionContentHash :=
      "7b7680f5036d2e456048d7a133029a5fb143b80b521377d01317ea7c8ccd58dc",
    referenceArtifactCommit := "fdc741dee8ebffec424ff8845177c16931347861",
    subjectTree := .unknown
      "the suite result files do not record the Tgrad commit/tree they executed"
      "rerun the frozen 34-file contract with an attributable suite runner",
    verifierTree := .unknown
      "the suite result files do not record the exact verifier tree that interpreted them"
      "rerun with the clean verifier commit and tree in every observation envelope",
    environmentManifestHash := .unknown
      "the suite result files do not identify Python, packages, host, or Metal selectors"
      "rerun under a content-addressed environment manifest with retained diagnostics",
    rawDiagnosticsBundleHash := .unknown
      "the suite result files retain counters but no stdout, stderr, or JUnit identities"
      "rerun with content-addressed stdout, stderr, and JUnit artifacts for every file",
    equivalenceRelationRegistryHash := .unknown
      "the historical run does not bind each observation to a reviewed equivalence relation"
      "rerun against a content-addressed relation registry joined to every obligation",
    validatorCalibrationBundleHash := .unknown
      "the historical run does not identify rejected mutants for its validators"
      "calibrate each validator/relation/environment tuple and bind the falsifier bundle",
    counts :=
      { total := 590,
        required := 471,
        excluded := 104,
        notApplicable := 15,
        observed := 34,
        unobservedRequired := 437,
        passed := 0,
        collectionFailures := 29,
        executionFailures := 5 },
    upstream :=
      { files := 138,
        passedFiles := 133,
        collectionFailures := 3,
        executionFailures := 1,
        emptyFiles := 1,
        timedOutFiles := 0,
        passedTests := 3419 },
    publicOracle :=
      { files := 34,
        passedFiles := 34,
        passedTests := 1003,
        skippedTests := 146 } }

theorem first_public_surface_projection_is_accounted_for :
    firstPublicSurfaceProjection.wellFormed = true := by
  native_decide

theorem first_public_surface_projection_has_no_passing_tgrad_file :
    firstPublicSurfaceProjection.counts.passed = 0 := by
  rfl

theorem first_public_surface_projection_preserves_failure_stage :
    firstPublicSurfaceProjection.counts.collectionFailures = 29 &&
    firstPublicSurfaceProjection.counts.executionFailures = 5 := by
  native_decide

theorem first_public_surface_projection_cannot_import_observed_cells :
    firstPublicSurfaceProjection.canImportObservedCells = false := by
  native_decide

end Tgrad.Spec.Parity

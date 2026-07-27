import Tgrad.Conformance.Claims

/-! # Tgrad.Evidence.Observations — revision-bound behavioral evidence

Evidence is an observation made by a specific verifier, through a specific
adapter, in a specific environment, against an exact subject tree.  A passing
artifact is not promotable when any of those identities drift.
-/

namespace Tgrad.Evidence

open Tgrad.Requirements
open Tgrad.Specification

inductive TargetDisposition where
  | extractedCandidate
  | promoted
  deriving DecidableEq, BEq, Repr, Inhabited

structure TargetRef where
  repository : String
  revision : String
  manifestHash : String
  profile : ProfileId
  disposition : TargetDisposition
  deriving DecidableEq, BEq, Repr, Inhabited

def TargetRef.wellFormed (target : TargetRef) : Bool :=
  !target.repository.trimAscii.isEmpty &&
  !target.revision.trimAscii.isEmpty &&
  !target.manifestHash.trimAscii.isEmpty &&
  target.profile.valid

structure TreeRef where
  revision : String
  contentHash : String
  dirty : Bool
  deriving DecidableEq, BEq, Repr, Inhabited

def TreeRef.wellFormed (tree : TreeRef) : Bool :=
  !tree.revision.trimAscii.isEmpty && !tree.contentHash.trimAscii.isEmpty

structure BoundaryIdentity where
  verifierTree : TreeRef
  adapterHash : String
  environmentId : String
  environmentHash : String
  scenarioManifestHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

def BoundaryIdentity.wellFormed (identity : BoundaryIdentity) : Bool :=
  identity.verifierTree.wellFormed &&
  !identity.adapterHash.trimAscii.isEmpty &&
  !identity.environmentId.trimAscii.isEmpty &&
  !identity.environmentHash.trimAscii.isEmpty &&
  !identity.scenarioManifestHash.trimAscii.isEmpty

inductive CalibrationOutcome where
  | validatorRejectedMutant
  | mutantSurvived
  | indeterminate
  deriving DecidableEq, BEq, Repr, Inhabited

structure Calibration where
  faultModel : String
  mutantTree : String
  artifactHash : String
  outcome : CalibrationOutcome
  deriving DecidableEq, BEq, Repr, Inhabited

def Calibration.establishesSensitivity (calibration : Calibration) : Bool :=
  !calibration.faultModel.trimAscii.isEmpty &&
  !calibration.mutantTree.trimAscii.isEmpty &&
  !calibration.artifactHash.trimAscii.isEmpty &&
  calibration.outcome == .validatorRejectedMutant

structure ValidatorRef where
  id : String
  version : String
  dimensions : List ObservationDimension
  calibrations : List Calibration
  deriving DecidableEq, BEq, Repr, Inhabited

def ValidatorRef.calibrated (validator : ValidatorRef) : Bool :=
  !validator.id.trimAscii.isEmpty &&
  !validator.version.trimAscii.isEmpty &&
  !validator.dimensions.isEmpty &&
  !validator.calibrations.isEmpty &&
  validator.calibrations.all Calibration.establishesSensitivity

inductive ObservationOutcome where
  | passed
  | failed
  | blocked
  | verifierError
  deriving DecidableEq, BEq, Repr, Inhabited

structure Observation where
  id : String
  requirement : RequirementId
  specification : SpecId
  targetRevision : String
  subjectTree : TreeRef
  boundary : BoundaryIdentity
  validatorId : String
  dimensions : List ObservationDimension
  outcome : ObservationOutcome
  blocker : String
  artifactHash : String
  observedAt : String
  deriving DecidableEq, BEq, Repr, Inhabited

def Observation.wellFormed (observation : Observation) : Bool :=
  !observation.id.trimAscii.isEmpty &&
  observation.requirement.valid &&
  observation.specification.valid &&
  !observation.targetRevision.trimAscii.isEmpty &&
  observation.subjectTree.wellFormed &&
  observation.boundary.wellFormed &&
  !observation.validatorId.trimAscii.isEmpty &&
  !observation.dimensions.isEmpty &&
  (if observation.outcome == .blocked
   then !observation.blocker.trimAscii.isEmpty
   else observation.blocker.trimAscii.isEmpty) &&
  !observation.artifactHash.trimAscii.isEmpty &&
  !observation.observedAt.trimAscii.isEmpty

structure PromotionContext where
  target : TargetRef
  subjectTree : TreeRef
  boundary : Option BoundaryIdentity
  deriving DecidableEq, BEq, Repr, Inhabited

def Observation.currentIn
    (observation : Observation) (context : PromotionContext) : Bool :=
  match context.boundary with
  | none => false
  | some boundary =>
      observation.targetRevision == context.target.revision &&
      observation.subjectTree == context.subjectTree &&
      !observation.subjectTree.dirty &&
      observation.boundary == boundary

def Observation.covers
    (observation : Observation) (requirement : Requirement) : Bool :=
  requirement.relation.dimensions.all observation.dimensions.contains

def Observation.validatorCalibrated
    (observation : Observation) (validators : List ValidatorRef) : Bool :=
  validators.any fun validator =>
    validator.id == observation.validatorId &&
    observation.dimensions.all validator.dimensions.contains &&
    validator.calibrated

def Observation.behaviorallyQualified
    (observation : Observation) (context : PromotionContext)
    (requirement : Requirement) (validators : List ValidatorRef) : Bool :=
  observation.wellFormed &&
  observation.requirement == requirement.id &&
  observation.currentIn context &&
  observation.covers requirement &&
  observation.validatorCalibrated validators &&
  observation.outcome == .passed

def Observation.promotable
    (observation : Observation) (context : PromotionContext)
    (requirement : Requirement) (validators : List ValidatorRef) : Bool :=
  context.target.disposition == .promoted &&
  observation.behaviorallyQualified context requirement validators

end Tgrad.Evidence

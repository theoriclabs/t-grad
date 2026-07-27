import Tgrad.Requirements.Relation

/-! # Tgrad.Requirements.Requirements — requirements and problem frames

An upstream symbol or test is provenance for a requirement, not the
requirement itself.  A requirement constrains shared world phenomena through
a typed observation relation.
-/

namespace Tgrad.Requirements

/-- An optative statement: behavior required in the problem world. -/
structure Requirement where
  id : RequirementId
  frame : FrameId
  profiles : List ProfileId
  monitored : List PhenomenonId
  controlled : List PhenomenonId
  assumptions : List AssumptionId
  relation : ObservationRelation
  provenance : List SourceRef
  statement : String
  deriving Repr, Inhabited

def Requirement.wellFormed (requirement : Requirement) : Bool :=
  requirement.id.valid &&
  requirement.frame.valid &&
  !requirement.profiles.isEmpty &&
  requirement.profiles.all ProfileId.valid &&
  !requirement.monitored.isEmpty &&
  requirement.monitored.all PhenomenonId.valid &&
  !requirement.controlled.isEmpty &&
  requirement.controlled.all PhenomenonId.valid &&
  requirement.assumptions.all AssumptionId.valid &&
  requirement.relation.wellFormed &&
  !requirement.provenance.isEmpty &&
  requirement.provenance.all SourceRef.wellFormed &&
  !requirement.statement.trimAscii.isEmpty

inductive ProblemFrameKind where
  | substitution
  | transformation
  | workpiece
  | commandedBehavior
  | causalState
  | connection
  | quality
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProblemFrame where
  id : FrameId
  kind : ProblemFrameKind
  domains : List DomainId
  sharedPhenomena : List PhenomenonId
  requirementIds : List RequirementId
  description : String
  deriving DecidableEq, BEq, Repr, Inhabited

def ProblemFrame.wellFormed (frame : ProblemFrame) : Bool :=
  frame.id.valid &&
  !frame.domains.isEmpty &&
  frame.domains.all DomainId.valid &&
  !frame.sharedPhenomena.isEmpty &&
  frame.sharedPhenomena.all PhenomenonId.valid &&
  !frame.requirementIds.isEmpty &&
  frame.requirementIds.all RequirementId.valid &&
  !frame.description.trimAscii.isEmpty

inductive Responsibility where
  | environment
  | compatibilityBoundary
  | product
  | verifier
  | specificationOwner
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ObstacleKind where
  | missingWorldFact
  | ambiguousRequirement
  | missingSpecification
  | missingImplementation
  | failedObservation
  | blockedObservation
  | uncalibratedValidator
  | unavailableEnvironment
  deriving DecidableEq, BEq, Repr, Inhabited

structure Obstacle where
  id : String
  kind : ObstacleKind
  obstructs : List RequirementId
  responsibility : Responsibility
  condition : String
  resolveBy : String
  deriving DecidableEq, BEq, Repr, Inhabited

def Obstacle.wellFormed (obstacle : Obstacle) : Bool :=
  !obstacle.id.trimAscii.isEmpty &&
  !obstacle.obstructs.isEmpty &&
  obstacle.obstructs.all RequirementId.valid &&
  !obstacle.condition.trimAscii.isEmpty &&
  !obstacle.resolveBy.trimAscii.isEmpty

structure RequirementCatalog where
  domains : List WorldDomain
  phenomena : List Phenomenon
  assumptions : List Assumption
  profiles : List CompatibilityProfile
  requirements : List Requirement
  frames : List ProblemFrame
  obstacles : List Obstacle
  deriving Repr, Inhabited

def RequirementCatalog.idsUnique (catalog : RequirementCatalog) : Bool :=
  (catalog.domains.map (·.id)).Nodup &&
  (catalog.phenomena.map (·.id)).Nodup &&
  (catalog.assumptions.map (·.id)).Nodup &&
  (catalog.profiles.map (·.id)).Nodup &&
  (catalog.requirements.map (·.id)).Nodup &&
  (catalog.frames.map (·.id)).Nodup

def RequirementCatalog.referencesKnown (catalog : RequirementCatalog) : Bool :=
  let domainIds := catalog.domains.map (·.id)
  let phenomenonIds := catalog.phenomena.map (·.id)
  let assumptionIds := catalog.assumptions.map (·.id)
  let profileIds := catalog.profiles.map (·.id)
  let requirementIds := catalog.requirements.map (·.id)
  let frameIds := catalog.frames.map (·.id)
  catalog.phenomena.all (fun phenomenon => domainIds.contains phenomenon.domain) &&
  catalog.assumptions.all (fun assumption =>
    assumption.domains.all domainIds.contains) &&
  catalog.profiles.all (fun profile =>
    profile.includedFrames.all frameIds.contains) &&
  catalog.requirements.all (fun requirement =>
    frameIds.contains requirement.frame &&
    requirement.profiles.all profileIds.contains &&
    requirement.monitored.all phenomenonIds.contains &&
    requirement.controlled.all phenomenonIds.contains &&
    requirement.assumptions.all assumptionIds.contains) &&
  catalog.frames.all (fun frame =>
    frame.domains.all domainIds.contains &&
    frame.sharedPhenomena.all phenomenonIds.contains &&
    frame.requirementIds.all requirementIds.contains) &&
  catalog.obstacles.all (fun obstacle =>
    obstacle.obstructs.all requirementIds.contains)

def RequirementCatalog.wellFormed (catalog : RequirementCatalog) : Bool :=
  !catalog.domains.isEmpty && catalog.domains.all WorldDomain.wellFormed &&
  !catalog.phenomena.isEmpty && catalog.phenomena.all Phenomenon.wellFormed &&
  catalog.assumptions.all Assumption.wellFormed &&
  !catalog.profiles.isEmpty && catalog.profiles.all CompatibilityProfile.wellFormed &&
  !catalog.requirements.isEmpty && catalog.requirements.all Requirement.wellFormed &&
  !catalog.frames.isEmpty && catalog.frames.all ProblemFrame.wellFormed &&
  catalog.obstacles.all Obstacle.wellFormed &&
  catalog.idsUnique && catalog.referencesKnown

end Tgrad.Requirements

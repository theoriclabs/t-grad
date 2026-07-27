import Tgrad.Growth.Derived

/-! # Typed projection target for the prospective broadcast-add manifest -/

namespace Tgrad.Growth.BroadcastAddManifest

open Tgrad.Requirements
open Tgrad.Growth

inductive OperandKind where
  | tensor
  | scalar
  deriving DecidableEq, BEq, Repr, Inhabited

inductive PilotDtype where
  | float32
  | int32
  deriving DecidableEq, BEq, Repr, Inhabited

inductive TraceEvent where
  | constructLeft
  | constructRight
  | snapshotInputs
  | invokeAdd
  | captureResultIdentity
  | observeShape
  | observeDtype
  | realize1
  | readback1
  | realize2
  | readback2
  | snapshotInputsAfter
  | observeShapeIfConstructed
  | recordTerminalStageClassMessage
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ObservationToken where
  | construction
  | shape
  | shapeAccess
  | dtype
  | exactValues
  | realizeIdentity
  | repeatedReadback
  | inputsUnchanged
  | exceptionStage
  | exceptionClass
  | exceptionMessage
  deriving DecidableEq, BEq, Repr, Inhabited

structure OperandFact where
  kind : OperandKind
  construction : String
  shape : List Nat
  dtype : PilotDtype
  valueCount : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure ScenarioFact where
  id : String
  requirementIds : List RequirementId
  left : OperandFact
  right : OperandFact
  observations : List ObservationToken
  deriving DecidableEq, BEq, Repr, Inhabited

def ScenarioFact.wellFormed (scenario : ScenarioFact) : Bool :=
  !scenario.id.trimAscii.isEmpty &&
  !scenario.requirementIds.isEmpty && scenario.requirementIds.Nodup &&
  !scenario.left.construction.trimAscii.isEmpty &&
  !scenario.right.construction.trimAscii.isEmpty &&
  scenario.left.valueCount > 0 && scenario.right.valueCount > 0 &&
  !scenario.observations.isEmpty && scenario.observations.Nodup

structure MutationObligation where
  id : String
  targetScenarios : List String
  fault : String
  implementation : String
  mustFail : List ObservationDimension
  mustNotChange : List ObservationDimension
  mayBeUnobserved : List ObservationDimension
  deriving DecidableEq, BEq, Repr, Inhabited

def MutationObligation.wellFormed (mutation : MutationObligation) : Bool :=
  !mutation.id.trimAscii.isEmpty &&
  !mutation.targetScenarios.isEmpty && mutation.targetScenarios.Nodup &&
  !mutation.fault.trimAscii.isEmpty &&
  !mutation.implementation.trimAscii.isEmpty &&
  !mutation.mustFail.isEmpty &&
  (mutation.mustFail ++ mutation.mustNotChange ++
    mutation.mayBeUnobserved).Nodup

inductive ForbiddenInference where
  | conformant
  | promoted
  | fullTinygradParity
  deriving DecidableEq, BEq, Repr, Inhabited

structure FrozenManifest where
  schemaVersion : Nat
  packetId : String
  baselineRevision : String
  upstreamRevision : String
  backend : String
  deterministic : Bool
  operator : String
  legalTrace : List TraceEvent
  incompatibleTrace : List TraceEvent
  exclusions : List String
  path : String
  contentHash : String
  requirementIds : List RequirementId
  scenarios : List ScenarioFact
  mutations : List MutationObligation
  expectedBefore : ObservationState
  allowedAfter : List ObservationState
  forbiddenInferences : List ForbiddenInference
  deriving Repr, Inhabited

def FrozenManifest.wellFormed (manifest : FrozenManifest) : Bool :=
  manifest.schemaVersion > 0 &&
  !manifest.packetId.trimAscii.isEmpty &&
  !manifest.baselineRevision.trimAscii.isEmpty &&
  !manifest.upstreamRevision.trimAscii.isEmpty &&
  !manifest.backend.trimAscii.isEmpty && manifest.deterministic &&
  !manifest.operator.trimAscii.isEmpty &&
  !manifest.legalTrace.isEmpty && manifest.legalTrace.Nodup &&
  !manifest.incompatibleTrace.isEmpty && manifest.incompatibleTrace.Nodup &&
  !manifest.exclusions.isEmpty && manifest.exclusions.Nodup &&
  !manifest.path.trimAscii.isEmpty &&
  !manifest.contentHash.trimAscii.isEmpty &&
  !manifest.requirementIds.isEmpty && manifest.requirementIds.Nodup &&
  !manifest.scenarios.isEmpty && manifest.scenarios.all ScenarioFact.wellFormed &&
  (manifest.scenarios.map (·.id)).Nodup &&
  manifest.scenarios.all (fun scenario =>
    scenario.requirementIds.all manifest.requirementIds.contains) &&
  !manifest.mutations.isEmpty &&
  manifest.mutations.all MutationObligation.wellFormed &&
  (manifest.mutations.map (·.id)).Nodup &&
  manifest.expectedBefore == .unobserved &&
  !manifest.allowedAfter.isEmpty && manifest.allowedAfter.Nodup &&
  !manifest.forbiddenInferences.isEmpty && manifest.forbiddenInferences.Nodup

end Tgrad.Growth.BroadcastAddManifest

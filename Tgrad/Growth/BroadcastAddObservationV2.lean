import Tgrad.Growth.BroadcastAddObservationV1
import Tgrad.Growth.BroadcastAddConstructorCandidateV1

/-! # Post-constructor calibrated TGrad observation -/

namespace Tgrad.Growth.BroadcastAddObservationV2

def evidenceId : String :=
  "8f6a7f9dc4bf105d23773e6fafc7dc145b8b3e7d8c5b6bf0a0cc8edfd9531de1"

def subjectRevision : String :=
  "4a489e3"

def constructedFloat32Scenarios : List String :=
  ["ADD-SAME-SHAPE-F32", "ADD-SINGLETON-AXIS-F32",
   "ADD-TWO-SIDED-BROADCAST-F32", "ADD-INCOMPATIBLE-SHAPES"]

def unsupportedInt32Scenarios : List String :=
  ["ADD-RANK-EXTENSION-I32", "ADD-I32-F32-SCALAR-PROMOTION"]

def snapshotBlockedScenarios : List String :=
  ["ADD-SAME-SHAPE-F32", "ADD-SINGLETON-AXIS-F32",
   "ADD-TWO-SIDED-BROADCAST-F32"]

def snapshotFailure : String := "'Tensor' object has no attribute 'tolist'"

def incompatibleTerminalStage : String := "invoke_add"

theorem v2_constructor_prediction_was_exact :
    constructedFloat32Scenarios =
      BroadcastAddConstructorCandidateV1.float32ScenariosExpectedToCrossConstruction ∧
    unsupportedInt32Scenarios =
      BroadcastAddConstructorCandidateV1.int32ScenariosExpectedToRemainUnsupported := by
  decide

theorem three_legal_float32_scenarios_expose_snapshot_readback :
    snapshotBlockedScenarios.length = 3 := by decide

end Tgrad.Growth.BroadcastAddObservationV2

import Tgrad.Growth.BroadcastAddObservationV2

/-! # Prospective float32 view-readback candidate -/

namespace Tgrad.Growth.BroadcastAddReadbackCandidateV1

def packetId : String := "WORK-PY-F32-VIEW-READBACK-V1"

def baselineRevision : String :=
  "6a646a54007be78c443db7373e14b2a35e186c75"

def triggerEvidenceId : String := BroadcastAddObservationV2.evidenceId

def productWriteSet : List String := ["python/tgrad.py", "Tgrad/Pipeline.lean"]

def verificationWriteSet : List String := ["scripts/spec/test_tensor_tolist.py"]

def snapshotScenariosExpectedToCross : List String :=
  BroadcastAddObservationV2.snapshotBlockedScenarios

def copyTypeMapping : List (String × String) :=
  [("bf16", "ushort"), ("f32", "uint")]

def predictedArithmeticOutcome : Option Bool := none

def predictedParityPromotion : Bool := false

theorem three_snapshot_boundaries_are_owned :
    snapshotScenariosExpectedToCross.length = 3 := by native_decide

theorem typed_copy_mapping_is_total_for_candidate_scope :
    copyTypeMapping = [("bf16", "ushort"), ("f32", "uint")] := by rfl

theorem candidate_does_not_predict_addition :
    predictedArithmeticOutcome = none ∧ predictedParityPromotion = false := by
  decide

end Tgrad.Growth.BroadcastAddReadbackCandidateV1

import Tgrad.Growth.BroadcastAddConstructorCandidateV1

/-! # V2 public-constructor work-shape amendment

V1 was rejected before commit because it omitted a verifier-side consumer of
the internal constructor. V2 changes only the write set and adds a migration
obligation; behavioral predictions remain inherited.
-/

namespace Tgrad.Growth.BroadcastAddConstructorCandidateV2

def packetId : String := "WORK-PY-TENSOR-PUBLIC-CONSTRUCTOR-V2"

def baselineRevision : String :=
  "58d18cedc82d6ec8b3ac282afe6be919329e5ca7"

def productWriteSet : List String := ["python/tgrad.py"]

def internalConsumerMigration : List String :=
  ["scripts/parity/fused_matmul_differential.py"]

def verificationWriteSet : List String :=
  ["scripts/spec/test_tensor_public_constructor.py"]

def inheritedCrossingPrediction : List String :=
  BroadcastAddConstructorCandidateV1.float32ScenariosExpectedToCrossConstruction

def inheritedUnsupportedPrediction : List String :=
  BroadcastAddConstructorCandidateV1.int32ScenariosExpectedToRemainUnsupported

def forbiddenWriteSet : List String :=
  BroadcastAddConstructorCandidateV1.forbiddenWriteSet

theorem v2_changes_only_the_work_shape :
    inheritedCrossingPrediction =
      BroadcastAddConstructorCandidateV1.float32ScenariosExpectedToCrossConstruction ∧
    inheritedUnsupportedPrediction =
      BroadcastAddConstructorCandidateV1.int32ScenariosExpectedToRemainUnsupported ∧
    forbiddenWriteSet = BroadcastAddConstructorCandidateV1.forbiddenWriteSet := by
  simp [inheritedCrossingPrediction, inheritedUnsupportedPrediction,
    forbiddenWriteSet]

theorem omitted_consumer_is_now_owned :
    internalConsumerMigration = ["scripts/parity/fused_matmul_differential.py"] := by
  rfl

end Tgrad.Growth.BroadcastAddConstructorCandidateV2

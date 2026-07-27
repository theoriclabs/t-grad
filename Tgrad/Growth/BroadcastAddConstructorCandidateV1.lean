import Tgrad.Growth.BroadcastAddObservationV1

/-! # Prospective public Tensor construction candidate

This packet is intentionally an observability step. It predicts that the
float32 scenarios cross public construction; it makes no prediction about the
pointwise kernel, broadcasting result, realization, or parity.
-/

namespace Tgrad.Growth.BroadcastAddConstructorCandidateV1

open Tgrad.Growth.BroadcastAddObservationV1

def packetId : String := "WORK-PY-TENSOR-PUBLIC-CONSTRUCTOR-V1"

def baselineRevision : String := "08ece776a380adf93abb691e0417b2bbe51bb105"

def triggerEvidenceId : String := tgradEvidenceId

def productWriteSet : List String := ["python/tgrad.py"]

def verificationWriteSet : List String :=
  ["scripts/spec/test_tensor_public_constructor.py"]

def forbiddenWriteSet : List String :=
  ["scripts/parity/shim/tinygrad/__init__.py",
   "scripts/parity/shim/tinygrad/dtype.py",
   "Tgrad/PythonFFI.lean", "Tgrad/Pipeline.lean", "c/tgrad_python.c"]

def float32ScenariosExpectedToCrossConstruction : List String :=
  ["ADD-SAME-SHAPE-F32", "ADD-SINGLETON-AXIS-F32",
   "ADD-TWO-SIDED-BROADCAST-F32", "ADD-INCOMPATIBLE-SHAPES"]

def int32ScenariosExpectedToRemainUnsupported : List String :=
  ["ADD-RANK-EXTENSION-I32", "ADD-I32-F32-SCALAR-PROMOTION"]

def requiredDataForms : List String :=
  ["scalar", "flat_list", "nested_list", "numpy_array"]

def requiredRanks : List Nat := [0, 1, 2, 3]

def predictedArithmeticOutcome : Option Bool := none

def predictedParityPromotion : Bool := false

theorem packet_partitions_the_six_frozen_scenarios :
    (float32ScenariosExpectedToCrossConstruction ++
      int32ScenariosExpectedToRemainUnsupported).Nodup ∧
    (float32ScenariosExpectedToCrossConstruction ++
      int32ScenariosExpectedToRemainUnsupported).length = results.length := by
  native_decide

theorem candidate_has_one_product_file : productWriteSet = ["python/tgrad.py"] := by
  rfl

theorem shim_and_runtime_boundaries_are_frozen :
    forbiddenWriteSet.contains "scripts/parity/shim/tinygrad/__init__.py" ∧
    forbiddenWriteSet.contains "scripts/parity/shim/tinygrad/dtype.py" ∧
    forbiddenWriteSet.contains "Tgrad/PythonFFI.lean" ∧
    forbiddenWriteSet.contains "Tgrad/Pipeline.lean" ∧
    forbiddenWriteSet.contains "c/tgrad_python.c" := by
  native_decide

theorem no_arithmetic_or_promotion_claim_is_predicted :
    predictedArithmeticOutcome = none ∧ predictedParityPromotion = false := by
  decide

end Tgrad.Growth.BroadcastAddConstructorCandidateV1

import Tgrad.Growth.BroadcastAddObservationV5
import Tgrad.Renderer.Elementwise
import Tgrad.Pipeline

/-! # Prospective int32 elementwise/storage candidate

The functionality boundary is wider than one scenario: once int32 can cross
the host/storage boundary, the existing Lean LUB should also make mixed
int32/float32 addition observable. The prediction therefore names both causal
consumers rather than adding an artificial mixed-dtype rejection.
-/

namespace Tgrad.Growth.BroadcastAddInt32CandidateV1

def packetId : String := "WORK-DTYPE-I32-ELEMENTWISE-V1"
def baselineRevision : String := "fe820050e14cf6746f2a866204b0fade10127b3d"
def triggerEvidenceId : String := BroadcastAddObservationV5.evidenceId

def productWriteSet : List String :=
  ["python/tgrad.py", "Tgrad/Renderer/Elementwise.lean", "Tgrad/Pipeline.lean"]

def adapterWriteSet : List String :=
  ["scripts/parity/shim/tinygrad/dtype.py"]

def verificationWriteSet : List String :=
  ["scripts/spec/test_int32_elementwise.py",
   "scripts/spec/test_tensor_public_constructor.py",
   "scripts/parity/test_substitution_shim.py",
   "scripts/devcheck.sh"]

def newlyMatchingScenarios : List String :=
  ["ADD-RANK-EXTENSION-I32", "ADD-I32-F32-SCALAR-PROMOTION"]

def aggregateBefore : Nat × Nat × Nat := (35, 13, 13)
def aggregateAfter : Nat × Nat × Nat := (57, 3, 1)
def precisionStressBoundary : Nat := 16777216
def promotionAllowed : Bool := false

theorem write_shape_is_partitioned :
    productWriteSet.length = 3 ∧ adapterWriteSet.length = 1 ∧
      verificationWriteSet.length = 4 := by
  decide

theorem representation_has_two_named_consumers :
    newlyMatchingScenarios.length = 2 ∧ newlyMatchingScenarios.Nodup := by
  decide

theorem predicted_partition_accounts_for_all_dimensions :
    let (sameBefore, differentBefore, unobservedBefore) := aggregateBefore
    let (sameAfter, differentAfter, unobservedAfter) := aggregateAfter
    sameBefore + differentBefore + unobservedBefore = 61 ∧
      sameAfter + differentAfter + unobservedAfter = 61 := by
  decide

theorem float_exact_integer_boundary_is_explicit : precisionStressBoundary = 2^24 := by
  decide

theorem scenario_success_is_not_requirement_promotion : promotionAllowed = false := by
  rfl

-- Product symbol pins: changes to the load-bearing Lean boundaries break the
-- spec build instead of leaving this packet as parallel prose.
#check (Tgrad.Renderer.Metal.mslScalarType : Tgrad.Dtype → Option String)
#check (Tgrad.Renderer.Metal.elementwiseKernelDeclRanked :
  Tgrad.BinOp → List Nat → Tgrad.UOp → Tgrad.UOp → Tgrad.Dtype →
    Tgrad.Dtype → Tgrad.Dtype → String → Option Tgrad.Renderer.Metal.KernelDecl)
#check (Tgrad.Pipeline.materializeViewKernelDeclForDtype :
  Tgrad.Schedule.View → Tgrad.UOp → Tgrad.Dtype →
    Tgrad.Renderer.Metal.KernelDecl)
#check (Tgrad.Dtype.lub : Tgrad.Dtype → Tgrad.Dtype → Tgrad.Dtype)

end Tgrad.Growth.BroadcastAddInt32CandidateV1

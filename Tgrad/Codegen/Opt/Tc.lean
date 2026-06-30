import Tgrad.Dtype

/-! # Tgrad.Codegen.Opt.Tc — TensorCore inventory (Metal)

  Lift from theograd_phases/08_wmma_matching/Demo.lean.
-/
namespace Tgrad

namespace Codegen

namespace Opt

structure TensorCore where
  dims              : List Nat
  threads           : Nat
  elementsPerThread : List Nat
  dtypeIn           : Dtype
  dtypeOut          : Dtype
  deriving Repr, Inhabited

/-- Metal's bf16/f32 TC inventory (the matmul fixture's target tile is
    the 8×8×8 bf16→f32 entry). Matches what phase 08's python_spec
    captured from `tinygrad.codegen.opt.tc.metal`. -/
def metalTensorCores : List TensorCore :=
  [ { dims := [8, 8, 8], threads := 32,
      elementsPerThread := [2, 2, 2], dtypeIn := .bfloat16_, dtypeOut := .float32_ } ]

end Opt

end Codegen

end Tgrad

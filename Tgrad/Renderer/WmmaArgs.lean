import Tgrad.Renderer.Base

/-! # Tgrad.Renderer.WmmaArgs — typed WMMA arg synthesis

  Captures the `__WMMA_<DIMS>_<dtypeIn>_<dtypeOut>` wrapper name and
  argument shape that tinygrad's MetalRenderer emits as the prelude
  for a TC matmul. At L3 we model the shape used by the bf16/f32
  matmul fixture (8×8×8 → bfloat2 × bfloat2 → float2).

  Full algebraic emit of the wrapper body lands at L8.
-/
namespace Tgrad

namespace Renderer

namespace Wmma

structure WmmaArgs where
  M        : Nat
  N        : Nat
  K        : Nat
  dtypeIn  : ScalarTy
  dtypeOut : ScalarTy
  /-- The width of the per-thread `thread_elements` chunk (`elementsPerThread[0]`). -/
  elementsPerThread : Nat
  deriving Repr, Inhabited

/-- Mirror tinygrad's `wmma_args` formatter:
    `__WMMA_<M>_<N>_<K>___<dtypeIn>_<dtypeOut>`. -/
def WmmaArgs.toFnName (w : WmmaArgs) : String :=
  "__WMMA_" ++ toString w.M ++ "_" ++ toString w.N ++ "_" ++ toString w.K ++
    "___" ++ w.dtypeIn.toName ++ "_" ++ w.dtypeOut.toName

/-- The bf16 64×64 matmul's WMMA prelude args. Matches phase-11's
    captured kernel: dim=(8,8,8), bfloat in, float out, EPT=2. -/
def matmul64x64Args : WmmaArgs :=
  { M := 8, N := 8, K := 8,
    dtypeIn := .bfloat_, dtypeOut := .float_,
    elementsPerThread := 2 }

end Wmma

end Renderer

end Tgrad

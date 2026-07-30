import Tgrad.Backend.Cuda

/-! # CUDA renderer leaf

This module translates shared typed fill semantics into CUDA C++ spelling.
It does not own dtype admission, scalar values, launch geometry, kernel/cache
identity, artifact binding, or availability.
-/
namespace Tgrad.Renderer.Cuda

open Tgrad.Backend

structure KernelSource where
  source : String
  scalar : RenderedScalar
  elementCount : Nat
  outputIndex : RenderedOutputIndex
  bounds : RenderedBounds
  deriving BEq, Repr, DecidableEq

def rendererIdentity : RendererContractIdentity :=
  (RendererContractIdentity.build "cuda-c-fill1d-v1").toOption.get
    (by native_decide)

private def valueType : FillValue → String
  | .int32 _ => "int"
  | .float32Bits _ => "float"

private def valueLiteral : FillValue → String
  | .int32 value => "((int)" ++ toString value.value ++ ")"
  | .float32Bits bits => "__uint_as_float(" ++ toString bits.toNat ++ "U)"

private def outputIndexExpression : OutputIndexPolicy → String
  | .linearBlockThreadX =>
      "(((unsigned long long)blockIdx.x * " ++
      "(unsigned long long)blockDim.x) + " ++
      "(unsigned long long)threadIdx.x)"

private def renderSource (plan : FillPlan) : String :=
  let index := outputIndexExpression plan.outputIndexPolicy
  let count := toString plan.elementCount.value ++ "ULL"
  String.intercalate "\n" [
    "extern \"C\" __global__ void " ++ plan.kernelName ++ "(" ++
      valueType plan.value ++ "* out) {",
    "  const unsigned long long idx = " ++ index ++ ";",
    "  if (idx < " ++ count ++ ") {",
    "    out[idx] = " ++ valueLiteral plan.value ++ ";",
    "  }",
    "}",
    ""]

def render (plan : FillPlan) : KernelSource := {
  source := renderSource plan
  scalar := plan.value.renderedScalar
  elementCount := plan.elementCount.value
  outputIndex := plan.outputIndexPolicy.rendered
  bounds := plan.boundsPolicy.rendered
}

def projectSemantics (payload : KernelSource) : Option RenderedSemantics :=
  some {
    scalar := payload.scalar
    elementCount := payload.elementCount
    outputIndex := payload.outputIndex
    bounds := payload.bounds
  }

def renderer : RendererContract KernelSource := {
  identity := rendererIdentity
  render
  projectSemantics
  renderPreserves := fun _ => rfl
}

end Tgrad.Renderer.Cuda

import Tgrad.Backend.FillPlan

/-! # CUDA leaf for the backend-neutral fill plan

This renderer owns CUDA spelling only.  Dtype/value/count, byte arithmetic,
launch geometry, and cache identity arrive already decided by Lean's shared
`Backend.FillPlan`.
-/
namespace Tgrad
namespace Renderer
namespace Cuda

inductive RenderError where
  | wrongBackend (actual : Backend.Id)
  | invalidProfile
  | invalidPlan (reason : Backend.FillPlanError)
  deriving BEq, Repr

structure KernelSource where
  kernelName : String
  source : String
  cacheIdentity : String
  deriving BEq, Repr

def valueType : Backend.FillValue → String
  | .int32 _ => "int"
  | .float32Bits _ => "float"

def valueLiteral : Backend.FillValue → String
  | .int32 value => "((int)" ++ toString value.value ++ ")"
  | .float32Bits bits => "__uint_as_float(" ++ toString bits.toNat ++ "U)"

def outputIndexExpression : Backend.OutputIndex → String
  | .globalLinear1D =>
      "(((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) + " ++
      "(unsigned long long)threadIdx.x)"

def renderFill (plan : Backend.FillPlan) : Except RenderError KernelSource :=
  if plan.profile.backend != .cuda then
    .error (.wrongBackend plan.profile.backend)
  else if !plan.profile.valid then
    .error .invalidProfile
  else match plan.revalidate with
  | .error reason => .error (.invalidPlan reason)
  | .ok () =>
    let name := plan.kernelName
    let idx := outputIndexExpression plan.outputIndex
    let count := toString plan.elementCount.value ++ "ULL"
    let source := String.intercalate "\n" [
      "extern \"C\" __global__ void " ++ name ++ "(" ++
        valueType plan.value ++ "* out) {",
      "  const unsigned long long idx = " ++ idx ++ ";",
      "  if (idx < " ++ count ++ ") {",
      "    out[idx] = " ++ valueLiteral plan.value ++ ";",
      "  }",
      "}",
      ""]
    .ok {
      kernelName := name
      source
      cacheIdentity := "dialect=cuda-c-v1|" ++ plan.cacheIdentity
    }

end Cuda
end Renderer
end Tgrad

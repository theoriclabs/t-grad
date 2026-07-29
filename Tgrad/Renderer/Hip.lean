import Tgrad.Backend.FillPlan

/-! # HIP source leaf for the neutral fill plan -/
namespace Tgrad.Renderer.Hip

open Tgrad.Backend

inductive RenderError where
  | wrongBackend (actual : BackendId)
  deriving BEq, Repr

def storageType : FillValue → String
  | .int32 _ => "int"
  | .float32Bits _ => "float"

def scalarLiteral : FillValue → String
  | .int32 value => toString value
  | .float32Bits bits => s!"__builtin_bit_cast(float, {bits}u)"

/-- HIP is only a dialect leaf: indexing, guard, count, dtype, value and
identity have already been fixed in the validated Lean plan. -/
def renderFill (plan : FillPlan) : Except RenderError String := do
  if plan.identity.backend != .hip then
    throw (.wrongBackend plan.identity.backend)
  let ty := storageType plan.value
  let literal := scalarLiteral plan.value
  pure <| String.intercalate "\n"
    ["#include <hip/hip_runtime.h>",
     s!"extern \"C\" __global__ void tgrad_fill({ty} *out) " ++ "{",
     "  const unsigned long long idx = ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) + (unsigned long long)threadIdx.x;",
     s!"  if (idx < {plan.elementCount}ULL) out[idx] = {literal};",
     "}", ""]

end Tgrad.Renderer.Hip

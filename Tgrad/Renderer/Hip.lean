import Tgrad.Backend.FillPlan

/-! # HIP source leaf for the neutral fill plan -/
namespace Tgrad.Renderer.Hip

open Tgrad.Backend

inductive RenderError where
  | wrongBackend (actual : BackendId)
  | kernelIdentityMismatch
  | sourceIdentityMismatch
  | sourceTextMismatch
  deriving BEq, Repr

/-- Public candidate data is not executable authority.  Only exact validation
against the plan-derived HIP rendering can construct private `KernelSource`. -/
structure KernelSourceCandidate where
  kernelIdentity : KernelIdentity
  sourceIdentity : SourceIdentity
  sourceText : String
  deriving BEq, Repr

/-- Plan-bound rendered artifact.  The constructor is private so an arbitrary
source string cannot be passed to the runtime compiler boundary. -/
structure KernelSource where
  private mk ::
  candidate : KernelSourceCandidate
  deriving Repr

def KernelSource.kernelIdentity (source : KernelSource) : KernelIdentity :=
  source.candidate.kernelIdentity

def KernelSource.sourceIdentity (source : KernelSource) : SourceIdentity :=
  source.candidate.sourceIdentity

def KernelSource.sourceText (source : KernelSource) : String :=
  source.candidate.sourceText

def KernelSource.kernelName (source : KernelSource) : String :=
  source.kernelIdentity.kernelName

def KernelSource.cacheIdentity (source : KernelSource) : String :=
  source.kernelIdentity.semanticIdentity

def storageType : FillValue → String
  | .int32 _ => "int"
  | .float32Bits _ => "float"

def scalarLiteral : FillValue → String
  | .int32 value => toString value
  | .float32Bits bits => s!"__builtin_bit_cast(float, {bits}u)"

def outputIndexExpression : OutputIndexPolicy → String
  | .linearBlockThreadX =>
      "((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) + (unsigned long long)threadIdx.x"

def guardedStore (policy : BoundsPolicy) (plan : FillPlan)
    (literal : String) : String :=
  match policy with
  | .guardElementCount =>
      s!"  if (idx < {plan.elementCount}ULL) out[idx] = {literal};"

private def renderText (plan : FillPlan) : String :=
  let ty := storageType plan.value
  let literal := scalarLiteral plan.value
  let index := outputIndexExpression plan.outputIndexPolicy
  String.intercalate "\n"
    ["#include <hip/hip_runtime.h>",
     s!"extern \"C\" __global__ void {plan.kernelName}({ty} *out) " ++ "{",
     s!"  const unsigned long long idx = {index};",
     guardedStore plan.boundsPolicy plan literal,
     "}", ""]

private def expectedCandidate (plan : FillPlan) : KernelSourceCandidate :=
  { kernelIdentity := plan.kernelIdentity
    sourceIdentity := plan.sourceIdentity
    sourceText := renderText plan }

/-- Exact candidate validation is intentionally public for negative boundary
tests, but only returns a private-constructor `KernelSource` after all plan,
dialect, kernel, cache, profile and source text fields agree. -/
def validateCandidate (plan : FillPlan) (candidate : KernelSourceCandidate) :
    Except RenderError KernelSource := do
  if plan.identity.backend != .hip then
    throw (.wrongBackend plan.identity.backend)
  if candidate.kernelIdentity != plan.kernelIdentity then
    throw .kernelIdentityMismatch
  if candidate.sourceIdentity != plan.sourceIdentity then
    throw .sourceIdentityMismatch
  if candidate.sourceText != renderText plan then
    throw .sourceTextMismatch
  pure { candidate := candidate }

def KernelSource.validateForPlan (plan : FillPlan) (source : KernelSource) :
    Except RenderError Unit := do
  let _ ← validateCandidate plan source.candidate
  pure ()

/-- HIP is only a dialect leaf: indexing, guard, count, dtype, value and
identity have already been fixed in the validated Lean plan. -/
def renderFill (plan : FillPlan) : Except RenderError KernelSource := do
  if plan.identity.backend != .hip then
    throw (.wrongBackend plan.identity.backend)
  validateCandidate plan (expectedCandidate plan)

end Tgrad.Renderer.Hip

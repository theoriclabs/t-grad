import Tgrad.Backend.Hip

/-! # HIP source renderer over the shared fill contract

HIP syntax is vendor-local.  The payload also projects the exact scalar,
element-count, output-index, and bounds meanings required by the shared
renderer contract.
-/
namespace Tgrad.Renderer.Hip

open Tgrad.Backend

inductive Dialect where
  | hip
  deriving BEq, Repr, DecidableEq

structure SourcePayload where
  dialect : Dialect
  sourceText : String
  scalar : RenderedScalar
  elementCount : Nat
  outputIndex : RenderedOutputIndex
  bounds : RenderedBounds
  deriving BEq, Repr, DecidableEq

def rendererIdentity : RendererContractIdentity :=
  (RendererContractIdentity.build "hip-source-v1").toOption.get
    (by native_decide)

def storageType : FillValue → String
  | .int32 _ => "int"
  | .float32Bits _ => "float"

def scalarLiteral : FillValue → String
  | .int32 value => toString value.value
  | .float32Bits bits =>
      "__builtin_bit_cast(float, " ++ toString bits.toNat ++ "u)"

def outputIndexExpression : OutputIndexPolicy → String
  | .linearBlockThreadX =>
      "((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) + " ++
      "(unsigned long long)threadIdx.x"

def guardedStore (policy : BoundsPolicy) (elementCount : Nat)
    (literal : String) : String :=
  match policy with
  | .guardElementCount =>
      "  if (idx < " ++ toString elementCount ++ "ULL) out[idx] = " ++
      literal ++ ";"

def renderSourceText (plan : FillPlan) : String :=
  let ty := storageType plan.value
  let literal := scalarLiteral plan.value
  let index := outputIndexExpression plan.outputIndexPolicy
  String.intercalate "\n"
    ["#include <hip/hip_runtime.h>",
     "extern \"C\" __global__ void " ++ plan.kernelName ++ "(" ++ ty ++
       " *out) {",
     "  const unsigned long long idx = " ++ index ++ ";",
     guardedStore plan.boundsPolicy plan.elementCount.value literal,
     "}", ""]

def renderPayload (plan : FillPlan) : SourcePayload :=
  { dialect := .hip
    sourceText := renderSourceText plan
    scalar := plan.value.renderedScalar
    elementCount := plan.elementCount.value
    outputIndex := plan.outputIndexPolicy.rendered
    bounds := plan.boundsPolicy.rendered }

def projectSemantics (payload : SourcePayload) : Option RenderedSemantics :=
  some {
    scalar := payload.scalar
    elementCount := payload.elementCount
    outputIndex := payload.outputIndex
    bounds := payload.bounds }

theorem renderPreserves (plan : FillPlan) :
    projectSemantics (renderPayload plan) = some plan.renderedSemantics := rfl

def rendererContract : RendererContract SourcePayload :=
  { identity := rendererIdentity
    render := renderPayload
    projectSemantics
    renderPreserves }

abbrev HipSourceArtifact (plan : FillPlan) :=
  SourceArtifact rendererContract plan

abbrev HipCompileRequest (plan : FillPlan) :=
  CompileRequest rendererContract plan

inductive RenderError where
  | profile (reason : Backend.Hip.ProfileError)
  | artifact (reason : ArtifactBindingError)
  deriving BEq, Repr

/-- Vendor-gated artifact entry point.  Generic shared rendering remains total
for its universal preservation proof, while this HIP leaf rejects every
non-HIP or malformed vendor profile before returning an artifact. -/
def renderFill (plan : FillPlan) : Except RenderError (HipSourceArtifact plan) := do
  match Backend.Hip.validateProfile plan.profile with
  | .ok () => pure ()
  | .error reason => throw (.profile reason)
  pure (rendererContract.renderArtifact plan)

def validateCandidate (plan : FillPlan)
    (candidate : SourceArtifactCandidate SourcePayload) :
    Except RenderError (HipSourceArtifact plan) := do
  match Backend.Hip.validateProfile plan.profile with
  | .ok () => pure ()
  | .error reason => throw (.profile reason)
  match SourceArtifact.validateCandidate rendererContract plan candidate with
  | .ok artifact => pure artifact
  | .error reason => throw (.artifact reason)

def buildCompileRequest (plan : FillPlan) (artifact : HipSourceArtifact plan) :
    Except RenderError (HipCompileRequest plan) := do
  match Backend.Hip.validateProfile plan.profile with
  | .ok () => pure ()
  | .error reason => throw (.profile reason)
  match CompileRequest.build rendererContract plan artifact with
  | .ok request => pure request
  | .error reason => throw (.artifact reason)

def HipSourceArtifact.sourceText {plan : FillPlan}
    (artifact : HipSourceArtifact plan) : String :=
  artifact.payload.sourceText

def HipCompileRequest.sourceText {plan : FillPlan}
    (request : HipCompileRequest plan) : String :=
  request.sourcePayload.sourceText

end Tgrad.Renderer.Hip

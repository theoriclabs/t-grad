import Tgrad.Backend.Hip
import Tgrad.Renderer.Hip
import Tgrad.Runtime.Hip

set_option maxRecDepth 2048

open Tgrad
open Tgrad.Backend

private def check (name : String) (condition : Bool) : IO Nat := do
  if condition then
    IO.println ("PASS " ++ name)
    pure 0
  else
    IO.eprintln ("FAIL " ++ name)
    pure 1

private def isError : Except ε α → Bool
  | .error _ => true
  | .ok _ => false

private def getOrThrow (result : Except ε α) [Repr ε] (label : String) : IO α :=
  match result with
  | .ok value => pure value
  | .error reason =>
      throw (IO.userError (label ++ " failed: " ++ reprStr reason))

private def hasProfileError (expected : Backend.Hip.ProfileError)
    (result : Except Backend.Hip.ProfileError DeviceProfile) : Bool :=
  match result with
  | .error actual => actual == expected
  | .ok _ => false

private def hasRenderProfileError (result : Except Renderer.Hip.RenderError α) : Bool :=
  match result with
  | .error (.profile _) => true
  | _ => false

private def negativeProbeFact : Runtime.Hip.ProbeFact → Bool
  | .negative _ => true
  | .reportedDeviceCount _ => false

private def hasRuntimeFailure (expectedClass : RuntimeErrorClass)
    (expectedStage : RuntimeStage) : RuntimeResult Runtime.Hip.Detail α → Bool
  | .error failure =>
      failure.errorClass == expectedClass && failure.stage == expectedStage
  | .ok _ => false

def main : IO Unit := do
  let runtimeProfile ← getOrThrow
    (Backend.Hip.buildProfile "gfx1100" .runtime "hiprtc" "rocm-6.3" 1024)
    "runtime profile"
  let archProfile ← getOrThrow
    (Backend.Hip.buildProfile "gfx942" .runtime "hiprtc" "rocm-6.3" 1024)
    "architecture profile"
  let offlineProfile ← getOrThrow
    (Backend.Hip.buildProfile "gfx1100" .offline "hipcc" "rocm-6.3" 1024)
    "offline profile"
  let versionProfile ← getOrThrow
    (Backend.Hip.buildProfile "gfx1100" .runtime "hiprtc" "rocm-6.4" 1024)
    "version profile"

  let floatPlan ← getOrThrow
    (Backend.Hip.buildFillPlan runtimeProfile .float32_
      (.float32Bits 0x40500000) 257 256) "float plan"
  let intPlan ← getOrThrow
    (Backend.Hip.buildFillPlan runtimeProfile .int32_ (.signed (-7)) 17 64)
    "int plan"
  let archPlan ← getOrThrow
    (Backend.Hip.buildFillPlan archProfile .float32_
      (.float32Bits 0x40500000) 257 256) "architecture plan"
  let offlinePlan ← getOrThrow
    (Backend.Hip.buildFillPlan offlineProfile .float32_
      (.float32Bits 0x40500000) 257 256) "offline plan"
  let versionPlan ← getOrThrow
    (Backend.Hip.buildFillPlan versionProfile .float32_
      (.float32Bits 0x40500000) 257 256) "version plan"

  let otherBackend ← getOrThrow (BackendIdentity.build "cuda") "other backend"
  let otherArch ← getOrThrow (ArchitectureIdentity.build "sm_80") "other arch"
  let otherTool ← getOrThrow (CompilerToolIdentity.build "nvrtc") "other tool"
  let otherVersion ← getOrThrow
    (CompilerVersionIdentity.build "cuda-12") "other version"
  let wrongProfile ← getOrThrow
    (DeviceProfile.build otherBackend otherArch {
      mode := .runtime, tool := otherTool, version := otherVersion } 1024)
    "wrong-backend descriptive profile"
  let wrongPlan ← getOrThrow
    (FillPlan.build wrongProfile .float32_ (.float32Bits 0x40500000) 257 256)
    "wrong-backend descriptive plan"

  let mut failures := 0

  failures := failures + (← check "HIP vendor identity is exact"
    (Backend.Hip.backendIdentity.stableName == "hip-rocm" &&
     runtimeProfile.backend == Backend.Hip.backendIdentity &&
     floatPlan.semanticIdentity.profile.backend == Backend.Hip.backendIdentity))
  failures := failures + (← check "gfx grammar admits observed architecture forms"
    (Backend.Hip.validArchitecture "gfx1100" &&
     Backend.Hip.validArchitecture "gfx942" &&
     Backend.Hip.validArchitecture "gfx90a"))
  failures := failures + (← check "gfx grammar rejects vendor over-admission"
    (!Backend.Hip.validArchitecture "gfxbanana" &&
     !Backend.Hip.validArchitecture "gfxz" &&
     !Backend.Hip.validArchitecture "gfx1z" &&
     !Backend.Hip.validArchitecture "gfx" &&
     !Backend.Hip.validArchitecture "gfx1" &&
     !Backend.Hip.validArchitecture "gfx12345" &&
     !Backend.Hip.validArchitecture "sm_80"))
  failures := failures + (← check "HIP compiler tool and mode coupling"
    (hasProfileError (.compilerModeMismatch .runtime .hipcc)
      (Backend.Hip.buildProfile "gfx1100" .runtime "hipcc" "rocm-6.3" 1024) &&
     hasProfileError (.compilerModeMismatch .offline .hiprtc)
      (Backend.Hip.buildProfile "gfx1100" .offline "hiprtc" "rocm-6.3" 1024) &&
     hasProfileError (.unsupportedCompilerTool "clang")
      (Backend.Hip.buildProfile "gfx1100" .runtime "clang" "rocm-6.3" 1024)))
  failures := failures + (← check "HIP leaf owns device thread limit admission"
    (hasProfileError (.invalidThreadLimit 0)
      (Backend.Hip.buildProfile "gfx1100" .runtime "hiprtc" "rocm-6.3" 0) &&
     hasProfileError (.invalidThreadLimit 1025)
      (Backend.Hip.buildProfile "gfx1100" .runtime "hiprtc" "rocm-6.3" 1025)))
  failures := failures + (← check "shared identity binds gfx tool mode and version"
    (floatPlan.semanticIdentity != archPlan.semanticIdentity &&
     floatPlan.semanticIdentity != offlinePlan.semanticIdentity &&
     floatPlan.semanticIdentity != versionPlan.semanticIdentity &&
     floatPlan.cacheIdentity != archPlan.cacheIdentity &&
     floatPlan.cacheIdentity != offlinePlan.cacheIdentity &&
     floatPlan.cacheIdentity != versionPlan.cacheIdentity))

  let floatArtifact ← getOrThrow (Renderer.Hip.renderFill floatPlan)
    "float source artifact"
  let intArtifact ← getOrThrow (Renderer.Hip.renderFill intPlan)
    "int source artifact"
  let floatSource := floatArtifact.sourceText
  let intSource := intArtifact.sourceText
  failures := failures + (← check "HIP source symbol and dialect are plan-bound"
    (floatArtifact.payload.dialect == .hip &&
     floatArtifact.identity ==
       floatPlan.sourceIdentity Renderer.Hip.rendererIdentity &&
     floatSource.contains ("void " ++ floatPlan.kernelName ++ "(float *out)") &&
     !floatSource.contains "metal" && !floatSource.contains "cuda"))
  failures := failures + (← check "HIP output-index source semantics"
    ((floatSource.splitOn "\n").contains
      ("  const unsigned long long idx = " ++
       "((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) + " ++
       "(unsigned long long)threadIdx.x;")))
  failures := failures + (← check "HIP bounds source semantics"
    (floatSource.contains "if (idx < 257ULL)" &&
     intSource.contains "if (idx < 17ULL)"))
  failures := failures + (← check "HIP exact scalar storage semantics"
    (floatSource.contains
       "out[idx] = __builtin_bit_cast(float, 1078984704u);" &&
     intSource.contains "void " && intSource.contains "(int *out)" &&
     intSource.contains "out[idx] = -7;"))
  failures := failures + (← check "renderer projection preserves shared semantics"
    (Renderer.Hip.rendererContract.projectSemantics floatArtifact.payload ==
      some floatPlan.renderedSemantics &&
     Renderer.Hip.rendererContract.projectSemantics intArtifact.payload ==
      some intPlan.renderedSemantics))

  let forgedText : SourceArtifactCandidate Renderer.Hip.SourcePayload := {
    identity := floatArtifact.identity
    payload := { floatArtifact.payload with
      sourceText := floatArtifact.payload.sourceText ++ "// arbitrary" } }
  let forgedScalar : SourceArtifactCandidate Renderer.Hip.SourcePayload := {
    identity := floatArtifact.identity
    payload := { floatArtifact.payload with scalar := .float32Bits 0x40900000 } }
  let crossProfile : SourceArtifactCandidate Renderer.Hip.SourcePayload := {
    identity := archPlan.sourceIdentity Renderer.Hip.rendererIdentity
    payload := Renderer.Hip.rendererContract.render archPlan }
  failures := failures + (← check "shared artifact rejects arbitrary HIP source"
    (isError (Renderer.Hip.validateCandidate floatPlan forgedText)))
  failures := failures + (← check "shared artifact rejects altered source meaning"
    (isError (Renderer.Hip.validateCandidate floatPlan forgedScalar)))
  failures := failures + (← check "shared artifact binds exact HIP profile"
    (isError (Renderer.Hip.validateCandidate floatPlan crossProfile)))
  failures := failures + (← check "HIP renderer rejects non-HIP profile"
    (hasRenderProfileError (Renderer.Hip.renderFill wrongPlan)))
  let compileRequest := Renderer.Hip.buildCompileRequest floatPlan floatArtifact
  failures := failures + (← check "exact shared artifact admits compile request"
    (match compileRequest with
    | .ok request =>
        request.sourceIdentity == floatArtifact.identity &&
        request.sourceText == floatSource &&
        request.renderedSemantics == some floatPlan.renderedSemantics
    | .error _ => false))

  failures := failures + (← check "sealed probe classes fail closed"
    (Runtime.Hip.sealedProbeSelfCheck &&
     negativeProbeFact (Runtime.Hip.classifyProbeRaw 0) &&
     negativeProbeFact (Runtime.Hip.classifyProbeRaw 2) &&
     negativeProbeFact (Runtime.Hip.classifyProbeRaw 3) &&
     negativeProbeFact (Runtime.Hip.classifyProbeRaw 256) &&
     Runtime.Hip.classifyProbeRaw 257 == .reportedDeviceCount 1))
  failures := failures + (← check "shared runtime artifact and copy interfaces bind"
    Runtime.Hip.staticInterfaceSelfCheck)
  failures := failures + (← check "runtime operations cannot fall back"
    (← Runtime.Hip.noFallbackSelfCheck))
  failures := failures + (← check "vendor status translation is structured"
    ((Runtime.Hip.translateVendorResult .launch 0).isOk &&
     hasRuntimeFailure .vendorFailure .launch
       (Runtime.Hip.translateVendorResult .launch 700)))

  let actualAvailability ← Runtime.Hip.localBoundary.availability
  match actualAvailability with
  | .unavailable requested reason =>
      failures := failures + (← check "local HIP availability names exact vendor"
        (requested == Backend.Hip.backendIdentity))
      failures := failures + (← check "local HIP availability has no fallback"
        (!actualAvailability.isAvailable))
      IO.println ("UNOBSERVED_ENVIRONMENT hip_runtime_or_complete_profile " ++
        reprStr reason.reasonClass ++ " " ++ reprStr reason.detail)
  | .available capability =>
      IO.eprintln ("FAIL unexpected executable HIP capability " ++ reprStr capability.deviceOf)
      failures := failures + 1

  IO.println ("SUMMARY failures=" ++ toString failures)
  if failures != 0 then IO.Process.exit 1

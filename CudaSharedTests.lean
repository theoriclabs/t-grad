import Tgrad.Backend.Cuda
import Tgrad.Renderer.Cuda
import Tgrad.Runtime.Cuda

open Tgrad
open Tgrad.Backend

def check (name : String) (condition : Bool) : IO Nat := do
  if condition then
    IO.println ("PASS " ++ name)
    pure 0
  else
    IO.eprintln ("FAIL " ++ name)
    pure 1

def isError : Except ε α → Bool
  | .error _ => true
  | .ok _ => false

def hasUnavailableClass (expected : UnavailableClass) :
    Availability Capability String → Bool
  | .unavailable _ reason => reason.reasonClass == expected
  | .available _ => false

def requestsCuda : Availability Capability String → Bool
  | .unavailable requested _ => requested == Backend.Cuda.identity
  | .available _ => false

def getIdentity (result : Except ε α) (what : String) : IO α :=
  match result with
  | .ok value => pure value
  | .error _ => throw (IO.userError ("failed to build " ++ what))

def getProfile (architecture : String) (tool : Backend.Cuda.CompilerTool)
    (version : String := "12.4") : IO DeviceProfile :=
  match Backend.Cuda.buildProfile architecture tool version 1024 with
  | .ok profile => pure profile
  | .error reason =>
      throw (IO.userError ("CUDA profile construction failed: " ++ reprStr reason))

def getPlan (profile : DeviceProfile) (dtype : Dtype) (input : ScalarInput)
    (count threads : Nat) : IO FillPlan :=
  match FillPlan.build profile dtype input count threads with
  | .ok plan => pure plan
  | .error reason =>
      throw (IO.userError ("CUDA plan construction failed: " ++ reprStr reason))

def contains (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

def main : IO Unit := do
  let runtimeProfile <- getProfile "sm_80" .nvrtc
  let offlineProfile <- getProfile "sm_80" .nvcc
  let floatPlan <- getPlan runtimeProfile .float32_
    (.float32Bits 1065353216) 513 256
  let intPlan <- getPlan runtimeProfile .int32_ (.signed (-7)) 257 256
  let offlinePlan <- getPlan offlineProfile .float32_
    (.float32Bits 1065353216) 513 256

  let otherBackend <- getIdentity (BackendIdentity.build "hip") "other backend"
  let otherArch <- getIdentity (ArchitectureIdentity.build "gfx1100") "other architecture"
  let otherTool <- getIdentity (CompilerToolIdentity.build "hiprtc") "other compiler tool"
  let otherVersion <- getIdentity
    (CompilerVersionIdentity.build "6.1") "other compiler version"
  let otherProfile <- getIdentity
    (DeviceProfile.build otherBackend otherArch {
      mode := .runtime, tool := otherTool, version := otherVersion } 1024)
    "other profile"

  let runtimeSource := Renderer.Cuda.renderer.render floatPlan
  let intSource := Renderer.Cuda.renderer.render intPlan
  let artifact := Renderer.Cuda.renderer.renderArtifact floatPlan
  let compileRequest := CompileRequest.build Renderer.Cuda.renderer floatPlan artifact
  let alteredCandidate : SourceArtifactCandidate Renderer.Cuda.KernelSource := {
    identity := artifact.identity
    payload := { artifact.payload with source := artifact.payload.source ++ "// altered" }
  }
  let crossProfileCandidate : SourceArtifactCandidate Renderer.Cuda.KernelSource := {
    identity := artifact.identity
    payload := artifact.payload
  }
  let crossProfilePayloadCandidate :
      SourceArtifactCandidate Renderer.Cuda.KernelSource := {
    identity := offlinePlan.sourceIdentity Renderer.Cuda.renderer.identity
    payload := artifact.payload
  }

  let localAvailability <- Runtime.Cuda.localAvailability
  let boundaryAvailability <- Runtime.Cuda.runtime.availability

  let mut failures := 0
  failures := failures + (← check "vendor identity is exact CUDA"
    (Backend.Cuda.identity.stableName == "cuda" &&
     Renderer.Cuda.renderer.identity.stableName == "cuda-c-fill1d-v1" &&
     runtimeProfile.backend == Backend.Cuda.identity &&
     runtimeProfile.backend != otherBackend))
  failures := failures + (← check "tool and architecture admission is vendor-local"
    (Backend.Cuda.profileAdmitted runtimeProfile &&
     Backend.Cuda.profileAdmitted offlineProfile &&
     runtimeProfile.architecture.stableName == "sm_80" &&
     runtimeProfile.compiler.mode == .runtime &&
     runtimeProfile.compiler.tool.stableName == "nvrtc" &&
     offlineProfile.compiler.mode == .offline &&
     offlineProfile.compiler.tool.stableName == "nvcc" &&
     isError (Backend.Cuda.buildProfile "gfx1100" .nvrtc "12.4" 1024) &&
     isError (Backend.Cuda.buildProfile "sm_" .nvrtc "12.4" 1024) &&
     isError (Backend.Cuda.buildProfile "sm_8a" .nvcc "12.4" 1024) &&
     isError (Backend.Cuda.buildProfile "sm_80" .nvrtc "12.4" 0) &&
     isError (Backend.Cuda.buildProfile "sm_80" .nvrtc "12.4" 1025) &&
     !Backend.Cuda.profileAdmitted otherProfile))
  failures := failures + (← check "renderer preserves exact shared semantics"
    (Renderer.Cuda.renderer.projectSemantics runtimeSource ==
       some floatPlan.renderedSemantics &&
     Renderer.Cuda.renderer.projectSemantics intSource ==
       some intPlan.renderedSemantics))
  failures := failures + (← check "renderer emits CUDA index and exact guard"
    (contains runtimeSource.source "blockIdx.x" &&
     contains runtimeSource.source "blockDim.x" &&
     contains runtimeSource.source "threadIdx.x" &&
     contains runtimeSource.source "if (idx < 513ULL)" &&
     contains intSource.source "if (idx < 257ULL)" &&
     contains runtimeSource.source floatPlan.kernelName))
  failures := failures + (← check "renderer preserves exact storage literals"
    (contains runtimeSource.source "float* out" &&
     contains runtimeSource.source "__uint_as_float(1065353216U)" &&
     contains intSource.source "int* out" &&
     contains intSource.source "((int)-7)"))
  failures := failures + (← check "shared artifact binds source and profile identity"
    (floatPlan.cacheIdentity != offlinePlan.cacheIdentity &&
     floatPlan.kernelIdentity != offlinePlan.kernelIdentity &&
     artifact.identity != offlinePlan.sourceIdentity Renderer.Cuda.renderer.identity &&
     !isError compileRequest &&
     isError (SourceArtifact.validateCandidate Renderer.Cuda.renderer floatPlan
       alteredCandidate) &&
     isError (SourceArtifact.validateCandidate Renderer.Cuda.renderer
       offlinePlan crossProfileCandidate) &&
     isError (SourceArtifact.validateCandidate Renderer.Cuda.renderer
       offlinePlan crossProfilePayloadCandidate)))
  failures := failures + (← check "availability is sealed and fail-closed"
    (!localAvailability.isAvailable && !boundaryAvailability.isAvailable &&
     hasUnavailableClass .probeAbsent localAvailability &&
     hasUnavailableClass .probeAbsent boundaryAvailability &&
     requestsCuda localAvailability && requestsCuda boundaryAvailability))
  failures := failures + (← check "negative probe classes never become capability"
    (hasUnavailableClass .runtimeLibraryMissing
       (Runtime.Cuda.availabilityFromProbe (some .runtimeMissing)) &&
     hasUnavailableClass .negativeProbe
       (Runtime.Cuda.availabilityFromProbe (some (.negative "driver rejected probe"))) &&
     hasUnavailableClass .probeFailed
       (Runtime.Cuda.availabilityFromProbe (some (.failed "driver error"))) &&
     hasUnavailableClass .noDevice
       (Runtime.Cuda.availabilityFromProbe (some .noDevice))))
  failures := failures + (← check "missing CUDA runtime has no backend fallback"
    (requestsCuda localAvailability && requestsCuda boundaryAvailability &&
     Backend.Cuda.identity != otherBackend))

  if failures != 0 then
    IO.eprintln ("cuda-shared-tests: " ++ toString failures ++ " failure(s)")
    IO.Process.exit 1
  IO.println "cuda-shared-tests: all focused checks green"

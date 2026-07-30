import Tgrad.Renderer.Hip

/-! # Fail-closed HIP runtime leaf over the shared runtime boundary

The local C bridge performs header-free runtime discovery and device counting
only.  Count-only observations can never mint the shared sealed capability.
No compile, allocation, launch, synchronization, or copy fallback exists.
-/
namespace Tgrad.Runtime.Hip

open Tgrad.Backend

inductive Detail where
  | runtimeLibraryMissing
  | deviceQuerySymbolMissing
  | deviceQueryFailed
  | noDevices
  | malformedProbe (raw : UInt64)
  | incompleteDeviceProfile (deviceCount : Nat)
  | profileRejected (reason : Backend.Hip.ProfileError)
  | vendorCode (stage : RuntimeStage) (code : UInt32)
  | implementationMissing (stage : RuntimeStage)
  deriving BEq, Repr

/-- Public wire facts are descriptive only.  They are never execution
authority and contain no credential accepted by the shared probe boundary. -/
inductive ProbeFact where
  | negative (detail : Detail)
  | reportedDeviceCount (count : Nat)
  deriving BEq, Repr

/-- Stable raw protocol from `c/hip_probe.c`: 0..3 are negative outcomes;
`256+n` reports `n>0` devices. -/
def classifyProbeRaw (raw : UInt64) : ProbeFact :=
  if raw == 0 then .negative .runtimeLibraryMissing
  else if raw == 1 then .negative .deviceQuerySymbolMissing
  else if raw == 2 then .negative .deviceQueryFailed
  else if raw == 3 then .negative .noDevices
  else if raw > 256 then .reportedDeviceCount (raw - 256).toNat
  else .negative (.malformedProbe raw)

structure ProbeCredential where
  private mk ::
  token : Unit

private def probeCredential : ProbeCredential := { token := () }

def probeAuthority : ProbeAuthority ProbeCredential :=
  { requestedBackend := Backend.Hip.backendIdentity
    admitsProfile := fun _ profile => Backend.Hip.admitsProfile profile }

abbrev DeviceCapability := AvailableCapability probeAuthority

def capabilityContract : CapabilityContract DeviceCapability :=
  AvailableCapability.contract probeAuthority

def observationFromRaw (raw : UInt64) :
    ProbeObservation probeAuthority Detail :=
  match classifyProbeRaw raw with
  | .negative .runtimeLibraryMissing => .runtimeMissing .runtimeLibraryMissing
  | .negative .deviceQuerySymbolMissing => .negative .deviceQuerySymbolMissing
  | .negative .deviceQueryFailed => .failed .deviceQueryFailed
  | .negative .noDevices => .profileRejected .noDevice .noDevices
  | .negative detail => .failed detail
  | .reportedDeviceCount count =>
      .countOnly count (.incompleteDeviceProfile count)

def availabilityFromRaw (raw : UInt64) : Availability DeviceCapability Detail :=
  availabilityFromProbe probeAuthority (observationFromRaw raw)

@[extern "lean_tgrad_hip_probe"]
opaque probeRaw : IO UInt64

def availability : IO (Availability DeviceCapability Detail) :=
  availabilityFromRaw <$> probeRaw

structure BufferHandle where
  raw : UInt64
  deriving BEq, Repr

structure KernelHandle where
  raw : UInt64
  deriving BEq, Repr

def runtimeFailure (errorClass : RuntimeErrorClass) (stage : RuntimeStage)
    (detail : Detail) : RuntimeFailure Detail :=
  { errorClass, stage, detail }

/-- Stable vendor status translation.  It records the exact HIP code but does
not reinterpret it as Tensor behavior. -/
def translateVendorResult (stage : RuntimeStage) (code : UInt32) :
    RuntimeResult Detail Unit :=
  if code == 0 then .ok ()
  else .error (runtimeFailure .vendorFailure stage (.vendorCode stage code))

private def unavailableFor (stage : RuntimeStage) :
    IO (RuntimeResult Detail α) := do
  match ← availability with
  | .unavailable _ reason =>
      pure (.error (runtimeFailure .unavailable stage reason.detail))
  | .available _ =>
      pure (.error (runtimeFailure .operationNotImplemented stage
        (.implementationMissing stage)))

/-- Honest no-hardware boundary.  Every operation rechecks the sealed local
availability path and returns a structured failure.  It never runs on CPU,
Metal, another backend, or fabricated caller facts. -/
def localBoundary : RuntimeBoundary DeviceCapability Detail BufferHandle
    KernelHandle Renderer.Hip.SourcePayload :=
  { capabilityContract
    renderer := Renderer.Hip.rendererContract
    availability
    allocate := fun _ => unavailableFor .allocation
    compile := fun _ _ => unavailableFor .compilation
    launch := fun _ _ _ => unavailableFor .launch
    synchronize := fun _ _ => unavailableFor .synchronization
    copyIn := fun _ _ _ => unavailableFor .copy
    copyOut := fun _ _ _ => unavailableFor .copy
    releaseBuffer := fun _ _ => unavailableFor .release
    releaseKernel := fun _ _ => unavailableFor .release }

private def isError : Except ε α → Bool
  | .error _ => true
  | .ok _ => false

/-- CPU-only audit of the exact shared capability, artifact, buffer, kernel,
and copy interfaces.  The private credential is used only inside this module;
it is not evidence of a real probe or hardware. -/
def staticInterfaceSelfCheck : Bool :=
  match Backend.Hip.buildProfile "gfx1100" .runtime "hiprtc" "rocm-static" 1024,
      Backend.Hip.buildProfile "gfx942" .runtime "hiprtc" "rocm-static" 1024 with
  | .ok profile, .ok otherProfile =>
      match Backend.Hip.buildFillPlan profile .float32_
          (.float32Bits 0x40500000) 257 256,
          Backend.Hip.buildFillPlan otherProfile .float32_
          (.float32Bits 0x40500000) 257 256 with
      | .ok plan, .ok otherPlan =>
          let observed := profileObservationFromProbe probeAuthority
            probeCredential 1 0 (.complete profile)
              (Detail.implementationMissing .availability)
          match availabilityFromProbe probeAuthority observed with
          | .unavailable _ _ => false
          | .available capability =>
              match AuthorizedPlan.build capabilityContract capability plan with
              | .error _ => false
              | .ok authorized =>
                  let artifact := Renderer.Hip.rendererContract.renderArtifact plan
                  match CompileRequest.build Renderer.Hip.rendererContract plan artifact with
                  | .error _ => false
                  | .ok request =>
                      let bufferArtifact : BufferArtifact BufferHandle := {
                        handle := { raw := 1 }
                        device := authorized.device
                        ownership := .runtimeOwned
                        byteCapacity := plan.byteCount }
                      let kernelArtifact : CompiledKernelArtifact KernelHandle := {
                        handle := { raw := 2 }
                        device := authorized.device
                        identity := plan.kernelIdentity }
                      match BoundBuffer.build authorized bufferArtifact,
                          BoundCompiledKernel.build authorized kernelArtifact with
                      | .ok buffer, .ok _kernel =>
                          let bytes := ByteArray.mk #[1, 2, 3, 4]
                          let copyIn := CopyInRequest.build bytes buffer.byteCapacity
                          let fourBytes : AbiU64 := { value := 4, fits := by decide }
                          let copyOut := CopyOutRequest.build fourBytes buffer.byteCapacity
                          request.renderedSemantics == some plan.renderedSemantics &&
                          !isError copyIn && !isError copyOut &&
                          isError (AuthorizedPlan.build capabilityContract capability otherPlan)
                      | _, _ => false
      | _, _ => false
  | _, _ => false

def sealedProbeSelfCheck : Bool :=
  let invalidProfile := do
    let architecture ← (ArchitectureIdentity.build "sm_80").toOption
    let tool ← (CompilerToolIdentity.build "hiprtc").toOption
    let version ← (CompilerVersionIdentity.build "rocm-static").toOption
    (DeviceProfile.build Backend.Hip.backendIdentity architecture {
      mode := .runtime, tool, version } 1024).toOption
  let invalidAvailability := invalidProfile.map (fun profile =>
    availabilityFromProbe probeAuthority (profileObservationFromProbe
      probeAuthority probeCredential 1 0 (.complete profile)
        (Detail.profileRejected (.invalidArchitecture "sm_80"))))
  match availabilityFromRaw 0, availabilityFromRaw 257, invalidAvailability with
  | .unavailable requestedMissing missing,
      .unavailable requestedCount countOnly,
      some (.unavailable _ invalidReason) =>
      requestedMissing == Backend.Hip.backendIdentity &&
      requestedCount == Backend.Hip.backendIdentity &&
      missing.reasonClass == .runtimeLibraryMissing &&
      countOnly.reasonClass == .countOnly &&
      invalidReason.reasonClass == .invalidDeviceProfile
  | _, _, _ => false

private def isUnavailableAt (stage : RuntimeStage) :
    RuntimeResult Detail α → Bool
  | .error failure =>
      failure.errorClass == .unavailable && failure.stage == stage
  | .ok _ => false

/-- Exercise real local entry points with a module-private synthetic authority.
The authority exists only to reach the dependent call edge; every operation
must still re-probe and return unavailable, never a host/other-backend result. -/
def noFallbackSelfCheck : IO Bool := do
  let .ok profile := Backend.Hip.buildProfile
      "gfx1100" .runtime "hiprtc" "rocm-static" 1024
    | pure false
  let .ok plan := Backend.Hip.buildFillPlan profile .int32_ (.signed 1) 1 1
    | pure false
  let observed := profileObservationFromProbe probeAuthority probeCredential
    1 0 (.complete profile) (Detail.implementationMissing .availability)
  let .available capability := availabilityFromProbe probeAuthority observed
    | pure false
  let .ok authorized := AuthorizedPlan.build capabilityContract capability plan
    | pure false
  let artifact := localBoundary.renderer.renderArtifact authorized.plan
  let .ok request := CompileRequest.build localBoundary.renderer
      authorized.plan artifact
    | pure false
  let allocated ← localBoundary.allocate authorized
  let compiled ← localBoundary.compile authorized request
  let synchronized ← localBoundary.synchronize capability .device
  pure (isUnavailableAt .allocation allocated &&
    isUnavailableAt .compilation compiled &&
    isUnavailableAt .synchronization synchronized)

end Tgrad.Runtime.Hip

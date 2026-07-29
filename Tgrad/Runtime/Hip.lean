import Tgrad.Renderer.Hip

/-! # Fail-closed HIP runtime boundary

The C bridge performs only dynamic runtime discovery and device enumeration.
All semantic planning remains in Lean.  Compilation and execution are explicit
typed operations but deliberately unavailable in this no-ROCm initial slice.
-/
namespace Tgrad.Runtime.Hip

open Tgrad.Backend

inductive UnavailableReason where
  | runtimeLibraryMissing
  | deviceQuerySymbolMissing
  | deviceQueryFailed
  | noDevices
  | incompleteDeviceProfile (deviceCount : Nat)
  | invalidDeviceProfile
  | malformedProbe (raw : UInt64)
  deriving BEq, Repr

structure DeviceCapability where
  private mk ::
  profile : BackendIdentity
  deviceOrdinal : Nat
  deviceCount : Nat
  deriving BEq, Repr

inductive Availability where
  | available (capability : DeviceCapability)
  | unavailable (reason : UnavailableReason)
  deriving BEq, Repr

def Availability.isAvailable : Availability → Bool
  | .available _ => true
  | .unavailable _ => false

/-- Public protocol parsing is descriptive only; it is not an execution
capability.  In particular, a caller passing `257` can describe the wire value
but cannot construct the private `DeviceCapability` consumed by execution. -/
inductive ProbeFact where
  | negative (reason : UnavailableReason)
  | reportedDeviceCount (count : Nat)
  deriving BEq, Repr

/-- Stable raw probe protocol: 0..3 are negative results; 256+n reports `n`
devices.  `n = 0` is malformed and must not become availability. -/
def classifyProbeRaw (raw : UInt64) : ProbeFact :=
  if raw == 0 then .negative .runtimeLibraryMissing
  else if raw == 1 then .negative .deviceQuerySymbolMissing
  else if raw == 2 then .negative .deviceQueryFailed
  else if raw == 3 then .negative .noDevices
  else if raw > 256 then .reportedDeviceCount (raw - 256).toNat
  else .negative (.malformedProbe raw)

private def availabilityFromRaw (raw : UInt64) : Availability :=
  match classifyProbeRaw raw with
  | .negative reason => .unavailable reason
  | .reportedDeviceCount count =>
      -- Device count alone cannot authorize a plan: architecture, compiler
      -- profile, ordinal and block limit have not been observed.
      .unavailable (.incompleteDeviceProfile count)

/-- Complete output required from a future device-profile probe.  The private
constructor prevents caller-supplied profile claims from minting execution
authority. -/
structure ProfiledProbeResult where
  private mk ::
  profile : BackendIdentity
  deviceOrdinal : Nat
  deviceCount : Nat

private def availabilityFromProfiledProbe
    (probe : ProfiledProbeResult) : Availability :=
  if probe.profile.backend != .hip ||
      !probe.profile.validArchitecture ||
      !probe.profile.compilerMatches ||
      probe.profile.compilerIdentity.isEmpty ||
      probe.profile.maxThreadsPerBlock = 0 ||
      probe.profile.maxThreadsPerBlock > 1024 ||
      probe.deviceCount = 0 || probe.deviceOrdinal >= probe.deviceCount then
    .unavailable .invalidDeviceProfile
  else
    .available {
      profile := probe.profile
      deviceOrdinal := probe.deviceOrdinal
      deviceCount := probe.deviceCount
    }

def DeviceCapability.authorizesPlan (capability : DeviceCapability)
    (plan : FillPlan) : Bool :=
  capability.profile == plan.identity &&
  capability.profile.backend == .hip &&
  capability.profile.validArchitecture &&
  capability.profile.compilerMatches &&
  capability.deviceCount > 0 &&
  capability.deviceOrdinal < capability.deviceCount &&
  plan.launch.blockSize ≤ capability.profile.maxThreadsPerBlock

/-- Static self-check for the private, probe-issued full-profile capability
contract.  It does not report hardware availability or expose a constructible
capability. -/
def capabilityCouplingSelfCheck : Bool :=
  let runtimeProfile : BackendIdentity :=
    { backend := .hip, architecture := "gfx1100", compilerMode := .runtime,
      compilerTool := .hiprtc, compilerIdentity := "rocm-v1",
      maxThreadsPerBlock := 1024 }
  let otherArch : BackendIdentity :=
    { runtimeProfile with architecture := "gfx942" }
  let otherCompiler : BackendIdentity :=
    { runtimeProfile with compilerMode := .offline, compilerTool := .hipcc }
  let otherLimit : BackendIdentity :=
    { runtimeProfile with maxThreadsPerBlock := 512 }
  let invalidLimit : BackendIdentity :=
    { runtimeProfile with maxThreadsPerBlock := 2048 }
  let issued := availabilityFromProfiledProbe {
    profile := runtimeProfile, deviceOrdinal := 0, deviceCount := 1 }
  let invalidIssued := availabilityFromProfiledProbe {
    profile := invalidLimit, deviceOrdinal := 0, deviceCount := 1 }
  match issued, invalidIssued,
        mkFillPlan runtimeProfile .int32_ (.signed 1) 257 256,
        mkFillPlan otherArch .int32_ (.signed 1) 257 256,
        mkFillPlan otherCompiler .int32_ (.signed 1) 257 256,
        mkFillPlan otherLimit .int32_ (.signed 1) 257 256 with
  | .available capability, .unavailable .invalidDeviceProfile,
      .ok matching, .ok wrongArch, .ok wrongCompiler, .ok wrongLimit =>
      capability.authorizesPlan matching &&
      !capability.authorizesPlan wrongArch &&
      !capability.authorizesPlan wrongCompiler &&
      !capability.authorizesPlan wrongLimit
  | _, _, _, _, _, _ => false

/-- Count-only success is deliberately insufficient to mint availability. -/
def incompleteProbeSelfCheck : Bool :=
  match availabilityFromRaw 257 with
  | .unavailable (.incompleteDeviceProfile 1) => true
  | _ => false

@[extern "lean_tgrad_hip_probe"]
opaque probeRaw : IO UInt64

def availability : IO Availability :=
  availabilityFromRaw <$> probeRaw

inductive BufferOwnership where
  | runtimeOwned
  deriving BEq, Repr

inductive CopyDirection where
  | hostToDevice | deviceToHost
  deriving BEq, Repr

inductive Synchronization where
  | device
  deriving BEq, Repr

inductive RuntimeStage where
  | availability | allocation | compilation | moduleLoad | launch
  | synchronization | copy | release
  deriving BEq, Repr

inductive RuntimeError where
  | unavailable (reason : UnavailableReason)
  | vendorError (stage : RuntimeStage) (code : UInt32)
  | invalidHandle
  | wrongBufferBackend (actual : BackendId)
  | invalidCopySize (requested capacity : Nat)
  | kernelBinding (reason : Renderer.Hip.RenderError)
  | capabilityProfileMismatch
  | operationNotImplemented (stage : RuntimeStage)
  deriving BEq, Repr

abbrev RuntimeResult (α : Type) := Except RuntimeError α

/-- Stable translation at the runtime boundary: HIP success is zero; every
nonzero code retains its stage and exact vendor value. -/
def translateVendorResult (stage : RuntimeStage) (code : UInt32) :
    RuntimeResult Unit :=
  if code == 0 then .ok () else .error (.vendorError stage code)

structure Buffer where
  raw : UInt64
  backend : BackendId
  ownership : BufferOwnership
  byteCount : Nat
  deriving BEq, Repr

structure CompiledKernel where
  raw : UInt64
  backend : BackendId
  identity : KernelIdentity
  deriving BEq, Repr

/-- The only compilation request type.  It binds a validated plan to the
private renderer artifact and rechecks exact source/kernel/cache/profile
linkage before runtime entry. -/
structure CompileRequest where
  private mk ::
  plan : FillPlan
  kernel : Renderer.Hip.KernelSource

def CompileRequest.build (plan : FillPlan)
    (kernel : Renderer.Hip.KernelSource) : RuntimeResult CompileRequest := do
  match kernel.validateForPlan plan with
  | .ok () => pure { plan, kernel }
  | .error reason => throw (.kernelBinding reason)

def validateTransfer (buffer : Buffer) (direction : CopyDirection)
    (requested : Nat) : RuntimeResult Unit := do
  if buffer.raw == 0 then throw .invalidHandle
  if buffer.backend != .hip then throw (.wrongBufferBackend buffer.backend)
  if requested > buffer.byteCount then
    throw (.invalidCopySize requested buffer.byteCount)
  match direction with
  | .hostToDevice | .deviceToHost => pure ()

/-- Explicit compile/launch interface.  Implementations receive the validated
plan used by the renderer and launch; byte counts and geometry are not loose
parallel arguments. -/
structure RuntimeBoundary where
  availability : IO Availability
  allocate : FillPlan → IO (RuntimeResult Buffer)
  compile : CompileRequest → IO (RuntimeResult CompiledKernel)
  launch : FillPlan → CompiledKernel → Buffer → IO (RuntimeResult Unit)
  synchronize : Synchronization → IO (RuntimeResult Unit)
  copy : Buffer → CopyDirection → Nat → IO (RuntimeResult ByteArray)
  releaseBuffer : Buffer → IO (RuntimeResult Unit)
  releaseKernel : CompiledKernel → IO (RuntimeResult Unit)

private def unavailableFor (stage : RuntimeStage) : IO (RuntimeResult α) := do
  match ← availability with
  | .unavailable reason => pure (.error (.unavailable reason))
  | .available _ => pure (.error (.operationNotImplemented stage))

private def unavailableForPlan (plan : FillPlan) (stage : RuntimeStage) :
    IO (RuntimeResult α) := do
  match ← availability with
  | .unavailable reason => pure (.error (.unavailable reason))
  | .available capability =>
      if capability.authorizesPlan plan then
        pure (.error (.operationNotImplemented stage))
      else pure (.error .capabilityProfileMismatch)

/-- Honest local boundary: it can probe, but it cannot compile or execute until
the vendor bridge exists.  An absent runtime always returns structured error. -/
def localBoundary : RuntimeBoundary :=
  { availability := availability
    allocate := fun plan => unavailableForPlan plan .allocation
    compile := fun request => unavailableForPlan request.plan .compilation
    launch := fun plan kernel buffer =>
      if kernel.backend != .hip || buffer.backend != .hip ||
          kernel.identity != plan.kernelIdentity then
        pure (.error .capabilityProfileMismatch)
      else unavailableForPlan plan .launch
    synchronize := fun _ => unavailableFor .synchronization
    copy := fun _ _ _ => unavailableFor .copy
    releaseBuffer := fun _ => unavailableFor .release
    releaseKernel := fun _ => unavailableFor .release }

end Tgrad.Runtime.Hip

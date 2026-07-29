import Tgrad.Renderer.Cuda

/-! # Fail-closed CUDA runtime boundary

No CUDA bridge is linked in this slice.  `localAvailability` therefore says
`probeAbsent`; it can never turn the host's Metal device (or a source renderer)
into CUDA availability.  A future C bridge must produce `ProbeResult` and then
pass through this same typed admission boundary.
-/
namespace Tgrad
namespace Runtime

inductive UnavailableReason where
  | probeAbsent
  | runtimeLibraryMissing
  | probeFailed (detail : String)
  | noDevice
  | invalidDeviceProfile
  deriving BEq, Repr

structure AvailableBackend where
  private mk ::
  profile : Backend.Profile
  deviceOrdinal : Nat
  deriving BEq, Repr

inductive Availability where
  | available (backend : AvailableBackend)
  | unavailable (requested : Backend.Id) (reason : UnavailableReason)
  deriving BEq, Repr

def Availability.isAvailable : Availability → Bool
  | .available _ => true
  | .unavailable _ _ => false

inductive StableRuntimeError where
  | unavailable (requested : Backend.Id) (reason : UnavailableReason)
  | invalidPlan (detail : String)
  | allocationFailure (detail : String)
  | compilationFailure (detail : String)
  | launchFailure (detail : String)
  | synchronizationFailure (detail : String)
  | copyFailure (detail : String)
  deriving BEq, Repr

abbrev Result (α : Type) := Except StableRuntimeError α

namespace Cuda

inductive BufferOwnership where
  | runtimeOwned
  | borrowed
  deriving BEq, Repr

structure DeviceBuffer where
  backend : Backend.Id
  deviceOrdinal : Nat
  ownership : BufferOwnership
  byteCount : Backend.AbiU64
  handle : UInt64

inductive CopyDirection where
  | hostToDevice
  | deviceToHost
  | deviceToDevice
  deriving BEq, Repr

structure CopyRequest where
  direction : CopyDirection
  byteCount : Backend.AbiU64
  deviceBuffer : DeviceBuffer

inductive SyncScope where
  | device
  | stream (handle : UInt64)
  deriving BEq, Repr

structure CompileRequest where
  plan : Backend.FillPlan
  kernel : Renderer.Cuda.KernelSource

structure CompiledModule where
  backend : Backend.Id
  deviceOrdinal : Nat
  cacheIdentity : String
  handle : UInt64

structure LaunchRequest where
  capability : AvailableBackend
  plan : Backend.FillPlan
  module : CompiledModule
  output : DeviceBuffer

structure SynchronizeRequest where
  capability : AvailableBackend
  scope : SyncScope

def CompileRequest.build (plan : Backend.FillPlan)
    (kernel : Renderer.Cuda.KernelSource) : Result CompileRequest :=
  match plan.revalidate with
  | .error reason => .error (.invalidPlan (reprStr reason))
  | .ok () =>
      match Renderer.Cuda.renderFill plan with
      | .error reason => .error (.invalidPlan (reprStr reason))
      | .ok expected =>
          if expected == kernel then .ok { plan, kernel }
          else .error (.invalidPlan "source/cache identity does not match validated plan")

def CopyRequest.build (direction : CopyDirection) (byteCount : Backend.AbiU64)
    (deviceBuffer : DeviceBuffer) : Result CopyRequest :=
  if deviceBuffer.backend != .cuda then
    .error (.copyFailure "copy buffer is not CUDA")
  else if deviceBuffer.handle == 0 then
    .error (.copyFailure "copy buffer handle is null")
  else if byteCount.value > deviceBuffer.byteCount.value then
    .error (.copyFailure "copy exceeds device buffer byte size")
  else .ok { direction, byteCount, deviceBuffer }

structure ProbeResult where
  private mk ::
  runtimeLoaded : Bool
  deviceCount : Option Nat
  profile : Option Backend.Profile
  deriving BEq, Repr

def ProbeResult.runtimeMissing : ProbeResult := {
  runtimeLoaded := false
  deviceCount := none
  profile := none
}

def availabilityFromProbe : Option ProbeResult → Availability
  | none => .unavailable .cuda .probeAbsent
  | some probe =>
      if !probe.runtimeLoaded then .unavailable .cuda .runtimeLibraryMissing
      else match probe.deviceCount with
        | none => .unavailable .cuda (.probeFailed "device count unavailable")
        | some 0 => .unavailable .cuda .noDevice
        | some (_ + 1) =>
            match probe.profile with
            | some profile =>
                if profile.backend == .cuda && profile.valid then
                  .available { profile, deviceOrdinal := 0 }
                else .unavailable .cuda .invalidDeviceProfile
            | none => .unavailable .cuda .invalidDeviceProfile

/-- Honest local state until an actual CUDA runtime bridge is linked. -/
def localAvailability : IO Availability :=
  pure (availabilityFromProbe none)

def requireAvailable : Availability → Result AvailableBackend
  | .available backend =>
      if backend.profile.backend == .cuda && backend.profile.valid then
        .ok backend
      else .error (.unavailable .cuda .invalidDeviceProfile)
  | .unavailable requested reason => .error (.unavailable requested reason)

def LaunchRequest.build (availability : Availability) (plan : Backend.FillPlan)
    (module : CompiledModule) (output : DeviceBuffer) : Result LaunchRequest := do
  let capability ← requireAvailable availability
  match plan.revalidate with
  | .error reason => throw (.invalidPlan (reprStr reason))
  | .ok () => pure ()
  if plan.profile != capability.profile then
    throw (.invalidPlan "plan profile differs from available CUDA capability")
  if module.backend != .cuda || output.backend != .cuda then
    throw (.launchFailure "module or output buffer is not CUDA")
  if module.deviceOrdinal != capability.deviceOrdinal ||
      output.deviceOrdinal != capability.deviceOrdinal then
    throw (.launchFailure "device ordinal mismatch")
  if module.handle == 0 || output.handle == 0 then
    throw (.launchFailure "module or output handle is null")
  let kernel ← match Renderer.Cuda.renderFill plan with
    | .ok kernel => pure kernel
    | .error reason => throw (.invalidPlan (reprStr reason))
  if module.cacheIdentity != kernel.cacheIdentity then
    throw (.launchFailure "compiled module identity differs from plan")
  if output.byteCount.value < plan.byteCount.value then
    throw (.launchFailure "output buffer is too small for plan")
  pure { capability, plan, module, output }

def SynchronizeRequest.build (availability : Availability) (scope : SyncScope) :
    Result SynchronizeRequest := do
  let capability ← requireAvailable availability
  match scope with
  | .device => pure { capability, scope }
  | .stream handle =>
      if handle == 0 then throw (.synchronizationFailure "stream handle is null")
      pure { capability, scope }

end Cuda
end Runtime
end Tgrad

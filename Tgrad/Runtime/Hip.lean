import Tgrad.Backend.FillPlan

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
  | malformedProbe (raw : UInt64)
  deriving BEq, Repr

structure DeviceIdentity where
  private mk ::
  ordinal : Nat
  deviceCount : Nat
  deriving BEq, Repr

inductive Availability where
  | available (device : DeviceIdentity)
  | unavailable (reason : UnavailableReason)
  deriving BEq, Repr

/-- Public protocol parsing is descriptive only; it is not an execution
capability.  In particular, a caller passing `257` can describe the wire value
but cannot construct the private `DeviceIdentity` consumed by execution. -/
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
      .available { ordinal := 0, deviceCount := count }

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
  cacheIdentity : String
  deriving BEq, Repr

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
  compile : FillPlan → String → IO (RuntimeResult CompiledKernel)
  launch : FillPlan → CompiledKernel → Buffer → IO (RuntimeResult Unit)
  synchronize : Synchronization → IO (RuntimeResult Unit)
  copy : Buffer → CopyDirection → Nat → IO (RuntimeResult ByteArray)
  releaseBuffer : Buffer → IO (RuntimeResult Unit)
  releaseKernel : CompiledKernel → IO (RuntimeResult Unit)

private def unavailableFor (stage : RuntimeStage) : IO (RuntimeResult α) := do
  match ← availability with
  | .unavailable reason => pure (.error (.unavailable reason))
  | .available _ => pure (.error (.operationNotImplemented stage))

/-- Honest local boundary: it can probe, but it cannot compile or execute until
the vendor bridge exists.  An absent runtime always returns structured error. -/
def localBoundary : RuntimeBoundary :=
  { availability := availability
    allocate := fun _ => unavailableFor .allocation
    compile := fun _ _ => unavailableFor .compilation
    launch := fun _ _ _ => unavailableFor .launch
    synchronize := fun _ => unavailableFor .synchronization
    copy := fun _ _ _ => unavailableFor .copy
    releaseBuffer := fun _ => unavailableFor .release
    releaseKernel := fun _ => unavailableFor .release }

end Tgrad.Runtime.Hip

import Tgrad.Renderer.Cuda

/-! # Fail-closed CUDA runtime leaf

No CUDA bridge is linked in this CPU/static slice. The leaf exposes the full
shared runtime operation shape, but local availability is exactly probe-absent
and every otherwise unreachable operation reports a structured missing
implementation. It never substitutes Metal, CPU, or another backend.
-/
namespace Tgrad.Runtime.Cuda

open Tgrad.Backend

private structure ProbeCredential where
  token : Unit

private def probeAuthority : ProbeAuthority ProbeCredential := {
  requestedBackend := Backend.Cuda.identity
  admitsProfile := fun _ profile => Backend.Cuda.profileAdmitted profile
}

inductive ProbeResult where
  | runtimeMissing
  | negative (detail : String)
  | failed (detail : String)
  | noDevice
  deriving BEq, Repr

def availabilityFromProbe : Option ProbeResult →
    Availability (AvailableCapability probeAuthority) String
  | none => Backend.availabilityFromProbe probeAuthority
      (.absent "CUDA probe is not linked")
  | some .runtimeMissing =>
      Backend.availabilityFromProbe probeAuthority
        (.runtimeMissing "CUDA runtime library missing")
  | some (.negative detail) =>
      Backend.availabilityFromProbe probeAuthority (.negative detail)
  | some (.failed detail) =>
      Backend.availabilityFromProbe probeAuthority (.failed detail)
  | some .noDevice =>
      Backend.availabilityFromProbe probeAuthority
        (.countOnly 0 "CUDA device count is zero")

def localAvailability : IO
    (Availability (AvailableCapability probeAuthority) String) :=
  pure (availabilityFromProbe none)

structure BufferHandle where
  private mk ::
  raw : UInt64
  deriving BEq, Repr

structure KernelHandle where
  private mk ::
  moduleHandle : UInt64
  functionHandle : UInt64
  deriving BEq, Repr

private def missingFailure (stage : RuntimeStage) : RuntimeFailure String := {
  errorClass := .operationNotImplemented
  stage
  detail := "CUDA runtime bridge is not linked"
}

private def missingResult (stage : RuntimeStage) :
    IO (RuntimeResult String α) :=
  pure (.error (missingFailure stage))

def runtime : RuntimeBoundary
    (AvailableCapability probeAuthority) String BufferHandle KernelHandle
    Renderer.Cuda.KernelSource := {
  capabilityContract := AvailableCapability.contract probeAuthority
  renderer := Renderer.Cuda.renderer
  availability := localAvailability
  allocate := fun _ => missingResult .allocation
  compile := fun _ _ => missingResult .compilation
  launch := fun _ _ _ => missingResult .launch
  synchronize := fun _ _ => missingResult .synchronization
  copyIn := fun _ _ _ => missingResult .copy
  copyOut := fun _ _ _ => missingResult .copy
  releaseBuffer := fun _ _ => missingResult .release
  releaseKernel := fun _ _ => missingResult .release
}

end Tgrad.Runtime.Cuda

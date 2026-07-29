import Tgrad.Dtype

/-! # Backend-neutral accelerator planning and runtime contract

This module is the semantic intersection extracted from two independently
accepted accelerator fill prototypes. It owns dtype/value admission, checked
byte and launch arithmetic, typed semantic/source identity, artifact binding,
and the types crossing an accelerator runtime boundary.

It intentionally knows no vendor dialect, architecture spelling, compiler
tool, probe protocol, handle type, or runtime implementation. Vendor leaves
validate their architecture/tool compatibility, render their dialect, mint
private capabilities from real probes, and implement this interface.
-/
namespace Tgrad.Backend

def maxAbiU32 : Nat := 4294967295
def maxAbiU64 : Nat := 18446744073709551615

structure AbiU32 where
  value : Nat
  fits : value ≤ maxAbiU32
  deriving Repr

structure AbiU64 where
  value : Nat
  fits : value ≤ maxAbiU64
  deriving Repr

inductive IdentityError where
  | emptyBackend
  | emptyArchitecture
  | emptyCompilerTool
  | emptyCompilerVersion
  | emptyRendererContract
  | invalidThreadLimit (limit : Nat)
  deriving BEq, Repr

structure BackendIdentity where
  private mk ::
  stableName : String
  deriving BEq, Repr

def BackendIdentity.build (name : String) : Except IdentityError BackendIdentity :=
  if name.trimAscii.isEmpty then .error .emptyBackend
  else .ok { stableName := name }

/-- Opaque architecture identity. Its vendor admissibility is a leaf concern. -/
structure ArchitectureIdentity where
  private mk ::
  stableName : String
  deriving BEq, Repr

def ArchitectureIdentity.build (name : String) :
    Except IdentityError ArchitectureIdentity :=
  if name.trimAscii.isEmpty then .error .emptyArchitecture
  else .ok { stableName := name }

inductive CompilerMode where
  | offline
  | runtime
  deriving BEq, Repr, DecidableEq

def CompilerMode.stableName : CompilerMode → String
  | .offline => "offline"
  | .runtime => "runtime"

structure CompilerToolIdentity where
  private mk ::
  stableName : String
  deriving BEq, Repr

def CompilerToolIdentity.build (name : String) :
    Except IdentityError CompilerToolIdentity :=
  if name.trimAscii.isEmpty then .error .emptyCompilerTool
  else .ok { stableName := name }

structure CompilerVersionIdentity where
  private mk ::
  stableName : String
  deriving BEq, Repr

def CompilerVersionIdentity.build (version : String) :
    Except IdentityError CompilerVersionIdentity :=
  if version.trimAscii.isEmpty then .error .emptyCompilerVersion
  else .ok { stableName := version }

structure CompilerIdentity where
  mode : CompilerMode
  tool : CompilerToolIdentity
  version : CompilerVersionIdentity
  deriving BEq, Repr

/-- A complete descriptive profile. Vendor leaves separately validate that its
opaque identities are compatible before minting an execution capability. -/
structure DeviceProfile where
  private mk ::
  backend : BackendIdentity
  architecture : ArchitectureIdentity
  compiler : CompilerIdentity
  maxThreadsPerBlock : AbiU32
  threadsPositive : 0 < maxThreadsPerBlock.value
  deriving Repr

def DeviceProfile.build (backend : BackendIdentity)
    (architecture : ArchitectureIdentity) (compiler : CompilerIdentity)
    (maxThreadsPerBlock : Nat) : Except IdentityError DeviceProfile :=
  if positive : 0 < maxThreadsPerBlock then
    if fits : maxThreadsPerBlock ≤ maxAbiU32 then
      .ok {
        backend
        architecture
        compiler
        maxThreadsPerBlock := { value := maxThreadsPerBlock, fits }
        threadsPositive := positive
      }
    else .error (.invalidThreadLimit maxThreadsPerBlock)
  else .error (.invalidThreadLimit maxThreadsPerBlock)

def DeviceProfile.beq (left right : DeviceProfile) : Bool :=
  left.backend == right.backend && left.architecture == right.architecture &&
  left.compiler == right.compiler &&
  left.maxThreadsPerBlock.value == right.maxThreadsPerBlock.value

instance : BEq DeviceProfile := ⟨DeviceProfile.beq⟩

/-- Exact device binding used by capabilities and runtime artifacts. -/
structure DeviceIdentity where
  profile : DeviceProfile
  ordinal : Nat
  deriving Repr

def DeviceIdentity.beq (left right : DeviceIdentity) : Bool :=
  left.profile == right.profile && left.ordinal == right.ordinal

instance : BEq DeviceIdentity := ⟨DeviceIdentity.beq⟩

structure Int32Value where
  value : Int
  lower : (-2147483648 : Int) ≤ value
  upper : value ≤ (2147483647 : Int)
  deriving Repr

inductive FillValue where
  | int32 (value : Int32Value)
  | float32Bits (bits : UInt32)
  deriving Repr

def FillValue.beq : FillValue → FillValue → Bool
  | .int32 left, .int32 right => left.value == right.value
  | .float32Bits left, .float32Bits right => left == right
  | _, _ => false

instance : BEq FillValue := ⟨FillValue.beq⟩

def FillValue.dtype : FillValue → Tgrad.Dtype
  | .int32 _ => .int32_
  | .float32Bits _ => .float32_

def FillValue.stableTag : FillValue → String
  | .int32 value => "i32:" ++ toString value.value
  | .float32Bits bits => "f32bits:" ++ toString bits.toNat

inductive ScalarInput where
  | signed (value : Int)
  | float32Bits (bits : UInt32)
  deriving BEq, Repr

inductive OutputIndexPolicy where
  | linearBlockThreadX
  deriving BEq, Repr, DecidableEq

def OutputIndexPolicy.stableName : OutputIndexPolicy → String
  | .linearBlockThreadX => "linear-block-thread-x"

inductive BoundsPolicy where
  | guardElementCount
  deriving BEq, Repr, DecidableEq

def BoundsPolicy.stableName : BoundsPolicy → String
  | .guardElementCount => "guard-element-count"

inductive PlanError where
  | unsupportedDtype (dtype : Tgrad.Dtype)
  | valueDtypeMismatch (dtype : Tgrad.Dtype) (input : ScalarInput)
  | int32OutOfRange (value : Int)
  | elementCountOutOfRange (count : Nat)
  | byteCountOverflow (count bytesPerElement : Nat)
  | invalidBlockSize (threads limit : Nat)
  | gridSizeOutOfRange (blocks : Nat)
  | capabilityProfileMismatch
  deriving BEq, Repr

def admitFillValue (dtype : Tgrad.Dtype) (input : ScalarInput) :
    Except PlanError FillValue :=
  match dtype, input with
  | .int32_, .signed value =>
      if lower : (-2147483648 : Int) ≤ value then
        if upper : value ≤ (2147483647 : Int) then
          .ok (.int32 { value, lower, upper })
        else .error (.int32OutOfRange value)
      else .error (.int32OutOfRange value)
  | .float32_, .float32Bits bits => .ok (.float32Bits bits)
  | .int32_, input | .float32_, input =>
      .error (.valueDtypeMismatch dtype input)
  | unsupported, _ => .error (.unsupportedDtype unsupported)

def ceilDiv (numer denom : Nat) : Nat :=
  if numer == 0 then 0 else ((numer - 1) / denom) + 1

structure BytePlan where
  elementCount : AbiU64
  bytesPerElement : Nat
  byteCount : AbiU64
  bytesExact : byteCount.value = elementCount.value * bytesPerElement

structure Launch1D (elementCount : Nat) where
  threadsPerBlock : AbiU32
  blocksPerGrid : AbiU32
  threadsPositive : 0 < threadsPerBlock.value
  blocksExact : blocksPerGrid.value = ceilDiv elementCount threadsPerBlock.value

structure FillPlan where
  private mk ::
  profile : DeviceProfile
  value : FillValue
  bytes : BytePlan
  launch : Launch1D bytes.elementCount.value
  outputIndexPolicy : OutputIndexPolicy
  boundsPolicy : BoundsPolicy
  threadsWithinProfile :
    launch.threadsPerBlock.value ≤ profile.maxThreadsPerBlock.value

def FillPlan.elementCount (plan : FillPlan) : AbiU64 := plan.bytes.elementCount
def FillPlan.byteCount (plan : FillPlan) : AbiU64 := plan.bytes.byteCount
def FillPlan.dtype (plan : FillPlan) : Tgrad.Dtype := plan.value.dtype

def FillPlan.build (profile : DeviceProfile) (dtype : Tgrad.Dtype)
    (input : ScalarInput) (elementCount threadsPerBlock : Nat) :
    Except PlanError FillPlan := do
  let value ← admitFillValue dtype input
  if countFits : elementCount ≤ maxAbiU64 then
    let bytes := elementCount * value.dtype.sizeBytes
    if bytesFit : bytes ≤ maxAbiU64 then
      if threadsPositive : 0 < threadsPerBlock then
        if threadsWithinProfile :
            threadsPerBlock ≤ profile.maxThreadsPerBlock.value then
          if threadsFit : threadsPerBlock ≤ maxAbiU32 then
            let blocks := ceilDiv elementCount threadsPerBlock
            if blocksFit : blocks ≤ maxAbiU32 then
              .ok {
                profile
                value
                bytes := {
                  elementCount := { value := elementCount, fits := countFits }
                  bytesPerElement := value.dtype.sizeBytes
                  byteCount := { value := bytes, fits := bytesFit }
                  bytesExact := rfl
                }
                launch := {
                  threadsPerBlock := { value := threadsPerBlock, fits := threadsFit }
                  blocksPerGrid := { value := blocks, fits := blocksFit }
                  threadsPositive
                  blocksExact := rfl
                }
                outputIndexPolicy := .linearBlockThreadX
                boundsPolicy := .guardElementCount
                threadsWithinProfile
              }
            else .error (.gridSizeOutOfRange blocks)
          else .error (.invalidBlockSize threadsPerBlock
            profile.maxThreadsPerBlock.value)
        else .error (.invalidBlockSize threadsPerBlock
          profile.maxThreadsPerBlock.value)
      else .error (.invalidBlockSize threadsPerBlock
        profile.maxThreadsPerBlock.value)
    else .error (.byteCountOverflow elementCount value.dtype.sizeBytes)
  else .error (.elementCountOutOfRange elementCount)

/-- Authoritative typed semantic/cache identity. Serialization is only a
derived boundary view; equality and binding use this value directly. -/
structure SemanticIdentity where
  profile : DeviceProfile
  value : FillValue
  outputIndexPolicy : OutputIndexPolicy
  boundsPolicy : BoundsPolicy
  elementCount : Nat
  bytesPerElement : Nat
  byteCount : Nat
  threadsPerBlock : Nat
  blocksPerGrid : Nat
  deriving Repr

def SemanticIdentity.beq (left right : SemanticIdentity) : Bool :=
  left.profile == right.profile && left.value == right.value &&
  left.outputIndexPolicy == right.outputIndexPolicy &&
  left.boundsPolicy == right.boundsPolicy &&
  left.elementCount == right.elementCount &&
  left.bytesPerElement == right.bytesPerElement &&
  left.byteCount == right.byteCount &&
  left.threadsPerBlock == right.threadsPerBlock &&
  left.blocksPerGrid == right.blocksPerGrid

instance : BEq SemanticIdentity := ⟨SemanticIdentity.beq⟩

def FillPlan.semanticIdentity (plan : FillPlan) : SemanticIdentity :=
  { profile := plan.profile
    value := plan.value
    outputIndexPolicy := plan.outputIndexPolicy
    boundsPolicy := plan.boundsPolicy
    elementCount := plan.elementCount.value
    bytesPerElement := plan.bytes.bytesPerElement
    byteCount := plan.byteCount.value
    threadsPerBlock := plan.launch.threadsPerBlock.value
    blocksPerGrid := plan.launch.blocksPerGrid.value }

private def identityField (name value : String) : String :=
  name ++ "=" ++ toString value.length ++ ":" ++ value

def SemanticIdentity.serialize (identity : SemanticIdentity) : String :=
  String.intercalate "|" [
    "schema=fill1d-v2",
    identityField "backend" identity.profile.backend.stableName,
    identityField "architecture" identity.profile.architecture.stableName,
    "compiler-mode=" ++ identity.profile.compiler.mode.stableName,
    identityField "compiler-tool" identity.profile.compiler.tool.stableName,
    identityField "compiler-version" identity.profile.compiler.version.stableName,
    "max-threads=" ++ toString identity.profile.maxThreadsPerBlock.value,
    "dtype=" ++ identity.value.dtype.toStr,
    "value=" ++ identity.value.stableTag,
    "index=" ++ identity.outputIndexPolicy.stableName,
    "bounds=" ++ identity.boundsPolicy.stableName,
    "count=" ++ toString identity.elementCount,
    "bytes-per-element=" ++ toString identity.bytesPerElement,
    "bytes=" ++ toString identity.byteCount,
    "threads=" ++ toString identity.threadsPerBlock,
    "blocks=" ++ toString identity.blocksPerGrid]

def FillPlan.cacheIdentity (plan : FillPlan) : String :=
  plan.semanticIdentity.serialize

private def identifierEncode (value : String) : String :=
  String.intercalate "_" (value.toList.map (fun char => toString char.toNat))

structure KernelIdentity where
  semantic : SemanticIdentity
  kernelName : String
  deriving Repr

def KernelIdentity.beq (left right : KernelIdentity) : Bool :=
  left.semantic == right.semantic && left.kernelName == right.kernelName

instance : BEq KernelIdentity := ⟨KernelIdentity.beq⟩

def FillPlan.kernelIdentity (plan : FillPlan) : KernelIdentity :=
  { semantic := plan.semanticIdentity
    kernelName := "tgrad_fill1d_" ++ identifierEncode plan.cacheIdentity }

def FillPlan.kernelName (plan : FillPlan) : String := plan.kernelIdentity.kernelName

structure RendererContractIdentity where
  private mk ::
  stableName : String
  deriving BEq, Repr

def RendererContractIdentity.build (name : String) :
    Except IdentityError RendererContractIdentity :=
  if name.trimAscii.isEmpty then .error .emptyRendererContract
  else .ok { stableName := name }

structure SourceIdentity where
  rendererContract : RendererContractIdentity
  kernel : KernelIdentity
  deriving Repr

def SourceIdentity.beq (left right : SourceIdentity) : Bool :=
  left.rendererContract == right.rendererContract && left.kernel == right.kernel

instance : BEq SourceIdentity := ⟨SourceIdentity.beq⟩

def FillPlan.sourceIdentity (plan : FillPlan)
    (rendererContract : RendererContractIdentity) : SourceIdentity :=
  { rendererContract, kernel := plan.kernelIdentity }

/-- A vendor leaf supplies a distinct payload type and a pure renderer. -/
structure RendererContract (Payload : Type) where
  identity : RendererContractIdentity
  render : FillPlan → Payload

structure SourceArtifactCandidate (Payload : Type) where
  identity : SourceIdentity
  payload : Payload

inductive ArtifactBindingError where
  | sourceIdentityMismatch
  | sourcePayloadMismatch
  deriving BEq, Repr

/-- Private, exact-plan source authority. Public candidate data is descriptive
only until both typed identity and leaf-rendered payload match. -/
structure SourceArtifact {Payload : Type} (renderer : RendererContract Payload)
    (plan : FillPlan) where
  private mk ::
  candidate : SourceArtifactCandidate Payload

def SourceArtifact.validateCandidate [DecidableEq Payload]
    (renderer : RendererContract Payload) (plan : FillPlan)
    (candidate : SourceArtifactCandidate Payload) :
    Except ArtifactBindingError (SourceArtifact renderer plan) := do
  if candidate.identity != plan.sourceIdentity renderer.identity then
    throw .sourceIdentityMismatch
  if candidate.payload = renderer.render plan then
    pure { candidate }
  else throw .sourcePayloadMismatch

def RendererContract.renderArtifact
    (renderer : RendererContract Payload) (plan : FillPlan) :
    SourceArtifact renderer plan :=
  { candidate := {
      identity := plan.sourceIdentity renderer.identity
      payload := renderer.render plan } }

def SourceArtifact.identity {Payload : Type}
    {renderer : RendererContract Payload} {plan : FillPlan}
    (artifact : SourceArtifact renderer plan) : SourceIdentity :=
  artifact.candidate.identity

def SourceArtifact.payload {Payload : Type}
    {renderer : RendererContract Payload} {plan : FillPlan}
    (artifact : SourceArtifact renderer plan) : Payload :=
  artifact.candidate.payload

/-- The only compile request accepted by the shared runtime shape. It rechecks
the private artifact against the exact plan and renderer contract. -/
structure CompileRequest {Payload : Type} (renderer : RendererContract Payload)
    (plan : FillPlan) where
  private mk ::
  artifact : SourceArtifact renderer plan

def CompileRequest.build [DecidableEq Payload] (renderer : RendererContract Payload)
    (plan : FillPlan) (artifact : SourceArtifact renderer plan) :
    Except ArtifactBindingError (CompileRequest renderer plan) := do
  let rebound ← SourceArtifact.validateCandidate renderer plan artifact.candidate
  pure { artifact := rebound }

def CompileRequest.sourceIdentity {Payload : Type}
    {renderer : RendererContract Payload} {plan : FillPlan}
    (request : CompileRequest renderer plan) : SourceIdentity :=
  request.artifact.identity

def CompileRequest.sourcePayload {Payload : Type}
    {renderer : RendererContract Payload} {plan : FillPlan}
    (request : CompileRequest renderer plan) : Payload :=
  request.artifact.payload

/-! ## Runtime-neutral intersection -/

inductive BufferOwnership where
  | runtimeOwned
  deriving BEq, Repr, DecidableEq

inductive CopyDirection where
  | hostToDevice
  | deviceToHost
  deriving BEq, Repr, DecidableEq

inductive SynchronizationScope where
  | device
  deriving BEq, Repr, DecidableEq

inductive RuntimeStage where
  | availability
  | allocation
  | compilation
  | moduleLoad
  | launch
  | synchronization
  | copy
  | release
  deriving BEq, Repr, DecidableEq

inductive UnavailableClass where
  | probeAbsent
  | runtimeLibraryMissing
  | probeFailed
  | noDevice
  | incompleteDeviceProfile
  | invalidDeviceProfile
  deriving BEq, Repr, DecidableEq

structure UnavailableReason (Detail : Type) where
  reasonClass : UnavailableClass
  detail : Detail
  deriving Repr

inductive RuntimeErrorClass where
  | unavailable
  | invalidPlan
  | capabilityProfileMismatch
  | artifactBindingFailure
  | invalidBuffer
  | allocationFailure
  | compilationFailure
  | moduleLoadFailure
  | launchFailure
  | synchronizationFailure
  | copyFailure
  | releaseFailure
  | operationNotImplemented
  | vendorFailure
  deriving BEq, Repr, DecidableEq

structure RuntimeFailure (Detail : Type) where
  errorClass : RuntimeErrorClass
  stage : RuntimeStage
  detail : Detail
  deriving Repr

abbrev RuntimeResult (Detail : Type) (α : Type) :=
  Except (RuntimeFailure Detail) α

inductive Availability (Capability Detail : Type) where
  | available (capability : Capability)
  | unavailable (requested : BackendIdentity)
      (reason : UnavailableReason Detail)
  deriving Repr

def Availability.isAvailable : Availability Capability Detail → Bool
  | .available _ => true
  | .unavailable _ _ => false

/-- A leaf exposes only this projection from its private capability. -/
structure CapabilityContract (Capability : Type) where
  deviceOf : Capability → DeviceIdentity

structure AuthorizedPlan {Capability : Type}
    (contract : CapabilityContract Capability) where
  private mk ::
  capability : Capability
  plan : FillPlan

def AuthorizedPlan.build (contract : CapabilityContract Capability)
    (capability : Capability) (plan : FillPlan) :
    Except PlanError (AuthorizedPlan contract) :=
  if (contract.deviceOf capability).profile == plan.profile then
    .ok { capability, plan }
  else .error .capabilityProfileMismatch

def AuthorizedPlan.device (authorized : AuthorizedPlan contract) : DeviceIdentity :=
  contract.deviceOf authorized.capability

/-- Leaf-private handles are wrapped with exact common metadata. -/
structure BufferArtifact (Handle : Type) where
  handle : Handle
  device : DeviceIdentity
  ownership : BufferOwnership
  byteCapacity : AbiU64

inductive BufferBindingError where
  | wrongDevice
  | insufficientCapacity (required capacity : Nat)
  deriving BEq, Repr

structure BoundBuffer {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    (authorized : AuthorizedPlan contract) where
  private mk ::
  artifact : BufferArtifact Handle

def BoundBuffer.build {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    (authorized : AuthorizedPlan contract)
    (artifact : BufferArtifact Handle) :
    Except BufferBindingError (BoundBuffer (Handle := Handle) authorized) := do
  if artifact.device != authorized.device then throw .wrongDevice
  if artifact.byteCapacity.value < authorized.plan.byteCount.value then
    throw (.insufficientCapacity authorized.plan.byteCount.value
      artifact.byteCapacity.value)
  pure { artifact }

def BoundBuffer.byteCapacity {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    {authorized : AuthorizedPlan contract}
    (buffer : BoundBuffer (Handle := Handle) authorized) : AbiU64 :=
  buffer.artifact.byteCapacity

def BoundBuffer.device {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    {authorized : AuthorizedPlan contract}
    (buffer : BoundBuffer (Handle := Handle) authorized) : DeviceIdentity :=
  buffer.artifact.device

def BoundBuffer.handle {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    {authorized : AuthorizedPlan contract}
    (buffer : BoundBuffer (Handle := Handle) authorized) : Handle :=
  buffer.artifact.handle

def BoundBuffer.ownership {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    {authorized : AuthorizedPlan contract}
    (buffer : BoundBuffer (Handle := Handle) authorized) : BufferOwnership :=
  buffer.artifact.ownership

structure CompiledKernelArtifact (Handle : Type) where
  handle : Handle
  device : DeviceIdentity
  identity : KernelIdentity

inductive CompiledKernelBindingError where
  | wrongDevice
  | kernelIdentityMismatch
  deriving BEq, Repr

structure BoundCompiledKernel {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    (authorized : AuthorizedPlan contract) where
  private mk ::
  artifact : CompiledKernelArtifact Handle

def BoundCompiledKernel.build {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    (authorized : AuthorizedPlan contract)
    (artifact : CompiledKernelArtifact Handle) :
    Except CompiledKernelBindingError
      (BoundCompiledKernel (Handle := Handle) authorized) := do
  if artifact.device != authorized.device then throw .wrongDevice
  if artifact.identity != authorized.plan.kernelIdentity then
    throw .kernelIdentityMismatch
  pure { artifact }

def BoundCompiledKernel.handle {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    {authorized : AuthorizedPlan contract}
    (kernel : BoundCompiledKernel (Handle := Handle) authorized) : Handle :=
  kernel.artifact.handle

def BoundCompiledKernel.identity {Capability Handle : Type}
    {contract : CapabilityContract Capability}
    {authorized : AuthorizedPlan contract}
    (kernel : BoundCompiledKernel (Handle := Handle) authorized) : KernelIdentity :=
  kernel.artifact.identity

structure CopyInRequest (capacity : AbiU64) where
  private mk ::
  bytes : ByteArray
  byteCount : AbiU64
  sizeExact : byteCount.value = bytes.size

structure CopyOutRequest (capacity : AbiU64) where
  private mk ::
  byteCount : AbiU64

inductive CopyRequestError where
  | hostByteCountOverflow (count : Nat)
  | exceedsCapacity (requested capacity : Nat)
  deriving BEq, Repr

def CopyInRequest.build (bytes : ByteArray) (capacity : AbiU64) :
    Except CopyRequestError (CopyInRequest capacity) :=
  if fits : bytes.size ≤ maxAbiU64 then
    if bytes.size ≤ capacity.value then
      .ok {
        bytes
        byteCount := { value := bytes.size, fits }
        sizeExact := rfl }
    else .error (.exceedsCapacity bytes.size capacity.value)
  else .error (.hostByteCountOverflow bytes.size)

def CopyOutRequest.build (byteCount capacity : AbiU64) :
    Except CopyRequestError (CopyOutRequest capacity) :=
  if byteCount.value ≤ capacity.value then .ok { byteCount }
  else .error (.exceedsCapacity byteCount.value capacity.value)

def CopyInRequest.direction (_request : CopyInRequest capacity) : CopyDirection :=
  .hostToDevice

def CopyInRequest.hostBytes (request : CopyInRequest capacity) : ByteArray :=
  request.bytes

def CopyInRequest.requestedByteCount
    (request : CopyInRequest capacity) : AbiU64 := request.byteCount

def CopyOutRequest.direction (_request : CopyOutRequest capacity) : CopyDirection :=
  .deviceToHost

def CopyOutRequest.requestedByteCount
    (request : CopyOutRequest capacity) : AbiU64 := request.byteCount

/-- A copy-out implementation must return exactly the requested bytes. -/
structure CopyOutResult {capacity : AbiU64}
    (request : CopyOutRequest capacity) where
  bytes : ByteArray
  sizeExact : bytes.size = request.byteCount.value

/-- The common runtime operation shape. Source payload and handles remain leaf
types. Compile, launch, copy, and release accept only exact renderer/plan/device
bound values; implementations remain wholly vendor-local. -/
structure RuntimeBoundary
    (Capability Detail BufferHandle KernelHandle SourcePayload : Type)
    [DecidableEq SourcePayload] where
  capabilityContract : CapabilityContract Capability
  renderer : RendererContract SourcePayload
  availability : IO (Availability Capability Detail)
  allocate : (authorized : AuthorizedPlan capabilityContract) →
    IO (RuntimeResult Detail (BoundBuffer (Handle := BufferHandle) authorized))
  compile : (authorized : AuthorizedPlan capabilityContract) →
    CompileRequest renderer authorized.plan →
    IO (RuntimeResult Detail
      (BoundCompiledKernel (Handle := KernelHandle) authorized))
  launch : (authorized : AuthorizedPlan capabilityContract) →
    BoundCompiledKernel (Handle := KernelHandle) authorized →
    BoundBuffer (Handle := BufferHandle) authorized →
    IO (RuntimeResult Detail Unit)
  synchronize : Capability → SynchronizationScope →
    IO (RuntimeResult Detail Unit)
  copyIn : (authorized : AuthorizedPlan capabilityContract) →
    (buffer : BoundBuffer (Handle := BufferHandle) authorized) →
    CopyInRequest buffer.byteCapacity → IO (RuntimeResult Detail Unit)
  copyOut : (authorized : AuthorizedPlan capabilityContract) →
    (buffer : BoundBuffer (Handle := BufferHandle) authorized) →
    (request : CopyOutRequest buffer.byteCapacity) →
    IO (RuntimeResult Detail (CopyOutResult request))
  releaseBuffer : (authorized : AuthorizedPlan capabilityContract) →
    BoundBuffer (Handle := BufferHandle) authorized →
    IO (RuntimeResult Detail Unit)
  releaseKernel : (authorized : AuthorizedPlan capabilityContract) →
    BoundCompiledKernel (Handle := KernelHandle) authorized →
    IO (RuntimeResult Detail Unit)

end Tgrad.Backend

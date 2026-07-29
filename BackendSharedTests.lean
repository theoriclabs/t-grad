import Tgrad.Backend.FillPlan

set_option maxRecDepth 2048

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

def getIdentity (builder : Except IdentityError α) (label : String) : IO α :=
  match builder with
  | .ok value => pure value
  | .error reason =>
      throw (IO.userError (label ++ " construction failed: " ++ reprStr reason))

def getPlan (profile : DeviceProfile) (dtype : Dtype) (input : ScalarInput)
    (count threads : Nat) : IO FillPlan :=
  match FillPlan.build profile dtype input count threads with
  | .ok plan => pure plan
  | .error reason =>
      throw (IO.userError ("plan construction failed: " ++ reprStr reason))

def hasLaunch (profile : DeviceProfile) (count threads blocks : Nat) : Bool :=
  match FillPlan.build profile .float32_ (.float32Bits 1065353216)
      count threads with
  | .ok plan => plan.launch.blocksPerGrid.value == blocks
  | .error _ => false

/-! Two intentionally distinct neutral witness leaves. They share only the
public plan/artifact/runtime contract under test. -/

structure WitnessCapabilityA where
  device : DeviceIdentity

structure WitnessCapabilityB where
  device : DeviceIdentity

def capabilityContractA : CapabilityContract WitnessCapabilityA :=
  { deviceOf := WitnessCapabilityA.device }

def capabilityContractB : CapabilityContract WitnessCapabilityB :=
  { deviceOf := WitnessCapabilityB.device }

structure WitnessSourceA where
  declaration : String
  deriving BEq, Repr, DecidableEq

structure WitnessSourceB where
  words : List Nat
  deriving BEq, Repr, DecidableEq

/-- A deliberately untrustworthy Bool equality. Artifact admission must use
propositional DecidableEq instead, so distinct payloads remain distinguishable. -/
structure LawlessPayload where
  value : String
  deriving Repr, DecidableEq

instance : BEq LawlessPayload := ⟨fun _ _ => true⟩

def rendererA (identity : RendererContractIdentity) :
    RendererContract WitnessSourceA :=
  { identity
    render := fun plan => {
      declaration := plan.kernelName ++ ":" ++ plan.value.stableTag
    } }

def rendererB (identity : RendererContractIdentity) :
    RendererContract WitnessSourceB :=
  { identity
    render := fun plan => {
      words := [plan.elementCount.value, plan.byteCount.value,
        plan.launch.threadsPerBlock.value, plan.launch.blocksPerGrid.value]
    } }

structure BufferHandleA where raw : Nat deriving BEq, Repr
structure BufferHandleB where token : String deriving BEq, Repr
structure KernelHandleA where raw : Nat deriving BEq, Repr
structure KernelHandleB where token : String deriving BEq, Repr

def runtimeFailure (errorClass : RuntimeErrorClass) (stage : RuntimeStage)
    (detail : String) : RuntimeFailure String :=
  { errorClass, stage, detail }

def zeroCopyOut (request : CopyOutRequest capacity) : CopyOutResult request :=
  { bytes := ByteArray.mk (Array.replicate request.byteCount.value 0)
    sizeExact := by
      change (Array.replicate request.byteCount.value 0).size = _
      exact Array.size_replicate }

def boundaryA (renderer : RendererContract WitnessSourceA)
    (capability : WitnessCapabilityA) :
    RuntimeBoundary WitnessCapabilityA String BufferHandleA KernelHandleA
      WitnessSourceA :=
  { capabilityContract := capabilityContractA
    renderer
    availability := pure (.available capability)
    allocate := fun authorized =>
      match BoundBuffer.build authorized {
          handle := { raw := 11 }
          device := authorized.device
          ownership := .runtimeOwned
          byteCapacity := authorized.plan.byteCount } with
      | .ok buffer => pure (.ok buffer)
      | .error reason => pure (.error
          (runtimeFailure .invalidBuffer .allocation (reprStr reason)))
    compile := fun authorized request =>
      let payload := request.sourcePayload
      if request.sourceIdentity != authorized.plan.sourceIdentity renderer.identity ||
          payload.declaration.isEmpty then
        pure (.error (runtimeFailure .artifactBindingFailure .compilation
          "witness A source projection mismatch"))
      else
        match BoundCompiledKernel.build authorized {
            handle := { raw := payload.declaration.length + 1 }
            device := authorized.device
            identity := authorized.plan.kernelIdentity } with
        | .ok kernel => pure (.ok kernel)
        | .error reason => pure (.error
            (runtimeFailure .artifactBindingFailure .compilation (reprStr reason)))
    launch := fun authorized kernel buffer =>
      if kernel.handle.raw == 0 || buffer.handle.raw == 0 ||
          buffer.ownership != .runtimeOwned ||
          kernel.identity != authorized.plan.kernelIdentity then
        pure (.error (runtimeFailure .launchFailure .launch
          "witness A projected launch binding mismatch"))
      else pure (.ok ())
    synchronize := fun _ _ => pure (.ok ())
    copyIn := fun _ _ request =>
      if request.direction != .hostToDevice || request.hostBytes.isEmpty ||
          request.requestedByteCount.value != request.hostBytes.size then
        pure (.error (runtimeFailure .copyFailure .copy
          "witness A copy-in projection mismatch"))
      else pure (.ok ())
    copyOut := fun _ _ request => pure (.ok (zeroCopyOut request))
    releaseBuffer := fun _ buffer =>
      if buffer.handle.raw == 0 then
        pure (.error (runtimeFailure .releaseFailure .release
          "witness A buffer handle missing"))
      else pure (.ok ())
    releaseKernel := fun _ kernel =>
      if kernel.handle.raw == 0 then
        pure (.error (runtimeFailure .releaseFailure .release
          "witness A kernel handle missing"))
      else pure (.ok ()) }

def boundaryB (renderer : RendererContract WitnessSourceB)
    (capability : WitnessCapabilityB) :
    RuntimeBoundary WitnessCapabilityB String BufferHandleB KernelHandleB
      WitnessSourceB :=
  { capabilityContract := capabilityContractB
    renderer
    availability := pure (.available capability)
    allocate := fun authorized =>
      match BoundBuffer.build authorized {
          handle := { token := "buffer-b" }
          device := authorized.device
          ownership := .runtimeOwned
          byteCapacity := authorized.plan.byteCount } with
      | .ok buffer => pure (.ok buffer)
      | .error reason => pure (.error
          (runtimeFailure .invalidBuffer .allocation (reprStr reason)))
    compile := fun authorized request =>
      let payload := request.sourcePayload
      if request.sourceIdentity != authorized.plan.sourceIdentity renderer.identity ||
          payload.words.isEmpty then
        pure (.error (runtimeFailure .artifactBindingFailure .compilation
          "witness B source projection mismatch"))
      else
        match BoundCompiledKernel.build authorized {
            handle := { token := "kernel-b-" ++ toString payload.words.length }
            device := authorized.device
            identity := authorized.plan.kernelIdentity } with
        | .ok kernel => pure (.ok kernel)
        | .error reason => pure (.error
            (runtimeFailure .artifactBindingFailure .compilation (reprStr reason)))
    launch := fun authorized kernel buffer =>
      if kernel.handle.token.isEmpty || buffer.handle.token.isEmpty ||
          buffer.ownership != .runtimeOwned ||
          kernel.identity != authorized.plan.kernelIdentity then
        pure (.error (runtimeFailure .launchFailure .launch
          "witness B projected launch binding mismatch"))
      else pure (.ok ())
    synchronize := fun _ _ => pure (.ok ())
    copyIn := fun _ _ request =>
      if request.direction != .hostToDevice || request.hostBytes.isEmpty ||
          request.requestedByteCount.value != request.hostBytes.size then
        pure (.error (runtimeFailure .copyFailure .copy
          "witness B copy-in projection mismatch"))
      else pure (.ok ())
    copyOut := fun _ _ request => pure (.ok (zeroCopyOut request))
    releaseBuffer := fun _ buffer =>
      if buffer.handle.token.isEmpty then
        pure (.error (runtimeFailure .releaseFailure .release
          "witness B buffer handle missing"))
      else pure (.ok ())
    releaseKernel := fun _ kernel =>
      if kernel.handle.token.isEmpty then
        pure (.error (runtimeFailure .releaseFailure .release
          "witness B kernel handle missing"))
      else pure (.ok ()) }

def main : IO Unit := do
  let backendA ← getIdentity (BackendIdentity.build "accelerator-a") "backend"
  let backendB ← getIdentity (BackendIdentity.build "accelerator-b") "backend"
  let archA ← getIdentity (ArchitectureIdentity.build "architecture-a") "architecture"
  let archB ← getIdentity (ArchitectureIdentity.build "architecture-b") "architecture"
  let toolA ← getIdentity (CompilerToolIdentity.build "tool-a") "compiler tool"
  let toolB ← getIdentity (CompilerToolIdentity.build "tool-b") "compiler tool"
  let versionA ← getIdentity (CompilerVersionIdentity.build "1.0") "compiler version"
  let versionB ← getIdentity (CompilerVersionIdentity.build "2.0") "compiler version"
  let rendererIdentityA ← getIdentity
    (RendererContractIdentity.build "witness-renderer-a-v1") "renderer"
  let rendererIdentityB ← getIdentity
    (RendererContractIdentity.build "witness-renderer-b-v1") "renderer"
  let profile ← getIdentity
    (DeviceProfile.build backendA archA {
      mode := .runtime, tool := toolA, version := versionA } 1024) "profile"
  let otherBackend ← getIdentity
    (DeviceProfile.build backendB archA {
      mode := .runtime, tool := toolA, version := versionA } 1024) "profile"
  let otherArch ← getIdentity
    (DeviceProfile.build backendA archB {
      mode := .runtime, tool := toolA, version := versionA } 1024) "profile"
  let otherMode ← getIdentity
    (DeviceProfile.build backendA archA {
      mode := .offline, tool := toolA, version := versionA } 1024) "profile"
  let otherTool ← getIdentity
    (DeviceProfile.build backendA archA {
      mode := .runtime, tool := toolB, version := versionA } 1024) "profile"
  let otherVersion ← getIdentity
    (DeviceProfile.build backendA archA {
      mode := .runtime, tool := toolA, version := versionB } 1024) "profile"
  let otherLimit ← getIdentity
    (DeviceProfile.build backendA archA {
      mode := .runtime, tool := toolA, version := versionA } 512) "profile"

  let floatPlan ← getPlan profile .float32_ (.float32Bits 1065353216) 513 256
  let intPlan ← getPlan profile .int32_ (.signed (-7)) 17 16
  let backendPlan ← getPlan otherBackend .float32_ (.float32Bits 1065353216) 513 256
  let archPlan ← getPlan otherArch .float32_ (.float32Bits 1065353216) 513 256
  let modePlan ← getPlan otherMode .float32_ (.float32Bits 1065353216) 513 256
  let toolPlan ← getPlan otherTool .float32_ (.float32Bits 1065353216) 513 256
  let versionPlan ← getPlan otherVersion .float32_ (.float32Bits 1065353216) 513 256
  let limitPlan ← getPlan otherLimit .float32_ (.float32Bits 1065353216) 513 256
  let valuePlan ← getPlan profile .float32_ (.float32Bits 2147483648) 513 256
  let launchPlan ← getPlan profile .float32_ (.float32Bits 1065353216) 513 128
  let identityPlans := [backendPlan, archPlan, modePlan, toolPlan, versionPlan,
    limitPlan, valuePlan, launchPlan]

  let mut failures := 0
  failures := failures + (← check "identity rejects empty backend"
    (isError (BackendIdentity.build "  ")))
  failures := failures + (← check "identity rejects empty architecture"
    (isError (ArchitectureIdentity.build "")))
  failures := failures + (← check "identity rejects empty compiler tool"
    (isError (CompilerToolIdentity.build "\t")))
  failures := failures + (← check "identity rejects empty compiler version"
    (isError (CompilerVersionIdentity.build "")))
  failures := failures + (← check "identity rejects empty renderer contract"
    (isError (RendererContractIdentity.build "")))
  failures := failures + (← check "profile rejects zero thread limit"
    (isError (DeviceProfile.build backendA archA {
      mode := .runtime, tool := toolA, version := versionA } 0)))
  failures := failures + (← check "profile rejects ABI-overflow thread limit"
    (isError (DeviceProfile.build backendA archA {
      mode := .runtime, tool := toolA, version := versionA } (maxAbiU32 + 1))))

  failures := failures + (← check "float32 exact bits admitted"
    (floatPlan.dtype == .float32_ && floatPlan.value.stableTag == "f32bits:1065353216"))
  failures := failures + (← check "int32 exact value admitted"
    (intPlan.dtype == .int32_ && intPlan.value.stableTag == "i32:-7"))
  failures := failures + (← check "exact byte product"
    (floatPlan.elementCount.value == 513 && floatPlan.bytes.bytesPerElement == 4 &&
     floatPlan.byteCount.value == 2052))
  failures := failures + (← check "exact ceil-div launch boundaries"
    (hasLaunch profile 0 256 0 && hasLaunch profile 1 256 1 &&
     hasLaunch profile 255 256 1 && hasLaunch profile 256 256 1 &&
     hasLaunch profile 257 256 2 && hasLaunch profile 512 256 2 &&
     hasLaunch profile 513 256 3))
  failures := failures + (← check "typed index and bounds policies"
    (floatPlan.outputIndexPolicy == .linearBlockThreadX &&
     floatPlan.boundsPolicy == .guardElementCount))
  failures := failures + (← check "unsupported dtype rejected"
    (isError (FillPlan.build profile .float64_ (.float32Bits 0) 1 1)))
  failures := failures + (← check "dtype value mismatch rejected"
    (isError (FillPlan.build profile .int32_ (.float32Bits 0) 1 1)))
  failures := failures + (← check "int32 underflow rejected"
    (isError (FillPlan.build profile .int32_ (.signed (-2147483649)) 1 1)))
  failures := failures + (← check "int32 overflow rejected"
    (isError (FillPlan.build profile .int32_ (.signed 2147483648) 1 1)))
  failures := failures + (← check "zero block rejected"
    (isError (FillPlan.build profile .float32_ (.float32Bits 0) 1 0)))
  failures := failures + (← check "device block limit rejected"
    (isError (FillPlan.build profile .float32_ (.float32Bits 0) 1 1025)))
  failures := failures + (← check "element ABI overflow rejected"
    (isError (FillPlan.build profile .float32_ (.float32Bits 0) (maxAbiU64 + 1) 1)))
  failures := failures + (← check "byte product overflow rejected"
    (isError (FillPlan.build profile .float32_ (.float32Bits 0) maxAbiU64 1)))
  failures := failures + (← check "launch width overflow rejected"
    (isError (FillPlan.build profile .float32_ (.float32Bits 0)
      (maxAbiU32 + 1) 1)))

  failures := failures + (← check "typed semantic identity separates every axis"
    (identityPlans.all (fun candidate =>
      candidate.semanticIdentity != floatPlan.semanticIdentity)))
  failures := failures + (← check "derived cache view separates every axis"
    (identityPlans.all (fun candidate => candidate.cacheIdentity != floatPlan.cacheIdentity)))
  failures := failures + (← check "kernel identity contains typed semantic authority"
    (floatPlan.kernelIdentity.semantic == floatPlan.semanticIdentity &&
     identityPlans.all (fun candidate =>
       candidate.kernelIdentity != floatPlan.kernelIdentity)))

  let delimiterBackend ← getIdentity
    (BackendIdentity.build "a|architecture=1:b") "backend"
  let delimiterArch ← getIdentity
    (ArchitectureIdentity.build "b|compiler-mode=runtime") "architecture"
  let delimiterProfile ← getIdentity
    (DeviceProfile.build delimiterBackend archA {
      mode := .runtime, tool := toolA, version := versionA } 1024) "profile"
  let delimiterProfile2 ← getIdentity
    (DeviceProfile.build backendA delimiterArch {
      mode := .runtime, tool := toolA, version := versionA } 1024) "profile"
  let delimiterPlan ← getPlan delimiterProfile .float32_
    (.float32Bits 1065353216) 513 256
  let delimiterPlan2 ← getPlan delimiterProfile2 .float32_
    (.float32Bits 1065353216) 513 256
  failures := failures + (← check "derived identity view resists delimiters"
    (delimiterPlan.cacheIdentity != delimiterPlan2.cacheIdentity))

  let leafRendererA := rendererA rendererIdentityA
  let leafRendererB := rendererB rendererIdentityB
  let artifactA := leafRendererA.renderArtifact floatPlan
  let artifactB := leafRendererB.renderArtifact floatPlan
  failures := failures + (← check "distinct leaves share typed semantic plan"
    (artifactA.identity.kernel.semantic == artifactB.identity.kernel.semantic &&
     artifactA.identity.rendererContract != artifactB.identity.rendererContract))
  let forgedA : SourceArtifactCandidate WitnessSourceA := {
    identity := artifactA.identity
    payload := { artifactA.payload with
      declaration := artifactA.payload.declaration ++ ":forged" }
  }
  failures := failures + (← check "arbitrary source payload rejected before compile"
    (isError (SourceArtifact.validateCandidate leafRendererA floatPlan forgedA)))
  let lawlessRenderer : RendererContract LawlessPayload := {
    identity := rendererIdentityA
    render := fun _ => { value := "expected" }
  }
  let lawlessCandidate : SourceArtifactCandidate LawlessPayload := {
    identity := floatPlan.sourceIdentity rendererIdentityA
    payload := { value := "forged" }
  }
  failures := failures + (← check "artifact binding uses propositional equality"
    (isError (SourceArtifact.validateCandidate lawlessRenderer floatPlan
      lawlessCandidate)))
  let wrongIdentityA : SourceArtifactCandidate WitnessSourceA := {
    identity := intPlan.sourceIdentity rendererIdentityA
    payload := leafRendererA.render floatPlan
  }
  failures := failures + (← check "wrong-plan source identity rejected before compile"
    (isError (SourceArtifact.validateCandidate leafRendererA floatPlan wrongIdentityA)))
  let compileRequestA := CompileRequest.build leafRendererA floatPlan artifactA
  let compileRequestB := CompileRequest.build leafRendererB floatPlan artifactB
  failures := failures + (← check "exact bound artifacts admit compile requests"
    (!isError compileRequestA && !isError compileRequestB))

  let device : DeviceIdentity := { profile, ordinal := 0 }
  let otherDevice : DeviceIdentity := { profile, ordinal := 1 }
  let capabilityA : WitnessCapabilityA := { device }
  let capabilityB : WitnessCapabilityB := { device }
  let authorizationA := AuthorizedPlan.build capabilityContractA capabilityA floatPlan
  let authorizationB := AuthorizedPlan.build capabilityContractB capabilityB floatPlan
  let mismatchedCapability : WitnessCapabilityA := {
    device := { profile := otherArch, ordinal := 0 } }
  failures := failures + (← check "capabilities bind exact plan profile"
    (!isError authorizationA && !isError authorizationB &&
     isError (AuthorizedPlan.build capabilityContractA mismatchedCapability floatPlan)))
  let .ok authorizedA := authorizationA
    | throw (IO.userError "witness A authorization unexpectedly rejected")
  let .ok authorizedB := authorizationB
    | throw (IO.userError "witness B authorization unexpectedly rejected")

  let wrongDeviceBuffer : BufferArtifact BufferHandleA := {
    handle := { raw := 1 }, device := otherDevice, ownership := .runtimeOwned,
    byteCapacity := floatPlan.byteCount }
  let tooSmall : AbiU64 := { value := 2051, fits := by decide }
  let smallBuffer : BufferArtifact BufferHandleA := {
    handle := { raw := 1 }, device, ownership := .runtimeOwned,
    byteCapacity := tooSmall }
  failures := failures + (← check "buffer binding rejects wrong device and capacity"
    (isError (BoundBuffer.build authorizedA wrongDeviceBuffer) &&
     isError (BoundBuffer.build authorizedA smallBuffer)))
  let wrongDeviceKernel : CompiledKernelArtifact KernelHandleA := {
    handle := { raw := 1 }, device := otherDevice,
    identity := floatPlan.kernelIdentity }
  let wrongIdentityKernel : CompiledKernelArtifact KernelHandleA := {
    handle := { raw := 1 }, device, identity := intPlan.kernelIdentity }
  failures := failures + (← check "compiled binding rejects wrong device and plan"
    (isError (BoundCompiledKernel.build authorizedA wrongDeviceKernel) &&
     isError (BoundCompiledKernel.build authorizedA wrongIdentityKernel)))

  let incompleteAvailability : Availability WitnessCapabilityA String :=
    .unavailable backendA {
      reasonClass := .incompleteDeviceProfile
      detail := "count-only probe" }
  failures := failures + (← check "count-only availability remains structured unavailable"
    (!incompleteAvailability.isAvailable))
  failures := failures + (← check "stable error vocabulary includes binding and release"
    (RuntimeErrorClass.artifactBindingFailure != .invalidBuffer &&
     RuntimeErrorClass.releaseFailure != .operationNotImplemented &&
     RuntimeStage.release != .moduleLoad))

  let runtimeA := boundaryA leafRendererA capabilityA
  let runtimeB := boundaryB leafRendererB capabilityB
  let runtimeArtifactA := runtimeA.renderer.renderArtifact authorizedA.plan
  let runtimeArtifactB := runtimeB.renderer.renderArtifact authorizedB.plan
  let .ok requestA := CompileRequest.build runtimeA.renderer authorizedA.plan
      runtimeArtifactA
    | throw (IO.userError "witness A compile request unexpectedly rejected")
  let .ok requestB := CompileRequest.build runtimeB.renderer authorizedB.plan
      runtimeArtifactB
    | throw (IO.userError "witness B compile request unexpectedly rejected")
  let allocatedA ← runtimeA.allocate authorizedA
  let allocatedB ← runtimeB.allocate authorizedB
  let compiledA ← runtimeA.compile authorizedA requestA
  let compiledB ← runtimeB.compile authorizedB requestB
  let .ok bufferA := allocatedA
    | throw (IO.userError "witness A allocation unexpectedly failed")
  let .ok bufferB := allocatedB
    | throw (IO.userError "witness B allocation unexpectedly failed")
  let .ok kernelA := compiledA
    | throw (IO.userError "witness A compilation unexpectedly failed")
  let .ok kernelB := compiledB
    | throw (IO.userError "witness B compilation unexpectedly failed")
  let hostBytes := ByteArray.mk #[1, 2, 3, 4]
  let copyInA := CopyInRequest.build hostBytes bufferA.byteCapacity
  let copyInB := CopyInRequest.build hostBytes bufferB.byteCapacity
  let fourBytes : AbiU64 := { value := 4, fits := by decide }
  let copyOutA := CopyOutRequest.build fourBytes bufferA.byteCapacity
  let copyOutB := CopyOutRequest.build fourBytes bufferB.byteCapacity
  let tooLarge : AbiU64 := { value := 2053, fits := by decide }
  let oversizedHost := ByteArray.mk (Array.replicate 2053 1)
  failures := failures + (← check "copy directions are request types"
    (match copyInA, copyOutA with
     | .ok input, .ok output =>
         input.direction == .hostToDevice && output.direction == .deviceToHost
     | _, _ => false))
  failures := failures + (← check "copy payload and sizes checked against bound buffers"
    (!isError copyInA && !isError copyInB && !isError copyOutA && !isError copyOutB &&
     isError (CopyInRequest.build oversizedHost bufferA.byteCapacity) &&
     isError (CopyOutRequest.build tooLarge bufferA.byteCapacity)))
  let .ok inputA := copyInA
    | throw (IO.userError "witness A copy-in unexpectedly rejected")
  let .ok inputB := copyInB
    | throw (IO.userError "witness B copy-in unexpectedly rejected")
  let .ok outputA := copyOutA
    | throw (IO.userError "witness A copy-out unexpectedly rejected")
  let .ok outputB := copyOutB
    | throw (IO.userError "witness B copy-out unexpectedly rejected")
  let launchedA ← runtimeA.launch authorizedA kernelA bufferA
  let launchedB ← runtimeB.launch authorizedB kernelB bufferB
  let copiedInA ← runtimeA.copyIn authorizedA bufferA inputA
  let copiedInB ← runtimeB.copyIn authorizedB bufferB inputB
  let copiedOutA ← runtimeA.copyOut authorizedA bufferA outputA
  let copiedOutB ← runtimeB.copyOut authorizedB bufferB outputB
  let releasedBufferA ← runtimeA.releaseBuffer authorizedA bufferA
  let releasedKernelA ← runtimeA.releaseKernel authorizedA kernelA
  let releasedBufferB ← runtimeB.releaseBuffer authorizedB bufferB
  let releasedKernelB ← runtimeB.releaseKernel authorizedB kernelB
  let copyOutGreen := match copiedOutA, copiedOutB with
    | .ok resultA, .ok resultB =>
        resultA.bytes.size == 4 && resultB.bytes.size == 4
    | _, _ => false
  let operationsGreen :=
    !isError launchedA && !isError launchedB &&
    !isError copiedInA && !isError copiedInB && copyOutGreen
  let releasesGreen :=
    !isError releasedBufferA && !isError releasedKernelA &&
    !isError releasedBufferB && !isError releasedKernelB
  failures := failures + (← check "two distinct leaves consume same plan-indexed interface"
    (operationsGreen && releasesGreen))

  if failures != 0 then
    IO.eprintln ("backend-shared-tests: " ++ toString failures ++ " failure(s)")
    IO.Process.exit 1
  IO.println "backend-shared-tests: all focused checks green"

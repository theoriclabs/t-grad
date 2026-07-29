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

def hasUnavailableClass (expected : UnavailableClass) :
    Availability Capability Detail → Bool
  | .unavailable _ reason => reason.reasonClass == expected
  | .available _ => false

def hasRuntimeError (expectedClass : RuntimeErrorClass)
    (expectedStage : RuntimeStage) : RuntimeResult String α → Bool
  | .error failure =>
      failure.errorClass == expectedClass && failure.stage == expectedStage
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

private structure WitnessCredentialA where
  token : Unit

private structure WitnessCredentialB where
  token : Unit

structure WitnessSourceA where
  scalar : RenderedScalar
  elementCount : Nat
  outputIndex : RenderedOutputIndex
  bounds : RenderedBounds
  declaration : String
  deriving BEq, Repr, DecidableEq

inductive WitnessSemanticTokenB where
  | scalar (value : RenderedScalar)
  | elementCount (value : Nat)
  | outputIndex (meaning : RenderedOutputIndex)
  | bounds (meaning : RenderedBounds)
  deriving BEq, Repr, DecidableEq

structure WitnessSourceB where
  semanticTokens : List WitnessSemanticTokenB
  words : List Nat
  deriving BEq, Repr, DecidableEq

/-- A deliberately untrustworthy Bool equality. Artifact admission must use
propositional DecidableEq instead, so distinct payloads remain distinguishable. -/
structure LawlessPayload where
  semantics : RenderedSemantics
  value : String
  deriving Repr, DecidableEq

instance : BEq LawlessPayload := ⟨fun _ _ => true⟩

def rendererA (identity : RendererContractIdentity) :
    RendererContract WitnessSourceA :=
  { identity
    render := fun plan => {
      scalar := plan.value.renderedScalar
      elementCount := plan.elementCount.value
      outputIndex := plan.outputIndexPolicy.rendered
      bounds := plan.boundsPolicy.rendered
      declaration := plan.kernelName ++ ":" ++ plan.value.stableTag
    }
    projectSemantics := fun payload => some {
      scalar := payload.scalar
      elementCount := payload.elementCount
      outputIndex := payload.outputIndex
      bounds := payload.bounds }
    renderPreserves := fun _ => rfl }

def rendererB (identity : RendererContractIdentity) :
    RendererContract WitnessSourceB :=
  { identity
    render := fun plan => {
      semanticTokens := [
        .scalar plan.value.renderedScalar,
        .elementCount plan.elementCount.value,
        .outputIndex plan.outputIndexPolicy.rendered,
        .bounds plan.boundsPolicy.rendered]
      words := [plan.elementCount.value, plan.byteCount.value,
        plan.launch.threadsPerBlock.value, plan.launch.blocksPerGrid.value]
    }
    projectSemantics := fun payload =>
      match payload.semanticTokens with
      | [.scalar scalar, .elementCount count, .outputIndex index, .bounds bounds] =>
          some {
            scalar
            elementCount := count
            outputIndex := index
            bounds }
      | _ => none
    renderPreserves := fun _ => rfl }

structure BufferHandleA where raw : Nat deriving BEq, Repr
structure BufferHandleB where token : String deriving BEq, Repr
structure KernelHandleA where raw : Nat deriving BEq, Repr
structure KernelHandleB where token : String deriving BEq, Repr

def runtimeFailure (errorClass : RuntimeErrorClass) (stage : RuntimeStage)
    (detail : String) : RuntimeFailure String :=
  { errorClass, stage, detail }

def missingResult (stage : RuntimeStage) : IO (RuntimeResult String α) :=
  pure (.error (runtimeFailure .operationNotImplemented stage "missing"))

def zeroCopyOut (request : CopyOutRequest capacity) : CopyOutResult request :=
  { bytes := ByteArray.mk (Array.replicate request.byteCount.value 0)
    sizeExact := by
      change (Array.replicate request.byteCount.value 0).size = _
      exact Array.size_replicate }

def boundaryA (renderer : RendererContract WitnessSourceA)
    (authority : ProbeAuthority WitnessCredentialA)
    (capability : AvailableCapability authority) :
    RuntimeBoundary (AvailableCapability authority) String BufferHandleA
      KernelHandleA WitnessSourceA :=
  { capabilityContract := AvailableCapability.contract authority
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
      match request.renderedSemantics with
      | none => pure (.error (runtimeFailure .artifactBindingFailure .compilation
          "witness A omitted semantic projection"))
      | some projected =>
          if request.sourceIdentity !=
                authorized.plan.sourceIdentity renderer.identity ||
              projected != authorized.plan.renderedSemantics ||
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
                (runtimeFailure .artifactBindingFailure .compilation
                  (reprStr reason)))
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
    (authority : ProbeAuthority WitnessCredentialB)
    (capability : AvailableCapability authority) :
    RuntimeBoundary (AvailableCapability authority) String BufferHandleB
      KernelHandleB WitnessSourceB :=
  { capabilityContract := AvailableCapability.contract authority
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
      match request.renderedSemantics with
      | none => pure (.error (runtimeFailure .artifactBindingFailure .compilation
          "witness B omitted semantic projection"))
      | some projected =>
          if request.sourceIdentity !=
                authorized.plan.sourceIdentity renderer.identity ||
              projected != authorized.plan.renderedSemantics ||
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
                (runtimeFailure .artifactBindingFailure .compilation
                  (reprStr reason)))
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

def missingBoundaryA (renderer : RendererContract WitnessSourceA)
    (authority : ProbeAuthority WitnessCredentialA)
    (capability : AvailableCapability authority) :
    RuntimeBoundary (AvailableCapability authority) String BufferHandleA
      KernelHandleA WitnessSourceA :=
  { capabilityContract := AvailableCapability.contract authority
    renderer
    availability := pure (.available capability)
    allocate := fun _ => missingResult .allocation
    compile := fun _ _ => missingResult .compilation
    launch := fun _ _ _ => missingResult .launch
    synchronize := fun _ _ => missingResult .synchronization
    copyIn := fun _ _ _ => missingResult .copy
    copyOut := fun _ _ _ => missingResult .copy
    releaseBuffer := fun _ _ => missingResult .release
    releaseKernel := fun _ _ => missingResult .release }

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
  let profileB ← getIdentity
    (DeviceProfile.build backendB archB {
      mode := .offline, tool := toolB, version := versionB } 512) "profile"
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
  /- This deliberately ABI-wide neutral profile is only for arithmetic-edge
  tests. Vendor leaves remain responsible for admitting real device limits. -/
  let wideProfile ← getIdentity
    (DeviceProfile.build backendA archA {
      mode := .runtime, tool := toolA, version := versionA } maxAbiU32) "profile"

  let floatPlan ← getPlan profile .float32_ (.float32Bits 1065353216) 513 256
  let floatPlanB ← getPlan profileB .float32_ (.float32Bits 1065353216) 513 256
  let intPlan ← getPlan profile .int32_ (.signed (-7)) 17 16
  let intMinPlan ← getPlan profile .int32_ (.signed (-2147483648)) 1 1
  let intMaxPlan ← getPlan profile .int32_ (.signed 2147483647) 1 1
  let plusZeroPlan ← getPlan profile .float32_ (.float32Bits 0x00000000) 1 1
  let minusZeroPlan ← getPlan profile .float32_ (.float32Bits 0x80000000) 1 1
  let plusInfPlan ← getPlan profile .float32_ (.float32Bits 0x7f800000) 1 1
  let minusInfPlan ← getPlan profile .float32_ (.float32Bits 0xff800000) 1 1
  let nanOnePlan ← getPlan profile .float32_ (.float32Bits 0x7fc00001) 1 1
  let nanTwoPlan ← getPlan profile .float32_ (.float32Bits 0x7fc00002) 1 1
  let negativeOrdinaryPlan ← getPlan profile .float32_
    (.float32Bits 0xc0200000) 1 1
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
  failures := failures + (← check "int32 endpoints admitted exactly"
    (intMinPlan.value.stableTag == "i32:-2147483648" &&
     intMaxPlan.value.stableTag == "i32:2147483647"))
  let specialFloatPlans := [plusZeroPlan, minusZeroPlan, plusInfPlan,
    minusInfPlan, nanOnePlan, nanTwoPlan, negativeOrdinaryPlan]
  let specialFloatTags := specialFloatPlans.map (fun plan => plan.value.stableTag)
  failures := failures + (← check "float32 storage edge bits remain distinct"
    (specialFloatTags == ["f32bits:0", "f32bits:2147483648",
      "f32bits:2139095040", "f32bits:4286578688", "f32bits:2143289345",
      "f32bits:2143289346", "f32bits:3223322624"] &&
     specialFloatPlans.all (fun plan => plan.dtype == .float32_)))
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
  let maxByteElements := maxAbiU64 / 4
  failures := failures + (← check "maximum representable float byte plan accepted"
    (match FillPlan.build wideProfile .float32_ (.float32Bits 0)
        maxByteElements maxAbiU32 with
     | .ok plan => plan.byteCount.value == maxByteElements * 4
     | .error _ => false))
  failures := failures + (← check "first byte overflow and element overflow distinguish"
    (match FillPlan.build wideProfile .float32_ (.float32Bits 0)
        (maxByteElements + 1) maxAbiU32,
      FillPlan.build wideProfile .float32_ (.float32Bits 0)
        (maxAbiU64 + 1) maxAbiU32 with
     | .error (.byteCountOverflow _ _), .error (.elementCountOutOfRange _) => true
     | _, _ => false))
  failures := failures + (← check "maximum grid accepted and first overflow rejected"
    (match FillPlan.build profile .float32_ (.float32Bits 0)
        (maxAbiU32 * 1024) 1024,
      FillPlan.build profile .float32_ (.float32Bits 0)
        (maxAbiU32 * 1024 + 1) 1024 with
     | .ok maximum, .error (.gridSizeOutOfRange _) =>
         maximum.launch.blocksPerGrid.value == maxAbiU32
     | _, _ => false))

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
  let unicodeBackend ← getIdentity (BackendIdentity.build "α|backend") "backend"
  let unicodeProfile ← getIdentity
    (DeviceProfile.build unicodeBackend archA {
      mode := .runtime, tool := toolA, version := versionA } 1024) "profile"
  let unicodePlan ← getPlan unicodeProfile .float32_
    (.float32Bits 1065353216) 513 256
  failures := failures + (← check "opaque unicode identity remains distinct"
    (unicodePlan.cacheIdentity != delimiterPlan.cacheIdentity &&
     unicodePlan.cacheIdentity != floatPlan.cacheIdentity))

  let leafRendererA := rendererA rendererIdentityA
  let leafRendererB := rendererB rendererIdentityB
  let artifactA := leafRendererA.renderArtifact floatPlan
  let artifactB := leafRendererB.renderArtifact floatPlanB
  failures := failures + (← check "distinct leaves share typed semantic plan"
    (leafRendererA.projectSemantics artifactA.payload ==
       leafRendererB.projectSemantics artifactB.payload &&
     floatPlan.renderedSemantics == floatPlanB.renderedSemantics &&
     artifactA.identity.kernel.semantic != artifactB.identity.kernel.semantic &&
     artifactA.identity.rendererContract != artifactB.identity.rendererContract))
  let forgedA : SourceArtifactCandidate WitnessSourceA := {
    identity := artifactA.identity
    payload := { artifactA.payload with
      declaration := artifactA.payload.declaration ++ ":forged" }
  }
  failures := failures + (← check "arbitrary source payload rejected before compile"
    (isError (SourceArtifact.validateCandidate leafRendererA floatPlan forgedA)))
  let wrongIndexA : SourceArtifactCandidate WitnessSourceA := {
    identity := artifactA.identity
    payload := { artifactA.payload with outputIndex := .blockOnlyX }
  }
  let wrongBoundsA : SourceArtifactCandidate WitnessSourceA := {
    identity := artifactA.identity
    payload := { artifactA.payload with bounds := .unguarded }
  }
  let omittedIndexB : SourceArtifactCandidate WitnessSourceB := {
    identity := artifactB.identity
    payload := { artifactB.payload with
      semanticTokens := artifactB.payload.semanticTokens.filter (fun token =>
        match token with | .outputIndex _ => false | _ => true) }
  }
  failures := failures + (← check "wrong or omitted renderer semantics rejected"
    (isError (SourceArtifact.validateCandidate leafRendererA floatPlan wrongIndexA) &&
     isError (SourceArtifact.validateCandidate leafRendererA floatPlan wrongBoundsA) &&
     isError (SourceArtifact.validateCandidate leafRendererB floatPlanB omittedIndexB)))
  let lawlessRenderer : RendererContract LawlessPayload := {
    identity := rendererIdentityA
    render := fun plan => { semantics := plan.renderedSemantics, value := "expected" }
    projectSemantics := fun payload => some payload.semantics
    renderPreserves := fun _ => rfl
  }
  let lawlessCandidate : SourceArtifactCandidate LawlessPayload := {
    identity := floatPlan.sourceIdentity rendererIdentityA
    payload := { semantics := floatPlan.renderedSemantics, value := "forged" }
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
  let wrongRendererA : SourceArtifactCandidate WitnessSourceA := {
    identity := floatPlan.sourceIdentity rendererIdentityB
    payload := leafRendererA.render floatPlan }
  failures := failures + (← check "wrong renderer and cross-profile artifacts reject"
    (isError (SourceArtifact.validateCandidate leafRendererA floatPlan wrongRendererA) &&
     isError (SourceArtifact.validateCandidate leafRendererB floatPlan
       artifactB.candidate)))
  let compileRequestA := CompileRequest.build leafRendererA floatPlan artifactA
  let compileRequestB := CompileRequest.build leafRendererB floatPlanB artifactB
  failures := failures + (← check "exact bound artifacts admit compile requests"
    (!isError compileRequestA && !isError compileRequestB))

  let credentialA : WitnessCredentialA := { token := () }
  let credentialB : WitnessCredentialB := { token := () }
  let authorityA : ProbeAuthority WitnessCredentialA := {
    requestedBackend := backendA
    admitsProfile := fun _ candidate =>
      candidate.architecture == archA &&
      candidate.compiler.mode == .runtime &&
      candidate.compiler.tool == toolA }
  let authorityB : ProbeAuthority WitnessCredentialB := {
    requestedBackend := backendB
    admitsProfile := fun _ candidate =>
      candidate.architecture == archB &&
      candidate.compiler.mode == .offline &&
      candidate.compiler.tool == toolB }

  let absentA := availabilityFromProbe authorityA (.absent "absent")
  let runtimeMissingA := availabilityFromProbe authorityA
    (.runtimeMissing "runtime missing")
  let negativeA := availabilityFromProbe authorityA (.negative "negative")
  let failedA := availabilityFromProbe authorityA (.failed "failed")
  let zeroA := availabilityFromProbe authorityA (.countOnly 0 "zero")
  let countOnlyA := availabilityFromProbe authorityA (.countOnly 1 "count only")
  let incompleteA := availabilityFromProbe authorityA
    (.profiled credentialA 1 0 .missing "missing profile")
  let invalidRawA := availabilityFromProbe authorityA
    (.profiled credentialA 1 0 .invalid "invalid profile")
  let invalidOrdinalA := availabilityFromProbe authorityA
    (.profiled credentialA 1 1 (.complete profile) "invalid ordinal")
  let wrongBackendA := availabilityFromProbe authorityA
    (.profiled credentialA 1 0 (.complete otherBackend) "wrong backend")
  let incompatibleProfileA := availabilityFromProbe authorityA
    (.profiled credentialA 1 0 (.complete otherArch) "incompatible profile")
  let incompatibleCompilerA := availabilityFromProbe authorityA
    (.profiled credentialA 1 0 (.complete otherMode) "incompatible compiler")
  let availableA := availabilityFromProbe authorityA
    (.profiled credentialA 1 0 (.complete profile) "complete")
  let availableB := availabilityFromProbe authorityB
    (.profiled credentialB 1 0 (.complete profileB) "complete")
  failures := failures + (← check "probe admission negative matrix fails closed"
    (hasUnavailableClass .probeAbsent absentA &&
     hasUnavailableClass .runtimeLibraryMissing runtimeMissingA &&
     hasUnavailableClass .negativeProbe negativeA &&
     hasUnavailableClass .probeFailed failedA &&
     hasUnavailableClass .noDevice zeroA &&
     hasUnavailableClass .countOnly countOnlyA &&
     hasUnavailableClass .incompleteDeviceProfile incompleteA &&
     hasUnavailableClass .invalidDeviceProfile invalidRawA &&
     hasUnavailableClass .invalidDeviceOrdinal invalidOrdinalA &&
     hasUnavailableClass .wrongBackend wrongBackendA &&
     hasUnavailableClass .invalidDeviceProfile incompatibleProfileA &&
     hasUnavailableClass .invalidDeviceProfile incompatibleCompilerA))
  failures := failures + (← check "complete admitted profiles alone mint capability"
    (availableA.isAvailable && availableB.isAvailable))
  let .available capabilityA := availableA
    | throw (IO.userError "witness A complete probe unexpectedly unavailable")
  let .available capabilityB := availableB
    | throw (IO.userError "witness B complete probe unexpectedly unavailable")
  let device := capabilityA.deviceOf
  let otherDevice : DeviceIdentity := { profile, ordinal := 1 }
  let contractA := AvailableCapability.contract authorityA
  let contractB := AvailableCapability.contract authorityB
  let authorizationA := AuthorizedPlan.build contractA capabilityA floatPlan
  let authorizationB := AuthorizedPlan.build contractB capabilityB floatPlanB
  failures := failures + (← check "capabilities bind exact plan profile"
    (!isError authorizationA && !isError authorizationB &&
     isError (AuthorizedPlan.build contractA capabilityA archPlan)))
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

  failures := failures + (← check "stable error vocabulary includes binding and release"
    (RuntimeErrorClass.artifactBindingFailure != .invalidBuffer &&
     RuntimeErrorClass.releaseFailure != .operationNotImplemented &&
     RuntimeStage.release != .moduleLoad))

  let runtimeA := boundaryA leafRendererA authorityA capabilityA
  let runtimeB := boundaryB leafRendererB authorityB capabilityB
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
  let missingRuntime := missingBoundaryA leafRendererA authorityA capabilityA
  let missingAllocation ← missingRuntime.allocate authorizedA
  failures := failures + (← check "missing implementation is structured failure"
    (hasRuntimeError .operationNotImplemented .allocation missingAllocation))
  let .ok zeroBuffer := BoundBuffer.build authorizedA {
      handle := ({ raw := 0 } : BufferHandleA)
      device := authorizedA.device
      ownership := .runtimeOwned
      byteCapacity := authorizedA.plan.byteCount }
    | throw (IO.userError "zero-handle buffer metadata unexpectedly rejected")
  let .ok zeroKernel := BoundCompiledKernel.build authorizedA {
      handle := ({ raw := 0 } : KernelHandleA)
      device := authorizedA.device
      identity := authorizedA.plan.kernelIdentity }
    | throw (IO.userError "zero-handle kernel metadata unexpectedly rejected")
  let invalidLaunch ← runtimeA.launch authorizedA zeroKernel zeroBuffer
  let invalidBufferRelease ← runtimeA.releaseBuffer authorizedA zeroBuffer
  let invalidKernelRelease ← runtimeA.releaseKernel authorizedA zeroKernel
  failures := failures + (← check "leaf invalid handles fail launch and release"
    (hasRuntimeError .launchFailure .launch invalidLaunch &&
     hasRuntimeError .releaseFailure .release invalidBufferRelease &&
     hasRuntimeError .releaseFailure .release invalidKernelRelease))
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

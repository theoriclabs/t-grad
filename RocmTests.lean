import Tgrad.Backend.FillPlan
import Tgrad.Renderer.Hip
import Tgrad.Runtime.Hip

open Tgrad
open Tgrad.Backend

private def hipIdentity (mode : CompilerMode := .runtime)
    (arch : String := "gfx1100") : BackendIdentity :=
  { backend := .hip, architecture := arch, compilerMode := mode,
    compilerTool := if mode == .runtime then .hiprtc else .hipcc,
    compilerIdentity := "rocm-v1", maxThreadsPerBlock := 1024 }

private def cudaIdentity : BackendIdentity :=
  { backend := .cuda, architecture := "sm_80", compilerMode := .runtime,
    compilerTool := .nvrtc, compilerIdentity := "cuda-v1",
    maxThreadsPerBlock := 1024 }

private def check (name : String) (condition : Bool) : IO Nat := do
  if condition then
    IO.println s!"PASS {name}"
    pure 0
  else
    IO.eprintln s!"FAIL {name}"
    pure 1

private def plan? (identity : BackendIdentity) (dtype : Dtype)
    (input : ScalarInput) (count block : Nat) : Option FillPlan :=
  (mkFillPlan identity dtype input count block).toOption

private def rendered? (dtype : Dtype) (input : ScalarInput)
    (count block : Nat) : Option Renderer.Hip.KernelSource := do
  let plan ← plan? hipIdentity dtype input count block
  (Renderer.Hip.renderFill plan).toOption

private def hasPlanError (expected : PlanError)
    (result : Except PlanError FillPlan) : Bool :=
  match result with
  | .error actual => actual == expected
  | .ok _ => false

private def negativeProbeFact (value : Runtime.Hip.ProbeFact) : Bool :=
  match value with
  | .negative _ => true
  | .reportedDeviceCount _ => false

private def hasRuntimeError (expected : Runtime.Hip.RuntimeError)
    (result : Runtime.Hip.RuntimeResult α) : Bool :=
  match result with
  | .error actual => actual == expected
  | .ok _ => false

def main : IO UInt32 := do
  let mut failures := 0
  IO.println "ROCM_STATIC_SLICE_TESTS"

  -- Geometry boundary set required by the backend contract.
  for (count, expected) in [(0, 0), (1, 1), (255, 1), (256, 1),
                            (257, 2), (512, 2)] do
    let actual := (plan? hipIdentity .int32_ (.signed 7) count 256).map
      (fun plan => plan.launch.gridSize)
    failures := failures + (← check s!"ceil-div count={count}"
      (actual == some expected))

  let sample := plan? hipIdentity .float32_ (.float32Bits 1078984704) 257 256
  failures := failures + (← check "byte plan 257*f32=1028"
    (sample.map (fun p => p.byteCount) == some 1028))
  failures := failures + (← check "launch 257/256 gives grid=2 block=256"
    (sample.map (fun p => (p.launch.gridSize, p.launch.blockSize)) ==
      some (2, 256)))

  let floatArtifact := rendered? .float32_ (.float32Bits 1078984704) 257 256
  let floatSource := floatArtifact.map Renderer.Hip.KernelSource.sourceText
  failures := failures + (← check "HIP output index expression"
    (floatSource.map (fun source => (source.splitOn "\n").contains
      "  const unsigned long long idx = ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) + (unsigned long long)threadIdx.x;") == some true))
  failures := failures + (← check "HIP exact element-count guard"
    (floatSource.map (·.contains "if (idx < 257ULL)") == some true))
  failures := failures + (← check "HIP f32 dtype/value emission"
    (match sample, floatArtifact with
    | some plan, some artifact =>
        artifact.sourceText.contains
          ("void " ++ plan.kernelName ++ "(float *out)") &&
        artifact.sourceText.contains
          "out[idx] = __builtin_bit_cast(float, 1078984704u);"
    | _, _ => false))
  let negativeZeroSource :=
    (rendered? .float32_ (.float32Bits 2147483648) 1 64).map
      Renderer.Hip.KernelSource.sourceText
  failures := failures + (← check "HIP f32 signed-zero bit emission"
    (negativeZeroSource.map (·.contains
      "out[idx] = __builtin_bit_cast(float, 2147483648u);") == some true))
  let intSource := (rendered? .int32_ (.signed (-7)) 17 64).map
    Renderer.Hip.KernelSource.sourceText
  failures := failures + (← check "HIP i32 dtype/value emission"
    (intSource.map (fun source =>
      source.contains "(int *out)" &&
      source.contains "out[idx] = -7;") == some true))

  failures := failures + (← check "unsupported dtype rejected"
    (hasPlanError (.unsupportedDtype .float16_)
      (mkFillPlan hipIdentity .float16_ (.float32Bits 1065353216) 1 64)))
  failures := failures + (← check "dtype/value mismatch rejected"
    (hasPlanError (.valueDtypeMismatch .float32_)
      (mkFillPlan hipIdentity .float32_ (.signed 1) 1 64)))
  failures := failures + (← check "int32 lower bound admitted"
    ((plan? hipIdentity .int32_ (.signed (-2147483648)) 1 64).isSome))
  failures := failures + (← check "int32 upper bound admitted"
    ((plan? hipIdentity .int32_ (.signed 2147483647) 1 64).isSome))
  failures := failures + (← check "int32 below lower bound rejected"
    (hasPlanError (.int32OutOfRange (-2147483649))
      (mkFillPlan hipIdentity .int32_ (.signed (-2147483649)) 1 64)))
  failures := failures + (← check "int32 above upper bound rejected"
    (hasPlanError (.int32OutOfRange 2147483648)
      (mkFillPlan hipIdentity .int32_ (.signed 2147483648) 1 64)))
  failures := failures + (← check "zero block rejected"
    (hasPlanError .zeroBlockSize
      (mkFillPlan hipIdentity .int32_ (.signed 1) 1 0)))
  failures := failures + (← check "device block limit admitted"
    ((plan? hipIdentity .int32_ (.signed 1) 1 1024).isSome))
  failures := failures + (← check "device block limit + 1 rejected"
    (hasPlanError (.blockExceedsDeviceLimit 1025 1024)
      (mkFillPlan hipIdentity .int32_ (.signed 1) 1 1025)))
  failures := failures + (← check "byte multiplication overflow rejected"
    (hasPlanError (.byteCountOverflow (abiUInt64Max / 4 + 1) 4)
      (mkFillPlan hipIdentity .float32_ (.float32Bits 1065353216)
        (abiUInt64Max / 4 + 1) 256)))
  failures := failures + (← check "element count ABI max reaches later byte rejection"
    (hasPlanError (.byteCountOverflow abiUInt64Max 4)
      (mkFillPlan hipIdentity .int32_ (.signed 1) abiUInt64Max 256)))
  failures := failures + (← check "element count ABI max + 1 rejected"
    (hasPlanError (.elementCountOverflow (abiUInt64Max + 1))
      (mkFillPlan hipIdentity .int32_ (.signed 1) (abiUInt64Max + 1) 256)))
  let tooWideCount := abiUInt32Max * 256 + 1
  failures := failures + (← check "launch ABI width overflow rejected"
    (hasPlanError (.launchWidthOverflow (abiUInt32Max + 1) 256)
      (mkFillPlan hipIdentity .int32_ (.signed 1) tooWideCount 256)))
  failures := failures + (← check "launch ABI width max admitted"
    ((plan? hipIdentity .int32_ (.signed 1) (abiUInt32Max * 256) 256).map
      (fun p => p.launch.gridSize) == some abiUInt32Max))
  let mismatchedIdentity : BackendIdentity :=
    { hipIdentity with compilerMode := .runtime, compilerTool := .hipcc }
  failures := failures + (← check "HIP compiler/profile mismatch rejected"
    (hasPlanError (.compilerProfileMismatch .hip .runtime .hipcc)
      (mkFillPlan mismatchedIdentity .int32_ (.signed 1) 1 64)))
  let invalidArchIdentity : BackendIdentity :=
    { hipIdentity with architecture := "sm_80" }
  failures := failures + (← check "HIP architecture profile mismatch rejected"
    (hasPlanError (.invalidArchitecture .hip "sm_80")
      (mkFillPlan invalidArchIdentity .int32_ (.signed 1) 1 64)))

  let hipPlan := plan? hipIdentity .int32_ (.signed 7) 257 256
  let cudaPlan := plan? cudaIdentity .int32_ (.signed 7) 257 256
  let offlinePlan := plan? (hipIdentity .offline) .int32_ (.signed 7) 257 256
  let archPlan := plan? (hipIdentity .runtime "gfx942") .int32_ (.signed 7) 257 256
  let valuePlan := plan? hipIdentity .int32_ (.signed 8) 257 256
  let launchPlan := plan? hipIdentity .int32_ (.signed 7) 257 128
  failures := failures + (← check "cache separates backend identity"
    ((hipPlan.map FillPlan.cacheIdentity) != (cudaPlan.map FillPlan.cacheIdentity)))
  failures := failures + (← check "cache separates compiler mode"
    ((hipPlan.map FillPlan.cacheIdentity) != (offlinePlan.map FillPlan.cacheIdentity)))
  failures := failures + (← check "cache separates architecture"
    ((hipPlan.map FillPlan.cacheIdentity) != (archPlan.map FillPlan.cacheIdentity)))
  failures := failures + (← check "cache separates value"
    ((hipPlan.map FillPlan.cacheIdentity) != (valuePlan.map FillPlan.cacheIdentity)))
  failures := failures + (← check "cache separates launch semantics"
    ((hipPlan.map FillPlan.cacheIdentity) != (launchPlan.map FillPlan.cacheIdentity)))
  failures := failures + (← check "HIP renderer rejects a CUDA plan"
    (cudaPlan.bind (fun p => (Renderer.Hip.renderFill p).toOption)).isNone)

  let gfx1100Plan := plan? hipIdentity .float32_ (.float32Bits 1078984704) 257 256
  let gfx942Plan := plan? (hipIdentity .runtime "gfx942") .float32_
    (.float32Bits 1078984704) 257 256
  let hipccPlan := plan? (hipIdentity .offline "gfx1100") .float32_
    (.float32Bits 1078984704) 257 256
  let gfx1100Source := gfx1100Plan.bind
    (fun plan => (Renderer.Hip.renderFill plan).toOption)
  let gfx942Source := gfx942Plan.bind
    (fun plan => (Renderer.Hip.renderFill plan).toOption)
  let hipccSource := hipccPlan.bind
    (fun plan => (Renderer.Hip.renderFill plan).toOption)
  failures := failures + (← check "kernel source is bound to plan identities"
    (match gfx1100Plan, gfx1100Source with
    | some plan, some source =>
        source.kernelIdentity == plan.kernelIdentity &&
        source.sourceIdentity == plan.sourceIdentity &&
        source.kernelName == plan.kernelName &&
        source.cacheIdentity == plan.cacheIdentity
    | _, _ => false))
  failures := failures + (← check "source symbol is plan-derived"
    (match gfx1100Plan, gfx1100Source with
    | some plan, some source =>
        source.sourceText.contains
          ("void " ++ plan.kernelName ++ "(float *out)")
    | _, _ => false))
  failures := failures + (← check "gfx and compiler profiles bind distinct artifacts"
    (match gfx1100Source, gfx942Source, hipccSource with
    | some a, some b, some c =>
        a.kernelName != b.kernelName && a.kernelName != c.kernelName &&
        a.cacheIdentity != b.cacheIdentity && a.cacheIdentity != c.cacheIdentity &&
        a.sourceIdentity != b.sourceIdentity && a.sourceIdentity != c.sourceIdentity
    | _, _, _ => false))
  failures := failures + (← check
    "equivalent fill bodies cannot collapse profile identity"
    (match gfx1100Source, gfx942Source, hipccSource with
    | some a, some b, some c =>
        [a, b, c].all (fun source =>
          source.sourceText.contains
            "out[idx] = __builtin_bit_cast(float, 1078984704u);" &&
          source.sourceText.contains "if (idx < 257ULL)") &&
        a.cacheIdentity != b.cacheIdentity && a.cacheIdentity != c.cacheIdentity
    | _, _, _ => false))
  failures := failures + (← check "arbitrary source candidate rejected"
    (match gfx1100Plan, gfx1100Source with
    | some plan, some source =>
        let forged : Renderer.Hip.KernelSourceCandidate :=
          { kernelIdentity := source.kernelIdentity
            sourceIdentity := source.sourceIdentity
            sourceText := source.sourceText ++ "// arbitrary" }
        match Renderer.Hip.validateCandidate plan forged with
        | .error .sourceTextMismatch => true
        | _ => false
    | _, _ => false))
  failures := failures + (← check "mismatched profile compile request rejected"
    (match gfx942Plan, gfx1100Source with
    | some plan, some source =>
        hasRuntimeError (.kernelBinding .kernelIdentityMismatch)
          (Runtime.Hip.CompileRequest.build plan source)
    | _, _ => false))
  failures := failures + (← check "private capability couples full profile"
    Runtime.Hip.capabilityCouplingSelfCheck)
  failures := failures + (← check "count-only probe cannot mint capability"
    Runtime.Hip.incompleteProbeSelfCheck)

  failures := failures + (← check "absent runtime probe is unavailable"
    (negativeProbeFact (Runtime.Hip.classifyProbeRaw 0)))
  failures := failures + (← check "negative query probe is unavailable"
    (negativeProbeFact (Runtime.Hip.classifyProbeRaw 2)))
  failures := failures + (← check "zero-device probe is unavailable"
    (negativeProbeFact (Runtime.Hip.classifyProbeRaw 3)))
  failures := failures + (← check "malformed probe is unavailable"
    (negativeProbeFact (Runtime.Hip.classifyProbeRaw 256)))
  failures := failures + (← check "positive raw input is protocol fact, not capability"
    (Runtime.Hip.classifyProbeRaw 257 == .reportedDeviceCount 1))
  -- Execution below re-probes through the opaque IO boundary; this parsed
  -- caller value never flows into `localBoundary`.

  let invalidBuffer : Runtime.Hip.Buffer :=
    { raw := 0, backend := .hip, ownership := .runtimeOwned, byteCount := 16 }
  failures := failures + (← check "copy rejects invalid handle"
    (hasRuntimeError .invalidHandle
      (Runtime.Hip.validateTransfer invalidBuffer .deviceToHost 4)))
  let smallBuffer : Runtime.Hip.Buffer :=
    { raw := 1, backend := .hip, ownership := .runtimeOwned, byteCount := 16 }
  failures := failures + (← check "copy rejects oversize"
    (hasRuntimeError (.invalidCopySize 17 16)
      (Runtime.Hip.validateTransfer smallBuffer .deviceToHost 17)))
  let wrongBuffer : Runtime.Hip.Buffer :=
    { raw := 1, backend := .cuda, ownership := .runtimeOwned, byteCount := 16 }
  failures := failures + (← check "copy rejects wrong backend"
    (hasRuntimeError (.wrongBufferBackend .cuda)
      (Runtime.Hip.validateTransfer wrongBuffer .hostToDevice 4)))
  failures := failures + (← check "vendor success translation is stable"
    ((Runtime.Hip.translateVendorResult .launch 0).isOk))
  failures := failures + (← check "vendor error translation is stable"
    (hasRuntimeError (.vendorError .launch 700)
      (Runtime.Hip.translateVendorResult .launch 700)))

  let actualAvailability ← Runtime.Hip.availability
  match actualAvailability with
  | .unavailable reason =>
      IO.println s!"UNOBSERVED_ENVIRONMENT hip_runtime_or_device {repr reason}"
      match sample, floatArtifact with
      | some plan, some source =>
          match Runtime.Hip.CompileRequest.build plan source with
          | .error _ =>
              failures := failures + (← check
                "exact compile request accepted before unavailable runtime" false)
          | .ok request =>
              failures := failures + (← check
                "exact compile request accepted before unavailable runtime" true)
              let compileResult ← Runtime.Hip.localBoundary.compile request
              failures := failures + (← check
                "arbitrary positive raw fact cannot authorize execution"
                (hasRuntimeError (.unavailable reason) compileResult))
      | _, _ =>
          failures := failures + (← check
            "arbitrary positive raw fact cannot authorize execution" false)
  | .available device =>
      IO.eprintln s!"FAIL unexpected HIP runtime/device availability {repr device}"
      failures := failures + 1

  IO.println s!"SUMMARY failures={failures}"
  pure failures.toUInt32

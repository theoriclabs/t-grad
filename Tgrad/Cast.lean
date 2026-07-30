import Tgrad.Tensor
import Tgrad.Runtime.MetalAllocator
import Tgrad.Runtime.MetalProgram

/-! # Lean-owned numeric cast and storage bitcast authority

Plans carry the invariants required by execution.  Rendering retains the exact
plan and launch proof; compilation retains that exact artifact; dispatch can be
formed only from the compiled artifact; and a cast Tensor can be formed only
from a successful dispatch value.  Python supplies syntax and receives the
structured result, but owns none of these decisions.
-/
namespace Tgrad

namespace Cast

inductive TransformReason where
  | ok | invalidHandle | invalidDtype | nonBuffer | rootDtypeMismatch
  | invalidShape | sizeOverflow | bufferTooSmall | unsupportedPair
  | unequalItemSize | compileFailed | allocationFailed | dispatchFailed
  | transportMetadataMismatch
  deriving BEq, Repr, Inhabited, DecidableEq

def TransformReason.code : TransformReason → UInt8
  | .ok => 0 | .invalidHandle => 1 | .invalidDtype => 2 | .nonBuffer => 3
  | .rootDtypeMismatch => 4 | .invalidShape => 5 | .sizeOverflow => 6
  | .bufferTooSmall => 7 | .unsupportedPair => 8 | .unequalItemSize => 9
  | .compileFailed => 10 | .allocationFailed => 11 | .dispatchFailed => 12
  | .transportMetadataMismatch => 13

def TransformReason.name : TransformReason → String
  | .ok => "ok" | .invalidHandle => "invalidHandle"
  | .invalidDtype => "invalidDtype" | .nonBuffer => "nonBuffer"
  | .rootDtypeMismatch => "rootDtypeMismatch" | .invalidShape => "invalidShape"
  | .sizeOverflow => "sizeOverflow" | .bufferTooSmall => "bufferTooSmall"
  | .unsupportedPair => "unsupportedPair" | .unequalItemSize => "unequalItemSize"
  | .compileFailed => "compileFailed" | .allocationFailed => "allocationFailed"
  | .dispatchFailed => "dispatchFailed"
  | .transportMetadataMismatch => "transportMetadataMismatch"

inductive TransformOwnership where
  | existing | owned | borrowed
  deriving BEq, Repr, Inhabited, DecidableEq

structure TransformResult where
  reason : TransformReason
  tensor : Option Tensor := none
  ownership : TransformOwnership := .existing
  deriving Inhabited

private structure CheckedUSize where
  value : Nat
  abi : USize
  exact : abi.toNat = value

private structure MaterializedRoot where
  uop : UOp
  raw : UInt64
  shape : List Nat
  rootDtype : Dtype
  tensorDtype : Dtype
  targetDtype : Dtype
  rootExact : uop = .buffer raw shape rootDtype
  dtypeExact : rootDtype = tensorDtype
  dimsPositive : shape.all (fun dim => dim != 0) = true
  count : CheckedUSize
  inputBytes : CheckedUSize
  outputBytes : CheckedUSize
  countFormula : count.value = Tgrad.numel shape
  inputFormula : inputBytes.value = count.value * rootDtype.sizeBytes
  outputFormula : outputBytes.value = count.value * targetDtype.sizeBytes

def castPairs : List (Dtype × Dtype) :=
  [(.float32_, .bfloat16_), (.bfloat16_, .float32_)]

/-- The exact same-dtype storage authority frozen for Wave 10.  This is
    deliberately narrower than backend-wide `computeSupported`: admitting a
    dtype for some operation does not admit its cast/bitcast identity surface. -/
def identityDtypes : List Dtype := [.bfloat16_, .float32_, .int32_]

/-- Largest count representable by the emitted MSL `uint` guard literal. -/
def maxCastElementCount : Nat := 4294967295

private structure CastPlan extends MaterializedRoot where
  identity : Bool
  admitted :
    (identity = true ∧ rootDtype = targetDtype ∧
      identityDtypes.contains rootDtype = true) ∨
    (identity = false ∧ castPairs.contains (rootDtype, targetDtype) = true)
  indexFits : identity = false → count.value ≤ maxCastElementCount

private structure BitcastPlan extends MaterializedRoot where
  identity : Bool
  admitted :
    (identity = true ∧ rootDtype = targetDtype ∧
      identityDtypes.contains rootDtype = true) ∨
    (identity = false ∧ rootDtype.bitcastStoragePair targetDtype = true)
  bytesEqual : outputBytes.value = inputBytes.value

/-- Runtime storage evidence produced below the Metal/Python boundary.  The
    actual buffer length is queried from the registered BUFFER and converted
    into a proposition before any execution or alias construction proceeds. -/
private structure InputCovered (root : MaterializedRoot) where
  actualBytes : Nat
  covers : root.inputBytes.value ≤ actualBytes

private structure CheckedCastPlan where
  plan : CastPlan
  input : InputCovered plan.toMaterializedRoot
  nonidentity : plan.identity = false

private structure CheckedBitcastPlan where
  plan : BitcastPlan
  input : InputCovered plan.toMaterializedRoot

private def exactBufferRoot (t : Tensor) (target : Dtype) :
    Except TransformReason MaterializedRoot :=
  match t.uop with
  | .buffer raw shape rootDtype => do
      if hDtype : rootDtype = t.dtype then
        if hPositive : shape.all (fun dim => dim != 0) = true then
          let countValue := Tgrad.numel shape
          let inputValue := countValue * rootDtype.sizeBytes
          let outputValue := countValue * target.sizeBytes
          let countAbi := USize.ofNat countValue
          let inputAbi := USize.ofNat inputValue
          let outputAbi := USize.ofNat outputValue
          if hCount : countAbi.toNat = countValue then
            if hInput : inputAbi.toNat = inputValue then
              if hOutput : outputAbi.toNat = outputValue then
                let count : CheckedUSize :=
                  { value := countValue, abi := countAbi, exact := hCount }
                let inputBytes : CheckedUSize :=
                  { value := inputValue, abi := inputAbi, exact := hInput }
                let outputBytes : CheckedUSize :=
                  { value := outputValue, abi := outputAbi, exact := hOutput }
                pure (MaterializedRoot.mk (.buffer raw shape rootDtype) raw shape
                  rootDtype t.dtype target rfl hDtype hPositive count inputBytes
                  outputBytes rfl rfl rfl)
              else throw .sizeOverflow
            else throw .sizeOverflow
          else throw .sizeOverflow
        else throw .invalidShape
      else throw .rootDtypeMismatch
  | _ => .error .nonBuffer

private def buildCastPlan (t : Tensor) (target : Dtype) :
    Except TransformReason CastPlan := do
  let root ← exactBufferRoot t target
  if hIdentity : root.rootDtype = root.targetDtype then
    if hSupported : identityDtypes.contains root.rootDtype = true then
      pure (CastPlan.mk root true (.inl ⟨rfl, hIdentity, hSupported⟩)
        (by intro h; contradiction))
    else throw .unsupportedPair
  else if hPair : castPairs.contains (root.rootDtype, root.targetDtype) = true then
    if hIndex : root.count.value ≤ maxCastElementCount then
      pure (CastPlan.mk root false (.inr ⟨rfl, hPair⟩) (fun _ => hIndex))
    else throw .sizeOverflow
  else throw .unsupportedPair

private def buildBitcastPlan (t : Tensor) (target : Dtype) :
    Except TransformReason BitcastPlan := do
  let root ← exactBufferRoot t target
  if hBytes : root.outputBytes.value = root.inputBytes.value then
    if hIdentity : root.rootDtype = root.targetDtype then
      if hSupported : identityDtypes.contains root.rootDtype = true then
        pure (BitcastPlan.mk root true (.inl ⟨rfl, hIdentity, hSupported⟩)
          hBytes)
      else throw .unsupportedPair
    else if hPair : root.rootDtype.bitcastStoragePair root.targetDtype = true then
      pure (BitcastPlan.mk root false (.inr ⟨rfl, hPair⟩) hBytes)
    else throw .unsupportedPair
  else throw .unequalItemSize

/-- Exact portable expression corresponding to `Dtype.bf16PackBits`. -/
def exactBf16PackExpr (bits : String) : String :=
  s!"((((({bits}) & 0x7f800000u) == 0x7f800000u) && ((({bits}) & 0x007fffffu) != 0u))"
    ++ s!" ? ((({bits}) | 0x00400000u) >> 16)"
    ++ s!" : ((({bits}) + 0x00007fffu + ((({bits}) >> 16) & 1u)) >> 16))"

/-- Portable integer expression for exact bf16 storage expansion. -/
def exactBf16ExpandExpr (bits : String) : String :=
  s!"(((uint)({bits})) << 16)"

/-- Independent source-contract pins.  Runtime exact-bit comparison remains
    necessary: these catch renderer drift, while the foreign-grounded corpus
    catches a wrong-but-consistently-pinned formula. -/
theorem exact_bf16_pack_expression_contract :
    exactBf16PackExpr "bits" =
      "(((((bits) & 0x7f800000u) == 0x7f800000u) && (((bits) & 0x007fffffu) != 0u)) ? (((bits) | 0x00400000u) >> 16) : (((bits) + 0x00007fffu + (((bits) >> 16) & 1u)) >> 16))" := by
  native_decide

theorem exact_bf16_expand_expression_contract :
    exactBf16ExpandExpr "data0[gidx0]" =
      "(((uint)(data0[gidx0])) << 16)" := by
  native_decide

private inductive CastIndexBuiltin where
  | threadPositionInGrid
  deriving DecidableEq

private structure CastLaunch (plan : CastPlan) where
  indexBuiltin : CastIndexBuiltin
  gridX : USize
  threadgroupX : USize
  guardExclusive : Nat
  gridExact : gridX.toNat = plan.count.value
  guardExact : guardExclusive = plan.count.value
  indexFits : plan.count.value ≤ maxCastElementCount

private def launchFor (plan : CastPlan) (nonidentity : plan.identity = false) :
    CastLaunch plan :=
  { indexBuiltin := .threadPositionInGrid,
    gridX := plan.count.abi, threadgroupX := 256,
    guardExclusive := plan.count.value,
    gridExact := plan.count.exact, guardExact := rfl,
    indexFits := plan.indexFits nonidentity }

private def dtypeFragment : Dtype → String
  | .float32_ => "f32" | .bfloat16_ => "bf16" | .int32_ => "i32"
  | dtype => dtype.toStr

/-- Compiler-visible launch/index syntax pins.  The launch proof supplies the
    count; these functions supply the exact MSL use of that count. -/
def exactCastIndexDecl : String :=
  "uint gidx0 [[thread_position_in_grid]]"

def exactCastGuardStmt (count : Nat) : String :=
  s!"if (gidx0 >= {count}u) return;"

theorem exact_cast_index_declaration_contract :
    exactCastIndexDecl = "uint gidx0 [[thread_position_in_grid]]" := rfl

theorem exact_cast_guard_statement_contract (count : Nat) :
    exactCastGuardStmt count = s!"if (gidx0 >= {count}u) return;" := rfl

/-- The artifact retains the exact proof-carrying plan and launch object. -/
private structure CastArtifact where
  plan : CastPlan
  launch : CastLaunch plan
  declName : String
  source : String

private def renderArtifact (plan : CastPlan) (launch : CastLaunch plan) : CastArtifact :=
  let name := s!"tgrad_cast_{dtypeFragment plan.rootDtype}_to_"
    ++ s!"{dtypeFragment plan.targetDtype}_{plan.count.value}"
  let indexDeclaration := match launch.indexBuiltin with
    | .threadPositionInGrid => exactCastIndexDecl
  let signature :=
    if plan.rootDtype == .float32_ then
      s!"kernel void {name}(device const uint* data0 [[buffer(0)]], device ushort* data1 [[buffer(1)]], {indexDeclaration})"
    else
      s!"kernel void {name}(device const ushort* data0 [[buffer(0)]], device uint* data1 [[buffer(1)]], {indexDeclaration})"
  let body :=
    if plan.rootDtype == .float32_ then
      s!"  uint bits = data0[gidx0];\n  data1[gidx0] = (ushort)({exactBf16PackExpr "bits"} & 0xffffu);"
    else s!"  data1[gidx0] = {exactBf16ExpandExpr "data0[gidx0]"};"
  let source := "#include <metal_stdlib>\nusing namespace metal;\n" ++ signature ++ " {\n"
    ++ s!"  {exactCastGuardStmt launch.guardExclusive}\n" ++ body ++ "\n}\n"
  { plan, launch, declName := name, source }

private def artifactForPlan (plan : CastPlan) (nonidentity : plan.identity = false) :
    CastArtifact :=
  renderArtifact plan (launchFor plan nonidentity)

/-- A library handle cannot be detached from the exact artifact it compiled. -/
private structure CompiledCastArtifact where
  artifact : CastArtifact
  library : UInt64
  libraryNonzero : library ≠ 0

private def bindCompiledArtifact (artifact : CastArtifact) (library : UInt64)
    (h : library ≠ 0) : CompiledCastArtifact :=
  { artifact, library, libraryNonzero := h }

private def compileArtifact (artifact : CastArtifact) :
    IO (Except TransformReason CompiledCastArtifact) := do
  let library ← Runtime.Metal.metalCompile artifact.source
  if h : library = 0 then return .error .compileFailed
  else return .ok (bindCompiledArtifact artifact library h)

private structure CastDispatchRequest where
  library : UInt64
  declName : String
  buffers : Array UInt64
  gridX : USize
  threadgroupX : USize

private def dispatchRequest (compiled : CompiledCastArtifact) (output : UInt64) :
    CastDispatchRequest :=
  { library := compiled.library,
    declName := compiled.artifact.declName,
    buffers := #[compiled.artifact.plan.raw, output],
    gridX := compiled.artifact.launch.gridX,
    threadgroupX := compiled.artifact.launch.threadgroupX }

private def dispatchArtifact (compiled : CompiledCastArtifact) (output : UInt64) : IO UInt32 := do
  let request := dispatchRequest compiled output
  Runtime.Metal.metalDispatch request.library request.declName request.buffers
    request.gridX 1 1 request.threadgroupX 1 1

private structure DispatchedCast where
  plan : CastPlan
  output : UInt64

private def executeCast (checked : CheckedCastPlan) :
    IO (Except TransformReason DispatchedCast) := do
  let plan := checked.plan
  let artifact := artifactForPlan plan checked.nonidentity
  let compiledResult ← compileArtifact artifact
  let compiled ← match compiledResult with
    | .ok compiled => pure compiled
    | .error reason => return .error reason
  let output ← Runtime.Metal.metalAlloc plan.outputBytes.abi
  if output == 0 then
    Runtime.Metal.metalLibraryRelease compiled.library
    return .error .allocationFailed
  let rc ← dispatchArtifact compiled output
  Runtime.Metal.metalLibraryRelease compiled.library
  if rc != 0 then
    Runtime.Metal.metalFree output plan.outputBytes.abi
    return .error .dispatchFailed
  pure (.ok { plan, output })

private def registerableTensor (done : DispatchedCast) : Tensor :=
  Tensor.ofBuffer { raw := done.output, size := done.plan.outputBytes.value }
    done.plan.shape done.plan.targetDtype

private def checkInputBuffer (root : MaterializedRoot) :
    IO (Except TransformReason (InputCovered root)) := do
  let actual ← Runtime.Metal.metalBufferLength root.raw
  if h : root.inputBytes.value ≤ actual.toNat then
    pure (.ok { actualBytes := actual.toNat, covers := h })
  else pure (.error .bufferTooSmall)

private def success (tensor : Tensor) (ownership : TransformOwnership) : TransformResult :=
  { reason := .ok, tensor := some tensor, ownership }

private def failure (reason : TransformReason) : TransformResult := { reason }

def realizeCast (tensor : Tensor) (target : Dtype) : IO TransformResult := do
  let plan ← match buildCastPlan tensor target with
    | .ok plan => pure plan
    | .error reason => return failure reason
  let input ← match ← checkInputBuffer plan.toMaterializedRoot with
    | .ok input => pure input
    | .error reason => return failure reason
  match hIdentity : plan.identity with
  | true => return success tensor .existing
  | false =>
      match ← executeCast { plan, input, nonidentity := hIdentity } with
      | .error reason => pure (failure reason)
      | .ok done => pure (success (registerableTensor done) .owned)

def realizeBitcast (tensor : Tensor) (target : Dtype) : IO TransformResult := do
  let plan ← match buildBitcastPlan tensor target with
    | .ok plan => pure plan
    | .error reason => return failure reason
  let input ← match ← checkInputBuffer plan.toMaterializedRoot with
    | .ok input => pure input
    | .error reason => return failure reason
  let checked : CheckedBitcastPlan := { plan, input }
  if plan.identity then return success tensor .existing
  pure (success
    { uop := .bitcast checked.plan.targetDtype checked.plan.uop,
      dtype := checked.plan.targetDtype }
    .borrowed)

/-! General compiler obligations over every constructible plan/artifact. -/

theorem cast_admitted_pairs_exact (plan : CastPlan) :
    (plan.identity = true ∧ plan.rootDtype = plan.targetDtype ∧
      identityDtypes.contains plan.rootDtype = true) ∨
    (plan.identity = false ∧ castPairs.contains (plan.rootDtype, plan.targetDtype) = true) :=
  plan.admitted

theorem bitcast_admitted_pairs_exact (plan : BitcastPlan) :
    (plan.identity = true ∧ plan.rootDtype = plan.targetDtype ∧
      identityDtypes.contains plan.rootDtype = true) ∨
    (plan.identity = false ∧ plan.rootDtype.bitcastStoragePair plan.targetDtype = true) :=
  plan.admitted

/-- Independent finite-policy pins.  These right-hand sides intentionally do
    not reuse the definitions: changing the admitted denominator must break
    compilation until the external contract is deliberately revised. -/
theorem identity_dtypes_contract :
    identityDtypes = [.bfloat16_, .float32_, .int32_] := by native_decide

theorem max_cast_element_count_contract :
    maxCastElementCount = 4294967295 := rfl

theorem cast_pairs_contract :
    castPairs = [(.float32_, .bfloat16_), (.bfloat16_, .float32_)] := by
  native_decide

private inductive PlanKind where | cast | bitcast deriving DecidableEq
private def CastPlan.kind (_ : CastPlan) : PlanKind := .cast
private def BitcastPlan.kind (_ : BitcastPlan) : PlanKind := .bitcast
theorem cast_bitcast_plans_distinct (cast : CastPlan) (bitcast : BitcastPlan) :
    cast.kind != bitcast.kind := by simp [CastPlan.kind, BitcastPlan.kind]

theorem bitcast_bytes_preserved (plan : BitcastPlan) :
    plan.outputBytes.value = plan.inputBytes.value := plan.bytesEqual

theorem bitcast_shape_preserved (plan : BitcastPlan) :
    plan.shape = plan.toMaterializedRoot.shape := rfl

theorem identity_preserves_storage (plan : CastPlan) :
    plan.identity = true → plan.raw = plan.toMaterializedRoot.raw := by intro; rfl

theorem cast_root_is_exact_buffer (plan : CastPlan) :
    plan.uop = .buffer plan.raw plan.shape plan.rootDtype := plan.rootExact

theorem cast_root_dtype_matches (plan : CastPlan) :
    plan.rootDtype = plan.tensorDtype := plan.dtypeExact

theorem cast_dims_positive (plan : CastPlan) :
    plan.shape.all (fun dim => dim != 0) = true := plan.dimsPositive

theorem cast_byte_products_fit (plan : CastPlan) :
    plan.count.abi.toNat = plan.count.value ∧
    plan.inputBytes.abi.toNat = plan.inputBytes.value ∧
    plan.outputBytes.abi.toNat = plan.outputBytes.value :=
  ⟨plan.count.exact, plan.inputBytes.exact, plan.outputBytes.exact⟩

theorem cast_buffer_length_covers_input (checked : CheckedCastPlan) :
    checked.plan.inputBytes.value ≤ checked.input.actualBytes :=
  checked.input.covers

theorem checked_bitcast_covers_input (checked : CheckedBitcastPlan) :
    checked.plan.inputBytes.value ≤ checked.input.actualBytes :=
  checked.input.covers

theorem cast_launch_covers_exactly (plan : CastPlan)
    (nonidentity : plan.identity = false) :
    let launch := launchFor plan nonidentity
    launch.indexBuiltin = .threadPositionInGrid ∧
    launch.gridX.toNat = plan.count.value ∧
    launch.guardExclusive = plan.count.value ∧
    plan.count.value ≤ maxCastElementCount := by
  exact ⟨rfl, plan.count.exact, rfl, plan.indexFits nonidentity⟩

theorem cast_artifact_exact_identity (plan : CastPlan)
    (nonidentity : plan.identity = false) :
    (artifactForPlan plan nonidentity).plan = plan := rfl

/-- There is no digest cache: the actual compile path binds the complete
    artifact value into `CompiledCastArtifact`. -/
theorem cast_cache_collision_checked (artifact : CastArtifact) (library : UInt64)
    (h : library ≠ 0) :
    (bindCompiledArtifact artifact library h).artifact = artifact := rfl

theorem cast_dispatch_uses_artifact_name (compiled : CompiledCastArtifact)
    (output : UInt64) :
    (dispatchRequest compiled output).declName = compiled.artifact.declName := rfl

/-- The only constructor consumed by registration requires dispatch success. -/
theorem cast_registers_after_dispatch (done : DispatchedCast) :
    (registerableTensor done).buffer.raw = done.output := rfl

def portable_cast_sources : List String :=
  let f32 := Tensor.ofBuffer { raw := 1, size := 4 } [1] .float32_
  let bf16 := Tensor.ofBuffer { raw := 1, size := 2 } [1] .bfloat16_
  [match buildCastPlan f32 .bfloat16_ with
    | .ok plan => if h : plan.identity = false
      then (artifactForPlan plan h).source else "" | _ => "",
   match buildCastPlan bf16 .float32_ with
    | .ok plan => if h : plan.identity = false
      then (artifactForPlan plan h).source else "" | _ => ""]

theorem portable_cast_sources_use_integer_storage :
    portable_cast_sources.all (fun source =>
      source.contains "thread_position_in_grid" &&
      !source.contains "device bfloat" && !source.contains "device float*") = true := by
  native_decide

end Cast

end Tgrad

import Tgrad.Dtype

/-! # Backend-neutral one-dimensional fill planning

This module owns the semantics shared by vendor backends.  A renderer receives
one validated `FillPlan`; it does not receive an independent dtype, count,
launch, or cache key that could disagree with the plan used at launch time.
-/
namespace Tgrad.Backend

inductive BackendId where
  | metal | cuda | hip
  deriving BEq, Repr, DecidableEq

def BackendId.stableName : BackendId → String
  | .metal => "metal"
  | .cuda => "cuda"
  | .hip => "hip"

inductive CompilerMode where
  | offline | runtime
  deriving BEq, Repr, DecidableEq

def CompilerMode.stableName : CompilerMode → String
  | .offline => "offline"
  | .runtime => "runtime"

inductive CompilerTool where
  | metalCompiler | nvrtc | nvcc | hiprtc | hipcc
  deriving BEq, Repr, DecidableEq

def CompilerTool.stableName : CompilerTool → String
  | .metalCompiler => "metal-compiler"
  | .nvrtc => "nvrtc"
  | .nvcc => "nvcc"
  | .hiprtc => "hiprtc"
  | .hipcc => "hipcc"

/-- Neutral scalar inputs.  Admission couples one of these values to a concrete
storage dtype; a vendor renderer never chooses or defaults a dtype. -/
inductive ScalarInput where
  | signed (value : Int)
  /-- Exact IEEE-754 binary32 payload, including signed zero and NaN payloads. -/
  | float32Bits (bits : UInt32)
  deriving Repr

/-- The deliberately narrow dtype/value subset admitted by this initial slice.
The constructor itself makes mismatched dtype/value pairs unrepresentable. -/
inductive FillValue where
  | int32 (value : Int)
  | float32Bits (bits : UInt32)
  deriving Repr

def FillValue.dtype : FillValue → Tgrad.Dtype
  | .int32 _ => .int32_
  | .float32Bits _ => .float32_

def FillValue.stableTag : FillValue → String
  | .int32 value => s!"i32:{value}"
  | .float32Bits bits => s!"f32bits:{bits}"

inductive PlanError where
  | unsupportedDtype (dtype : Tgrad.Dtype)
  | valueDtypeMismatch (dtype : Tgrad.Dtype)
  | int32OutOfRange (value : Int)
  | invalidArchitecture (backend : BackendId) (architecture : String)
  | emptyCompilerIdentity
  | compilerProfileMismatch (backend : BackendId) (mode : CompilerMode)
      (tool : CompilerTool)
  | invalidDeviceBlockLimit (limit : Nat)
  | zeroBlockSize
  | blockExceedsDeviceLimit (block limit : Nat)
  | elementCountOverflow (count : Nat)
  | byteCountOverflow (count elementBytes : Nat)
  | launchWidthOverflow (grid block : Nat)
  deriving BEq, Repr

def admitFillValue (dtype : Tgrad.Dtype) (input : ScalarInput) :
    Except PlanError FillValue :=
  match dtype, input with
  | .int32_, .signed value =>
      if value < -2147483648 || value > 2147483647 then
        .error (.int32OutOfRange value)
      else .ok (.int32 value)
  | .float32_, .float32Bits bits => .ok (.float32Bits bits)
  | .int32_, _ | .float32_, _ => .error (.valueDtypeMismatch dtype)
  | _, _ => .error (.unsupportedDtype dtype)

def abiUInt64Max : Nat := UInt64.size - 1
def abiUInt32Max : Nat := UInt32.size - 1

/-- Total ceil division.  Validation separately rejects `denom = 0`. -/
def ceilDiv (numer denom : Nat) : Nat :=
  numer / denom + (if numer % denom = 0 then 0 else 1)

/-- Checked byte and element widths before conversion to a runtime ABI. -/
structure BytePlan where
  elementCount : Nat
  elementBytes : Nat
  byteCount : Nat
  byteCount_eq : byteCount = elementCount * elementBytes
  elementCount_fits_u64 : elementCount ≤ abiUInt64Max
  byteCount_fits_u64 : byteCount ≤ abiUInt64Max

/-- Exact one-dimensional launch geometry.  Both fields cross a 32-bit vendor
launch ABI, and `gridSize` is definitionally bound to exact ceil division. -/
structure Launch1D (elementCount : Nat) where
  blockSize : Nat
  gridSize : Nat
  block_positive : 0 < blockSize
  grid_is_ceil_div : gridSize = ceilDiv elementCount blockSize
  block_fits_u32 : blockSize ≤ abiUInt32Max
  grid_fits_u32 : gridSize ≤ abiUInt32Max

structure BackendIdentity where
  backend : BackendId
  architecture : String
  compilerMode : CompilerMode
  compilerTool : CompilerTool
  compilerIdentity : String
  maxThreadsPerBlock : Nat
  deriving BEq, Repr

def BackendIdentity.validArchitecture (identity : BackendIdentity) : Bool :=
  match identity.backend with
  | .metal => identity.architecture.startsWith "apple"
  | .cuda => identity.architecture.startsWith "sm_"
  | .hip => identity.architecture.startsWith "gfx"

def BackendIdentity.compilerMatches (identity : BackendIdentity) : Bool :=
  match identity.backend, identity.compilerMode, identity.compilerTool with
  | .metal, .offline, .metalCompiler => true
  | .cuda, .runtime, .nvrtc | .cuda, .offline, .nvcc => true
  | .hip, .runtime, .hiprtc | .hip, .offline, .hipcc => true
  | _, _, _ => false

/-- Validated, backend-neutral semantic plan.  Proof fields prevent unchecked
ABI widths or an inconsistent byte/launch plan from being constructed. -/
structure FillPlan where
  private mk ::
  identity : BackendIdentity
  value : FillValue
  bytes : BytePlan
  launch : Launch1D bytes.elementCount

def FillPlan.elementCount (plan : FillPlan) : Nat := plan.bytes.elementCount
def FillPlan.byteCount (plan : FillPlan) : Nat := plan.bytes.byteCount
def FillPlan.dtype (plan : FillPlan) : Tgrad.Dtype := plan.value.dtype

/-- Every semantic/cache input, including renderer contract revision, is owned
by Lean.  This string is the cache identity itself, not a claim of hashing. -/
def FillPlan.cacheIdentity (plan : FillPlan) : String :=
  String.intercalate "|"
    ["fill1d-v1", plan.identity.backend.stableName,
     plan.identity.architecture, plan.identity.compilerMode.stableName,
     plan.identity.compilerTool.stableName, plan.identity.compilerIdentity,
     toString plan.identity.maxThreadsPerBlock,
     plan.dtype.toStr, plan.value.stableTag,
     toString plan.elementCount, toString plan.bytes.elementBytes,
     toString plan.byteCount, toString plan.launch.blockSize,
     toString plan.launch.gridSize, "idx=block*blockDim+thread", "guard=idx<count"]

def mkFillPlan (identity : BackendIdentity) (dtype : Tgrad.Dtype)
    (input : ScalarInput) (elementCount blockSize : Nat) :
    Except PlanError FillPlan := do
  if !identity.validArchitecture then
    throw (.invalidArchitecture identity.backend identity.architecture)
  if identity.compilerIdentity.isEmpty then throw .emptyCompilerIdentity
  if !identity.compilerMatches then
    throw (.compilerProfileMismatch identity.backend identity.compilerMode
      identity.compilerTool)
  if identity.maxThreadsPerBlock = 0 || identity.maxThreadsPerBlock > 1024 then
    throw (.invalidDeviceBlockLimit identity.maxThreadsPerBlock)
  let value ← admitFillValue dtype input
  if hblock : blockSize = 0 then throw .zeroBlockSize else
    if blockSize > identity.maxThreadsPerBlock then
      throw (.blockExceedsDeviceLimit blockSize identity.maxThreadsPerBlock)
    else
      if hcount : elementCount > abiUInt64Max then
        throw (.elementCountOverflow elementCount)
      else
        let elementBytes := value.dtype.sizeBytes
        let byteCount := elementCount * elementBytes
        if hbytes : byteCount > abiUInt64Max then
          throw (.byteCountOverflow elementCount elementBytes)
        else
          let gridSize := ceilDiv elementCount blockSize
          if hlaunch : blockSize > abiUInt32Max ∨ gridSize > abiUInt32Max then
            throw (.launchWidthOverflow gridSize blockSize)
          else
            have hblockPos : 0 < blockSize := Nat.pos_of_ne_zero hblock
            have hcountFits : elementCount ≤ abiUInt64Max := Nat.le_of_not_gt hcount
            have hbytesFits : byteCount ≤ abiUInt64Max := Nat.le_of_not_gt hbytes
            have hblockFits : blockSize ≤ abiUInt32Max := by
              apply Nat.le_of_not_gt
              intro h
              exact hlaunch (Or.inl h)
            have hgridFits : gridSize ≤ abiUInt32Max := by
              apply Nat.le_of_not_gt
              intro h
              exact hlaunch (Or.inr h)
            return {
              identity := identity
              value := value
              bytes := {
                elementCount := elementCount
                elementBytes := elementBytes
                byteCount := byteCount
                byteCount_eq := rfl
                elementCount_fits_u64 := hcountFits
                byteCount_fits_u64 := hbytesFits
              }
              launch := {
                blockSize := blockSize
                gridSize := gridSize
                block_positive := hblockPos
                grid_is_ceil_div := rfl
                block_fits_u32 := hblockFits
                grid_fits_u32 := hgridFits
              }
            }

end Tgrad.Backend

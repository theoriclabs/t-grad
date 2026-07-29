import Tgrad.Dtype

/-! # Backend-neutral one-dimensional fill planning

This module is the semantic spine of the first non-Metal backend slice.  It
does not emit a vendor language or call a runtime.  Lean admits the dtype and
value, checks byte and launch arithmetic against ABI widths, and constructs
the complete cache identity.  Vendor leaves may consume a `FillPlan`; they do
not get to recompute any of these decisions.
-/
namespace Tgrad
namespace Backend

inductive Id where
  | metal | cuda | hip
  deriving BEq, Repr, DecidableEq

def Id.tag : Id → String
  | .metal => "metal"
  | .cuda => "cuda"
  | .hip => "hip"

inductive CompilerMode where
  | metalOffline (version : String)
  | nvrtc (version : String)
  | nvcc (version : String)
  | hiprtc (version : String)
  | hipcc (version : String)
  deriving BEq, Repr, DecidableEq

def CompilerMode.tag : CompilerMode → String
  | .metalOffline version => "metal-offline:" ++ version
  | .nvrtc version => "nvrtc:" ++ version
  | .nvcc version => "nvcc:" ++ version
  | .hiprtc version => "hiprtc:" ++ version
  | .hipcc version => "hipcc:" ++ version

def CompilerMode.supports : CompilerMode → Id → Bool
  | .metalOffline _, .metal => true
  | .nvrtc _, .cuda => true
  | .nvcc _, .cuda => true
  | .hiprtc _, .hip => true
  | .hipcc _, .hip => true
  | _, _ => false

/-- A backend/compiler/architecture identity supplied by a runtime probe or a
focused static test.  `maxThreadsPerBlock` is explicit: launch admission never
silently assumes a different device's limit. -/
structure Profile where
  backend : Id
  architecture : String
  compiler : CompilerMode
  maxThreadsPerBlock : Nat
  deriving BEq, Repr

def Profile.valid (profile : Profile) : Bool :=
  profile.compiler.supports profile.backend &&
  !profile.architecture.trimAscii.isEmpty &&
  profile.maxThreadsPerBlock > 0 &&
  profile.maxThreadsPerBlock ≤ 4294967295

def maxAbiU32 : Nat := 4294967295
def maxAbiU64 : Nat := 18446744073709551615

/-- A natural number accompanied by the proof needed for lossless UInt32 ABI
marshalling. -/
structure AbiU32 where
  value : Nat
  fits : value ≤ maxAbiU32

/-- A natural number accompanied by the proof needed for lossless UInt64 ABI
marshalling. -/
structure AbiU64 where
  value : Nat
  fits : value ≤ maxAbiU64

structure Int32Value where
  value : Int
  lower : (-2147483648 : Int) ≤ value
  upper : value ≤ (2147483647 : Int)

/-- Exact admitted storage value.  Float32 uses its 32-bit storage pattern so
NaNs, signed zero, infinities, value emission, and cache identity cannot be
silently changed by host Float formatting. -/
inductive FillValue where
  | int32 (value : Int32Value)
  | float32Bits (bits : UInt32)

def FillValue.dtype : FillValue → Dtype
  | .int32 _ => .int32_
  | .float32Bits _ => .float32_

def FillValue.scalarTag : FillValue → String
  | .int32 value => "i32:" ++ toString value.value
  | .float32Bits bits => "f32bits:" ++ toString bits.toNat

inductive RawFillValue where
  | signed (value : Int)
  | float32Bits (bits : UInt32)
  deriving BEq, Repr

inductive FillPlanError where
  | unsupportedDtype (dtype : Dtype)
  | dtypeValueMismatch (dtype : Dtype) (value : RawFillValue)
  | int32OutOfRange (value : Int)
  | invalidProfile
  | elementCountOutOfRange (count : Nat)
  | byteCountOverflow (count bytesPerElement : Nat)
  | invalidBlockSize (threads limit : Nat)
  | gridSizeOutOfRange (blocks : Nat)
  | invalidConstructedPlan
  deriving BEq, Repr

def admitValue (dtype : Dtype) (raw : RawFillValue) :
    Except FillPlanError FillValue :=
  match dtype, raw with
  | .int32_, .signed value =>
      if lower : (-2147483648 : Int) ≤ value then
        if upper : value ≤ (2147483647 : Int) then
          .ok (.int32 { value, lower, upper })
        else .error (.int32OutOfRange value)
      else .error (.int32OutOfRange value)
  | .float32_, .float32Bits bits => .ok (.float32Bits bits)
  | .int32_, raw => .error (.dtypeValueMismatch dtype raw)
  | .float32_, raw => .error (.dtypeValueMismatch dtype raw)
  | unsupported, _ => .error (.unsupportedDtype unsupported)

def ceilDiv (n d : Nat) : Nat :=
  if n == 0 then 0 else ((n - 1) / d) + 1

inductive OutputIndex where
  | globalLinear1D
  deriving BEq, Repr, DecidableEq

def OutputIndex.tag : OutputIndex → String
  | .globalLinear1D => "global-linear-1d"

/-- The only constructible execution plan consumed by vendor renderers.  The
proof fields make checked byte products and exact launch geometry compiler
obligations, not test-only comments. -/
structure FillPlan where
  profile : Profile
  value : FillValue
  elementCount : AbiU64
  byteCount : AbiU64
  threadsPerBlock : AbiU32
  blocksPerGrid : AbiU32
  outputIndex : OutputIndex
  threadsPositive : 0 < threadsPerBlock.value
  threadsWithinProfile : threadsPerBlock.value ≤ profile.maxThreadsPerBlock
  bytesExact : byteCount.value = elementCount.value * value.dtype.sizeBytes
  launchExact :
    blocksPerGrid.value = ceilDiv elementCount.value threadsPerBlock.value

def FillPlan.build (profile : Profile) (dtype : Dtype) (raw : RawFillValue)
    (elementCount threadsPerBlock : Nat) : Except FillPlanError FillPlan := do
  if !profile.valid then throw .invalidProfile
  let value ← admitValue dtype raw
  if countFits : elementCount ≤ maxAbiU64 then
    let bytes := elementCount * value.dtype.sizeBytes
    if bytesFit : bytes ≤ maxAbiU64 then
      if threadsPositive : 0 < threadsPerBlock then
        if threadsFitProfile : threadsPerBlock ≤ profile.maxThreadsPerBlock then
          if threadsFitAbi : threadsPerBlock ≤ maxAbiU32 then
            let blocks := ceilDiv elementCount threadsPerBlock
            if blocksFit : blocks ≤ maxAbiU32 then
              return {
                profile
                value
                elementCount := { value := elementCount, fits := countFits }
                byteCount := { value := bytes, fits := bytesFit }
                threadsPerBlock := { value := threadsPerBlock, fits := threadsFitAbi }
                blocksPerGrid := { value := blocks, fits := blocksFit }
                outputIndex := .globalLinear1D
                threadsPositive
                threadsWithinProfile := threadsFitProfile
                bytesExact := rfl
                launchExact := rfl
              }
            else throw (FillPlanError.gridSizeOutOfRange blocks)
          else throw (FillPlanError.invalidBlockSize threadsPerBlock profile.maxThreadsPerBlock)
        else throw (FillPlanError.invalidBlockSize threadsPerBlock profile.maxThreadsPerBlock)
      else throw (FillPlanError.invalidBlockSize threadsPerBlock profile.maxThreadsPerBlock)
    else throw (FillPlanError.byteCountOverflow elementCount value.dtype.sizeBytes)
  else throw (FillPlanError.elementCountOutOfRange elementCount)

def FillValue.toRaw : FillValue → RawFillValue
  | .int32 value => .signed value.value
  | .float32Bits bits => .float32Bits bits

/-- Consumer-side validation closes the public-structure construction escape
hatch.  Proof fields already make inconsistent arithmetic unconstructible in
safe Lean; this executable check additionally protects every foreign/runtime
consumer from forged or unsafely decoded values. -/
def FillPlan.revalidate (plan : FillPlan) : Except FillPlanError Unit :=
  match FillPlan.build plan.profile plan.value.dtype plan.value.toRaw
      plan.elementCount.value plan.threadsPerBlock.value with
  | .error error => .error error
  | .ok rebuilt =>
      if rebuilt.byteCount.value == plan.byteCount.value &&
         rebuilt.blocksPerGrid.value == plan.blocksPerGrid.value then
        .ok ()
      else .error .invalidConstructedPlan

/-- Stable, complete compilation-cache identity.  Backend, architecture,
compiler mode/version, dtype, constant, bytes, and launch semantics all take
part; a vendor leaf cannot collapse any of them. -/
def FillPlan.cacheIdentity (plan : FillPlan) : String :=
  String.intercalate "|" [
    "schema=fill1d-v1",
    "backend=" ++ plan.profile.backend.tag,
    "arch=" ++ plan.profile.architecture,
    "compiler=" ++ plan.profile.compiler.tag,
    "dtype=" ++ plan.value.dtype.toStr,
    "value=" ++ plan.value.scalarTag,
    "index=" ++ plan.outputIndex.tag,
    "count=" ++ toString plan.elementCount.value,
    "bytes=" ++ toString plan.byteCount.value,
    "threads=" ++ toString plan.threadsPerBlock.value,
    "blocks=" ++ toString plan.blocksPerGrid.value]

private def identifierEncode (value : String) : String :=
  String.intercalate "_" (value.toList.map (fun char => toString char.toNat))

/-- The symbol identity is an injective identifier-safe encoding of the full
cache identity, rather than a digest that could hide a collision. -/
def FillPlan.kernelName (plan : FillPlan) : String :=
  "tgrad_fill1d_" ++ identifierEncode plan.cacheIdentity

end Backend
end Tgrad

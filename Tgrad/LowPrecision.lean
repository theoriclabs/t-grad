import Tgrad.Dtype

/-! # Lean-owned low-precision scalar conversion

This module is a closed, cache-free authority over IEEE binary64 bit payloads.
Python and C transport integers only; all descriptor selection, admission,
rounding, special-value handling and result bits live here.
-/
namespace Tgrad.LowPrecision

inductive Error where
  | unsupportedDtype
  | unsupportedOperation
  | invalidPayload
  | invalidValue
  | valueOverflow
  deriving BEq, Repr, DecidableEq

def Error.code : Error → UInt64
  | .unsupportedDtype => 1
  | .unsupportedOperation => 2
  | .invalidPayload => 3
  | .invalidValue => 4
  | .valueOverflow => 5

private structure Fp8Descriptor where
  dtype : Dtype
  bias : Nat
  sigBits : Nat
  mantMask : UInt64
  minDenormHalfBits : UInt64
  overflowThreshold : UInt64
  maxNormalPayload : UInt64
  minNormalBits : UInt64
  fnuz : Bool
  deriving BEq, Repr

private def descriptorFor? : Dtype → Option Fp8Descriptor
  | .fp8e4m3_ => some {
      dtype := .fp8e4m3_, bias := 7, sigBits := 4, mantMask := 0x7,
      minDenormHalfBits := 0x3F50000000000000,
      overflowThreshold := 0x407D000000000000,
      maxNormalPayload := 0x7E, minNormalBits := 0x3F90000000000000,
      fnuz := false }
  | .fp8e5m2_ => some {
      dtype := .fp8e5m2_, bias := 15, sigBits := 3, mantMask := 0x3,
      minDenormHalfBits := 0x3EE0000000000000,
      overflowThreshold := 0x40EDFFFFFFFFFFFF,
      maxNormalPayload := 0x7B, minNormalBits := 0x3F10000000000000,
      fnuz := false }
  | .fp8e4m3fnuz_ => some {
      dtype := .fp8e4m3fnuz_, bias := 8, sigBits := 4, mantMask := 0x7,
      minDenormHalfBits := 0x3F40000000000000,
      overflowThreshold := 0x406EFFFFFFFFFFFF,
      maxNormalPayload := 0x7F, minNormalBits := 0x3F80000000000000,
      fnuz := true }
  | .fp8e5m2fnuz_ => some {
      dtype := .fp8e5m2fnuz_, bias := 16, sigBits := 3, mantMask := 0x3,
      minDenormHalfBits := 0x3ED0000000000000,
      overflowThreshold := 0x40EDFFFFFFFFFFFF,
      maxNormalPayload := 0x7F, minNormalBits := 0x3F00000000000000,
      fnuz := true }
  | _ => none

private def fp8Dtypes : List Dtype :=
  [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_]

private def listNodup [BEq α] : List α → Bool
  | [] => true
  | x :: rest => !rest.contains x && listNodup rest

private def descriptorInvariant (dtype : Dtype) : Bool :=
  match descriptorFor? dtype with
  | none => false
  | some descriptor =>
      descriptor.dtype == dtype &&
      (descriptor.sigBits == 3 || descriptor.sigBits == 4) &&
      descriptor.mantMask == ((1 : UInt64) <<< UInt64.ofNat (descriptor.sigBits - 1)) - 1 &&
      Dtype.finfo dtype == some (8 - descriptor.sigBits, descriptor.sigBits - 1) &&
      descriptor.minDenormHalfBits < descriptor.minNormalBits &&
      descriptor.minNormalBits ≤ descriptor.overflowThreshold &&
      descriptor.fnuz == [.fp8e4m3fnuz_, .fp8e5m2fnuz_].contains dtype

def descriptorInvariantCheck : Bool := fp8Dtypes.all descriptorInvariant

theorem descriptor_invariants_hold : descriptorInvariantCheck = true := by
  native_decide

theorem supported_descriptor_dtypes_nodup : listNodup fp8Dtypes = true := by
  native_decide

private def pow2 (shift : Nat) : UInt64 :=
  (1 : UInt64) <<< UInt64.ofNat shift

private def signedPayload (bits : UInt64) : UInt64 :=
  ((bits >>> 63) &&& 1) <<< 7

private def fp8EncodeWith (descriptor : Fp8Descriptor) (bits : UInt64) : UInt64 :=
  let sign := signedPayload bits
  let rawExponent := (bits >>> 52) &&& 0x7FF
  let fraction := bits &&& 0x000FFFFFFFFFFFFF
  let absolute := bits &&& 0x7FFFFFFFFFFFFFFF
  let finite := rawExponent != 0x7FF
  if descriptor.fnuz && !finite then 0x80
  else if descriptor.fnuz && absolute == 0 then 0
  else if descriptor.dtype == .fp8e4m3_ && !finite then
    sign ||| 0x7F
  else if descriptor.dtype == .fp8e5m2_ && !finite then
    sign ||| (if fraction == 0 then 0x7C else 0x7F)
  else
    let exponent : Int := Int.ofNat rawExponent.toNat - 1023 + Int.ofNat descriptor.bias
    let mantissa := (bits >>> UInt64.ofNat (53 - descriptor.sigBits)) &&& descriptor.mantMask
    let halfUlp := pow2 (52 - descriptor.sigBits)
    let result :=
      if absolute ≤ descriptor.minDenormHalfBits then 0
      else if absolute > descriptor.overflowThreshold then descriptor.maxNormalPayload
      else if absolute ≥ descriptor.minNormalBits then
        let retained := (UInt64.ofNat exponent.toNat <<< UInt64.ofNat (descriptor.sigBits - 1)) ||| mantissa
        let roundBits := bits &&& ((halfUlp <<< 1) - 1)
        if roundBits > halfUlp || (roundBits == halfUlp && (mantissa &&& 1) == 1)
        then retained + 1 else retained
      else
        let shift := (1 - exponent).toNat
        let significand := mantissa ||| pow2 (descriptor.sigBits - 1)
        let retained := significand >>> UInt64.ofNat shift
        let half := halfUlp <<< UInt64.ofNat shift
        let roundBits := (bits ||| pow2 52) &&& ((half <<< 1) - 1)
        if roundBits > half || (roundBits == half && (retained &&& 1) == 1)
        then retained + 1 else retained
    if descriptor.fnuz && result == 0 then 0 else result ||| sign

def fp8EncodeBits (dtype : Dtype) (bits : UInt64) : Except Error UInt64 := do
  let some descriptor := descriptorFor? dtype | throw .unsupportedDtype
  pure (fp8EncodeWith descriptor bits)

private def floorLog2Small (value : UInt64) : Nat :=
  if value ≥ 4 then 2 else if value ≥ 2 then 1 else 0

private def canonicalNaN (negative : Bool := false) : UInt64 :=
  (if negative then (1 : UInt64) <<< 63 else 0) ||| 0x7FF8000000000000

private def fp8DecodeWith (descriptor : Fp8Descriptor) (payload : UInt64) : UInt64 :=
  let byte := payload &&& 0xFF
  let sign := (byte >>> 7) &&& 1
  let sign64 := sign <<< 63
  let absolute := byte &&& 0x7F
  let mantissaBits := descriptor.sigBits - 1
  let exponentBits := 8 - descriptor.sigBits
  let exponentMax := pow2 exponentBits - 1
  let mantissaMax := pow2 mantissaBits - 1
  let exponent := (byte >>> UInt64.ofNat mantissaBits) &&& exponentMax
  let mantissa := byte &&& mantissaMax
  if descriptor.fnuz && byte == 0x80 then canonicalNaN
  else if absolute == 0 then sign64
  else if !descriptor.fnuz && exponent == exponentMax && descriptor.dtype == .fp8e5m2_ then
    if mantissa == 0 then sign64 ||| 0x7FF0000000000000
    else canonicalNaN (sign == 1)
  else if !descriptor.fnuz && exponent == exponentMax && mantissa == mantissaMax then
    canonicalNaN
  else if exponent == 0 then
    let leading := floorLog2Small mantissa
    let unbiased : Int := 1 - Int.ofNat descriptor.bias - Int.ofNat mantissaBits + Int.ofNat leading
    let doubleExponent := UInt64.ofNat (unbiased + 1023).toNat
    let leadingBit := pow2 leading
    let doubleFraction := (mantissa - leadingBit) <<< UInt64.ofNat (52 - leading)
    sign64 ||| (doubleExponent <<< 52) ||| doubleFraction
  else
    let unbiased : Int := Int.ofNat exponent.toNat - Int.ofNat descriptor.bias
    let doubleExponent := UInt64.ofNat (unbiased + 1023).toNat
    let doubleFraction := mantissa <<< UInt64.ofNat (52 - mantissaBits)
    sign64 ||| (doubleExponent <<< 52) ||| doubleFraction

def fp8DecodeBits (dtype : Dtype) (payload : UInt64) : Except Error UInt64 := do
  let some descriptor := descriptorFor? dtype | throw .unsupportedDtype
  pure (fp8DecodeWith descriptor payload)

/-- Public decode admission preserves arbitrary Python integers as Lean
`Int`. Tag 1 denotes an actual Python integral object (including bool); all
other tags reject after dtype admission. -/
def fp8DecodePublic (dtype : Dtype) (payload : Int) (payloadTag : UInt8) :
    Except Error UInt64 := do
  let some descriptor := descriptorFor? dtype | throw .unsupportedDtype
  if payloadTag != 1 then throw .invalidPayload
  pure (fp8DecodeWith descriptor (UInt64.ofInt payload &&& 0xFF))

def fp8TruncateBits (dtype : Dtype) (bits : UInt64) : Except Error UInt64 := do
  let payload ← fp8EncodeBits dtype bits
  fp8DecodeBits dtype payload

/-- Public FP8 encode/truncate admission. Python transports one of three
stable boundary states: tag 1 carries exact binary64 bits, tag 2 records a
binary64-marshalling overflow, and every other tag is invalid/non-real.
Dtype and operation admission deliberately precede value-kind admission. -/
def convertPublicValue (operation : UInt8) (dtype : Dtype) (bits : UInt64)
    (valueTag : UInt8) : Except Error UInt64 := do
  let _ ← match descriptorFor? dtype with
    | some descriptor => pure descriptor
    | none => throw .unsupportedDtype
  if operation != 0 && operation != 3 then throw .unsupportedOperation
  if valueTag == 2 then throw .valueOverflow
  if valueTag != 1 then throw .invalidValue
  if operation == 0 then fp8EncodeBits dtype bits else fp8TruncateBits dtype bits

example : convertPublicValue 0 .fp8e4m3_ 0 2 = .error .valueOverflow := by
  rfl

example : convertPublicValue 0 .float32_ 0 2 = .error .unsupportedDtype := by
  rfl

private def fp16PackBits (bits : UInt64) : UInt64 :=
  let sign : UInt64 := (bits >>> 63) <<< 15
  let rawExponent : UInt64 := (bits >>> 52) &&& 0x7FF
  let fraction : UInt64 := bits &&& 0x000FFFFFFFFFFFFF
  if rawExponent == 0x7FF then
    sign ||| (if fraction == 0 then (0x7C00 : UInt64) else 0x7E00)
  else if rawExponent == 0 then sign
  else
    let exponent : Int := Int.ofNat rawExponent.toNat - 1023
    if exponent > 15 then sign ||| (0x7C00 : UInt64)
    else if exponent ≥ -14 then
      let kept := fraction >>> 42
      let remainder := fraction &&& 0x3FFFFFFFFFF
      let halfway : UInt64 := 0x20000000000
      let result := UInt64.ofNat (exponent + 15).toNat <<< 10 ||| kept
      sign ||| (if remainder > halfway || (remainder == halfway && (kept &&& 1) == 1)
                then result + 1 else result)
    else if exponent < -25 then sign
    else
      let significand := pow2 52 ||| fraction
      let drop := (28 - exponent).toNat
      let kept := significand >>> UInt64.ofNat drop
      let mask := pow2 drop - 1
      let remainder := significand &&& mask
      let halfway := pow2 (drop - 1)
      sign ||| (if remainder > halfway || (remainder == halfway && (kept &&& 1) == 1)
                then kept + 1 else kept)

private def fp16ExpandBits (half : UInt64) : UInt64 :=
  let value : UInt64 := half &&& 0xFFFF
  let sign : UInt64 := (value >>> 15) <<< 63
  let exponent : UInt64 := (value >>> 10) &&& 0x1F
  let mantissa : UInt64 := value &&& 0x3FF
  if exponent == 0x1F then
    sign ||| (0x7FF0000000000000 : UInt64) ||| (mantissa <<< 42)
  else if exponent == 0 then
    if mantissa == 0 then sign
    else
      let leading :=
        if mantissa ≥ 512 then 9 else if mantissa ≥ 256 then 8
        else if mantissa ≥ 128 then 7 else if mantissa ≥ 64 then 6
        else if mantissa ≥ 32 then 5 else if mantissa ≥ 16 then 4
        else if mantissa ≥ 8 then 3 else if mantissa ≥ 4 then 2
        else if mantissa ≥ 2 then 1 else 0
      let unbiased : Int := 1 - 15 - 10 + Int.ofNat leading
      let doubleExponent := UInt64.ofNat (unbiased + 1023).toNat
      let doubleFraction := (mantissa - pow2 leading) <<< UInt64.ofNat (52 - leading)
      sign ||| (doubleExponent <<< 52) ||| doubleFraction
  else
    let doubleExponent := exponent - 15 + 1023
    sign ||| (doubleExponent <<< 52) ||| (mantissa <<< 42)

def fp16RoundedBits (bits : UInt64) : UInt64 :=
  fp16ExpandBits (fp16PackBits bits)

/-- Operation codes: 0 encode FP8, 1 decode FP8, 2 round through fp16,
3 truncate through FP8. Every success is an exact integer result payload. -/
def convert (operation : UInt8) (dtype : Dtype) (input : UInt64) : Except Error UInt64 :=
  match operation with
  | 0 => fp8EncodeBits dtype input
  | 1 => fp8DecodeBits dtype input
  | 2 => if dtype == .float16_ then pure (fp16RoundedBits input)
         else throw .unsupportedDtype
  | 3 => fp8TruncateBits dtype input
  | _ => throw .unsupportedOperation

theorem fp8_fnuz_nonfinite_sentinel :
    (match fp8EncodeBits .fp8e4m3fnuz_ 0x7FF0000000000000 with
     | .ok value => value == 0x80 | .error _ => false) = true := by
  native_decide

theorem fp8_fnuz_negative_zero_canonical :
    (match fp8EncodeBits .fp8e5m2fnuz_ 0x8000000000000000 with
     | .ok value => value == 0 | .error _ => false) = true := by
  native_decide

theorem fp8_unsupported_rejected :
    (match fp8EncodeBits .float32_ 0 with
     | .error .unsupportedDtype => true | _ => false) = true := by
  native_decide

theorem fp16_overflow_tie :
    fp16RoundedBits 0x40EFFE0000000000 = 0x7FF0000000000000 := by
  native_decide

end Tgrad.LowPrecision

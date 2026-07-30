/-! # Tgrad.Dtype

The current scalar dtype ontology is grounded in pinned tinygrad revision
`19c4d736f2bc`.  `immediateParents` is the one current promotion authority;
reachability and least-upper promotion are derived from it by a finite closure.

The original 14×14 release fixtures predate weakfloat and FP8.  Their readings
remain available only through the explicitly historical `legacy*` definitions
at the end of this file.  They are not the product's current dtype semantics.
-/
namespace Tgrad

/-- Public scalar semantic identities plus `void`, which is a non-lattice
sentinel. Aliases such as `half` and `float` are names for these identities,
not additional constructors. -/
inductive Dtype where
  | bool_
  | weakint_
  | int8_ | uint8_
  | int16_ | uint16_
  | int32_ | uint32_
  | int64_ | uint64_
  | weakfloat_
  | fp8e4m3_ | fp8e5m2_
  | fp8e4m3fnuz_ | fp8e5m2fnuz_
  | float16_ | bfloat16_
  | float32_ | float64_
  | void_
  deriving BEq, Repr, Inhabited, DecidableEq

/-- Stable public names used by the foreign requirement and Python boundary. -/
def Dtype.toStr : Dtype → String
  | .bool_ => "bool" | .weakint_ => "weakint"
  | .int8_ => "int8" | .uint8_ => "uint8"
  | .int16_ => "int16" | .uint16_ => "uint16"
  | .int32_ => "int32" | .uint32_ => "uint32"
  | .int64_ => "int64" | .uint64_ => "uint64"
  | .weakfloat_ => "weakfloat"
  | .fp8e4m3_ => "fp8e4m3" | .fp8e5m2_ => "fp8e5m2"
  | .fp8e4m3fnuz_ => "fp8e4m3fnuz" | .fp8e5m2fnuz_ => "fp8e5m2fnuz"
  | .float16_ => "float16" | .bfloat16_ => "bfloat16"
  | .float32_ => "float32" | .float64_ => "float64"
  | .void_ => "void"

/-- Backend spelling from upstream `DType.name`. -/
def Dtype.backendName : Dtype → String
  | .bool_ => "bool" | .weakint_ => "weakint"
  | .int8_ => "signed char" | .uint8_ => "unsigned char"
  | .int16_ => "short" | .uint16_ => "unsigned short"
  | .int32_ => "int" | .uint32_ => "unsigned int"
  | .int64_ => "long" | .uint64_ => "unsigned long"
  | .weakfloat_ => "weakfloat"
  | .fp8e4m3_ => "float8_e4m3" | .fp8e5m2_ => "float8_e5m2"
  | .fp8e4m3fnuz_ => "float8_e4m3fnuz" | .fp8e5m2fnuz_ => "float8_e5m2fnuz"
  | .float16_ => "half" | .bfloat16_ => "__bf16"
  | .float32_ => "float" | .float64_ => "double"
  | .void_ => "void"

/-- Tinygrad runtime spellings used by symbolic fixtures. -/
def Dtype.toSymbolicStr : Dtype → String
  | .int32_ => "int" | .int64_ => "long"
  | .float16_ => "half" | .bfloat16_ => "__bf16"
  | .float32_ => "float" | .float64_ => "double"
  | d => d.toStr

def Dtype.ofSymbolicStr : String → Option Dtype
  | "bool" => some .bool_ | "weakint" => some .weakint_
  | "int8" => some .int8_ | "uint8" => some .uint8_
  | "int16" => some .int16_ | "uint16" => some .uint16_
  | "int" | "int32" => some .int32_
  | "uint32" => some .uint32_
  | "long" | "int64" => some .int64_
  | "uint64" => some .uint64_
  | "weakfloat" => some .weakfloat_
  | "fp8e4m3" => some .fp8e4m3_ | "fp8e5m2" => some .fp8e5m2_
  | "fp8e4m3fnuz" => some .fp8e4m3fnuz_
  | "fp8e5m2fnuz" => some .fp8e5m2fnuz_
  | "half" | "float16" => some .float16_
  | "__bf16" | "bfloat16" => some .bfloat16_
  | "float" | "float32" => some .float32_
  | "double" | "float64" => some .float64_
  | "void" => some .void_
  | _ => none

/-- Stable FFI code. Existing compute codes 0..3 are preserved. -/
def Dtype.code : Dtype → UInt8
  | .bfloat16_ => 0 | .float32_ => 1 | .float16_ => 2 | .int32_ => 3
  | .bool_ => 4 | .weakint_ => 5 | .int8_ => 6 | .uint8_ => 7
  | .int16_ => 8 | .uint16_ => 9 | .uint32_ => 10
  | .int64_ => 11 | .uint64_ => 12 | .weakfloat_ => 13
  | .fp8e4m3_ => 14 | .fp8e5m2_ => 15
  | .fp8e4m3fnuz_ => 16 | .fp8e5m2fnuz_ => 17
  | .float64_ => 18 | .void_ => 254

def Dtype.ofCode? : UInt8 → Option Dtype
  | 0 => some .bfloat16_ | 1 => some .float32_ | 2 => some .float16_
  | 3 => some .int32_ | 4 => some .bool_ | 5 => some .weakint_
  | 6 => some .int8_ | 7 => some .uint8_ | 8 => some .int16_
  | 9 => some .uint16_ | 10 => some .uint32_ | 11 => some .int64_
  | 12 => some .uint64_ | 13 => some .weakfloat_
  | 14 => some .fp8e4m3_ | 15 => some .fp8e5m2_
  | 16 => some .fp8e4m3fnuz_ | 17 => some .fp8e5m2fnuz_
  | 18 => some .float64_ | 254 => some .void_
  | _ => none

/-- Complete current lattice, in upstream declaration order. -/
def Dtype.semanticAll : List Dtype :=
  [.weakint_, .bool_, .int8_, .uint8_, .int16_, .uint16_, .int32_, .uint32_,
   .int64_, .uint64_, .weakfloat_, .fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_,
   .fp8e5m2fnuz_, .float16_, .bfloat16_, .float32_, .float64_]

def Dtype.allWithVoid : List Dtype := .void_ :: Dtype.semanticAll

/-- Bytes per concrete storage element. Weak values are compile-time scalar
types and have no tensor storage admission; their itemsize mirrors upstream
metadata only. -/
def Dtype.sizeBytes : Dtype → Nat
  | .bool_ | .int8_ | .uint8_ | .fp8e4m3_ | .fp8e5m2_
  | .fp8e4m3fnuz_ | .fp8e5m2fnuz_ => 1
  | .int16_ | .uint16_ | .float16_ | .bfloat16_ => 2
  | .int32_ | .uint32_ | .float32_ => 4
  | .int64_ | .uint64_ | .float64_ => 8
  | .weakint_ | .weakfloat_ => 100
  | .void_ => 0

def Dtype.bits : Dtype → Nat
  | .bool_ => 1 | .weakint_ | .weakfloat_ => 800
  | .int8_ | .uint8_ | .fp8e4m3_ | .fp8e5m2_
  | .fp8e4m3fnuz_ | .fp8e5m2fnuz_ => 8
  | .int16_ | .uint16_ | .float16_ | .bfloat16_ => 16
  | .int32_ | .uint32_ | .float32_ => 32
  | .int64_ | .uint64_ | .float64_ => 64
  | .void_ => 0

def Dtype.priority : Dtype → Int
  | .void_ => -1 | .weakint_ | .bool_ => 0
  | .int8_ => 1 | .uint8_ => 2 | .int16_ => 3 | .uint16_ => 4
  | .int32_ => 5 | .uint32_ => 6 | .int64_ => 7 | .uint64_ => 8
  | .weakfloat_ => 9
  | .fp8e4m3_ | .fp8e4m3fnuz_ => 10
  | .fp8e5m2_ | .fp8e5m2fnuz_ => 11
  | .float16_ => 12 | .bfloat16_ => 13 | .float32_ => 14 | .float64_ => 15

/-- Stable format code: 0 means upstream `None`, otherwise ASCII. -/
def Dtype.formatCode : Dtype → UInt8
  | .bool_ => '?'.toNat.toUInt8
  | .int8_ => 'b'.toNat.toUInt8 | .uint8_ => 'B'.toNat.toUInt8
  | .int16_ => 'h'.toNat.toUInt8 | .uint16_ => 'H'.toNat.toUInt8
  | .int32_ => 'i'.toNat.toUInt8 | .uint32_ => 'I'.toNat.toUInt8
  | .int64_ => 'q'.toNat.toUInt8 | .uint64_ => 'Q'.toNat.toUInt8
  | .float16_ => 'e'.toNat.toUInt8 | .float32_ => 'f'.toNat.toUInt8
  | .float64_ => 'd'.toNat.toUInt8
  | _ => 0

def Dtype.isFloat : Dtype → Bool
  | .weakfloat_ | .fp8e4m3_ | .fp8e5m2_ | .fp8e4m3fnuz_
  | .fp8e5m2fnuz_ | .float16_ | .bfloat16_ | .float32_ | .float64_ => true
  | _ => false

def Dtype.isInt : Dtype → Bool
  | .weakint_ | .int8_ | .uint8_ | .int16_ | .uint16_ | .int32_
  | .uint32_ | .int64_ | .uint64_ => true
  | _ => false

def Dtype.isUnsigned : Dtype → Bool
  | .uint8_ | .uint16_ | .uint32_ | .uint64_ => true
  | _ => false

def Dtype.isBool (d : Dtype) : Bool := d == .bool_

/-- Metadata capability is intentionally broader than compute capability. -/
def Dtype.computeSupported : Dtype → Bool
  | .bfloat16_ | .float32_ | .int32_ => true
  | _ => false

def Dtype.computeSupportedSet : List Dtype :=
  Dtype.semanticAll.filter Dtype.computeSupported

/-- Dtypes implemented through a foreign backend emulation rather than their
    native storage/arithmetic.  The current Metal product has no such route.
    Keeping the empty relation in Lean lets compatibility metadata expose the
    truthful answer without recreating upstream ContextVar machinery. -/
def Dtype.emulatedSet : List Dtype := []

def Dtype.isEmulated (d : Dtype) : Bool := Dtype.emulatedSet.contains d

theorem emulated_set_is_empty : Dtype.emulatedSet = [] := rfl

def Dtype.defaultInt : Dtype := .int32_
def Dtype.defaultFloat : Dtype := .float32_

/-- Exact foreign-admitted runtime integer defaults.  This is metadata
admission, not tensor-compute admission. -/
def Dtype.integerDefaultSet : List Dtype :=
  [.int8_, .int16_, .int32_, .int64_]

/-- Exact foreign-admitted runtime floating defaults.  FP8, fp16 and fp64 are
valid semantic defaults even though the current compute backend cannot create
tensors of all of them. -/
def Dtype.floatingDefaultSet : List Dtype :=
  [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_,
   .float16_, .bfloat16_, .float32_, .float64_]

def Dtype.integerDefaultAllowed (d : Dtype) : Bool :=
  Dtype.integerDefaultSet.contains d

def Dtype.floatingDefaultAllowed (d : Dtype) : Bool :=
  Dtype.floatingDefaultSet.contains d

/-- Strengthen a weak dtype under an explicit runtime-default environment. -/
def Dtype.strongWithDefaults (defaultInt defaultFloat : Dtype) : Dtype → Dtype
  | .weakint_ => defaultInt | .weakfloat_ => defaultFloat
  | d => d

/-- Initial-process strengthening, retained as a pure specification function.
Runtime entry points use `strongWithDefaults` with the shared Lean state. -/
def Dtype.strong (d : Dtype) : Dtype :=
  Dtype.strongWithDefaults Dtype.defaultInt Dtype.defaultFloat d

/-- Primitive authoring tags accepted by the Python boundary:
0 bool/Invalid, 1 int, 2 float, 3 empty aggregate. Python owns only recursive
syntax normalization; this Lean function owns admission and dtype meaning. -/
def Dtype.pythonTagDtypeWithDefaults? (defaultInt defaultFloat : Dtype) :
    UInt8 → Option Dtype
  | 0 => some .bool_
  | 1 => some defaultInt
  | 2 | 3 => some defaultFloat
  | _ => none

def Dtype.pythonTagDtype? : UInt8 → Option Dtype :=
  Dtype.pythonTagDtypeWithDefaults? Dtype.defaultInt Dtype.defaultFloat

/-- Foreign public names that denote an existing semantic identity. -/
def Dtype.aliases : List (String × Dtype) :=
  [("half", .float16_), ("float", .float32_), ("double", .float64_),
   ("uchar", .uint8_), ("ushort", .uint16_), ("uint", .uint32_),
   ("ulong", .uint64_), ("char", .int8_), ("short", .int16_),
   ("int", .int32_), ("long", .int64_),
   ("default_float", .float32_), ("default_int", .int32_)]

/-- Public `str`/`repr` name, derived from foreign alias declaration order.
The final non-default alias wins exactly as upstream's inverse dictionary does. -/
def Dtype.displayName (d : Dtype) : String :=
  Dtype.aliases.foldl (fun current entry =>
    if entry.1 == "default_float" || entry.1 == "default_int" then current
    else if entry.2 == d then entry.1 else current) d.toStr

/-- Named foreign classification collections. These are data in Lean so
membership and alias correspondence are compiler-checked. -/
def Dtype.collections : List (String × List Dtype) :=
  [("fp8_ocp", [.fp8e4m3_, .fp8e5m2_]),
   ("fp8_fnuz", [.fp8e4m3fnuz_, .fp8e5m2fnuz_]),
   ("fp8s", [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_]),
   ("floats", [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_,
                .float16_, .bfloat16_, .float32_, .float64_]),
   ("int8s", [.uint8_, .int8_]), ("int16s", [.uint16_, .int16_]),
   ("int32s", [.uint32_, .int32_]), ("int64s", [.uint64_, .int64_]),
   ("uints", [.uint8_, .uint16_, .uint32_, .uint64_]),
   ("sints", [.int8_, .int16_, .int32_, .int64_]),
   ("ints", [.uint8_, .uint16_, .uint32_, .uint64_,
              .int8_, .int16_, .int32_, .int64_]),
   ("weaks", [.weakint_, .weakfloat_]),
   ("all", [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_,
             .float16_, .bfloat16_, .float32_, .float64_,
             .uint8_, .uint16_, .uint32_, .uint64_,
             .int8_, .int16_, .int32_, .int64_, .bool_])]

/-- Current pinned promotion edges. This is the sole current graph authority. -/
def Dtype.immediateParents : Dtype → List Dtype
  | .bool_ => [.weakint_]
  | .weakint_ => [.int8_, .uint8_]
  | .int8_ => [.int16_] | .int16_ => [.int32_] | .int32_ => [.int64_]
  | .int64_ => [.weakfloat_]
  | .uint8_ => [.int16_, .uint16_]
  | .uint16_ => [.int32_, .uint32_]
  | .uint32_ => [.int64_, .uint64_]
  | .uint64_ => [.weakfloat_]
  | .weakfloat_ => [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_]
  | .fp8e4m3_ | .fp8e5m2_ | .fp8e4m3fnuz_ | .fp8e5m2fnuz_ =>
      [.float16_, .bfloat16_]
  | .float16_ | .bfloat16_ => [.float32_]
  | .float32_ => [.float64_]
  | .float64_ | .void_ => []

private def Dtype.addUnique (xs : List Dtype) (x : Dtype) : List Dtype :=
  if xs.contains x then xs else xs ++ [x]

private def Dtype.expandReachable (seen : List Dtype) : List Dtype :=
  seen.foldl (fun acc d => d.immediateParents.foldl Dtype.addUnique acc) seen

private def Dtype.closeWithFuel : Nat → List Dtype → List Dtype
  | 0, seen => seen
  | fuel + 1, seen => Dtype.closeWithFuel fuel (Dtype.expandReachable seen)

/-- Reflexive transitive closure, mechanically derived from
`immediateParents` with enough fuel for the finite semantic universe. -/
def Dtype.closureSet (d : Dtype) : List Dtype :=
  Dtype.closeWithFuel Dtype.semanticAll.length [d]

def Dtype.reaches (a b : Dtype) : Bool := a.closureSet.contains b

def Dtype.lt (a b : Dtype) : Bool :=
  if a.priority < b.priority then true
  else if a.priority > b.priority then false
  else if a.bits < b.bits then true
  else if a.bits > b.bits then false
  else compare a.backendName b.backendName == .lt

/-- Infer the dtype of a normalized Python aggregate. Empty input means an
empty top-level aggregate. Aggregation follows upstream's DType order and the
winner is strengthened in Lean. -/
def Dtype.inferPythonTagsWithDefaults? (defaultInt defaultFloat : Dtype)
    (tags : List UInt8) : Option Dtype := do
  let dtypes ← tags.mapM
    (Dtype.pythonTagDtypeWithDefaults? defaultInt defaultFloat)
  match dtypes with
  | [] => some defaultFloat
  | first :: rest =>
      some (Dtype.strongWithDefaults defaultInt defaultFloat
        (rest.foldl (fun current next =>
          if current.lt next then next else current) first))

def Dtype.inferPythonTags? (tags : List UInt8) : Option Dtype :=
  Dtype.inferPythonTagsWithDefaults? Dtype.defaultInt Dtype.defaultFloat tags

private def Dtype.intersect (xs ys : List Dtype) : List Dtype :=
  xs.filter (fun x => ys.contains x)

def Dtype.lub (a b : Dtype) : Dtype :=
  let common := Dtype.intersect a.closureSet b.closureSet
  match common with
  | [] => .float64_
  | x :: xs => xs.foldl (fun best d => if d.lt best then d else best) x

/-- True foreign varargs promotion: intersect every closure at once, then
choose the foreign minimum. This is not a binary fold because the pinned
binary operation is non-associative. -/
def Dtype.leastUpperMany? : List Dtype → Option Dtype
  | [] => none
  | first :: rest =>
    let common := rest.foldl (fun acc d => Dtype.intersect acc d.closureSet)
      first.closureSet
    match common with
    | [] => none
    | x :: xs => some (xs.foldl (fun best d => if d.lt best then d else best) x)

def Dtype.le (a b : Dtype) : Bool := a.reaches b

def Dtype.leastUpperFloatWithDefault (defaultFloat d : Dtype) : Dtype :=
  if d == .weakint_ then .weakfloat_
  else if d.isFloat then d
  else Dtype.lub d defaultFloat

def Dtype.leastUpperFloat (d : Dtype) : Dtype :=
  Dtype.leastUpperFloatWithDefault Dtype.defaultFloat d

def Dtype.canLosslessCast (dt0 dt1 : Dtype) : Bool :=
  if dt0 == dt1 || dt0 == .bool_ then true
  else match dt1 with
  | .weakint_ => [.int8_, .uint8_, .int16_, .uint16_, .int32_, .uint32_,
                   .int64_, .uint64_].contains dt0
  | .float64_ => [.float32_, .float16_, .bfloat16_, .fp8e4m3_, .fp8e5m2_,
                   .fp8e4m3fnuz_, .fp8e5m2fnuz_, .uint32_, .uint16_, .uint8_,
                   .int32_, .int16_, .int8_].contains dt0
  | .float32_ => [.float16_, .bfloat16_, .fp8e4m3_, .fp8e5m2_,
                   .fp8e4m3fnuz_, .fp8e5m2fnuz_, .uint16_, .uint8_,
                   .int16_, .int8_].contains dt0
  | .float16_ => [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_,
                   .uint8_, .int8_].contains dt0
  | .uint64_ => [.uint32_, .uint16_, .uint8_].contains dt0
  | .uint32_ => [.uint16_, .uint8_].contains dt0
  | .uint16_ => dt0 == .uint8_
  | .int64_ => [.uint32_, .uint16_, .uint8_, .int32_, .int16_, .int8_].contains dt0
  | .int32_ => [.uint16_, .uint8_, .int16_, .int8_].contains dt0
  | .int16_ => [.uint8_, .int8_].contains dt0
  | _ => false

/-- Range kind at the FFI boundary: 0 bool, 1 signed integer, 2 unsigned
integer, 3 floating infinity bounds, 4 abstract weak value, 255 none. -/
def Dtype.rangeKind : Dtype → UInt8
  | .bool_ => 0
  | .int8_ | .int16_ | .int32_ | .int64_ => 1
  | .uint8_ | .uint16_ | .uint32_ | .uint64_ => 2
  | .fp8e4m3_ | .fp8e5m2_ | .fp8e4m3fnuz_ | .fp8e5m2fnuz_
  | .float16_ | .bfloat16_ | .float32_ | .float64_ => 3
  | .weakint_ | .weakfloat_ => 4
  | .void_ => 255

/-- Sign-extended 64-bit two's-complement representation of the concrete
minimum.  The FFI has one width-independent integer representation: consumers
decode bit 63, never reinterpret the value using the source dtype width. -/
def Dtype.rangeMinBits (d : Dtype) : UInt64 :=
  match d.rangeKind with
  | 0 | 2 | 3 | 4 | 255 => 0
  | _ => (0 : UInt64) - ((1 : UInt64) <<< UInt64.ofNat (d.bits - 1))

/-- Bit pattern for the concrete maximum. -/
def Dtype.rangeMaxBits (d : Dtype) : UInt64 :=
  match d.rangeKind with
  | 0 => 1
  | 1 => ((1 : UInt64) <<< UInt64.ofNat (d.bits - 1)) - 1
  | 2 => if d.bits == 64 then (0 : UInt64) - 1
         else ((1 : UInt64) <<< UInt64.ofNat d.bits) - 1
  | _ => 0

def Dtype.finfo : Dtype → Option (Nat × Nat)
  | .float16_ => some (5, 10) | .bfloat16_ => some (8, 7)
  | .float32_ => some (8, 23) | .float64_ => some (11, 52)
  | .fp8e4m3_ | .fp8e4m3fnuz_ => some (4, 3)
  | .fp8e5m2_ | .fp8e5m2fnuz_ => some (5, 2)
  | _ => none

/-! ## Lean-owned bf16 storage conversion -/

/-- Round an IEEE-754 fp32 bit pattern to bf16 storage bits using
round-to-nearest-even. NaNs are quieted so a signaling NaN cannot collapse to
infinity when the low 16 payload bits are removed. -/
def Dtype.bf16PackBits (u : UInt32) : UInt32 :=
  let exponent := u &&& 0x7F800000
  let mantissa := u &&& 0x007FFFFF
  if exponent == 0x7F800000 && mantissa != 0 then
    ((u ||| 0x00400000) >>> 16) &&& 0xFFFF
  else
    ((u + (0x7FFF : UInt32) + ((u >>> 16) &&& (1 : UInt32))) >>> 16) &&& 0xFFFF

/-- Upstream's scalar `float_to_bf16` result as fp32 bits. Non-finite fp32
values pass through unchanged; finite values are rounded to bf16 precision and
returned in fp32 storage. This is distinct from `bf16PackBits`, which produces
the 16-bit tensor-storage representation. -/
def Dtype.bf16RoundedF32Bits (u : UInt32) : UInt32 :=
  let exponent := u &&& 0x7F800000
  if exponent == 0x7F800000 then u else Dtype.bf16PackBits u <<< 16

def Dtype.bf16ExpandBits (u : UInt32) : UInt32 := (u &&& 0xFFFF) <<< 16

/-! ## Exact zero-copy storage reinterpretation authority -/

/-- Proof-relevant complete nonidentity storage-bitcast relation. -/
inductive Dtype.BitcastStorageRelation : Dtype → Dtype → Prop where
  | float32ToInt32 : Dtype.BitcastStorageRelation .float32_ .int32_
  | int32ToFloat32 : Dtype.BitcastStorageRelation .int32_ .float32_

def Dtype.decideBitcastStorageRelation :
    (source target : Dtype) → Decidable (Dtype.BitcastStorageRelation source target)
  | source, target =>
      if hForward : source = .float32_ ∧ target = .int32_ then
        isTrue (by simpa [hForward.1, hForward.2] using
          Dtype.BitcastStorageRelation.float32ToInt32)
      else if hReverse : source = .int32_ ∧ target = .float32_ then
        isTrue (by simpa [hReverse.1, hReverse.2] using
          Dtype.BitcastStorageRelation.int32ToFloat32)
      else
        isFalse (by
          intro relation
          cases relation <;> simp_all)

instance (source target : Dtype) :
    Decidable (Dtype.BitcastStorageRelation source target) :=
  Dtype.decideBitcastStorageRelation source target

/-- Boolean decision view consumed by plans and Tensor classification.  It
    lives below both modules, preventing parallel admission policies. -/
def Dtype.bitcastStoragePair (source target : Dtype) : Bool :=
  decide (Dtype.BitcastStorageRelation source target)

/-- Independent exhaustive contract for the shared finite relation. -/
theorem Dtype.bitcastStoragePair_contract (source target : Dtype) :
    source.bitcastStoragePair target =
      ((source == .float32_ && target == .int32_) ||
       (source == .int32_ && target == .float32_)) := by
  cases source <;> cases target <;> native_decide

theorem Dtype.bitcastStoragePair_itemsize {source target : Dtype}
    (h : source.bitcastStoragePair target = true) :
    source.sizeBytes = target.sizeBytes := by
  have relation : Dtype.BitcastStorageRelation source target :=
    of_decide_eq_true h
  cases relation <;> rfl

private def Dtype.readU32LE (bytes : ByteArray) (offset : Nat) : UInt32 :=
  UInt32.ofNat ((bytes[offset]!).toNat
    + ((bytes[offset + 1]!).toNat * 256)
    + ((bytes[offset + 2]!).toNat * 65536)
    + ((bytes[offset + 3]!).toNat * 16777216))

private def Dtype.pushU16LE (out : ByteArray) (u : UInt32) : ByteArray :=
  (out.push (u &&& 0xFF).toUInt8).push ((u >>> 8) &&& 0xFF).toUInt8

private def Dtype.pushU32LE (out : ByteArray) (u : UInt32) : ByteArray :=
  let out := out.push (u &&& 0xFF).toUInt8
  let out := out.push ((u >>> 8) &&& 0xFF).toUInt8
  let out := out.push ((u >>> 16) &&& 0xFF).toUInt8
  out.push ((u >>> 24) &&& 0xFF).toUInt8

def Dtype.packFp32BytesToBf16? (bytes : ByteArray) : Option ByteArray :=
  if bytes.size % 4 != 0 then none else
    some ((List.range (bytes.size / 4)).foldl (fun out i =>
      Dtype.pushU16LE out (Dtype.bf16PackBits (Dtype.readU32LE bytes (i * 4))))
      ByteArray.empty)

def Dtype.expandBf16BytesToFp32? (bytes : ByteArray) : Option ByteArray :=
  if bytes.size % 2 != 0 then none else
    some ((List.range (bytes.size / 2)).foldl (fun out i =>
      let lo := (bytes[i * 2]!).toNat
      let hi := (bytes[i * 2 + 1]!).toNat
      Dtype.pushU32LE out (Dtype.bf16ExpandBits (UInt32.ofNat (lo + (hi <<< 8)))))
      ByteArray.empty)

/-! ## Finite compiler obligations over the foreign-grounded universe -/

private def listNodup [BEq α] (xs : List α) : Bool :=
  match xs with
  | [] => true
  | x :: rest => !rest.contains x && listNodup rest

def Dtype.codesUniqueCheck : Bool := listNodup (Dtype.allWithVoid.map Dtype.code)
def Dtype.namesUniqueCheck : Bool := listNodup (Dtype.allWithVoid.map Dtype.toStr)
def Dtype.edgeEndpointsCheck : Bool := Dtype.semanticAll.all (fun d =>
  d.immediateParents.all (fun p => Dtype.semanticAll.contains p))
def Dtype.reachesFloat64Check : Bool :=
  Dtype.semanticAll.all (fun d => d.reaches .float64_)
def Dtype.lubTotalCheck : Bool := Dtype.semanticAll.all (fun a =>
  Dtype.semanticAll.all (fun b =>
    !(Dtype.intersect a.closureSet b.closureSet).isEmpty))
def Dtype.lubIdemCheck : Bool := Dtype.semanticAll.all (fun a => Dtype.lub a a == a)
def Dtype.lub_comm_check : Bool := Dtype.semanticAll.all (fun a =>
  Dtype.semanticAll.all (fun b => Dtype.lub a b == Dtype.lub b a))
def Dtype.lubUpperBoundCheck : Bool := Dtype.semanticAll.all (fun a =>
  Dtype.semanticAll.all (fun b => a.reaches (Dtype.lub a b) && b.reaches (Dtype.lub a b)))
def Dtype.nonAssociativeTriples : List (Dtype × Dtype × Dtype) :=
  Dtype.semanticAll.flatMap (fun a =>
    Dtype.semanticAll.flatMap (fun b =>
      Dtype.semanticAll.filterMap (fun c =>
        if Dtype.lub a (Dtype.lub b c) != Dtype.lub (Dtype.lub a b) c
        then some (a, b, c) else none)))
def Dtype.classificationCoherentCheck : Bool := Dtype.allWithVoid.all (fun d =>
  !(d.isFloat && d.isInt) && !(d.isBool && (d.isFloat || d.isInt)))
def Dtype.aliasesValidCheck : Bool :=
  listNodup (Dtype.aliases.map Prod.fst) &&
  Dtype.aliases.all (fun entry => Dtype.allWithVoid.contains entry.2)
def Dtype.collectionsValidCheck : Bool :=
  listNodup (Dtype.collections.map Prod.fst) &&
  Dtype.collections.all (fun entry =>
    listNodup entry.2 && entry.2.all (fun d => Dtype.semanticAll.contains d))

theorem dtype_codes_unique : Dtype.codesUniqueCheck = true := by native_decide
theorem dtype_names_unique : Dtype.namesUniqueCheck = true := by native_decide
theorem promotion_edge_endpoints_valid : Dtype.edgeEndpointsCheck = true := by native_decide
theorem every_semantic_dtype_reaches_float64 : Dtype.reachesFloat64Check = true := by native_decide
theorem lub_total_holds : Dtype.lubTotalCheck = true := by native_decide
theorem lub_idempotent_holds : Dtype.lubIdemCheck = true := by native_decide
theorem lub_comm_holds : Dtype.lub_comm_check = true := by native_decide
theorem lub_upper_bound_holds : Dtype.lubUpperBoundCheck = true := by native_decide
theorem dtype_classification_coherent : Dtype.classificationCoherentCheck = true := by native_decide
theorem dtype_alias_targets_valid : Dtype.aliasesValidCheck = true := by native_decide
theorem dtype_collections_valid : Dtype.collectionsValidCheck = true := by native_decide
theorem compute_supported_exact :
    Dtype.computeSupportedSet = [.int32_, .bfloat16_, .float32_] := by
  native_decide
theorem compute_supported_count : Dtype.computeSupportedSet.length = 3 := by
  native_decide
theorem integer_default_admission_exact :
    Dtype.allWithVoid.filter Dtype.integerDefaultAllowed =
      [.int8_, .int16_, .int32_, .int64_] := by
  native_decide
theorem floating_default_admission_exact :
    Dtype.allWithVoid.filter Dtype.floatingDefaultAllowed =
      [.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_,
       .float16_, .bfloat16_, .float32_, .float64_] := by
  native_decide
theorem float_rejected_as_integer_default :
    Dtype.integerDefaultAllowed .float32_ = false := by rfl
theorem integer_rejected_as_floating_default :
    Dtype.floatingDefaultAllowed .int32_ = false := by rfl
theorem weak_and_void_rejected_as_defaults :
    ([.weakint_, .weakfloat_, .void_] : List Dtype).all (fun d =>
      !d.integerDefaultAllowed && !d.floatingDefaultAllowed) = true := by
  native_decide
theorem float16_compute_rejected : Dtype.computeSupported .float16_ = false := by
  rfl
theorem all_fp8_compute_rejected :
    ([.fp8e4m3_, .fp8e5m2_, .fp8e4m3fnuz_, .fp8e5m2fnuz_] : List Dtype).all
      (fun d => !d.computeSupported) = true := by native_decide
theorem python_infer_bool : Dtype.inferPythonTags? [0] = some .bool_ := by native_decide
theorem python_infer_invalid_as_bool : Dtype.inferPythonTags? [0] = some .bool_ := by
  native_decide
theorem python_infer_int : Dtype.inferPythonTags? [1] = some .int32_ := by native_decide
theorem python_infer_float : Dtype.inferPythonTags? [2] = some .float32_ := by native_decide
theorem python_infer_empty : Dtype.inferPythonTags? [] = some .float32_ := by native_decide
theorem python_infer_empty_marker : Dtype.inferPythonTags? [3] = some .float32_ := by
  native_decide
theorem python_infer_bool_list : Dtype.inferPythonTags? [0] = some .bool_ := by
  native_decide
theorem python_infer_bool_int : Dtype.inferPythonTags? [0, 1] = some .int32_ := by
  native_decide
theorem python_infer_bool_float : Dtype.inferPythonTags? [0, 2] = some .float32_ := by
  native_decide
theorem python_infer_int_float : Dtype.inferPythonTags? [1, 2] = some .float32_ := by
  native_decide
theorem python_infer_bool_int_float :
    Dtype.inferPythonTags? [0, 1, 2] = some .float32_ := by native_decide
theorem python_infer_nested_empty_int :
    Dtype.inferPythonTags? [3, 1] = some .float32_ := by native_decide
theorem python_infer_invalid_tag_rejected : Dtype.inferPythonTags? [255] = none := by
  native_decide
theorem dtype_display_float32 : Dtype.displayName .float32_ = "float" := by native_decide
theorem dtype_display_int32 : Dtype.displayName .int32_ = "int" := by native_decide
theorem dtype_display_float16 : Dtype.displayName .float16_ = "half" := by native_decide

/- The range boundary is a single sign-extended UInt64 protocol.  These
examples pin both ends for every signed storage width, so changing either the
Lean range rule or its FFI encoding breaks the build. -/
theorem int8_range_extrema :
    Dtype.rangeMinBits .int8_ = 0xFFFFFFFFFFFFFF80 ∧
      Dtype.rangeMaxBits .int8_ = 0x7F := by native_decide
theorem int16_range_extrema :
    Dtype.rangeMinBits .int16_ = 0xFFFFFFFFFFFF8000 ∧
      Dtype.rangeMaxBits .int16_ = 0x7FFF := by native_decide
theorem int32_range_extrema :
    Dtype.rangeMinBits .int32_ = 0xFFFFFFFF80000000 ∧
      Dtype.rangeMaxBits .int32_ = 0x7FFFFFFF := by native_decide
theorem int64_range_extrema :
    Dtype.rangeMinBits .int64_ = 0x8000000000000000 ∧
      Dtype.rangeMaxBits .int64_ = 0x7FFFFFFFFFFFFFFF := by native_decide

theorem int64_uint64_promote_to_weakfloat :
    Dtype.lub .int64_ .uint64_ = .weakfloat_ := by native_decide
theorem weakfloat_fp8_promotes_to_fp8 :
    Dtype.lub .weakfloat_ .fp8e4m3_ = .fp8e4m3_ := by native_decide
theorem cross_fp8_promotes_to_float16 :
    Dtype.lub .fp8e4m3_ .fp8e5m2_ = .float16_ := by native_decide
theorem foreign_lub_nonassociative_left :
    Dtype.lub .bfloat16_ (Dtype.lub .fp8e4m3_ .fp8e4m3fnuz_) = .float32_ := by
  native_decide
theorem foreign_lub_nonassociative_right :
    Dtype.lub (Dtype.lub .bfloat16_ .fp8e4m3_) .fp8e4m3fnuz_ = .bfloat16_ := by
  native_decide
theorem foreign_lub_nonassociative_witness :
    Dtype.lub .bfloat16_ (Dtype.lub .fp8e4m3_ .fp8e4m3fnuz_) !=
      Dtype.lub (Dtype.lub .bfloat16_ .fp8e4m3_) .fp8e4m3fnuz_ := by
  native_decide
theorem foreign_lub_nonassociative_count :
    Dtype.nonAssociativeTriples.length = 24 := by native_decide
theorem foreign_nary_fp8_bf16_example :
    Dtype.leastUpperMany? [.fp8e4m3_, .fp8e5m2_, .bfloat16_] =
      some .bfloat16_ := by native_decide
theorem foreign_nary_nonassoc_witness_example :
    Dtype.leastUpperMany? [.bfloat16_, .fp8e4m3_, .fp8e4m3fnuz_] =
      some .bfloat16_ := by native_decide
theorem foreign_nary_empty_rejected : Dtype.leastUpperMany? [] = none := by rfl

theorem bf16_round_down_example : Dtype.bf16PackBits 0x3F807000 = 0x3F80 := by native_decide
theorem bf16_round_up_example : Dtype.bf16PackBits 0x3F80C000 = 0x3F81 := by native_decide
theorem bf16_even_tie_example : Dtype.bf16PackBits 0x3F808000 = 0x3F80 := by native_decide
theorem bf16_odd_tie_example : Dtype.bf16PackBits 0x3F818000 = 0x3F82 := by native_decide
theorem bf16_nan_example : Dtype.bf16PackBits 0x7F800001 = 0x7FC0 := by native_decide
theorem bf16_quiet_nan_scalar_passthrough :
    Dtype.bf16RoundedF32Bits 0x7FC00001 = 0x7FC00001 := by native_decide
theorem bf16_positive_infinity_example : Dtype.bf16PackBits 0x7F800000 = 0x7F80 := by native_decide
theorem bf16_negative_infinity_example : Dtype.bf16PackBits 0xFF800000 = 0xFF80 := by native_decide
theorem bf16_max_finite_example : Dtype.bf16PackBits 0x7F7F7FFF = 0x7F7F := by native_decide
theorem bf16_overflow_tie_example : Dtype.bf16PackBits 0x7F7F8000 = 0x7F80 := by native_decide

theorem canLosslessCast_self (d : Dtype) : Dtype.canLosslessCast d d = true := by
  cases d <;> rfl
theorem canLosslessCast_from_bool (d : Dtype) : Dtype.canLosslessCast .bool_ d = true := by
  cases d <;> rfl

/-! ## Historical L1 subset

These definitions reproduce the old observer's 14×14 readings. They are named
legacy so no current caller can mistake them for the pinned product lattice.
-/

def Dtype.legacyAll : List Dtype :=
  [.bool_, .weakint_, .int8_, .uint8_, .int16_, .uint16_, .int32_, .uint32_,
   .int64_, .uint64_, .float16_, .bfloat16_, .float32_, .float64_]

/-- Compatibility alias for older callers that only enumerate L1 fixtures. -/
def Dtype.all : List Dtype := Dtype.legacyAll

private def Dtype.legacyImmediateParents : Dtype → List Dtype
  | .bool_ => [.weakint_] | .weakint_ => [.int8_, .uint8_]
  | .int8_ => [.int16_] | .int16_ => [.int32_] | .int32_ => [.int64_]
  | .int64_ => [.uint64_]
  | .uint8_ => [.int16_, .uint16_] | .uint16_ => [.int32_, .uint32_]
  | .uint32_ => [.int64_, .uint64_]
  | .uint64_ => [.float16_, .bfloat16_]
  | .float16_ | .bfloat16_ => [.float32_] | .float32_ => [.float64_]
  | _ => []

private def Dtype.legacyExpand (seen : List Dtype) : List Dtype :=
  seen.foldl (fun acc d => d.legacyImmediateParents.foldl Dtype.addUnique acc) seen
private def Dtype.legacyClose : Nat → List Dtype → List Dtype
  | 0, seen => seen
  | fuel + 1, seen => Dtype.legacyClose fuel (Dtype.legacyExpand seen)
private def Dtype.legacyClosure (d : Dtype) : List Dtype :=
  Dtype.legacyClose Dtype.legacyAll.length [d]

def Dtype.legacyLub (a b : Dtype) : Dtype :=
  match Dtype.intersect (Dtype.legacyClosure a) (Dtype.legacyClosure b) with
  | [] => .float64_
  | x :: xs => xs.foldl (fun best d => if d.lt best then d else best) x

def Dtype.legacyLubAssocCheck : Bool := Dtype.legacyAll.all (fun a =>
  Dtype.legacyAll.all (fun b => Dtype.legacyAll.all (fun c =>
    Dtype.legacyLub a (Dtype.legacyLub b c) ==
      Dtype.legacyLub (Dtype.legacyLub a b) c)))

theorem legacy_lub_assoc_holds : Dtype.legacyLubAssocCheck = true := by native_decide

def Dtype.lubRowToJson (t : Dtype × Dtype × Dtype) : String :=
  let (a, b, result) := t
  "  {\n    \"a\": \"" ++ a.toStr ++ "\",\n" ++
  "    \"b\": \"" ++ b.toStr ++ "\",\n" ++
  "    \"lub\": \"" ++ result.toStr ++ "\"\n  }"

def Dtype.lubTableToJson (rows : List (Dtype × Dtype × Dtype)) : String :=
  "[\n" ++ String.intercalate ",\n" (rows.map Dtype.lubRowToJson) ++ "\n]"

def Dtype.computeLubTable : List (Dtype × Dtype × Dtype) :=
  Dtype.legacyAll.flatMap (fun a =>
    Dtype.legacyAll.map (fun b => (a, b, Dtype.legacyLub a b)))

def Dtype.castRowToJson (t : Dtype × Dtype × Bool) : String :=
  let (a, b, admitted) := t
  "  {\n    \"dt0\": \"" ++ a.toStr ++ "\",\n" ++
  "    \"dt1\": \"" ++ b.toStr ++ "\",\n" ++
  "    \"can_cast\": " ++ (if admitted then "true" else "false") ++ "\n  }"

def Dtype.castTableToJson (rows : List (Dtype × Dtype × Bool)) : String :=
  "[\n" ++ String.intercalate ",\n" (rows.map Dtype.castRowToJson) ++ "\n]"

def Dtype.computeCastTable : List (Dtype × Dtype × Bool) :=
  Dtype.legacyAll.flatMap (fun a =>
    Dtype.legacyAll.map (fun b => (a, b, Dtype.canLosslessCast a b)))

end Tgrad

/-! # Tgrad.Dtype

  Port of tinygrad's `promo_lattice` + `least_upper_dtype` +
  `can_lossless_cast`. Lifted from theograd_phases/17_dtype_system/Demo.lean
  with namespace and naming adapted into the unified `Tgrad` namespace.

  Two stringifications coexist:
    Dtype.toStr         — phase-17 names ("int32", "bfloat16", ...);
                          used by lub_table / can_lossless_cast emit.
    Dtype.toSymbolicStr — tinygrad runtime names ("int", "long",
                          "float", ...); used by symbolic DAG emit
                          (phase-03 fixture format).
-/
namespace Tgrad

/-- The 14 dtypes we model. Trailing underscore on each constructor
    avoids clashes with Lean's built-in `Bool`/`Int`. The lattice and
    LUB enumerate over `allDtypes` below. -/
inductive Dtype where
  | bool_
  | weakint_
  | int8_  | uint8_
  | int16_ | uint16_
  | int32_ | uint32_
  | int64_ | uint64_
  | float16_ | bfloat16_
  | float32_ | float64_
  /-- Sentinel for SINK / STORE / END dtype slots. Not part of the
      promotion lattice — excluded from `Dtype.all` so `lub_comm_holds`
      and `lub_assoc_holds` stay over the 14-element set. -/
  | void_
  deriving BEq, Repr, Inhabited, DecidableEq

/-- Phase-17 emit names (used by `lub_table.json` /
    `can_lossless_cast_table.json` byte-diffs). -/
def Dtype.toStr : Dtype → String
  | .bool_     => "bool"
  | .weakint_  => "weakint"
  | .int8_     => "int8"
  | .uint8_    => "uint8"
  | .int16_    => "int16"
  | .uint16_   => "uint16"
  | .int32_    => "int32"
  | .uint32_   => "uint32"
  | .int64_    => "int64"
  | .uint64_   => "uint64"
  | .float16_  => "float16"
  | .bfloat16_ => "bfloat16"
  | .float32_  => "float32"
  | .float64_  => "float64"
  | .void_     => "void"

/-- Tinygrad-runtime stringification (matches phase 02/03's `Dtype.toStr`,
    used by the symbolic DAG fixture format). Subset coverage matches the
    fixture inputs; the unmapped variants fall through to `toStr`. -/
def Dtype.toSymbolicStr : Dtype → String
  | .bool_     => "bool"
  | .int32_    => "int"
  | .int64_    => "long"
  | .float32_  => "float"
  | .float64_  => "double"
  | .bfloat16_ => "__bf16"
  | d          => d.toStr

/-- Parse a tinygrad-runtime dtype string into Dtype. Used by the
    symbolic-DAG `toTree` fixture loader + phase-04-rangeify loader. -/
def Dtype.ofSymbolicStr : String → Option Dtype
  | "bool"    => some .bool_
  | "int"     => some .int32_
  | "long"    => some .int64_
  | "float"   => some .float32_
  | "double"  => some .float64_
  | "__bf16"  => some .bfloat16_
  | "weakint" => some .weakint_
  | "void"    => some .void_
  | _         => none

/-- Bytes per element. Used by buffer / dispatch code at L4+. -/
def Dtype.sizeBytes : Dtype → Nat
  | .bool_     => 1
  | .weakint_  => 8   -- tinygrad treats weakint as Python int; we use 8
  | .int8_     => 1   | .uint8_    => 1
  | .int16_    => 2   | .uint16_   => 2
  | .int32_    => 4   | .uint32_   => 4
  | .int64_    => 8   | .uint64_   => 8
  | .float16_  => 2   | .bfloat16_ => 2
  | .float32_  => 4   | .float64_  => 8
  | .void_     => 0   -- sentinel; void has no in-memory representation

-- ============================================================================
-- Lattice: `parents` adjacency mirrors tinygrad's `promo_lattice`.
-- ============================================================================

/-- Immediate promotion parents (RHS of `promo_lattice` in
    `tinygrad/dtype.py:219-225`). `closureSet` walks this graph
    transitively. KEEP ALIGNED with `tinygrad.dtype.promo_lattice`. -/
def Dtype.parents : Dtype → List Dtype
  | .bool_     => [.weakint_]
  | .weakint_  => [.int8_, .uint8_]
  | .int8_     => [.int16_]
  | .int16_    => [.int32_]
  | .int32_    => [.int64_]
  | .int64_    => [.uint64_]
  | .uint8_    => [.int16_, .uint16_]
  | .uint16_   => [.int32_, .uint32_]
  | .uint32_   => [.int64_, .uint64_]
  | .uint64_   => [.float16_, .bfloat16_]
  | .float16_  => [.float32_]
  | .bfloat16_ => [.float32_]
  | .float32_  => [.float64_]
  | .float64_  => []
  | .void_     => []   -- void is not part of the lattice

/-- Transitive closure of `parents` (hardcoded per-dtype rather than
    fuel-bounded BFS; `partial def` doesn't reduce in proofs). -/
def Dtype.closureSet : Dtype → List Dtype
  | .bool_     => [.bool_, .weakint_, .int8_, .uint8_, .int16_, .uint16_,
                   .int32_, .uint32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .weakint_  => [.weakint_, .int8_, .uint8_, .int16_, .uint16_,
                   .int32_, .uint32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .int8_     => [.int8_, .int16_, .int32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .uint8_    => [.uint8_, .int16_, .uint16_, .int32_, .uint32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .int16_    => [.int16_, .int32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .uint16_   => [.uint16_, .int32_, .uint32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .int32_    => [.int32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .uint32_   => [.uint32_, .int64_, .uint64_,
                   .float16_, .bfloat16_, .float32_, .float64_]
  | .int64_    => [.int64_, .uint64_, .float16_, .bfloat16_, .float32_, .float64_]
  | .uint64_   => [.uint64_, .float16_, .bfloat16_, .float32_, .float64_]
  | .float16_  => [.float16_, .float32_, .float64_]
  | .bfloat16_ => [.bfloat16_, .float32_, .float64_]
  | .float32_  => [.float32_, .float64_]
  | .float64_  => [.float64_]
  | .void_     => [.void_]   -- void is its own closure (not in lattice)

/-- DType.__lt__ tuple key — (priority, bits) is total on our 14 dtypes. -/
def Dtype.priority : Dtype → Nat
  | .bool_     => 0
  | .weakint_  => 0
  | .int8_     => 1
  | .uint8_    => 2
  | .int16_    => 3
  | .uint16_   => 4
  | .int32_    => 5
  | .uint32_   => 6
  | .int64_    => 7
  | .uint64_   => 8
  | .float16_  => 11
  | .bfloat16_ => 12
  | .float32_  => 13
  | .float64_  => 14
  | .void_     => 0   -- void doesn't participate in priority comparison

def Dtype.bits : Dtype → Nat
  | .bool_     => 1
  | .weakint_  => 800
  | .int8_     => 8   | .uint8_    => 8
  | .int16_    => 16  | .uint16_   => 16
  | .int32_    => 32  | .uint32_   => 32
  | .int64_    => 64  | .uint64_   => 64
  | .float16_  => 16  | .bfloat16_ => 16
  | .float32_  => 32  | .float64_  => 64
  | .void_     => 0

def Dtype.lt (a b : Dtype) : Bool :=
  let pa := a.priority; let pb := b.priority
  if pa < pb then true
  else if pa > pb then false
  else a.bits < b.bits

private def Dtype.intersect (xs ys : List Dtype) : List Dtype :=
  xs.filter (fun x => ys.contains x)

/-- `leastUpperDtype a b = min (by Dtype.lt) of (closureSet a ∩ closureSet b)`.
    Total on our 14-element set: `float64` is in every closure. -/
def Dtype.lub (a b : Dtype) : Dtype :=
  let inter := Dtype.intersect a.closureSet b.closureSet
  match inter with
  | []      => .float64_  -- unreachable; closures always share float64
  | x :: xs =>
      xs.foldl (fun acc d => if d.lt acc then d else acc) x

/-- The 14-element Dtype set, in priority order. Reused by lub_table emit
    and the commutativity/associativity proofs. -/
def Dtype.all : List Dtype :=
  [.bool_, .weakint_,
   .int8_, .uint8_, .int16_, .uint16_, .int32_, .uint32_, .int64_, .uint64_,
   .float16_, .bfloat16_, .float32_, .float64_]

-- ============================================================================
-- Theorems — proven by `decide`/`native_decide` on the finite lattice.
-- ============================================================================

/-- Commutativity of `lub` checked as a boolean over the 196 pairs. -/
def Dtype.lub_comm_check : Bool :=
  Dtype.all.all (fun a =>
    Dtype.all.all (fun b => Dtype.lub a b == Dtype.lub b a))

/-- `decide` proves `lub_comm_check = true`. If `parents`/`closureSet`
    changes such that `lub` stops being commutative on our 14-element
    set, this proof fails at compile time. -/
theorem lub_comm_holds : Dtype.lub_comm_check = true := by native_decide

/-- Associativity checked as a boolean over the 2744 triples. -/
def Dtype.lub_assoc_check : Bool :=
  Dtype.all.all (fun a =>
    Dtype.all.all (fun b =>
      Dtype.all.all (fun c =>
        Dtype.lub a (Dtype.lub b c) == Dtype.lub (Dtype.lub a b) c)))

/-- `native_decide` compiles + runs the 14³ check at Lean compile time. -/
theorem lub_assoc_holds : Dtype.lub_assoc_check = true := by native_decide

-- ============================================================================
-- can_lossless_cast (numpy-style "is dt0 safely representable in dt1").
-- ============================================================================

private def Dtype.memDtype (d : Dtype) (xs : List Dtype) : Bool :=
  xs.contains d

/-- Ports `tinygrad.dtype.can_lossless_cast`. Returns `true` iff every
    representable value of `dt0` is also representable in `dt1`. The
    case table mirrors the Python switch on `dt1`. -/
def Dtype.canLosslessCast (dt0 dt1 : Dtype) : Bool :=
  if dt0 == dt1 ∨ dt0 == .bool_ then true
  else
    match dt1 with
    | .weakint_  => Dtype.memDtype dt0 [.int8_, .uint8_, .int16_, .uint16_,
                                        .int32_, .uint32_, .int64_, .uint64_]
    | .float64_  => Dtype.memDtype dt0 [.float32_, .float16_, .bfloat16_,
                                        .uint32_, .uint16_, .uint8_,
                                        .int32_, .int16_, .int8_]
    | .float32_  => Dtype.memDtype dt0 [.float16_, .bfloat16_,
                                        .uint16_, .uint8_, .int16_, .int8_]
    | .float16_  => Dtype.memDtype dt0 [.uint8_, .int8_]
    | .uint64_   => Dtype.memDtype dt0 [.uint32_, .uint16_, .uint8_]
    | .uint32_   => Dtype.memDtype dt0 [.uint16_, .uint8_]
    | .uint16_   => Dtype.memDtype dt0 [.uint8_]
    | .int64_    => Dtype.memDtype dt0 [.uint32_, .uint16_, .uint8_,
                                        .int32_, .int16_, .int8_]
    | .int32_    => Dtype.memDtype dt0 [.uint16_, .uint8_, .int16_, .int8_]
    | .int16_    => Dtype.memDtype dt0 [.uint8_, .int8_]
    | _          => false

/-- BL invariant: every dtype losslessly casts to itself. -/
theorem canLosslessCast_self (d : Dtype) : Dtype.canLosslessCast d d = true := by
  cases d <;> rfl

/-- BL invariant: bool losslessly casts to every dtype (it's the bottom
    of the lattice in tinygrad's `can_lossless_cast`). -/
theorem canLosslessCast_from_bool (d : Dtype) : Dtype.canLosslessCast .bool_ d = true := by
  cases d <;> rfl

-- ============================================================================
-- JSON emit — Python `json.dumps(indent=2)`-byte-equivalent layouts.
-- ============================================================================

/-- Render one row of the lub table as JSON, matching python_spec.py output
    byte-for-byte (2-space indent, double-quoted strings). -/
def Dtype.lubRowToJson (t : Dtype × Dtype × Dtype) : String :=
  let (a, b, lub) := t
  "  {\n    \"a\": \"" ++ a.toStr ++ "\",\n" ++
  "    \"b\": \"" ++ b.toStr ++ "\",\n" ++
  "    \"lub\": \"" ++ lub.toStr ++ "\"\n  }"

def Dtype.lubTableToJson (rows : List (Dtype × Dtype × Dtype)) : String :=
  "[\n" ++ String.intercalate ",\n" (rows.map Dtype.lubRowToJson) ++ "\n]"

/-- Compute the full 14×14 Lean-side lub table. -/
def Dtype.computeLubTable : List (Dtype × Dtype × Dtype) :=
  Dtype.all.flatMap (fun a =>
    Dtype.all.map (fun b => (a, b, Dtype.lub a b)))

/-- Render one row of the cast table. -/
def Dtype.castRowToJson (t : Dtype × Dtype × Bool) : String :=
  let (a, b, c) := t
  "  {\n    \"dt0\": \"" ++ a.toStr ++ "\",\n" ++
  "    \"dt1\": \"" ++ b.toStr ++ "\",\n" ++
  "    \"can_cast\": " ++ (if c then "true" else "false") ++ "\n  }"

def Dtype.castTableToJson (rows : List (Dtype × Dtype × Bool)) : String :=
  "[\n" ++ String.intercalate ",\n" (rows.map Dtype.castRowToJson) ++ "\n]"

def Dtype.computeCastTable : List (Dtype × Dtype × Bool) :=
  Dtype.all.flatMap (fun a =>
    Dtype.all.map (fun b => (a, b, Dtype.canLosslessCast a b)))

end Tgrad

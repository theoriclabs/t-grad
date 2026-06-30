/-! # Tgrad.Shape

  Port of tinygrad's shape arithmetic: `prod` (numel), `_align_left`,
  `_broadcast_shape`, plus movement-op shape transformers (reshape,
  permute, expand). Lifted from theograd_phases/18_shape_arithmetic/Demo.lean
  into the unified `Tgrad` namespace.

  Two shape types coexist:
    Shape       = List Nat  (concrete shapes — v1 fast path)
    SintShape   = List Sint (symbolic-aware shapes — `Sint = nat | uop`)

  Theorems:
    numel_nil                          numel ([] : Shape) = 1
    numel_append                       numel (xs ++ ys) = numel xs * numel ys
    sintNumel_lift                     sintNumel (Shape.lift xs) = .nat (numel xs)
    reshape_preserves_numel_concrete   reshapeShape preserves numel on the
                                        all-concrete path.
-/
namespace Tgrad

-- ============================================================================
-- v1: concrete shape (List Nat) + numel + alignLeft + broadcast.
-- ============================================================================

abbrev Shape := List Nat

/-- `numel []` = 1 (scalar = single element). Matches tinygrad's
    `prod(()) = 1` (the `functools.reduce(*, [], 1)` start value). -/
def numel : Shape → Nat
  | []      => 1
  | s :: ss => s * numel ss

/-- `padLeft target xs`: prepend `(target - xs.length)` 1s onto `xs`. If
    `xs` is already at least `target` long, return it unchanged. -/
def padLeft (target : Nat) (xs : Shape) : Shape :=
  let n := xs.length
  if target ≤ n then xs
  else List.replicate (target - n) 1 ++ xs

/-- `alignLeft a b`: pad both to `max(a.length, b.length)` with leading 1s.
    Matches tinygrad's `_align_left` for the 2-shape case. -/
def alignLeft (a b : Shape) : Shape × Shape :=
  let m := Nat.max a.length b.length
  (padLeft m a, padLeft m b)

/-- Pairwise zip with `Nat.max`, but if any operand is 0 the result is 0.
    Matches tinygrad's `_broadcast_shape`:
        0 if 0 in nth_dim_sizes else smax(nth_dim_sizes)

    Note: tinygrad's `smax` is just `max`; it does *not* error on
    incompatible non-1 dims. We preserve that quirk. -/
def broadcastPairwise : List Nat → List Nat → List Nat
  | [], _              => []
  | _, []              => []
  | x :: xs, y :: ys   =>
      let v := if x == 0 ∨ y == 0 then 0 else Nat.max x y
      v :: broadcastPairwise xs ys

/-- `broadcast a b`: align-left then pairwise-max-with-zero. -/
def broadcast (a b : Shape) : Shape :=
  let (a', b') := alignLeft a b
  broadcastPairwise a' b'

-- ============================================================================
-- v1 theorems.
-- ============================================================================

theorem numel_nil : numel ([] : Shape) = 1 := rfl

theorem numel_append (xs ys : Shape) : numel (xs ++ ys) = numel xs * numel ys := by
  induction xs with
  | nil       => simp [numel]
  | cons h t ih =>
      simp [numel, ih, Nat.mul_assoc]

theorem padLeft_id (target : Nat) (xs : Shape) (h : target ≤ xs.length) :
    padLeft target xs = xs := by
  unfold padLeft
  simp [h]

-- ============================================================================
-- JSON emit — shape_table.json layout (matches python_spec.py byte-for-byte).
-- ============================================================================

private def shapeJson (s : Shape) : String :=
  if s.isEmpty then "[]"
  else "[\n      " ++ String.intercalate ",\n      " (s.map toString) ++ "\n    ]"

structure ShapeRow where
  a         : Shape
  b         : Shape
  alignedA  : Shape
  alignedB  : Shape
  bcast     : Shape
  numelA    : Nat
  numelB    : Nat
  deriving Repr, Inhabited

def ShapeRow.toJson (r : ShapeRow) : String :=
  "  {\n" ++
  "    \"a\": "         ++ shapeJson r.a        ++ ",\n" ++
  "    \"b\": "         ++ shapeJson r.b        ++ ",\n" ++
  "    \"aligned_a\": " ++ shapeJson r.alignedA ++ ",\n" ++
  "    \"aligned_b\": " ++ shapeJson r.alignedB ++ ",\n" ++
  "    \"broadcast\": " ++ shapeJson r.bcast    ++ ",\n" ++
  "    \"numel_a\": "   ++ toString r.numelA    ++ ",\n" ++
  "    \"numel_b\": "   ++ toString r.numelB    ++ "\n  }"

def shapeTableToJson (rows : List ShapeRow) : String :=
  "[\n" ++ String.intercalate ",\n" (rows.map ShapeRow.toJson) ++ "\n]"

def computeShapeRow (pair : Shape × Shape) : ShapeRow :=
  let (a, b) := pair
  let (al, bl) := alignLeft a b
  { a := a, b := b,
    alignedA := al, alignedB := bl,
    bcast := broadcast a b,
    numelA := numel a, numelB := numel b }

def computeShapeTable (pairs : List (Shape × Shape)) : List ShapeRow :=
  pairs.map computeShapeRow

/-- The fixed 12-pair list captured by `theograd_phases/18` python_spec.py
    (FIXTURE_PAIRS). Keep aligned with `shape_table.json`. -/
def Shape.fixturePairs : List (Shape × Shape) :=
  [ -- rank-equal, no broadcast
    ([2, 3],     [2, 3]),
    ([1, 1],     [1, 1]),
    -- rank-equal, broadcast via 1
    ([1, 3],     [2, 1]),
    ([5, 1, 4],  [1, 3, 1]),
    -- rank-different, left-align
    ([3],        [2, 3]),
    ([4, 5],     [2, 3, 4, 5]),
    -- scalar with non-scalar
    ([],         [2, 3]),
    -- both empty (scalars)
    ([],         []),
    -- zero-dim (tinygrad's "0 in zip" propagates 0)
    ([0, 3],     [2, 3]),
    ([2, 0],     [2, 1]),
    -- mismatched non-1 dims (tinygrad doesn't error, just smax)
    ([2, 3],     [4, 5]),
    -- 3D + 1D
    ([8, 1, 5],  [5]) ]

-- ============================================================================
-- v2: Sint (int | uop_key) + symbolic-aware shape.
-- ============================================================================

/-- Symbolic dimension — either a concrete Nat or a key referencing an
    upstream UOp. The `uop` key is whatever tinygrad emits as the symbolic
    dim's repr; drift trips the byte-diff fixture gate. -/
inductive Sint where
  | nat (n : Nat)
  | uop (key : String)
  deriving BEq, Repr, Inhabited

def Sint.toStr : Sint → String
  | .nat n   => toString n
  | .uop k   => "\"" ++ k ++ "\""

abbrev SintShape := List Sint

/-- Lift a Nat-shape into a SintShape. -/
def Shape.lift (s : Shape) : SintShape := s.map Sint.nat

/-- numel over a SintShape: returns `Sint.nat (∏ nats)` when all dims are
    concrete, or `Sint.uop "prod_sym"` if any dim is symbolic. -/
def sintNumel : SintShape → Sint
  | []      => .nat 1
  | s :: ss =>
      let rest := sintNumel ss
      match s, rest with
      | .nat n, .nat m => .nat (n * m)
      | _, _           => .uop ("prod_sym")

-- ============================================================================
-- Movement-op shape transformers.
-- ============================================================================

/-- Reshape semantics: target shape must preserve numel. Returns `some out`
    iff numel-preserving (all-Nat case); always `some` for symbolic dims
    (we trust upstream validation). -/
def reshapeShape (inS outS : SintShape) : Option SintShape :=
  match sintNumel inS, sintNumel outS with
  | .nat n, .nat m => if n == m then some outS else none
  | _, _           => some outS

/-- Permute: reorder axes by `axes`. Returns `none` if axes length ≠
    shape length or contains an out-of-range index. -/
def permuteShape (inS : SintShape) (axes : List Nat) : Option SintShape :=
  if axes.length ≠ inS.length then none
  else
    let arr := inS.toArray
    let n := arr.size
    if axes.all (fun a => a < n) then
      some (axes.map (fun i => arr[i]!))
    else none

/-- Expand: each input dim must be 1, or equal to the target dim. -/
def expandPairwise : List Sint → List Sint → Option (List Sint)
  | [], []                  => some []
  | _, []                   => none
  | [], _                   => none
  | s :: ss, t :: ts        =>
      let dimOk := match s, t with
        | .nat 1, _      => true
        | .nat n, .nat m => n == m
        | _, _           => true
      if dimOk then
        match expandPairwise ss ts with
        | some rest => some (t :: rest)
        | none      => none
      else none

def expandShape (inS targetS : SintShape) : Option SintShape :=
  if inS.length ≠ targetS.length then none
  else expandPairwise inS targetS

-- ============================================================================
-- v2 theorems.
-- ============================================================================

/-- `sintNumel (Shape.lift xs) = .nat (numel xs)`: the v1 `numel` agrees
    with v2 `sintNumel` on the all-Nat path. -/
theorem sintNumel_lift (xs : Shape) : sintNumel (Shape.lift xs) = .nat (numel xs) := by
  induction xs with
  | nil       => rfl
  | cons h t ih =>
      show sintNumel (Sint.nat h :: Shape.lift t) = .nat (h * numel t)
      unfold sintNumel
      rw [ih]

/-- Reshape preserves numel on the all-concrete path. -/
theorem reshape_preserves_numel_concrete (inS outS : SintShape) (n m : Nat)
    (h_in : sintNumel inS = .nat n)
    (h_out : sintNumel outS = .nat m)
    (h_some : reshapeShape inS outS = some outS) :
    n = m := by
  unfold reshapeShape at h_some
  rw [h_in, h_out] at h_some
  simp at h_some
  exact h_some

-- ============================================================================
-- Movement-op fixture serialization.
-- ============================================================================

inductive MovementOpKind where
  | reshape | permute | expand
  deriving BEq, Repr, Inhabited

def MovementOpKind.toStr : MovementOpKind → String
  | .reshape => "reshape"
  | .permute => "permute"
  | .expand  => "expand"

private def sintShapeJson (s : SintShape) : String :=
  if s.isEmpty then "[]"
  else "[\n      " ++ String.intercalate ",\n      " (s.map Sint.toStr) ++ "\n    ]"

private def natListJson (xs : List Nat) : String :=
  if xs.isEmpty then "[]"
  else "[\n      " ++ String.intercalate ",\n      " (xs.map toString) ++ "\n    ]"

private def optShapeJson (os : Option SintShape) : String :=
  match os with
  | none   => "null"
  | some s => sintShapeJson s

structure MovementRow where
  op       : MovementOpKind
  inShape  : SintShape
  arg      : List Nat
  expected : Option SintShape
  deriving Repr, Inhabited

def MovementRow.toJson (r : MovementRow) : String :=
  "  {\n" ++
  "    \"op\": \""       ++ r.op.toStr        ++ "\",\n" ++
  "    \"in_shape\": "   ++ sintShapeJson r.inShape ++ ",\n" ++
  "    \"arg\": "        ++ natListJson r.arg ++ ",\n" ++
  "    \"expected\": "   ++ optShapeJson r.expected ++ "\n  }"

def movementTableToJson (rows : List MovementRow) : String :=
  "[\n" ++ String.intercalate ",\n" (rows.map MovementRow.toJson) ++ "\n]"

/-- Apply the appropriate shape transformer; returns the computed expected. -/
def applyMovementRow (r : MovementRow) : Option SintShape :=
  let argShape := r.arg.map Sint.nat
  match r.op with
  | .reshape => reshapeShape r.inShape argShape
  | .permute => permuteShape r.inShape r.arg
  | .expand  => expandShape  r.inShape argShape

/-- Captured rows from theograd_phases/18 python_spec.py (15 cases in
    the exact order tinygrad emitted them). Keep aligned with
    `movement_table.json`. -/
def Shape.movementFixtureRows : List MovementRow :=
  -- 5 reshape cases
  [ { op := .reshape, inShape := [.nat 2, .nat 3],    arg := [6],
      expected := some [.nat 6] },
    { op := .reshape, inShape := [.nat 2, .nat 3],    arg := [3, 2],
      expected := some [.nat 3, .nat 2] },
    { op := .reshape, inShape := [.nat 4],            arg := [2, 2],
      expected := some [.nat 2, .nat 2] },
    { op := .reshape, inShape := [.nat 6],            arg := [2, 3],
      expected := some [.nat 2, .nat 3] },
    { op := .reshape, inShape := [.nat 2, .nat 3],    arg := [5],
      expected := none },
    -- 5 permute cases
    { op := .permute, inShape := [.nat 2, .nat 3],    arg := [1, 0],
      expected := some [.nat 3, .nat 2] },
    { op := .permute, inShape := [.nat 2, .nat 3, .nat 4], arg := [2, 0, 1],
      expected := some [.nat 4, .nat 2, .nat 3] },
    { op := .permute, inShape := [.nat 2, .nat 3, .nat 4], arg := [0, 1, 2],
      expected := some [.nat 2, .nat 3, .nat 4] },
    { op := .permute, inShape := [.nat 3],            arg := [0],
      expected := some [.nat 3] },
    { op := .permute, inShape := [.nat 2, .nat 3],    arg := [0, 1, 2],
      expected := none },
    -- 5 expand cases
    { op := .expand,  inShape := [.nat 1, .nat 3],    arg := [2, 3],
      expected := some [.nat 2, .nat 3] },
    { op := .expand,  inShape := [.nat 1, .nat 1, .nat 5], arg := [2, 4, 5],
      expected := some [.nat 2, .nat 4, .nat 5] },
    { op := .expand,  inShape := [.nat 3],            arg := [3],
      expected := some [.nat 3] },
    { op := .expand,  inShape := [.nat 2, .nat 1],    arg := [2, 4],
      expected := some [.nat 2, .nat 4] },
    { op := .expand,  inShape := [.nat 2, .nat 3],    arg := [4, 5],
      expected := none } ]

end Tgrad

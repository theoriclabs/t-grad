import Tgrad.UOp

/-! # Tgrad.Schedule.Indexing — movement-op forward decls + backward algebra

  Lift from theograd_phases/04_rangeify/Demo.lean.

  v1 narrow scope: RESHAPE + PERMUTE only. PAD/SHRINK/EXPAND/FLIP land
  with the full `pm_mops` library at L10.

  `applyReshape1D` handles the 1D→nD reshape case (the v1 fixture's
  `chain_forward` starts at shape (4,) and reshapes to (2,2)). Higher-
  rank input reshapes need symbolic `%` / `//` and live in L9+ (after
  the full symbolic rule library lands).
-/
namespace Tgrad

/-- v1 movement-op surface: reshape + permute. -/
inductive MovementOp where
  | reshape (inShape : List Nat) (outShape : List Nat)
  | permute (inShape : List Nat) (axes : List Nat)
  deriving Repr, Inhabited

def MovementOp.kindStr : MovementOp → String
  | .reshape _ _ => "RESHAPE"
  | .permute _ _ => "PERMUTE"

-- ============================================================================
-- Helpers used by applyMovementOp.
-- ============================================================================

/-- `argsort axes` returns the index permutation that sorts `axes` ascending.
    Inverts PERMUTE: tinygrad's `apply_movement_op(PERMUTE, ..., arg=axes)`
    returns `tuple(rngs[p] for p in argsort(arg))`. -/
def argsort (xs : List Nat) : List Nat :=
  let idxs := List.range xs.length
  idxs.mergeSort (fun i j => xs[i]! ≤ xs[j]!)

/-- Row-major strides. `stridesFor [2,2] = [2, 1]`,
    `stridesFor [3,4,5] = [20, 5, 1]`. -/
def stridesFor (shape : List Nat) : List Nat :=
  let rec go (acc : Nat) : List Nat → List Nat
    | []        => []
    | s :: rest => acc :: go (acc * s) rest
  (go 1 shape.reverse).reverse

/-- Build `∑_i rngs[i] * strides[i]` in the simplified canonical form
    (`* 1` dropped, `+ 0` dropped, constants moved left in `MUL`).
    Pre: `inShape.length = 1` (v1 narrow scope). -/
def applyReshape1D (outShape : List Nat) (rngs : List UOp) : List UOp :=
  let strides := stridesFor outShape
  let pairs := rngs.zip strides
  let terms := pairs.map (fun p =>
    let (r, s) := p
    if s == 1 then r
    else .binop .mul r (.const .weakint_ (.i (Int.ofNat s))) .weakint_)
  match terms with
  | []        => [.const .weakint_ (.i 0)]
  | [t]       => [t]
  | t :: rest => [rest.foldl (fun acc x => .binop .add acc x .weakint_) t]

/-- Apply one movement op *backward* (given output ranges, return input
    ranges). Matches `tinygrad.schedule.indexing.apply_movement_op` for
    `Ops.PERMUTE` and the simplified-form output for `Ops.RESHAPE` from
    a 1D input. -/
def applyMovementOp (op : MovementOp) (rngs : List UOp) : List UOp :=
  match op with
  | .permute _ axes =>
      let ord := argsort axes
      ord.map (fun i => rngs[i]!)
  | .reshape _ outShape =>
      applyReshape1D outShape rngs

end Tgrad

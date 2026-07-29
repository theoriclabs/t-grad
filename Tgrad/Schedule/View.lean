import Tgrad.UOp

/-! # Tgrad.Schedule.View — strides-and-offset view algebra

  The sort that was missing. Every movement op (PERMUTE / RESHAPE /
  EXPAND / SLICE) over a contiguous buffer is a pure transform of
  `(shape, strides, offset)`, and the LOAD index for a view is then
  `Σ_i idx_i * strides_i + offset`. This is tinygrad's `View` from
  `tinygrad/shape/view.py`, reduced to the non-masked case.

  Before this module, `Pipeline.viewIndexUOpForA/B` derived load
  indices from a six-arm pattern table that matched *literal* axis
  lists (`.permute _ [1, 0]`) and fell through to `panic!`. That table
  had two silent wrong-answer bugs which are structurally impossible
  here:

  - it matched `.slice _ (slc :: _)` and discarded every axis after
    the first, so `a[0:8, 4:12]` read column 0;
  - its `.expand` arm assumed the broadcast axis was axis 1, so a
    `(1,N) -> (K,N)` broadcast produced a K-varying index.

  A strides representation cannot drop an axis or mistake which axis
  is broadcast, because every axis contributes a term by construction.
-/
namespace Tgrad
namespace Schedule

/-- A strided view over a flat buffer. `strides` and `shape` are
    parallel, one entry per axis. A stride of `0` is a broadcast axis
    (the index does not advance). -/
structure View where
  shape   : List Nat
  strides : List Nat
  offset  : Nat
  deriving Repr, Inhabited, DecidableEq, BEq

namespace View

/-- Row-major strides for `shape`: `[2,3,4] => [12,4,1]`. -/
def stridesOf (shape : List Nat) : List Nat :=
  let rec go (acc : Nat) : List Nat → List Nat
    | []        => []
    | s :: rest => acc :: go (acc * s) rest
  (go 1 shape.reverse).reverse

/-- The view a freshly allocated row-major buffer presents. -/
def contiguous (shape : List Nat) : View :=
  { shape := shape, strides := stridesOf shape, offset := 0 }

def rank (v : View) : Nat := v.shape.length

def numel (v : View) : Nat := v.shape.foldl (· * ·) 1

/-- Prepend synthetic size-one axes so operands of different ranks can be
right-aligned for broadcasting. Synthetic axes have stride zero: moving along
one must continue to address the sole underlying element. -/
def padLeftToRank (v : View) (targetRank : Nat) : Option View :=
  if targetRank < v.rank then none
  else
    let padding := targetRank - v.rank
    some
      { shape := List.replicate padding 1 ++ v.shape
        strides := List.replicate padding 0 ++ v.strides
        offset := v.offset }

theorem padLeftToRank_preserves_parallel_axes (v : View) (targetRank : Nat)
    (hParallel : v.shape.length = v.strides.length) :
    ∀ padded, v.padLeftToRank targetRank = some padded →
      padded.shape.length = padded.strides.length := by
  intro padded hPadded
  simp [padLeftToRank, View.rank] at hPadded
  rw [← hPadded.2]
  simp [hParallel]

/-- True when `v` is exactly what `contiguous v.shape` would produce.
    Reshape is only sound on such a view. -/
def isContiguous (v : View) : Bool :=
  v.offset == 0 && v.strides == stridesOf v.shape

/-- PERMUTE: output axis `i` takes input axis `axes[i]`, matching
    `Tensor.permuteShapeNat`. Rejects anything that is not a genuine
    permutation of `0..rank-1` — note the old `permuteShape` accepted
    duplicate axes such as `[0,0]`. -/
def permute (v : View) (axes : List Nat) : Option View :=
  if axes.length != v.rank then none
  else if !(axes.all (· < v.rank)) then none
  else if (axes.eraseDups).length != axes.length then none
  else some
    { shape   := axes.map (fun i => v.shape[i]!)
      strides := axes.map (fun i => v.strides[i]!)
      offset  := v.offset }

/-- RESHAPE: sound only on a contiguous view, and only when the
    element count is preserved. Returning `none` otherwise is the
    honest answer — a non-contiguous reshape needs a second view,
    which this representation cannot hold. -/
def reshape (v : View) (newShape : List Nat) : Option View :=
  if !v.isContiguous then none
  else if newShape.foldl (· * ·) 1 != v.numel then none
  else some (contiguous newShape)

/-- EXPAND: broadcast size-1 axes up. The expanded axis gets stride 0
    so the index never advances along it. Any axis that is neither
    unchanged nor a 1 -> n broadcast is rejected. -/
def expand (v : View) (newShape : List Nat) : Option View :=
  if newShape.length != v.rank then none
  else
    let paired := (v.shape.zip v.strides).zip newShape
    let ok := paired.all (fun p =>
      let ((old, _), new) := p
      old == new || old == 1)
    if !ok then none
    else some
      { shape   := newShape
        strides := paired.map (fun p =>
          let ((old, st), new) := p
          if old == new then st else 0)
        offset  := v.offset }

/-- SLICE: `(start, stop, step)` per axis, numpy semantics. Axes
    beyond `slices` are untouched. Every sliced axis contributes to
    the offset, which is precisely what the old pattern table got
    wrong. `step = 0` is treated as `1`, matching `sliceShapeNat`. -/
def slice (v : View) (slices : List Slice) : Option View :=
  if slices.length > v.rank then none
  else
    let idxs := List.range v.rank
    let parts := idxs.map (fun i =>
      let dim := v.shape[i]!
      let st  := v.strides[i]!
      match slices[i]? with
      | none    => (dim, st, 0)
      | some sl =>
        let stop  := Nat.min sl.stop dim
        let start := Nat.min sl.start stop
        let step  := if sl.step == 0 then 1 else sl.step
        let n     := (stop - start + step - 1) / step
        (n, st * step, start * st))
    some
      { shape   := parts.map (fun p => p.1)
        strides := parts.map (fun p => p.2.1)
        offset  := v.offset + (parts.foldl (fun acc p => acc + p.2.2) 0) }

/-- Build `Σ_i vars[i] * strides[i] + offset` as an index UOp.

    Canonicalisation, chosen so the common cases render exactly as the
    hand-written table used to:
    - stride 0 terms are dropped (broadcast axis contributes nothing);
    - stride 1 terms render as the bare variable;
    - terms are emitted in descending stride order;
    - a zero offset is dropped; an empty sum renders as `0`. -/
def indexOf (v : View) (vars : List UOp) : UOp :=
  let terms := (vars.zip v.strides).filter (fun p => p.2 != 0)
  let sorted := terms.mergeSort (fun a b => a.2 ≥ b.2)
  let mk (p : UOp × Nat) : UOp :=
    if p.2 == 1 then p.1
    else .binop .mul p.1 (.const .int32_ (.i (Int.ofNat p.2))) .int32_
  let summands := sorted.map mk
  let all := if v.offset == 0 then summands
             else summands ++ [.const .int32_ (.i (Int.ofNat v.offset))]
  match all with
  | []      => .const .int32_ (.i 0)
  | t :: ts => ts.foldl (fun acc x => .binop .add acc x .int32_) t

end View

/-- Collapse a movement chain rooted at a `.buffer` into a single
    `View`. `none` for any chain this representation cannot express
    (e.g. reshape of a transposed view), which callers must handle —
    no `panic!`. -/
def viewOfUOp : UOp → Option View
  | .buffer _ shape _  => some (View.contiguous shape)
  | .permute src axes  => (viewOfUOp src).bind (fun v => v.permute axes)
  | .reshape src ns    => (viewOfUOp src).bind (fun v => v.reshape ns)
  | .expand  src ns    => (viewOfUOp src).bind (fun v => v.expand ns)
  | .slice   src sls   => (viewOfUOp src).bind (fun v => v.slice sls)
  | _                  => none

/-! The equations below are deliberately named obligations, not tests that
    grep constructor spellings. L14.B.3 imports these declarations as the
    structural contract for the shared view derivation. If an implementation
    arm is removed or stops delegating to the corresponding `View` transform,
    its equation no longer type-checks even when another function in this file
    happens to match the same `UOp` constructor. -/

theorem viewOfUOp_buffer_eq (handle : UInt64) (shape : List Nat) (dtype : Dtype) :
    viewOfUOp (.buffer handle shape dtype) = some (View.contiguous shape) := by
  simp [viewOfUOp]

theorem viewOfUOp_permute_eq (src : UOp) (axes : List Nat) :
    viewOfUOp (.permute src axes) =
      (viewOfUOp src).bind (fun v => v.permute axes) := by
  simp [viewOfUOp]

theorem viewOfUOp_reshape_eq (src : UOp) (newShape : List Nat) :
    viewOfUOp (.reshape src newShape) =
      (viewOfUOp src).bind (fun v => v.reshape newShape) := by
  simp [viewOfUOp]

theorem viewOfUOp_expand_eq (src : UOp) (newShape : List Nat) :
    viewOfUOp (.expand src newShape) =
      (viewOfUOp src).bind (fun v => v.expand newShape) := by
  simp [viewOfUOp]

theorem viewOfUOp_slice_eq (src : UOp) (slices : List Slice) :
    viewOfUOp (.slice src slices) =
      (viewOfUOp src).bind (fun v => v.slice slices) := by
  simp [viewOfUOp]

/-- The `.buffer` leaf under a movement chain, if there is one. -/
partial def bufferRootOf : UOp → Option UOp
  | u@(.buffer _ _ _) => some u
  | .permute src _    => bufferRootOf src
  | .reshape src _    => bufferRootOf src
  | .expand  src _    => bufferRootOf src
  | .slice   src _    => bufferRootOf src
  | _                 => none

/-- Canonical loop variables for a rank-`n` view: `idx0 .. idx{n-1}`. -/
def canonicalVars (n : Nat) : List UOp :=
  (List.range n).map (fun i => .var s!"idx{i}" .int32_)

end Schedule
end Tgrad

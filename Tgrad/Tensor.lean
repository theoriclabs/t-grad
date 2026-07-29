import Tgrad.Shape
import Tgrad.Dtype
import Tgrad.UOp
import Tgrad.Runtime.Buffer

/-! # Tgrad.Tensor — Lean-owned tensor (uop graph + dtype)

  At L5 this was `{ shape, dtype, buffer }` — a thin wrapper over a
  concrete `BufferHandle`. At L14.A the structure became `{ uop, dtype }`:
  a Lean-owned UOp graph whose root is (for contiguous tensors) the
  `UOp.buffer` leaf carrying the raw MTLBuffer pointer + shape + dtype.

  L14.B.1 extends `Tensor.shape` and `Tensor.buffer` to walk view
  chains (PERMUTE / RESHAPE / EXPAND / SLICE composed atop a BUFFER
  leaf). View methods `Tensor.{transpose, reshape, permute, expand,
  slice}` push movement nodes onto the uop tree — pure graph
  transforms; no buffer allocation; no kernel dispatch.

  L14.A invariant preserved: when the uop is a pure BUFFER (the
  L11 / L13 / L13_F contiguous path), `Tensor.shape` and
  `Tensor.buffer` produce bit-identical values to the L5..L13
  field-based version.

  L14.B.2 (separate sub-sub-gate) wires `Schedule.Rangeify.rangeify`
  into `Pipeline.realize` so a viewed Tensor's matmul actually
  produces correct bytes via index-expression-driven codegen. Until
  then, the matmul FFI raises `MatmulOnNonBufferUop` (a typed Python
  exception surfaced via the L14.B.1 guard in PythonFFI) for any
  non-BUFFER uop.
-/
namespace Tgrad

/-- L14.A: Tensor is a `uop : UOp` graph rooted at one or more BUFFER
    leaves, plus the result `dtype`. At L14.A the root is always a
    BUFFER (contiguous-tensor path); L14.B+ adds movement-op roots. -/
structure Tensor where
  uop   : UOp
  dtype : Dtype
  deriving Inhabited

/-! ## Concrete shape helpers used by `Tensor.shape`

    The helpers in `Tgrad.Shape` operate on `SintShape` (`List Sint`,
    supporting symbolic dims). L14.B.1's view methods only produce
    concrete shapes, so we provide `List Nat` variants that avoid the
    `Sint` round-trip + the `Option` for cases L14.B.1 guarantees
    validity by construction (view methods are pure; if the user calls
    `t.reshape(...)` with a bad shape, the resulting `Tensor.shape`
    returns the user-supplied dims — the rangeify pass at L14.B.2
    is responsible for rejecting incompatible chains). -/

/-- Reorder `shape` by `axes`. Pre: `axes` is a permutation of
    `[0, ..., shape.length - 1]`. Returns the original on
    out-of-range axis (graceful degradation; L14.B.2's rangeify pass
    is the authoritative validator). -/
def permuteShapeNat (shape : List Nat) (axes : List Nat) : List Nat :=
  axes.map (fun i => (shape[i]?).getD 0)

/-- Apply per-axis SLICE `(start, stop, step)` to `shape`. Each axis's
    new dim is `((stop - start) + step - 1) / step` (rounded up). -/
def sliceShapeNat (shape : List Nat) (slices : List Slice) : List Nat :=
  shape.zip slices |>.map (fun p =>
    let (dim, sl) := p
    let stop  := Nat.min sl.stop dim
    let start := Nat.min sl.start stop
    let step  := if sl.step == 0 then 1 else sl.step
    (stop - start + step - 1) / step)

/-- Derived: extract the shape from `t.uop`, walking PERMUTE / RESHAPE
    / EXPAND / SLICE atop a BUFFER leaf. Non-shape-bearing uops panic
    (contract: Tensor's uop is always a movement chain over a BUFFER). -/
def Tensor.shape (t : Tensor) : Shape :=
  walk t.uop
where
  walk : UOp → Shape
  | .buffer _ shape _   => shape
  | .permute src axes   => permuteShapeNat (walk src) axes
  | .reshape _ newShape => newShape
  | .expand _ newShape  => newShape
  | .slice src slices   => sliceShapeNat (walk src) slices
  -- Pointwise: both operands carry the same shape at this stage, so the
  -- left one is representative. Broadcasting between differing shapes
  -- is rejected before a graph is built, not silently resolved here.
  | .binop _ a _ _      => walk a
  -- Keepdim, matching what the reduce kernels actually write: a
  -- contracted axis becomes 1 rather than vanishing, so rank is stable
  -- across a reduction. This arm used to be absent, which meant a
  -- reduce node fell through to the panic and reported rank 0 --- one
  -- of the `panicsToDefault` morphisms `Ontology.lean` grades.
  | .reduce _ body axes =>
      let ax := axes.filterMap fun u =>
        match u with
        | .const _ (.i n) => some n.toNat
        | _               => none
      (walk body).mapIdx fun i d => if ax.contains i then 1 else d
  | _                   => panic! "L14.B.1: Tensor.shape: unsupported uop kind"

/-- Derived: extract the underlying MTLBuffer handle from the BUFFER
    leaf of `t.uop`, walking through any movement-op chain. The
    L14.A `size` formula (`numel shape * dtype.sizeBytes`) uses the
    LEAF's shape — not the view's effective shape — so view methods
    don't change the size that's reported to the Metal allocator. -/
def Tensor.buffer (t : Tensor) : Runtime.BufferHandle :=
  walk t.uop
where
  walk : UOp → Runtime.BufferHandle
  | .buffer h s d => { raw := h, size := Tgrad.numel s * d.sizeBytes }
  | .permute src _ => walk src
  | .reshape src _ => walk src
  | .expand src _  => walk src
  | .slice src _   => walk src
  | _              => panic! "L14.B.1: Tensor.buffer: unsupported uop kind"

/-- Construct a Tensor from a raw BufferHandle + shape + dtype. Wraps
    the buffer in a `.buffer` UOp. Construction sites that built the
    L5..L13 `{ shape, dtype, buffer }` record migrate to this helper. -/
def Tensor.ofBuffer (buf : Runtime.BufferHandle) (shape : Shape) (dtype : Dtype) : Tensor :=
  { uop := .buffer buf.raw shape dtype, dtype := dtype }

/-- Construct the canonical storage-free tensor representation exactly when
    the logical shape has zero elements.  A nonzero registry handle identifies
    the Tensor; raw buffer zero means there is intentionally no Metal object.
    This invariant is range-independent and prevents callers from encoding an
    empty tensor as a dummy one-element allocation. -/
def Tensor.ofEmpty? (shape : Shape) (dtype : Dtype) : Option Tensor :=
  if Tgrad.numel shape == 0 then
    some (Tensor.ofBuffer { raw := 0, size := 0 } shape dtype)
  else none

example : (Tensor.ofEmpty? [0] .int32_).map Tensor.shape = some [0] := by
  native_decide

example : (Tensor.ofEmpty? [0] .int64_).map Tensor.dtype = some .int64_ := by
  native_decide

example : (Tensor.ofEmpty? [0] .float32_).map (fun t => t.buffer.raw) = some 0 := by
  native_decide

example : (Tensor.ofEmpty? [0] .bfloat16_).map (fun t => t.buffer.size) = some 0 := by
  native_decide

example : (Tensor.ofEmpty? [1] .int32_).isNone := by
  native_decide

/-! ## View methods (L14.B.1)

    Each composes a movement node onto `t.uop`. Pure (no IO); no
    buffer alloc; no dispatch; same `dtype` carried through. The
    resulting Tensor's `shape` is derived by walking the chain. -/

/-- 2-D transpose (axes [1, 0]). For higher-rank inputs use `permute`
    with explicit axis order. -/
def Tensor.transpose (t : Tensor) : Tensor :=
  { t with uop := .permute t.uop [1, 0] }

/-- General permute by `axes`. `axes.length` should match
    `t.shape.length`; mismatches degrade gracefully (out-of-range
    indices yield 0 dim — rangeify rejects at L14.B.2). -/
def Tensor.permute (t : Tensor) (axes : List Nat) : Tensor :=
  { t with uop := .permute t.uop axes }

/-- Reshape to `newShape`. Must preserve numel — rangeify validates
    at L14.B.2. -/
def Tensor.reshape (t : Tensor) (newShape : Shape) : Tensor :=
  { t with uop := .reshape t.uop newShape }

/-- Expand (broadcast) to `newShape`. Each input dim must be 1 or
    equal to the corresponding target dim. -/
def Tensor.expand (t : Tensor) (newShape : Shape) : Tensor :=
  { t with uop := .expand t.uop newShape }

/-- Per-axis slice. -/
def Tensor.slice (t : Tensor) (slices : List Slice) : Tensor :=
  { t with uop := .slice t.uop slices }

/-- numel over the tensor's shape. -/
def Tensor.numel (t : Tensor) : Nat := Tgrad.numel t.shape

/-- Number of bytes the tensor occupies in GPU memory. -/
def Tensor.sizeBytes (t : Tensor) : Nat := t.numel * t.dtype.sizeBytes

/-- L14.B.1: query whether `t.uop`'s root is a BUFFER (the
    pre-rangeify-wiring contract). Used by the matmul FFI to raise
    `MatmulOnNonBufferUop` if either input has a movement-op root. -/
def Tensor.isBufferUop (t : Tensor) : Bool :=
  match t.uop with
  | .buffer _ _ _ => true
  | _             => false

end Tgrad

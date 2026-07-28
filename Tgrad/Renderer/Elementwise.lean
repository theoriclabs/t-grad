import Tgrad.Renderer.Metal
import Tgrad.Schedule.View

/-! # Tgrad.Renderer.Elementwise — one kernel for every pointwise binary op

  Matmul needed a bespoke generator: an accumulator, a K-loop, a tiled
  WMMA fragment schedule. Elementwise needs none of that. One thread
  computes one output element, so the whole kernel is two loads, one
  operator and a store:

      out[i] = a[idx_a(i)] OP b[idx_b(i)]

  Both indices come from `Schedule.View.indexOf`, which already handles
  permute, reshape, slice and stride-0 broadcast. So operating on views
  is free here rather than a special case — `a.T + b` needs no code
  that `a + b` does not.

  This is why the plan puts the *easiest* op through the general path
  before adding more ops specially: the operator is a table row, and
  everything expensive is shared.

  Scope: rank 0 through 3, matching Metal's dispatch dimensionality and the
  current public Tensor constructor. Differently-ranked operands are padded
  on the left with stride-zero size-one axes before `View.expand`.
-/
namespace Tgrad
namespace Renderer
namespace Metal

/-- MSL scalar type for a Tgrad dtype. `none` means the backend has no
    spelling for it yet, which is a refusal rather than a fallback: a
    silently wrong element type would corrupt every load and store. -/
def mslScalarType : Dtype → Option String
  | .bfloat16_ => some "bfloat"
  | .float32_  => some "float"
  | .float16_  => some "half"
  | .int32_    => some "int"
  | _          => none

/-- MSL spelling of the pointwise operators that are meaningful for a
    float dtype. The integer and comparison members of `BinOp`
    (`shl`, `andB`, `cmplt`, …) are deliberately absent: they are not
    meaningful for bf16 and become reachable when the dtype lattice is
    wired, not before. Returning `none` keeps that honest — an
    unsupported op fails to build a kernel rather than emitting
    something plausible. -/
def elementwiseOpStr : BinOp → Option String
  | .add => some "+"
  | .sub => some "-"
  | .mul => some "*"
  | _    => none

/-- Operator name for the kernel symbol. The operator MUST appear in the
    kernel name: the compiled-library cache is keyed on that name, and
    without it `a - b` over the same views as `a + b` hits the cached
    add kernel and silently returns the wrong numbers. That bug was
    real, and the numpy differential is what caught it. -/
def elementwiseOpName : BinOp → String
  | .add => "add"
  | .sub => "sub"
  | .mul => "mul"
  | _    => "unsupported"

/-- Compute in the promoted output domain. Native int32 arithmetic must not
pass through float: values above `2^24` would be rounded before the store.
Every currently admitted non-int32 output is a floating dtype, including mixed
int32/float32 operations selected by `Dtype.lub`. -/
def elementwiseComputeExpr (outTy : Dtype) (opStr : String) : String :=
  if outTy == .int32_ then s!"val0{opStr}val1"
  else s!"((float)val0){opStr}((float)val1)"

/-- MSL coordinate component for one of Metal's three grid axes. -/
private def gridComponent : Nat → String
  | 0 => "gid.x"
  | 1 => "gid.y"
  | _ => "gid.z"

private def shapeTag (shape : List Nat) : String :=
  if shape.isEmpty then "scalar" else String.intercalate "x" (shape.map toString)

/-- Kernel for `out = a OP b` over a rank-0-through-3 output shape, with
per-operand index expressions so either side may be an arbitrary view. -/
def elementwiseKernelDeclRanked (op : BinOp) (outShape : List Nat)
    (aIdx bIdx : UOp) (aTy bTy outTy : Dtype) (tag : String) : Option KernelDecl :=
  if outShape.length > 3 then none else
  match elementwiseOpStr op, mslScalarType aTy, mslScalarType bTy,
        mslScalarType outTy with
  | some opStr, some aS, some bS, some outS =>
    let vars := (List.range outShape.length).map (fun i =>
      UOp.var s!"gidx{i}" .int32_)
    let coordDecls := (List.range outShape.length).map (fun i =>
      Stmt.declInt s!"gidx{i}" (gridComponent i) ((outShape[i]?).map toString))
    let outIdx := Schedule.View.indexOf (Schedule.View.contiguous outShape) vars
    some
      { name     := s!"ew_{elementwiseOpName op}_{aTy.toStr}_{bTy.toStr}_{outTy.toStr}_{tag}_{shapeTag outShape}",
        wmmaArgs := [],
        args     := [
          .buffer { qualifier := "device", baseType := outS, name := "data0" },
          .buffer { qualifier := "device", baseType := aS,   name := "data1" },
          .buffer { qualifier := "device", baseType := bS,   name := "data2" },
          .attr   { baseType := "uint3", name := "gid",
                    attrStr := "[[threadgroup_position_in_grid]]" },
        ],
        body     := coordDecls ++ [
          .loadIndexed s!"{aS} val0" "data1" aIdx,
          .loadIndexed s!"{bS} val1" "data2" bIdx,
          -- Compute in fp32 and let `storeIndexed` apply the bf16 cast,
          -- matching how the matmul kernels accumulate.
          .storeIndexedAs outS "data0" outIdx
            (elementwiseComputeExpr outTy opStr)
        ],
        trailingNewline := false }
  | _, _, _, _ => none

/-- Backward-compatible rank-2 entry used by existing callers and theorems. -/
def elementwiseKernelDecl (op : BinOp) (rows cols : Nat)
    (aIdx bIdx : UOp) (aTy bTy outTy : Dtype) (tag : String) : Option KernelDecl :=
  elementwiseKernelDeclRanked op [rows, cols] aIdx bIdx aTy bTy outTy tag


/-- **Regression obligation for a bug that shipped in this file.**

    The compiled-library cache is keyed on the kernel name. When the
    name omitted the operator, `a - b` over the same views as `a + b`
    hit the cached add kernel and returned silently wrong numbers —
    `a + b` and `a.T * b.T` both passed while `a - b` and `a * b` did
    not, because only the latter collided. Distinct operators must
    therefore produce distinct kernel identities. -/
theorem elementwise_names_separate_operators :
    (elementwiseKernelDecl .add 4 4 (.var "i" .int32_) (.var "j" .int32_) .bfloat16_ .bfloat16_ .bfloat16_ "t").map
        (fun d => d.name)
      ≠ (elementwiseKernelDecl .sub 4 4 (.var "i" .int32_) (.var "j" .int32_) .bfloat16_ .bfloat16_ .bfloat16_ "t").map
        (fun d => d.name) := by
  native_decide

theorem int32_compute_is_native :
    elementwiseComputeExpr .int32_ "+" = "val0+val1" := by
  rfl

theorem promoted_float_compute_converts_operands :
    elementwiseComputeExpr .float32_ "+" =
      "((float)val0)+((float)val1)" := by
  rfl

/-- Unsupported operators build no kernel at all, rather than emitting
    a plausible one. -/
theorem elementwise_rejects_unsupported :
    (elementwiseKernelDecl .andB 4 4 (.var "i" .int32_) (.var "j" .int32_) .bfloat16_ .bfloat16_ .bfloat16_ "t").isNone := by
  native_decide


/-! ## Reduction

  The accumulate-over-a-loop shape already exists in the matmul kernel:
  zero an accumulator, walk the contracted axis, write back. This
  generalises it to one contracted axis of a rank-2 operand, so `sum`
  and `prod` are the same kernel with a different identity element and
  operator — a table row each, exactly like the pointwise ops.

  Keepdim is the only mode: reducing axis 1 of `rows x cols` yields
  `rows x 1`. Everything downstream is rank-2, and a rank-changing
  reduce would need shape machinery that is not built yet. Saying so is
  better than silently dropping an axis.
-/

/-- Identity element for the accumulator, as MSL. -/
def reduceInit : BinOp → Option String
  | .add => some "0.0f"
  | .mul => some "1.0f"
  | _    => none

def reduceOpName : BinOp → String
  | .add => "sum"
  | .mul => "prod"
  | _    => "unsupported"

/-- `out = reduce OP over one axis of a rank-2 operand`, keepdim.

    `idxOf` receives the output coordinate and the loop variable and
    returns the operand index, so the caller decides which axis is
    contracted and views cost nothing. -/
def reduceKernelDecl (op : BinOp) (rows cols axis : Nat)
    (operandIdx : UOp) (aTy outTy : Dtype) (tag : String) : Option KernelDecl :=
  match elementwiseOpStr op, reduceInit op, mslScalarType aTy, mslScalarType outTy with
  | some opStr, some init, some aS, some outS =>
    let bound := if axis == 1 then cols else rows
    let outLen := if axis == 1 then rows else cols
    some
      { name     := s!"red_{reduceOpName op}_{aTy.toStr}_{outTy.toStr}_a{axis}_{tag}_{rows}x{cols}",
        wmmaArgs := [],
        args     := [
          .buffer { qualifier := "device", baseType := outS, name := "data0" },
          .buffer { qualifier := "device", baseType := aS,   name := "data1" },
          .attr   { baseType := "uint3", name := "gid",
                    attrStr := "[[threadgroup_position_in_grid]]" },
        ],
        body     := [
          .declInt "gidx0" "gid.x" (some s!"{outLen}"),
          .declFloat "acc" init,
          .forLoop "ridx0" bound [
            .loadIndexed s!"{aS} val0" "data1" operandIdx,
            .assign "acc" s!"acc{opStr}((float)val0)"
          ],
          .storeIndexedAs outS "data0" (.var "gidx0" .int32_) "acc"
        ],
        trailingNewline := false }
  | _, _, _, _ => none

/-- Distinct reductions must not share a kernel identity, for the same
    cache reason the pointwise operators must not. -/
theorem reduce_names_separate_operators :
    (reduceKernelDecl .add 4 4 1 (.var "i" .int32_) .bfloat16_ .bfloat16_ "t").map
        (fun d => d.name)
      ≠ (reduceKernelDecl .mul 4 4 1 (.var "i" .int32_) .bfloat16_ .bfloat16_ "t").map
        (fun d => d.name) := by
  native_decide

/-- Reducing a different axis is a different kernel. -/
theorem reduce_names_separate_axes :
    (reduceKernelDecl .add 4 4 0 (.var "i" .int32_) .bfloat16_ .bfloat16_ "t").map
        (fun d => d.name)
      ≠ (reduceKernelDecl .add 4 4 1 (.var "i" .int32_) .bfloat16_ .bfloat16_ "t").map
        (fun d => d.name) := by
  native_decide


/-! ## Fused reduce-of-elementwise

  `reduce add (mul a b)` is matmul. Lowering it literally — materialise
  the elementwise product, then reduce it — allocates `M*N*K` elements:
  two gigabytes at 1024³. tinygrad never does that; its scheduler fuses
  the pair into one kernel whose accumulator consumes the product as it
  is produced.

  This is that fused lowering, generically: one thread per output
  element, accumulating `a OP_ew b` along the contracted axis, with no
  intermediate. It is the general path that makes matmul expressible
  without a matmul-shaped generator.

  It is not a replacement for the WMMA kernels. Those stay as an
  optimisation for shapes that qualify; this is the correct-everywhere
  fallback they are selected *over*. The differential is what keeps the
  two honest about agreeing.
-/

/-- Fused `reduce(redOp) over axis of (a ewOp b)`, keepdim.

    Operand indices are supplied by the caller from the `View` algebra
    and may reference the output coordinates `gidx0`/`gidx1` and the
    contraction variable `ridx0`, so expanded and permuted operands need
    no special handling. -/
def fusedReduceKernelDecl (redOp ewOp : BinOp) (outRows outCols bound : Nat)
    (aIdx bIdx : UOp) (aTy bTy outTy : Dtype) (tag : String) :
    Option KernelDecl :=
  match elementwiseOpStr redOp, reduceInit redOp, elementwiseOpStr ewOp,
        mslScalarType aTy, mslScalarType bTy, mslScalarType outTy with
  | some redStr, some init, some ewStr, some aS, some bS, some outS =>
    let outIdx : UOp :=
      .binop .add
        (.binop .mul (.var "gidx0" .int32_)
                     (.const .int32_ (.i (Int.ofNat outCols))) .int32_)
        (.var "gidx1" .int32_) .int32_
    some
      { name := s!"fused_{reduceOpName redOp}_{elementwiseOpName ewOp}_{aTy.toStr}_{bTy.toStr}_{outTy.toStr}_{tag}_{outRows}x{outCols}x{bound}",
        wmmaArgs := [],
        args     := [
          .buffer { qualifier := "device", baseType := outS, name := "data0" },
          .buffer { qualifier := "device", baseType := aS,   name := "data1" },
          .buffer { qualifier := "device", baseType := bS,   name := "data2" },
          .attr   { baseType := "uint3", name := "gid",
                    attrStr := "[[threadgroup_position_in_grid]]" },
        ],
        body     := [
          .declInt "gidx0" "gid.x" (some s!"{outRows}"),
          .declInt "gidx1" "gid.y" (some s!"{outCols}"),
          .declFloat "acc" init,
          .forLoop "ridx0" bound [
            .loadIndexed s!"{aS} val0" "data1" aIdx,
            .loadIndexed s!"{bS} val1" "data2" bIdx,
            .assign "acc" s!"acc{redStr}(((float)val0){ewStr}((float)val1))"
          ],
          .storeIndexedAs outS "data0" outIdx "acc"
        ],
        trailingNewline := false }
  | _, _, _, _, _, _ => none

/-- The fused kernel's identity must separate both operators, not just
    one: `reduce add (mul ..)` and `reduce add (add ..)` are different
    computations that would otherwise share a cache entry. -/
theorem fused_names_separate_inner_operator :
    (fusedReduceKernelDecl .add .mul 4 4 4 (.var "i" .int32_) (.var "j" .int32_)
        .bfloat16_ .bfloat16_ .bfloat16_ "t").map (fun d => d.name)
      ≠ (fusedReduceKernelDecl .add .add 4 4 4 (.var "i" .int32_) (.var "j" .int32_)
        .bfloat16_ .bfloat16_ .bfloat16_ "t").map (fun d => d.name) := by
  native_decide

end Metal
end Renderer
end Tgrad

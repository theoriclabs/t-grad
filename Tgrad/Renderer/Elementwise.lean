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

  Scope: rank-2, same output shape, one dtype. Broadcasting between
  differently-shaped operands is a `View.expand` away but is not wired
  until the dtype/promotion step, so it is rejected rather than guessed.
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

/-- Kernel for `out = a OP b` over `rows x cols`, with per-operand
    index expressions so either side may be an arbitrary view.

    `gidx0` walks rows and `gidx1` walks columns, matching the
    convention `Pipeline.viewIndexUOpForA/B` already emit, so the same
    `View`-derived index UOps work unchanged. -/
def elementwiseKernelDecl (op : BinOp) (rows cols : Nat)
    (aIdx bIdx : UOp) (aTy bTy outTy : Dtype) (tag : String) : Option KernelDecl :=
  match elementwiseOpStr op, mslScalarType aTy, mslScalarType bTy,
        mslScalarType outTy with
  | some opStr, some aS, some bS, some outS =>
    let outIdx : UOp :=
      .binop .add
        (.binop .mul (.var "gidx0" .int32_)
                     (.const .int32_ (.i (Int.ofNat cols))) .int32_)
        (.var "gidx1" .int32_) .int32_
    some
      { name     := s!"ew_{elementwiseOpName op}_{aTy.toStr}_{bTy.toStr}_{outTy.toStr}_{tag}_{rows}x{cols}",
        wmmaArgs := [],
        args     := [
          .buffer { qualifier := "device", baseType := outS, name := "data0" },
          .buffer { qualifier := "device", baseType := aS,   name := "data1" },
          .buffer { qualifier := "device", baseType := bS,   name := "data2" },
          .attr   { baseType := "uint3", name := "gid",
                    attrStr := "[[threadgroup_position_in_grid]]" },
        ],
        body     := [
          .declInt "gidx0" "gid.x" (some s!"{rows}"),
          .declInt "gidx1" "gid.y" (some s!"{cols}"),
          .loadIndexed s!"{aS} val0" "data1" aIdx,
          .loadIndexed s!"{bS} val1" "data2" bIdx,
          -- Compute in fp32 and let `storeIndexed` apply the bf16 cast,
          -- matching how the matmul kernels accumulate.
          -- Compute in fp32 regardless of operand width, then cast once
          -- to the promoted output type.
          .storeIndexedAs outS "data0" outIdx
            s!"((float)val0){opStr}((float)val1)"
        ],
        trailingNewline := false }
  | _, _, _, _ => none


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

/-- Unsupported operators build no kernel at all, rather than emitting
    a plausible one. -/
theorem elementwise_rejects_unsupported :
    (elementwiseKernelDecl .andB 4 4 (.var "i" .int32_) (.var "j" .int32_) .bfloat16_ .bfloat16_ .bfloat16_ "t").isNone := by
  native_decide

end Metal
end Renderer
end Tgrad

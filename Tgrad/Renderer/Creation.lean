import Tgrad.Renderer.Elementwise
import Tgrad.Schedule.View

/-! # Tgrad.Renderer.Creation — nullary creation kernels

  Nullary tensor creation (`full` / `ones` / `zeros` / `arange`) is not an
  elementwise binary op: there are no operands, only a destination
  buffer and a constant. Keeping the generator in its own module
  rather than growing `Elementwise.lean` makes that boundary
  visible — Elementwise stays "two loads, one operator, one store";
  Creation is "one store of a constant per thread".

  Conventions match `elementwiseKernelDeclRanked`: `KernelDecl`,
  typed `Stmt` constructors, dtype-parameterised kernel name, and
  rank 0–3 (Metal's three grid axes plus scalar). The fill value is
  baked into the MSL as a literal; the name carries a fill tag so
  distinct constants cannot collide in the compiled-library cache.
-/
namespace Tgrad
namespace Renderer
namespace Metal

/-- MSL coordinate component for one of Metal's three grid axes. -/
private def fillGridComponent : Nat → String
  | 0 => "gid.x"
  | 1 => "gid.y"
  | _ => "gid.z"

private def fillShapeTag (shape : List Nat) : String :=
  if shape.isEmpty then "scalar" else String.intercalate "x" (shape.map toString)

/-- MSL right-hand side for a constant fill of the given dtype.

    Floating values use a float literal; `storeScalarStmts` performs the
    dtype-specific storage lowering (including portable bf16 packing).
    int32 uses a truncated integer literal.
    Unsupported dtypes refuse rather than invent a spelling. -/
def fillValueLiteral (ty : Dtype) (fill : Float) : Option String :=
  match ty with
  | .int32_              => some (toString fill.toInt64)
  | .float32_ | .bfloat16_ => some (fill.toString ++ "f")
  | _                    => none

/-- Cache-key fragment for the fill value. Floats use IEEE bits so
    distinct bit patterns never share a kernel identity; ints use the
    truncated integer. -/
def fillValueTag (ty : Dtype) (fill : Float) : String :=
  match ty with
  | .int32_ => toString fill.toInt64
  | _       => toString fill.toBits

/-- One-thread-per-element constant fill: `out[i] = fill`.

    Returns `none` when the shape rank exceeds Metal's three-axis
    grid or the dtype has no MSL scalar spelling — refusal, not a
    silent fallback. -/
def fillKernelDecl (shape : List Nat) (ty : Dtype) (fill : Float) :
    Option KernelDecl :=
  if shape.length > 3 then none else
  match mslStorageType ty, fillValueLiteral ty fill with
  | some outS, some lit =>
    let tag := fillValueTag ty fill
    let vars := (List.range shape.length).map (fun i =>
      UOp.var s!"gidx{i}" .int32_)
    let coordDecls := (List.range shape.length).map (fun i =>
      Stmt.declInt s!"gidx{i}" (fillGridComponent i) ((shape[i]?).map toString))
    let outIdx := Schedule.View.indexOf (Schedule.View.contiguous shape) vars
    some
      { name     := s!"fill_{ty.toStr}_{tag}_{fillShapeTag shape}",
        wmmaArgs := [],
        args     := [
          .buffer { qualifier := "device", baseType := outS, name := "data0" },
          .attr   { baseType := "uint3", name := "gid",
                    attrStr := "[[threadgroup_position_in_grid]]" },
        ],
        body     := coordDecls ++
          storeScalarStmts "result" ty outS "data0" outIdx lit,
        trailingNewline := false }
  | _, _ => none


/-- **Regression obligation for library-cache collisions.**

    The compiled-library cache is keyed on the kernel name. An i32
    fill and an f32 fill over the same shape must not share an
    identity — that class of bug shipped once for elementwise ops
    (`elementwise_names_separate_operators`) and would silently
    reinterpret bits across the dtype boundary. -/
theorem fill_names_separate_dtypes :
    (fillKernelDecl [2, 2] .float32_ 1.0).map (fun d => d.name)
      ≠ (fillKernelDecl [2, 2] .int32_ 1.0).map (fun d => d.name) := by
  native_decide

/-- bf16 and f32 fills are also distinct identities. -/
theorem fill_names_separate_float_dtypes :
    (fillKernelDecl [2, 2] .bfloat16_ 1.0).map (fun d => d.name)
      ≠ (fillKernelDecl [2, 2] .float32_ 1.0).map (fun d => d.name) := by
  native_decide

/-- Unsupported dtypes build no kernel, rather than emitting a
    plausible one with a guessed MSL type. -/
theorem fill_rejects_float64 :
    (fillKernelDecl [2, 2] .float64_ 1.0).isNone := by
  native_decide

/-- Constant creation has no load, but its rendered destination and packing
    must obey the same portable bf16 storage contract.  This is the structural
    check that distinguishes the fix from the old native-`bfloat` kernel even
    on an M4 where both versions compile and produce the same numbers. -/
theorem bf16_fill_render_is_portable :
    portableBf16ScalarSource
      ((fillKernelDecl [2, 2] .bfloat16_ 1.0).map renderKernel |>.getD "")
      ["data0"] false = true := by
  native_decide


-- ----------------------------------------------------------------------
-- Arithmetic-progression creation (`Tensor.arange`).
-- ----------------------------------------------------------------------

private def rangeIntTag (value : Int) : String :=
  if value < 0 then s!"n{(-value).toNat}" else s!"p{value.toNat}"

/-- Compute expression for one integral arithmetic-progression element.

    Lean applies pinned tinygrad's one-sided lo/hi admission rule. The kernel
    then evaluates the progression in a signed 64-bit Metal `long` before the
    final dtype storage cast, preserving admitted wrap at that cast while also
    avoiding intermediate int32 overflow for a progression such as
    `[-2^31, -1, 2^31-2]`. -/
def rangeValueExpr (ty : Dtype) (start step : Int) : Option String :=
  let integral :=
    s!"(((long){start}) + (((long)gid.x) * ((long){step})))"
  match ty with
  | .int32_ => some integral
  | .float32_ | .bfloat16_ => some s!"((float){integral})"
  | _ => none

/-- One-thread-per-element arithmetic progression: `out[i] = start + i*step`.

    Length, normalized start, step, dtype, and representability are decided by
    the Lean range resolver before this declaration can be constructed. -/
def rangeKernelDecl (length : Nat) (ty : Dtype) (start step : Int) :
    Option KernelDecl :=
  match mslStorageType ty, rangeValueExpr ty start step with
  | some outS, some rhs =>
    let outIdx := UOp.var "gid.x" .uint32_
    some
      { name := s!"range_{ty.toStr}_{rangeIntTag start}_{rangeIntTag step}_{length}",
        wmmaArgs := [],
        args := [
          .buffer { qualifier := "device", baseType := outS, name := "data0" },
          .attr { baseType := "uint3", name := "gid",
                  attrStr := "[[threadgroup_position_in_grid]]" },
        ],
        body := storeScalarStmts "result" ty outS "data0" outIdx rhs,
        trailingNewline := false }
  | _, _ => none

theorem range_names_separate_values :
    (rangeKernelDecl 10 .int32_ 0 1).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .int32_ 1 1).map (fun d => d.name) := by
  native_decide

theorem range_names_separate_steps :
    (rangeKernelDecl 10 .int32_ 0 1).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .int32_ 0 2).map (fun d => d.name) := by
  native_decide

theorem range_rejects_float16 :
    (rangeKernelDecl 10 .float16_ 0 1).isNone := by
  native_decide

theorem bf16_range_render_is_portable :
    portableBf16ScalarSource
      ((rangeKernelDecl 10 .bfloat16_ 0 1).map renderKernel |>.getD "")
      ["data0"] false = true := by
  native_decide

end Metal
end Renderer
end Tgrad

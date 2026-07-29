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

inductive RangeKernelScalar where
  | integral (value : Int)
  | floating (value : Float)
  deriving Repr

namespace RangeKernelScalar

/-- Bit-exact cache identity.  Floating values are keyed by their IEEE-754
    binary64 payload, not by a decimal rendering that might merge distinct
    authoring inputs. -/
def tag : RangeKernelScalar → String
  | .integral value =>
      if value < 0 then s!"in{(-value).toNat}" else s!"ip{value.toNat}"
  | .floating value => s!"f{value.toBits}"

end RangeKernelScalar

private def rangeStorageType : Dtype → Option String
  | .int8_ => some "char"
  | .int32_ => some "int"
  | .int64_ => some "long"
  | .bfloat16_ => some "ushort"
  | .float32_ => some "float"
  | _ => none

private def f32BitsExpr (value : Float) : String :=
  s!"as_type<float>({value.toFloat32.toBits}u)"

private def rangeScalarFloat? : RangeKernelScalar → Option Float
  | .integral value =>
      let converted := Float.ofInt value
      if converted.isFinite then some converted else none
  | .floating value => some value

private def rangeFloatParts? (start step : RangeKernelScalar) :
    Option (Float × Float) := do
  let stepValue ← rangeScalarFloat? step
  let offset ← match start, step with
    | .integral start, .integral step => rangeScalarFloat? (.integral (start - step))
    | .floating start, .floating step => some (start - step)
    | _, _ => none
  some (stepValue, offset)

/-- Compute expression for one integral arithmetic-progression element.

    Lean applies pinned tinygrad's one-sided lo/hi admission rule. The kernel
    then evaluates the progression in a signed 64-bit Metal `long` before the
    final dtype storage cast, preserving admitted wrap at that cast while also
    avoiding intermediate int32 overflow for a progression such as
    `[-2^31, -1, 2^31-2]`. -/
def rangeValueExpr (ty : Dtype) (start step : RangeKernelScalar) : Option String :=
  match ty, start, step with
  | .int8_, .integral start, .integral step =>
    let integral :=
      s!"(((long){start}) + (((long)gid.x) * ((long){step})))"
    some integral
  | .int32_, .integral start, .integral step =>
    let integral :=
      s!"(((long){start}) + (((long)gid.x) * ((long){step})))"
    some integral
  | .int64_, .integral start, .integral step =>
    let integral :=
      s!"(((long){start}) + (((long)gid.x) * ((long){step})))"
    some integral
  | .float32_, start, step => do
    let (stepValue, offset) ← rangeFloatParts? start step
    some s!"fma(((float)(gid.x + 1u)), {f32BitsExpr stepValue}, {f32BitsExpr offset})"
  | .bfloat16_, start, step => do
    let (stepValue, offset) ← rangeFloatParts? start step
    let prefixExpr := s!"(((float)(gid.x + 1u)) * {f32BitsExpr stepValue})"
    let prefixBits := s!"as_type<uint>({prefixExpr})"
    let rounded := roundedBf16BitsExpr prefixBits
    let prefixBf16 := s!"as_type<float>(((uint)({rounded} >> 16)) << 16)"
    some s!"({prefixBf16} + {f32BitsExpr offset})"
  | _, _, _ => none

/-- One-thread-per-element arithmetic progression: `out[i] = start + i*step`.

    Length, normalized start, step, dtype, and representability are decided by
    the Lean range resolver before this declaration can be constructed. -/
def rangeKernelDecl (length : Nat) (ty : Dtype)
    (start step : RangeKernelScalar) :
    Option KernelDecl :=
  match rangeStorageType ty, rangeValueExpr ty start step with
  | some outS, some rhs =>
    let outIdx := UOp.var "gid.x" .uint32_
    some
      { name := s!"range_{ty.toStr}_{start.tag}_{step.tag}_{length}",
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
    (rangeKernelDecl 10 .int32_ (.integral 0) (.integral 1)).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .int32_ (.integral 1) (.integral 1)).map (fun d => d.name) := by
  native_decide

theorem range_names_separate_steps :
    (rangeKernelDecl 10 .int32_ (.integral 0) (.integral 1)).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .int32_ (.integral 0) (.integral 2)).map (fun d => d.name) := by
  native_decide

theorem range_names_separate_float_bits :
    (rangeKernelDecl 10 .float32_ (.floating 0.0) (.floating 0.0)).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .float32_ (.floating 0.0) (.floating (-0.0))).map (fun d => d.name) := by
  native_decide

theorem range_names_separate_numeric_modes :
    (rangeKernelDecl 10 .float32_ (.integral 0) (.integral 1)).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .float32_ (.floating 0.0) (.floating 1.0)).map (fun d => d.name) := by
  native_decide

theorem range_names_separate_dtypes :
    (rangeKernelDecl 10 .float32_ (.floating 0.0) (.floating 0.25)).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .bfloat16_ (.floating 0.0) (.floating 0.25)).map (fun d => d.name) := by
  native_decide

theorem range_names_separate_lengths :
    (rangeKernelDecl 10 .float32_ (.floating 0.0) (.floating 0.25)).map (fun d => d.name) ≠
      (rangeKernelDecl 11 .float32_ (.floating 0.0) (.floating 0.25)).map (fun d => d.name) := by
  native_decide

theorem range_names_separate_nearby_float_bits :
    (rangeKernelDecl 10 .float32_ (.floating 0.0)
      (.floating (Float.ofBits 4599075939470750515))).map (fun d => d.name) ≠
      (rangeKernelDecl 10 .float32_ (.floating 0.0)
        (.floating (Float.ofBits 4599075939470750516))).map (fun d => d.name) := by
  native_decide

theorem range_rejects_float16 :
    (rangeKernelDecl 10 .float16_ (.integral 0) (.integral 1)).isNone := by
  native_decide

theorem range_admits_required_int_storage :
    (rangeKernelDecl 10 .int8_ (.integral 0) (.integral 1)).isSome &&
      (rangeKernelDecl 10 .int64_ (.integral 0) (.integral 1)).isSome = true := by
  native_decide

theorem bf16_range_render_is_portable :
    portableBf16ScalarSource
      ((rangeKernelDecl 10 .bfloat16_ (.integral 0) (.integral 1)).map renderKernel |>.getD "")
      ["data0"] false = true := by
  native_decide

end Metal
end Renderer
end Tgrad

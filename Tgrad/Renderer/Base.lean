/-! # Tgrad.Renderer.Base — scalar types + addr-space + value-ty AST

  Lift from theograd_phases/01_renderer/Demo.lean.

  L3 keeps the typed MSL AST (Option B in phase 01's design doc) at
  module-scope so downstream `Renderer.CStyle` / `Renderer.Metal`
  share the grammar surface.
-/
namespace Tgrad

namespace Renderer

inductive ScalarTy where
  | int_ | int64_ | float_ | bfloat_ | bool_ | void_
  deriving BEq, Repr, Inhabited

inductive ValueTy where
  | scalar (s : ScalarTy)
  | vec    (s : ScalarTy) (n : Nat)
  deriving BEq, Repr, Inhabited

inductive AddrSpace where
  | global_ | threadgroup_ | reg_
  deriving BEq, Repr, Inhabited

/-- The renderer's `Dtype` is richer than Tgrad.Dtype: it carries
    address-space info for pointers. (Tgrad.Dtype is the value-only
    lattice; this is the renderer-internal type.) -/
inductive RenderDtype where
  | val (v : ValueTy)
  | ptr (base : ValueTy) (addr : AddrSpace) (size : Nat)
  deriving BEq, Repr, Inhabited

def ScalarTy.toName : ScalarTy → String
  | .int_    => "int"
  | .int64_  => "long"
  | .float_  => "float"
  | .bfloat_ => "bfloat"
  | .bool_   => "bool"
  | .void_   => "void"

def ValueTy.toName : ValueTy → String
  | .scalar s => s.toName
  | .vec s n  => s.toName ++ toString n

def AddrSpace.toPrefix : AddrSpace → String
  | .global_      => "device "
  | .threadgroup_ => "threadgroup "
  | .reg_         => ""

def RenderDtype.toName : RenderDtype → String
  | .val v             => v.toName
  | .ptr base addr _sz => addr.toPrefix ++ base.toName ++ "*"

end Renderer

end Tgrad

import Tgrad.Renderer.CStyle

/-! # Tgrad.Renderer.CodeForOp — per-op C-style emit

  Lift from theograd_phases/01_renderer/Demo.lean's per-op render
  functions. At L3 this is the lower-level emit primitive each
  renderer composes; Metal-specific specialisations live in
  `Renderer.Metal`.
-/
namespace Tgrad

namespace Renderer

/-- C-style binop emission. -/
def emitBinop (op : Binop) (a b : String) : String :=
  let a' := if op.isAssoc then stripParens a else a
  let b' := if op.isAssoc then stripParens b else b
  "(" ++ a' ++ op.toStr ++ b' ++ ")"

/-- C-style scalar cast: `((Ty)(expr))`. -/
def emitCastVal (toTy : ValueTy) (e : String) : String :=
  "((" ++ toTy.toName ++ ")(" ++ e ++ "))"

/-- C-style pointer cast: `((addr Ty*)(expr))`. -/
def emitCastPtr (base : ValueTy) (addr : AddrSpace) (e : String) : String :=
  "((" ++ addr.toPrefix ++ base.toName ++ "*)(" ++ e ++ "))"

/-- C-style pointer arithmetic: `(buf+idx)`. -/
def emitIndex (buf idx : String) : String :=
  "(" ++ buf ++ "+" ++ idx ++ ")"

/-- C-style dereference: `(*addr)`. -/
def emitLoad (addr : String) : String :=
  "(*" ++ addr ++ ")"

/-- C-style assignment: `*addr = value;` (callers add the leading indent). -/
def emitStore (addr value : String) : String :=
  "*" ++ addr ++ " = " ++ value ++ ";"

/-- Vector element access: `expr.x` / `.y` / `.z` / `.w` for lanes 0..3,
    `v{N}` for larger lanes. -/
def emitGep (e : String) (lane : Nat) : String :=
  let suffix := match lane with
    | 0 => "x" | 1 => "y" | 2 => "z" | 3 => "w"
    | n => "v" ++ toString n
  e ++ "." ++ suffix

end Renderer

end Tgrad

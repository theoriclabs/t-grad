import Tgrad.Renderer.Base

/-! # Tgrad.Renderer.CStyle — shared C-style render helpers

  Lift from theograd_phases/01_renderer/Demo.lean. Subset shared
  across backends (CUDA / OpenCL / Metal would all use these).
-/
namespace Tgrad

namespace Renderer

inductive Binop where
  | add | sub | mul | shl | shr | xor | and_ | or_
  deriving BEq, Repr

def Binop.toStr : Binop → String
  | .add  => "+" | .sub  => "-" | .mul => "*"
  | .shl  => "<<" | .shr => ">>" | .xor => "^"
  | .and_ => "&"  | .or_ => "|"

def Binop.isAssoc : Binop → Bool
  | .add | .mul | .xor | .and_ | .or_ => true
  | _ => false

/-- Strip a single matched pair of outer parens. Used for
    associative-ALU children where tinygrad flattens
    `(a+(b+c))` → `(a+b+c)`. Operates on the character list to
    dodge 4.29's `String.drop` / `dropRight` Slice quirks. -/
def stripParens (s : String) : String :=
  let cs := s.toList
  match cs with
  | '(' :: rest =>
      match rest.reverse with
      | ')' :: mid => String.ofList mid.reverse
      | _          => s
  | _ => s

end Renderer

end Tgrad

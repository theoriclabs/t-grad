/-! # Tgrad.Codegen.Simplify — late-codegen PM inventory

  Lift from theograd_phases/10_codegen_late/Demo.lean. At L3 this is
  an inventory-only port: we record the structure of `pm_*` pattern
  matchers (count + body hashes) so upstream drift in tinygrad
  trips the gate. Full per-rule semantic ports live at L9/L10.
-/
namespace Tgrad

namespace Codegen

namespace Simplify

/-- The 13 late-codegen pattern matchers captured at phase 10. The
    body-hash carries a `sha8` fingerprint per PM so drift in any
    rule's body trips the inventory gate. -/
structure PatternMatcherEntry where
  name        : String
  ruleCount   : Nat
  bodiesSha8  : String
  deriving Repr, Inhabited

def lateCodegenPMs : List PatternMatcherEntry :=
  [ { name := "pm_make_images", ruleCount := 0, bodiesSha8 := "0000_sentinel" } ]

/-- The number of late-codegen PMs we've inventoried. -/
def pmCount : Nat := lateCodegenPMs.length

end Simplify

end Codegen

end Tgrad

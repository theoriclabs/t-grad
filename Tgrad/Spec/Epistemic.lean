/-! # Tgrad.Spec.Epistemic — claims with upgrade paths

The value is only half of a claim. The other half is the work that established
it, or the work that would change it. This is the four-state vocabulary from
Clockworks' “Lean as a Universal Specification Language” guide.
-/

namespace Tgrad.Spec

inductive Epistemic (alpha : Type) where
  | confirmed (value : alpha) (evidence : String)
  | tentative (value : alpha) (basis : String) (upgradeBy : String)
  | unknown (whatMissing : String) (resolveBy : String)
  | deferred (whatMissing : String) (rationale : String)
  deriving Repr, Inhabited

namespace Epistemic

def value? : Epistemic alpha -> Option alpha
  | .confirmed value _ => some value
  | .tentative value _ _ => some value
  | .unknown _ _ => none
  | .deferred _ _ => none

def isConfirmed : Epistemic alpha -> Bool
  | .confirmed _ evidence => !evidence.isEmpty
  | _ => false

/-- An unknown/tentative claim is actionable only when its upgrade prose is
specific enough to be non-empty. This is deliberately a weak structural check;
semantic quality still requires review. -/
def hasUpgradePath : Epistemic alpha -> Bool
  | .confirmed _ evidence => !evidence.isEmpty
  | .tentative _ basis upgradeBy =>
      !basis.isEmpty && !upgradeBy.isEmpty
  | .unknown whatMissing resolveBy =>
      !whatMissing.isEmpty && !resolveBy.isEmpty
  | .deferred whatMissing rationale =>
      !whatMissing.isEmpty && !rationale.isEmpty

end Epistemic
end Tgrad.Spec

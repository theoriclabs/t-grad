import Tgrad.Requirements.World

/-! # Tgrad.Requirements.Relation — typed observation equivalence

Requirements select relations from this algebra rather than naming a
comparator in an unchecked string.  The pilot keeps the algebra intentionally
small; new constructors should enter only when a world requirement needs one.
-/

namespace Tgrad.Requirements

/-- A non-negative rational bound represented without floating-point
comparison in the specification layer. -/
structure RationalBound where
  numerator : Nat
  denominator : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

def RationalBound.wellFormed (bound : RationalBound) : Bool :=
  bound.denominator > 0

inductive ObservationDimension where
  | importResolution
  | publicSurface
  | value
  | shape
  | dtype
  | exception
  | effect
  | storageAndLifetime
  | performance
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Relations over observations in the problem world.  These constructors do
not prescribe how Tgrad implements the behavior. -/
inductive ObservationRelation where
  | importResolvesWithoutFallback (provider : String)
  | publicNamesContain (names : List String)
  | exactTensorValue
  | numericalTensorValue (relative absolute : RationalBound)
  | sameShape
  | sameDtype
  | sameException
  | sameEffects
  | sameStorageAndLifetime
  | performanceDistribution (statistic protocol : String)
  | all (relations : List ObservationRelation)
  deriving Repr, Inhabited

partial def ObservationRelation.wellFormed : ObservationRelation → Bool
  | .importResolvesWithoutFallback provider => !provider.trimAscii.isEmpty
  | .publicNamesContain names =>
      !names.isEmpty && names.all (fun name => !name.trimAscii.isEmpty)
  | .numericalTensorValue relative absolute =>
      relative.wellFormed && absolute.wellFormed
  | .performanceDistribution statistic protocol =>
      !statistic.trimAscii.isEmpty && !protocol.trimAscii.isEmpty
  | .all relations =>
      !relations.isEmpty && relations.all ObservationRelation.wellFormed
  | _ => true

partial def ObservationRelation.dimensions :
    ObservationRelation → List ObservationDimension
  | .importResolvesWithoutFallback _ => [.importResolution]
  | .publicNamesContain _ => [.publicSurface]
  | .exactTensorValue => [.value]
  | .numericalTensorValue _ _ => [.value]
  | .sameShape => [.shape]
  | .sameDtype => [.dtype]
  | .sameException => [.exception]
  | .sameEffects => [.effect]
  | .sameStorageAndLifetime => [.storageAndLifetime]
  | .performanceDistribution _ _ => [.performance]
  | .all relations =>
      (relations.flatMap ObservationRelation.dimensions).eraseDups

end Tgrad.Requirements

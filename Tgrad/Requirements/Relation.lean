import Tgrad.Requirements.World

/-! # Tgrad.Requirements.Relation — typed observation-relation descriptors

Requirements select relations from this algebra rather than naming a
comparator in an unchecked string.  The pilot keeps the algebra intentionally
small; new constructors should enter only when a world requirement needs one.
These constructors currently classify required dimensions; they do not yet
denote executable relations over typed traces.
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
  | exceptionStage
  | exceptionClass
  | exceptionMessage
  | terminalOutcome
  | effect
  | objectIdentity
  | readbackStability
  | inputImmutability
  | storageAndLifetime
  | performance
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Descriptors for relations over observations in the problem world.  They do
not prescribe how Tgrad implements the behavior, and executable denotations are
an explicit scale-gate obligation. -/
inductive ObservationRelation where
  | importResolvesWithoutFallback (provider : String)
  | publicNamesContain (names : List String)
  | exactTensorValue
  | numericalTensorValue (relative absolute : RationalBound)
  | sameShape
  | sameDtype
  | sameException
  | sameExceptionStage
  | sameExceptionClass
  | sameExceptionMessage
  | sameTerminalOutcome
  | sameEffects
  | realizeReturnsSelf
  | repeatedReadbackStable
  | inputsUnchanged
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
  | .sameExceptionStage => [.exceptionStage]
  | .sameExceptionClass => [.exceptionClass]
  | .sameExceptionMessage => [.exceptionMessage]
  | .sameTerminalOutcome => [.terminalOutcome]
  | .sameEffects => [.effect]
  | .realizeReturnsSelf => [.objectIdentity]
  | .repeatedReadbackStable => [.readbackStability]
  | .inputsUnchanged => [.inputImmutability]
  | .sameStorageAndLifetime => [.storageAndLifetime]
  | .performanceDistribution _ _ => [.performance]
  | .all relations =>
      (relations.flatMap ObservationRelation.dimensions).eraseDups

end Tgrad.Requirements

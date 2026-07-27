import Tgrad.Requirements.Requirements
import Tgrad.Spec.Epistemic

/-! # Tgrad.Specification.Boundary — machine specifications and adequacy

A boundary specification constrains only shared phenomena.  Structural
coverage below is useful, but deliberately does not prove the Jackson/Zave
obligation `D ∧ S ⊨ R`; adequacy remains an epistemic claim of its own.
-/

namespace Tgrad.Specification

open Tgrad.Requirements
open Tgrad.Spec

structure SpecId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

def SpecId.valid (id : SpecId) : Bool := !id.value.trimAscii.isEmpty

structure BoundarySpec where
  id : SpecId
  requirement : RequirementId
  observes : List PhenomenonId
  controls : List PhenomenonId
  assumptions : List AssumptionId
  guarantees : List ObservationDimension
  failureSemantics : String
  deriving DecidableEq, BEq, Repr, Inhabited

def BoundarySpec.wellFormed (specification : BoundarySpec) : Bool :=
  specification.id.valid &&
  specification.requirement.valid &&
  !specification.observes.isEmpty &&
  specification.observes.all PhenomenonId.valid &&
  !specification.controls.isEmpty &&
  specification.controls.all PhenomenonId.valid &&
  specification.assumptions.all AssumptionId.valid &&
  !specification.guarantees.isEmpty &&
  !specification.failureSemantics.trimAscii.isEmpty

/-- A structural prerequisite for adequacy, not an adequacy proof. -/
def BoundarySpec.structurallyCovers
    (specification : BoundarySpec) (requirement : Requirement) : Bool :=
  specification.wellFormed &&
  specification.requirement == requirement.id &&
  requirement.monitored.all specification.observes.contains &&
  requirement.controlled.all specification.controls.contains &&
  requirement.assumptions.all specification.assumptions.contains &&
  requirement.relation.dimensions.all specification.guarantees.contains

inductive AdequacyResult where
  | argued
  | proved
  | refuted
  deriving DecidableEq, BEq, Repr, Inhabited

structure AdequacyClaim where
  requirement : RequirementId
  specification : SpecId
  assumptions : List AssumptionId
  result : Epistemic AdequacyResult
  deriving Repr, Inhabited

def AdequacyClaim.actionable (claim : AdequacyClaim) : Bool :=
  claim.requirement.valid &&
  claim.specification.valid &&
  !claim.assumptions.isEmpty &&
  claim.assumptions.all AssumptionId.valid &&
  claim.result.hasUpgradePath

end Tgrad.Specification

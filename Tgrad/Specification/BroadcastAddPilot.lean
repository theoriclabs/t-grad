import Tgrad.Requirements.BroadcastAddPilot
import Tgrad.Specification.Boundary

/-! # Prospective broadcast-add boundary specifications

These records state what the complete substituted Tgrad machine must expose at
the Python boundary. Structural coverage is checked below; adequacy remains
explicitly unknown until the finite scenarios are interpreted against each
world requirement.
-/

namespace Tgrad.Specification.BroadcastAddPilot

open Tgrad.Requirements
open Tgrad.Requirements.BroadcastAddPilot
open Tgrad.Specification
open Tgrad.Spec

private def boundaryFor (id : String) (requirement : Requirement)
    (failureSemantics : String) : BoundarySpec :=
  { id := ⟨id⟩
    requirement := requirement.id
    observes := requirement.monitored
    controls := requirement.controlled
    assumptions := requirement.assumptions
    guarantees := requirement.relation.dimensions
    failureSemantics }

def legalSameDtypeBoundary : BoundarySpec :=
  boundaryFor "SPEC-ADD-LEGAL-SAME-DTYPE-V1" legalSameDtype
    "Construction, shape/dtype observation, realization, or readback failure is recorded at its actual stage; no fallback to upstream tinygrad is permitted."

def dtypePromotionBoundary : BoundarySpec :=
  boundaryFor "SPEC-ADD-DTYPE-PROMOTION-V1" dtypePromotion
    "The observer distinguishes construction, dtype, realization, and value mismatches and records the actual exception if execution does not reach readback."

def incompatibleShapeBoundary : BoundarySpec :=
  boundaryFor "SPEC-ADD-INCOMPATIBLE-SHAPE-V1" incompatibleShape
    "Construction and subsequent shape access are separate events; exception stage, class, and public message are compared exactly."

def realizationIdempotenceBoundary : BoundarySpec :=
  boundaryFor "SPEC-REALIZE-IDEMPOTENT-V1" realizationIdempotence
    "Object identity, repeated readback, and post-call inputs are independent mandatory dimensions; failure of one does not author a verdict for another."

private def openAdequacy (requirement : Requirement)
    (specification : BoundarySpec) : AdequacyClaim :=
  { requirement := requirement.id
    specification := specification.id
    assumptions := requirement.assumptions
    result := .unknown
      "No reviewed D ∧ S ⊨ R argument exists for this prospective boundary."
      "Interpret the pinned upstream scenarios and review whether the finite observations suffice for the stated requirement." }

def legalSameDtypeAdequacy : AdequacyClaim :=
  openAdequacy legalSameDtype legalSameDtypeBoundary

def dtypePromotionAdequacy : AdequacyClaim :=
  openAdequacy dtypePromotion dtypePromotionBoundary

def incompatibleShapeAdequacy : AdequacyClaim :=
  openAdequacy incompatibleShape incompatibleShapeBoundary

def realizationIdempotenceAdequacy : AdequacyClaim :=
  openAdequacy realizationIdempotence realizationIdempotenceBoundary

theorem boundaries_structurally_cover_requirements :
    legalSameDtypeBoundary.structurallyCovers legalSameDtype ∧
    dtypePromotionBoundary.structurallyCovers dtypePromotion ∧
    incompatibleShapeBoundary.structurallyCovers incompatibleShape ∧
    realizationIdempotenceBoundary.structurallyCovers realizationIdempotence := by
  native_decide

theorem no_adequacy_claim_is_accepted :
    !legalSameDtypeAdequacy.result.isConfirmed ∧
    !dtypePromotionAdequacy.result.isConfirmed ∧
    !incompatibleShapeAdequacy.result.isConfirmed ∧
    !realizationIdempotenceAdequacy.result.isConfirmed := by
  native_decide

end Tgrad.Specification.BroadcastAddPilot

import Tgrad.Requirements.Pilot
import Tgrad.Specification.Boundary

/-! # Tgrad.Specification.Pilot — boundary contracts for the pilot

The three boundary specifications are implementation-neutral.  Their
structural coverage is checked; their real adequacy obligations remain open
until scenario semantics establish `D ∧ S ⇒ R`.
-/

namespace Tgrad.Specification.Pilot

open Tgrad.Requirements
open Tgrad.Requirements.Pilot
open Tgrad.Specification

def helpersBoundary : BoundarySpec :=
  { id := ⟨"SPEC-PY-IMPORT-HELPERS"⟩
    requirement := importHelpers.id
    observes := importHelpers.monitored
    controls := importHelpers.controlled
    assumptions := importHelpers.assumptions
    guarantees := importHelpers.relation.dimensions
    failureSemantics := "An unavailable or contaminated module resolution is a visible import failure; upstream fallback is never success." }

def addBoundary : BoundarySpec :=
  { id := ⟨"SPEC-TENSOR-ADD-BROADCAST"⟩
    requirement := broadcastAdd.id
    observes := broadcastAdd.monitored
    controls := broadcastAdd.controlled
    assumptions := broadcastAdd.assumptions
    guarantees := broadcastAdd.relation.dimensions
    failureSemantics := "Illegal broadcasting or unsupported dtype behavior is reported as a public exception, never a silent wrong tensor." }

def viewBoundary : BoundarySpec :=
  { id := ⟨"SPEC-VIEW-READBACK-TRANSPOSE"⟩
    requirement := viewReadbackLifetime.id
    observes := viewReadbackLifetime.monitored
    controls := viewReadbackLifetime.controlled
    assumptions := viewReadbackLifetime.assumptions
    guarantees := viewReadbackLifetime.relation.dimensions
    failureSemantics := "A released base reference cannot turn a surviving view into stale storage, incorrect readback, or process failure." }

def helpersAdequacy : AdequacyClaim :=
  { requirement := importHelpers.id
    specification := helpersBoundary.id
    assumptions := importHelpers.assumptions
    result := .unknown
      "a checked argument that the module-boundary guarantees imply the world requirement"
      "execute the isolated no-fallback import scenarios and encode the resulting adequacy argument" }

def addAdequacy : AdequacyClaim :=
  { requirement := broadcastAdd.id
    specification := addBoundary.id
    assumptions := broadcastAdd.assumptions
    result := .unknown
      "a checked argument connecting boundary values, shapes, dtypes, exceptions, and effects to broadcast-add compatibility"
      "define the finite pilot scenario semantics and check boundary traces against the pinned upstream runtime" }

def viewAdequacy : AdequacyClaim :=
  { requirement := viewReadbackLifetime.id
    specification := viewBoundary.id
    assumptions := viewReadbackLifetime.assumptions
    result := .unknown
      "a checked argument connecting storage validity and readback guarantees to the view-lifetime world requirement"
      "run base-release/readback scenarios and encode the storage-lifetime adequacy argument" }

/-- Structural coverage is a consistency check, not adequacy evidence. -/
theorem pilot_specs_structurally_cover_requirements :
    helpersBoundary.structurallyCovers importHelpers = true ∧
    addBoundary.structurallyCovers broadcastAdd = true ∧
    viewBoundary.structurallyCovers viewReadbackLifetime = true := by
  native_decide

theorem pilot_adequacy_obligations_are_actionable :
    helpersAdequacy.actionable = true ∧
    addAdequacy.actionable = true ∧
    viewAdequacy.actionable = true := by
  native_decide

end Tgrad.Specification.Pilot

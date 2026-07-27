import Tgrad.Requirements.Pilot

/-! # Prospective broadcast-add requirements

This module splits the retrospective, bf16-only `broadcastAdd` pilot row into
four independently observable world requirements.  It was authored before any
product candidate for this cycle.  The finite scenario and mutation domain is
frozen in `fixtures/requirements/broadcast_add_prospective_v1.json`.

These are optative requirements, not statements about what Tgrad currently
implements.  This module deliberately imports no product module.
-/

namespace Tgrad.Requirements.BroadcastAddPilot

open Tgrad.Requirements
open Tgrad.Requirements.Pilot

private def revision := "19c4d736f2bc8e26d21f08b28ffd6298408da00f"

def scenarioManifest : SourceRef :=
  { id := ⟨"SRC-BROADCAST-ADD-PROSPECTIVE-V1"⟩
    kind := .projectDecision
    revision := "baseline:51244a9"
    locator := "fixtures/requirements/broadcast_add_prospective_v1.json" }

def pinnedUpstream : SourceRef :=
  { id := ⟨"SRC-UPSTREAM-BROADCAST-ADD-PIN"⟩
    kind := .upstreamSource
    revision
    locator := "fixtures/parity/upstream_19c4d736f2bc.json" }

def returnedObjectIdentity : Phenomenon :=
  { id := ⟨"PHEN-RETURNED-OBJECT-IDENTITY"⟩
    domain := storageWorld.id
    kind := .storageIdentity
    controlledBy := .machine
    description := "Whether realization returns the same public Tensor object." }

def repeatedReadback : Phenomenon :=
  { id := ⟨"PHEN-REPEATED-READBACK"⟩
    domain := storageWorld.id
    kind := .realization
    controlledBy := .machine
    description := "Values and errors observed across repeated realization and readback." }

def postCallInputs : Phenomenon :=
  { id := ⟨"PHEN-POST-CALL-INPUTS"⟩
    domain := tensorValues.id
    kind := .tensorValue
    controlledBy := .machine
    description := "Input tensor values, shapes, and dtypes after addition and realization." }

def terminalEvent : Phenomenon :=
  { id := ⟨"PHEN-ADD-TERMINAL-EVENT"⟩
    domain := pythonProgram.id
    kind := .effect
    controlledBy := .machine
    description := "Whether the declared add trace returns a Tensor or terminates with an exception." }

def exceptionStage : Phenomenon :=
  { id := ⟨"PHEN-ADD-EXCEPTION-STAGE"⟩
    domain := pythonProgram.id
    kind := .exception
    controlledBy := .machine
    description := "The first declared trace event at which an exception becomes observable." }

def finiteScenarioAssumption : Assumption :=
  { id := ⟨"ASM-BROADCAST-ADD-FINITE-V1"⟩
    domains := [tensorValues.id, storageWorld.id]
    statement := "The prospective claim is restricted to the six deterministic scenarios and explicit exclusions in the version-one manifest."
    provenance := [scenarioManifest] }

def transformationFrameId : FrameId := ⟨"FRAME-BROADCAST-ADD-PROSPECTIVE"⟩
def realizationFrameId : FrameId := ⟨"FRAME-REALIZATION-IDEMPOTENCE-PROSPECTIVE"⟩

def profile : CompatibilityProfile :=
  { id := ⟨"PROFILE-BROADCAST-ADD-PROSPECTIVE-V1"⟩
    upstreamRevision := revision
    includedFrames := [transformationFrameId, realizationFrameId]
    environments := ["CPython on macOS", "Metal-capable Apple device"]
    description := "Six deterministic add scenarios; excludes autograd, symbolic and zero-size dimensions, broad dtype coverage, performance, and concurrency." }

def legalSameDtype : Requirement :=
  { id := ⟨"REQ-ADD-LEGAL-SAME-DTYPE"⟩
    frame := transformationFrameId
    profiles := [profile.id]
    monitored := [tensorCall.id, tensorArguments.id]
    controlled :=
      [returnedValue.id, returnedShape.id, returnedDtype.id, terminalEvent.id]
    assumptions := [pinnedRevision.id, finiteScenarioAssumption.id]
    relation := .all
      [.exactTensorValue, .sameShape, .sameDtype, .sameTerminalOutcome]
    provenance := [pinnedUpstream, scenarioManifest]
    statement := "For the four declared legal same-dtype scenarios, addition has the pinned upstream terminal outcome, exact values, right-aligned broadcast shape, and dtype." }

def dtypePromotion : Requirement :=
  { id := ⟨"REQ-ADD-DTYPE-PROMOTION"⟩
    frame := transformationFrameId
    profiles := [profile.id]
    monitored := [tensorCall.id, tensorArguments.id]
    controlled :=
      [returnedValue.id, returnedShape.id, returnedDtype.id, terminalEvent.id]
    assumptions := [pinnedRevision.id, finiteScenarioAssumption.id]
    relation := .all
      [.exactTensorValue, .sameShape, .sameDtype, .sameTerminalOutcome]
    provenance := [pinnedUpstream, scenarioManifest]
    statement := "For the declared int32 Tensor plus typed zero-dimensional float32 Tensor scenario, addition has the pinned upstream values, shape, promoted dtype, and terminal outcome." }

def incompatibleShape : Requirement :=
  { id := ⟨"REQ-ADD-INCOMPATIBLE-SHAPE"⟩
    frame := transformationFrameId
    profiles := [profile.id]
    monitored := [tensorCall.id, tensorArguments.id]
    controlled := [terminalEvent.id, exceptionStage.id, raisedException.id]
    assumptions := [pinnedRevision.id, finiteScenarioAssumption.id]
    relation := .all
      [.sameTerminalOutcome, .sameExceptionStage, .sameExceptionClass,
       .sameExceptionMessage]
    provenance := [pinnedUpstream, scenarioManifest]
    statement := "For shapes (2,3) and (2,2), the declared construction/add/shape-access trace exposes the same terminal event, exception stage, class, and public message as pinned upstream." }

def realizationIdempotence : Requirement :=
  { id := ⟨"REQ-REALIZE-IDEMPOTENT"⟩
    frame := realizationFrameId
    profiles := [profile.id]
    monitored := [tensorCall.id, tensorArguments.id]
    controlled :=
      [returnedObjectIdentity.id, repeatedReadback.id, postCallInputs.id]
    assumptions := [pinnedRevision.id, finiteScenarioAssumption.id]
    relation := .all
      [.realizeReturnsSelf, .repeatedReadbackStable, .inputsUnchanged]
    provenance := [pinnedUpstream, scenarioManifest]
    statement := "For every declared legal scenario, after input snapshots and add, two ordered realize/readback cycles return the original result Tensor, produce stable values/errors, and leave both Tensor inputs unchanged." }

def transformationFrame : ProblemFrame :=
  { id := transformationFrameId
    kind := .transformation
    domains := [pythonProgram.id, upstreamContract.id, tensorValues.id]
    sharedPhenomena :=
      [tensorCall.id, tensorArguments.id, returnedValue.id, returnedShape.id,
       returnedDtype.id, terminalEvent.id, exceptionStage.id,
       raisedException.id]
    requirementIds := [legalSameDtype.id, dtypePromotion.id, incompatibleShape.id]
    description := "Interpret the finite addition value, broadcast, promotion, and incompatible-shape boundary." }

def realizationFrame : ProblemFrame :=
  { id := realizationFrameId
    kind := .commandedBehavior
    domains := [pythonProgram.id, tensorValues.id, storageWorld.id]
    sharedPhenomena :=
      [tensorCall.id, tensorArguments.id, returnedObjectIdentity.id,
       repeatedReadback.id, postCallInputs.id]
    requirementIds := [realizationIdempotence.id]
    description := "Preserve the finite scenarios' public realization identity and effect observations." }

def catalog : RequirementCatalog :=
  { domains := [pythonProgram, upstreamContract, tensorValues, storageWorld]
    phenomena :=
      [tensorCall, tensorArguments, returnedValue, returnedShape, returnedDtype,
       raisedException, terminalEvent, exceptionStage, returnedObjectIdentity,
       repeatedReadback, postCallInputs]
    assumptions := [pinnedRevision, finiteScenarioAssumption]
    profiles := [profile]
    requirements :=
      [legalSameDtype, dtypePromotion, incompatibleShape, realizationIdempotence]
    frames := [transformationFrame, realizationFrame]
    obstacles := [] }

/-- Internal consistency of the authored requirement interpretation only. -/
theorem catalog_is_structurally_well_formed : catalog.wellFormed := by
  native_decide

theorem requirements_are_distinct :
    (catalog.requirements.map (·.id)).Nodup := by native_decide

inductive LegacyDisposition where
  | retainedHistorical
  | supersessionCandidate
  | superseded
  deriving DecidableEq, BEq, Repr, Inhabited

/-- An interpretation decision, not a migration or coverage claim. -/
structure RequirementRefinementCandidate where
  legacy : RequirementId
  proposedRefinements : List RequirementId
  disposition : LegacyDisposition
  rationale : String
  deriving DecidableEq, BEq, Repr, Inhabited

def legacyBroadcastDisposition : RequirementRefinementCandidate :=
  { legacy := Pilot.broadcastAdd.id
    proposedRefinements := catalog.requirements.map (·.id)
    disposition := .retainedHistorical
    rationale := "Historical evidence and derived state still refer to the bf16-only row. The four prospective requirements are evaluated separately until an explicit migration decision is reviewed and promoted." }

theorem legacy_requirement_is_not_yet_superseded :
    legacyBroadcastDisposition.disposition = .retainedHistorical := by rfl

end Tgrad.Requirements.BroadcastAddPilot

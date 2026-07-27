import Tgrad.Growth.Derived
import Tgrad.Growth.BroadcastAddManifestGenerated
import Tgrad.Requirements.BroadcastAddPilot
import Tgrad.Specification.BroadcastAddPilot

/-! # Frozen prospective work packet for broadcast addition

This packet describes work *on* the codebase: build and calibrate an observer,
then derive whether product work is needed.  It does not prescribe a product
patch and does not claim any requirement is satisfied.
-/

namespace Tgrad.Growth.BroadcastAddPacket

open Tgrad.Requirements
open Tgrad.Requirements.Pilot
open Tgrad.Requirements.BroadcastAddPilot
open Tgrad.Growth
open Tgrad.Growth.BroadcastAddManifest
open Tgrad.Growth.BroadcastAddManifestGenerated

inductive PacketResource where
  | pythonAuthoring
  | leanSpecification
  | cpuVerification
  | leanBuild
  | metalGpu
  deriving DecidableEq, BEq, Repr, Inhabited

structure ObservationOutcomeEnvelope where
  requirement : RequirementId
  before : ObservationState
  allowedAfter : List ObservationState
  deriving DecidableEq, BEq, Repr, Inhabited

def ObservationOutcomeEnvelope.wellFormed
    (delta : ObservationOutcomeEnvelope) : Bool :=
  delta.requirement.valid &&
  delta.before == .unobserved &&
  !delta.allowedAfter.isEmpty &&
  delta.allowedAfter.Nodup

structure ProspectivePacket where
  id : String
  baselineRevision : String
  manifestPath : String
  manifestHash : String
  requirements : List RequirementId
  scenarios : List String
  mutations : List MutationObligation
  candidateComponents : List String
  definitionWriteSet : List String
  observerWriteSet : List String
  definitionResources : List PacketResource
  observerResources : List PacketResource
  serialVerification : Bool
  definitionExpectedObservationChanges : List RequirementId
  observationOutcomeEnvelope : List ObservationOutcomeEnvelope
  mustRemainUnchanged : List RequirementId
  forbiddenInferences : List ForbiddenInference
  closure : String
  recovery : String
  deriving Repr, Inhabited

def ProspectivePacket.wellFormed (packet : ProspectivePacket) : Bool :=
  !packet.id.trimAscii.isEmpty &&
  !packet.baselineRevision.trimAscii.isEmpty &&
  !packet.manifestPath.trimAscii.isEmpty &&
  !packet.manifestHash.trimAscii.isEmpty &&
  !packet.requirements.isEmpty && packet.requirements.Nodup &&
  !packet.scenarios.isEmpty && packet.scenarios.Nodup &&
  !packet.mutations.isEmpty && packet.mutations.all MutationObligation.wellFormed &&
  (packet.mutations.map (·.id)).Nodup &&
  !packet.candidateComponents.isEmpty &&
  !packet.definitionWriteSet.isEmpty && packet.definitionWriteSet.Nodup &&
  !packet.observerWriteSet.isEmpty && packet.observerWriteSet.Nodup &&
  !packet.definitionResources.isEmpty && packet.definitionResources.Nodup &&
  !packet.observerResources.isEmpty && packet.observerResources.Nodup &&
  packet.serialVerification &&
  packet.definitionExpectedObservationChanges.isEmpty &&
  packet.observationOutcomeEnvelope.length == packet.requirements.length &&
  packet.observationOutcomeEnvelope.all ObservationOutcomeEnvelope.wellFormed &&
  packet.observationOutcomeEnvelope.map (·.requirement) == packet.requirements &&
  packet.mustRemainUnchanged.Nodup &&
  !packet.forbiddenInferences.isEmpty && packet.forbiddenInferences.Nodup &&
  !packet.closure.trimAscii.isEmpty &&
  !packet.recovery.trimAscii.isEmpty

def packet : ProspectivePacket :=
  { id := manifest.packetId
    baselineRevision := manifest.baselineRevision
    manifestPath := manifest.path
    manifestHash := manifest.contentHash
    requirements := manifest.requirementIds
    scenarios := manifest.scenarios.map (·.id)
    mutations := manifest.mutations
    candidateComponents :=
      ["world requirement interpretation", "Python-boundary specification",
       "pinned-upstream/Tgrad observer", "mutation calibration harness"]
    definitionWriteSet :=
      ["Tgrad/Requirements/BroadcastAddPilot.lean",
       "Tgrad/Requirements/Relation.lean",
       "Tgrad/Specification/BroadcastAddPilot.lean",
       "Tgrad/Growth/BroadcastAddManifest.lean",
       "Tgrad/Growth/BroadcastAddManifestGenerated.lean",
       "Tgrad/Growth/BroadcastAddPacket.lean",
       "fixtures/requirements/broadcast_add_prospective_v1.json",
       "scripts/spec/generate_broadcast_add_manifest.py",
       "scripts/spec/test_broadcast_add_manifest.py", "TgradSpec.lean",
       "scripts/devcheck.sh", "docs/requirement_engineering.md",
       "docs/plan_2026-07-27.md"]
    observerWriteSet :=
      ["scripts/spec/observe_broadcast_add.py",
       "fixtures/requirements/broadcast_add_<subject>.json",
       "Tgrad/Evidence/BroadcastAddGenerated.lean"]
    definitionResources :=
      [.pythonAuthoring, .leanSpecification, .cpuVerification, .leanBuild]
    observerResources := [.pythonAuthoring, .cpuVerification, .metalGpu]
    serialVerification := true
    definitionExpectedObservationChanges := []
    observationOutcomeEnvelope :=
      [{ requirement := legalSameDtype.id
         before := manifest.expectedBefore
         allowedAfter := manifest.allowedAfter },
       { requirement := dtypePromotion.id
         before := manifest.expectedBefore
         allowedAfter := manifest.allowedAfter },
       { requirement := incompatibleShape.id
         before := manifest.expectedBefore
         allowedAfter := manifest.allowedAfter },
       { requirement := realizationIdempotence.id
         before := manifest.expectedBefore
         allowedAfter := manifest.allowedAfter }]
    mustRemainUnchanged :=
      [Pilot.broadcastAdd.id, importHelpers.id, viewReadbackLifetime.id]
    forbiddenInferences := manifest.forbiddenInferences
    closure := "All six scenarios run against the pinned upstream and Tgrad boundaries; all eight atomic mutants are rejected and localized; generated evidence and derived state reproduce without changing this packet or derivation source."
    recovery := "Reject the observer or candidate result, retain explicit unknown/failed states, and start a new frozen baseline if requirement, scenario, mutation, or derivation semantics must change." }

theorem packet_is_well_formed : packet.wellFormed := by native_decide

theorem packet_has_six_scenarios_and_eight_atomic_mutations :
    packet.scenarios.length = 6 ∧ packet.mutations.length = 8 := by
  native_decide

theorem manifest_requirements_bind_interpreted_requirements :
    manifest.requirementIds =
      [legalSameDtype.id, dtypePromotion.id, incompatibleShape.id,
       realizationIdempotence.id] := by
  native_decide

theorem authored_unchanged_set_is_internally_consistent :
    packet.mustRemainUnchanged =
      [Pilot.broadcastAdd.id, importHelpers.id, viewReadbackLifetime.id] := by
  rfl

theorem packet_does_not_predetermine_a_green_result :
    packet.observationOutcomeEnvelope.all (fun delta =>
      delta.allowedAfter.contains .failed &&
      delta.allowedAfter.contains .passedCalibrated &&
      delta.allowedAfter.contains .blocked &&
      delta.allowedAfter.contains .verifierError) := by
  native_decide

theorem definition_stage_predicts_no_observation_change :
    packet.definitionExpectedObservationChanges = [] := by rfl

theorem open_adequacy_cannot_be_derived_conformant
    (context : Tgrad.Evidence.PromotionContext) (state : ObservationState) :
    derivePromotion context .open state != .conformant := by
  unfold derivePromotion
  split
  · decide
  · rw [show (.open != AdequacyState.accepted) = true by native_decide]
    rfl

end Tgrad.Growth.BroadcastAddPacket

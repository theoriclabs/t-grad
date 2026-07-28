import Tgrad.Growth.PilotState

/-! # Tgrad.Growth.Work — work compiled from the pilot's current gaps

At commit `0031066` an earlier version emitted `WORK-PY-COMPAT-HELPERS` from
the observed helper failure and its two downstream blockages.  The post-change
result exposed a non-generic reference to failure-only evidence, so closure
handling was generalized afterward.  The current version now yields no helper
packet for the calibrated passing state, without an authored status-cell edit.
This retrospective cycle is not evidence of derivation stability; the next
prospective cycle must freeze this module first.  No authored `goalDistance`
participates in selection or closure.
-/

namespace Tgrad.Growth.Work

open Tgrad.Requirements
open Tgrad.Requirements.Pilot
open Tgrad.Evidence
open Tgrad.Growth
open Tgrad.Growth.PilotState

inductive AuthoringResource where
  | python
  | leanSpecification
  deriving DecidableEq, BEq, Repr, Inhabited

inductive VerificationResource where
  | cpu
  | leanBuild
  | metalGpu
  deriving DecidableEq, BEq, Repr, Inhabited

structure PriorityVector where
  requirementsUnblocked : Nat
  currentFailure : Bool
  missingImplementation : Bool
  cpuOnly : Bool
  deriving DecidableEq, BEq, Repr, Inhabited

structure ExpectedTransition where
  requirement : RequirementId
  before : ObservationState
  after : ObservationState
  deriving DecidableEq, BEq, Repr, Inhabited

structure WorkPacket where
  id : String
  closes : List String
  primaryRequirement : RequirementId
  requirementsUnblocked : List RequirementId
  problemFrame : FrameId
  candidateComponents : List String
  oracle : ObservationRelation
  authoringResources : List AuthoringResource
  verificationResources : List VerificationResource
  expectedTransitions : List ExpectedTransition
  priority : PriorityVector
  recovery : String
  deriving Repr, Inhabited

def WorkPacket.wellFormed (packet : WorkPacket) : Bool :=
  !packet.id.trimAscii.isEmpty &&
  !packet.closes.isEmpty &&
  packet.closes.all (fun gap => !gap.trimAscii.isEmpty) &&
  packet.primaryRequirement.valid &&
  !packet.requirementsUnblocked.isEmpty &&
  packet.requirementsUnblocked.all RequirementId.valid &&
  packet.problemFrame.valid &&
  !packet.candidateComponents.isEmpty &&
  packet.candidateComponents.all (fun component => !component.trimAscii.isEmpty) &&
  packet.oracle.wellFormed &&
  !packet.authoringResources.isEmpty &&
  !packet.verificationResources.isEmpty &&
  !packet.expectedTransitions.isEmpty &&
  !packet.recovery.trimAscii.isEmpty

def helperClosingGapsFor (helpers : RequirementState) : List Gap :=
  (gapsFor helpers).filter (fun gap =>
    gap.kind == .implementation || gap.kind == .failedBehavior)

def helperBlockedRequirementsFor (ctx : PromotionContext) : List RequirementId :=
  ((Tgrad.Evidence.PilotGenerated.blockages.filter (fun blockage =>
      blockage.sourceObservationId ==
        Tgrad.Evidence.PilotGenerated.helperObservation.id &&
      blockage.currentIn ctx)).flatMap (·.blocks)).eraseDups

def deriveHelperSurfaceWorkFor
    (helpers : RequirementState) (ctx : PromotionContext) : Option WorkPacket :=
  let blocked := helperBlockedRequirementsFor ctx
  let closing := helperClosingGapsFor helpers
  if helpers.implementation == .noCandidate &&
     helpers.observation == .failed &&
     !closing.isEmpty &&
     !blocked.isEmpty then
    some
      { id := "WORK-PY-COMPAT-HELPERS"
        closes := closing.map (·.id)
        primaryRequirement := importHelpers.id
        requirementsUnblocked := blocked
        problemFrame := importHelpers.frame
        candidateComponents := ["Python substitution boundary"]
        oracle := importHelpers.relation
        authoringResources := [.python]
        verificationResources := [.cpu, .leanBuild]
        expectedTransitions :=
          [{ requirement := importHelpers.id, before := .failed, after := .passedCalibrated },
           { requirement := broadcastAdd.id, before := .blocked, after := .unobserved },
           { requirement := viewReadbackLifetime.id, before := .blocked, after := .unobserved }]
        priority :=
          { requirementsUnblocked := blocked.length
            currentFailure := true
            missingImplementation := true
            cpuOnly := true }
        recovery := "Revert the compatibility-surface candidate and regenerate the exact baseline observation." }
  else none

def deriveHelperSurfaceWork : Option WorkPacket :=
  deriveHelperSurfaceWorkFor helpersState context

def frontierFor (helpers : RequirementState) (ctx : PromotionContext) : List WorkPacket :=
  (deriveHelperSurfaceWorkFor helpers ctx).toList

def frontier : List WorkPacket := frontierFor helpersState context

theorem current_helper_state_does_not_emit_the_failure_packet :
    deriveHelperSurfaceWork = none := by
  native_decide

theorem current_frontier_is_empty_for_the_single_packet_schema :
    frontier = [] := by
  native_decide

private def quote (value : String) : String :=
  "\"" ++ (value.replace "\\" "\\\\").replace "\"" "\\\"" ++ "\""

def workJsonFor (helpers : RequirementState) (ctx : PromotionContext) : String :=
  let packets := (frontierFor helpers ctx).map fun packet =>
    "    {\n" ++
    s!"      \"id\": {quote packet.id},\n" ++
    s!"      \"primary_requirement\": {quote packet.primaryRequirement.value},\n" ++
    s!"      \"requirements_unblocked\": {packet.priority.requirementsUnblocked},\n" ++
    "      \"verification\": [\"cpu\", \"lean_build\"]\n" ++
    "    }"
  "[\n" ++ String.intercalate ",\n" packets ++ "\n  ]"

def workJson : String := workJsonFor helpersState context

end Tgrad.Growth.Work

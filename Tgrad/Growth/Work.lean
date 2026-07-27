import Tgrad.Growth.PilotState

/-! # Tgrad.Growth.Work — work compiled from the pilot's current gaps

The work packet below exists only when the observed helper failure, absent
implementation candidate, and active downstream blockage all exist together.
No authored `goalDistance` participates in selection.
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

def helperClosingGaps : List Gap :=
  (gapsFor helpersState).filter (fun gap =>
    gap.kind == .implementation || gap.kind == .failedBehavior)

def deriveHelperSurfaceWork : Option WorkPacket :=
  let blocked := Tgrad.Evidence.PilotGenerated.helpersBlockage.blocks
  if helpersState.implementation == .noCandidate &&
     helpersState.observation == .failed &&
     !helperClosingGaps.isEmpty &&
     !blocked.isEmpty then
    some
      { id := "WORK-PY-COMPAT-HELPERS"
        closes := helperClosingGaps.map (·.id)
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

def frontier : List WorkPacket := deriveHelperSurfaceWork.toList

theorem observed_state_derives_the_helper_surface_packet :
    deriveHelperSurfaceWork.isSome = true := by
  native_decide

theorem derived_packet_is_well_formed_and_unblocks_two_requirements :
    frontier.all WorkPacket.wellFormed = true ∧
    frontier.map (fun packet => packet.priority.requirementsUnblocked) = [2] := by
  native_decide

private def quote (value : String) : String :=
  "\"" ++ (value.replace "\\" "\\\\").replace "\"" "\\\"" ++ "\""

def workJson : String :=
  let packets := frontier.map fun packet =>
    "    {\n" ++
    s!"      \"id\": {quote packet.id},\n" ++
    s!"      \"primary_requirement\": {quote packet.primaryRequirement.value},\n" ++
    s!"      \"requirements_unblocked\": {packet.priority.requirementsUnblocked},\n" ++
    "      \"verification\": [\"cpu\", \"lean_build\"]\n" ++
    "    }"
  "[\n" ++ String.intercalate ",\n" packets ++ "\n  ]"

end Tgrad.Growth.Work

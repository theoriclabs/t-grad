import Tgrad.Growth.BroadcastAddObservationV3

/-! # Prospective realization-identity candidate -/

namespace Tgrad.Growth.BroadcastAddRealizeCandidateV1

def packetId : String := "WORK-PY-REALIZE-IDENTITY-V1"
def baselineRevision : String := "0eda71fd99cabfde6538197e15bcacc8980e7f27"
def triggerEvidenceId : String := BroadcastAddObservationV3.evidenceId
def productWriteSet : List String := ["python/tgrad.py"]
def expectedFullTraceScenarios : List String :=
  BroadcastAddObservationV3.realizationBlockedScenarios
def allocationAllowed : Bool := false
def dispatchAllowed : Bool := false
def parityPromotionPredicted : Bool := false

theorem exactly_two_scenarios_are_targeted : expectedFullTraceScenarios.length = 2 := by
  decide

theorem realization_is_observational_identity_only :
    !allocationAllowed && !dispatchAllowed && !parityPromotionPredicted := by
  decide

end Tgrad.Growth.BroadcastAddRealizeCandidateV1

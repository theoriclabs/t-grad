import Tgrad.Growth.BroadcastAddObservationV5

/-! # Recurring-verification amendment for ranked broadcasting

The product packet created a focused test but did not name the recurring
development runner that must invoke it. This is a verifier-work correction,
not a retrospective expansion of the product write set.
-/

namespace Tgrad.Growth.BroadcastAddRankedVerificationAmendmentV1

def packetId : String := "WORK-EW-RANKED-BROADCAST-VERIFY-V1"
def baselineRevision : String := BroadcastAddObservationV5.subjectRevision
def triggerEvidenceId : String := BroadcastAddObservationV5.evidenceId
def productWriteSet : List String := []
def verifierWriteSet : List String :=
  ["scripts/spec/test_ranked_broadcast.py", "scripts/devcheck.sh"]
def predictedRecurringCommand : String :=
  "python -m unittest scripts.spec.test_ranked_broadcast"
def changesConformanceState : Bool := false

theorem verifier_only_amendment :
    productWriteSet.isEmpty = true ∧ verifierWriteSet.length = 2 := by
  decide

theorem recurring_registration_is_not_new_evidence :
    changesConformanceState = false := by
  rfl

end Tgrad.Growth.BroadcastAddRankedVerificationAmendmentV1

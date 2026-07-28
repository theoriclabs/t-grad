import Tgrad.Growth.BroadcastAddObservationV4

/-! # Prospective rank-0-through-3 elementwise broadcast candidate -/

namespace Tgrad.Growth.BroadcastAddRankedCandidateV1

def packetId : String := "WORK-EW-RANKED-BROADCAST-V1"
def baselineRevision : String := "2d144bd991d8fa023c0faf05f9c52ee79db7b5e0"
def triggerEvidenceId : String := BroadcastAddObservationV4.evidenceId

def productWriteSet : List String :=
  ["python/tgrad.py", "Tgrad/Schedule/View.lean",
   "Tgrad/Renderer/Elementwise.lean", "Tgrad/PythonFFI.lean"]

def verificationWriteSet : List String := ["scripts/spec/test_ranked_broadcast.py"]
def supportedRanks : List Nat := [0, 1, 2, 3]
def targetScenario : String := "ADD-TWO-SIDED-BROADCAST-F32"
def alreadyMatchingScenarios : List String :=
  BroadcastAddObservationV4.fullyMatchingScenarios
def promotionPredicted : Bool := false

theorem write_set_has_four_product_boundaries : productWriteSet.length = 4 := by
  decide

theorem metal_rank_scope_is_bounded_and_nodup :
    supportedRanks = [0, 1, 2, 3] ∧ supportedRanks.Nodup := by
  decide

theorem one_new_scenario_does_not_imply_promotion : promotionPredicted = false := by
  rfl

end Tgrad.Growth.BroadcastAddRankedCandidateV1

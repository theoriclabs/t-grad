import Tgrad.Growth.BroadcastAddInt32CandidateV1

/-! # Post-int32 calibrated TGrad observation -/

namespace Tgrad.Growth.BroadcastAddObservationV6

def evidenceId : String :=
  "006ceb03875aaf932a6038866e5e3bf1de20f9b621b608129f9fe74866fe5fdd"

def subjectRevision : String :=
  "c4984c8fa81b4f1e74bafb559a0c4bd825dbe165"

def fullyMatchingScenarios : List String :=
  BroadcastAddObservationV5.fullyMatchingScenarios ++
    BroadcastAddInt32CandidateV1.newlyMatchingScenarios

def remainingScenario : String := "ADD-INCOMPATIBLE-SHAPES"
def aggregateComparison : Nat × Nat × Nat := (57, 3, 1)

def promotionBlockers : List String :=
  ["incompatible-shape exception relation still differs",
   "backing-storage width is not directly observed",
   "precision-stress behavior is not mutation-calibrated in the observer",
   "source-to-runtime-binary provenance remains open"]

def prospectivePredictionHeld : Bool := true
def promotionAllowed : Bool := false

theorem exactly_five_scenarios_match_every_applicable_dimension :
    fullyMatchingScenarios.length = 5 ∧ fullyMatchingScenarios.Nodup := by
  decide

theorem aggregate_matches_the_prospective_packet :
    aggregateComparison = BroadcastAddInt32CandidateV1.aggregateAfter := by
  rfl

theorem successful_causal_prediction_is_not_promotion :
    prospectivePredictionHeld = true ∧ promotionAllowed = false ∧
      promotionBlockers.length = 4 := by
  decide

end Tgrad.Growth.BroadcastAddObservationV6

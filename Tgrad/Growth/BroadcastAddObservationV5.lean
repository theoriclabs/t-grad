import Tgrad.Growth.BroadcastAddRankedCandidateV1

/-! # Post-ranked-broadcast calibrated TGrad observation

This module records facts read from immutable evidence. It deliberately does
not promote either broadcast-add adequacy or general tinygrad parity.
-/

namespace Tgrad.Growth.BroadcastAddObservationV5

def evidenceId : String :=
  "23d0daf8a3a75f29d8deecb52665e5353a6531ad4cfdf3fe76d3e31556ff67bf"

def subjectRevision : String :=
  "8016524c73cb78681d8d784c79cdfb487b47c6e8"

def fullyMatchingScenarios : List String :=
  BroadcastAddObservationV4.fullyMatchingScenarios ++
    [BroadcastAddRankedCandidateV1.targetScenario]

def unchangedIncompleteScenarios : List (String × Nat × Nat × Nat) :=
  [("ADD-RANK-EXTENSION-I32", 0, 5, 6),
   ("ADD-I32-F32-SCALAR-PROMOTION", 0, 5, 6),
   ("ADD-INCOMPATIBLE-SHAPES", 2, 3, 1)]

def dimensionsPerMatchingScenario : Nat := 11
def prospectivePredictionHeld : Bool := true
def promotionAllowed : Bool := false

theorem exactly_three_scenarios_match_all_dimensions :
    fullyMatchingScenarios.length = 3 ∧ dimensionsPerMatchingScenario = 11 := by
  decide

theorem excluded_fronts_remained_incomplete :
    unchangedIncompleteScenarios.length = 3 := by
  decide

theorem successful_prediction_is_not_requirement_promotion :
    prospectivePredictionHeld = true ∧ promotionAllowed = false := by
  decide

end Tgrad.Growth.BroadcastAddObservationV5

import Tgrad.Growth.BroadcastAddObservationV3

/-! # First fully matching broadcast-add scenarios -/

namespace Tgrad.Growth.BroadcastAddObservationV4

def evidenceId : String :=
  "bfbde2d1477526daaf84b1b3aa06b8996fe13396d8e36ba3d87899c21718dcc9"

def fullyMatchingScenarios : List String :=
  ["ADD-SAME-SHAPE-F32", "ADD-SINGLETON-AXIS-F32"]

def dimensionsPerMatchingScenario : Nat := 11

def remainingScenarioFronts : List (String × String) :=
  [("ADD-TWO-SIDED-BROADCAST-F32", "generalized right-aligned broadcasting"),
   ("ADD-RANK-EXTENSION-I32", "int32 construction and storage"),
   ("ADD-I32-F32-SCALAR-PROMOTION", "int32 plus float32 promotion"),
   ("ADD-INCOMPATIBLE-SHAPES", "exception class and message relation")]

def promotionAllowed : Bool := false

theorem exactly_two_scenarios_match_all_dimensions :
    fullyMatchingScenarios.length = 2 ∧ dimensionsPerMatchingScenario = 11 := by
  decide

theorem four_scenario_fronts_remain : remainingScenarioFronts.length = 4 := by
  decide

theorem partial_conformance_is_not_promotion : promotionAllowed = false := by rfl

end Tgrad.Growth.BroadcastAddObservationV4

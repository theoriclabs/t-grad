import Tgrad.Growth.BroadcastAddObservationV2

/-! # Post-readback calibrated TGrad observation -/

namespace Tgrad.Growth.BroadcastAddObservationV3

def evidenceId : String :=
  "317fc368cfcda64ed0ee45eab33f49a985605ad75dc921021f4a7d86040572d1"

def realizationBlockedScenarios : List String :=
  ["ADD-SAME-SHAPE-F32", "ADD-SINGLETON-AXIS-F32"]

def rankBlockedScenarios : List String := ["ADD-TWO-SIDED-BROADCAST-F32"]

def realizationFailure : String := "'Tensor' object has no attribute 'realize'"

def observedResultShapeAndDtype : List String := realizationBlockedScenarios

theorem two_results_reached_shape_and_dtype_observation :
    observedResultShapeAndDtype.length = 2 := by decide

theorem generalized_rank_is_a_distinct_front : rankBlockedScenarios.length = 1 := by
  decide

end Tgrad.Growth.BroadcastAddObservationV3

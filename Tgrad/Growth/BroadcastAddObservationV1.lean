/-! # First calibrated TGrad broadcast-add observation

This module records the interpreted result of the first TGrad run against the
calibrated upstream broadcast-add baseline.  It is an evidence projection, not
a conformance theorem: every scenario stopped at public Tensor construction,
so arithmetic, broadcasting, promotion, realization, and readback remain
unobserved.
-/

namespace Tgrad.Growth.BroadcastAddObservationV1

inductive ScenarioClass where
  | legal
  | incompatible
  deriving DecidableEq, BEq, Repr, Inhabited

structure ScenarioResult where
  id : String
  scenarioClass : ScenarioClass
  terminalStage : String
  exceptionClass : String
  exceptionMessage : String
  constructionObserved : Bool
  leftConstructed : Bool
  rightConstructed : Bool
  downstreamDimensionsUnobserved : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

def upstreamEvidenceId : String :=
  "84a58222575eab06ecc72889e1dbbe2a2084849356673a8d299c21ad2e41a844"

def tgradEvidenceId : String :=
  "a4fcdc9090a20ce6c01283d3795bb55e099a3fcd45e29f608f2e128182022bba"

def constructorError : String :=
  "Tensor.__init__() missing 2 required positional arguments: 'size' and 'shape'"

def legalScenarioIds : List String :=
  ["ADD-SAME-SHAPE-F32", "ADD-SINGLETON-AXIS-F32",
   "ADD-RANK-EXTENSION-I32", "ADD-TWO-SIDED-BROADCAST-F32",
   "ADD-I32-F32-SCALAR-PROMOTION"]

def incompatibleScenarioId : String := "ADD-INCOMPATIBLE-SHAPES"

def resultFor (id : String) (scenarioClass : ScenarioClass) : ScenarioResult :=
  { id
    scenarioClass
    terminalStage := "construct_left"
    exceptionClass := "TypeError"
    exceptionMessage := constructorError
    constructionObserved := true
    leftConstructed := false
    rightConstructed := false
    downstreamDimensionsUnobserved := if scenarioClass == .legal then 6 else 1 }

def results : List ScenarioResult :=
  legalScenarioIds.map (resultFor · .legal) ++
    [resultFor incompatibleScenarioId .incompatible]

def productFinding : String :=
  "The strict tinygrad substitution aliases tinygrad.Tensor to tgrad.Tensor, but Tgrad exposes only its internal buffer constructor rather than tinygrad's public Tensor(data, dtype=...) constructor."

def forbiddenInference : String :=
  "No arithmetic, broadcast, promotion, realization, readback, or incompatibility-error conformance follows from this observation."

theorem six_scenarios_were_observed : results.length = 6 := by native_decide

theorem every_scenario_stopped_at_public_construction :
    results.all (fun result =>
      result.terminalStage == "construct_left" &&
      result.constructionObserved &&
      !result.leftConstructed && !result.rightConstructed) := by
  native_decide

theorem legal_semantics_remain_unobserved :
    (results.filter (fun result => result.scenarioClass == .legal)).all
      (fun result => result.downstreamDimensionsUnobserved == 6) := by
  native_decide

end Tgrad.Growth.BroadcastAddObservationV1

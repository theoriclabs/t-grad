import Tgrad.Spec.Parity

/-! # Tgrad.Spec.ParityCalibration — when an upstream suite is an oracle

Running tinygrad's tests against tinygrad is necessary but not sufficient.
This module separates a complete, diagnosable run from a repeatability
comparison and from promotion.  In particular, a partial or merely
aggregate-compatible run cannot become oracle evidence. -/

namespace Tgrad.Spec.Parity

inductive SuiteFileStatus where
  | pass
  | passWithSkips
  | fail
  | collectError
  | timeout
  | empty
  | runnerError
  deriving DecidableEq, BEq, Repr, Inhabited

structure SuiteFileObservation where
  path : String
  status : SuiteFileStatus
  collected : Nat
  passed : Nat
  failed : Nat
  errors : Nat
  skipped : Nat
  semanticHash : String
  stdoutHash : String
  stderrHash : String
  junitHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

def SuiteFileObservation.accountedFor (observation : SuiteFileObservation) : Bool :=
  observation.collected == observation.passed + observation.failed +
    observation.errors + observation.skipped

def SuiteFileObservation.diagnosable (observation : SuiteFileObservation) : Bool :=
  !observation.path.isEmpty &&
  !observation.semanticHash.isEmpty &&
  !observation.stdoutHash.isEmpty &&
  !observation.stderrHash.isEmpty &&
  observation.accountedFor

structure SuiteCalibrationRun where
  upstreamRevision : String
  upstreamTree : String
  targetManifestHash : String
  environmentManifestHash : String
  scenarioHash : String
  inventoryCount : Nat
  complete : Bool
  outcomeHash : String
  diagnosticManifestHash : String
  files : List SuiteFileObservation
  deriving Repr, Inhabited

def SuiteCalibrationRun.wellFormed (run : SuiteCalibrationRun) : Bool :=
  !run.upstreamRevision.isEmpty &&
  !run.upstreamTree.isEmpty &&
  !run.targetManifestHash.isEmpty &&
  !run.environmentManifestHash.isEmpty &&
  !run.scenarioHash.isEmpty &&
  !run.outcomeHash.isEmpty &&
  !run.diagnosticManifestHash.isEmpty &&
  run.files.length == run.inventoryCount &&
  (run.files.map (·.path)).eraseDups.length == run.files.length &&
  run.files.all SuiteFileObservation.diagnosable

def SuiteCalibrationRun.sameQuestion
    (left right : SuiteCalibrationRun) : Bool :=
  left.upstreamRevision == right.upstreamRevision &&
  left.upstreamTree == right.upstreamTree &&
  left.targetManifestHash == right.targetManifestHash &&
  left.environmentManifestHash == right.environmentManifestHash &&
  left.scenarioHash == right.scenarioHash &&
  left.inventoryCount == right.inventoryCount

structure SuiteCalibrationComparison where
  first : SuiteCalibrationRun
  second : SuiteCalibrationRun
  deriving Repr, Inhabited

def SuiteCalibrationComparison.repeatable
    (comparison : SuiteCalibrationComparison) : Bool :=
  comparison.first.wellFormed &&
  comparison.second.wellFormed &&
  comparison.first.complete &&
  comparison.second.complete &&
  comparison.first.sameQuestion comparison.second &&
  comparison.first.outcomeHash == comparison.second.outcomeHash

/-- Promotion is stricter than repeatability.  Stable failures and stable
skips are useful calibration facts, but they are not a green oracle. -/
def SuiteCalibrationComparison.canPromote
    (comparison : SuiteCalibrationComparison) : Bool :=
  comparison.repeatable &&
  comparison.first.files.all fun observation => observation.status == .pass &&
  comparison.second.files.all fun observation => observation.status == .pass

theorem empty_comparison_cannot_promote :
    SuiteCalibrationComparison.canPromote {} = false := by
  native_decide

end Tgrad.Spec.Parity

import Tgrad.Spec.Parity
import Tgrad.PythonFFI

/-! # Tgrad.Spec.PerformanceBoundary — comparable work before ratios

Performance evidence is meaningful only after both operands name the same
kind of work.  This module records the canonical prepared-runtime boundary;
it does not contain a threshold or a measured result. -/

namespace Tgrad.Spec

#check Tgrad.PythonFFI.matmulPrepare
#check Tgrad.PythonFFI.matmulRunPrepared
#check Tgrad.PythonFFI.matmulPlanRelease

inductive RuntimeBoundaryComponent where
  | fixedShape
  | compiledExecutionState
  | deviceResidentInputs
  | reusableOutput
  | hostInvocation
  | runtimeLookupOrReplay
  | commandSubmission
  | deviceCompletion
  deriving DecidableEq, BEq, Repr, Inhabited

structure PreparedRuntimeBoundary where
  id : String
  preparationExcludes : List String
  timedComponents : List RuntimeBoundaryComponent
  synchronized : Bool
  outputReused : Bool
  deriving DecidableEq, BEq, Repr, Inhabited

def PreparedRuntimeBoundary.wellFormed
    (boundary : PreparedRuntimeBoundary) : Bool :=
  !boundary.id.isEmpty &&
  !boundary.preparationExcludes.isEmpty &&
  boundary.timedComponents.eraseDups.length == boundary.timedComponents.length &&
  boundary.timedComponents.contains .fixedShape &&
  boundary.timedComponents.contains .compiledExecutionState &&
  boundary.timedComponents.contains .deviceResidentInputs &&
  boundary.timedComponents.contains .reusableOutput &&
  boundary.timedComponents.contains .hostInvocation &&
  boundary.timedComponents.contains .runtimeLookupOrReplay &&
  boundary.timedComponents.contains .commandSubmission &&
  boundary.timedComponents.contains .deviceCompletion &&
  boundary.synchronized && boundary.outputReused

def PreparedRuntimeBoundary.comparable
    (left right : PreparedRuntimeBoundary) : Bool :=
  left.wellFormed && right.wellFormed &&
  left.timedComponents == right.timedComponents &&
  left.synchronized == right.synchronized &&
  left.outputReused == right.outputReused

private def canonicalComponents : List RuntimeBoundaryComponent :=
  [.fixedShape, .compiledExecutionState, .deviceResidentInputs,
   .reusableOutput, .hostInvocation, .runtimeLookupOrReplay,
   .commandSubmission, .deviceCompletion]

def tgradPreparedRuntime : PreparedRuntimeBoundary :=
  { id := "tgrad.PreparedMatmul.run",
    preparationExcludes :=
      ["route selection", "rendering", "compilation", "output allocation"],
    timedComponents := canonicalComponents,
    synchronized := true,
    outputReused := true }

def tinygradPreparedRuntime : PreparedRuntimeBoundary :=
  { id := "tinygrad.TinyJit.replay",
    preparationExcludes :=
      ["graph construction", "capture", "lowering", "compilation", "output allocation"],
    timedComponents := canonicalComponents,
    synchronized := true,
    outputReused := true }

theorem canonical_prepared_boundaries_are_comparable :
    tgradPreparedRuntime.comparable tinygradPreparedRuntime = true := by
  native_decide

end Tgrad.Spec

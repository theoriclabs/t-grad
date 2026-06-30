import Tgrad.Model.Spec
import Tgrad.Model.Factory

/-! # Tgrad.Model.Coverage — ground the factory roadmap against the spec frontier

  The consolidating link between the two independently-built halves of the
  model: the **grounded** capability lattice (`Spec.frontier`, whose op/dtype
  axes are tied to the real `UOp`/`Dtype` via `classify`/`dtypeClassOf`) and
  the **hand-authored** factory capability map (`Factory.fullReplacementCapabilityMap`).

  It answers one cross-check question: *does the factory's capability map
  cover every capability the grounded spec says is below target?*

  It is honest about the answer rather than asserting coverage. The
  op-ontology axis (elementwise / reduction / memory / loop-structure growth)
  is NOT covered by the current factory map — the same gap `Model.Plan`
  found on the S1 side (`op_axis_has_no_change_subject`), now confirmed
  independently from the factory side.

  `coveredAreas` is read from the REAL `fullReplacementCapabilityMap`, so if
  the factory map gains/loses an entry this cross-check updates; and
  `capabilityArea` is total over the REAL `Capability`, so a new spec axis
  forces a coverage decision. The check is grounded on both sides.
-/

namespace Tgrad
namespace Model

/-- Map a grounded spec `Capability` to the factory `CapabilityArea` it
    belongs to. TOTAL over `Capability` (no wildcard): adding a spec axis
    forces a coverage decision here. -/
def capabilityArea : Capability → CapabilityArea
  | .op .leafIO        => .authoringApi
  | .op .movementView  => .shapeViewSemantics
  | .op .memoryAccess  => .codegen
  | .op .aluBinary     => .opOntology
  | .op .aluConvert    => .opOntology
  | .op .reduction     => .opOntology
  | .op .loopStructure => .codegen
  | .op .tensorCore    => .codegen
  | .dtype _           => .dtypeSystem
  | .backend _         => .backend
  | .exec .singleKernel     => .inference
  | .exec .multiKernelInfer => .inference
  | .exec .training         => .training
  | .sched _           => .scheduler

/-- Areas the factory capability map actually covers — read from the real
    `fullReplacementCapabilityMap`, so it tracks Factory edits. -/
def coveredAreas : List CapabilityArea :=
  fullReplacementCapabilityMap.map (fun c => c.area)

def frontierAreaCovered (c : Capability) : Bool :=
  coveredAreas.contains (capabilityArea c)

/-- Frontier capabilities the factory roadmap/map does NOT cover. -/
def uncoveredFrontier : List Capability :=
  frontier.filter (fun c => !frontierAreaCovered c)

def coveredFrontier : List Capability :=
  frontier.filter frontierAreaCovered

def coverageReport : String :=
  let cov := coveredFrontier.map (fun c =>
    "  + " ++ c.toStr ++ "  ->  area:" ++ (capabilityArea c).toStr)
  let unc := uncoveredFrontier.map (fun c =>
    "  - " ++ c.toStr ++ "  ->  area:" ++ (capabilityArea c).toStr ++ "  (NO factory-map coverage)")
  "FRONTIER (" ++ toString frontier.length ++ ") ↔ FACTORY CAPABILITY-MAP COVERAGE\n"
    ++ "covered-areas: " ++ String.intercalate "," (coveredAreas.map CapabilityArea.toStr) ++ "\n\n"
    ++ "covered (" ++ toString coveredFrontier.length ++ "):\n" ++ String.intercalate "\n" cov ++ "\n\n"
    ++ "uncovered gap (" ++ toString uncoveredFrontier.length ++ "):\n" ++ String.intercalate "\n" unc

#eval IO.println coverageReport

/-! ## Cross-check facts -/

/-- The dtype and backend frontier capabilities ARE covered by the factory
    capability map — those axes are consolidated across the two halves. -/
theorem dtype_backend_frontier_covered :
    (frontier.filter (fun c => match c with
      | .dtype _ => true | .backend _ => true | _ => false)).all frontierAreaCovered = true := by
  native_decide

/-- FINDING: the grounded frontier has capabilities the factory map does NOT
    cover. Grounds the hand-authored map against the grounded lattice and
    surfaces the same op-axis gap `Plan.op_axis_has_no_change_subject` found
    on the S1 side — now from the factory side. -/
theorem frontier_coverage_has_gap :
    uncoveredFrontier.isEmpty = false := by native_decide

/-- Concretely: the binary-ALU op capability (general elementwise) is
    uncovered by the factory capability map. -/
theorem alu_op_uncovered :
    frontierAreaCovered (.op .aluBinary) = false := by native_decide

/-- The view/movement op capability IS covered (via `cap.views.general`),
    so the gap is specifically the op-ontology/codegen growth, not all ops. -/
theorem movement_op_covered :
    frontierAreaCovered (.op .movementView) = true := by native_decide

end Model
end Tgrad

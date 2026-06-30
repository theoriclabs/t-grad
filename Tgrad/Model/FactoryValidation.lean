import Tgrad.Model.Factory

/-! # Tgrad.Model.FactoryValidation

  First validation layer for the TGrad factory model.

  These checks validate the intensional Lean control-plane model: registry
  coverage, roadmap dependency shape, evidence expectations, and sample
  admission boundaries. They do not claim operational reality. Filesystem,
  ledger, evidence, and repo-ref validation need separate IO checkers.
-/

namespace Tgrad
namespace Model

inductive ValidationLayer where
  | model
  | operational
  | economic
  deriving BEq, Repr, Inhabited, DecidableEq

def ValidationLayer.toStr : ValidationLayer -> String
  | .model => "model"
  | .operational => "operational"
  | .economic => "economic"

inductive ValidationStatus where
  | pass
  | fail
  | warn
  deriving BEq, Repr, Inhabited, DecidableEq

def ValidationStatus.toStr : ValidationStatus -> String
  | .pass => "pass"
  | .fail => "fail"
  | .warn => "warn"

structure ValidationCheck where
  id : String
  layer : ValidationLayer
  status : ValidationStatus
  statement : String
  repair : String := ""
  deriving Repr, Inhabited

def ValidationCheck.passed (check : ValidationCheck) : Bool :=
  check.status == .pass

def ValidationCheck.line (check : ValidationCheck) : String :=
  check.id ++ " [" ++ check.layer.toStr ++ "; " ++ check.status.toStr ++ "] " ++
    check.statement ++
    (if check.repair == "" then "" else " repair: " ++ check.repair)

private def statusOf (ok : Bool) : ValidationStatus :=
  if ok then .pass else .fail

private def check (id : String) (layer : ValidationLayer) (ok : Bool)
    (statement repair : String) : ValidationCheck := {
  id := id
  layer := layer
  status := statusOf ok
  statement := statement
  repair := if ok then "" else repair
}

private def hasOutputKind (item : RoadmapItem) (kind : ProductArtifactKind) : Bool :=
  item.expectedOutputs.any (fun output => output == kind)

private def roadmapItemShapeOk (item : RoadmapItem) : Bool :=
  item.id != "" &&
  item.title != "" &&
  item.acceptanceBoundary != "" &&
  !item.expectedOutputs.isEmpty

private def roadmapItemPerformanceOutputOk (item : RoadmapItem) : Bool :=
  !item.needsPerformanceEvidence || hasOutputKind item .performanceEvidence

private def roadmapItemRepoTransformOk (item : RoadmapItem) : Bool :=
  item.requiresRepoTransform

def roadmapItemsWithBadShape (roadmap : TgradRoadmap) : List RoadmapItem :=
  roadmap.items.filter (fun item => !roadmapItemShapeOk item)

def roadmapItemsMissingPerformanceOutput (roadmap : TgradRoadmap) :
    List RoadmapItem :=
  roadmap.items.filter (fun item => !roadmapItemPerformanceOutputOk item)

def roadmapItemsWithoutRepoTransform (roadmap : TgradRoadmap) : List RoadmapItem :=
  roadmap.items.filter (fun item => !roadmapItemRepoTransformOk item)

def tgradModelValidationChecks : List ValidationCheck := [
  check "model.factory.registry.partition" .model
    tgradFactory.partitionsRegistry
    "Production-function homes partition the factory registry."
    "Update tgradProcesses so each production function is homed exactly once.",
  check "model.product-design.first-class" .model
    (!productDesignFunctions.isEmpty)
    "Product description/specification are produced by at least one production function."
    "Add or repair TPF1 product framing.",
  check "model.roadmap.dependencies-known" .model
    tgradReplacementRoadmap.dependenciesKnown
    "Every roadmap dependency names another roadmap item."
    "Fix misspelled/missing dependency ids or add the missing roadmap item.",
  check "model.roadmap.has-ready-foundation" .model
    (!readyRoadmapNow.isEmpty)
    "At least one foundation roadmap item is ready with no accepted items."
    "Repair initial roadmap dependencies so a first item can run.",
  check "model.roadmap.foundation-unblocks-core" .model
    ((readyRoadmapAfterFoundation.map (fun item => item.id)).any
      (fun id => id == "rm.dtype-runtime-fp16"))
    "Accepting product spec and capability map unblocks a semantic-core slice."
    "Repair roadmap dependencies after foundation.",
  check "model.roadmap.item-shape" .model
    ((roadmapItemsWithBadShape tgradReplacementRoadmap).isEmpty)
    "Every roadmap item has id, title, acceptance boundary, and expected outputs."
    "Fill missing roadmap fields.",
  check "model.roadmap.performance-output" .model
    ((roadmapItemsMissingPerformanceOutput tgradReplacementRoadmap).isEmpty)
    "Every performance-required roadmap item expects performance evidence."
    "Add performanceEvidence to expected outputs or lower the repo transform requirement.",
  check "model.roadmap.repo-transform" .model
    ((roadmapItemsWithoutRepoTransform tgradReplacementRoadmap).isEmpty)
    "Every current roadmap item requires an accepted repo-ref transformation."
    "Mark non-product/process-only items explicitly or require a repo transform.",
  check "model.workorder.sample-fp16" .model
    sampleFp16WorkOrder.admissible
    "A sample fp16 work order with repo ref is admissible."
    "Repair work-order construction or admission predicate.",
  check "model.workorder.reject-missing-repo" .model
    (!sampleBackendWorkOrderWithoutRepo.admissible)
    "A product-changing backend work order without repo ref is rejected."
    "Require repo refs for product-changing work."
]

def failedValidationChecks (checks : List ValidationCheck) : List ValidationCheck :=
  checks.filter (fun check => check.status == .fail)

def warningValidationChecks (checks : List ValidationCheck) : List ValidationCheck :=
  checks.filter (fun check => check.status == .warn)

def allValidationChecksPass (checks : List ValidationCheck) : Bool :=
  (failedValidationChecks checks).isEmpty

def validationReport (checks : List ValidationCheck) : String :=
  let failures := failedValidationChecks checks
  let warnings := warningValidationChecks checks
  "validation-checks: " ++ toString checks.length ++
  "\nfailures: " ++ toString failures.length ++
  "\nwarnings: " ++ toString warnings.length ++
  "\n" ++ String.intercalate "\n" (checks.map ValidationCheck.line)

def tgradModelValidationReport : String :=
  validationReport tgradModelValidationChecks

theorem tgrad_model_validation_passes :
    allValidationChecksPass tgradModelValidationChecks = true := by
  native_decide

theorem performance_required_items_have_performance_output :
    (roadmapItemsMissingPerformanceOutput tgradReplacementRoadmap).isEmpty = true := by
  native_decide

theorem roadmap_items_have_required_shape :
    (roadmapItemsWithBadShape tgradReplacementRoadmap).isEmpty = true := by
  native_decide

end Model
end Tgrad

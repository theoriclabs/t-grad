import Tgrad.Dtype

/-! # Tgrad.Model.Impact

  A small typed query model for reasoning about Tgrad changes before
  implementation. The first query is impact analysis:

      impactOf : DesiredChange -> List Impact

  The point is not to mirror every file. The point is to encode the
  durable architecture concepts from the mechanistic model so planner,
  architect, and builder agents can ask structured questions without
  re-parsing prose.
-/

namespace Tgrad
namespace Model

/-- Architectural plane touched by a change. -/
inductive Plane where
  | authoring
  | graph
  | compiler
  | runtime
  deriving BEq, Repr, Inhabited, DecidableEq

def Plane.toStr : Plane -> String
  | .authoring => "authoring"
  | .graph     => "graph"
  | .compiler  => "compiler"
  | .runtime   => "runtime"

/-- How directly a component is wired into current execution. -/
inductive Ring where
  | hotPath
  | decisionLogic
  | modeledSpec
  deriving BEq, Repr, Inhabited, DecidableEq

def Ring.toStr : Ring -> String
  | .hotPath       => "hot-path"
  | .decisionLogic => "decision-logic"
  | .modeledSpec   => "modeled-spec"

/-- Coarse components agents should reason about. -/
inductive Component where
  | pythonTensor
  | pythonFFI
  | cTrampoline
  | tensorRegistry
  | tensor
  | dtype
  | shape
  | uop
  | rangeify
  | viewIndexing
  | scheduler
  | optimizer
  | dispatchHeuristic
  | renderer
  | scalarMatmul
  | tensorCoreMatmul
  | runtimeExterns
  | metalBridge
  | backendAbstraction
  | bufferLifetime
  | graphRealize
  | modelAuthoring
  deriving BEq, Repr, Inhabited, DecidableEq

def Component.toStr : Component -> String
  | .pythonTensor       => "python-tensor"
  | .pythonFFI          => "python-ffi"
  | .cTrampoline        => "c-trampoline"
  | .tensorRegistry     => "tensor-registry"
  | .tensor             => "tensor"
  | .dtype              => "dtype"
  | .shape              => "shape"
  | .uop                => "uop"
  | .rangeify           => "rangeify"
  | .viewIndexing       => "view-indexing"
  | .scheduler          => "scheduler"
  | .optimizer          => "optimizer"
  | .dispatchHeuristic  => "dispatch-heuristic"
  | .renderer           => "renderer"
  | .scalarMatmul       => "scalar-matmul"
  | .tensorCoreMatmul   => "tensor-core-matmul"
  | .runtimeExterns     => "runtime-externs"
  | .metalBridge        => "metal-bridge"
  | .backendAbstraction => "backend-abstraction"
  | .bufferLifetime     => "buffer-lifetime"
  | .graphRealize       => "graph-realize"
  | .modelAuthoring     => "model-authoring"

/-- Missing abstraction an impact item may require before implementation. -/
inductive MissingAbstraction where
  | generalGraphRealize
  | generalRangeify
  | backendDescriptor
  | runtimeBackend
  | dtypeCodec
  | rendererTypeclass
  | scheduleSearchState
  | bufferLifetimePlan
  | modelGraphApi
  | opOntology
  | none
  deriving BEq, Repr, Inhabited, DecidableEq

def MissingAbstraction.toStr : MissingAbstraction -> String
  | .generalGraphRealize => "general-graph-realize"
  | .generalRangeify     => "general-rangeify"
  | .backendDescriptor   => "backend-descriptor"
  | .runtimeBackend      => "runtime-backend"
  | .dtypeCodec          => "dtype-codec"
  | .rendererTypeclass   => "renderer-typeclass"
  | .scheduleSearchState => "schedule-search-state"
  | .bufferLifetimePlan  => "buffer-lifetime-plan"
  | .modelGraphApi       => "model-graph-api"
  | .opOntology          => "op-ontology"
  | .none                => "none"

/-- Type of work implied by an impact item. -/
inductive ImpactKind where
  | addConcept
  | extendBoundary
  | refactor
  | routeChange
  | implementLowering
  | implementRuntime
  | validatePolicy
  deriving BEq, Repr, Inhabited, DecidableEq

def ImpactKind.toStr : ImpactKind -> String
  | .addConcept        => "add-concept"
  | .extendBoundary    => "extend-boundary"
  | .refactor          => "refactor"
  | .routeChange       => "route-change"
  | .implementLowering => "implement-lowering"
  | .implementRuntime  => "implement-runtime"
  | .validatePolicy    => "validate-policy"

/-- Backend families named by the first prototype. -/
inductive BackendFamily where
  | metal
  | cuda
  | rocm
  | cpu
  | webgpu
  deriving BEq, Repr, Inhabited, DecidableEq

def BackendFamily.toStr : BackendFamily -> String
  | .metal => "metal"
  | .cuda  => "cuda"
  | .rocm  => "rocm"
  | .cpu   => "cpu"
  | .webgpu => "webgpu"

/-- View semantics that may be added or generalized. -/
inductive ViewFeature where
  | negativeStepSlice
  | flip
  | pad
  | generalMovementComposition
  deriving BEq, Repr, Inhabited, DecidableEq

def ViewFeature.toStr : ViewFeature -> String
  | .negativeStepSlice        => "negative-step-slice"
  | .flip                     => "flip"
  | .pad                      => "pad"
  | .generalMovementComposition => "general-movement-composition"

/-- Larger workload targets. -/
inductive Workload where
  | transformerInference
  | fullInference
  deriving BEq, Repr, Inhabited, DecidableEq

def Workload.toStr : Workload -> String
  | .transformerInference => "transformer-inference"
  | .fullInference        => "full-inference"

/-- What the requested change is about. This is the change-description language's noun phrase. -/
inductive ChangeSubject where
  | runtimeDtype (dtype : Dtype)
  | viewFeature (feature : ViewFeature)
  | rangeifyAsRuntimePath
  | backend (family : BackendFamily)
  | workload (target : Workload)
  | scheduleSearch
  deriving Repr, Inhabited

def ChangeSubject.toStr : ChangeSubject -> String
  | .runtimeDtype d         => "runtime-dtype:" ++ d.toStr
  | .viewFeature f          => "view-feature:" ++ f.toStr
  | .rangeifyAsRuntimePath  => "rangeify-as-runtime-path"
  | .backend b              => "backend:" ++ b.toStr
  | .workload w             => "workload:" ++ w.toStr
  | .scheduleSearch         => "schedule-search"

/-- The change verb. -/
inductive ChangeVerb where
  | add
  | remove
  | replace
  | generalize
  | growToward
  deriving BEq, Repr, Inhabited, DecidableEq

def ChangeVerb.toStr : ChangeVerb -> String
  | .add        => "add"
  | .remove     => "remove"
  | .replace    => "replace"
  | .generalize => "generalize"
  | .growToward => "grow-toward"

/-- Constraints/preferences on a change request. -/
inductive ChangeConstraint where
  | keepPythonThin
  | preserveHotPath
  | preferPureLeanModel
  | requireRuntimeExecution
  | correctnessOnly
  | performanceRelevant
  deriving BEq, Repr, Inhabited, DecidableEq

def ChangeConstraint.toStr : ChangeConstraint -> String
  | .keepPythonThin       => "keep-python-thin"
  | .preserveHotPath      => "preserve-hot-path"
  | .preferPureLeanModel  => "prefer-pure-lean-model"
  | .requireRuntimeExecution => "require-runtime-execution"
  | .correctnessOnly      => "correctness-only"
  | .performanceRelevant  => "performance-relevant"

/-- A typed change description that planner agents can construct. -/
structure DesiredChange where
  verb        : ChangeVerb
  subject     : ChangeSubject
  constraints : List ChangeConstraint := []
  deriving Repr, Inhabited

def DesiredChange.toStr (c : DesiredChange) : String :=
  let cs := String.intercalate "," (c.constraints.map ChangeConstraint.toStr)
  if c.constraints.isEmpty then
    c.verb.toStr ++ " " ++ c.subject.toStr
  else
    c.verb.toStr ++ " " ++ c.subject.toStr ++ " [" ++ cs ++ "]"

def DesiredChange.hasConstraint (c : DesiredChange) (target : ChangeConstraint) : Bool :=
  c.constraints.any (fun x => x == target)

def DesiredChange.constraintNotes (c : DesiredChange) : List String :=
  let notes : List (ChangeConstraint × String) := [
    (.keepPythonThin, "keep Python as a handle/marshalling layer; semantic ownership should stay in Lean"),
    (.preserveHotPath, "avoid regressing the current matmul hot path while adding the abstraction"),
    (.preferPureLeanModel, "add or adjust pure Lean types/functions before introducing IO behavior"),
    (.requireRuntimeExecution, "documentation/spec-only work is insufficient; the change must reach executable dispatch or runtime state"),
    (.correctnessOnly, "performance parity is not required for the first version; route can favor simple scalar/lowering correctness"),
    (.performanceRelevant, "include route quality, measurement boundary, or scheduling policy in the design")
  ]
  notes.filterMap (fun (constraint, note) =>
    if c.hasConstraint constraint then some note else none)

/-- One ordered implementation impact. -/
structure Impact where
  order    : Nat
  planes   : List Plane
  component : Component
  ring     : Ring
  kind     : ImpactKind
  missing  : List MissingAbstraction
  reason   : String
  deriving Repr, Inhabited

def Impact.summary (i : Impact) : String :=
  let planes := String.intercalate "," (i.planes.map Plane.toStr)
  let missing :=
    if i.missing.isEmpty then "none"
    else String.intercalate "," (i.missing.map MissingAbstraction.toStr)
  s!"{i.order}: {i.component.toStr} [{planes}; {i.ring.toStr}; {i.kind.toStr}; missing={missing}] {i.reason}"

def Impact.toReportLine (i : Impact) : String :=
  "- " ++ i.summary

private def noneMissing : List MissingAbstraction := []

private def dtypeRuntimeNote (d : Dtype) : String :=
  if d == .float16_ then
    "Extend Dtype semantics only if existing float16 constructor is not enough for runtime policy."
  else
    "Dtype " ++ d.toStr ++ " already may exist in the lattice, but runtime support still requires storage, conversion, rendering, and route policy."

private def impactRuntimeDtype (d : Dtype) : List Impact := [
  { order := 1, planes := [.graph], component := .dtype, ring := .decisionLogic,
    kind := .addConcept, missing := noneMissing,
    reason := dtypeRuntimeNote d },
  { order := 2, planes := [.authoring, .runtime], component := .pythonTensor, ring := .hotPath,
    kind := .extendBoundary, missing := [.dtypeCodec],
    reason := "Python needs host conversion and byte lifting for " ++ d.toStr ++ ", not just bf16 truncation." },
  { order := 3, planes := [.authoring, .runtime], component := .pythonFFI, ring := .hotPath,
    kind := .extendBoundary, missing := [.dtypeCodec],
    reason := "Stable dtype codes and tensor queries must carry " ++ d.toStr ++ " through the handle boundary." },
  { order := 4, planes := [.compiler], component := .renderer, ring := .hotPath,
    kind := .implementLowering, missing := noneMissing,
    reason := "MSL buffer types, casts, and accumulator policy must render " ++ d.toStr ++ " correctly." },
  { order := 5, planes := [.compiler], component := .dispatchHeuristic, ring := .decisionLogic,
    kind := .validatePolicy, missing := noneMissing,
    reason := "TC/scalar routing depends on dtype pair and backend capability." }
]

private def impactViewFeature (feature : ViewFeature) : List Impact := [
  { order := 1, planes := [.graph], component := .uop, ring := .decisionLogic,
    kind := .addConcept, missing := noneMissing,
    reason := "Represent " ++ feature.toStr ++ " in the UOp/view ontology with typed fields." },
  { order := 2, planes := [.graph], component := .tensor, ring := .hotPath,
    kind := .implementLowering, missing := noneMissing,
    reason := "Tensor.shape and Tensor.buffer walkers must understand the new movement semantics." },
  { order := 3, planes := [.compiler], component := .rangeify, ring := .modeledSpec,
    kind := .implementLowering, missing := [.generalRangeify],
    reason := "Sustainable view support should lower movement chains to index UOps instead of adding Pipeline special cases." },
  { order := 4, planes := [.compiler], component := .viewIndexing, ring := .hotPath,
    kind := .refactor, missing := [.generalRangeify],
    reason := "Current view matmul derives A/B indices by hand; the new feature should either extend this or replace it with rangeify output." },
  { order := 5, planes := [.authoring, .runtime], component := .pythonFFI, ring := .hotPath,
    kind := .extendBoundary, missing := noneMissing,
    reason := "Expose the new view operation as handle-in, handle-out if Python needs a public method." }
]

private def impactRealRangeify : List Impact := [
  { order := 1, planes := [.compiler], component := .rangeify, ring := .modeledSpec,
    kind := .implementLowering, missing := [.generalRangeify],
    reason := "Replace identity/fixture-scope rangeify with a real movement-chain to indexed-load lowering." },
  { order := 2, planes := [.compiler], component := .viewIndexing, ring := .hotPath,
    kind := .refactor, missing := [.generalRangeify],
    reason := "Remove or shrink viewIndexUOpForA/B once rangeify can produce the required index expressions." },
  { order := 3, planes := [.compiler], component := .scalarMatmul, ring := .hotPath,
    kind := .implementLowering, missing := noneMissing,
    reason := "Keep scalarMatmulKernelDeclWithIdx as the consumer of rangeified index UOps." },
  { order := 4, planes := [.compiler], component := .graphRealize, ring := .modeledSpec,
    kind := .refactor, missing := [.generalGraphRealize],
    reason := "A real rangeify pass is a step toward making the modeled scheduler/lowerer load-bearing." },
  { order := 5, planes := [.authoring], component := .pythonTensor, ring := .hotPath,
    kind := .validatePolicy, missing := noneMissing,
    reason := "Python should remain a handle ferry; view semantics should stay in Lean." }
]

private def impactBackend (family : BackendFamily) : List Impact := [
  { order := 1, planes := [.compiler, .runtime], component := .backendAbstraction, ring := .decisionLogic,
    kind := .addConcept, missing := [.backendDescriptor, .runtimeBackend],
    reason := "Supporting " ++ family.toStr ++ " requires a first-class backend descriptor, not just a renderer branch." },
  { order := 2, planes := [.compiler], component := .renderer, ring := .hotPath,
    kind := .refactor, missing := [.rendererTypeclass],
    reason := "Metal-specific KernelDecl rendering must be split from backend-independent kernel intent." },
  { order := 3, planes := [.runtime], component := .runtimeExterns, ring := .hotPath,
    kind := .implementRuntime, missing := [.runtimeBackend],
    reason := "Compile, dispatch, buffer allocation, and synchronization need backend-specific extern/runtime implementations." },
  { order := 4, planes := [.compiler], component := .dispatchHeuristic, ring := .decisionLogic,
    kind := .validatePolicy, missing := [.backendDescriptor],
    reason := "Launch dims and TC eligibility depend on backend capabilities and tile inventory." },
  { order := 5, planes := [.authoring, .runtime], component := .pythonFFI, ring := .hotPath,
    kind := .extendBoundary, missing := [.backendDescriptor],
    reason := "Python handle creation and library loading need a way to select or carry device/backend identity." }
]

private def impactInference (w : Workload) : List Impact := [
  { order := 1, planes := [.graph], component := .uop, ring := .decisionLogic,
    kind := .addConcept, missing := [.opOntology],
    reason := w.toStr ++ " needs elementwise, reduction, normalization, gather, and multi-op graph vocabulary beyond matmul." },
  { order := 2, planes := [.compiler], component := .graphRealize, ring := .modeledSpec,
    kind := .implementLowering, missing := [.generalGraphRealize],
    reason := "The current runtime route must become one instance of a general UOp DAG realization path." },
  { order := 3, planes := [.compiler], component := .scheduler, ring := .modeledSpec,
    kind := .implementLowering, missing := [.scheduleSearchState, .bufferLifetimePlan],
    reason := "Inference requires multi-kernel ordering, fusion choices, and buffer lifetime planning." },
  { order := 4, planes := [.compiler], component := .optimizer, ring := .modeledSpec,
    kind := .addConcept, missing := [.scheduleSearchState],
    reason := "Better scheduling must become typed transformations over schedule states, not shape-only branches." },
  { order := 5, planes := [.authoring], component := .modelAuthoring, ring := .hotPath,
    kind := .extendBoundary, missing := [.modelGraphApi],
    reason := "Python needs a stable way to author or import model graphs while keeping compiler policy in Lean." },
  { order := 6, planes := [.runtime], component := .bufferLifetime, ring := .hotPath,
    kind := .implementRuntime, missing := [.bufferLifetimePlan],
    reason := "The append-only TensorRegistry and per-call buffer lifecycle are insufficient for long inference runs." }
]

private def impactScheduleSearch : List Impact := [
  { order := 1, planes := [.compiler], component := .optimizer, ring := .modeledSpec,
    kind := .addConcept, missing := [.scheduleSearchState],
    reason := "Define typed schedule states and actions before implementing search." },
  { order := 2, planes := [.compiler], component := .scheduler, ring := .modeledSpec,
    kind := .implementLowering, missing := [.scheduleSearchState],
    reason := "Enumeration, apply-action, and scoring should operate on schedule states." },
  { order := 3, planes := [.compiler], component := .dispatchHeuristic, ring := .decisionLogic,
    kind := .refactor, missing := noneMissing,
    reason := "pickDispatchPlan can become the BEAM=0 seed or fallback policy." },
  { order := 4, planes := [.runtime], component := .runtimeExterns, ring := .hotPath,
    kind := .validatePolicy, missing := noneMissing,
    reason := "Empirical search requires a measured execution boundary; pure search only needs estimates." }
]

/-- First mechanistic query: impact analysis for a desired change. -/
def impactOf (change : DesiredChange) : List Impact :=
  match change.subject with
  | .runtimeDtype d => impactRuntimeDtype d
  | .viewFeature f => impactViewFeature f
  | .rangeifyAsRuntimePath => impactRealRangeify
  | .backend b => impactBackend b
  | .workload w => impactInference w
  | .scheduleSearch => impactScheduleSearch

/-! ## Seeded changes from the first experiment plan. -/

def addFp16RuntimeDtype : DesiredChange :=
  { verb := .add, subject := .runtimeDtype .float16_,
    constraints := [.keepPythonThin, .requireRuntimeExecution] }

def addNegativeStepSlice : DesiredChange :=
  { verb := .add, subject := .viewFeature .negativeStepSlice,
    constraints := [.keepPythonThin, .correctnessOnly] }

def replaceViewSpecialCasesWithRangeify : DesiredChange :=
  { verb := .replace, subject := .rangeifyAsRuntimePath,
    constraints := [.preferPureLeanModel, .preserveHotPath] }

def addCudaBackend : DesiredChange :=
  { verb := .add, subject := .backend .cuda,
    constraints := [.preferPureLeanModel, .performanceRelevant] }

def growTowardTransformerInference : DesiredChange :=
  { verb := .growToward, subject := .workload .transformerInference,
    constraints := [.preferPureLeanModel, .requireRuntimeExecution] }

def seededChanges : List DesiredChange := [
  addFp16RuntimeDtype,
  addNegativeStepSlice,
  replaceViewSpecialCasesWithRangeify,
  addCudaBackend,
  growTowardTransformerInference
]

/-- Compact text summaries useful in `#eval` and future CLI/tool surfaces. -/
def summarizeImpact (change : DesiredChange) : List String :=
  (impactOf change).map Impact.summary

private def addUniqueMissing (xs : List MissingAbstraction) (m : MissingAbstraction) :
    List MissingAbstraction :=
  if m == .none then xs
  else if xs.any (fun x => x == m) then xs
  else xs ++ [m]

private def uniqueMissing (xs : List MissingAbstraction) : List MissingAbstraction :=
  xs.foldl addUniqueMissing []

/-- Second small query: missing abstractions that block or precede a change. -/
def blockersOf (change : DesiredChange) : List MissingAbstraction :=
  uniqueMissing ((impactOf change).foldl (fun acc i => acc ++ i.missing) [])

/-- Impact records directly wired into the runtime hot path. -/
def hotPathImpacts (change : DesiredChange) : List Impact :=
  (impactOf change).filter (fun i => i.ring == .hotPath)

/-- Impact records in currently modeled/spec code rather than the hot path. -/
def modeledSpecImpacts (change : DesiredChange) : List Impact :=
  (impactOf change).filter (fun i => i.ring == .modeledSpec)

/-- Does this change pressure modeled/spec modules to become real runtime machinery? -/
def requiresModeledSpecPromotion (change : DesiredChange) : Bool :=
  let blockers := blockersOf change
  !(modeledSpecImpacts change).isEmpty ||
    blockers.any (fun b =>
      b == .generalGraphRealize ||
      b == .generalRangeify ||
      b == .scheduleSearchState)

/-- Human-readable report surface for agents and future CLI/tool wrappers. -/
def impactReport (change : DesiredChange) : String :=
  let title := "change: " ++ change.toStr
  let blockers := blockersOf change
  let blockerLine :=
    if blockers.isEmpty then "blockers: none"
    else "blockers: " ++ String.intercalate ", " (blockers.map MissingAbstraction.toStr)
  let promotionLine :=
    "requires-modeled-spec-promotion: " ++
      (if requiresModeledSpecPromotion change then "yes" else "no")
  let notes := change.constraintNotes
  let notesLine :=
    if notes.isEmpty then "constraint-notes: none"
    else "constraint-notes:\n" ++ String.intercalate "\n" (notes.map (fun n => "- " ++ n))
  let impacts := String.intercalate "\n" ((impactOf change).map Impact.toReportLine)
  title ++ "\n" ++ blockerLine ++ "\n" ++ promotionLine ++ "\n" ++ notesLine ++ "\nimpacts:\n" ++ impacts

/-- Smoke-check the seeded model is populated. -/
def seededImpactCounts : List Nat :=
  seededChanges.map (fun c => (impactOf c).length)

theorem seededImpactCounts_nonempty :
    seededImpactCounts.all (fun n => n > 0) = true := by
  native_decide

theorem seededBlockers_nonempty_for_backend :
    (blockersOf addCudaBackend).isEmpty = false := by
  native_decide

theorem rangeify_requires_promotion :
    requiresModeledSpecPromotion replaceViewSpecialCasesWithRangeify = true := by
  native_decide

theorem fp16_has_hot_path_impacts :
    (hotPathImpacts addFp16RuntimeDtype).isEmpty = false := by
  native_decide

theorem correctnessOnly_has_note :
    (addNegativeStepSlice.constraintNotes).isEmpty = false := by
  native_decide

end Model
end Tgrad

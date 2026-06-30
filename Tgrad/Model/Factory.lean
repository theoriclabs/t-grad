import Tgrad.Model.Impact

/-! # Tgrad.Model.Factory

  A first typed control-plane model for producing Tgrad as a product.

  This intentionally mirrors the useful shape of the Webmapper factory without
  depending on that repository: product artifacts, repo refs, production
  functions, process stations, actor roles, gates, and small computable queries.

  Lean does not execute the factory. Lean names the factory's contracts so
  planner, architect, builder, and verifier agents can reason over them before
  mutating the product repo.
-/

namespace Tgrad
namespace Model

/-- The product line this factory is responsible for. -/
inductive ProductLine where
  | tgradSeed
  | fullTinygradReplacement
  deriving BEq, Repr, Inhabited, DecidableEq

def ProductLine.toStr : ProductLine -> String
  | .tgradSeed => "tgrad-seed"
  | .fullTinygradReplacement => "full-tinygrad-replacement"

/-- Artifact classes that are part of the Tgrad product or its factory. -/
inductive ProductArtifactKind where
  | sourceRepoRef
  | productDescription
  | productSpec
  | compatibilitySpec
  | capabilityMap
  | mechanisticModel
  | leanQueryModel
  | workOrder
  | designPacket
  | implementationPatch
  | verificationEvidence
  | performanceEvidence
  | releaseCandidate
  | learningRecord
  deriving BEq, Repr, Inhabited, DecidableEq

def ProductArtifactKind.toStr : ProductArtifactKind -> String
  | .sourceRepoRef => "source-repo-ref"
  | .productDescription => "product-description"
  | .productSpec => "product-spec"
  | .compatibilitySpec => "compatibility-spec"
  | .capabilityMap => "capability-map"
  | .mechanisticModel => "mechanistic-model"
  | .leanQueryModel => "lean-query-model"
  | .workOrder => "work-order"
  | .designPacket => "design-packet"
  | .implementationPatch => "implementation-patch"
  | .verificationEvidence => "verification-evidence"
  | .performanceEvidence => "performance-evidence"
  | .releaseCandidate => "release-candidate"
  | .learningRecord => "learning-record"

/-- Quality ladder for factory artifacts. -/
inductive ArtifactQuality where
  | raw
  | scoped
  | checked
  | accepted
  | released
  deriving BEq, Repr, Inhabited, DecidableEq

def ArtifactQuality.toStr : ArtifactQuality -> String
  | .raw => "raw"
  | .scoped => "scoped"
  | .checked => "checked"
  | .accepted => "accepted"
  | .released => "released"

/-- A typed artifact port consumed or produced by a production function. -/
structure ArtifactPort where
  name : String
  kind : ProductArtifactKind
  minimumQuality : ArtifactQuality
  description : String := ""
  deriving Repr, Inhabited

def artifact (name : String) (kind : ProductArtifactKind)
    (quality : ArtifactQuality) (description : String := "") : ArtifactPort := {
  name := name
  kind := kind
  minimumQuality := quality
  description := description
}

/-- Product-growth actors. Agents may occupy these roles, but the role is the contract. -/
inductive ActorRole where
  | principal
  | factoryDri
  | productDesigner
  | architect
  | builder
  | verifier
  | performanceEngineer
  | releaseOwner
  deriving BEq, Repr, Inhabited, DecidableEq

def ActorRole.toStr : ActorRole -> String
  | .principal => "principal"
  | .factoryDri => "factory-dri"
  | .productDesigner => "product-designer"
  | .architect => "architect"
  | .builder => "builder"
  | .verifier => "verifier"
  | .performanceEngineer => "performance-engineer"
  | .releaseOwner => "release-owner"

/-- Gates are named contracts; their checkers may be shell, Lean, Python, or human review. -/
inductive GateKind where
  | build
  | typecheck
  | compatibility
  | correctness
  | performance
  | codeReview
  | productReview
  | evidenceReview
  deriving BEq, Repr, Inhabited, DecidableEq

def GateKind.toStr : GateKind -> String
  | .build => "build"
  | .typecheck => "typecheck"
  | .compatibility => "compatibility"
  | .correctness => "correctness"
  | .performance => "performance"
  | .codeReview => "code-review"
  | .productReview => "product-review"
  | .evidenceReview => "evidence-review"

structure QualityGate where
  id : String
  kind : GateKind
  requiredQuality : ArtifactQuality
  method : String
  description : String
  deriving Repr, Inhabited

def gate (id : String) (kind : GateKind) (quality : ArtifactQuality)
    (method description : String) : QualityGate := {
  id := id
  kind := kind
  requiredQuality := quality
  method := method
  description := description
}

/-- Immutable product input. The ambient checkout is evidence, not the semantic input. -/
structure ProductRepoRef where
  id : String
  repoUrlOrPath : String
  inputShaOrTag : String
  expectedTreeHash : String
  dirtyAllowed : Bool := false
  isolatedWorktreeOrBranch : String
  notes : String := ""
  deriving Repr, Inhabited

namespace ProductRepoRef

def declared (ref : ProductRepoRef) : Bool :=
  ref.id != "" &&
  ref.repoUrlOrPath != "" &&
  ref.inputShaOrTag != "" &&
  ref.expectedTreeHash != "" &&
  ref.isolatedWorktreeOrBranch != ""

end ProductRepoRef

/-- Accepted product output. Product mutation is a repo-ref transformation. -/
structure ProductOutputRef where
  id : String
  repoUrlOrPath : String
  parentShaOrTag : String
  outputShaOrTag : String
  diffHash : String
  verificationRefs : List String
  acceptanceRefs : List String
  deriving Repr, Inhabited

namespace ProductOutputRef

def declared (ref : ProductOutputRef) : Bool :=
  ref.id != "" &&
  ref.repoUrlOrPath != "" &&
  ref.parentShaOrTag != "" &&
  ref.outputShaOrTag != "" &&
  ref.diffHash != "" &&
  !ref.verificationRefs.isEmpty &&
  !ref.acceptanceRefs.isEmpty

def parentMatchesInput (output : ProductOutputRef) (input : ProductRepoRef) : Bool :=
  output.parentShaOrTag == input.inputShaOrTag

end ProductOutputRef

/-- A function transforms typed inputs into typed outputs under named gates. -/
structure ProductionFunction where
  id : String
  name : String
  purpose : String
  inputs : List ArtifactPort
  outputs : List ArtifactPort
  owner : ActorRole
  gates : List QualityGate
  touchesProductCode : Bool := false
  notes : String := ""
  deriving Repr, Inhabited

def ProductionFunction.produces (pf : ProductionFunction)
    (kind : ProductArtifactKind) : Bool :=
  pf.outputs.any (fun p => p.kind == kind)

def ProductionFunction.consumes (pf : ProductionFunction)
    (kind : ProductArtifactKind) : Bool :=
  pf.inputs.any (fun p => p.kind == kind)

/-- A process station organizes one or more production functions. -/
structure ProductionProcess where
  id : String
  name : String
  purpose : String
  functionIds : List String
  functions : List ProductionFunction
  owner : ActorRole
  exitRule : String
  deriving Repr, Inhabited

def ProductionProcess.homesFunction (process : ProductionProcess) (id : String) : Bool :=
  process.functionIds.any (fun fid => fid == id)

/-- The product/factory graph as a typed registry. -/
structure TgradFactory where
  id : String
  productLine : ProductLine
  product : ArtifactPort
  standingInputs : List ArtifactPort
  processes : List ProductionProcess
  invariants : List String
  deriving Repr, Inhabited

def TgradFactory.functions (factory : TgradFactory) : List ProductionFunction :=
  factory.processes.foldl (fun acc p => acc ++ p.functions) []

def TgradFactory.functionIds (factory : TgradFactory) : List String :=
  (factory.functions).map (fun pf => pf.id)

private def containsString (xs : List String) (x : String) : Bool :=
  xs.any (fun y => y == x)

private def countString (xs : List String) (x : String) : Nat :=
  xs.foldl (fun n y => if y == x then n + 1 else n) 0

def TgradFactory.coversRegistry (factory : TgradFactory) : Bool :=
  let registry := factory.functionIds
  let homed := factory.processes.foldl (fun acc p => acc ++ p.functionIds) []
  registry.all (fun id => containsString homed id)

def TgradFactory.partitionsRegistry (factory : TgradFactory) : Bool :=
  let registry := factory.functionIds
  let homed := factory.processes.foldl (fun acc p => acc ++ p.functionIds) []
  factory.coversRegistry &&
    registry.all (fun id => countString homed id == 1)

def TgradFactory.functionsProducing (factory : TgradFactory)
    (kind : ProductArtifactKind) : List ProductionFunction :=
  (factory.functions).filter (fun pf => pf.produces kind)

/-! ## Tgrad production functions. -/

def pfProductFrame : ProductionFunction := {
  id := "TPF1"
  name := "Frame Tgrad product"
  purpose :=
    "Turn a repo+sha, user goal, and tinygrad replacement ambition into product description, product specification, and claim boundary."
  inputs := [
    artifact "input product repo ref" .sourceRepoRef .checked,
    artifact "user/product goal" .workOrder .scoped
  ]
  outputs := [
    artifact "Tgrad product description" .productDescription .checked,
    artifact "Tgrad product specification" .productSpec .checked
  ]
  owner := .productDesigner
  gates := [
    gate "TPF1-product-review" .productReview .checked "human+agent review"
      "The product unit, non-goals, compatibility target, and claim boundary are explicit."
  ]
}

def pfCapabilityMap : ProductionFunction := {
  id := "TPF2"
  name := "Map capability gaps"
  purpose :=
    "Compare the Tgrad seed against the full tinygrad replacement target and emit a typed capability/gap map."
  inputs := [
    artifact "Tgrad product specification" .productSpec .checked,
    artifact "input product repo ref" .sourceRepoRef .checked
  ]
  outputs := [
    artifact "tinygrad replacement capability map" .capabilityMap .checked,
    artifact "compatibility specification" .compatibilitySpec .checked
  ]
  owner := .architect
  gates := [
    gate "TPF2-compatibility-scope" .compatibility .checked "spec review"
      "The map names supported, partial, missing, and explicitly deferred tinygrad capabilities."
  ]
}

def pfMechanisticModel : ProductionFunction := {
  id := "TPF3"
  name := "Update mechanistic model"
  purpose :=
    "Maintain the code graph, impact query model, missing abstractions, and architectural consequences before implementation."
  inputs := [
    artifact "product specification" .productSpec .checked,
    artifact "capability map" .capabilityMap .checked,
    artifact "input product repo ref" .sourceRepoRef .checked
  ]
  outputs := [
    artifact "mechanistic model" .mechanisticModel .checked,
    artifact "Lean query model" .leanQueryModel .checked
  ]
  owner := .architect
  gates := [
    gate "TPF3-lean-build" .typecheck .checked "lake build Tgrad"
      "The query model compiles and seeded queries remain total.",
    gate "TPF3-impact-coverage" .evidenceReview .checked "impact query probe"
      "The requested change has affected planes, components, hot-path classification, missing abstractions, and ordering."
  ]
}

def pfDesignSlice : ProductionFunction := {
  id := "TPF4"
  name := "Design one product slice"
  purpose :=
    "Convert a capability gap or DesiredChange into a bounded design packet with implementation order and acceptance gates."
  inputs := [
    artifact "capability map" .capabilityMap .checked,
    artifact "mechanistic model" .mechanisticModel .checked,
    artifact "Lean query model" .leanQueryModel .checked
  ]
  outputs := [
    artifact "design packet" .designPacket .checked,
    artifact "work order" .workOrder .checked
  ]
  owner := .architect
  gates := [
    gate "TPF4-design-is-bounded" .productReview .checked "architecture review"
      "The slice has a small surface, explicit changed planes, missing abstractions, and rollback boundary."
  ]
}

def pfImplementSlice : ProductionFunction := {
  id := "TPF5"
  name := "Implement one product slice"
  purpose :=
    "Transform an explicit input repo+sha and design packet into a patch or output repo ref."
  inputs := [
    artifact "input product repo ref" .sourceRepoRef .checked,
    artifact "design packet" .designPacket .checked,
    artifact "work order" .workOrder .checked
  ]
  outputs := [
    artifact "implementation patch" .implementationPatch .checked,
    artifact "candidate output repo ref" .sourceRepoRef .checked
  ]
  owner := .builder
  gates := [
    gate "TPF5-build" .build .checked "lake build Tgrad"
      "Lean product builds after the slice.",
    gate "TPF5-no-ambient-input" .evidenceReview .checked "repo-ref audit"
      "Accepted mutation records parent repo+sha and output repo+sha or explicit blocker."
  ]
  touchesProductCode := true
}

def pfVerifySlice : ProductionFunction := {
  id := "TPF6"
  name := "Verify one product slice"
  purpose :=
    "Run correctness, compatibility, and performance gates appropriate to the slice and attach evidence."
  inputs := [
    artifact "candidate output repo ref" .sourceRepoRef .checked,
    artifact "implementation patch" .implementationPatch .checked,
    artifact "design packet" .designPacket .checked
  ]
  outputs := [
    artifact "verification evidence" .verificationEvidence .checked,
    artifact "performance evidence" .performanceEvidence .scoped
  ]
  owner := .verifier
  gates := [
    gate "TPF6-correctness" .correctness .checked "gate/test suite"
      "The slice's positive and negative checks pass at the declared boundary.",
    gate "TPF6-performance" .performance .scoped "benchmark or explicit non-claim"
      "Performance is measured when the slice claims hot-path or scheduling value."
  ]
}

def pfPromoteReleaseCandidate : ProductionFunction := {
  id := "TPF7"
  name := "Promote release candidate"
  purpose :=
    "Accept a verified repo-ref transformation as the next product version or reject it with explicit blockers."
  inputs := [
    artifact "candidate output repo ref" .sourceRepoRef .checked,
    artifact "verification evidence" .verificationEvidence .checked
  ]
  outputs := [
    artifact "release candidate" .releaseCandidate .accepted
  ]
  owner := .releaseOwner
  gates := [
    gate "TPF7-review" .codeReview .accepted "architect/release review"
      "Product output ref parent matches input ref, gates pass, and residual risks are recorded."
  ]
  touchesProductCode := true
}

def pfLearn : ProductionFunction := {
  id := "TPF8"
  name := "Learn from production run"
  purpose :=
    "Turn accepted, rejected, or blocked run evidence into updated product spec, capability map, model, or next work."
  inputs := [
    artifact "release candidate or rejected patch" .releaseCandidate .scoped,
    artifact "verification evidence" .verificationEvidence .checked
  ]
  outputs := [
    artifact "learning record" .learningRecord .checked,
    artifact "next work order" .workOrder .scoped
  ]
  owner := .factoryDri
  gates := [
    gate "TPF8-learning" .evidenceReview .checked "retrospective"
      "Every run either changes product state, improves factory state, or records a named deferral."
  ]
}

def tgradProcesses : List ProductionProcess := [
  {
    id := "TS1"
    name := "Product design and specification"
    purpose := "Make product description and specification first-class factory outputs."
    functionIds := ["TPF1"]
    functions := [pfProductFrame]
    owner := .productDesigner
    exitRule := "Product description/specification are checked and tied to an input repo ref."
  },
  {
    id := "TS2"
    name := "Capability and compatibility mapping"
    purpose := "Describe the full tinygrad replacement target as typed gaps and compatibility promises."
    functionIds := ["TPF2"]
    functions := [pfCapabilityMap]
    owner := .architect
    exitRule := "Capability map and compatibility spec are checked."
  },
  {
    id := "TS3"
    name := "Mechanistic architecture model"
    purpose := "Keep code-shape and impact queries current before implementation."
    functionIds := ["TPF3", "TPF4"]
    functions := [pfMechanisticModel, pfDesignSlice]
    owner := .architect
    exitRule := "A bounded design packet exists with query-backed consequences."
  },
  {
    id := "TS4"
    name := "Implementation"
    purpose := "Apply one bounded product slice as a repo-ref transformation."
    functionIds := ["TPF5"]
    functions := [pfImplementSlice]
    owner := .builder
    exitRule := "Patch builds and emits candidate output repo ref or explicit blocker."
  },
  {
    id := "TS5"
    name := "Verification"
    purpose := "Attach correctness, compatibility, and performance evidence to the candidate."
    functionIds := ["TPF6"]
    functions := [pfVerifySlice]
    owner := .verifier
    exitRule := "Evidence passes or the run is rejected with a classified failure."
  },
  {
    id := "TS6"
    name := "Promotion"
    purpose := "Promote verified product output to next version."
    functionIds := ["TPF7"]
    functions := [pfPromoteReleaseCandidate]
    owner := .releaseOwner
    exitRule := "Accepted candidate records parent, output, diff, evidence, and residual risk."
  },
  {
    id := "TS7"
    name := "Learning"
    purpose := "Update factory/product knowledge from every run."
    functionIds := ["TPF8"]
    functions := [pfLearn]
    owner := .factoryDri
    exitRule := "Run learning is converted into changed spec/model/process or a next work order."
  }
]

def tgradFactory : TgradFactory := {
  id := "factory.tgrad.v0"
  productLine := .fullTinygradReplacement
  product := artifact "Tgrad" .releaseCandidate .released
    "Lean-owned tinygrad replacement: authoring API, tensor graph, compiler, schedulers, renderers, runtimes, and compatibility surface."
  standingInputs := [
    artifact "Tgrad source repo ref" .sourceRepoRef .checked,
    artifact "current mechanistic model" .mechanisticModel .checked
  ]
  processes := tgradProcesses
  invariants := [
    "Product design and product specification are product artifacts, not side-channel notes.",
    "Product-changing work transforms explicit input repo+sha into explicit output repo+sha or an explicit blocker.",
    "Python remains an authoring/marshalling boundary unless a work order explicitly changes that boundary.",
    "Semantic compiler/runtime concepts enter Lean types or pure functions before IO behavior claims them.",
    "Hot-path changes require either measured performance evidence or an explicit non-performance claim."
  ]
}

/-! ## Capability map for full tinygrad replacement. -/

inductive CapabilityArea where
  | authoringApi
  | dtypeSystem
  | shapeViewSemantics
  | opOntology
  | autograd
  | scheduler
  | codegen
  | runtime
  | backend
  | inference
  | training
  deriving BEq, Repr, Inhabited, DecidableEq

def CapabilityArea.toStr : CapabilityArea -> String
  | .authoringApi => "authoring-api"
  | .dtypeSystem => "dtype-system"
  | .shapeViewSemantics => "shape-view-semantics"
  | .opOntology => "op-ontology"
  | .autograd => "autograd"
  | .scheduler => "scheduler"
  | .codegen => "codegen"
  | .runtime => "runtime"
  | .backend => "backend"
  | .inference => "inference"
  | .training => "training"

inductive CapabilityStatus where
  | seeded
  | partialSupport
  | absent
  | blocked
  | accepted
  deriving BEq, Repr, Inhabited, DecidableEq

def CapabilityStatus.toStr : CapabilityStatus -> String
  | .seeded => "seed"
  | .partialSupport => "partial"
  | .absent => "missing"
  | .blocked => "blocked"
  | .accepted => "accepted"

structure CapabilitySpec where
  id : String
  area : CapabilityArea
  target : ProductLine
  status : CapabilityStatus
  requiredArtifacts : List ProductArtifactKind
  likelyChanges : List DesiredChange
  acceptanceBoundary : String
  deriving Repr, Inhabited

def CapabilitySpec.needsWork (cap : CapabilitySpec) : Bool :=
  cap.status == .seeded ||
  cap.status == .partialSupport ||
  cap.status == .absent ||
  cap.status == .blocked

def fullReplacementCapabilityMap : List CapabilitySpec := [
  {
    id := "cap.dtype.full-runtime"
    area := .dtypeSystem
    target := .fullTinygradReplacement
    status := .partialSupport
    requiredArtifacts := [.productSpec, .compatibilitySpec, .leanQueryModel, .verificationEvidence]
    likelyChanges := [addFp16RuntimeDtype]
    acceptanceBoundary := "dtype storage, Python marshalling, renderer casts, and runtime dispatch support all declared dtypes."
  },
  {
    id := "cap.views.general"
    area := .shapeViewSemantics
    target := .fullTinygradReplacement
    status := .partialSupport
    requiredArtifacts := [.mechanisticModel, .leanQueryModel, .verificationEvidence]
    likelyChanges := [addNegativeStepSlice, replaceViewSpecialCasesWithRangeify]
    acceptanceBoundary := "movement chains lower through general rangeify/indexing rather than per-view matmul special cases."
  },
  {
    id := "cap.backends.multi"
    area := .backend
    target := .fullTinygradReplacement
    status := .absent
    requiredArtifacts := [.productSpec, .designPacket, .verificationEvidence, .performanceEvidence]
    likelyChanges := [addCudaBackend]
    acceptanceBoundary := "backend descriptor, renderer, runtime externs, and dispatch policy support at least one non-Metal backend."
  },
  {
    id := "cap.inference.transformer"
    area := .inference
    target := .fullTinygradReplacement
    status := .absent
    requiredArtifacts := [.productSpec, .capabilityMap, .designPacket, .verificationEvidence, .performanceEvidence]
    likelyChanges := [growTowardTransformerInference]
    acceptanceBoundary := "a bounded transformer inference graph executes through general graph realization with evidence."
  },
  {
    id := "cap.training.autograd"
    area := .training
    target := .fullTinygradReplacement
    status := .absent
    requiredArtifacts := [.productSpec, .capabilityMap, .mechanisticModel, .verificationEvidence]
    likelyChanges := []
    acceptanceBoundary := "autograd graph construction, backward lowering, optimizer steps, and training-loop state are modeled and executable."
  },
  {
    id := "cap.scheduler.search"
    area := .scheduler
    target := .fullTinygradReplacement
    status := .seeded
    requiredArtifacts := [.mechanisticModel, .leanQueryModel, .performanceEvidence]
    likelyChanges := [{ verb := .add, subject := .scheduleSearch, constraints := [.preferPureLeanModel, .performanceRelevant] }]
    acceptanceBoundary := "schedule states/actions/scoring exist and can improve or preserve measured routes."
  }
]

def capabilitiesNeedingWork : List CapabilitySpec :=
  fullReplacementCapabilityMap.filter CapabilitySpec.needsWork

def capabilitiesForArea (area : CapabilityArea) : List CapabilitySpec :=
  fullReplacementCapabilityMap.filter (fun cap => cap.area == area)

/-! ## Planner queries. -/

def productDesignFunctions : List ProductionFunction :=
  (tgradFactory.functions).filter (fun pf =>
    pf.produces .productDescription || pf.produces .productSpec)

def productChangingFunctions : List ProductionFunction :=
  (tgradFactory.functions).filter (fun pf => pf.touchesProductCode)

def productionFunctionIdsForChange (change : DesiredChange) : List String :=
  let core := ["TPF3", "TPF4", "TPF5", "TPF6", "TPF7", "TPF8"]
  match change.subject with
  | .backend _ => ["TPF1", "TPF2"] ++ core
  | .workload _ => ["TPF1", "TPF2"] ++ core
  | .scheduleSearch => core
  | _ => core

def productionFunctionsForChange (change : DesiredChange) : List ProductionFunction :=
  let ids := productionFunctionIdsForChange change
  (tgradFactory.functions).filter (fun pf => containsString ids pf.id)

def changeTouchesProductDesign (change : DesiredChange) : Bool :=
  (productionFunctionIdsForChange change).any (fun id => id == "TPF1" || id == "TPF2")

def changeRequiresRepoRefTransform (change : DesiredChange) : Bool :=
  (productionFunctionsForChange change).any (fun pf => pf.touchesProductCode)

def processIdsForChange (change : DesiredChange) : List String :=
  let functionIds := productionFunctionIdsForChange change
  tgradFactory.processes.foldl (fun acc p =>
    if p.functionIds.any (fun id => containsString functionIds id) then acc ++ [p.id] else acc) []

def factoryPlanSummary (change : DesiredChange) : String :=
  let pfs := String.intercalate "," (productionFunctionIdsForChange change)
  let ps := String.intercalate "," (processIdsForChange change)
  "change: " ++ change.toStr ++
  "\nprocesses: " ++ ps ++
  "\nproduction-functions: " ++ pfs ++
  "\ntouches-product-design: " ++ (if changeTouchesProductDesign change then "yes" else "no") ++
  "\nrequires-repo-ref-transform: " ++ (if changeRequiresRepoRefTransform change then "yes" else "no")

/-! ## Work admission and product-change acceptance. -/

inductive WorkOrderStatus where
  | draft
  | scoped
  | admitted
  | blocked
  | accepted
  | rejected
  deriving BEq, Repr, Inhabited, DecidableEq

def WorkOrderStatus.toStr : WorkOrderStatus -> String
  | .draft => "draft"
  | .scoped => "scoped"
  | .admitted => "admitted"
  | .blocked => "blocked"
  | .accepted => "accepted"
  | .rejected => "rejected"

def gateIdsForChange (change : DesiredChange) : List String :=
  (productionFunctionsForChange change).foldl
    (fun acc pf => acc ++ pf.gates.map (fun g => g.id))
    []

/-- Admitted factory work binds a change to process route, gates, and repo input. -/
structure TgradWorkOrder where
  id : String
  title : String
  status : WorkOrderStatus
  change : DesiredChange
  requestedBy : ActorRole
  owner : ActorRole
  inputRepoRef : Option ProductRepoRef := none
  productionFunctionIds : List String
  processIds : List String
  expectedOutputKinds : List ProductArtifactKind
  gateIds : List String
  downstreamConsumer : String
  notes : String := ""
  deriving Repr, Inhabited

def TgradWorkOrder.hasDeclaredInputRepo (wo : TgradWorkOrder) : Bool :=
  match wo.inputRepoRef with
  | none => false
  | some ref => ref.declared

def TgradWorkOrder.requiresRepoRefTransform (wo : TgradWorkOrder) : Bool :=
  changeRequiresRepoRefTransform wo.change

def TgradWorkOrder.admissible (wo : TgradWorkOrder) : Bool :=
  wo.id != "" &&
  wo.title != "" &&
  wo.status == .scoped &&
  !wo.productionFunctionIds.isEmpty &&
  !wo.processIds.isEmpty &&
  !wo.expectedOutputKinds.isEmpty &&
  !wo.gateIds.isEmpty &&
  wo.downstreamConsumer != "" &&
  (if wo.requiresRepoRefTransform then wo.hasDeclaredInputRepo else true)

def workOrderForChange (id title : String) (change : DesiredChange)
    (inputRepoRef : Option ProductRepoRef := none) : TgradWorkOrder := {
  id := id
  title := title
  status := .scoped
  change := change
  requestedBy := .factoryDri
  owner := .architect
  inputRepoRef := inputRepoRef
  productionFunctionIds := productionFunctionIdsForChange change
  processIds := processIdsForChange change
  expectedOutputKinds := [.designPacket, .implementationPatch, .verificationEvidence, .releaseCandidate, .learningRecord]
  gateIds := gateIdsForChange change
  downstreamConsumer := "release candidate or next work order"
}

inductive FactoryDecision where
  | accept
  | reject
  | block
  deriving BEq, Repr, Inhabited, DecidableEq

def FactoryDecision.toStr : FactoryDecision -> String
  | .accept => "accept"
  | .reject => "reject"
  | .block => "block"

/-- Acceptance packet for a product-changing run. -/
structure ProductChangePacket where
  id : String
  workOrder : TgradWorkOrder
  decision : FactoryDecision
  outputRef : Option ProductOutputRef := none
  evidenceRefs : List String
  residualRisks : List String := []
  learningRefs : List String := []
  deriving Repr, Inhabited

def ProductChangePacket.inputOutputMatch (packet : ProductChangePacket) : Bool :=
  match packet.workOrder.inputRepoRef, packet.outputRef with
  | some inputRef, some outputRef => outputRef.parentMatchesInput inputRef
  | _, _ => false

def ProductChangePacket.acceptable (packet : ProductChangePacket) : Bool :=
  packet.id != "" &&
  packet.decision == .accept &&
  packet.workOrder.admissible &&
  !packet.evidenceRefs.isEmpty &&
  !packet.learningRefs.isEmpty &&
  (match packet.outputRef with
   | none => false
   | some outputRef => outputRef.declared) &&
  packet.inputOutputMatch

def sampleInputRepoRef : ProductRepoRef := {
  id := "repo.tgrad.sample"
  repoUrlOrPath := "."
  inputShaOrTag := "sample-input-sha"
  expectedTreeHash := "sample-tree-hash"
  isolatedWorktreeOrBranch := "sample-worktree"
}

def sampleFp16WorkOrder : TgradWorkOrder :=
  workOrderForChange "WO.sample.fp16" "Add fp16 runtime dtype" addFp16RuntimeDtype
    (some sampleInputRepoRef)

def sampleBackendWorkOrderWithoutRepo : TgradWorkOrder :=
  workOrderForChange "WO.sample.cuda" "Add CUDA backend abstraction" addCudaBackend none

/-! ## Typed roadmap. -/

inductive RoadmapHorizon where
  | foundation
  | semanticCore
  | compilerRuntime
  | backendExpansion
  | inference
  | training
  | parity
  deriving BEq, Repr, Inhabited, DecidableEq

def RoadmapHorizon.toStr : RoadmapHorizon -> String
  | .foundation => "foundation"
  | .semanticCore => "semantic-core"
  | .compilerRuntime => "compiler-runtime"
  | .backendExpansion => "backend-expansion"
  | .inference => "inference"
  | .training => "training"
  | .parity => "parity"

inductive RoadmapItemKind where
  | productSpec
  | capabilityMap
  | architecture
  | implementation
  | validation
  | release
  deriving BEq, Repr, Inhabited, DecidableEq

def RoadmapItemKind.toStr : RoadmapItemKind -> String
  | .productSpec => "product-spec"
  | .capabilityMap => "capability-map"
  | .architecture => "architecture"
  | .implementation => "implementation"
  | .validation => "validation"
  | .release => "release"

/-- Roadmap status is intent/state. Accepted progress is tracked separately by packets. -/
inductive RoadmapItemStatus where
  | specified
  | planned
  | active
  | acceptedByPacket
  | deferred
  deriving BEq, Repr, Inhabited, DecidableEq

def RoadmapItemStatus.toStr : RoadmapItemStatus -> String
  | .specified => "specified"
  | .planned => "planned"
  | .active => "active"
  | .acceptedByPacket => "accepted-by-packet"
  | .deferred => "deferred"

inductive RepoTransformRequirement where
  | notRequired
  | required
  | requiredWithPerformanceEvidence
  deriving BEq, Repr, Inhabited, DecidableEq

def RepoTransformRequirement.toStr : RepoTransformRequirement -> String
  | .notRequired => "not-required"
  | .required => "required"
  | .requiredWithPerformanceEvidence => "required-with-performance-evidence"

/-- A roadmap item is a typed promise about one accepted step toward replacement. -/
structure RoadmapItem where
  id : String
  title : String
  horizon : RoadmapHorizon
  kind : RoadmapItemKind
  status : RoadmapItemStatus
  capabilityIds : List String
  desiredChanges : List DesiredChange
  dependencies : List String
  expectedOutputs : List ProductArtifactKind
  repoTransform : RepoTransformRequirement
  acceptanceBoundary : String
  notes : String := ""
  deriving Repr, Inhabited

def RoadmapItem.requiresRepoTransform (item : RoadmapItem) : Bool :=
  item.repoTransform == .required ||
  item.repoTransform == .requiredWithPerformanceEvidence

def RoadmapItem.needsPerformanceEvidence (item : RoadmapItem) : Bool :=
  item.repoTransform == .requiredWithPerformanceEvidence

def RoadmapItem.dependsOn (item : RoadmapItem) (id : String) : Bool :=
  containsString item.dependencies id

def RoadmapItem.blockedBy (acceptedItemIds : List String) (item : RoadmapItem) :
    List String :=
  item.dependencies.filter (fun dep => !containsString acceptedItemIds dep)

def RoadmapItem.readyGiven (acceptedItemIds : List String) (item : RoadmapItem) :
    Bool :=
  item.status == .planned &&
  !containsString acceptedItemIds item.id &&
  (item.blockedBy acceptedItemIds).isEmpty

structure TgradRoadmap where
  id : String
  target : ProductLine
  items : List RoadmapItem
  invariants : List String
  deriving Repr, Inhabited

def TgradRoadmap.itemIds (roadmap : TgradRoadmap) : List String :=
  roadmap.items.map (fun item => item.id)

def TgradRoadmap.hasItemId (roadmap : TgradRoadmap) (id : String) : Bool :=
  containsString roadmap.itemIds id

def TgradRoadmap.dependenciesKnown (roadmap : TgradRoadmap) : Bool :=
  roadmap.items.all (fun item =>
    item.dependencies.all (fun dep => roadmap.hasItemId dep))

def TgradRoadmap.readyItems (roadmap : TgradRoadmap)
    (acceptedItemIds : List String) : List RoadmapItem :=
  roadmap.items.filter (fun item => item.readyGiven acceptedItemIds)

def TgradRoadmap.blockedItems (roadmap : TgradRoadmap)
    (acceptedItemIds : List String) : List RoadmapItem :=
  roadmap.items.filter (fun item =>
    item.status == .planned && !(item.blockedBy acceptedItemIds).isEmpty)

def TgradRoadmap.itemsForCapability (roadmap : TgradRoadmap)
    (capabilityId : String) : List RoadmapItem :=
  roadmap.items.filter (fun item => containsString item.capabilityIds capabilityId)

def TgradRoadmap.itemsRequiringRepoTransform (roadmap : TgradRoadmap) :
    List RoadmapItem :=
  roadmap.items.filter RoadmapItem.requiresRepoTransform

def roadmapItemLine (acceptedItemIds : List String) (item : RoadmapItem) : String :=
  let deps :=
    if item.dependencies.isEmpty then "none"
    else String.intercalate "," item.dependencies
  let blocked :=
    match item.blockedBy acceptedItemIds with
    | [] => "none"
    | xs => String.intercalate "," xs
  item.id ++ " [" ++ item.horizon.toStr ++ "; " ++ item.kind.toStr ++
    "; " ++ item.status.toStr ++ "; repo=" ++ item.repoTransform.toStr ++
    "; deps=" ++ deps ++ "; blocked-by=" ++ blocked ++ "] " ++ item.title

def roadmapReport (roadmap : TgradRoadmap) (acceptedItemIds : List String) :
    String :=
  let ready := roadmap.readyItems acceptedItemIds
  let blocked := roadmap.blockedItems acceptedItemIds
  "roadmap: " ++ roadmap.id ++
  "\ntarget: " ++ roadmap.target.toStr ++
  "\ndependencies-known: " ++ (if roadmap.dependenciesKnown then "yes" else "no") ++
  "\nready:\n" ++
    (if ready.isEmpty then "- none"
     else String.intercalate "\n" (ready.map (fun item => "- " ++ roadmapItemLine acceptedItemIds item))) ++
  "\nblocked:\n" ++
    (if blocked.isEmpty then "- none"
     else String.intercalate "\n" (blocked.map (fun item => "- " ++ roadmapItemLine acceptedItemIds item)))

/-- Acceptance ties roadmap progress to a product-change packet. -/
structure RoadmapAcceptance where
  id : String
  roadmapItemId : String
  packet : ProductChangePacket
  deriving Repr, Inhabited

def RoadmapAcceptance.validFor (roadmap : TgradRoadmap)
    (acceptance : RoadmapAcceptance) : Bool :=
  acceptance.id != "" &&
  roadmap.hasItemId acceptance.roadmapItemId &&
  acceptance.packet.acceptable

def acceptedRoadmapItemIds (roadmap : TgradRoadmap)
    (acceptances : List RoadmapAcceptance) : List String :=
  acceptances.foldl (fun acc acceptance =>
    if acceptance.validFor roadmap then acc ++ [acceptance.roadmapItemId] else acc) []

def tgradReplacementRoadmap : TgradRoadmap := {
  id := "roadmap.tgrad.full-replacement.v0"
  target := .fullTinygradReplacement
  items := [
    {
      id := "rm.product-spec-v0"
      title := "Accept product description and full-replacement product spec"
      horizon := .foundation
      kind := .productSpec
      status := .planned
      capabilityIds := []
      desiredChanges := []
      dependencies := []
      expectedOutputs := [.productDescription, .productSpec]
      repoTransform := .required
      acceptanceBoundary := "The repo contains checked product description/specification tied to the replacement target."
    },
    {
      id := "rm.capability-map-v0"
      title := "Accept tinygrad compatibility and capability gap map"
      horizon := .foundation
      kind := .capabilityMap
      status := .planned
      capabilityIds := []
      desiredChanges := []
      dependencies := ["rm.product-spec-v0"]
      expectedOutputs := [.compatibilitySpec, .capabilityMap]
      repoTransform := .required
      acceptanceBoundary := "Supported, partial, missing, and deferred tinygrad capabilities are typed and reviewable."
    },
    {
      id := "rm.dtype-runtime-fp16"
      title := "Add fp16 runtime dtype path"
      horizon := .semanticCore
      kind := .implementation
      status := .planned
      capabilityIds := ["cap.dtype.full-runtime"]
      desiredChanges := [addFp16RuntimeDtype]
      dependencies := ["rm.capability-map-v0"]
      expectedOutputs := [.designPacket, .implementationPatch, .verificationEvidence, .releaseCandidate]
      repoTransform := .required
      acceptanceBoundary := "fp16 crosses Python marshalling, FFI, renderer, dispatch, and runtime correctness gates."
    },
    {
      id := "rm.views-general-rangeify"
      title := "Replace view special cases with general rangeify/indexing path"
      horizon := .semanticCore
      kind := .architecture
      status := .planned
      capabilityIds := ["cap.views.general"]
      desiredChanges := [addNegativeStepSlice, replaceViewSpecialCasesWithRangeify]
      dependencies := ["rm.capability-map-v0"]
      expectedOutputs := [.mechanisticModel, .designPacket, .implementationPatch, .verificationEvidence, .performanceEvidence]
      repoTransform := .requiredWithPerformanceEvidence
      acceptanceBoundary := "Movement chains lower through rangeify/indexing while preserving current matmul hot-path behavior or recording a measured non-claim."
    },
    {
      id := "rm.backend-descriptor-v0"
      title := "Introduce backend descriptor and renderer/runtime split"
      horizon := .backendExpansion
      kind := .architecture
      status := .planned
      capabilityIds := ["cap.backends.multi"]
      desiredChanges := [addCudaBackend]
      dependencies := ["rm.capability-map-v0"]
      expectedOutputs := [.mechanisticModel, .designPacket, .implementationPatch, .verificationEvidence]
      repoTransform := .required
      acceptanceBoundary := "Backend-independent kernel intent is separated from Metal renderer/runtime implementation."
    },
    {
      id := "rm.cuda-skeleton-v0"
      title := "Add CUDA backend skeleton with one tiny executable path or honest unsupported rejection"
      horizon := .backendExpansion
      kind := .implementation
      status := .planned
      capabilityIds := ["cap.backends.multi"]
      desiredChanges := [addCudaBackend]
      dependencies := ["rm.backend-descriptor-v0"]
      expectedOutputs := [.implementationPatch, .verificationEvidence, .performanceEvidence, .releaseCandidate]
      repoTransform := .requiredWithPerformanceEvidence
      acceptanceBoundary := "CUDA route has descriptor/runtime/renderer boundary and either one correctness path or typed unsupported-route rejection."
    },
    {
      id := "rm.graph-realize-v0"
      title := "Generalize matmul realization into multi-op graph realization"
      horizon := .compilerRuntime
      kind := .architecture
      status := .planned
      capabilityIds := ["cap.inference.transformer", "cap.scheduler.search", "cap.training.autograd"]
      desiredChanges := [growTowardTransformerInference]
      dependencies := ["rm.dtype-runtime-fp16", "rm.views-general-rangeify"]
      expectedOutputs := [.mechanisticModel, .designPacket, .implementationPatch, .verificationEvidence]
      repoTransform := .required
      acceptanceBoundary := "A UOp DAG realization route exists beyond one matmul special path."
    },
    {
      id := "rm.buffer-lifetime-v0"
      title := "Add buffer lifetime plan for multi-kernel execution"
      horizon := .compilerRuntime
      kind := .implementation
      status := .planned
      capabilityIds := ["cap.inference.transformer", "cap.training.autograd"]
      desiredChanges := [growTowardTransformerInference]
      dependencies := ["rm.graph-realize-v0"]
      expectedOutputs := [.designPacket, .implementationPatch, .verificationEvidence]
      repoTransform := .required
      acceptanceBoundary := "Temporary buffers, output ownership, and reuse/lifetime rules are explicit for multi-kernel graphs."
    },
    {
      id := "rm.schedule-search-v0"
      title := "Introduce typed schedule search states/actions/scoring"
      horizon := .compilerRuntime
      kind := .architecture
      status := .planned
      capabilityIds := ["cap.scheduler.search"]
      desiredChanges := [{ verb := .add, subject := .scheduleSearch, constraints := [.preferPureLeanModel, .performanceRelevant] }]
      dependencies := ["rm.graph-realize-v0"]
      expectedOutputs := [.mechanisticModel, .leanQueryModel, .performanceEvidence]
      repoTransform := .requiredWithPerformanceEvidence
      acceptanceBoundary := "Search can enumerate schedule states and compare routes against measured or declared cost signals."
    },
    {
      id := "rm.tiny-transformer-inference-v0"
      title := "Run one bounded transformer-like inference graph"
      horizon := .inference
      kind := .validation
      status := .planned
      capabilityIds := ["cap.inference.transformer"]
      desiredChanges := [growTowardTransformerInference]
      dependencies := ["rm.graph-realize-v0", "rm.buffer-lifetime-v0"]
      expectedOutputs := [.implementationPatch, .verificationEvidence, .performanceEvidence, .releaseCandidate]
      repoTransform := .requiredWithPerformanceEvidence
      acceptanceBoundary := "A small static transformer-like graph executes end-to-end with correctness evidence and explicit performance boundary."
    },
    {
      id := "rm.autograd-core-v0"
      title := "Model autograd graph construction and backward lowering"
      horizon := .training
      kind := .architecture
      status := .planned
      capabilityIds := ["cap.training.autograd"]
      desiredChanges := []
      dependencies := ["rm.graph-realize-v0"]
      expectedOutputs := [.productSpec, .mechanisticModel, .designPacket, .verificationEvidence]
      repoTransform := .required
      acceptanceBoundary := "Forward graph, backward graph, gradient accumulation, and optimizer boundary are typed before training runtime claims."
    },
    {
      id := "rm.training-loop-v0"
      title := "Run one tiny training loop"
      horizon := .training
      kind := .validation
      status := .planned
      capabilityIds := ["cap.training.autograd"]
      desiredChanges := []
      dependencies := ["rm.autograd-core-v0", "rm.buffer-lifetime-v0"]
      expectedOutputs := [.implementationPatch, .verificationEvidence, .performanceEvidence, .releaseCandidate]
      repoTransform := .requiredWithPerformanceEvidence
      acceptanceBoundary := "A tiny training loop updates parameters through typed autograd and runtime state with correctness evidence."
    },
    {
      id := "rm.tinygrad-compatibility-scaleout"
      title := "Scale compatibility coverage across tinygrad API/op surface"
      horizon := .parity
      kind := .release
      status := .planned
      capabilityIds := [
        "cap.dtype.full-runtime",
        "cap.views.general",
        "cap.backends.multi",
        "cap.inference.transformer",
        "cap.training.autograd",
        "cap.scheduler.search"
      ]
      desiredChanges := []
      dependencies := [
        "rm.tiny-transformer-inference-v0",
        "rm.training-loop-v0",
        "rm.cuda-skeleton-v0",
        "rm.schedule-search-v0"
      ]
      expectedOutputs := [.compatibilitySpec, .verificationEvidence, .performanceEvidence, .releaseCandidate]
      repoTransform := .requiredWithPerformanceEvidence
      acceptanceBoundary := "Compatibility matrix and replacement claims expand only as evidence-backed accepted repo transforms."
    }
  ]
  invariants := [
    "A roadmap item is planned intent until a valid RoadmapAcceptance ties it to an acceptable ProductChangePacket.",
    "Every dependency names another roadmap item.",
    "Every product-changing roadmap item requires a repo-ref transformation.",
    "Performance-sensitive roadmap items require performance evidence or an explicit non-performance claim."
  ]
}

def readyRoadmapNow : List RoadmapItem :=
  tgradReplacementRoadmap.readyItems []

def readyRoadmapAfterFoundation : List RoadmapItem :=
  tgradReplacementRoadmap.readyItems ["rm.product-spec-v0", "rm.capability-map-v0"]

/-! ## Smoke theorems for the factory control plane. -/

theorem tgradFactory_has_functions :
    (tgradFactory.functions).isEmpty = false := by
  native_decide

theorem tgradFactory_partitions_registry :
    tgradFactory.partitionsRegistry = true := by
  native_decide

theorem product_design_is_first_class :
    productDesignFunctions.isEmpty = false := by
  native_decide

theorem full_replacement_has_open_work :
    capabilitiesNeedingWork.isEmpty = false := by
  native_decide

theorem backend_change_touches_product_design :
    changeTouchesProductDesign addCudaBackend = true := by
  native_decide

theorem backend_change_requires_repo_ref_transform :
    changeRequiresRepoRefTransform addCudaBackend = true := by
  native_decide

theorem sample_fp16_work_order_admissible :
    sampleFp16WorkOrder.admissible = true := by
  native_decide

theorem backend_work_order_without_repo_not_admissible :
    sampleBackendWorkOrderWithoutRepo.admissible = false := by
  native_decide

theorem roadmap_dependencies_known :
    tgradReplacementRoadmap.dependenciesKnown = true := by
  native_decide

theorem roadmap_has_ready_foundation_item :
    readyRoadmapNow.isEmpty = false := by
  native_decide

theorem roadmap_foundation_unblocks_semantic_core :
    (readyRoadmapAfterFoundation.map (fun item => item.id)).any
      (fun id => id == "rm.dtype-runtime-fp16") = true := by
  native_decide

theorem roadmap_requires_repo_transforms :
    (tgradReplacementRoadmap.itemsRequiringRepoTransform).isEmpty = false := by
  native_decide

end Model
end Tgrad

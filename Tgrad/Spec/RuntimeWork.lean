import Tgrad.Ontology
import Tgrad.Spec.Architecture
import Tgrad.Pipeline
import Tgrad.Runtime.MetalProgram
import Tgrad.Renderer.MatmulTc

/-! # Tgrad.Spec.RuntimeWork — work performed by the codebase

“The codebase does work” has three meanings in Tgrad:

* product work transforms user tensors into results;
* verification work transforms candidate revisions and scenarios into evidence;
* specification work transforms the checked schema into findings and work queues.

These are repeatable capabilities, not executions and not roadmap items. A
particular run is an observation of a capability. Repository evolution belongs
to `Tgrad.Spec.Work` and is deliberately a different type.
-/

namespace Tgrad.Spec.Runtime

structure WorkId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

def workId (value : String) : WorkId := { value }

instance : ToString WorkId where
  toString id := id.value

inductive Realm where
  | product
  | verification
  | specification
  deriving DecidableEq, BEq, Repr, Inhabited

inductive Artifact where
  | externalTensor
  | hostBytes
  | tensorHandle
  | uopGraph
  | view
  | indexExpr
  | launchPlan
  | kernelDecl
  | metalSource
  | compiledProgram
  | deviceBuffer
  | tensorResult
  | candidateRevision
  | scenario
  | executionObservation
  | validationEvidence
  | findingSet
  | evolutionGraph
  | report
  deriving DecidableEq, BEq, Repr, Inhabited

inductive Verb where
  | ingest
  | represent
  | normalize
  | select
  | lower
  | render
  | compile
  | allocate
  | execute
  | synchronize
  | materialize
  | observe
  | validate
  | query
  deriving DecidableEq, BEq, Repr, Inhabited

inductive Resource where
  | pythonHeap
  | tensorRegistry
  | hostMemory
  | metalBufferPool
  | metalLibraryCache
  | metalPipelineCache
  | metalCommandQueue
  | metalGpu
  | sourceTree
  | leanBuildTree
  | tmpNamespace
  | evidenceStore
  deriving DecidableEq, BEq, Repr, Inhabited

/-- The operational status of a repeatable capability.

`referenceOnly` means the implementation exists to compare against, but must
not be confused with generated product behavior. `bypassed` means the claimed
abstraction exists but the path does not use it. -/
inductive CapabilityState where
  | loadBearing
  | bounded
  | referenceOnly
  | bypassed
  | missing
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ScaleDriver where
  | fixedHostOverhead
  | inputBytes
  | outputElements
  | reductionElements
  | dispatchCount
  | scenarioCount
  | evidenceCount
  deriving DecidableEq, BEq, Repr, Inhabited

structure WorkUnit where
  id : WorkId
  realm : Realm
  verb : Verb
  consumes : List Artifact
  produces : List Artifact
  component : Component
  resources : List Resource
  scaleDrivers : List ScaleDriver
  implementation : Epistemic (List String)
  support : Epistemic CapabilityState
  observationMethod : String
  failureSurface : String
  deriving Repr, Inhabited

private def implemented
    (id name : String) (realm : Realm) (verb : Verb)
    (consumes produces : List Artifact) (component : Component)
    (resources : List Resource) (scaleDrivers : List ScaleDriver)
    (files : List String) (state : CapabilityState)
    (observationMethod failureSurface : String) : WorkUnit :=
  { id := workId id, realm, verb, consumes, produces, component, resources,
    scaleDrivers,
    implementation := .confirmed files s!"traced implementation of {name}",
    support := .confirmed state s!"current support classification for {name}",
    observationMethod, failureSurface }

private def absent
    (id name : String) (realm : Realm) (verb : Verb)
    (consumes produces : List Artifact) (component : Component)
    (resources : List Resource) (scaleDrivers : List ScaleDriver)
    (resolveBy observationMethod failureSurface : String) : WorkUnit :=
  { id := workId id, realm, verb, consumes, produces, component, resources,
    scaleDrivers,
    implementation := .unknown s!"no load-bearing implementation of {name}" resolveBy,
    support := .confirmed .missing s!"{name} is an explicit capability gap",
    observationMethod, failureSurface }

def workUnits : List WorkUnit :=
  [ implemented "product.ingest-tensor" "tensor ingestion"
      .product .ingest [.externalTensor] [.hostBytes, .tensorHandle]
      .pythonAuthoring [.pythonHeap, .hostMemory, .metalBufferPool]
      [.inputBytes] ["python/tgrad.py", "Tgrad/PythonFFI.lean"] .bounded
      "construct tensors from numpy/bytes and inspect shape, dtype, and allocation"
      "invalid shape/dtype/payload must reject before device access",
    implemented "product.record-view" "lazy movement representation"
      .product .represent [.tensorHandle] [.uopGraph]
      .tensorIr [.tensorRegistry] [.fixedHostOverhead]
      ["Tgrad/Tensor.lean", "Tgrad/PythonFFI.lean"] .bounded
      "query Tensor.uop and movement constructors"
      "unsupported or malformed view chains can remain latent until realization",
    implemented "product.compose-view" "view composition"
      .product .normalize [.uopGraph] [.view, .indexExpr]
      .viewAlgebra [.hostMemory] [.fixedHostOverhead]
      ["Tgrad/Schedule/View.lean"] .bounded
      "run View regressions and inspect computed shape/strides/offset/index"
      "invalid reshape/permute/expand/slice returns none or must be rejected upstream",
    implemented "product.rangeify" "movement elimination"
      .product .normalize [.uopGraph] [.uopGraph]
      .scheduler [.hostMemory] [.fixedHostOverhead]
      ["Tgrad/Schedule/Rangeify.lean", "Tgrad/Pipeline.lean"] .bounded
      "assert movement nodes are eliminated and inspect rangeify trace"
      "unsupported UOp forms can remain outside the bounded movement algebra",
    implemented "product.select-matmul-route" "matmul route selection"
      .product .select [.uopGraph] [.launchPlan]
      .scheduler [.hostMemory] [.fixedHostOverhead]
      ["python/tgrad.py", "Tgrad/PythonFFI.lean", "Tgrad/Codegen/Opt/Heuristic.lean"] .bounded
      "record chosen sentinel/TC/scalar/view route and dispatch geometry"
      "duplicated eligibility rules and tables can disagree",
    implemented "product.lower-scalar-matmul" "scalar matmul lowering"
      .product .lower [.uopGraph, .view, .indexExpr] [.kernelDecl]
      .renderer [.hostMemory] [.outputElements, .reductionElements]
      ["Tgrad/Renderer/MatmulScalar.lean", "Tgrad/Pipeline.lean"] .bounded
      "render and execute general and view-indexed scalar kernels"
      "correct fallback is intentionally very slow and has no broad performance contract",
    implemented "product.lower-tc-matmul" "tensor-core matmul lowering"
      .product .lower [.uopGraph, .launchPlan] [.kernelDecl]
      .renderer [.hostMemory] [.outputElements, .reductionElements]
      ["Tgrad/Renderer/MatmulTc.lean", "Tgrad/Pipeline.lean"] .loadBearing
      "render and execute generated TC kernels; differentially compare every sentinel against the independent captured oracle"
      "the generated route is authoritative for its explicit aligned domain; general graph-to-kernel lowering remains outside this matmul-specialized capability",
    implemented "product.render-metal" "Metal source rendering"
      .product .render [.kernelDecl] [.metalSource]
      .renderer [.hostMemory] [.fixedHostOverhead]
      ["Tgrad/Renderer/Metal.lean"] .bounded
      "call renderKernel and compile the resulting source"
      "renderer is partial and malformed typed/string nodes can surface only at Metal compile",
    implemented "product.compile-metal" "Metal compilation and cache lookup"
      .product .compile [.metalSource] [.compiledProgram]
      .metalRuntime [.metalLibraryCache, .metalPipelineCache]
      [.fixedHostOverhead] ["Tgrad/PythonFFI.lean", "c/metal_alloc.m"] .bounded
      "compile source, resolve function, and inspect return code/cache behavior"
      "Metal diagnostics are compressed into integer errors and caches have incomplete lifecycle",
    implemented "product.allocate-output" "device output allocation"
      .product .allocate [.launchPlan] [.deviceBuffer]
      .metalRuntime [.metalBufferPool] [.outputElements]
      ["Tgrad/PythonFFI.lean", "Tgrad/Runtime/MetalAllocator.lean", "c/metal_alloc.m"] .bounded
      "compare requested bytes with buffer length and output shape"
      "some lower-level APIs still trust caller-supplied size and ownership",
    implemented "product.dispatch-metal" "Metal dispatch"
      .product .execute [.compiledProgram, .deviceBuffer, .launchPlan] [.deviceBuffer]
      .metalRuntime [.metalCommandQueue, .metalGpu, .metalPipelineCache]
      [.dispatchCount, .outputElements, .reductionElements]
      ["Tgrad/Runtime/MetalProgram.lean", "c/metal_alloc.m"] .bounded
      "dispatch generated geometry and inspect command-buffer completion"
      "single global queue/cache, no timeout, and weak diagnostics constrain safe use",
    implemented "product.synchronize" "device synchronization"
      .product .synchronize [.deviceBuffer] [.tensorResult]
      .metalRuntime [.metalCommandQueue, .metalGpu]
      [.dispatchCount] ["c/metal_alloc.m"] .loadBearing
      "confirm waitUntilCompleted occurs before the FFI call returns"
      "command errors are not surfaced with full diagnostic context",
    implemented "product.readback-buffer" "contiguous buffer readback"
      .product .materialize [.tensorResult] [.hostBytes]
      .pythonAuthoring [.hostMemory, .metalBufferPool]
      [.inputBytes] ["python/tgrad.py", "Tgrad/Runtime/Buffer.lean"] .bounded
      "compare contiguous numpy/to_bytes results against expected bf16 bytes"
      "contiguous tensors read directly; movement views must first pass bounded materialization",
    implemented "product.materialize-view" "view materialization"
      .product .materialize [.view, .indexExpr] [.tensorResult, .hostBytes]
      .renderer [.hostMemory, .metalBufferPool, .metalGpu]
      [.inputBytes, .dispatchCount]
      ["Tgrad/Pipeline.lean", "Tgrad/PythonFFI.lean", "python/tgrad.py"] .bounded
      "materialize transpose, slice, reshape, expand, and supported chains; compare exact bf16 bytes and numpy shape"
      "unsupported or empty views reject explicitly; each readback currently compiles, registers, and dispatches a fresh copy",

    implemented "verify.lean-build" "Lean build validation"
      .verification .validate [.candidateRevision] [.validationEvidence]
      .gateHarness [.sourceTree, .leanBuildTree] [.fixedHostOverhead]
      ["lakefile.lean", "lean-toolchain"] .loadBearing
      "lake build the product and specification roots"
      "mechanical correctness does not imply runtime or numerical correctness",
    implemented "verify.unit-tests" "Lean assertion suite"
      .verification .validate [.candidateRevision, .scenario] [.validationEvidence]
      .gateHarness [.sourceTree, .leanBuildTree] [.scenarioCount]
      ["Tests.lean"] .bounded
      "run tgrad-tests and require every named assertion"
      "coverage is bounded and many historical gates are structural rather than semantic",
    implemented "verify.codegen-differential" "captured/generated codegen differential"
      .verification .validate [.candidateRevision, .scenario] [.executionObservation, .validationEvidence]
      .gateHarness [.sourceTree, .tmpNamespace, .metalGpu, .evidenceStore]
      [.scenarioCount, .dispatchCount]
      ["Main.lean", "scripts/differential_codegen.sh"] .loadBearing
      "run all sentinels in one process per shape; require different sources and bit-identical captured/generated outputs"
      "L12 C3 now requires the executable differential; store non-aliasing and right-placement remain complementary obligations",
    implemented "verify.numpy-differential" "numpy numerical differential"
      .verification .validate [.candidateRevision, .scenario] [.executionObservation, .validationEvidence]
      .gateHarness [.sourceTree, .metalGpu] [.scenarioCount, .dispatchCount]
      ["python/tgrad_bench.py", "scripts/gates/L13_C.sh", "scripts/gates/L14_C.sh"] .bounded
      "run supported shapes/views against a numpy bf16 reference"
      "tolerances and sampled domains do not establish complete semantics",
    implemented "verify.performance" "performance comparison"
      .verification .observe [.candidateRevision, .scenario] [.executionObservation, .validationEvidence]
      .evidenceStore [.tmpNamespace, .metalGpu, .evidenceStore]
      [.scenarioCount, .dispatchCount]
      ["python/tgrad_bench.py", "scripts/gates/L7.sh", "scripts/gates/L11.sh"] .bypassed
      "interleave both runtimes in one session across the same timed boundary; retain raw paired samples and derive any threshold from observed within-run and between-run variance"
      "the current gate compares a live generated route with a frozen asymmetric baseline, so no ratio threshold is admissible",
    implemented "verify.performance-repeatability" "performance repeatability diagnosis"
      .verification .observe [.candidateRevision, .scenario] [.executionObservation, .findingSet]
      .evidenceStore [.tmpNamespace, .metalGpu, .evidenceStore]
      [.scenarioCount, .dispatchCount]
      ["python/tgrad_bench.py", "scripts/gates/L11.sh", "scripts/gates/L12.sh"] .bounded
      "repeat an identical benchmark configuration serially and report raw distributions, miss counts, and within-run and between-run variance without promoting a parity verdict"
      "it can falsify a gate's repeatability, but a frozen tinygrad denominator means it cannot validate relative performance",
    implemented "verify.evidence-audit" "gate evidence must stay untracked"
      .verification .observe [.candidateRevision, .validationEvidence] [.findingSet, .report]
      .evidenceStore [.sourceTree, .evidenceStore] [.evidenceCount]
      ["scripts/dev/gate_evidence_not_tracked.py"] .loadBearing
      "fail if fixtures/gate_evidence/ is tracked by git or any *.json under it is staged"
      "re-tracking evidence makes umbrella [[ -f ]] checks vacuous again",
    implemented "verify.evidence-integrity" "evidence hash integrity validation"
      .verification .validate [.candidateRevision, .validationEvidence] [.validationEvidence]
      .evidenceStore [.sourceTree, .evidenceStore] [.evidenceCount]
      ["scripts/lib/checks.sh", "fixtures/gate_evidence"] .bypassed
      "recompute every recorded hash against runtime evidence produced in this tree"
      "current checks confirm presence and schema keys but do not recompute recorded hashes",

    implemented "spec.query-findings" "finding query"
      .specification .query [.findingSet] [.report]
      .specification [.leanBuildTree] [.evidenceCount]
      ["Tgrad/Spec/Findings.lean", "Tgrad/Spec/Work.lean"] .loadBearing
      "run tgrad-spec and inspect tracked open findings"
      "the registry is intentionally scoped and not yet the complete audit backlog",
    implemented "spec.compute-evolution-frontier" "evolution frontier computation"
      .specification .query [.evolutionGraph] [.report]
      .specification [.leanBuildTree, .sourceTree] [.evidenceCount]
      ["Tgrad/Spec/Work.lean"] .loadBearing
      "run tgrad-spec and compare ready work with active write sets"
      "file paths and live writers must still be re-observed when work starts",
    implemented "spec.check-growth-loop" "growth-loop integrity checks"
      .specification .validate [.findingSet, .evolutionGraph] [.validationEvidence, .report]
      .specification [.leanBuildTree] [.evidenceCount]
      ["Tgrad/Spec/Growth.lean", "Tgrad/Spec/Work.lean"] .loadBearing
      "compile native_decide checks over runtime, finding, evolution, and promotion links"
      "the checks establish reference integrity, not the truth of prose evidence" ]

def workIds : List WorkId := workUnits.map (·.id)

def uniqueWorkIds : Bool := workIds.eraseDups.length == workIds.length

def externalArtifacts : List Artifact :=
  [.externalTensor, .candidateRevision, .scenario, .findingSet, .evolutionGraph]

def producedArtifacts : List Artifact :=
  (workUnits.flatMap (·.produces)).eraseDups

def artifactFlowClosed : Bool :=
  workUnits.all (fun unit =>
    unit.consumes.all (fun artifact =>
      externalArtifacts.contains artifact || producedArtifacts.contains artifact))

def WorkUnit.isState (unit : WorkUnit) (state : CapabilityState) : Bool :=
  unit.support.value? == some state

def unitsInRealm (realm : Realm) : List WorkUnit :=
  workUnits.filter (fun unit => unit.realm == realm)

def unitsInState (state : CapabilityState) : List WorkUnit :=
  workUnits.filter (fun unit => unit.isState state)

def missingOrBypassed : List WorkUnit :=
  workUnits.filter (fun unit =>
    unit.isState .missing || unit.isState .bypassed || unit.isState .referenceOnly)

def workUnitFor? (id : WorkId) : Option WorkUnit :=
  workUnits.find? (fun unit => unit.id == id)

def allClaimsHaveUpgradePaths : Bool :=
  workUnits.all (fun unit =>
    unit.implementation.hasUpgradePath && unit.support.hasUpgradePath &&
    !unit.observationMethod.isEmpty && !unit.failureSurface.isEmpty)

theorem runtime_work_graph_well_formed :
    uniqueWorkIds && artifactFlowClosed && allClaimsHaveUpgradePaths = true := by
  native_decide

theorem work_by_the_codebase_has_three_realms :
    (unitsInRealm .product).length > 0 &&
    (unitsInRealm .verification).length > 0 &&
    (unitsInRealm .specification).length > 0 := by
  native_decide

theorem view_materialization_candidate_is_bounded :
    (workUnitFor? (workId "product.materialize-view")).map
      (fun unit => unit.isState .bounded) = some true := by
  native_decide

theorem tensor_core_lowering_is_load_bearing :
    (workUnitFor? (workId "product.lower-tc-matmul")).map
      (fun unit => unit.isState .loadBearing) = some true := by
  native_decide

theorem performance_and_evidence_are_not_promoted :
    (workUnitFor? (workId "verify.performance")).map
        (fun unit => unit.isState .bypassed) = some true &&
    (workUnitFor? (workId "verify.evidence-integrity")).map
        (fun unit => unit.isState .bypassed) = some true := by
  native_decide

theorem repeatability_diagnosis_does_not_promote_performance :
    (workUnitFor? (workId "verify.performance-repeatability")).map
        (fun unit => unit.isState .bounded) = some true &&
    (workUnitFor? (workId "verify.performance")).map
        (fun unit => unit.isState .bypassed) = some true := by
  native_decide

theorem evidence_audit_is_load_bearing_but_enforcement_is_bypassed :
    (workUnitFor? (workId "verify.evidence-audit")).map
        (fun unit => unit.isState .loadBearing) = some true &&
    (workUnitFor? (workId "verify.evidence-integrity")).map
        (fun unit => unit.isState .bypassed) = some true := by
  native_decide

/-! Product-symbol pins: these make drift from the described computational
surface visible at compile time. -/

#check (Tgrad.Tensor.transpose : Tgrad.Tensor -> Tgrad.Tensor)
#check (Tgrad.Schedule.View.indexOf : Tgrad.Schedule.View -> List Tgrad.UOp -> Tgrad.UOp)
#check (Tgrad.Schedule.Rangeify.rangeify : Tgrad.UOp -> Tgrad.UOp)
#check (Tgrad.Pipeline.materializeView :
  Tgrad.Tensor -> IO (Except Tgrad.PipelineError Tgrad.Tensor))
#check (Tgrad.Renderer.Metal.renderKernel : Tgrad.Renderer.Metal.KernelDecl -> String)
#check (Tgrad.Renderer.Metal.tcMatmulKernelDeclManualLoadWide :
  Nat -> Nat -> Nat -> Except Tgrad.Renderer.Metal.CodegenError Tgrad.Renderer.Metal.KernelDecl)
#check (Tgrad.Runtime.Metal.metalCompile : String -> IO UInt64)
#check (Tgrad.Runtime.Metal.metalDispatch :
  UInt64 -> String -> Array UInt64 -> USize -> USize -> USize -> USize -> USize -> USize -> IO UInt32)

end Tgrad.Spec.Runtime

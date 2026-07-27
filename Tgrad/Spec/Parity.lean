import Tgrad.Spec.Epistemic
import Tgrad.Spec.ParityTarget
import Tgrad.Dtype
import Tgrad.UOp
import Tgrad.Tensor
import Tgrad.Codegen.Opt.Apply
import Tgrad.Schedule.View
import Tgrad.Renderer.Metal
import Tgrad.Pipeline

/-! # Tgrad.Spec.Parity — a versioned destination and a compiler from gaps to work

Compatibility is always relative to an upstream revision, a declared profile,
and evidence produced from a particular Tgrad tree.  This module types those
stable relationships.  It deliberately does not claim that the current tree is
at parity: the upstream revision and generated manifests have not been pinned.

The long-horizon parity program is not the live execution ledger.  A
`ProgramTemplate` is a desired delta.  It becomes an executable
`Tgrad.Spec.Work.WorkItem` only after an agent has resolved its exact base tree,
write set, resources, oracle, validators, falsifiers, and recovery action.
-/

namespace Tgrad.Spec.Parity

structure UpstreamRef where
  repository : String
  revision : String
  observedAt : String
  sourceManifestHash : String
  apiManifestHash : String
  testManifestHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

def UpstreamRef.wellFormed (ref : UpstreamRef) : Bool :=
  !ref.repository.isEmpty &&
  !ref.revision.isEmpty &&
  !ref.observedAt.isEmpty &&
  !ref.sourceManifestHash.isEmpty &&
  !ref.apiManifestHash.isEmpty &&
  !ref.testManifestHash.isEmpty

/-- The exact foreign revision and generated-manifest identities selected by
the owner after reviewing the extracted candidate. -/
def pinnedUpstream : UpstreamRef :=
  { repository := ParityTarget.repository,
    revision := ParityTarget.revision,
    observedAt := ParityTarget.committedAt,
    sourceManifestHash := ParityTarget.sourceManifestSha256,
    apiManifestHash := ParityTarget.apiManifestSha256,
    testManifestHash := ParityTarget.testManifestSha256 }

/-- The reviewed foreign manifest pins the first convergence target.  This
confirms the denominator only; it does not claim that any Tgrad requirement is
conformant. -/
def targetUpstream : Epistemic UpstreamRef :=
  .confirmed pinnedUpstream
    s!"{ParityTarget.manifestPath}; content sha256 {ParityTarget.manifestContentSha256}; extractor sha256 {ParityTarget.extractorSha256}"

inductive Profile where
  | semanticCore
  | publicApi
  | metal
  | portable
  | ecosystem
  | allBackends
  deriving DecidableEq, BEq, Repr, Inhabited

inductive Domain where
  | scalarSemantics
  | dtypeSystem
  | tensorSurface
  | shapeAndView
  | uopIr
  | rewriteSystem
  | lazySchedule
  | lowering
  | optimization
  | renderer
  | runtime
  | effectsAndAliasing
  | autograd
  | jit
  | nnAndState
  | multiDevice
  | interoperability
  | workloads
  deriving DecidableEq, BEq, Repr, Inhabited

inductive Dimension where
  | api
  | semantic
  | numerical
  | gradient
  | compiler
  | runtime
  | ecosystem
  | backend
  | performance
  deriving DecidableEq, BEq, Repr, Inhabited

inductive OracleOrigin where
  | upstreamSuite
  | upstreamRuntime
  | independentFramework
  | mathematicalLaw
  | internalDifferential
  | selfReferential
  deriving DecidableEq, BEq, Repr, Inhabited

inductive EvidenceKind where
  | manifestDiff
  | compileCheck
  | unit
  | property
  | metamorphic
  | differential
  | upstreamTest
  | backendMatrix
  | workload
  | performanceDistribution
  | provenanceAudit
  deriving DecidableEq, BEq, Repr, Inhabited

structure EvidenceRef where
  id : String
  kind : EvidenceKind
  origin : OracleOrigin
  upstreamRevision : String
  subjectTree : String
  verifierTree : String
  adapterHash : String
  equivalenceRelationVersion : String
  environmentId : String
  environmentHash : String
  scenarioManifestHash : String
  artifactHash : String
  requirementIds : List String
  validatorId : String
  deriving DecidableEq, BEq, Repr, Inhabited

inductive CalibrationOutcome where
  | validatorRejectedMutant
  | mutantSurvived
  | indeterminate
  deriving DecidableEq, BEq, Repr, Inhabited

structure CalibrationEvidence where
  mutantTree : String
  faultModel : String
  scenarioManifestHash : String
  artifactHash : String
  outcome : CalibrationOutcome
  deriving DecidableEq, BEq, Repr, Inhabited

def CalibrationEvidence.establishesSensitivity
    (calibration : CalibrationEvidence) : Bool :=
  !calibration.mutantTree.isEmpty &&
  !calibration.faultModel.isEmpty &&
  !calibration.scenarioManifestHash.isEmpty &&
  !calibration.artifactHash.isEmpty &&
  calibration.outcome == .validatorRejectedMutant

structure ValidatorRef where
  id : String
  verifierTree : String
  version : String
  dimensions : List Dimension
  calibrations : List CalibrationEvidence
  deriving DecidableEq, BEq, Repr, Inhabited

def ValidatorRef.calibrated (validator : ValidatorRef) : Bool :=
  !validator.id.isEmpty &&
  !validator.verifierTree.isEmpty &&
  !validator.version.isEmpty &&
  !validator.dimensions.isEmpty &&
  !validator.calibrations.isEmpty &&
  validator.calibrations.all CalibrationEvidence.establishesSensitivity

def EvidenceRef.wellFormed (evidence : EvidenceRef) : Bool :=
  !evidence.id.isEmpty &&
  !evidence.upstreamRevision.isEmpty &&
  !evidence.subjectTree.isEmpty &&
  !evidence.verifierTree.isEmpty &&
  !evidence.adapterHash.isEmpty &&
  !evidence.equivalenceRelationVersion.isEmpty &&
  !evidence.environmentId.isEmpty &&
  !evidence.environmentHash.isEmpty &&
  !evidence.scenarioManifestHash.isEmpty &&
  !evidence.artifactHash.isEmpty &&
  !evidence.requirementIds.isEmpty &&
  !evidence.validatorId.isEmpty

/-- Self-reference may support debugging, but cannot promote a compatibility
claim.  Every promoting check must also have been falsified on purpose. -/
def EvidenceRef.canPromote
    (validators : List ValidatorRef) (dimensions : List Dimension)
    (evidence : EvidenceRef) : Bool :=
  evidence.wellFormed &&
  evidence.origin != .selfReferential &&
  validators.any fun validator =>
    validator.id == evidence.validatorId &&
    validator.verifierTree == evidence.verifierTree &&
    dimensions.all validator.dimensions.contains &&
    validator.calibrated

inductive CoverageState where
  | absent
  | scaffold
  | bounded
  | conformant
  | drifted
  | blockedByHardware
  deriving DecidableEq, BEq, Repr, Inhabited

structure Requirement where
  id : String
  domains : List Domain
  dimensions : List Dimension
  profiles : List Profile
  upstreamSymbols : List String
  upstreamTests : List String
  requiredEnvironments : List String
  equivalenceRelation : String
  deriving Repr, Inhabited

def Requirement.wellFormed (requirement : Requirement) : Bool :=
  !requirement.id.isEmpty &&
  !requirement.domains.isEmpty &&
  !requirement.dimensions.isEmpty &&
  (if requirement.dimensions.contains .performance
   then requirement.dimensions == [.performance]
   else true) &&
  !requirement.profiles.isEmpty &&
  (!requirement.upstreamSymbols.isEmpty || !requirement.upstreamTests.isEmpty) &&
  !requirement.requiredEnvironments.isEmpty &&
  !requirement.equivalenceRelation.isEmpty

def Requirement.isCompatibility (requirement : Requirement) : Bool :=
  requirement.dimensions.all fun dimension => dimension != .performance

def Requirement.isPerformance (requirement : Requirement) : Bool :=
  requirement.dimensions.contains .performance

structure CoverageCell where
  requirementId : String
  state : Epistemic CoverageState
  evidence : List EvidenceRef
  gap : String
  deriving Repr, Inhabited

def CoverageCell.promotable
    (validators : List ValidatorRef) (requirement : Requirement)
    (cell : CoverageCell) : Bool :=
  match cell.state with
  | .confirmed .conformant _ =>
      !cell.requirementId.isEmpty &&
      !cell.evidence.isEmpty &&
      (cell.evidence.all fun evidence =>
          evidence.requirementIds.contains requirement.id &&
          evidence.equivalenceRelationVersion == requirement.equivalenceRelation &&
          EvidenceRef.canPromote validators requirement.dimensions evidence) &&
      requirement.requiredEnvironments.all fun environment =>
        cell.evidence.any fun evidence => evidence.environmentId == environment
  | _ => false

def compatibilityDimensions : List Dimension :=
  [.api, .semantic, .numerical, .gradient, .compiler, .runtime,
   .ecosystem, .backend]

/-- Performance is deliberately a separate claim.  A compatible implementation
may be slow; a fast implementation may be incompatible. -/
def optimizationDimensions : List Dimension := [.performance]

structure Contract where
  upstream : UpstreamRef
  subjectTree : String
  profile : Profile
  requirements : List Requirement
  cells : List CoverageCell
  validators : List ValidatorRef
  deriving Repr, Inhabited

def Contract.requiredForProfile (contract : Contract) : List Requirement :=
  contract.requirements.filter fun requirement =>
    requirement.profiles.contains contract.profile &&
    requirement.isCompatibility

def Contract.performanceForProfile (contract : Contract) : List Requirement :=
  contract.requirements.filter fun requirement =>
    requirement.profiles.contains contract.profile &&
    requirement.isPerformance

def Contract.hasPromotableCell
    (contract : Contract) (requirement : Requirement) : Bool :=
  contract.cells.any fun cell =>
    cell.requirementId == requirement.id &&
    CoverageCell.promotable contract.validators requirement cell &&
    cell.evidence.all fun evidence =>
      evidence.upstreamRevision == contract.upstream.revision &&
      evidence.subjectTree == contract.subjectTree

def Contract.structurallyWellFormed (contract : Contract) : Bool :=
  let requirementIds := contract.requirements.map (·.id)
  let validatorIds := contract.validators.map (·.id)
  let cellIds := contract.cells.map (·.requirementId)
  !contract.subjectTree.isEmpty &&
  requirementIds.eraseDups.length == requirementIds.length &&
  validatorIds.eraseDups.length == validatorIds.length &&
  cellIds.eraseDups.length == cellIds.length &&
  contract.requirements.all Requirement.wellFormed &&
  contract.cells.all fun cell => requirementIds.contains cell.requirementId

/-- A compatibility claim is the product of required domains and required
dimensions, not a scalar percentage or a hand-authored milestone. -/
def Contract.complete (contract : Contract) : Bool :=
  contract.upstream.wellFormed &&
  contract.structurallyWellFormed &&
  !(contract.requiredForProfile).isEmpty &&
  (contract.requiredForProfile).all contract.hasPromotableCell

/-- Optimization qualification is queried independently from compatibility. -/
def Contract.performanceComplete (contract : Contract) : Bool :=
  contract.upstream.wellFormed &&
  contract.structurallyWellFormed &&
  !(contract.performanceForProfile).isEmpty &&
  (contract.performanceForProfile).all contract.hasPromotableCell

/-- Promotion obligations are derived from affected requirements; a candidate
or certificate does not get to choose a weaker obligation set. -/
def Contract.obligationsFor
    (contract : Contract) (requirementIds : List String) : List Dimension :=
  ((contract.requirements.filter fun requirement =>
      requirementIds.contains requirement.id).flatMap (·.dimensions)).eraseDups

/-! ## Generated target requirements

The names below come only from `ParityTarget`, which is rendered from the
foreign JSON manifest.  The category-to-domain/profile/equivalence mapping is
local policy and remains reviewable here; the denominator itself is not typed
by hand. -/

private def apiProfiles : List Profile :=
  [.publicApi, .metal, .portable, .ecosystem, .allBackends]

private def semanticProfiles : List Profile :=
  [.semanticCore, .publicApi, .metal, .portable, .ecosystem, .allBackends]

private def symbolRequirement (category symbol : String)
    (domains : List Domain) (dimensions : List Dimension)
    (profiles : List Profile) (relation : String) : Requirement :=
  { id := s!"{category}:{symbol}",
    domains,
    dimensions,
    profiles,
    upstreamSymbols := [symbol],
    upstreamTests := [],
    requiredEnvironments := ["host"],
    equivalenceRelation := relation }

private def testRequirement (group path : String)
    (profiles : List Profile) : Requirement :=
  { id := s!"test:{group}:{path}",
    domains := [.workloads],
    dimensions := [.api, .semantic, .runtime],
    profiles,
    upstreamSymbols := [],
    upstreamTests := [path],
    requiredEnvironments :=
      if group == "backend" then ["declared-backend"] else ["host"],
    equivalenceRelation := "tinygrad-test-file-v1" }

def tensorMethodRequirements : List Requirement :=
  ParityTarget.tensorMethods.map fun name =>
    symbolRequirement "tensor-method" s!"Tensor.{name}"
      [.tensorSurface] [.api, .semantic] apiProfiles "tinygrad-tensor-api-v1"

def tensorPropertyRequirements : List Requirement :=
  ParityTarget.tensorProperties.map fun name =>
    symbolRequirement "tensor-property" s!"Tensor.{name}"
      [.tensorSurface] [.api, .semantic] apiProfiles "tinygrad-tensor-api-v1"

def dtypeRequirements : List Requirement :=
  ParityTarget.dtypeNames.map fun name =>
    symbolRequirement "dtype" s!"dtypes.{name}"
      [.dtypeSystem, .scalarSemantics] [.api, .semantic, .numerical]
      semanticProfiles "tinygrad-dtype-v1"

def opsRequirements : List Requirement :=
  ParityTarget.opsMembers.map fun name =>
    symbolRequirement "ops" s!"Ops.{name}"
      [.uopIr, .rewriteSystem, .lowering] [.semantic, .compiler]
      semanticProfiles "tinygrad-ops-v1"

def backendRequirements : List Requirement :=
  ParityTarget.backendNames.map fun name =>
    { (symbolRequirement "backend" s!"tinygrad.runtime.ops_{name}"
        [.renderer, .runtime] [.backend, .runtime]
        (if name == "metal" then [.metal, .allBackends] else [.allBackends])
        "tinygrad-backend-v1") with
      requiredEnvironments := [name] }

def nullTestRequirements : List Requirement :=
  ParityTarget.nullTestFiles.map fun path =>
    testRequirement "null" path semanticProfiles

def unitTestRequirements : List Requirement :=
  ParityTarget.unitTestFiles.map fun path =>
    testRequirement "unit" path semanticProfiles

def backendTestRequirements : List Requirement :=
  ParityTarget.backendTestFiles.map fun path =>
    testRequirement "backend" path [.metal, .portable, .allBackends]

def targetRequirements : List Requirement :=
  tensorMethodRequirements ++ tensorPropertyRequirements ++
  dtypeRequirements ++ opsRequirements ++ backendRequirements ++
  nullTestRequirements ++ unitTestRequirements ++ backendTestRequirements

def targetRequirementIds : List String := targetRequirements.map (·.id)

def unknownCoverageCell (requirement : Requirement) : CoverageCell :=
  { requirementId := requirement.id,
    state := .unknown
      "Tgrad support has not been classified against this generated requirement"
      "run the pinned foreign test or manifest/API adapter and record a typed coverage state",
    evidence := [],
    gap := "unclassified generated upstream requirement" }

/-- A coverage-contract skeleton for a particular immutable Tgrad tree.  Every
cell starts unknown; importing the denominator does not grant conformance. -/
def contractSkeleton (subjectTree : String) (profile : Profile) : Contract :=
  { upstream := pinnedUpstream,
    subjectTree,
    profile,
    requirements := targetRequirements,
    cells := targetRequirements.map unknownCoverageCell,
    validators := [] }

theorem target_requirement_count_is_generated :
    targetRequirements.length = ParityTarget.requirementCount := by
  native_decide

theorem target_requirement_ids_are_unique : targetRequirementIds.Nodup := by
  native_decide

theorem target_requirements_are_well_formed :
    targetRequirements.all Requirement.wellFormed = true := by
  native_decide

theorem target_exclusions_are_explicitly_empty :
    ParityTarget.exclusions = [] := by
  rfl

/-- The denominator and skeleton now exist.  This particular contract remains
unknown until a subject tree, declared profile, applicability rules, calibrated
validators, and observed cells are imported. -/
def targetContract : Epistemic Contract :=
  .unknown
    "a subject-tree/profile instance of the generated 590-row target with observed coverage cells"
    "select an immutable Tgrad tree and profile, import applicable upstream tests, and replace each unknown skeleton cell with observed evidence or an explicit gap"

/-! ## Shape of the ideal codebase

These are ownership boundaries, not a claim that these paths exist today.
Dependencies point from a foundation to its consumer.  The rank is a compact,
checked architectural constraint: semantics cannot depend on a renderer,
verification cannot be imported into the product, and backends attach through
interfaces rather than forking the compiler.
-/

inductive ModuleRole where
  | upstreamContract
  | scalarSemantics
  | shapeAlgebra
  | uopIr
  | tensorEvaluator
  | rewriteEngine
  | tensorFrontend
  | scheduler
  | autograd
  | lowering
  | kernelEvaluator
  | backendInterface
  | renderer
  | runtime
  | runtimeSession
  | generatedAbi
  | jit
  | nnState
  | observability
  | conformance
  | evolution
  deriving DecidableEq, BEq, Repr, Inhabited

def ModuleRole.rank : ModuleRole -> Nat
  | .upstreamContract => 0
  | .scalarSemantics => 0
  | .shapeAlgebra => 0
  | .backendInterface => 0
  | .uopIr => 1
  | .tensorEvaluator => 2
  | .rewriteEngine => 2
  | .tensorFrontend => 2
  | .scheduler => 3
  | .autograd => 3
  | .lowering => 4
  | .kernelEvaluator => 5
  | .renderer => 5
  | .runtime => 6
  | .runtimeSession => 7
  | .generatedAbi => 8
  | .jit => 8
  | .nnState => 8
  | .observability => 8
  | .conformance => 9
  | .evolution => 10

structure Dependency where
  foundation : ModuleRole
  consumer : ModuleRole
  rationale : String
  deriving DecidableEq, BEq, Repr, Inhabited

def idealDependencies : List Dependency :=
  [ { foundation := .scalarSemantics, consumer := .uopIr,
      rationale := "IR payloads use one dtype and scalar semantics" },
    { foundation := .shapeAlgebra, consumer := .uopIr,
      rationale := "movement is represented by compositional views" },
    { foundation := .uopIr, consumer := .tensorEvaluator,
      rationale := "the pure tensor evaluator gives frontend IR a backend-independent meaning" },
    { foundation := .uopIr, consumer := .rewriteEngine,
      rationale := "rewrites preserve the meaning of the same IR" },
    { foundation := .uopIr, consumer := .tensorFrontend,
      rationale := "the public Tensor surface constructs lazy IR" },
    { foundation := .rewriteEngine, consumer := .scheduler,
      rationale := "scheduling consumes normalized graphs" },
    { foundation := .tensorFrontend, consumer := .autograd,
      rationale := "gradient transforms operate on frontend graphs" },
    { foundation := .scheduler, consumer := .lowering,
      rationale := "all operations cross one schedule-to-kernel boundary" },
    { foundation := .lowering, consumer := .kernelEvaluator,
      rationale := "kernel IR has an executable meaning independent of every renderer" },
    { foundation := .backendInterface, consumer := .renderer,
      rationale := "renderers are backend instances, not compiler forks" },
    { foundation := .lowering, consumer := .renderer,
      rationale := "renderers consume typed lowered programs" },
    { foundation := .renderer, consumer := .runtime,
      rationale := "runtime compiles and executes rendered programs" },
    { foundation := .runtime, consumer := .runtimeSession,
      rationale := "one session owns generational handles, devices, buffers, caches, and teardown" },
    { foundation := .runtimeSession, consumer := .generatedAbi,
      rationale := "the generated ABI exposes opaque session-scoped handles, not backend pointers" },
    { foundation := .runtimeSession, consumer := .jit,
      rationale := "JIT replays explicit executable and buffer contracts" },
    { foundation := .runtimeSession, consumer := .observability,
      rationale := "runtime traces are observations of session-owned graph, buffer, cache, and launch state" },
    { foundation := .autograd, consumer := .nnState,
      rationale := "training libraries consume the common gradient graph" },
    { foundation := .upstreamContract, consumer := .conformance,
      rationale := "verification is generated from an immutable foreign target" },
    { foundation := .tensorEvaluator, consumer := .conformance,
      rationale := "frontend properties and differentials have an independent semantic path" },
    { foundation := .kernelEvaluator, consumer := .conformance,
      rationale := "lowering and renderer checks have an independent kernel meaning" },
    { foundation := .observability, consumer := .conformance,
      rationale := "evidence names the artifact and route that actually governed execution" },
    { foundation := .runtime, consumer := .conformance,
      rationale := "backend observations are attached to exact executable trees" },
    { foundation := .conformance, consumer := .evolution,
      rationale := "only observed evidence promotes repository state" } ]

def idealDependenciesForward : Bool :=
  idealDependencies.all fun edge =>
    edge.foundation.rank < edge.consumer.rank && !edge.rationale.isEmpty

theorem ideal_architecture_is_ranked : idealDependenciesForward = true := by
  native_decide

/-! ## Shape of work

The program below is a dependency graph of desired deltas.  It is intentionally
coarser than executable work.  Agents refine a ready template into one or more
transactions small enough to own, falsify, integrate, and revert independently.
-/

inductive Stage where
  | trust
  | semantics
  | compiler
  | verticalSlices
  | training
  | runtime
  | ecosystem
  | backends
  | optimization
  | continuousParity
  deriving DecidableEq, BEq, Repr, Inhabited

def Stage.rank : Stage -> Nat
  | .trust => 0
  | .semantics => 1
  | .compiler => 2
  | .runtime => 3
  | .verticalSlices => 4
  | .training => 5
  | .ecosystem => 6
  | .backends => 7
  | .optimization => 8
  | .continuousParity => 9

inductive ParallelClass where
  | readOnlyDiscovery
  | disjointAuthoring
  | sharedBuildVerification
  | gpuCorrectness
  | gpuTiming
  | evidenceIntegration
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProgramTemplate where
  id : String
  stage : Stage
  dependsOn : List String
  domains : List Domain
  expectedDelta : String
  independentOracle : String
  falsifier : String
  splitRule : String
  parallelClass : ParallelClass
  deriving Repr, Inhabited

private def seed (id : String) (stage : Stage) (dependsOn : List String)
    (domains : List Domain) (expectedDelta independentOracle falsifier splitRule : String)
    (parallelClass : ParallelClass := .disjointAuthoring) : ProgramTemplate :=
  { id, stage, dependsOn, domains, expectedDelta, independentOracle, falsifier,
    splitRule, parallelClass }

def program : List ProgramTemplate :=
  [ seed "parity.pin-upstream" .trust []
      [.tensorSurface, .dtypeSystem, .uopIr]
      "immutable source/API/test manifests for one official tinygrad revision"
      "official tinygrad repository and generated manifests"
      "change one recorded upstream symbol or test hash and require drift detection"
      "split capture tooling from manifest review; never hand-edit generated inventory",
    seed "parity.import-test-contract" .trust ["parity.pin-upstream"]
      [.tensorSurface, .workloads]
      "a runner, unsupported ledger, and score for upstream null/unit/backend tests"
      "the pinned tinygrad test tree"
      "inject one expected upstream failure/pass transition and require the score to move"
      "shard by upstream test file; preserve upstream files as immutable inputs"
      .readOnlyDiscovery,
    seed "harness.namespace-temporaries" .trust []
      [.runtime]
      "concurrent-safe non-GPU verification with per-run temporary namespaces"
      "isolation tests over two concurrent harness processes"
      "run colliding scenarios simultaneously and require distinct artifacts"
      "convert one script family at a time before enabling parallel verification",
    seed "harness.paired-performance" .trust []
      [.optimization, .runtime]
      "live interleaved measurements of Tgrad and tinygrad with variance reported"
      "paired runs on one host and the same workload manifest"
      "repeat identical sessions until the decision rule is demonstrably stable"
      "separate measurement, statistical analysis, and threshold promotion"
      .gpuTiming,
    seed "semantics.dtype-scalar" .semantics ["parity.pin-upstream"]
      [.scalarSemantics, .dtypeSystem]
      "one total dtype lattice, cast, overflow, NaN, and scalar-operation semantics"
      "upstream dtype tests plus numpy/PyTorch edge cases"
      "exercise boundary values and every cast edge"
      "split by dtype family only when cross-family promotion laws stay shared",
    seed "semantics.movement-indexing" .semantics ["parity.pin-upstream"]
      [.shapeAndView]
      "n-dimensional movement/view composition, masks, inversion, aliasing, and indexed UOps"
      "upstream movement/indexing suites and mathematical index laws"
      "generate non-contiguous, masked, zero-size, and negative-step chains"
      "match current movement-UOp semantics; an internal View type may serve proofs but is not the compatibility target",
    seed "semantics.uop-schema" .semantics
      ["parity.pin-upstream", "semantics.dtype-scalar", "semantics.movement-indexing"]
      [.uopIr]
      "the pinned Ops/payload surface represented without semantic String escape hatches"
      "pinned upstream Ops manifest and constructor inventory"
      "delete or perturb one constructor mapping and require manifest or exhaustiveness failure"
      "split stable payload families, not individual operations",
    seed "semantics.reference-evaluator" .semantics
      ["semantics.dtype-scalar", "semantics.movement-indexing", "semantics.uop-schema"]
      [.uopIr, .rewriteSystem]
      "a pure backend-independent meaning for the supported UOp graph"
      "numpy/PyTorch differentials and algebraic laws"
      "mutate one operator meaning while compiler code remains unchanged"
      "grow one vertical semantic slice at a time",
    seed "compiler.stage-contracts" .compiler
      ["semantics.reference-evaluator", "parity.import-test-contract"]
      [.rewriteSystem, .lazySchedule, .lowering, .renderer]
      "stage-certified graph, pass, target, renderer, runtime, diagnostics, and trace contracts"
      "stage-validity specifications plus evaluator-preservation properties"
      "inject a stage-invalid node or bypass a declared boundary and require rejection"
      "stabilize only interfaces consumed by the immediately following CPU vertical slice",
    seed "compiler.first-cpu-slice" .compiler ["compiler.stage-contracts"]
      [.renderer, .runtime]
      "contiguous elementwise add traverses Tensor, rewrite, schedule, lower, linearize, and a simple CPU backend"
      "upstream add/backend tests plus tensor-evaluator, kernel-evaluator, and CPU differential"
      "run with Metal unavailable; bypass each stage and require trace/equivalence failure"
      "co-design the minimal target/runtime contracts with this slice; optimize only after conformance",
    seed "compiler.metal-spine" .compiler ["compiler.first-cpu-slice"]
      [.rewriteSystem, .lazySchedule, .lowering, .renderer, .runtime]
      "add and generic matmul use the same staged compiler on Metal; WMMA is an optimizer choice"
      "upstream operation tests, evaluator/CPU/Metal differential, and captured matmul reference"
      "disable each route, force generic fallback shapes, and verify the runtime trace names the governing artifact"
      "migrate one existing Metal route at a time; delete path-specific Python/FFI routing only after its last caller",
    seed "runtime.buffer-lifecycle" .runtime ["compiler.first-cpu-slice"]
      [.runtime, .interoperability]
      "validated buffer sizes, ownership, caching, diagnostics, and thread contract"
      "sanitizers, lifecycle stress tests, and upstream device tests"
      "drop parents, reuse pools, cross threads, and inject runtime failures"
      "separate safety/lifetime from performance caching",
    seed "slice.elementwise" .verticalSlices
      ["compiler.metal-spine", "semantics.dtype-scalar"]
      [.tensorSurface, .lowering]
      "elementwise Unary/Binary/Ternary operations across promoted dtypes"
      "upstream operation tests"
      "property-generate broadcasting and dtype combinations"
      "packet by semantic family, each crossing API through runtime",
    seed "slice.movement-indexing" .verticalSlices
      ["compiler.metal-spine", "semantics.movement-indexing"]
      [.tensorSurface, .shapeAndView, .lowering]
      "movement, gather/scatter, padding, masking, and indexing parity"
      "upstream movement/indexing tests"
      "compose long randomized view and indexing chains"
      "packet by operation family but share movement/indexing laws",
    seed "slice.reduction" .verticalSlices
      ["slice.elementwise", "slice.movement-indexing"]
      [.tensorSurface, .lazySchedule, .lowering]
      "reductions over arbitrary axes, shapes, identities, and accumulation dtypes"
      "upstream reduction tests and independent framework differential"
      "exercise empty, degenerate, non-contiguous, and numerically hostile reductions"
      "separate semantic reduction from backend optimization",
    seed "slice.matmul-conv" .verticalSlices ["slice.reduction"]
      [.tensorSurface, .lowering, .optimization]
      "general matmul and convolution expressed through the common compiler"
      "upstream operation/model tests plus captured-kernel differentials"
      "force fallback shapes, layouts, padding, dilation, and groups"
      "retain specialized kernels only as optimizer choices behind common semantics",
    seed "slice.random" .verticalSlices
      ["semantics.dtype-scalar", "compiler.metal-spine"]
      [.tensorSurface, .runtime]
      "seeded random generation and distribution semantics"
      "upstream random tests and statistical properties"
      "replay exact seeds and test distribution moments separately"
      "split deterministic PRNG core from device filling",
    seed "slice.effects-assignment" .verticalSlices
      ["slice.elementwise", "slice.movement-indexing", "runtime.buffer-lifecycle"]
      [.tensorSurface, .effectsAndAliasing, .runtime]
      "assignment, in-place operations, aliasing, mutation order, and realization effects"
      "upstream assignment/setitem/schedule tests plus lifecycle stress differentials"
      "vary overlapping views, repeated mutation, graph reuse, and realization order"
      "separate semantic effect ordering from backend synchronization mechanics",
    seed "training.autograd" .training
      ["slice.elementwise", "slice.movement-indexing", "slice.reduction",
       "slice.matmul-conv", "slice.effects-assignment"]
      [.autograd]
      "graph-derived gradients, no_grad, detach, accumulation, and higher-order behavior"
      "upstream gradient suite and finite differences"
      "mutate one derivative rule and require finite-difference failure"
      "packet by gradient rule family while retaining one reverse traversal",
    seed "runtime.tinyjit" .training
      ["runtime.buffer-lifecycle", "compiler.stage-contracts", "slice.effects-assignment"]
      [.jit, .runtime]
      "capture, input replacement, replay, graph batching, and invalidation parity"
      "upstream TinyJit tests"
      "change shape/device/alias inputs across capture boundaries"
      "capture semantics first; command-queue optimization second",
    seed "ecosystem.nn-state" .ecosystem
      ["training.autograd", "runtime.tinyjit"]
      [.nnAndState, .interoperability, .workloads]
      "layers, optimizers, state traversal, serialization, and canonical model workloads"
      "upstream nn/state tests and pinned example workloads"
      "round-trip parameters and train several steps against an independent reference"
      "packet by end-to-end workload slice, not by isolated class count",
    seed "backends.multi-device" .backends
      ["runtime.tinyjit", "slice.reduction"]
      [.multiDevice, .runtime]
      "device tuples, sharding, synchronization, collectives, and buffer transfer"
      "upstream multi-device tests on declared hardware profiles"
      "vary shard axes, uneven shapes, and device ordering"
      "keep unavailable hardware explicitly blocked, never inferred green",
    seed "backends.profile-matrix" .backends
      ["compiler.first-cpu-slice", "runtime.buffer-lifecycle", "backends.multi-device"]
      [.renderer, .runtime, .interoperability]
      "one conformance row for every declared backend profile"
      "upstream backend suite on each actual backend"
      "remove a backend runner and require profile completeness to become unknown"
      "one backend per workstream behind shared target/runtime contracts"
      .gpuCorrectness,
    seed "optimization.search" .optimization
      ["slice.matmul-conv", "runtime.tinyjit", "backends.profile-matrix"]
      [.optimization, .lazySchedule, .lowering]
      "typed Opt application, BEAM/search, memory planning, and stable performance distributions"
      "upstream speed tests plus paired live benchmarks"
      "compare repeated sessions and reject thresholds dominated by variance"
      "optimize one workload family without changing its compatibility oracle"
      .gpuTiming,
    seed "parity.upstream-drift-loop" .continuousParity
      ["parity.import-test-contract", "backends.profile-matrix", "ecosystem.nn-state"]
      [.tensorSurface, .dtypeSystem, .uopIr, .workloads]
      "upstream changes automatically produce typed gaps without silently moving the target"
      "manifest diff between two official upstream revisions"
      "add/remove an upstream symbol and require a new gap or explicit exclusion"
      "discovery may parallelize; one integrator promotes the next target revision"
      .evidenceIntegration ]

/-- Internal identity projection for the authored program-template table.

Checks over this list establish only that the plan is internally coherent.
They do not compare Tgrad with tinygrad and therefore are not parity evidence. -/
def programIds : List String := program.map (fun item => item.id)

def ProgramTemplate.structurallyReady (item : ProgramTemplate) : Bool :=
  !item.id.isEmpty &&
  !item.domains.isEmpty &&
  !item.expectedDelta.isEmpty &&
  !item.independentOracle.isEmpty &&
  !item.falsifier.isEmpty &&
  !item.splitRule.isEmpty &&
  item.dependsOn.all fun dependency =>
    program.any fun candidate =>
      candidate.id == dependency && candidate.stage.rank <= item.stage.rank

/-- `dependsOn` contains only hard readiness edges.  Context, hierarchy,
provenance, and invalidation/read edges belong in their own fields when a
template is refined.  Requiring hard dependencies to precede consumers gives
the template graph a checked topological order. -/
def dependenciesFollowOrder : List ProgramTemplate -> List String -> Bool
  | [], _ => true
  | item :: rest, seen =>
      item.dependsOn.all seen.contains &&
      dependenciesFollowOrder rest (item.id :: seen)

/-- Internal sanity check over the authored template table.  In particular,
this cannot establish upstream coverage, implementation support, or parity: its
entire input is the table defined in this module. -/
def programStructurallyReady : Bool :=
  program.all ProgramTemplate.structurallyReady &&
  dependenciesFollowOrder program []

/-- The authored template IDs are unique.  This is a self-consistency theorem,
not an observation about either product or the upstream contract. -/
theorem program_ids_are_unique : programIds.Nodup := by
  native_decide

/-- The authored dependency table is internally known and topologically
ordered.  This is useful plan validation, but is not compatibility evidence. -/
theorem program_dependencies_are_known_and_ordered :
    programStructurallyReady = true := by
  native_decide

def templatesReadyAfter (completed : List String) : List ProgramTemplate :=
  program.filter fun item =>
    !completed.contains item.id &&
    item.dependsOn.all completed.contains

example :
    (templatesReadyAfter []).map (fun item => item.id) =
      ["parity.pin-upstream", "harness.namespace-temporaries",
       "harness.paired-performance"] := by
  native_decide

/-- The transaction an agent must produce before authoring.  Unlike a
`ProgramTemplate`, this is intentionally specific to one immutable tree. -/
inductive PacketKind where
  | observation
  | authoring
  | verification
  | integration
  deriving DecidableEq, BEq, Repr, Inhabited

def PacketKind.mayBeReadOnly : PacketKind -> Bool
  | .observation | .verification => true
  | .authoring | .integration => false

inductive PacketAuthority where
  | ownerGoal
  | evidenceRepair
  | delegatedMaintenance
  deriving DecidableEq, BEq, Repr, Inhabited

structure ResourceRequest where
  resource : String
  exclusive : Bool
  observationHash : String
  observedAt : String
  deriving DecidableEq, BEq, Repr, Inhabited

def ResourceRequest.wellFormed (request : ResourceRequest) : Bool :=
  !request.resource.isEmpty &&
  !request.observationHash.isEmpty &&
  !request.observedAt.isEmpty

structure ResourceGrant where
  resource : String
  observationHash : String
  leaseToken : String
  heartbeatEpoch : Nat
  validThroughEpoch : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

def ResourceGrant.activeAt (grant : ResourceGrant) (epoch : Nat) : Bool :=
  !grant.resource.isEmpty &&
  !grant.observationHash.isEmpty &&
  !grant.leaseToken.isEmpty &&
  grant.heartbeatEpoch <= epoch &&
  epoch <= grant.validThroughEpoch

structure WorkPacket where
  seedId : String
  upstream : Epistemic UpstreamRef
  profile : Profile
  kind : PacketKind
  authority : PacketAuthority
  baseCommit : String
  baseTree : String
  hardDependencies : List String
  requirementIds : List String
  gap : String
  readSet : List String
  writeSet : List String
  effectSet : List String
  generatedOutputs : List String
  resources : List ResourceRequest
  oracle : String
  verifierTree : String
  validatorIds : List String
  validators : List String
  falsifiers : List String
  expectedArtifacts : List String
  recoveryByFailureClass : List String
  deriving Repr, Inhabited

def WorkPacket.structurallyExecutable (packet : WorkPacket) : Bool :=
  programIds.contains packet.seedId &&
  (if packet.seedId == "parity.pin-upstream"
   then packet.upstream.hasUpgradePath
   else packet.upstream.isConfirmed) &&
  !packet.baseCommit.isEmpty &&
  !packet.baseTree.isEmpty &&
  !packet.requirementIds.isEmpty &&
  !packet.gap.isEmpty &&
  (packet.kind.mayBeReadOnly || !packet.writeSet.isEmpty) &&
  !packet.effectSet.isEmpty &&
  !packet.resources.isEmpty &&
  packet.resources.all ResourceRequest.wellFormed &&
  !packet.oracle.isEmpty &&
  !packet.verifierTree.isEmpty &&
  !packet.validatorIds.isEmpty &&
  !packet.validators.isEmpty &&
  !packet.falsifiers.isEmpty &&
  !packet.expectedArtifacts.isEmpty &&
  !packet.recoveryByFailureClass.isEmpty

/-- Readiness is a replayed witness about the current world, not an assertion
inside the packet being authorized. -/
structure ReadyWitness where
  upstream : Epistemic UpstreamRef
  baseCommit : String
  baseTree : String
  promotedDependencies : List String
  availableValidatorIds : List String
  observedEpoch : Nat
  resourceGrants : List ResourceGrant
  authorityGranted : Bool
  deriving Repr, Inhabited

def ReadyWitness.supports (witness : ReadyWitness) (packet : WorkPacket) : Bool :=
  packet.structurallyExecutable &&
  witness.baseCommit == packet.baseCommit &&
  witness.baseTree == packet.baseTree &&
  witness.authorityGranted &&
  packet.hardDependencies.all witness.promotedDependencies.contains &&
  packet.validatorIds.all witness.availableValidatorIds.contains &&
  packet.resources.all fun request =>
    witness.resourceGrants.any fun grant =>
      grant.resource == request.resource &&
      grant.observationHash == request.observationHash &&
      grant.activeAt witness.observedEpoch &&
  (if packet.seedId == "parity.pin-upstream"
   then witness.upstream.hasUpgradePath
   else witness.upstream.isConfirmed &&
        witness.upstream.value? == packet.upstream.value?)

private def intersects (left right : List String) : Bool :=
  left.any right.contains

/-- Candidate authoring is parallel only when writes do not intersect the
other packet's writes or reads.  Generated outputs are part of the effect set. -/
def WorkPacket.authoringCompatible (left right : WorkPacket) : Bool :=
  !intersects (left.writeSet ++ left.generatedOutputs)
      (right.writeSet ++ right.generatedOutputs ++ right.readSet) &&
  !intersects (right.writeSet ++ right.generatedOutputs) left.readSet

inductive GapDisposition where
  | template (programTemplateId : String)
  | unknown (missing : String) (resolveBy : String)
  | deferred (missing : String) (rationale : String)
  | blocked (resource : String) (reobserveBy : String)
  deriving DecidableEq, BEq, Repr, Inhabited

def GapDisposition.wellFormed : GapDisposition -> Bool
  | .template id => programIds.contains id
  | .unknown missing resolveBy => !missing.isEmpty && !resolveBy.isEmpty
  | .deferred missing rationale => !missing.isEmpty && !rationale.isEmpty
  | .blocked resource reobserveBy => !resource.isEmpty && !reobserveBy.isEmpty

structure GapRecord where
  requirementId : String
  disposition : GapDisposition
  deriving DecidableEq, BEq, Repr, Inhabited

def Contract.gapRequirementIds (contract : Contract) : List String :=
  (contract.requiredForProfile.filter fun requirement =>
    !contract.hasPromotableCell requirement).map (·.id)

/-- Queue completeness means no mandatory gap disappears.  It is either mapped
to a program template or carries an explicit epistemic/resource reason. -/
def Contract.backlogCovers (contract : Contract) (backlog : List GapRecord) : Bool :=
  let gaps := contract.gapRequirementIds
  let backlogIds := backlog.map (·.requirementId)
  backlogIds.eraseDups.length == backlogIds.length &&
  (backlog.all fun record =>
    gaps.contains record.requirementId && record.disposition.wellFormed) &&
  gaps.all fun requirementId => backlogIds.contains requirementId

/-- A program template is not itself authority to mutate the repository. -/
example :
    (program.find? fun item => item.id == "compiler.first-cpu-slice").isSome = true := by
  native_decide

/-! ## Product-symbol drift pins

`Parity` is specification-side code, but it describes the product's dtype,
tensor, UOp, optimization, renderer, runtime, and view/lowering surfaces.  These
checks deliberately put representative product symbols in this module's
compile-time dependency cone.  A product rename or signature change must break
the spec build and force reconciliation.  They are drift alarms only: like the
table-coherence theorems above, they are not parity evidence. -/

#check (Tgrad.Dtype : Type)
#check (Tgrad.Dtype.lub : Tgrad.Dtype -> Tgrad.Dtype -> Tgrad.Dtype)
#check (Tgrad.ConstVal : Type)
#check (Tgrad.BinOp : Type)
#check (Tgrad.UOp : Type)
#check (Tgrad.UOpKind : Type)
#check (Tgrad.UOp.kind : Tgrad.UOp -> Tgrad.UOpKind)
#check (Tgrad.Tensor : Type)
#check (Tgrad.Tensor.transpose : Tgrad.Tensor -> Tgrad.Tensor)
#check (Tgrad.Schedule.View : Type)
#check (Tgrad.Schedule.View.indexOf :
  Tgrad.Schedule.View -> List Tgrad.UOp -> Tgrad.UOp)
#check (Tgrad.Codegen.Opt.OptOps : Type)
#check (Tgrad.Renderer.Metal.KernelDecl : Type)
#check (Tgrad.Renderer.Metal.renderKernel :
  Tgrad.Renderer.Metal.KernelDecl -> String)
#check (Tgrad.Pipeline.materializeView :
  Tgrad.Tensor -> IO (Except Tgrad.PipelineError Tgrad.Tensor))
#check (Tgrad.Runtime.Metal.metalCompile : String -> IO UInt64)
#check (Tgrad.Runtime.Metal.metalDispatch :
  UInt64 -> String -> Array UInt64 ->
  USize -> USize -> USize -> USize -> USize -> USize -> IO UInt32)

end Tgrad.Spec.Parity

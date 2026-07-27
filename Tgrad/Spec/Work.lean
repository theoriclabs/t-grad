import Tgrad.Spec.Evolution
import Tgrad.Spec.Parity

/-! # Tgrad.Spec.Work — work performed on the codebase

This is evolution work: changes performed by people and agents on the source
tree. It is distinct from `Runtime.WorkUnit`, which is repeatable computation
performed by the product, verifier, or specification. The Markdown plan
explains rationale; this module owns current evolution state and answers:

* which work is complete, and with what evidence;
* which planned work is dependency-ready;
* which ready items can be authored in parallel without file conflicts;
* which verification steps must serialize on shared resources;
* which action the priority picker selects next.

The unit is deliberately a small work item rather than a broad roadmap theme.
Completion requires a semantic validator, not merely “the command exited 0.”
-/

namespace Tgrad.Spec
namespace Evolution

private def ew (value : String) : Growth.EvolutionWorkId :=
  Growth.evolutionWorkId value

private def rw (value : String) : Runtime.WorkId := Runtime.workId value

inductive Phase where
  | explore | design | build | verify | promote
  deriving DecidableEq, BEq, Repr, Inhabited

inductive Authority where
  | userGoal | agentExecution | evidence
  deriving DecidableEq, BEq, Repr, Inhabited

structure Validation where
  expectedOutput : String
  procedure : String
  passCondition : String
  obligations : List Growth.ObligationKind
  deriving Repr, Inhabited

def Validation.meaningful (validation : Validation) : Bool :=
  !validation.expectedOutput.isEmpty &&
  !validation.procedure.isEmpty &&
  !validation.passCondition.isEmpty &&
  validation.passCondition != "exit 0" &&
  !validation.obligations.isEmpty

inductive Progress where
  | planned
  | inProgress (owner : String)
  | complete (evidence : String)
  | deferred (rationale : String) (returnWhen : String)
  deriving Repr, Inhabited

/-! `Progress` is the imported dashboard projection for the current roadmap.
It is useful for dependency queries, but `complete "commit"` is not a durable
promotion certificate. New execution history belongs in the event protocol in
`Tgrad.Spec.Evolution`; the migration state is explicit below. -/

def Progress.isComplete : Progress -> Bool
  | .complete evidence => !evidence.isEmpty
  | _ => false

def Progress.isPlanned : Progress -> Bool
  | .planned => true
  | _ => false

def Progress.isInProgress : Progress -> Bool
  | .inProgress _ => true
  | _ => false

def historicalEvolutionHistory : Epistemic (List Event) :=
  .unknown
    "the existing completed roadmap items predate attempt/candidate/check/promotion events"
    "record all new work with applyEvent/replay, then backfill only historical facts whose exact trees and check artifacts can be recovered"

structure WorkItem where
  id : Growth.EvolutionWorkId
  title : String
  phase : Phase
  authority : Authority
  closesFindings : List String
  dependsOn : List Growth.EvolutionWorkId
  runtimeScope : List Runtime.WorkId := []
  touches : List Component
  writes : List String
  authoringResources : List Resource
  verificationResources : List Resource
  cost : Nat
  goalDistance : Nat
  validation : Validation
  recovery : String
  progress : Progress
  deriving Repr, Inhabited

private def completeValidation (evidence : String) : Validation :=
  { expectedOutput := "committed behavior and discriminating regression",
    procedure := "see commit evidence",
    passCondition := evidence,
    obligations := [.build, .unitRegression, .differential] }

private def plannedValidation
    (expectedOutput procedure passCondition : String)
    (obligations : List Growth.ObligationKind) : Validation :=
  { expectedOutput, procedure, passCondition, obligations }

def workItems : List WorkItem :=
  [ { id := ew "foundation.renderer-runtime", title := "make renderKernel load-bearing",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-runtime-file-replay"], dependsOn := [],
      runtimeScope := [rw "product.render-metal", rw "product.compile-metal"],
      touches := [.renderer, .leanFfi],
      writes := ["Tgrad/PythonFFI.lean", "Tgrad/Pipeline.lean"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 2, goalDistance := 0,
      validation := completeValidation
        "fixtures/codegen hidden; rendered 64x64 remained byte-identical",
      recovery := "restore captured acquisition only if renderer output diverges",
      progress := .complete "f679bf7" },
    { id := ew "foundation.view-algebra", title := "replace movement special cases with View",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-rangeify-identity", "F-slice-drops-axes",
        "F-expand-assumes-axis1"], dependsOn := [],
      runtimeScope := [rw "product.compose-view", rw "product.rangeify",
        rw "product.lower-scalar-matmul"],
      touches := [.viewAlgebra, .scheduler],
      writes := ["Tgrad/Schedule/View.lean", "Tgrad/Schedule/Rangeify.lean",
        "Tgrad/Pipeline.lean"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 4, goalDistance := 0,
      validation := completeValidation
        "20 Lean assertions; 13 numpy differential view cases",
      recovery := "leave unsupported movement chain explicit rather than approximate",
      progress := .complete "f679bf7" },
    { id := ew "correctness.buffer-shape", title := "enforce shape/allocation agreement",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-buffer-shape-mismatch"],
      dependsOn := [ew "foundation.view-algebra"],
      runtimeScope := [rw "product.ingest-tensor", rw "product.allocate-output"],
      touches := [.pythonAuthoring], writes := ["python/tgrad.py"],
      authoringResources := [.sourceTree], verificationResources := [.metalGpu],
      cost := 1, goalDistance := 0,
      validation := completeValidation "short and oversized payloads raise TgradTypeError",
      recovery := "keep views exempt; their legal shape expansion is represented by stride 0",
      progress := .complete "e1d5760" },
    { id := ew "correctness.view-lifetime", title := "retain base Tensor from every view",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-view-parent-lifetime"],
      dependsOn := [ew "correctness.buffer-shape"],
      runtimeScope := [rw "product.record-view"],
      touches := [.pythonAuthoring], writes := ["python/tgrad.py"],
      authoringResources := [.sourceTree], verificationResources := [.metalGpu],
      cost := 1, goalDistance := 0,
      validation := completeValidation "temporary view survived GC and 80-allocation LRU churn",
      recovery := "inspect finalizer ownership and LRU handoff if lifetime probe fails",
      progress := .complete "e1d5760" },
    { id := ew "correctness.view-readback-safety",
      title := "refuse view readback until materialization exists",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-view-readback-wrong"],
      dependsOn := [ew "correctness.view-lifetime"],
      runtimeScope := [rw "product.readback-buffer", rw "product.materialize-view"],
      touches := [.pythonAuthoring], writes := ["python/tgrad.py"],
      authoringResources := [.sourceTree], verificationResources := [.metalGpu],
      cost := 1, goalDistance := 0,
      validation := completeValidation "view numpy/to_bytes raise; buffer and matmul readback pass",
      recovery := "never fall back to reshaping parent storage",
      progress := .complete "e1d5760" },
    { id := ew "view.materialize", title := "materialize a View with an indexed copy kernel",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-view-materialization-missing"],
      dependsOn := [ew "foundation.view-algebra", ew "correctness.view-readback-safety"],
      runtimeScope := [rw "product.materialize-view", rw "product.readback-buffer"],
      touches := [.pythonAuthoring, .leanFfi, .renderer, .metalRuntime],
      writes := ["python/tgrad.py", "Tgrad/PythonFFI.lean", "Tgrad/Pipeline.lean"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 4, goalDistance := 1,
      validation := plannedValidation
        "numpy/to_bytes preserve exact bf16 bits for transpose, slice, reshape, expand, and supported chains"
        "build the exact candidate; run the serial Metal suite over special bf16 payloads, invalid chains, and temporary-parent churn"
        "all supported forms match numpy shape/layout/bits; unsupported and empty views reject; rangeify supplies the executed source index"
        [.apiContract, .safety, .numerical, .differential],
      recovery := "retain explicit NotInLeanScope readback until every view form validates",
      progress := .complete "e6241bd" },
    { id := ew "codegen.differential-harness",
      title := "compare captured and generated kernels semantically",
      phase := .promote, authority := .agentExecution,
      closesFindings := [], dependsOn := [ew "foundation.renderer-runtime"],
      runtimeScope := [rw "verify.codegen-differential"],
      touches := [.gateHarness],
      writes := ["Main.lean", "scripts/differential_codegen.sh"],
      authoringResources := [.sourceTree],
      verificationResources := [.metalGpu, .tmpNamespace],
      cost := 2, goalDistance := 0,
      validation := plannedValidation
        "all 11 captured and generated sentinels compile, consume one seeded bf16 input pair per shape, and produce separately attributable outputs"
        "dispatch each function with its own geometry; require rendered source != capture; compare raw outputs bitwise; falsify with a wrong-but-distinct store offset"
        "11/11 source-inequality and bit-identity; 240 MB compared; c->c+2 mutation is rejected with 727933 differing bytes"
        [.build, .numerical, .semantic, .differential, .provenance, .resourceIsolation],
      recovery := "on divergence preserve both outputs and source hashes; do not route",
      progress := .complete "a6d5958" },
    { id := ew "codegen.warp-parameter",
      title := "generalize parametric WMMA generator to 64x64x64",
      phase := .build, authority := .agentExecution,
      closesFindings := [], dependsOn := [ew "foundation.renderer-runtime"],
      runtimeScope := [rw "product.lower-tc-matmul"],
      touches := [.renderer],
      writes := ["Tgrad/Renderer/Metal.lean", "Tgrad/Renderer/MatmulTc.lean"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 3, goalDistance := 0,
      validation := plannedValidation
        "wide generation covers 64x64x64 without changing production eligibility"
        "check wide_generates_64, strict_rejects_64, captured launch dimensions, and symbolic address equivalence"
        "wide/strict domains differ deliberately; 64x64 geometry and address/wiring/permutation match the capture"
        [.build, .semantic, .differential],
      recovery := "keep 64x64 transcription while larger shapes remain generated",
      progress := .complete "75f856b" },
    { id := ew "codegen.typed-stores",
      title := "emit TC stores through typed index statements",
      phase := .build, authority := .agentExecution,
      closesFindings := [], dependsOn := [ew "foundation.renderer-runtime"],
      runtimeScope := [rw "product.lower-tc-matmul", rw "product.render-metal"],
      touches := [.tensorIr, .renderer, .gateHarness],
      writes := ["Tgrad/UOp.lean", "Tgrad/Renderer/Metal.lean",
        "Tgrad/Renderer/MatmulTc.lean", "scripts/gates/L14_B_2_b.sh"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 3, goalDistance := 1,
      validation := plannedValidation
        "32 typed, non-aliasing output stores"
        "render and differentially execute all sentinel kernels"
        "store addresses cover each 32x128 tile exactly once"
        [.build, .semantic, .differential],
      recovery := "compare address set against existing accumulator permutation",
      progress := .complete "1787c83" },
    { id := ew "codegen.route-sentinels",
      title := "route sentinels and widen TC dispatch safely",
      phase := .build, authority := .userGoal,
      closesFindings := [],
      dependsOn := [ew "codegen.differential-harness", ew "codegen.warp-parameter",
        ew "codegen.typed-stores"],
      runtimeScope := [rw "product.select-matmul-route", rw "product.lower-tc-matmul"],
      touches := [.leanFfi, .renderer],
      writes := ["Tgrad/PythonFFI.lean", "Tgrad/Pipeline.lean",
        "Tgrad/Renderer/MatmulTc.lean", "Tgrad/Codegen/Opt/Heuristic.lean",
        "python/tgrad.py", "scripts/devcheck.sh"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 3, goalDistance := 0,
      validation := plannedValidation
        "all sentinel dispatches use generated kernels with nonzero tcLaunchDims geometry"
        "widen eligibility and dispatch together; update the devcheck eligibility triple; run routing and numerical differentials"
        "11/11 generated and numerically equivalent; (128,128,128)/(96,128,128)/(64,64,64) eligibility changes are explicit and every admitted shape has a nonzero grid"
        [.build, .apiContract, .safety, .numerical, .differential],
      recovery := "keep strict eligibility until geometry and the reviewed eligibility triple pass together; revert only divergent sentinel routes",
      progress := .complete "fd945b1" },
    { id := ew "gates.semantic-codegen", title := "add semantic layers, then retire L12 byte equality",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-byte-equality-gate"],
      dependsOn := [ew "codegen.differential-harness"],
      runtimeScope := [rw "verify.codegen-differential"],
      touches := [.gateHarness],
      writes := ["scripts/gates/L12.sh", "scripts/gates/L14_B_2_b.sh",
        "scripts/dev/l15_b_audit.py"],
      authoringResources := [.sourceTree],
      verificationResources := [.metalGpu, .tmpNamespace, .evidenceStore],
      cost := 3, goalDistance := 0,
      validation := plannedValidation
        "gate retains source byte-equality while transcription exists, and additionally checks store non-aliasing plus captured/generated execution"
        "build tileStoreOffsets_nodup; run the differential layer; prove c->c+2 stays Nodup but fails output comparison; retire byte equality only when transcription is deleted"
        "collision mutation fails the theorem; wrong-but-distinct placement fails the differential; 11/11 sources differ and outputs match"
        [.build, .semantic, .differential, .provenance],
      recovery := "leave L12 red; never weaken it to source substring checks",
      progress := .complete "aa67497" },
    { id := ew "codegen.delete-transcription",
      title := "delete MatmulDecls and lower_matmul transpiler",
      phase := .promote, authority := .userGoal,
      closesFindings := ["F-transcribed-sentinel-codegen"],
      dependsOn := [ew "codegen.route-sentinels", ew "gates.semantic-codegen"],
      runtimeScope := [rw "product.lower-tc-matmul", rw "product.render-metal"],
      touches := [.renderer, .gateHarness],
      writes := ["EXPERIMENT_RESULT.md", "GROWING_TGRAD.md", "Main.lean",
        "PLAN_CORRECTNESS_AND_CODEGEN.md", "PLAN_TINYGRAD_COMPAT.md", "README.md",
        "Tgrad.lean", "Tgrad/Pipeline.lean", "Tgrad/PythonFFI.lean",
        "Tgrad/Renderer/MatmulDecls.lean", "Tgrad/Renderer/MatmulScalar.lean",
        "Tgrad/Renderer/Metal.lean", "c/tgrad_python.c", "python/tgrad.py",
        "scripts/dev/l15_b_audit.py", "scripts/dev/lower_matmul.py",
        "scripts/differential_codegen.sh", "scripts/gates/L12.sh",
        "scripts/gates/L12_falsifiability.md",
        "scripts/gates/L13_F_STRICT_B_falsifiability.md", "scripts/gates/L14_A.sh",
        "scripts/gates/L14_B_2_a_falsifiability.md", "scripts/gates/L14_B_2_b.sh",
        "scripts/gates/L14_B_2_b_falsifiability.md"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu, .tmpNamespace],
      cost := 1, goalDistance := 0,
      validation := plannedValidation
        "no runtime/build dependency on transcribed declarations and no source-byte-equality gate layer"
        "inspect exact Git effects; build; run assertions, 11-shape semantic differential, and 50-case generated numerical sweep"
        "artifacts deleted; only explicit absence/history references remain; Nodup and semantic differential remain independent; all 11 generated kernels and 50 numerical rows pass"
        [.build, .unitRegression, .semantic, .numerical, .differential],
      recovery := "restore files from git while preserving generated routing work",
      progress := .complete "9f2ab91" },
    { id := ew "evidence.audit-tool", title := "make evidence provenance mechanically auditable",
      phase := .promote, authority := .evidence,
      closesFindings := [], dependsOn := [],
      runtimeScope := [rw "verify.evidence-audit"],
      touches := [.evidenceStore, .gateHarness],
      writes := ["scripts/dev/evidence_provenance_audit.py"],
      authoringResources := [.sourceTree],
      verificationResources := [.evidenceStore],
      cost := 2, goalDistance := 0,
      validation := plannedValidation
        "repeatable counts for commit reachability, hashes, roll-ups, and writer agreement"
        "run against committed evidence and a synthetic HEAD-tied good directory"
        "committed set fails with 37/37, 73/115, 28, 27; synthetic good evidence passes"
        [.semantic, .provenance],
      recovery := "keep the auditor nonfatal until evidence regeneration is owner-authorized",
      progress := .complete "bdc01b0" },
    { id := ew "spec.record-regeneration-observation",
      title := "record failed regeneration and performance repeatability",
      phase := .promote, authority := .evidence,
      closesFindings := [], dependsOn := [ew "evidence.audit-tool"],
      runtimeScope := [rw "verify.performance-repeatability",
        rw "verify.evidence-audit", rw "spec.check-growth-loop"],
      touches := [.specification, .evidenceStore, .gateHarness],
      writes := ["EXPERIMENT_RESULT.md", "GROWING_TGRAD.md",
        "PLAN_CORRECTNESS_AND_CODEGEN.md", "PLAN_TINYGRAD_COMPAT.md",
        "README.md", "Tgrad/Spec/Findings.lean",
        "Tgrad/Spec/RuntimeWork.lean", "Tgrad/Spec/Work.lean"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .evidenceStore],
      cost := 2, goalDistance := 0,
      validation := plannedValidation
        "checked distinction between bounded repeatability diagnosis, bypassed parity validation, and an abandoned partial-regeneration candidate"
        "build TgradSpec; replay tgrad-spec; require the exact current provenance failure; inspect the revision-scoped public narrative"
        "25 capabilities; 2 open findings; 73 events at the candidate tree; perf.rebaseline remains the sole frontier; 26/37, 76/115, 28, and 17 provenance defects are reproduced"
        [.build, .semantic, .provenance],
      recovery := "retain the direct observations but revert any prose or state transition that implies performance parity or complete evidence promotion",
      progress := .complete "0f4d594" },
    { id := ew "parity.pin-upstream",
      title := "capture a versioned upstream compatibility contract",
      phase := .design, authority := .userGoal,
      closesFindings := [], dependsOn := [], runtimeScope := [],
      touches := [.specification, .gateHarness, .evidenceStore],
      writes := ["scripts/parity", "fixtures/parity",
        "Tgrad/Spec/ParityTarget.lean", "Tgrad/Spec/Parity.lean",
        "Tgrad/Spec/Findings.lean", "PARITY.md"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .tmpNamespace, .evidenceStore],
      cost := 3, goalDistance := 0,
      validation := plannedValidation
        "one reproducible official tinygrad revision with generated source/API/Ops/dtype/backend/test manifests and an explicit empty-by-default exclusions ledger"
        "capture from a clean upstream checkout; rebuild manifests independently; compare hashes; build TgradSpec; mutate one symbol and one hash to calibrate drift detection"
        "the pin is exact; two captures agree; every manifest has provenance; mutations fail; no count or expected value is hand-authored; targetUpstream becomes confirmed only after review"
        [.build, .apiContract, .semantic, .provenance, .humanReview],
      recovery := "leave targetUpstream unknown, retain the candidate snapshot as research input, and repair the extractor rather than editing generated manifests",
      progress := .complete "8c87034" },
    { id := ew "harness.namespace-temporaries",
      title := "namespace every gate and devcheck temporary artifact",
      phase := .build, authority := .userGoal,
      closesFindings := [], dependsOn := [], runtimeScope := [],
      touches := [.gateHarness],
      writes := ["scripts/lib/run_context.sh", "scripts/lib/checks.sh",
        "scripts/gate.sh", "scripts/devcheck.sh",
        "scripts/runtime_independence.sh", "scripts/differential_codegen.sh",
        "scripts/dev/l15_a_audit.py", "scripts/dev/test_run_context.sh",
        "scripts/gates", "Tgrad/Pipeline.lean",
        "Tgrad/Spec/LiveConditions.lean", "Tgrad/Spec/RuntimeWork.lean",
        "PARITY.md", "GROWING_TGRAD.md"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .tmpNamespace],
      cost := 4, goalDistance := 0,
      validation := plannedValidation
        "zero fixed gate/devcheck /tmp paths; one owned root inherited by nested gates; Lean trace emission requires an explicit run-scoped path"
        "run two concurrent contexts with distinct artifacts and trace paths; bash -n every shell script; exercise missing and explicit Lean trace paths; run a focused build-independent entry point"
        "no collision, broad cleanup, or global symlink; direct and nested entry points initialize safely; only the root owner cleans; shared build/GPU/evidence resources remain serial"
        [.build, .semantic, .safety, .resourceIsolation],
      recovery := "retain serial verification and revert the affected script batch if any direct or nested entry point loses its artifacts",
      progress := .complete "602897e" },
    { id := ew "harness.paired-performance",
      title := "author a live paired performance observation harness",
      phase := .build, authority := .userGoal,
      closesFindings := [], dependsOn := [ew "codegen.delete-transcription"],
      runtimeScope := [rw "verify.paired-performance-observer"],
      touches := [.gateHarness],
      writes := ["scripts/perf/paired_runtime.py", "scripts/perf/README.md",
        "scripts/dev/test_paired_runtime.py", "Tgrad/Spec/RuntimeWork.lean",
        "Tgrad/Spec/Work.lean", "PARITY.md"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 3, goalDistance := 0,
      validation := plannedValidation
        "clean attributable subjects, unique run identity, deterministic AB/BA pairing, correctness-before-output, raw observations, absolute and relative distributions, and explicit non-kernel scope"
        "run CPU-only fakes twice; inject output/revision/timed failures and unequal session sizes; inspect schema; then run one serial exact-tree live smoke without treating it as repeatability evidence"
        "both orders occur; fixed fake identities reproduce; real invocations cannot alias; wrong outputs create no timed evidence; dirty subjects reject; session weighting is equal; no frozen baseline, threshold, verdict, or kernel-speed eligibility exists"
        [.build, .semantic, .performance, .provenance, .resourceIsolation],
      recovery := "keep verify.performance missing and delete the harness candidate if its fake calibration or metadata contract is incomplete",
      progress := .complete "42838f2" },
    { id := ew "perf.rebaseline", title := "measure symmetric generated-kernel performance",
      phase := .verify, authority := .evidence,
      closesFindings := ["F-performance-methodology"],
      dependsOn := [ew "codegen.delete-transcription",
        ew "harness.paired-performance"],
      runtimeScope := [rw "verify.performance"],
      touches := [.gateHarness, .evidenceStore],
      writes := ["fixtures/perf/generated_codegen.json", "EXPERIMENT_RESULT.md"],
      authoringResources := [.sourceTree],
      verificationResources := [.metalGpu, .tmpNamespace, .evidenceStore],
      cost := 5, goalDistance := 0,
      validation := plannedValidation
        "paired same-session distributions for both runtimes, with a repeatability model before any verdict"
        "interleave synchronized dispatch-only samples serially; repeat complete runs; retain raw samples; estimate within-run and between-run variance; predeclare a threshold derived from that variance or publish no pass/fail threshold"
        "same boundary and cache/JIT policy; both denominators live; uncertainty and throughput physically sane; the verdict is stable across repeats and no threshold is tuned to make a run green"
        [.performance, .provenance, .resourceIsolation],
      recovery := "report regression or indeterminate variance; never substitute a frozen denominator or tune the threshold after observing failures",
      progress := .planned },
    { id := ew "evidence.regenerate", title := "regenerate evidence at the shipped commit",
      phase := .promote, authority := .evidence,
      closesFindings := [],
      dependsOn := [ew "perf.rebaseline"],
      runtimeScope := [rw "verify.evidence-integrity"],
      touches := [.evidenceStore, .gateHarness],
      writes := ["fixtures/gate_evidence"],
      authoringResources := [.sourceTree],
      verificationResources := [.metalGpu, .tmpNamespace, .evidenceStore],
      cost := 4, goalDistance := 0,
      validation := plannedValidation
        "all 37 evidence files naming one shipped commit with recomputable hashes"
        "after perf.rebaseline is promoted, run gates serially; retain red results; recompute every recorded hash; reject a partial snapshot"
        "all 37 files come from their current scripts; commit equals the measured tree; all source, child, and baseline hashes resolve; L11 is not made green by threshold tuning"
        [.provenance, .semantic, .humanReview],
      recovery := "keep reproducible partial files as diagnostic history but do not promote the snapshot while any gate or provenance class remains red",
      progress := .planned },
    { id := ew "evidence.enforce-provenance", title := "make provenance audit a fatal release predicate",
      phase := .promote, authority := .evidence,
      closesFindings := ["F-evidence-provenance"],
      dependsOn := [ew "evidence.regenerate", ew "evidence.audit-tool"],
      runtimeScope := [rw "verify.evidence-integrity", rw "verify.evidence-audit"],
      touches := [.evidenceStore, .gateHarness],
      writes := ["scripts/dev/evidence_provenance_audit.py", "scripts/gate.sh",
        "scripts/lib/checks.sh"],
      authoringResources := [.sourceTree],
      verificationResources := [.evidenceStore],
      cost := 1, goalDistance := 0,
      validation := plannedValidation
        "release entry points fail on any absent commit, unresolved hash, bad roll-up, or writer mismatch"
        "run the audit after regeneration, then inject one defect from each class"
        "clean evidence passes; all four mutations fail before release publication"
        [.provenance, .semantic, .humanReview],
      recovery := "do not make a known-red audit fatal before the owner-authorized regeneration",
      progress := .planned } ]

private def observe
    (runtimeId scenario expected probe evidenceProduced : String) :
    Growth.ObservationSpec :=
  { runtimeWork := rw runtimeId, scenario, expected, probe, evidenceProduced }

private def delta
    (runtimeId : String) (kind : Growth.ChangeKind)
    (aspects : List Growth.DeltaAspect)
    (before after : Runtime.CapabilityState) (rationale : String) :
    Growth.CapabilityDelta :=
  { runtimeWork := rw runtimeId, kind, aspects, before, after, rationale }

private def promote
    (validators : List String) (obligations : List Growth.ObligationKind)
    (acceptance rollback : String) : Growth.Promotion :=
  { validators := validators.map rw, obligations, acceptance, rollback }

/-- Closed and open growth loops. A case connects runtime observations to
findings, evolution work, desired capability deltas, and reflexive validators.
The case does not encode how an agent invents a repair. -/
def growthCases : List Growth.Case :=
  [ { id := "G-upstream-contract",
      findingIds := ["F-upstream-ops-understatement",
        "F-upstream-tensor-mixin-understatement"],
      observations := [observe "verify.upstream-contract" "pinned official tinygrad tree"
        "the API and clean-checkout captures agree and generate 590 unique well-formed Lean requirements with no implicit exclusions"
        "run the offline mutation suite, local --check, official API --check, generated-target --check, and detached specification build"
        "canonical manifest, generated Lean target, mutation failures, and exact-tree build artifact"],
      evolutionWork := [ew "parity.pin-upstream"],
      deltas := [delta "verify.upstream-contract" .addCapability
        [.supportedDomain, .observability, .provenance] .missing .loadBearing
        "replace a prose candidate and unknown denominator with a reproducible foreign inventory and checked Lean requirement skeleton"],
      stage := .promoted,
      promotion := promote ["verify.upstream-contract", "verify.lean-build",
          "spec.check-growth-loop"]
        [.build, .apiContract, .semantic, .provenance, .humanReview]
        "two capture paths agree; calibrated mutations fail; targetUpstream is confirmed while every coverage cell remains unknown"
        "restore targetUpstream to unknown and retain the manifest as unpromoted research input",
      epistemic := .confirmed
        "tinygrad 19c4d736f2bc is the reviewed convergence target; this confirms only the denominator"
        "8c87034" },
    { id := "G-renderer-runtime",
      findingIds := ["F-runtime-file-replay"],
      observations := [observe "product.render-metal" "sentinel matmul"
        "runtime source is produced by renderKernel"
        "hide fixtures and dispatch a sentinel"
        "rendered source hash and numerical output"],
      evolutionWork := [ew "foundation.renderer-runtime"],
      deltas := [delta "product.render-metal" .replaceBypass
        [.architecture, .maintainability] .bypassed .loadBearing
        "replace fixture IO with the Lean renderer on the production path"],
      stage := .promoted,
      promotion := promote ["verify.unit-tests", "verify.numpy-differential"]
        [.build, .unitRegression, .numerical]
        "fixture-free runtime compiles and preserves sentinel output"
        "restore only acquisition while retaining the renderer fixes",
      epistemic := .confirmed "renderer is load-bearing" "f679bf7" },
    { id := "G-view-algebra",
      findingIds := ["F-rangeify-identity", "F-slice-drops-axes",
        "F-expand-assumes-axis1"],
      observations := [
        observe "product.compose-view" "multi-axis movement chains"
          "shape, strides, offset, and index agree with numpy"
          "run transpose/slice/expand differential cases"
          "view metadata and numerical outputs",
        observe "product.rangeify" "movement-bearing UOp"
          "movement nodes are eliminated"
          "inspect the rangeify result and trace"
          "normalized UOp and trace rows"],
      evolutionWork := [ew "foundation.view-algebra"],
      deltas := [
        delta "product.compose-view" .repairSemantics
          [.semantics, .supportedDomain] .bypassed .bounded
          "replace special-case index formulas with compositional View algebra",
        delta "product.rangeify" .replaceBypass
          [.semantics, .architecture] .bypassed .bounded
          "replace the identity scheduler pass with movement elimination"],
      stage := .promoted,
      promotion := promote ["verify.unit-tests", "verify.numpy-differential"]
        [.build, .unitRegression, .numerical, .semantic]
        "movement regressions fail on the old code and pass on the new algebra"
        "reject unsupported movement chains rather than approximate them",
      epistemic := .confirmed "bounded view algebra is load-bearing" "f679bf7" },
    { id := "G-host-correctness",
      findingIds := ["F-buffer-shape-mismatch", "F-view-parent-lifetime",
        "F-view-readback-wrong"],
      observations := [
        observe "product.ingest-tensor" "payload/shape mismatch"
          "invalid byte lengths reject before allocation use"
          "construct short and oversized tensors"
          "typed rejection result",
        observe "product.record-view" "view of temporary under LRU churn"
          "base buffer remains live"
          "force GC and more than 64 same-sized allocations"
          "unchanged bytes after churn",
        observe "product.readback-buffer" "readback from a lazy view"
          "never reinterpret parent storage as transformed output"
          "call numpy/to_bytes on transpose, slice, and expand"
          "explicit rejection until materialization exists"],
      evolutionWork := [ew "correctness.buffer-shape", ew "correctness.view-lifetime",
        ew "correctness.view-readback-safety"],
      deltas := [
        delta "product.ingest-tensor" .strengthenSafety [.safety, .semantics]
          .bounded .bounded "add the missing shape/allocation invariant",
        delta "product.record-view" .strengthenSafety [.safety]
          .bounded .bounded "retain the owning tensor for every view",
        delta "product.readback-buffer" .repairSemantics [.safety, .semantics]
          .bypassed .bounded "turn silent wrong readback into explicit failure"],
      stage := .promoted,
      promotion := promote ["verify.numpy-differential"]
        [.apiContract, .safety, .numerical]
        "negative safety cases reject and ordinary matmul/readback remains correct"
        "keep explicit rejection if full view materialization is unavailable",
      epistemic := .confirmed "silent host-side correctness failures removed" "e1d5760" },
    { id := "G-view-materialization",
      findingIds := ["F-view-materialization-missing"],
      observations := [observe "product.materialize-view" "view readback"
        "transpose, slice, expand, and chains return transformed contiguous bytes"
        "execute indexed copy kernel and compare against numpy"
        "shape, bytes, route trace, and temporary-parent result"],
      evolutionWork := [ew "view.materialize"],
      deltas := [delta "product.materialize-view" .addCapability
        [.supportedDomain, .semantics, .safety] .missing .bounded
        "make view readback a real runtime operation"],
      stage := .promoted,
      promotion := promote ["verify.numpy-differential", "verify.unit-tests"]
        [.apiContract, .safety, .numerical, .differential]
        "every supported movement form matches numpy in shape and values"
        "retain explicit view-readback rejection on any unsupported form",
      epistemic := .confirmed
        "e6241bd commits exact checked tree 790d413: rangeified ushort copy passes shape/bit/lifetime/rejection differentials"
        "two earlier candidate trees remain attached to abandoned stale-base attempts" },
    { id := "G-generated-sentinels",
      findingIds := ["F-transcribed-sentinel-codegen"],
      observations := [
        observe "product.lower-tc-matmul" "64x64x64 symbolic index domain"
          "W=min(4,N/32) preserves captured loads, stores, WMMA wiring, accumulator permutation, and dispatch dimensions"
          "re-run the symbolic enumerator over all 1024 checked index points"
          "address tuples, WMMA operand tuples, accumulator map, and dispatch plan",
        observe "product.select-matmul-route" "wide TC eligibility transition"
          "eligibility widens only with tcLaunchDims dispatch and no admitted shape has a zero grid dimension"
          "record the old and new (128,128,128)/(96,128,128)/(64,64,64) triple and test every admitted launch"
          "reviewed eligibility triple, route trace, and nonzero launch geometry",
        observe "product.lower-tc-matmul" "all 11 sentinels"
          "sentinels use generated typed TC declarations"
          "trace route and compare captured/generated/numpy outputs"
          "route, source hashes, launch geometry, and outputs"],
      evolutionWork := [ew "codegen.differential-harness", ew "codegen.warp-parameter",
        ew "codegen.typed-stores", ew "codegen.route-sentinels",
        ew "codegen.delete-transcription"],
      deltas := [delta "product.lower-tc-matmul" .replaceBypass
        [.architecture, .supportedDomain, .maintainability]
        .bounded .loadBearing
        "make parameterized TC lowering authoritative for sentinels"],
      stage := .promoted,
      promotion := promote ["verify.codegen-differential", "verify.numpy-differential"]
        [.build, .numerical, .semantic, .differential]
        "all production sentinels use generated declarations; transcription/build dependencies are absent; Nodup and 11/11 source-different execution remain green"
        "retain captured kernels only as executable references and revert only divergent generated routing",
      epistemic := .confirmed
        "9f2ab91 completes the fd945b1 generated route by deleting the per-shape declarations/parser and making semantic L12 authoritative"
        "exact tree 1401305: build/assertions green, 50/50 numerical, and 11/11 source-different captured/generated outputs bit-identical" },
    { id := "G-semantic-codegen-verifier",
      findingIds := ["F-byte-equality-gate"],
      observations := [observe "verify.codegen-differential" "captured versus generated"
        "one process compiles non-identical captured/generated sources, dispatches both over one seeded bf16 input pair with route-specific geometry, and compares attributable outputs"
        "assert source inequality, run positive and deliberately sabotaged output/geometry cases, then compare bitwise with a predeclared ULP fallback for Metal fast-math"
        "paired raw outputs, source hashes, function names, launch plans, seed, exact/ULP result, and rejection reason"],
      evolutionWork := [ew "codegen.differential-harness", ew "gates.semantic-codegen"],
      deltas := [delta "verify.codegen-differential" .addCapability
        [.semantics, .observability, .provenance] .missing .loadBearing
        "replace source identity with discriminating execution evidence"],
      stage := .promoted,
      promotion := promote ["verify.codegen-differential", "spec.check-growth-loop"]
        [.build, .numerical, .semantic, .differential, .provenance, .resourceIsolation]
        "L12 requires 11/11 execution equivalence with source inequality; wrong-but-distinct store/load mutations are caught independently of collision proofs"
        "never remove the semantic differential or the independent non-aliasing theorem",
      epistemic := .confirmed
        "aa67497 added C3 before deletion; 9f2ab91 retires only transcription byte equality and keeps 11/11 semantic execution load-bearing"
        "executable gate layer, source inequality, Nodup theorem, and reproduced wrong-but-distinct falsifiability rows" },
    { id := "G-run-isolation",
      findingIds := ["F-verification-temp-collision"],
      observations := [observe "verify.run-isolation" "concurrent script runs"
        "nested gate/devcheck artifacts remain under distinct owned roots and rangeify traces use explicit paths"
        "launch concurrent contexts, join nested consumers, attempt broad/unmarked roots, and exercise owner/non-owner cleanup"
        "root identities, artifact contents, trace paths, ownership markers, rejection results, and post-exit filesystem state"],
      evolutionWork := [ew "harness.namespace-temporaries"],
      deltas := [delta "verify.run-isolation" .addCapability
        [.safety, .observability, .maintainability] .missing .loadBearing
        "replace process-global temporary names with an owned, inherited run context"],
      stage := .promoted,
      promotion := promote ["verify.run-isolation", "verify.lean-build"]
        [.build, .semantic, .safety, .resourceIsolation]
        "the exact tree builds; direct/nested/concurrent contexts isolate artifacts; unsafe roots and missing trace paths reject"
        "serialize shared Lean builds, Metal work, timing, and evidence writes independently of temporary isolation",
      epistemic := .confirmed
        "602897e removes fixed gate/devcheck and Lean trace paths, adds owned roots, and calibrates concurrent and unsafe-root behavior"
        "exact detached tree acdbe442: build/spec report, shell syntax, devcheck entry, trace failure/success, and concurrent self-test pass" },
    { id := "G-symmetric-performance",
      findingIds := ["F-performance-methodology"],
      observations := [
        observe "verify.paired-performance-observer" "operational repeated-call instrument"
          "clean live subjects produce uniquely identified raw AB/BA observations, absolute and relative distributions, and an explicit non-kernel scope without a verdict"
          "run CPU falsifiers, then one exact-tree serial Metal smoke against the pinned tinygrad revision"
          "raw/summary hashes, subject revisions and trees, output hashes, ordering, boundary catalog, absolute throughput, ratio distribution, and kernel-speed eligibility",
        observe "verify.performance-repeatability" "identical generated-route repeats"
          "the old ratio predicate has bounded variance small enough to support a stable decision"
          "repeat the same 30/30 run serially on one code revision and GPU"
          "raw samples, per-run miss counts, ratio distributions, and within/between-run variance",
        observe "verify.performance" "paired generated matmul comparison"
          "both runtimes execute the same timed work boundary live in one interleaved session"
          "measure paired dispatch-only and end-to-end distributions serially, then repeat complete sessions"
          "paired raw samples, cache/JIT and boundary metadata, uncertainty, throughput, and thermal context"],
      evolutionWork := [ew "spec.record-regeneration-observation",
        ew "harness.paired-performance", ew "perf.rebaseline"],
      deltas := [
        delta "verify.paired-performance-observer" .addCapability
          [.performance, .observability, .provenance] .missing .bounded
          "add a calibrated paired observer without claiming boundary symmetry, repeatability, or a verdict",
        delta "verify.performance" .replaceBypass
          [.performance, .observability, .provenance] .bypassed .loadBearing
          "replace asymmetric frozen-baseline ratios with a paired experiment whose decision rule is derived from measured variance"],
      stage := .selected,
      promotion := promote ["verify.performance-repeatability", "verify.performance",
          "verify.evidence-integrity"]
        [.performance, .provenance, .resourceIsolation]
        "both sides are live and paired; repeated distributions support a predeclared variance-derived rule; reported throughput is physically plausible"
        "publish unknown/regression when variance is too large; never reuse stale ratios or choose a threshold after seeing the result",
      epistemic := .confirmed
        "42838f2 supplies the bounded observer, but the performance claim remains unpromotable: identical e90607f 30/30 runs varied from 2 to 25 to 10 misses"
        "exact-tree live smoke plus direct repeatability diagnosis; full paired multi-run variance and symmetric kernel boundaries remain missing" },
    { id := "G-evidence-provenance",
      findingIds := ["F-evidence-provenance"],
      observations := [
        observe "verify.evidence-audit" "committed and synthetic evidence"
          "the committed set fails with exact defect counts while HEAD-tied synthetic evidence passes"
          "run scripts/dev/evidence_provenance_audit.py against both directories"
          "commit/hash/roll-up/writer-agreement report",
        observe "verify.evidence-integrity" "release evidence snapshot"
          "every commit and hash resolves against the shipped tree"
          "recompute source, child-evidence, baseline, and memo hashes"
          "fatal structured provenance validation report"],
      evolutionWork := [ew "evidence.audit-tool",
        ew "spec.record-regeneration-observation", ew "evidence.regenerate",
        ew "evidence.enforce-provenance"],
      deltas := [
        delta "verify.evidence-audit" .improveObservability
          [.provenance, .observability] .missing .loadBearing
          "turn the manual evidence audit into a calibrated repeatable capability",
        delta "verify.evidence-integrity" .upgradeEvidence
          [.provenance, .observability] .bypassed .loadBearing
          "after owner-authorized regeneration, make the known-good audit fatal"],
      stage := .selected,
      promotion := promote ["verify.evidence-audit", "verify.evidence-integrity",
          "spec.check-growth-loop"]
        [.provenance, .semantic, .humanReview]
        "regenerated evidence passes the calibrated audit and release entry points enforce it"
        "keep audit diagnostic-only until regeneration; discard any dirty or unresolved snapshot",
      epistemic := .confirmed
        "7c7dc0f retains 11 script-produced files at e90607f; the current audit still reports 26/37 absent commits, 76/115 unresolved hashes, 28 roll-up disagreements, and 17 writer mismatches"
        "repeatable auditor calibrated to pass synthetic HEAD-tied evidence; the partial regeneration remains unpromoted" } ]

def itemIds : List Growth.EvolutionWorkId := workItems.map (·.id)

/-- The first complete live history recorded under the durable evolution
protocol. The attempt was authorized for two generator files, observed a
one-file diff, retained exact candidate/check provenance, and was promoted with
an explicit residual risk that routing remains strict. -/
def liveEvolutionEvents : List Event :=
  [ .attemptStarted
      { id := { value := "attempt-codegen-warp-parameter-20260726" },
        intent := ew "codegen.warp-parameter",
        actor := "claude-window-2-tab-1",
        base :=
          { commit := "1787c835b871551c9da460e35c71900b96d3876d",
            tree := "6e26bb4ad9e83f3f4f0479a2a991bc15f509b67b",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "Tgrad/Renderer/Metal.lean" },
            { kind := .modify, target := "Tgrad/Renderer/MatmulTc.lean" } ],
        lease :=
          { token := "coord-window-2-tab-1-warp-parameter",
            resources := [.sourceTree],
            validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-codegen-warp-parameter-75f856b" },
        attempt := { value := "attempt-codegen-warp-parameter-20260726" },
        tree := "0ae1b411ba394d920b7e4db7dfd188685f6ffe2d",
        observedEffects :=
          [ { kind := .modify, target := "Tgrad/Renderer/MatmulTc.lean" } ],
        summary := "parameterize W=min(4,N/32) in a wide generator while preserving strict routing" },
    .checkRecorded
      { id := { value := "check-warp-build-75f856b" },
        candidate := { value := "candidate-codegen-warp-parameter-75f856b" },
        tree := "0ae1b411ba394d920b7e4db7dfd188685f6ffe2d",
        validator := rw "verify.lean-build",
        obligation := .build,
        outcome := .passed,
        command := "lake build and devcheck, reported by the landing worker",
        artifactDigest := "sha256:71a5be78c016fe69a3f1ee8d8e73933aac5e13b253ff030dbb6ad5c6efe0eeb2" },
    .checkRecorded
      { id := { value := "check-warp-semantic-75f856b" },
        candidate := { value := "candidate-codegen-warp-parameter-75f856b" },
        tree := "0ae1b411ba394d920b7e4db7dfd188685f6ffe2d",
        validator := rw "verify.unit-tests",
        obligation := .semantic,
        outcome := .passed,
        command := "decide tcLaunchDims_matches_captured_64, wide_generates_64, and strict_rejects_64",
        artifactDigest := "sha256:0dde4e4286b60612bded22e044e13a9f6031a53bc666ea252bb1685ce2ca0e4f" },
    .checkRecorded
      { id := { value := "check-warp-symbolic-differential-75f856b" },
        candidate := { value := "candidate-codegen-warp-parameter-75f856b" },
        tree := "0ae1b411ba394d920b7e4db7dfd188685f6ffe2d",
        validator := rw "verify.unit-tests",
        obligation := .differential,
        outcome := .passed,
        command := "symbolically compare generated and captured 64x64 loads, stores, WMMA operands, and accumulator permutation",
        artifactDigest := "sha256:71a5be78c016fe69a3f1ee8d8e73933aac5e13b253ff030dbb6ad5c6efe0eeb2" },
    .promoted
      { growthCase := "G-generated-sentinels",
        candidate := { value := "candidate-codegen-warp-parameter-75f856b" },
        checkRuns :=
          [ { value := "check-warp-build-75f856b" },
            { value := "check-warp-semantic-75f856b" },
            { value := "check-warp-symbolic-differential-75f856b" } ],
        requiredObligations := [.build, .semantic, .differential],
        acceptedBy := ["harsh", "claude-window-2-tab-1"],
        target :=
          { commit := "75f856b990385fdbef215ec6e6a83423c90116a2",
            tree := "0ae1b411ba394d920b7e4db7dfd188685f6ffe2d",
            dirty := false },
        residualRisks :=
          [ "production tc eligibility remains strict",
            "PythonFFI dispatch still assumes a 128-wide N tile",
            "eligibility widening and zero-grid prevention belong to codegen.route-sentinels" ] },
    .attemptStarted
      { id := { value := "attempt-codegen-differential-harness-20260726" },
        intent := ew "codegen.differential-harness",
        actor := "claude-window-2-tab-1",
        base :=
          { commit := "75f856b990385fdbef215ec6e6a83423c90116a2",
            tree := "0ae1b411ba394d920b7e4db7dfd188685f6ffe2d",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "Main.lean" },
            { kind := .add, target := "scripts/differential_codegen.sh" } ],
        lease :=
          { token := "coord-window-2-tab-1-codegen-differential",
            resources := [.sourceTree],
            validThroughEpoch := 1000 } },
    .attemptStarted
      { id := { value := "attempt-view-materialize-20260726" },
        intent := ew "view.materialize",
        actor := "codex-primary",
        base :=
          { commit := "75f856b990385fdbef215ec6e6a83423c90116a2",
            tree := "0ae1b411ba394d920b7e4db7dfd188685f6ffe2d",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" } ],
        lease :=
          { token := "codex-primary-view-materialize",
            resources := [.sourceTree],
            validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-codegen-differential-a6d5958" },
        attempt := { value := "attempt-codegen-differential-harness-20260726" },
        tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
        observedEffects :=
          [ { kind := .modify, target := "Main.lean" },
            { kind := .add, target := "scripts/differential_codegen.sh" } ],
        summary := "execute non-identical captured and generated kernels over shared seeded inputs and compare raw output bits" },
    .checkRecorded
      { id := { value := "check-codegen-differential-build-a6d5958" },
        candidate := { value := "candidate-codegen-differential-a6d5958" },
        tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
        validator := rw "verify.lean-build",
        obligation := .build,
        outcome := .passed,
        command := "build tgrad-cli with matmul-differential and tileStoreOffsets_nodup",
        artifactDigest := "sha256:20b9ddb4e7a62f29203851413bb1b0acf46ece2cd03e59f57f7631e61a23f489" },
    .checkRecorded
      { id := { value := "check-codegen-differential-numerical-a6d5958" },
        candidate := { value := "candidate-codegen-differential-a6d5958" },
        tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
        validator := rw "verify.codegen-differential",
        obligation := .numerical,
        outcome := .passed,
        command := "bash scripts/differential_codegen.sh: 11/11 bit-identical, 240 MB compared",
        artifactDigest := "sha256:04d631bd95fd00c95d56608ce1e2ce011301275f18ae3d3d0a73b1fecf3f789e" },
    .checkRecorded
      { id := { value := "check-codegen-differential-semantic-a6d5958" },
        candidate := { value := "candidate-codegen-differential-a6d5958" },
        tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
        validator := rw "verify.codegen-differential",
        obligation := .semantic,
        outcome := .passed,
        command := "require source inequality for every sentinel; c->c+2 falsification stays Nodup but produces 727933 differing bytes",
        artifactDigest := "sha256:f10e6553cdf228322035c1c7d60ae8407c757fdb26d5f337a6502aa38df1998a" },
    .checkRecorded
      { id := { value := "check-codegen-differential-execution-a6d5958" },
        candidate := { value := "candidate-codegen-differential-a6d5958" },
        tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
        validator := rw "verify.codegen-differential",
        obligation := .differential,
        outcome := .passed,
        command := "compile both functions in one process, dispatch each geometry over one input pair, compare separately read outputs",
        artifactDigest := "sha256:04d631bd95fd00c95d56608ce1e2ce011301275f18ae3d3d0a73b1fecf3f789e" },
    .checkRecorded
      { id := { value := "check-codegen-differential-provenance-a6d5958" },
        candidate := { value := "candidate-codegen-differential-a6d5958" },
        tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
        validator := rw "verify.codegen-differential",
        obligation := .provenance,
        outcome := .passed,
        command := "retain shape, seed, function names, per-route launch plans, byte counts, source-inequality, and exact result",
        artifactDigest := "sha256:f10e6553cdf228322035c1c7d60ae8407c757fdb26d5f337a6502aa38df1998a" },
    .checkRecorded
      { id := { value := "check-codegen-differential-isolation-a6d5958" },
        candidate := { value := "candidate-codegen-differential-a6d5958" },
        tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
        validator := rw "verify.codegen-differential",
        obligation := .resourceIsolation,
        outcome := .passed,
        command := "harness allocates one run-scoped mktemp directory and removes it by trap",
        artifactDigest := "sha256:04d631bd95fd00c95d56608ce1e2ce011301275f18ae3d3d0a73b1fecf3f789e" },
    .promoted
      { growthCase := "G-semantic-codegen-verifier",
        candidate := { value := "candidate-codegen-differential-a6d5958" },
        checkRuns :=
          [ { value := "check-codegen-differential-build-a6d5958" },
            { value := "check-codegen-differential-numerical-a6d5958" },
            { value := "check-codegen-differential-semantic-a6d5958" },
            { value := "check-codegen-differential-execution-a6d5958" },
            { value := "check-codegen-differential-provenance-a6d5958" },
            { value := "check-codegen-differential-isolation-a6d5958" } ],
        requiredObligations :=
          [.build, .numerical, .semantic, .differential, .provenance,
            .resourceIsolation],
        acceptedBy := ["harsh", "claude-window-2-tab-1"],
        target :=
          { commit := "a6d59581e896495edb7cd7438d4ecf96b3dde470",
            tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
            dirty := false },
        residualRisks :=
          [ "the harness is not yet a mandatory L12 layer",
            "byte equality remains useful only while transcription exists",
            "tileStoreOffsets_nodup catches collisions; the differential catches wrong-but-distinct placement",
            "captured MSL remains the reference until generated routing is promoted" ] },
    .candidateProduced
      { id := { value := "candidate-view-materialize-995eb7e" },
        attempt := { value := "attempt-view-materialize-20260726" },
        tree := "995eb7e9964e8870800d558f496772f4d983577c",
        observedEffects :=
          [ { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" } ],
        summary := "rangeified, bit-preserving view copy plus bounded Python readback materialization" },
    .checkRecorded
      { id := { value := "check-view-materialize-build-995eb7e" },
        candidate := { value := "candidate-view-materialize-995eb7e" },
        tree := "995eb7e9964e8870800d558f496772f4d983577c",
        validator := rw "verify.lean-build",
        obligation := .build,
        outcome := .passed,
        command := "lake build Tgrad:shared; make -C c dylib; lake build TgradSpec tgrad-spec",
        artifactDigest := "sha256:8b54c1ec4d95e9abd72c203a53582231a183cb55ee6b0dcf6aa58923f1006638" },
    .checkRecorded
      { id := { value := "check-view-materialize-api-995eb7e" },
        candidate := { value := "candidate-view-materialize-995eb7e" },
        tree := "995eb7e9964e8870800d558f496772f4d983577c",
        validator := rw "verify.numpy-differential",
        obligation := .apiContract,
        outcome := .passed,
        command := "serial Metal view-materialization probe: supported readback succeeds; unsupported and empty chains raise NotInLeanScope",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-safety-995eb7e" },
        candidate := { value := "candidate-view-materialize-995eb7e" },
        tree := "995eb7e9964e8870800d558f496772f4d983577c",
        validator := rw "verify.numpy-differential",
        obligation := .safety,
        outcome := .passed,
        command := "serial Metal probe: temporary parent survives GC/LRU churn; payload mismatch and zero-sized dispatch reject",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-numerical-995eb7e" },
        candidate := { value := "candidate-view-materialize-995eb7e" },
        tree := "995eb7e9964e8870800d558f496772f4d983577c",
        validator := rw "verify.numpy-differential",
        obligation := .numerical,
        outcome := .passed,
        command := "serial Metal probe over NaNs, infinities, signed zero, subnormals, transpose, slice, reshape, expand, and chains",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-differential-995eb7e" },
        candidate := { value := "candidate-view-materialize-995eb7e" },
        tree := "995eb7e9964e8870800d558f496772f4d983577c",
        validator := rw "verify.numpy-differential",
        obligation := .differential,
        outcome := .passed,
        command := "compare exact output shape and raw bf16 layout against numpy for every supported movement form",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .attemptAbandoned
      { value := "attempt-view-materialize-20260726" }
      "HEAD advanced to a6d5958; exact-tree checks on 995eb7e cannot certify the rebased candidate",
    .attemptStarted
      { id := { value := "attempt-view-materialize-a6d5958-20260726" },
        intent := ew "view.materialize",
        actor := "codex-primary",
        base :=
          { commit := "a6d59581e896495edb7cd7438d4ecf96b3dde470",
            tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" } ],
        lease :=
          { token := "codex-primary-view-materialize-a6d5958",
            resources := [.sourceTree],
            validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-view-materialize-ac393d2" },
        attempt := { value := "attempt-view-materialize-a6d5958-20260726" },
        tree := "ac393d25d6afe54c0039e53ededa8df8a9cf7237",
        observedEffects :=
          [ { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" } ],
        summary := "rebase rangeified, bit-preserving view materialization onto promoted differential harness" },
    .checkRecorded
      { id := { value := "check-view-materialize-build-ac393d2" },
        candidate := { value := "candidate-view-materialize-ac393d2" },
        tree := "ac393d25d6afe54c0039e53ededa8df8a9cf7237",
        validator := rw "verify.lean-build",
        obligation := .build,
        outcome := .passed,
        command := "clean detached a6d5958 worktree: lake build Tgrad:shared; make -C c dylib",
        artifactDigest := "sha256:ad8b2c07fd9663fc486a795dbf7945a53bdf56a872112dd9d3297dd4c4cf8357" },
    .checkRecorded
      { id := { value := "check-view-materialize-api-ac393d2" },
        candidate := { value := "candidate-view-materialize-ac393d2" },
        tree := "ac393d25d6afe54c0039e53ededa8df8a9cf7237",
        validator := rw "verify.numpy-differential",
        obligation := .apiContract,
        outcome := .passed,
        command := "isolated serial Metal probe: supported readback succeeds; unsupported and empty chains raise NotInLeanScope",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-safety-ac393d2" },
        candidate := { value := "candidate-view-materialize-ac393d2" },
        tree := "ac393d25d6afe54c0039e53ededa8df8a9cf7237",
        validator := rw "verify.numpy-differential",
        obligation := .safety,
        outcome := .passed,
        command := "isolated serial Metal probe: temporary parent survives GC/LRU churn; malformed payload and zero dispatch reject",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-numerical-ac393d2" },
        candidate := { value := "candidate-view-materialize-ac393d2" },
        tree := "ac393d25d6afe54c0039e53ededa8df8a9cf7237",
        validator := rw "verify.numpy-differential",
        obligation := .numerical,
        outcome := .passed,
        command := "isolated serial Metal probe over exact NaN, infinity, signed-zero, subnormal, and ordinary bf16 payloads",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-differential-ac393d2" },
        candidate := { value := "candidate-view-materialize-ac393d2" },
        tree := "ac393d25d6afe54c0039e53ededa8df8a9cf7237",
        validator := rw "verify.numpy-differential",
        obligation := .differential,
        outcome := .passed,
        command := "isolated serial Metal probe: exact shape and bf16 layout match numpy for transpose/slice/reshape/expand/chains",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .attemptStarted
      { id := { value := "attempt-gates-semantic-codegen-20260726" },
        intent := ew "gates.semantic-codegen",
        actor := "claude-window-2-tab-1",
        base :=
          { commit := "a6d59581e896495edb7cd7438d4ecf96b3dde470",
            tree := "d94731a9fd44f755af5cdf57690f5c401e207e15",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "scripts/gates/L12.sh" },
            { kind := .modify, target := "scripts/gates/L12_falsifiability.md" } ],
        lease :=
          { token := "coord-window-2-tab-1-gates-semantic-codegen",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-gates-semantic-codegen-aa67497" },
        attempt := { value := "attempt-gates-semantic-codegen-20260726" },
        tree := "899f0f6619a76efd2666fb7b00033b58da59ec44",
        observedEffects :=
          [ { kind := .modify, target := "scripts/gates/L12.sh" },
            { kind := .modify, target := "scripts/gates/L12_falsifiability.md" } ],
        summary := "add semantic C3 before transcription deletion while preserving green byte-equality Layer C" },
    .checkRecorded
      { id := { value := "check-gates-semantic-build-aa67497" },
        candidate := { value := "candidate-gates-semantic-codegen-aa67497" },
        tree := "899f0f6619a76efd2666fb7b00033b58da59ec44",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "build remains green with tileStoreOffsets_nodup and both reproduced wrong-address mutations",
        artifactDigest := "sha256:dfb3613f10f41605c15dd29c44f5811db2460744ec737bebc70746929ef0c949" },
    .checkRecorded
      { id := { value := "check-gates-semantic-layer-aa67497" },
        candidate := { value := "candidate-gates-semantic-codegen-aa67497" },
        tree := "899f0f6619a76efd2666fb7b00033b58da59ec44",
        validator := rw "verify.codegen-differential", obligation := .semantic,
        outcome := .passed,
        command := "L12 C3 is additive; original green Layer C remains byte-for-byte intact",
        artifactDigest := "sha256:749783deae6da1bb2f96e75931484709b9938b1a439e33b27854ef302b4abdc2" },
    .checkRecorded
      { id := { value := "check-gates-semantic-differential-aa67497" },
        candidate := { value := "candidate-gates-semantic-codegen-aa67497" },
        tree := "899f0f6619a76efd2666fb7b00033b58da59ec44",
        validator := rw "verify.codegen-differential", obligation := .differential,
        outcome := .passed,
        command := "standalone C3: 11/11 bit-identical in 5.1s; c->c+2 and 24K+1->24K+2 both 11/11 divergent while build stays green",
        artifactDigest := "sha256:dfb3613f10f41605c15dd29c44f5811db2460744ec737bebc70746929ef0c949" },
    .checkRecorded
      { id := { value := "check-gates-semantic-provenance-aa67497" },
        candidate := { value := "candidate-gates-semantic-codegen-aa67497" },
        tree := "899f0f6619a76efd2666fb7b00033b58da59ec44",
        validator := rw "verify.codegen-differential", obligation := .provenance,
        outcome := .passed,
        command := "C3 requires diff_sources_byte_equal=0 and records the semantic layer in reproduced falsifiability rows",
        artifactDigest := "sha256:749783deae6da1bb2f96e75931484709b9938b1a439e33b27854ef302b4abdc2" },
    .promoted
      { growthCase := "G-semantic-codegen-verifier",
        candidate := { value := "candidate-gates-semantic-codegen-aa67497" },
        checkRuns :=
          [ { value := "check-gates-semantic-build-aa67497" },
            { value := "check-gates-semantic-layer-aa67497" },
            { value := "check-gates-semantic-differential-aa67497" },
            { value := "check-gates-semantic-provenance-aa67497" } ],
        requiredObligations := [.build, .semantic, .differential, .provenance],
        acceptedBy := ["harsh", "claude-window-2-tab-1"],
        target :=
          { commit := "aa67497412c9e91b9f928adeaecf87494a7bf7a1",
            tree := "899f0f6619a76efd2666fb7b00033b58da59ec44",
            dirty := false },
        residualRisks :=
          [ "Layer C remains a transcription round-trip until deletion",
            "C3 and tileStoreOffsets_nodup are complementary and both must remain",
            "the full gate was not run because it rewrites committed evidence" ] },
    .attemptStarted
      { id := { value := "attempt-evidence-audit-tool-20260726" },
        intent := ew "evidence.audit-tool",
        actor := "claude-window-2-tab-1",
        base :=
          { commit := "aa67497412c9e91b9f928adeaecf87494a7bf7a1",
            tree := "899f0f6619a76efd2666fb7b00033b58da59ec44",
            dirty := false },
        authorizedEffects :=
          [ { kind := .add, target := "scripts/dev/evidence_provenance_audit.py" } ],
        lease :=
          { token := "coord-window-2-tab-1-evidence-audit",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-evidence-audit-bdc01b0" },
        attempt := { value := "attempt-evidence-audit-tool-20260726" },
        tree := "658c79a62957e756a3c302134556d6615a0c5911",
        observedEffects :=
          [ { kind := .add, target := "scripts/dev/evidence_provenance_audit.py" } ],
        summary := "calibrated nonfatal audit of evidence commit, hashes, roll-ups, and writer agreement" },
    .checkRecorded
      { id := { value := "check-evidence-audit-semantic-bdc01b0" },
        candidate := { value := "candidate-evidence-audit-bdc01b0" },
        tree := "658c79a62957e756a3c302134556d6615a0c5911",
        validator := rw "verify.evidence-audit", obligation := .semantic,
        outcome := .passed,
        command := "synthetic HEAD-tied evidence returns OK; committed evidence returns FAIL",
        artifactDigest := "sha256:bf908e6fcf5238ee642c0f5e8d2a49300a9f244cca0bf946d3bc6a838aa4e70f" },
    .checkRecorded
      { id := { value := "check-evidence-audit-provenance-bdc01b0" },
        candidate := { value := "candidate-evidence-audit-bdc01b0" },
        tree := "658c79a62957e756a3c302134556d6615a0c5911",
        validator := rw "verify.evidence-audit", obligation := .provenance,
        outcome := .passed,
        command := "37/37 absent commit; 73/115 unresolved hashes; 28 roll-up disagreements; 27 writer-key mismatches",
        artifactDigest := "sha256:bf908e6fcf5238ee642c0f5e8d2a49300a9f244cca0bf946d3bc6a838aa4e70f" },
    .promoted
      { growthCase := "G-evidence-provenance",
        candidate := { value := "candidate-evidence-audit-bdc01b0" },
        checkRuns :=
          [ { value := "check-evidence-audit-semantic-bdc01b0" },
            { value := "check-evidence-audit-provenance-bdc01b0" } ],
        requiredObligations := [.semantic, .provenance],
        acceptedBy := ["harsh", "claude-window-2-tab-1"],
        target :=
          { commit := "bdc01b007c05afc4fe674e7939559778dcb54ebd",
            tree := "658c79a62957e756a3c302134556d6615a0c5911",
            dirty := false },
        residualRisks :=
          [ "committed evidence remains invalid",
            "the audit is deliberately nonfatal until owner-authorized regeneration",
            "regeneration requires a serial full gate sweep and rewrites 37 files" ] },
    .attemptAbandoned
      { value := "attempt-view-materialize-a6d5958-20260726" }
      "HEAD advanced through aa67497 and bdc01b0; ac393d2 checks remain exact only for the a6d5958-based tree",
    .attemptStarted
      { id := { value := "attempt-view-materialize-bdc01b0-20260726" },
        intent := ew "view.materialize",
        actor := "codex-primary",
        base :=
          { commit := "bdc01b007c05afc4fe674e7939559778dcb54ebd",
            tree := "658c79a62957e756a3c302134556d6615a0c5911",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" } ],
        lease :=
          { token := "codex-primary-view-materialize-bdc01b0",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-view-materialize-790d413" },
        attempt := { value := "attempt-view-materialize-bdc01b0-20260726" },
        tree := "790d41311b563c31dad92989ebe17081519ea771",
        observedEffects :=
          [ { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" } ],
        summary := "final bdc01b0-based rangeified bit-preserving view materialization" },
    .checkRecorded
      { id := { value := "check-view-materialize-build-790d413" },
        candidate := { value := "candidate-view-materialize-790d413" },
        tree := "790d41311b563c31dad92989ebe17081519ea771",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "clean detached bdc01b0 worktree: lake build Tgrad:shared; make -C c dylib",
        artifactDigest := "sha256:44b2baf39529cb1bb9381dd87a65674cff342194c1598dc6e257e1052fa99813" },
    .checkRecorded
      { id := { value := "check-view-materialize-api-790d413" },
        candidate := { value := "candidate-view-materialize-790d413" },
        tree := "790d41311b563c31dad92989ebe17081519ea771",
        validator := rw "verify.numpy-differential", obligation := .apiContract,
        outcome := .passed,
        command := "final isolated Metal probe: supported readback succeeds; unsupported and empty chains reject before dispatch",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-safety-790d413" },
        candidate := { value := "candidate-view-materialize-790d413" },
        tree := "790d41311b563c31dad92989ebe17081519ea771",
        validator := rw "verify.numpy-differential", obligation := .safety,
        outcome := .passed,
        command := "final isolated Metal probe: base retention, payload/allocation agreement, bounds, and zero-dispatch rejection",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-numerical-790d413" },
        candidate := { value := "candidate-view-materialize-790d413" },
        tree := "790d41311b563c31dad92989ebe17081519ea771",
        validator := rw "verify.numpy-differential", obligation := .numerical,
        outcome := .passed,
        command := "final isolated Metal probe preserves NaN, infinity, signed-zero, subnormal, and ordinary bf16 bits",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .checkRecorded
      { id := { value := "check-view-materialize-differential-790d413" },
        candidate := { value := "candidate-view-materialize-790d413" },
        tree := "790d41311b563c31dad92989ebe17081519ea771",
        validator := rw "verify.numpy-differential", obligation := .differential,
        outcome := .passed,
        command := "final isolated Metal probe matches numpy shape/layout for transpose, partial/multi/strided slice, reshape, expand, and chains",
        artifactDigest := "sha256:274c1d0b83fb67cc8dab4b50d379ac842dbbed5664aacf18154b91f9fccaa983" },
    .promoted
      { growthCase := "G-view-materialization",
        candidate := { value := "candidate-view-materialize-790d413" },
        checkRuns :=
          [ { value := "check-view-materialize-build-790d413" },
            { value := "check-view-materialize-api-790d413" },
            { value := "check-view-materialize-safety-790d413" },
            { value := "check-view-materialize-numerical-790d413" },
            { value := "check-view-materialize-differential-790d413" } ],
        requiredObligations := [.build, .apiContract, .safety, .numerical, .differential],
        acceptedBy := ["codex-primary"],
        target :=
          { commit := "e6241bdbf15c6a2ff594ebae0d551521ffcbf689",
            tree := "790d41311b563c31dad92989ebe17081519ea771",
            dirty := false },
        residualRisks :=
          [ "each view readback currently compiles and dispatches a fresh copy",
            "the Lean tensor registry and Metal pipeline cache remain append-only",
            "the overloaded tgrad_matmul_view(handle,0) ABI should become a named materialize entry" ] },
    .attemptStarted
      { id := { value := "attempt-route-sentinels-20260726" },
        intent := ew "codegen.route-sentinels",
        actor := "codex-primary",
        base :=
          { commit := "254fe08c2c6a554d681311ee535826a931c6b8c2",
            tree := "c04850ceb06f6fb7e44f3bdb2b19f918fbb6f55a",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" },
            { kind := .modify, target := "Tgrad/Renderer/MatmulTc.lean" },
            { kind := .modify, target := "Tgrad/Codegen/Opt/Heuristic.lean" },
            { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "scripts/devcheck.sh" } ],
        lease :=
          { token := "codex-primary-route-sentinels",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-route-sentinels-8475550" },
        attempt := { value := "attempt-route-sentinels-20260726" },
        tree := "84755502c57bb5cc459b51b488080c81c20384bd",
        observedEffects :=
          [ { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" },
            { kind := .modify, target := "Tgrad/Renderer/MatmulTc.lean" },
            { kind := .modify, target := "Tgrad/Codegen/Opt/Heuristic.lean" },
            { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "scripts/devcheck.sh" } ],
        summary := "make wide generated declarations, names, eligibility, and launch geometry authoritative for sentinel and aligned TC dispatch" },
    .checkRecorded
      { id := { value := "check-route-sentinels-build-fd945b1" },
        candidate := { value := "candidate-route-sentinels-8475550" },
        tree := "84755502c57bb5cc459b51b488080c81c20384bd",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "lake build Tgrad:shared TgradSpec tgrad-spec tgrad-tests tgrad-cli; make -C c dylib",
        artifactDigest := "sha256:16c156e60e6cc84ac2f60b689c5eadeea15c3fce69aea0b052da43fac36672d0" },
    .checkRecorded
      { id := { value := "check-route-sentinels-api-fd945b1" },
        candidate := { value := "candidate-route-sentinels-8475550" },
        tree := "84755502c57bb5cc459b51b488080c81c20384bd",
        validator := rw "verify.numpy-differential", obligation := .apiContract,
        outcome := .passed,
        command := "Python FFI eligibility: 128/96/64 accepted; M=31, K=7, and N=48 rejected",
        artifactDigest := "sha256:9ff1d0505956a9e85ce8a01aaab6b6ecbbb6c72745dd65b52ba85c37a739917d" },
    .checkRecorded
      { id := { value := "check-route-sentinels-safety-fd945b1" },
        candidate := { value := "candidate-route-sentinels-8475550" },
        tree := "84755502c57bb5cc459b51b488080c81c20384bd",
        validator := rw "verify.unit-tests", obligation := .safety,
        outcome := .passed,
        command := "decide generatedDispatchDimsFor_nonzero and generatedDispatchDimsFor_matches_capture for every sentinel",
        artifactDigest := "sha256:16c156e60e6cc84ac2f60b689c5eadeea15c3fce69aea0b052da43fac36672d0" },
    .checkRecorded
      { id := { value := "check-route-sentinels-numerical-fd945b1" },
        candidate := { value := "candidate-route-sentinels-8475550" },
        tree := "84755502c57bb5cc459b51b488080c81c20384bd",
        validator := rw "verify.numpy-differential", obligation := .numerical,
        outcome := .passed,
        command := "actual Python/C/Lean/Metal production route at 64x64x64, 96x128x128, and 128x128x128 versus numpy bf16 reference",
        artifactDigest := "sha256:9ff1d0505956a9e85ce8a01aaab6b6ecbbb6c72745dd65b52ba85c37a739917d" },
    .checkRecorded
      { id := { value := "check-route-sentinels-differential-fd945b1" },
        candidate := { value := "candidate-route-sentinels-8475550" },
        tree := "84755502c57bb5cc459b51b488080c81c20384bd",
        validator := rw "verify.codegen-differential", obligation := .differential,
        outcome := .passed,
        command := "bash scripts/differential_codegen.sh: 11/11 source-different and bit-identical over 240 MB",
        artifactDigest := "sha256:04d631bd95fd00c95d56608ce1e2ce011301275f18ae3d3d0a73b1fecf3f789e" },
    .promoted
      { growthCase := "G-generated-sentinels",
        candidate := { value := "candidate-route-sentinels-8475550" },
        checkRuns :=
          [ { value := "check-route-sentinels-build-fd945b1" },
            { value := "check-route-sentinels-api-fd945b1" },
            { value := "check-route-sentinels-safety-fd945b1" },
            { value := "check-route-sentinels-numerical-fd945b1" },
            { value := "check-route-sentinels-differential-fd945b1" } ],
        requiredObligations := [.build, .apiContract, .safety, .numerical, .differential],
        acceptedBy := ["harsh", "codex-primary"],
        target :=
          { commit := "fd945b10725da82edab8afa5246dba56452e4403",
            tree := "84755502c57bb5cc459b51b488080c81c20384bd",
            dirty := false },
        residualRisks :=
          [ "MatmulDecls and lower_matmul remain as transcription scaffolding",
            "L12 Layer C still checks transcription byte equality until deletion",
            "performance has not been rebaselined across a symmetric boundary",
            "dead threadgroup declarations and barrier marker remain structural-gate debt" ] },
    .attemptStarted
      { id := { value := "attempt-delete-transcription-20260726" },
        intent := ew "codegen.delete-transcription",
        actor := "codex-primary",
        base :=
          { commit := "96d3790e736a755bf3c8af7ea37033190b82cd17",
            tree := "37f64c5d6937aedbec84db4b0118c01f53d18889",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "EXPERIMENT_RESULT.md" },
            { kind := .modify, target := "GROWING_TGRAD.md" },
            { kind := .modify, target := "Main.lean" },
            { kind := .modify, target := "PLAN_CORRECTNESS_AND_CODEGEN.md" },
            { kind := .modify, target := "PLAN_TINYGRAD_COMPAT.md" },
            { kind := .modify, target := "README.md" },
            { kind := .modify, target := "Tgrad.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .delete, target := "Tgrad/Renderer/MatmulDecls.lean" },
            { kind := .modify, target := "Tgrad/Renderer/MatmulScalar.lean" },
            { kind := .modify, target := "Tgrad/Renderer/Metal.lean" },
            { kind := .modify, target := "c/tgrad_python.c" },
            { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "scripts/dev/l15_b_audit.py" },
            { kind := .delete, target := "scripts/dev/lower_matmul.py" },
            { kind := .modify, target := "scripts/differential_codegen.sh" },
            { kind := .modify, target := "scripts/gates/L12.sh" },
            { kind := .modify, target := "scripts/gates/L12_falsifiability.md" },
            { kind := .modify, target := "scripts/gates/L13_F_STRICT_B_falsifiability.md" },
            { kind := .modify, target := "scripts/gates/L14_A.sh" },
            { kind := .modify, target := "scripts/gates/L14_B_2_a_falsifiability.md" },
            { kind := .modify, target := "scripts/gates/L14_B_2_b.sh" },
            { kind := .modify, target := "scripts/gates/L14_B_2_b_falsifiability.md" } ],
        lease :=
          { token := "codex-primary-delete-transcription",
            resources := [.sourceTree], validThroughEpoch := 2000 } },
    .candidateProduced
      { id := { value := "candidate-delete-transcription-1401305" },
        attempt := { value := "attempt-delete-transcription-20260726" },
        tree := "1401305a6facaa4cc99ec6c5c266ed24fe925ad5",
        observedEffects :=
          [ { kind := .modify, target := "EXPERIMENT_RESULT.md" },
            { kind := .modify, target := "GROWING_TGRAD.md" },
            { kind := .modify, target := "Main.lean" },
            { kind := .modify, target := "PLAN_CORRECTNESS_AND_CODEGEN.md" },
            { kind := .modify, target := "PLAN_TINYGRAD_COMPAT.md" },
            { kind := .modify, target := "README.md" },
            { kind := .modify, target := "Tgrad.lean" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" },
            { kind := .modify, target := "Tgrad/PythonFFI.lean" },
            { kind := .delete, target := "Tgrad/Renderer/MatmulDecls.lean" },
            { kind := .modify, target := "Tgrad/Renderer/MatmulScalar.lean" },
            { kind := .modify, target := "Tgrad/Renderer/Metal.lean" },
            { kind := .modify, target := "c/tgrad_python.c" },
            { kind := .modify, target := "python/tgrad.py" },
            { kind := .modify, target := "scripts/dev/l15_b_audit.py" },
            { kind := .delete, target := "scripts/dev/lower_matmul.py" },
            { kind := .modify, target := "scripts/differential_codegen.sh" },
            { kind := .modify, target := "scripts/gates/L12.sh" },
            { kind := .modify, target := "scripts/gates/L12_falsifiability.md" },
            { kind := .modify, target := "scripts/gates/L13_F_STRICT_B_falsifiability.md" },
            { kind := .modify, target := "scripts/gates/L14_A.sh" },
            { kind := .modify, target := "scripts/gates/L14_B_2_a_falsifiability.md" },
            { kind := .modify, target := "scripts/gates/L14_B_2_b.sh" },
            { kind := .modify, target := "scripts/gates/L14_B_2_b_falsifiability.md" } ],
        summary := "delete transcription artifacts, switch CLI and audits to generated declarations, and make semantic L12 authoritative" },
    .checkRecorded
      { id := { value := "check-delete-transcription-build-9f2ab91" },
        candidate := { value := "candidate-delete-transcription-1401305" },
        tree := "1401305a6facaa4cc99ec6c5c266ed24fe925ad5",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "lake build Tgrad:shared TgradSpec tgrad-spec tgrad-tests tgrad-cli; make -C c dylib",
        artifactDigest := "sha256:fd8cb271292b84d9d5964469aef7daaaca2d0b4804fba67843b0713b77fc7e7a" },
    .checkRecorded
      { id := { value := "check-delete-transcription-unit-9f2ab91" },
        candidate := { value := "candidate-delete-transcription-1401305" },
        tree := "1401305a6facaa4cc99ec6c5c266ed24fe925ad5",
        validator := rw "verify.unit-tests", obligation := .unitRegression,
        outcome := .passed,
        command := ".lake/build/bin/tgrad-tests; require assertion marker and real View/rangeify assertions",
        artifactDigest := "sha256:a7a3396240e8acbf0c2110996c37b5da957d98e5fcddb989b334b8416a260388" },
    .checkRecorded
      { id := { value := "check-delete-transcription-semantic-9f2ab91" },
        candidate := { value := "candidate-delete-transcription-1401305" },
        tree := "1401305a6facaa4cc99ec6c5c266ed24fe925ad5",
        validator := rw "verify.codegen-differential", obligation := .semantic,
        outcome := .passed,
        command := "exact-tree git audit: deleted files/imports/old dispatch/byte-equality layer absent; generated route and Nodup theorem names present",
        artifactDigest := "sha256:8df85b5b7553f26b87658a2fd5c1269ebda0c09395dfac62db7ff83d9692b2f8" },
    .checkRecorded
      { id := { value := "check-delete-transcription-numerical-9f2ab91" },
        candidate := { value := "candidate-delete-transcription-1401305" },
        tree := "1401305a6facaa4cc99ec6c5c266ed24fe925ad5",
        validator := rw "verify.numpy-differential", obligation := .numerical,
        outcome := .passed,
        command := "serial alternate-generated-cache sweep: 50/50 distributions correct; legacy performance ratio is diagnostic only",
        artifactDigest := "sha256:cf7f9ad4859b5aad1a8d0a9d2d92a89ba05d3ef3ec640db25b7b315ceda5a60f" },
    .checkRecorded
      { id := { value := "check-delete-transcription-differential-9f2ab91" },
        candidate := { value := "candidate-delete-transcription-1401305" },
        tree := "1401305a6facaa4cc99ec6c5c266ed24fe925ad5",
        validator := rw "verify.codegen-differential", obligation := .differential,
        outcome := .passed,
        command := "bash scripts/differential_codegen.sh: 11/11 source-different and bit-identical over 240 MB",
        artifactDigest := "sha256:d594b04cf28e07e50c37b2340ba5eddce1018a9c9c78e77563bf2571545f19aa" },
    .promoted
      { growthCase := "G-generated-sentinels",
        candidate := { value := "candidate-delete-transcription-1401305" },
        checkRuns :=
          [ { value := "check-delete-transcription-build-9f2ab91" },
            { value := "check-delete-transcription-unit-9f2ab91" },
            { value := "check-delete-transcription-semantic-9f2ab91" },
            { value := "check-delete-transcription-numerical-9f2ab91" },
            { value := "check-delete-transcription-differential-9f2ab91" } ],
        requiredObligations := [.build, .unitRegression, .semantic, .numerical, .differential],
        acceptedBy := ["harsh", "codex-primary"],
        target :=
          { commit := "9f2ab912cb25d8b6bc2c13d4dca12e4ab7330d7f",
            tree := "1401305a6facaa4cc99ec6c5c266ed24fe925ad5",
            dirty := false },
        residualRisks :=
          [ "performance remains unknown until symmetric same-session measurement",
            "captured MSL remains as an executable differential oracle",
            "committed release evidence still has absent-commit and stale-hash provenance",
            "alternate generated cache and historical gates retain migration debt",
            "many untouched gates still share fixed /tmp paths" ] },
    /- `7c7dc0f` is deliberately a candidate, not a promotion. The serial
       regeneration repaired two real gate blockers and retained the eleven
       evidence files that their scripts could reproduce. L11 then falsified
       the performance predicate's repeatability, so no replacement L11
       evidence or all-green snapshot was accepted. A committed candidate can
       still be an abandoned attempt. -/
    .attemptStarted
      { id := { value := "attempt-evidence-regeneration-20260726" },
        intent := ew "evidence.regenerate",
        actor := "claude-window-2-tab-1",
        base :=
          { commit := "7dff169a08876e79e0c4d657cca386dacb8e199e",
            tree := "2e4bc9a16e22af0bf5538f36f1feb89d98f03894",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "scripts/gates/L12.sh" },
            { kind := .modify, target := "scripts/gates/L14_B_1.sh" },
            { kind := .modify, target := "fixtures/gate_evidence/L0.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L1.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L2.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L3.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L4.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L5.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L6.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L7.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L8.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L9.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L10.json" } ],
        lease :=
          { token := "coord-window-2-tab-1-evidence-regeneration",
            resources := [.sourceTree, .tmpNamespace, .metalGpu, .evidenceStore],
            validThroughEpoch := 3000 } },
    .candidateProduced
      { id := { value := "candidate-evidence-regeneration-5ccb293" },
        attempt := { value := "attempt-evidence-regeneration-20260726" },
        tree := "5ccb293d45874a5cde3f47d1d0516dce51359f1c",
        observedEffects :=
          [ { kind := .modify, target := "scripts/gates/L12.sh" },
            { kind := .modify, target := "scripts/gates/L14_B_1.sh" },
            { kind := .modify, target := "fixtures/gate_evidence/L0.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L1.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L2.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L3.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L4.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L5.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L6.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L7.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L8.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L9.json" },
            { kind := .modify, target := "fixtures/gate_evidence/L10.json" } ],
        summary := "repair two regeneration blockers and retain eleven script-produced evidence files while leaving the non-repeatable L11 result red" },
    .checkRecorded
      { id := { value := "check-partial-regeneration-provenance-7c7dc0f" },
        candidate := { value := "candidate-evidence-regeneration-5ccb293" },
        tree := "5ccb293d45874a5cde3f47d1d0516dce51359f1c",
        validator := rw "verify.evidence-audit", obligation := .provenance,
        outcome := .failed
          "only 11/37 files are reachable; 26 absent commits, 76/115 unresolved hashes, 28 bad roll-ups, and 17 writer-key mismatches remain",
        command := "python3 scripts/dev/evidence_provenance_audit.py",
        artifactDigest := "sha256:b79b7ad39140abb95b83cba7007ec5541558bcf205d8bc18fb77c4bb6cfc3f64" },
    .checkRecorded
      { id := { value := "check-L11-repeatability-e90607f" },
        candidate := { value := "candidate-evidence-regeneration-5ccb293" },
        tree := "5ccb293d45874a5cde3f47d1d0516dce51359f1c",
        validator := rw "verify.performance-repeatability", obligation := .performance,
        outcome := .failed
          "identical consecutive 30/30 runs missed 2/50, 25/50, and 10/50 with ratio maxima 1.655, 3.667, and 2.552",
        command := "reported serial L11 repeats on e90607f and one GPU; raw sample files were not committed, so the summary falsifies repeatability but cannot validate parity",
        artifactDigest := "sha256:a6336dee9a134b36c8e3a53bf331b28d7c3496a66f2abcbfb1cba70e2d7a9bc4" },
    .checkRecorded
      { id := { value := "check-L12-sample-sensitivity-e90607f" },
        candidate := { value := "candidate-evidence-regeneration-5ccb293" },
        tree := "5ccb293d45874a5cde3f47d1d0516dce51359f1c",
        validator := rw "verify.performance-repeatability", obligation := .performance,
        outcome := .failed
          "the same generated route changed from median/max 2.38/4.24 and 37 misses at 1/1 to 1.18/1.41 and zero misses at 30/30",
        command := "compare the reported L12 C2 1/1 and 30/30 diagnostics on e90607f; correctness remains 50/50",
        artifactDigest := "sha256:029ce9086b149fdfce30e023c8a09007e2818f4283c9746a388e0208d26ffae7" },
    .attemptAbandoned
      { value := "attempt-evidence-regeneration-20260726" }
      "perf.rebaseline is still unresolved and L11 is empirically non-repeatable; retain the 11 reproducible files and gate repairs as diagnostic progress, but do not promote a partial or threshold-tuned snapshot",
    .attemptStarted
      { id := { value := "attempt-record-regeneration-observation-20260726" },
        intent := ew "spec.record-regeneration-observation",
        actor := "codex-primary",
        base :=
          { commit := "7c7dc0ffe3a9c48b64c1b326509bb23c9f874056",
            tree := "5ccb293d45874a5cde3f47d1d0516dce51359f1c",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "EXPERIMENT_RESULT.md" },
            { kind := .modify, target := "GROWING_TGRAD.md" },
            { kind := .modify, target := "PLAN_CORRECTNESS_AND_CODEGEN.md" },
            { kind := .modify, target := "PLAN_TINYGRAD_COMPAT.md" },
            { kind := .modify, target := "README.md" },
            { kind := .modify, target := "Tgrad/Spec/Findings.lean" },
            { kind := .modify, target := "Tgrad/Spec/RuntimeWork.lean" },
            { kind := .modify, target := "Tgrad/Spec/Work.lean" } ],
        lease :=
          { token := "codex-primary-record-regeneration-observation",
            resources := [.sourceTree], validThroughEpoch := 3000 } },
    .candidateProduced
      { id := { value := "candidate-record-regeneration-c4bcee8" },
        attempt := { value := "attempt-record-regeneration-observation-20260726" },
        tree := "c4bcee871fe94edc99a37de2d71514036af7fa55",
        observedEffects :=
          [ { kind := .modify, target := "EXPERIMENT_RESULT.md" },
            { kind := .modify, target := "GROWING_TGRAD.md" },
            { kind := .modify, target := "PLAN_CORRECTNESS_AND_CODEGEN.md" },
            { kind := .modify, target := "PLAN_TINYGRAD_COMPAT.md" },
            { kind := .modify, target := "README.md" },
            { kind := .modify, target := "Tgrad/Spec/Findings.lean" },
            { kind := .modify, target := "Tgrad/Spec/RuntimeWork.lean" },
            { kind := .modify, target := "Tgrad/Spec/Work.lean" } ],
        summary := "separate failed repeatability diagnosis from performance validation, preserve the partial evidence attempt, and update the public account" },
    .checkRecorded
      { id := { value := "check-record-regeneration-build-0f4d594" },
        candidate := { value := "candidate-record-regeneration-c4bcee8" },
        tree := "c4bcee871fe94edc99a37de2d71514036af7fa55",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "lake build TgradSpec tgrad-spec",
        artifactDigest := "sha256:66366e930aa6dc15c8a350eac434f10f14fbc8107227faf8ba52677f0365a46c" },
    .checkRecorded
      { id := { value := "check-record-regeneration-semantic-0f4d594" },
        candidate := { value := "candidate-record-regeneration-c4bcee8" },
        tree := "c4bcee871fe94edc99a37de2d71514036af7fa55",
        validator := rw "spec.check-growth-loop", obligation := .semantic,
        outcome := .passed,
        command := ".lake/build/bin/tgrad-spec: 25 capabilities, two open findings, no active work, perf.rebaseline sole frontier, 73 replayed events",
        artifactDigest := "sha256:6b027885a7a0f104fbe20e01d1ae0c624408ef9fba1108eb773a367c52cd5b88" },
    .checkRecorded
      { id := { value := "check-record-regeneration-provenance-0f4d594" },
        candidate := { value := "candidate-record-regeneration-c4bcee8" },
        tree := "c4bcee871fe94edc99a37de2d71514036af7fa55",
        validator := rw "verify.evidence-audit", obligation := .provenance,
        outcome := .passed,
        command := "require evidence_provenance to fail with the exact current tuple 26/37 absent, 76/115 unresolved, 28 roll-up, 17 writer mismatch",
        artifactDigest := "sha256:b79b7ad39140abb95b83cba7007ec5541558bcf205d8bc18fb77c4bb6cfc3f64" },
    .promoted
      { growthCase := "G-evidence-provenance",
        candidate := { value := "candidate-record-regeneration-c4bcee8" },
        checkRuns :=
          [ { value := "check-record-regeneration-build-0f4d594" },
            { value := "check-record-regeneration-semantic-0f4d594" },
            { value := "check-record-regeneration-provenance-0f4d594" } ],
        requiredObligations := [.build, .semantic, .provenance],
        acceptedBy := ["harsh", "codex-primary"],
        target :=
          { commit := "0f4d594d7a1112eb11d43130b7b88e47c9d0d9a5",
            tree := "c4bcee871fe94edc99a37de2d71514036af7fa55",
            dirty := false },
        residualRisks :=
          [ "raw timing samples from the three L11 repeats were not committed",
            "verify.performance remains bypassed until both sides are paired live",
            "26 evidence files still name the phantom commit",
            "the current commit adds the promotion record after the promoted candidate tree" ] },
    .attemptStarted
      { id := { value := "attempt-parity-pin-upstream-20260727" },
        intent := ew "parity.pin-upstream", actor := "agent-boyle",
        base :=
          { commit := "d82d97992fcae3a9314bb45f5885505c70fa0f0a",
            tree := "95bdfcc22ed893bca8dc88205f61fe002166cc6a",
            dirty := false },
        authorizedEffects :=
          [ { kind := .add, target := "scripts/dev/capture_tinygrad_contract.py" },
            { kind := .add, target := "scripts/dev/test_capture_tinygrad_contract.py" },
            { kind := .add, target := "fixtures/upstream/19c4d736f2bc8e26d21f08b28ffd6298408da00f" },
            { kind := .add, target := "Tgrad/Spec/ParityTarget.lean" },
            { kind := .modify, target := "TgradSpec.lean" } ],
        lease :=
          { token := "bootstrap-parity-pin-agent-boyle",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .attemptStarted
      { id := { value := "attempt-harness-namespace-temporaries-20260727" },
        intent := ew "harness.namespace-temporaries", actor := "agent-carver",
        base :=
          { commit := "d82d97992fcae3a9314bb45f5885505c70fa0f0a",
            tree := "95bdfcc22ed893bca8dc88205f61fe002166cc6a",
            dirty := false },
        authorizedEffects :=
          [ { kind := .add, target := "scripts/lib/run_context.sh" },
            { kind := .add, target := "scripts/dev/test_run_context.sh" },
            { kind := .modify, target := "scripts/lib/checks.sh" },
            { kind := .modify, target := "scripts/gate.sh" },
            { kind := .modify, target := "scripts/devcheck.sh" },
            { kind := .modify, target := "scripts/runtime_independence.sh" },
            { kind := .modify, target := "scripts/differential_codegen.sh" },
            { kind := .modify, target := "scripts/dev/l15_a_audit.py" },
            { kind := .modify, target := "scripts/gates" } ],
        lease :=
          { token := "bootstrap-temp-namespace-agent-carver",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .attemptStarted
      { id := { value := "attempt-harness-paired-performance-20260727" },
        intent := ew "harness.paired-performance", actor := "agent-descartes",
        base :=
          { commit := "d82d97992fcae3a9314bb45f5885505c70fa0f0a",
            tree := "95bdfcc22ed893bca8dc88205f61fe002166cc6a",
            dirty := false },
        authorizedEffects :=
          [ { kind := .add, target := "scripts/perf/paired_runtime.py" },
            { kind := .add, target := "scripts/perf/README.md" },
            { kind := .add, target := "scripts/perf/__init__.py" },
            { kind := .add, target := "scripts/dev/test_paired_runtime.py" } ],
        lease :=
          { token := "bootstrap-paired-perf-agent-descartes",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .attemptAbandoned
      { value := "attempt-parity-pin-upstream-20260727" }
      "ownership transferred before authoring; agent-boyle made no repository changes and the Claude worker owns the non-overlapping scripts/parity extractor",
    .attemptStarted
      { id := { value := "attempt-parity-pin-upstream-claude-20260727" },
        intent := ew "parity.pin-upstream", actor := "claude-window-2-tab-1",
        base :=
          { commit := "d82d97992fcae3a9314bb45f5885505c70fa0f0a",
            tree := "95bdfcc22ed893bca8dc88205f61fe002166cc6a",
            dirty := false },
        authorizedEffects :=
          [ { kind := .add, target := "scripts/parity" },
            { kind := .add, target := "fixtures/parity/upstream_19c4d736f2bc.json" },
            { kind := .add, target := "Tgrad/Spec/ParityTarget.lean" },
            { kind := .modify, target := "TgradSpec.lean" } ],
        lease :=
          { token := "coord-window-2-tab-1-parity-pin",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-parity-extractor-3ed1e49" },
        attempt := { value := "attempt-parity-pin-upstream-claude-20260727" },
        tree := "dcbe77daa1a713336f54ee3578fe77a7dc56e6fc",
        observedEffects :=
          [ { kind := .add, target := "scripts/parity" },
            { kind := .add, target := "fixtures/parity/upstream_19c4d736f2bc.json" } ],
        summary := "foreign AST extraction produced the first concrete denominator but deliberately did not promote targetUpstream" },
    .attemptAbandoned
      { value := "attempt-parity-pin-upstream-claude-20260727" }
      "the extraction candidate was retained, but promotion requires an explicit exclusions ledger, generated Lean requirements, product drift pins, and reviewed target selection",
    .attemptStarted
      { id := { value := "attempt-parity-pin-upstream-codex-20260727" },
        intent := ew "parity.pin-upstream", actor := "codex-primary",
        base :=
          { commit := "3ed1e49506260aced45528c371334c293eec34d4",
            tree := "dcbe77daa1a713336f54ee3578fe77a7dc56e6fc",
            dirty := false },
        authorizedEffects :=
          [ { kind := .modify, target := "scripts/parity/extract_upstream.py" },
            { kind := .add, target := "scripts/parity/render_lean_target.py" },
            { kind := .add, target := "scripts/parity/test_target_generation.py" },
            { kind := .modify, target := "fixtures/parity/upstream_19c4d736f2bc.json" },
            { kind := .add, target := "Tgrad/Spec/ParityTarget.lean" },
            { kind := .modify, target := "Tgrad/Spec/Parity.lean" },
            { kind := .modify, target := "Tgrad/Spec/Findings.lean" },
            { kind := .modify, target := "PARITY.md" } ],
        lease :=
          { token := "codex-primary-parity-target-promotion",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-parity-target-8c87034" },
        attempt := { value := "attempt-parity-pin-upstream-codex-20260727" },
        tree := "081c50f1ac029ab496fec91d8b5828d54cb7bdae",
        observedEffects :=
          [ { kind := .modify, target := "scripts/parity/extract_upstream.py" },
            { kind := .add, target := "scripts/parity/render_lean_target.py" },
            { kind := .add, target := "scripts/parity/test_target_generation.py" },
            { kind := .modify, target := "fixtures/parity/upstream_19c4d736f2bc.json" },
            { kind := .add, target := "Tgrad/Spec/ParityTarget.lean" },
            { kind := .modify, target := "Tgrad/Spec/Parity.lean" },
            { kind := .modify, target := "Tgrad/Spec/Findings.lean" },
            { kind := .modify, target := "PARITY.md" } ],
        summary := "reviewed upstream pin, explicit exclusions and section hashes, deterministic Lean generation, 590 derived requirements, product drift pins, and anti-understatement findings" },
    .checkRecorded
      { id := { value := "check-parity-target-build-8c87034" },
        candidate := { value := "candidate-parity-target-8c87034" },
        tree := "081c50f1ac029ab496fec91d8b5828d54cb7bdae",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "detached 8c87034 worktree: lake build tgrad-spec; run tgrad-spec and require target confirmed=true",
        artifactDigest := "sha256:9d8df1edde988072d844a8edad0b5bcde9cb46f35bac892ec7cbe0cdd1978022" },
    .checkRecorded
      { id := { value := "check-parity-target-api-8c87034" },
        candidate := { value := "candidate-parity-target-8c87034" },
        tree := "081c50f1ac029ab496fec91d8b5828d54cb7bdae",
        validator := rw "verify.upstream-contract", obligation := .apiContract,
        outcome := .passed,
        command := "official GitHub API and clean 19c4d736 checkout both --check the 297 methods, 5 properties, 52 dtype names, 82 Ops, 16 backends, and 138 test files",
        artifactDigest := "sha256:0d4d2e018129d7e3aad48d6794e921435826ac37c881e4ac5e10ee0e3449eb05" },
    .checkRecorded
      { id := { value := "check-parity-target-semantic-8c87034" },
        candidate := { value := "candidate-parity-target-8c87034" },
        tree := "081c50f1ac029ab496fec91d8b5828d54cb7bdae",
        validator := rw "verify.upstream-contract", obligation := .semantic,
        outcome := .passed,
        command := "render_lean_target --check; native_decide proves 590 generated requirement rows are unique and well formed; target contract cells remain unknown",
        artifactDigest := "sha256:0c929b7a82ab8d13077f80ee5349c70522fdfb73c0b4bc293662bd293f5f6a91" },
    .checkRecorded
      { id := { value := "check-parity-target-provenance-8c87034" },
        candidate := { value := "candidate-parity-target-8c87034" },
        tree := "081c50f1ac029ab496fec91d8b5828d54cb7bdae",
        validator := rw "verify.upstream-contract", obligation := .provenance,
        outcome := .passed,
        command := "six offline tests reject the wrong Ops location, missing Tensor mixins, stale rehashed Ops inventory, corrupted content hash, and implicit exclusions; local/API captures agree",
        artifactDigest := "sha256:210324965d302087cf78ef6c67ffc0cc930e1b336ea8c2c6dc277921ea5e8c82" },
    .checkRecorded
      { id := { value := "check-parity-target-review-8c87034" },
        candidate := { value := "candidate-parity-target-8c87034" },
        tree := "081c50f1ac029ab496fec91d8b5828d54cb7bdae",
        validator := rw "spec.check-growth-loop", obligation := .humanReview,
        outcome := .passed,
        command := "owner reviewed 3ed1e49, verified the official revision, required product pins/self-reference labels/evidence wording, and authorized replacing unknown with the generated pin",
        artifactDigest := "sha256:57652041e1a5523fc32e4ca99e8e4b847b9f57d840298a3cc89192ee74548bf8" },
    .promoted
      { growthCase := "G-upstream-contract",
        candidate := { value := "candidate-parity-target-8c87034" },
        checkRuns :=
          [ { value := "check-parity-target-build-8c87034" },
            { value := "check-parity-target-api-8c87034" },
            { value := "check-parity-target-semantic-8c87034" },
            { value := "check-parity-target-provenance-8c87034" },
            { value := "check-parity-target-review-8c87034" } ],
        requiredObligations :=
          [.build, .apiContract, .semantic, .provenance, .humanReview],
        acceptedBy := ["harsh", "codex-primary", "claude-window-2-tab-1"],
        target :=
          { commit := "8c8703470b8bf4a51553c5d5db4dbe424ced8af6",
            tree := "081c50f1ac029ab496fec91d8b5828d54cb7bdae",
            dirty := false },
        residualRisks :=
          [ "targetContract is still unknown because no immutable Tgrad subject/profile coverage matrix exists",
            "test-file requirements are coarse inventory rows until the upstream adapter extracts test cases and applicability",
            "the candidate commit predates this promotion record" ] },
    .attemptAbandoned
      { value := "attempt-harness-namespace-temporaries-20260727" }
      "the agent-authored script batch was retained, but integration added Lean trace-path and shared-root safety obligations outside the original lease; ownership moved to an exact-tree integrator attempt",
    .attemptStarted
      { id := { value := "attempt-harness-run-isolation-integration-20260727" },
        intent := ew "harness.namespace-temporaries", actor := "codex-primary",
        base :=
          { commit := "6b946c77ecaf104d1ff5ad146fced211d3f6a864",
            tree := "fdfc7967774492fd484103c1b5c2c9ce404270eb",
            dirty := false },
        authorizedEffects :=
          [ { kind := .add, target := "scripts/lib/run_context.sh" },
            { kind := .add, target := "scripts/dev/test_run_context.sh" },
            { kind := .modify, target := "scripts/lib/checks.sh" },
            { kind := .modify, target := "scripts/gate.sh" },
            { kind := .modify, target := "scripts/devcheck.sh" },
            { kind := .modify, target := "scripts/runtime_independence.sh" },
            { kind := .modify, target := "scripts/differential_codegen.sh" },
            { kind := .modify, target := "scripts/dev/l15_a_audit.py" },
            { kind := .modify, target := "scripts/gates" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" },
            { kind := .modify, target := "Tgrad/Spec/LiveConditions.lean" },
            { kind := .modify, target := "Tgrad/Spec/RuntimeWork.lean" },
            { kind := .modify, target := "Tgrad/Spec/Work.lean" },
            { kind := .modify, target := "PARITY.md" },
            { kind := .modify, target := "GROWING_TGRAD.md" } ],
        lease :=
          { token := "codex-primary-run-isolation-integration",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-run-isolation-602897e" },
        attempt := { value := "attempt-harness-run-isolation-integration-20260727" },
        tree := "acdbe4423703a56009c196760c4d9984c5312743",
        observedEffects :=
          [ { kind := .add, target := "scripts/lib/run_context.sh" },
            { kind := .add, target := "scripts/dev/test_run_context.sh" },
            { kind := .modify, target := "scripts/lib/checks.sh" },
            { kind := .modify, target := "scripts/gate.sh" },
            { kind := .modify, target := "scripts/devcheck.sh" },
            { kind := .modify, target := "scripts/runtime_independence.sh" },
            { kind := .modify, target := "scripts/differential_codegen.sh" },
            { kind := .modify, target := "scripts/dev/l15_a_audit.py" },
            { kind := .modify, target := "scripts/gates" },
            { kind := .modify, target := "Tgrad/Pipeline.lean" },
            { kind := .modify, target := "Tgrad/Spec/LiveConditions.lean" },
            { kind := .modify, target := "Tgrad/Spec/RuntimeWork.lean" },
            { kind := .modify, target := "Tgrad/Spec/Work.lean" },
            { kind := .modify, target := "PARITY.md" },
            { kind := .modify, target := "GROWING_TGRAD.md" } ],
        summary := "owned run roots replace fixed gate/devcheck artifacts; nested consumers inherit identity; Lean tracing requires an explicit path; unsafe shared roots reject" },
    .checkRecorded
      { id := { value := "check-run-isolation-build-602897e" },
        candidate := { value := "candidate-run-isolation-602897e" },
        tree := "acdbe4423703a56009c196760c4d9984c5312743",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "clean detached worktree at 602897e: lake build tgrad-spec; run tgrad-spec and require 27 well-formed runtime capabilities",
        artifactDigest := "sha256:0450815f83b7ac7f9ced453160c5c773805220f30446e442a9eee800d6b84413" },
    .checkRecorded
      { id := { value := "check-run-isolation-semantic-602897e" },
        candidate := { value := "candidate-run-isolation-602897e" },
        tree := "acdbe4423703a56009c196760c4d9984c5312743",
        validator := rw "verify.run-isolation", obligation := .semantic,
        outcome := .passed,
        command := "detached tree: bash scripts/dev/test_run_context.sh; bash scripts/devcheck.sh L0; require nested consumers share only their parent root and preserve expected artifacts",
        artifactDigest := "sha256:cc35f0b61c8a83d1a3424ecff7de98a890561830d6fcc10736a458095da86f73" },
    .checkRecorded
      { id := { value := "check-run-isolation-safety-602897e" },
        candidate := { value := "candidate-run-isolation-602897e" },
        tree := "acdbe4423703a56009c196760c4d9984c5312743",
        validator := rw "verify.run-isolation", obligation := .safety,
        outcome := .passed,
        command := "detached tree: reject /tmp, invalid child names, non-empty unmarked shared roots, incomplete ownership, and missing TGRAD_RANGEIFY_TRACE_PATH; only exact marked owners clean",
        artifactDigest := "sha256:20a97eee1c224bf743f7b7f28616fc127e82fbb7efef288cbc3f222dd6d63672" },
    .checkRecorded
      { id := { value := "check-run-isolation-concurrency-602897e" },
        candidate := { value := "candidate-run-isolation-602897e" },
        tree := "acdbe4423703a56009c196760c4d9984c5312743",
        validator := rw "verify.run-isolation", obligation := .resourceIsolation,
        outcome := .passed,
        command := "detached tree: launch alpha/beta contexts concurrently; require distinct roots, artifacts, nested logs, and rangeify traces; rg finds no fixed /tmp path in gate/devcheck/Lean trace surfaces",
        artifactDigest := "sha256:9c6c4d12bee78d29e864164df3a5bc0b12f981e99ddd2ba6ea53c2b915006fe8" },
    .promoted
      { growthCase := "G-run-isolation",
        candidate := { value := "candidate-run-isolation-602897e" },
        checkRuns :=
          [ { value := "check-run-isolation-build-602897e" },
            { value := "check-run-isolation-semantic-602897e" },
            { value := "check-run-isolation-safety-602897e" },
            { value := "check-run-isolation-concurrency-602897e" } ],
        requiredObligations := [.build, .semantic, .safety, .resourceIsolation],
        acceptedBy := ["harsh", "codex-primary"],
        target :=
          { commit := "602897eef58ac4883aff7091662b851b808d5222",
            tree := "acdbe4423703a56009c196760c4d9984c5312743",
            dirty := false },
        residualRisks :=
          [ "shared .lake build outputs still serialize",
            "Metal correctness and all timing remain serial on one GPU",
            "committed evidence integration remains a single-writer operation",
            "the candidate commit predates this promotion record" ] },
    .attemptAbandoned
      { value := "attempt-harness-paired-performance-20260727" }
      "the agent-authored observer was retained, but integrator review added unique run identity, clean local-subject enforcement, equal-session bootstrap weighting, absolute throughput, structured non-kernel scope, and specification changes outside the original lease",
    .attemptStarted
      { id := { value := "attempt-harness-paired-performance-integration-20260727" },
        intent := ew "harness.paired-performance", actor := "codex-primary",
        base :=
          { commit := "1feb8c98de7c83f7b4df79bb52c746242aad8d23",
            tree := "8c5f4bc38fd2eeec954bb1456564cd744a3f6761",
            dirty := false },
        authorizedEffects :=
          [ { kind := .add, target := "scripts/perf/paired_runtime.py" },
            { kind := .add, target := "scripts/perf/README.md" },
            { kind := .add, target := "scripts/dev/test_paired_runtime.py" },
            { kind := .modify, target := "Tgrad/Spec/RuntimeWork.lean" },
            { kind := .modify, target := "Tgrad/Spec/Work.lean" },
            { kind := .modify, target := "PARITY.md" } ],
        lease :=
          { token := "codex-primary-paired-performance-integration",
            resources := [.sourceTree], validThroughEpoch := 1000 } },
    .candidateProduced
      { id := { value := "candidate-paired-observer-42838f2" },
        attempt := { value := "attempt-harness-paired-performance-integration-20260727" },
        tree := "525af6cdaf1b6cf2fcbd15235aa3a2d4feda56fa",
        observedEffects :=
          [ { kind := .add, target := "scripts/perf/paired_runtime.py" },
            { kind := .add, target := "scripts/perf/README.md" },
            { kind := .add, target := "scripts/dev/test_paired_runtime.py" },
            { kind := .modify, target := "Tgrad/Spec/RuntimeWork.lean" },
            { kind := .modify, target := "Tgrad/Spec/Work.lean" },
            { kind := .modify, target := "PARITY.md" } ],
        summary := "live same-process paired observer with pre-timing byte correctness, clean exact subjects, unique run identity, balanced ordering, raw evidence, equal-session uncertainty, absolute throughput, and no verdict or kernel-speed eligibility" },
    .checkRecorded
      { id := { value := "check-paired-observer-build-42838f2" },
        candidate := { value := "candidate-paired-observer-42838f2" },
        tree := "525af6cdaf1b6cf2fcbd15235aa3a2d4feda56fa",
        validator := rw "verify.lean-build", obligation := .build,
        outcome := .passed,
        command := "clean exact 42838f2 tree: lake build tgrad-spec; run report and require 28 well-formed runtime capabilities with verify.performance still bypassed",
        artifactDigest := "sha256:45a0201683b5fdc6f679e87ef3757d63ef1d37064f136fbb5793765b4aab3448" },
    .checkRecorded
      { id := { value := "check-paired-observer-semantic-42838f2" },
        candidate := { value := "candidate-paired-observer-42838f2" },
        tree := "525af6cdaf1b6cf2fcbd15235aa3a2d4feda56fa",
        validator := rw "verify.paired-performance-observer", obligation := .semantic,
        outcome := .passed,
        command := "run nine CPU-only fake tests: exact raw count, balanced ordering, deterministic fixed identities, correctness-before-output, retained timed errors, unique run identity, and equal session weighting",
        artifactDigest := "sha256:bc2660ba7202c25f851176154f7c6276e76c214565bdb93c99ae2669d04b4fc5" },
    .checkRecorded
      { id := { value := "check-paired-observer-live-42838f2" },
        candidate := { value := "candidate-paired-observer-42838f2" },
        tree := "525af6cdaf1b6cf2fcbd15235aa3a2d4feda56fa",
        validator := rw "verify.paired-performance-observer", obligation := .performance,
        outcome := .passed,
        command := "serial unsandboxed 64x64x64 smoke: one logical session, one warmup pair, two measured pairs; complete with BA/AB order and no errors; no performance conclusion",
        artifactDigest := "sha256:29982c0e1ecaee6de7ad7190d2e56ef5a07dae1abedb97a84f91e2b885400569" },
    .checkRecorded
      { id := { value := "check-paired-observer-provenance-42838f2" },
        candidate := { value := "candidate-paired-observer-42838f2" },
        tree := "525af6cdaf1b6cf2fcbd15235aa3a2d4feda56fa",
        validator := rw "verify.paired-performance-observer", obligation := .provenance,
        outcome := .passed,
        command := "live summary records clean Tgrad 42838f2/525af6c, pinned tinygrad 19c4d736/855cca3, unique run instance/time, exact equal 8192-byte output hashes, toolchain, boundaries, and diagnostic overrides",
        artifactDigest := "sha256:ab02139ea3e04ead3e59003c2b8421f33c3c64675c6de97cf8da3ac9eefde512" },
    .checkRecorded
      { id := { value := "check-paired-observer-isolation-42838f2" },
        candidate := { value := "candidate-paired-observer-42838f2" },
        tree := "525af6cdaf1b6cf2fcbd15235aa3a2d4feda56fa",
        validator := rw "verify.paired-performance-observer", obligation := .resourceIsolation,
        outcome := .passed,
        command := "inspect implementation and raw smoke: one pre-import execve at most, no per-sample subprocess, one long-lived process, synchronized calls, balanced AB/BA, serial GPU lane, and kernel_speed_claim_eligible=false",
        artifactDigest := "sha256:113469f0376ff38dec6900323bba4b2d4e3fb286c4039eeffb4d7e00fbeed19a" },
    .promoted
      { growthCase := "G-symmetric-performance",
        candidate := { value := "candidate-paired-observer-42838f2" },
        checkRuns :=
          [ { value := "check-paired-observer-build-42838f2" },
            { value := "check-paired-observer-semantic-42838f2" },
            { value := "check-paired-observer-live-42838f2" },
            { value := "check-paired-observer-provenance-42838f2" },
            { value := "check-paired-observer-isolation-42838f2" } ],
        requiredObligations :=
          [.build, .semantic, .performance, .provenance, .resourceIsolation],
        acceptedBy := ["harsh", "codex-primary"],
        target :=
          { commit := "42838f220e54a39d10a5925a6c3e1bb23e056706",
            tree := "525af6cdaf1b6cf2fcbd15235aa3a2d4feda56fa",
            dirty := false },
        residualRisks :=
          [ "the available operational boundary includes different host-side work and is not a kernel comparison",
            "one two-pair smoke establishes adapter execution, not repeatability or performance",
            "logical sessions share process-global caches and independent complete runs remain required",
            "verify.performance remains bypassed and F-performance-methodology remains open",
            "the candidate commit predates this promotion record" ] } ]

def liveEvolutionState : Except TransitionError State :=
  replay itemIds liveEvolutionEvents

def liveEvolutionStateValid : Bool :=
  match liveEvolutionState with
  | .ok state =>
      state.activeAttempts.length == 0 && state.candidates.length == 15 &&
      state.checks.length == 60 && state.promotions.length == 11 &&
      state.abandoned.length == 7
  | .error _ => false

def liveActiveIntentIds : List Growth.EvolutionWorkId :=
  match liveEvolutionState with
  | .ok state => state.activeAttempts.map (·.intent)
  | .error _ => []

def livePromotedIntentIds : List Growth.EvolutionWorkId :=
  match liveEvolutionState with
  | .error _ => []
  | .ok state => state.activePromotions.filterMap (fun certificate => do
      let candidate ← state.candidateFor? certificate.candidate
      let attempt ← state.attemptFor? candidate.attempt
      pure attempt.intent)

def completedIds : List Growth.EvolutionWorkId :=
  (workItems.filter (fun item => item.progress.isComplete)).map (·.id)

def dependenciesKnown : Bool :=
  workItems.all (fun item => item.dependsOn.all (fun dep => itemIds.contains dep))

def noSelfDependencies : Bool :=
  workItems.all (fun item => !(item.dependsOn.contains item.id))

def uniqueItemIds : Bool := itemIds.eraseDups.length == itemIds.length

def runtimeScopesKnown : Bool :=
  workItems.all (fun item => item.runtimeScope.all Runtime.workIds.contains)

def dependenciesSatisfied (item : WorkItem) : Bool :=
  item.dependsOn.all completedIds.contains

def dependencyReadyItems : List WorkItem :=
  workItems.filter (fun item => item.progress.isPlanned && dependenciesSatisfied item)

private def intersects [BEq alpha] (xs ys : List alpha) : Bool :=
  xs.any ys.contains

def authoringCompatible (left right : WorkItem) : Bool :=
  !intersects left.writes right.writes

def activeItems : List WorkItem :=
  workItems.filter (fun item =>
    item.progress.isInProgress || liveActiveIntentIds.contains item.id)

def liveEventsAgreeWithProgress : Bool :=
  liveActiveIntentIds.all (fun id =>
    workItems.any (fun item => item.id == id && item.progress.isInProgress)) &&
  livePromotedIntentIds.all (fun id =>
    workItems.any (fun item => item.id == id && item.progress.isComplete))

def clearOfActiveWriters (item : WorkItem) : Bool :=
  activeItems.all (authoringCompatible item)

/-- Dependency-ready work that does not overlap an already-active writer. -/
def readyItems : List WorkItem :=
  dependencyReadyItems.filter clearOfActiveWriters

def verificationCompatible (left right : WorkItem) : Bool :=
  !(left.verificationResources.any fun resource =>
    right.verificationResources.contains resource &&
    exclusiveForVerification resource)

def unblocksCount (id : Growth.EvolutionWorkId) : Nat :=
  (workItems.filter (fun item => item.dependsOn.contains id)).length

/-- Bootstrap picker score: directness first, then downstream work unblocked,
and finally lower estimated cost. Validation obligations are deliberately not
ranked; the picker may not trade correctness for performance or provenance. -/
def priorityScore (item : WorkItem) : Nat :=
  (10 - min item.goalDistance 10) * 100 +
  unblocksCount item.id * 20 +
  (6 - min item.cost 6)

def readyByPriority : List WorkItem :=
  readyItems.mergeSort (fun left right => priorityScore left >= priorityScore right)

def selectNext? : Option WorkItem := readyByPriority.head?

/-- Greedy, priority-ordered authoring frontier. Verification compatibility is
queried separately because its capacity is resource-specific. -/
def parallelAuthoringFrontier : List WorkItem :=
  readyByPriority.foldl (fun selected candidate =>
    if selected.all (authoringCompatible candidate) then
      selected ++ [candidate]
    else selected) []

def openFindingIds : List String := openFindings.map (·.id)

def openFindingsHaveWork : Bool :=
  openFindingIds.all (fun id => workItems.any (fun item => item.closesFindings.contains id))

def growthCasesWellFormed : Bool :=
  Growth.casesWellFormed itemIds completedIds growthCases

def openFindingsHaveGrowthCases : Bool :=
  Growth.openFindingsCovered growthCases

def currentGrowthMetrics : Growth.Metrics := Growth.metrics growthCases

def plannedWorkHasMeaningfulValidation : Bool :=
  workItems.all (fun item =>
    if item.progress.isPlanned || item.progress.isInProgress
    then item.validation.meaningful
    else true)

def frontierIds : List Growth.EvolutionWorkId := parallelAuthoringFrontier.map (·.id)

theorem work_graph_well_formed :
    dependenciesKnown && noSelfDependencies && uniqueItemIds && runtimeScopesKnown = true := by
  native_decide

theorem current_open_findings_are_actionable : openFindingsHaveWork = true := by
  native_decide

theorem planned_validation_is_specific : plannedWorkHasMeaningfulValidation = true := by
  native_decide

theorem historical_history_has_an_upgrade_path :
    historicalEvolutionHistory.hasUpgradePath = true := by
  native_decide

theorem live_evolution_event_replays_and_matches_progress :
    liveEvolutionStateValid && liveEventsAgreeWithProgress = true := by
  native_decide

theorem growth_loop_references_are_well_formed : growthCasesWellFormed = true := by
  native_decide

theorem every_open_finding_enters_a_growth_case : openFindingsHaveGrowthCases = true := by
  native_decide

theorem routing_is_promoted_and_released :
    (livePromotedIntentIds.contains (ew "codegen.route-sentinels") &&
      completedIds.contains (ew "codegen.route-sentinels") &&
      !(liveActiveIntentIds.contains (ew "codegen.route-sentinels"))) = true := by
  native_decide

theorem current_authoring_frontier_is_performance_rebaseline :
    frontierIds = [ew "perf.rebaseline"] := by
  native_decide

theorem materialization_and_differential_harness_are_promoted_and_released :
    (!(liveActiveIntentIds.contains (ew "view.materialize")) &&
      livePromotedIntentIds.contains (ew "view.materialize") &&
      livePromotedIntentIds.contains (ew "codegen.differential-harness")) = true := by
  native_decide

theorem view_materialization_candidate_has_required_checks :
    (match liveEvolutionState with
    | .error _ => false
    | .ok state =>
        let runs := state.checks.filter (fun run =>
          run.candidate.value == "candidate-view-materialize-790d413" &&
          run.outcome.passed?)
        runs.any (fun run => run.obligation == .build) &&
        runs.any (fun run => run.obligation == .apiContract) &&
        runs.any (fun run => run.obligation == .safety) &&
        runs.any (fun run => run.obligation == .numerical) &&
        runs.any (fun run => run.obligation == .differential)) = true := by
  native_decide

theorem generated_routing_candidate_has_required_checks :
    (match liveEvolutionState with
    | .error _ => false
    | .ok state =>
        let runs := state.checks.filter (fun run =>
          run.candidate.value == "candidate-route-sentinels-8475550" &&
          run.outcome.passed?)
        runs.any (fun run => run.obligation == .build) &&
        runs.any (fun run => run.obligation == .apiContract) &&
        runs.any (fun run => run.obligation == .safety) &&
        runs.any (fun run => run.obligation == .numerical) &&
        runs.any (fun run => run.obligation == .differential)) = true := by
  native_decide

theorem transcription_deletion_candidate_has_required_checks :
    (match liveEvolutionState with
    | .error _ => false
    | .ok state =>
        let runs := state.checks.filter (fun run =>
          run.candidate.value == "candidate-delete-transcription-1401305" &&
          run.outcome.passed?)
        runs.any (fun run => run.obligation == .build) &&
        runs.any (fun run => run.obligation == .unitRegression) &&
        runs.any (fun run => run.obligation == .semantic) &&
        runs.any (fun run => run.obligation == .numerical) &&
        runs.any (fun run => run.obligation == .differential)) = true := by
  native_decide

theorem codegen_differential_candidate_has_complementary_checks :
    (match liveEvolutionState with
    | .error _ => false
    | .ok state =>
        let runs := state.checks.filter (fun run =>
          run.candidate.value == "candidate-codegen-differential-a6d5958" &&
          run.outcome.passed?)
        runs.any (fun run => run.obligation == .build) &&
        runs.any (fun run => run.obligation == .semantic) &&
        runs.any (fun run => run.obligation == .differential) &&
        runs.any (fun run => run.obligation == .provenance) &&
        runs.any (fun run => run.obligation == .resourceIsolation)) = true := by
  native_decide

theorem semantic_gate_is_promoted_with_build_and_execution_obligations :
    (livePromotedIntentIds.contains (ew "gates.semantic-codegen") &&
      completedIds.contains (ew "gates.semantic-codegen")) = true := by
  native_decide

theorem evidence_audit_is_promoted_but_fatal_enforcement_is_not_ready :
    (livePromotedIntentIds.contains (ew "evidence.audit-tool") &&
      completedIds.contains (ew "evidence.audit-tool") &&
      !(readyItems.map (·.id)).contains (ew "evidence.enforce-provenance")) = true := by
  native_decide

theorem partial_evidence_regeneration_is_abandoned_not_promoted :
    (match liveEvolutionState with
    | .error _ => false
    | .ok state =>
        state.abandoned.contains
            { value := "attempt-evidence-regeneration-20260726" } &&
        state.checks.any (fun run =>
          run.id.value == "check-L11-repeatability-e90607f" &&
          !run.outcome.passed?) &&
        state.checks.any (fun run =>
          run.id.value == "check-partial-regeneration-provenance-7c7dc0f" &&
          !run.outcome.passed?) &&
        !state.promotions.any (fun certificate =>
          certificate.candidate.value ==
            "candidate-evidence-regeneration-5ccb293")) = true := by
  native_decide

theorem diagnostic_regeneration_did_not_satisfy_promotion_dependencies :
    (match liveEvolutionState with
    | .error _ => false
    | .ok state =>
        state.attempts.any (fun attempt =>
          attempt.id.value == "attempt-evidence-regeneration-20260726")) &&
      !(dependencyReadyItems.map (·.id)).contains
        (ew "evidence.regenerate") = true := by
  native_decide

theorem regeneration_observation_record_is_promoted_and_released :
    (livePromotedIntentIds.contains
        (ew "spec.record-regeneration-observation") &&
      completedIds.contains (ew "spec.record-regeneration-observation") &&
      !(liveActiveIntentIds.contains
        (ew "spec.record-regeneration-observation"))) = true := by
  native_decide

theorem warp_parameterization_is_promoted_and_released :
    (livePromotedIntentIds.contains (ew "codegen.warp-parameter") &&
      !(liveActiveIntentIds.contains (ew "codegen.warp-parameter")) &&
      completedIds.contains (ew "codegen.warp-parameter")) = true := by
  native_decide

theorem transcription_deletion_is_promoted_and_released :
    (livePromotedIntentIds.contains (ew "codegen.delete-transcription") &&
      completedIds.contains (ew "codegen.delete-transcription") &&
      !(liveActiveIntentIds.contains (ew "codegen.delete-transcription"))) = true := by
  native_decide

theorem upstream_contract_is_promoted_and_released :
    (livePromotedIntentIds.contains (ew "parity.pin-upstream") &&
      completedIds.contains (ew "parity.pin-upstream") &&
      !(liveActiveIntentIds.contains (ew "parity.pin-upstream")) &&
      Parity.targetUpstream.isConfirmed &&
      Parity.targetRequirements.length == ParityTarget.requirementCount) = true := by
  native_decide

theorem run_isolation_is_promoted_and_released :
    (livePromotedIntentIds.contains (ew "harness.namespace-temporaries") &&
      completedIds.contains (ew "harness.namespace-temporaries") &&
      !(liveActiveIntentIds.contains (ew "harness.namespace-temporaries")) &&
      !(exclusiveForVerification .tmpNamespace)) = true := by
  native_decide

theorem paired_observer_is_promoted_without_promoting_performance :
    (livePromotedIntentIds.contains (ew "harness.paired-performance") &&
      completedIds.contains (ew "harness.paired-performance") &&
      !(liveActiveIntentIds.contains (ew "harness.paired-performance")) &&
      (Runtime.workUnitFor? (rw "verify.paired-performance-observer")).map
        (fun unit => unit.isState .bounded) = some true &&
      (Runtime.workUnitFor? (rw "verify.performance")).map
        (fun unit => unit.isState .bypassed) = some true) := by
  native_decide

theorem generated_sentinels_case_matches_runtime_and_finding_state :
    ((growthCases.find? (fun growthCase => growthCase.id == "G-generated-sentinels")).map
        (fun growthCase => growthCase.stage == .promoted) = some true &&
      (Runtime.workUnitFor? (rw "product.lower-tc-matmul")).map
        (fun unit => unit.isState .loadBearing) = some true &&
      (findings.find? (fun finding => finding.id == "F-transcribed-sentinel-codegen")).map
        (fun finding => !finding.isOpen) = some true) := by
  native_decide

private def commaSeparated (values : List Growth.EvolutionWorkId) : String :=
  String.intercalate ", " (values.map (·.value))

private def readySummary (item : WorkItem) : String :=
  s!"{item.id} (priority={priorityScore item}, cost={item.cost})"

def printReport : IO Unit := do
  IO.println "Tgrad checked specification"
  IO.println s!"parity target confirmed={Parity.targetUpstream.isConfirmed}; long-horizon templates={Parity.program.length}"
  IO.println s!"parity bootstrap templates: {String.intercalate ", " ((Parity.templatesReadyAfter []).map (fun item => item.id))}"
  IO.println s!"runtime work ({Runtime.workUnits.length} capabilities):"
  IO.println s!"  product={(Runtime.unitsInRealm .product).length}, verification={(Runtime.unitsInRealm .verification).length}, specification={(Runtime.unitsInRealm .specification).length}"
  IO.println s!"  missing-or-bypassed={Runtime.missingOrBypassed.length}"
  for unit in Runtime.missingOrBypassed do
    IO.println s!"    - {unit.id}"
  IO.println s!"tracked open findings ({openFindings.length}):"
  for finding in openFindings do
    IO.println s!"  - {finding.id}: {finding.description}"
  IO.println s!"active work ({activeItems.length}):"
  for item in activeItems do
    IO.println s!"  - {item.id}"
  IO.println s!"safe ready work ({readyByPriority.length}):"
  for item in readyByPriority do
    IO.println s!"  - {readySummary item}"
  IO.println s!"growth loops: {currentGrowthMetrics.growthCases} total, {currentGrowthMetrics.promotedCases} promoted, {currentGrowthMetrics.openFindingsCovered}/{currentGrowthMetrics.openFindingsTotal} open findings covered"
  IO.println s!"durable evolution ledger: {liveEvolutionEvents.length} live event(s); historical events remain unknown"
  IO.println s!"parallel evolution frontier: {commaSeparated frontierIds}"
  IO.println "verification policy: run-scoped CPU artifacts may parallelize; shared Lean builds, Metal work, timing, and evidence integration serialize"

end Evolution
end Tgrad.Spec

def main : IO Unit := Tgrad.Spec.Evolution.printReport

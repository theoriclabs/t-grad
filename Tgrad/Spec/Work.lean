import Tgrad.Spec.Evolution

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
      closesFindings := ["F-transcribed-sentinel-codegen"],
      dependsOn := [ew "codegen.differential-harness", ew "codegen.warp-parameter",
        ew "codegen.typed-stores"],
      runtimeScope := [rw "product.select-matmul-route", rw "product.lower-tc-matmul"],
      touches := [.leanFfi, .renderer],
      writes := ["Tgrad/PythonFFI.lean", "Tgrad/Pipeline.lean", "Main.lean",
        "scripts/devcheck.sh"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu],
      cost := 3, goalDistance := 0,
      validation := plannedValidation
        "all sentinel dispatches use generated kernels with nonzero tcLaunchDims geometry"
        "widen eligibility and dispatch together; update the devcheck eligibility triple; run routing and numerical differentials"
        "11/11 generated and numerically equivalent; (128,128,128)/(96,128,128)/(64,64,64) eligibility changes are explicit and every admitted shape has a nonzero grid"
        [.build, .apiContract, .safety, .numerical, .differential],
      recovery := "keep strict eligibility until geometry and the reviewed eligibility triple pass together; revert only divergent sentinel routes",
      progress := .planned },
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
      closesFindings := [],
      dependsOn := [ew "codegen.route-sentinels", ew "gates.semantic-codegen"],
      runtimeScope := [rw "product.lower-tc-matmul", rw "product.render-metal"],
      touches := [.renderer, .gateHarness],
      writes := ["Tgrad/Renderer/MatmulDecls.lean", "scripts/dev/lower_matmul.py",
        "Tgrad.lean", "Main.lean", "scripts/gates/L12.sh"],
      authoringResources := [.sourceTree],
      verificationResources := [.leanBuildTree, .metalGpu, .tmpNamespace],
      cost := 1, goalDistance := 0,
      validation := plannedValidation
        "no runtime/build reference to transcribed decls and no source-byte-equality gate layer"
        "rg MatmulDecls/lower_matmul; retire Layer C while retaining C3; build; run semantic gate"
        "zero transcription references; Layer C3 plus non-aliasing remain; all 11 generated kernels pass"
        [.build, .semantic, .unitRegression],
      recovery := "restore files from git while preserving generated routing work",
      progress := .planned },
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
    { id := ew "perf.rebaseline", title := "measure symmetric generated-kernel performance",
      phase := .verify, authority := .evidence,
      closesFindings := ["F-performance-methodology"],
      dependsOn := [ew "codegen.delete-transcription"],
      runtimeScope := [rw "verify.performance"],
      touches := [.gateHarness, .evidenceStore],
      writes := ["fixtures/perf/generated_codegen.json", "EXPERIMENT_RESULT.md"],
      authoringResources := [.sourceTree],
      verificationResources := [.metalGpu, .tmpNamespace, .evidenceStore],
      cost := 5, goalDistance := 0,
      validation := plannedValidation
        "live same-session distributions for both runtimes"
        "run synchronized dispatch-only benchmark serially"
        "same boundary, TinyJit policy stated, distributions and throughput sane"
        [.performance, .provenance, .resourceIsolation],
      recovery := "report regression; do not substitute frozen or asymmetric baseline",
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
        "evidence naming HEAD with recomputable hashes"
        "run gates serially; recompute every recorded hash"
        "commit equals HEAD; all source, child, and baseline hashes resolve"
        [.provenance, .semantic, .humanReview],
      recovery := "discard generated snapshot if tree is dirty or any hash is unresolved",
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
  [ { id := "G-renderer-runtime",
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
      stage := .selected,
      promotion := promote ["verify.codegen-differential", "verify.numpy-differential"]
        [.build, .numerical, .semantic, .differential]
        "11/11 generated routes are numerically equivalent before transcription deletion"
        "retain the captured kernel as reference and revert only divergent routing",
      epistemic := .confirmed
        "wide generation covers 64x64x64 while production eligibility deliberately remains strict"
        "75f856b: tcLaunchDims_matches_captured_64, wide_generates_64, strict_rejects_64, and symbolic captured/generated address comparison" },
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
        "L12 retains its green transcription layer and adds 11/11 execution equivalence with source inequality; wrong-but-distinct store/load mutations are caught"
        "retain Layer C until transcription deletion; never remove C3 or the independent non-aliasing theorem",
      epistemic := .confirmed
        "aa67497 makes the a6d5958 differential a load-bearing L12 C3 layer: 11/11 in 5.1s; both c->c+2 and 24K+1->24K+2 diverge while build remains green"
        "executable gate layer and reproduced falsifiability rows" },
    { id := "G-symmetric-performance",
      findingIds := ["F-performance-methodology"],
      observations := [observe "verify.performance" "generated matmul comparison"
        "both runtimes execute the same timed work boundary in one session"
        "measure dispatch-only and end-to-end distributions serially"
        "raw samples, boundary metadata, throughput, and thermal context"],
      evolutionWork := [ew "perf.rebaseline"],
      deltas := [delta "verify.performance" .replaceBypass
        [.performance, .observability, .provenance] .bypassed .loadBearing
        "replace asymmetric frozen-baseline ratios with a symmetric experiment"],
      stage := .selected,
      promotion := promote ["verify.performance", "verify.evidence-integrity"]
        [.performance, .provenance, .resourceIsolation]
        "methods are symmetric and reported throughput is physically plausible"
        "publish unknown/regression rather than reusing stale ratios",
      epistemic := .confirmed "current performance claim is not promotable"
        "measurement-boundary and physical-plausibility audit" },
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
      evolutionWork := [ew "evidence.audit-tool", ew "evidence.regenerate",
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
        "bdc01b0 reports 37/37 absent commit, 73/115 unresolved hashes, 28 roll-up disagreements, and 27 writer mismatches"
        "repeatable auditor calibrated to pass synthetic HEAD-tied evidence" } ]

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
            "the overloaded tgrad_matmul_view(handle,0) ABI should become a named materialize entry" ] } ]

def liveEvolutionState : Except TransitionError State :=
  replay itemIds liveEvolutionEvents

def liveEvolutionStateValid : Bool :=
  match liveEvolutionState with
  | .ok state =>
      state.activeAttempts.length == 0 && state.candidates.length == 7 &&
      state.checks.length == 30 && state.promotions.length == 5 &&
      state.abandoned.length == 2
  | .error _ => false

def liveActiveIntentIds : List Growth.EvolutionWorkId :=
  match liveEvolutionState with
  | .ok state => state.activeAttempts.map (·.intent)
  | .error _ => []

def livePromotedIntentIds : List Growth.EvolutionWorkId :=
  match liveEvolutionState with
  | .error _ => []
  | .ok state => state.promotions.filterMap (fun certificate => do
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

/-- Greedy, priority-ordered authoring frontier. Verification is intentionally
not parallelized; see `verificationCompatible` and resource policies. -/
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

theorem routing_is_dependency_ready_and_released_for_authoring :
    ((dependencyReadyItems.map (·.id)).contains (ew "codegen.route-sentinels") &&
      (readyItems.map (·.id)).contains (ew "codegen.route-sentinels")) = true := by
  native_decide

theorem current_safe_frontier_is_generated_sentinel_routing :
    frontierIds = [ew "codegen.route-sentinels"] := by
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

theorem warp_parameterization_is_promoted_and_released :
    (livePromotedIntentIds.contains (ew "codegen.warp-parameter") &&
      !(liveActiveIntentIds.contains (ew "codegen.warp-parameter")) &&
      completedIds.contains (ew "codegen.warp-parameter")) = true := by
  native_decide

theorem transcription_deletion_is_not_ready :
    (readyItems.map (·.id)).contains (ew "codegen.delete-transcription") = false := by
  native_decide

private def commaSeparated (values : List Growth.EvolutionWorkId) : String :=
  String.intercalate ", " (values.map (·.value))

private def readySummary (item : WorkItem) : String :=
  s!"{item.id} (priority={priorityScore item}, cost={item.cost})"

def printReport : IO Unit := do
  IO.println "Tgrad checked specification"
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
  IO.println "verification policy: serial (shared Lean build, fixed /tmp namespace, one Metal GPU)"

end Evolution
end Tgrad.Spec

def main : IO Unit := Tgrad.Spec.Evolution.printReport

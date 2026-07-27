import Tgrad.Spec.Architecture

/-! # Tgrad.Spec.Findings — mutable claims about the product

Unlike ontology constructors, findings can move from open to remediated without
changing the vocabulary of the domain. A remediated finding remains in the
history with the evidence that closed it.

This registry is scoped to the current view/codegen/evidence program. It is not
yet an exhaustive normalization of every finding from the deep review; that
coverage claim is represented explicitly below instead of being implied by the
list's existence.
-/

namespace Tgrad.Spec

inductive Severity where
  | critical | high | medium | low
  deriving DecidableEq, BEq, Repr, Inhabited

inductive FindingState where
  | open
  | remediated (commit : String)
  | acceptedRisk
  deriving DecidableEq, BEq, Repr, Inhabited

structure Finding where
  id : String
  severity : Severity
  component : Component
  description : String
  state : Epistemic FindingState
  deriving Repr, Inhabited

def findingCoverage : Epistemic String :=
  .tentative "current user-directed view, codegen, performance, and evidence findings"
    "normalized from the findings that determine the active work graph"
    "ingest the remaining FFI, lifecycle, gate-integrity, and API-safety audit findings"

def findings : List Finding :=
  [ { id := "F-runtime-file-replay", severity := .high,
      component := .leanFfi,
      description := "benchmarked sentinel path read captured tinygrad MSL",
      state := .confirmed (.remediated "f679bf7")
        "fixtures/codegen hidden; rendered runtime still byte-matched" },
    { id := "F-rangeify-identity", severity := .high,
      component := .scheduler,
      description := "Schedule.Rangeify.rangeify was definitionally identity",
      state := .confirmed (.remediated "f679bf7")
        "rangeify now eliminates movement nodes; assertion is in Tests.lean" },
    { id := "F-slice-drops-axes", severity := .critical,
      component := .viewAlgebra,
      description := "view indexer ignored every slice axis after the first",
      state := .confirmed (.remediated "f679bf7")
        "a[0:8,4:12] differential test now passes; old result differed by 10.8" },
    { id := "F-expand-assumes-axis1", severity := .critical,
      component := .viewAlgebra,
      description := "expand indexer assumed the broadcast singleton was axis 1",
      state := .confirmed (.remediated "f679bf7")
        "axis-0 and axis-1 expand differential tests pass" },
    { id := "F-buffer-shape-mismatch", severity := .critical,
      component := .pythonAuthoring,
      description := "from_bf16_bytes accepted shapes larger than allocations",
      state := .confirmed (.remediated "e1d5760")
        "mismatched byte length now raises TgradTypeError" },
    { id := "F-view-parent-lifetime", severity := .critical,
      component := .pythonAuthoring,
      description := "views did not retain the tensor owning their MTLBuffer",
      state := .confirmed (.remediated "e1d5760")
        "view of temporary survives GC and 80-allocation LRU churn" },
    { id := "F-view-readback-wrong", severity := .high,
      component := .pythonAuthoring,
      description := "numpy/to_bytes returned pre-view storage as if transformed",
      state := .confirmed (.remediated "e1d5760")
        "view readback now fails explicitly; silent wrong answer removed" },
    { id := "F-view-materialization-missing", severity := .medium,
      component := .renderer,
      description := "views could not materialize into contiguous host-observable results",
      state := .confirmed (.remediated "e6241bd")
        "exact tree 790d413 was isolated-build/GPU checked on bdc01b0 and committed unchanged" },
    { id := "F-transcribed-sentinel-codegen", severity := .high,
      component := .renderer,
      description := "per-shape Lean declarations transcribed captured tinygrad arithmetic strings",
      state := .confirmed (.remediated "9f2ab91")
        "tree 1401305 deletes the declarations/parser, removes all product imports, and retains 11/11 source-different executable differentials" },
    { id := "F-byte-equality-gate", severity := .high,
      component := .gateHarness,
      description := "L12 relied on transcription round-trip without semantic generation evidence",
      state := .confirmed (.remediated "aa67497")
        "aa67497 added C3 before deletion; 9f2ab91 retired byte equality and made the 11/11 semantic differential authoritative" },
    { id := "F-performance-methodology", severity := .critical,
      component := .evidenceStore,
      description := "published ratios compare asymmetric measurement boundaries",
      state := .confirmed .open
        "tinygrad rebuilds graph/schedule without TinyJit; Tgrad dispatches cached kernel" },
    { id := "F-evidence-provenance", severity := .critical,
      component := .evidenceStore,
      description := "committed evidence names an absent commit and stale hashes",
      state := .confirmed .open
        "repeatable audit bdc01b0: 37/37 absent commit, 73/115 unresolved hashes, 28 roll-up disagreements, 27 writer-key mismatches" },
    { id := "F-division-semantics", severity := .medium,
      component := .rewriteEngine,
      description := "constant fold, Python oracle, and C renderer disagree for negatives",
      state := .deferred "negative division semantics remain unaligned"
        "return when general symbolic integer semantics enter the runtime path" },
    { id := "F-reduce-beq", severity := .medium,
      component := .tensorIr,
      description := "UOp.beq does not distinguish reduce operator payloads",
      state := .deferred "REDUCE cannot currently be constructed by UOp.ofParsed"
        "return before reduction becomes a supported runtime operation" },
    { id := "F-legacy-model-scaffolding", severity := .low,
      component := .specification,
      description := "unimported Tgrad/Model modules remain as historical self-referential scaffolding",
      state := .deferred "the legacy model is outside both build roots but still occupies the tree"
        "return after the current view/codegen migration, then delete it in an isolated change" } ]

def Finding.isOpen (finding : Finding) : Bool :=
  match finding.state.value? with
  | some .open => true
  | _ => false

def openFindings : List Finding := findings.filter Finding.isOpen

def remediatedFindings : List Finding := findings.filter (fun finding =>
  match finding.state.value? with
  | some (.remediated _) => true
  | _ => false)

example : openFindings.length = 2 := by native_decide
example : remediatedFindings.length = 10 := by native_decide
example : findings.all (fun finding => finding.state.hasUpgradePath) = true := by
  native_decide
example : findingCoverage.hasUpgradePath = true := by native_decide

end Tgrad.Spec

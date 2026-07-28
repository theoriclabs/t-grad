import Tgrad.Spec.Epistemic

/-! # Tgrad.Spec.LiveConditions — observable constraints on work

These are not timeless invariants. They are conditions of this repository and
machine that must be re-probed before relevant work. The observation method is
part of the claim.
-/

namespace Tgrad.Spec

inductive Resource where
  | sourceTree
  | leanBuildTree
  | tmpNamespace
  | metalGpu
  | evidenceStore
  deriving DecidableEq, BEq, Repr, Inhabited

structure ResourcePolicy where
  resource : Resource
  authoringCapacity : Nat
  verificationCapacity : Nat
  observation : String
  basis : Epistemic String
  deriving Repr, Inhabited

def resourcePolicies : List ResourcePolicy :=
  [ { resource := .sourceTree, authoringCapacity := 3,
      verificationCapacity := 1,
      observation := "git status --short; inspect overlapping write sets",
      basis := .tentative "three disjoint workstreams are practical"
        "file-level conflict matrix and ~239 MB build tree per worktree"
        "recompute before spawning worktrees; include current disk capacity" },
    { resource := .leanBuildTree, authoringCapacity := 1,
      verificationCapacity := 1,
      observation := "ps for lake/lean; inspect .lake/build",
      basis := .confirmed "one writer/verifier at a time"
        "shared incremental artifacts and 239 MB build tree" },
    { resource := .tmpNamespace, authoringCapacity := 1,
      verificationCapacity := 1,
      observation := "rg fixed /tmp/tgrad_ paths in scripts",
      basis := .confirmed "gate verification must be serial"
        "141 fixed /tmp/tgrad_* paths; only seven scripts use mktemp" },
    { resource := .metalGpu, authoringCapacity := 1,
      verificationCapacity := 1,
      observation := "ensure no other Tgrad benchmark is running",
      basis := .confirmed "timing and Metal verification are exclusive"
        "single physical GPU; process-global queue/cache in each runtime" },
    { resource := .evidenceStore, authoringCapacity := 1,
      verificationCapacity := 1,
      observation := "git ls-files fixtures/gate_evidence; confirm directory is gitignored",
      basis := .confirmed "gate evidence is a runtime artifact, never committed"
        "check_gate_evidence_not_tracked; umbrellas require children to have run in-tree" } ]

def policyFor? (resource : Resource) : Option ResourcePolicy :=
  resourcePolicies.find? (fun policy => policy.resource == resource)

def exclusiveForVerification (resource : Resource) : Bool :=
  match policyFor? resource with
  | some policy => policy.verificationCapacity <= 1
  | none => true

example : exclusiveForVerification .metalGpu = true := by native_decide
example : exclusiveForVerification .tmpNamespace = true := by native_decide

end Tgrad.Spec

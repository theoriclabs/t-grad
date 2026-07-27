import Tgrad.Growth.BroadcastAddPacketV4

/-! # V5 cross-run verifier-equivalence relation

Per-run Git provenance remains recorded. Cross-run executable equivalence uses
the observer, probe, schema and frozen verifier-file content identities.
-/

namespace Tgrad.Growth.BroadcastAddPacketV5

def comparisonFields : List String :=
  ["observer_sha256", "probe_sha256", "schema_version", "git.clean", "git.files"]

def retainedProvenanceFields : List String :=
  ["git.revision", "git.tree", "git.sha256"]

def productWriteSet : List String := []

theorem comparison_excludes_run_specific_git_identity :
    !comparisonFields.contains "git.revision" &&
    !comparisonFields.contains "git.tree" &&
    !comparisonFields.contains "git.sha256" := by native_decide

theorem provenance_is_not_deleted : retainedProvenanceFields.length = 3 := by decide

theorem product_authoring_remains_forbidden : productWriteSet = [] := by rfl

end Tgrad.Growth.BroadcastAddPacketV5

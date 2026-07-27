import Tgrad.Growth.BroadcastAddPacketV5

/-! # V6 subject-protocol / orchestration-provenance identity split -/

namespace Tgrad.Growth.BroadcastAddPacketV6

def subjectProtocolFields : List String :=
  ["probe_sha256", "schema_version", "manifest_effective_sha256",
   "semantic_lock_sha256"]

def provenanceOnlyFields : List String :=
  ["observer_sha256", "git.revision", "git.tree", "git.files", "git.sha256"]

def relationFaults : List String :=
  ["probe mismatch", "schema mismatch", "manifest mismatch",
   "semantic lock mismatch", "missing relation identity",
   "revision-only difference accepted"]

def productWriteSet : List String := []

theorem identities_are_disjoint :
    subjectProtocolFields.all (fun field => !provenanceOnlyFields.contains field) := by
  native_decide

theorem relation_has_six_calibration_obligations : relationFaults.length = 6 := by decide

theorem product_authoring_remains_forbidden : productWriteSet = [] := by rfl

end Tgrad.Growth.BroadcastAddPacketV6

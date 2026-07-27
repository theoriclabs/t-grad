import Tgrad.Growth.BroadcastAddManifestV3Generated

/-! # V3 broadcast-add calibration contract

V3 leaves the V2 behavioral manifest untouched. It narrows two verifier trace
footprints after a pinned-upstream diagnostic showed that V2 required completed,
identical stages to differ. The diagnostic remains a blocker, never a baseline.
-/

namespace Tgrad.Growth.BroadcastAddManifestV3

open Tgrad.Growth.BroadcastAddManifestV3Generated

def manifest := Tgrad.Growth.BroadcastAddManifestV2.manifest

def correctedTraceFootprint (mutationId : String) (inherited : List String) : List String :=
  match traceAmendments.find? (fun amendment => amendment.mutationId == mutationId) with
  | some amendment => amendment.after
  | none => inherited

theorem behavioral_manifest_is_inherited :
    manifest = Tgrad.Growth.BroadcastAddManifestV2.manifest := by rfl

theorem only_two_trace_contracts_are_amended :
    traceAmendments.map (·.mutationId) =
      ["MUT-ADD-WRONG-RIGHT-ALIGNMENT", "MUT-ADD-ACCEPT-INCOMPATIBLE"] := by
  native_decide

theorem amendments_strictly_remove_unobserved_trace_differences :
    traceAmendments.all (fun amendment =>
      amendment.after.all amendment.before.contains &&
      amendment.after.length < amendment.before.length) := by
  native_decide

end Tgrad.Growth.BroadcastAddManifestV3

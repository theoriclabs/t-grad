import Tgrad.Growth.BroadcastAddManifestV2Generated

/-! # Effective V2 broadcast-add manifest

V1 remains immutable and definition-refuted. V2 applies exactly the two
observability corrections recorded by the generated amendment; every other
scenario, requirement, trace and mutation field is inherited.
-/

namespace Tgrad.Growth.BroadcastAddManifestV2

open Tgrad.Requirements
open Tgrad.Growth.BroadcastAddManifest
open Tgrad.Growth.BroadcastAddManifestGenerated
open Tgrad.Growth.BroadcastAddManifestV2Generated

def isAmendedMutation (id : String) : Bool :=
  amendedMutationIds.contains id

def amendMutation (mutation : MutationObligation) : MutationObligation :=
  if isAmendedMutation mutation.id then
    { mutation with
      mustNotChange := mutation.mustNotChange.erase .dtype
      mayBeUnobserved := mutation.mayBeUnobserved ++ [.dtype] }
  else mutation

def manifest : FrozenManifest :=
  { BroadcastAddManifestGenerated.manifest with
    packetId := packetId
    baselineRevision := baselineRevision
    path := amendmentPath
    contentHash := effectiveManifestHash
    mutations := BroadcastAddManifestGenerated.manifest.mutations.map amendMutation }

theorem manifest_is_well_formed : manifest.wellFormed := by native_decide

theorem only_declared_mutations_change :
    (BroadcastAddManifestGenerated.manifest.mutations.zip manifest.mutations).all
      (fun pair =>
        if isAmendedMutation pair.1.id then pair.2 = amendMutation pair.1
        else pair.2 = pair.1) := by
  native_decide

theorem amended_mutants_allow_dtype_to_be_unobserved :
    manifest.mutations.filter (fun mutation => isAmendedMutation mutation.id) |>.all
      (fun mutation =>
        !mutation.mustNotChange.contains .dtype &&
        mutation.mayBeUnobserved.contains .dtype) := by
  native_decide

theorem all_other_manifest_axes_are_inherited :
    manifest.requirementIds = BroadcastAddManifestGenerated.manifest.requirementIds ∧
    manifest.scenarios = BroadcastAddManifestGenerated.manifest.scenarios ∧
    manifest.legalTrace = BroadcastAddManifestGenerated.manifest.legalTrace ∧
    manifest.incompatibleTrace = BroadcastAddManifestGenerated.manifest.incompatibleTrace ∧
    manifest.allowedAfter = BroadcastAddManifestGenerated.manifest.allowedAfter ∧
    manifest.forbiddenInferences =
      BroadcastAddManifestGenerated.manifest.forbiddenInferences := by
  native_decide

end Tgrad.Growth.BroadcastAddManifestV2

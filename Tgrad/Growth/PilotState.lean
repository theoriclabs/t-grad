import Tgrad.Growth.Derived
import Tgrad.Evidence.PilotGenerated

/-! # Tgrad.Growth.PilotState — the current derived requirements snapshot

The helper-surface observation is generated from a revision-bound execution.
It now passes with a calibrated no-fallback validator, while the two dependent
tensor requirements become observable but remain unobserved.  The target is
still only an extracted candidate and adequacy remains open, so no compatibility
claim is promoted merely because one behavior passed.
-/

namespace Tgrad.Growth.PilotState

open Tgrad.Requirements
open Tgrad.Requirements.Pilot
open Tgrad.Specification
open Tgrad.Specification.Pilot
open Tgrad.Conformance
open Tgrad.Evidence
open Tgrad.Growth
open Tgrad.Spec

def target : TargetRef :=
  { repository := "tinygrad/tinygrad"
    revision := "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
    manifestHash := "439370c2308bae30756329b6dfd96c0e004ce7a4aa78f155b4ee689554973874"
    profile := publicMetalPilot.id
    disposition := .extractedCandidate }

def productBaseline : TreeRef :=
  Tgrad.Evidence.PilotGenerated.subjectTree

/-- The generated observer binds verifier, adapter, environment, and scenario
identities to the current observation. -/
def context : PromotionContext :=
  { target
    subjectTree := productBaseline
    boundary := some Tgrad.Evidence.PilotGenerated.boundary }

def helpersState : RequirementState :=
  deriveRequirementState context importHelpers helpersBoundary helpersAdequacy
    pilotCandidates Tgrad.Evidence.PilotGenerated.validators
    Tgrad.Evidence.PilotGenerated.observations
    Tgrad.Evidence.PilotGenerated.blockages

def addState : RequirementState :=
  deriveRequirementState context broadcastAdd addBoundary addAdequacy
    pilotCandidates Tgrad.Evidence.PilotGenerated.validators
    Tgrad.Evidence.PilotGenerated.observations
    Tgrad.Evidence.PilotGenerated.blockages

def viewState : RequirementState :=
  deriveRequirementState context viewReadbackLifetime viewBoundary viewAdequacy
    pilotCandidates Tgrad.Evidence.PilotGenerated.validators
    Tgrad.Evidence.PilotGenerated.observations
    Tgrad.Evidence.PilotGenerated.blockages

def states : List RequirementState := [helpersState, addState, viewState]

def gaps : List Gap := (states.flatMap gapsFor).eraseDups

/- This theorem checks only the current snapshot.  It does not establish a
source-to-binary build chain or a derivation-stable before/after transition. -/
theorem current_snapshot_records_calibrated_scenario_without_overpromotion :
    helpersState.implementation = .behaviorObserved ∧
    addState.implementation = .candidateMapped ∧
    viewState.implementation = .candidateMapped ∧
    helpersState.observation = .passedCalibrated ∧
    addState.observation = .unobserved ∧
    viewState.observation = .unobserved ∧
    states.all (fun state =>
      state.adequacy == .open &&
      state.promotion == .targetUnpromoted) = true := by
  native_decide

theorem global_target_gap_is_deduplicated :
    (gaps.filter (fun gap => gap.id == "GAP-TARGET-UPSTREAM")).length = 1 := by
  native_decide

/-! ## Calibration of the derivation logic

These are model mutations, not product-fault evidence.  They establish that
the state derivation distinguishes stale, blocked, failed, uncalibrated, and
qualified observations.  Product validators still need their own real
mutants before capability promotion.
-/

private def verifierBoundary : BoundaryIdentity :=
  { verifierTree :=
      { revision := "requirements-pilot-validator-v1"
        contentHash := "validator-content-hash-v1"
        dirty := false }
    adapterHash := "strict-shim-content-hash-v1"
    runtimeArtifactHash := "runtime-artifact-hash-v1"
    environmentId := "pilot-environment-v1"
    environmentHash := "pilot-environment-hash-v1"
    scenarioManifestHash := "pilot-scenario-manifest-v1" }

private def promotedTarget : TargetRef :=
  { target with disposition := .promoted }

private def observedContext : PromotionContext :=
  { target := promotedTarget
    subjectTree := productBaseline
    boundary := some verifierBoundary }

private def calibratedAddValidator : ValidatorRef :=
  { id := "VALIDATOR-ADD-V1"
    version := "1"
    dimensions := broadcastAdd.relation.dimensions
    calibrations :=
      [{ faultModel := "broadcast index uses a non-zero stride on an expanded axis"
         dimensions := broadcastAdd.relation.dimensions
         mutantTree := "mutant-add-broadcast-stride"
         artifactHash := "artifact-rejected-broadcast-stride"
         outcome := .validatorRejectedMutant }] }

private def uncalibratedAddValidator : ValidatorRef :=
  { calibratedAddValidator with
    calibrations :=
      [{ faultModel := "broadcast index uses a non-zero stride on an expanded axis"
         dimensions := broadcastAdd.relation.dimensions
         mutantTree := "mutant-add-broadcast-stride"
         artifactHash := "artifact-survived-broadcast-stride"
         outcome := .mutantSurvived }] }

private def partiallyCalibratedAddValidator : ValidatorRef :=
  { calibratedAddValidator with
    calibrations :=
      [{ faultModel := "value-only mutation"
         dimensions := [.value]
         mutantTree := "mutant-add-value-only"
         artifactHash := "artifact-rejected-value-only"
         outcome := .validatorRejectedMutant }] }

private def addObservation (outcome : ObservationOutcome) : Observation :=
  { id := "OBS-ADD-PILOT"
    requirement := broadcastAdd.id
    specification := addBoundary.id
    targetRevision := promotedTarget.revision
    subjectTree := productBaseline
    boundary := verifierBoundary
    validatorId := calibratedAddValidator.id
    dimensions := broadcastAdd.relation.dimensions
    outcome
    blocker := if outcome == .blocked then "python substitution prerequisite" else ""
    artifactHash := "artifact-add-pilot"
    runId := "deterministic-model-mutation-add-v1" }

private def acceptedAddAdequacy : AdequacyClaim :=
  { addAdequacy with
    result := .confirmed .argued "checked pilot scenario argument D ∧ S ⊨ R" }

private def stateWith
    (observation : Observation) (validators : List ValidatorRef) : RequirementState :=
  deriveRequirementState observedContext broadcastAdd addBoundary acceptedAddAdequacy
    pilotCandidates validators [observation] []

theorem failed_observation_is_not_hidden_by_a_candidate_mapping :
    (stateWith (addObservation .failed) [calibratedAddValidator]).observation = .failed := by
  native_decide

theorem blocked_observation_is_not_reported_as_behavior_failure :
    (stateWith (addObservation .blocked) [calibratedAddValidator]).observation = .blocked := by
  native_decide

theorem surviving_mutant_prevents_calibrated_pass :
    (stateWith (addObservation .passed) [uncalibratedAddValidator]).observation =
      .passedUncalibrated := by
  native_decide

theorem missing_dimension_calibration_prevents_calibrated_pass :
    (stateWith (addObservation .passed) [partiallyCalibratedAddValidator]).observation =
      .passedUncalibrated := by
  native_decide

theorem observation_from_a_different_specification_does_not_qualify :
    let otherSpec := { addBoundary with id := ⟨"SPEC-OTHER-ADD"⟩ }
    (deriveRequirementState observedContext broadcastAdd otherSpec acceptedAddAdequacy
      pilotCandidates [calibratedAddValidator] [addObservation .passed] []).observation =
      .unobserved := by
  native_decide

theorem adequacy_for_a_different_specification_cannot_promote :
    let wrongClaim :=
      { acceptedAddAdequacy with specification := ⟨"SPEC-OTHER-ADD"⟩ }
    let state :=
      deriveRequirementState observedContext broadcastAdd addBoundary wrongClaim
        pilotCandidates [calibratedAddValidator] [addObservation .passed] []
    state.adequacy = .open ∧ state.promotion = .adequacyOpen := by
  native_decide

theorem calibrated_current_pass_can_promote_only_after_target_and_adequacy :
    let state := stateWith (addObservation .passed) [calibratedAddValidator]
    state.observation = .passedCalibrated ∧ state.promotion = .conformant := by
  native_decide

theorem stale_subject_changes_observation_axis_without_fabricating_failure :
    let stale := { addObservation .passed with
      subjectTree := { productBaseline with revision := "different-product-tree" } }
    (stateWith stale [calibratedAddValidator]).observation = .stale := by
  native_decide

private def inventoryToken : InventoryState → String
  | .catalogued => "catalogued"
  | .interpreted => "interpreted"

private def specificationToken : SpecificationState → String
  | .missing => "missing"
  | .structurallySpecified => "structurally_specified"

private def adequacyToken : AdequacyState → String
  | .open => "open"
  | .tentative => "tentative"
  | .accepted => "accepted"
  | .refuted => "refuted"

private def implementationToken : ImplementationState → String
  | .noCandidate => "no_candidate"
  | .candidateMapped => "candidate_mapped"
  | .behaviorObserved => "behavior_observed"

private def observationToken : ObservationState → String
  | .unobserved => "unobserved"
  | .stale => "stale"
  | .blocked => "blocked"
  | .failed => "failed"
  | .verifierError => "verifier_error"
  | .passedUncalibrated => "passed_uncalibrated"
  | .passedCalibrated => "passed_calibrated"

private def promotionToken : PromotionState → String
  | .targetUnpromoted => "target_unpromoted"
  | .adequacyOpen => "adequacy_open"
  | .observationInsufficient => "observation_insufficient"
  | .conformant => "conformant"

private def gapKindToken : GapKind → String
  | .targetPromotion => "target_promotion"
  | .specification => "specification"
  | .adequacy => "adequacy"
  | .implementation => "implementation"
  | .observation => "observation"
  | .validator => "validator"
  | .failedBehavior => "failed_behavior"
  | .prerequisite => "prerequisite"
  | .environment => "environment"

private def quote (value : String) : String :=
  "\"" ++ (value.replace "\\" "\\\\").replace "\"" "\\\"" ++ "\""

private def stateJson (state : RequirementState) : String :=
  "    {\n" ++
  s!"      \"requirement\": {quote state.requirement.value},\n" ++
  s!"      \"inventory\": {quote (inventoryToken state.inventory)},\n" ++
  s!"      \"specification\": {quote (specificationToken state.specification)},\n" ++
  s!"      \"adequacy\": {quote (adequacyToken state.adequacy)},\n" ++
  s!"      \"implementation\": {quote (implementationToken state.implementation)},\n" ++
  s!"      \"observation\": {quote (observationToken state.observation)},\n" ++
  s!"      \"promotion\": {quote (promotionToken state.promotion)}\n" ++
  "    }"

private def gapJson (gap : Gap) : String :=
  let requirement := match gap.requirement with
    | none => "null"
    | some id => quote id.value
  "    {\n" ++
  s!"      \"id\": {quote gap.id},\n" ++
  s!"      \"requirement\": {requirement},\n" ++
  s!"      \"kind\": {quote (gapKindToken gap.kind)}\n" ++
  "    }"

def statusJson : String :=
  "{\n" ++
  "  \"schema_version\": 1,\n" ++
  s!"  \"target_revision\": {quote target.revision},\n" ++
  "  \"target_disposition\": \"extracted_candidate\",\n" ++
  s!"  \"product_revision\": {quote productBaseline.revision},\n" ++
  "  \"observer_identity\": {\n" ++
  s!"    \"verifier_hash\": {quote Tgrad.Evidence.PilotGenerated.boundary.verifierTree.contentHash},\n" ++
  s!"    \"adapter_hash\": {quote Tgrad.Evidence.PilotGenerated.boundary.adapterHash},\n" ++
  s!"    \"runtime_artifact_hash\": {quote Tgrad.Evidence.PilotGenerated.boundary.runtimeArtifactHash},\n" ++
  s!"    \"environment_hash\": {quote Tgrad.Evidence.PilotGenerated.boundary.environmentHash},\n" ++
  s!"    \"scenario_hash\": {quote Tgrad.Evidence.PilotGenerated.boundary.scenarioManifestHash}\n" ++
  "  },\n" ++
  "  \"states\": [\n" ++
  String.intercalate ",\n" (states.map stateJson) ++ "\n  ],\n" ++
  "  \"gaps\": [\n" ++
  String.intercalate ",\n" (gaps.map gapJson) ++ "\n  ]\n" ++
  "}"

end Tgrad.Growth.PilotState

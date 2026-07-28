import Tgrad.Contract.Identity
import Tgrad.Contract.Assurance
import Tgrad.Contract.Completion
import Tgrad.Contract.Chronology

/-! # Tgrad.Contract.GeneratedClaim — canonical claim guard

Packet `mechanics.generated-claim-v1` / `mechanics.generated-claim-guard-v1`.

Public inputs are premises only. A private smart constructor validates a raw
profile-completion request, judging-input closure candidate, and cycle-registry
candidate, selects one promoted event, then cross-binds target, profile, source
closure, subject tree ↔ promoted candidate commit, runtime, assurance policy,
judging-closure identity, and claim-renderer identity. Callers cannot author a
complete/pass Boolean or a certificate digest.

Human-facing completion language is emitted only by the renderer from a
validated private certificate. Validation faults produce an attempt/failure
report that may list blockers but never emits completion language.

This packet is epistemically honest:

* `ContentDigest` remains a shape token, not a cryptographic binding;
* the synthetic one-requirement fixture is unauthenticated;
* no real Tgrad completeness / tinygrad-compatibility claim exists;
* agreement between Lean native_decide checks and the Python regenerator is
  not authentication of either artifact;
* field-provenance classification is an authored premise and is not
  cryptographically grounded to field denotations.

Activation is synthetic / non-release. Full cross-language generation from Lean
is not yet available; the Python checker mirrors the canonical encoding over
the same synthetic raw inputs and states that trust boundary explicitly.
-/

namespace Tgrad.Contract

/-! ## Schema / activation -/

def generatedClaimSchemaVersion : Nat := 1

/-- Activation class. Release gates must not treat synthetic claims as product
truth. -/
inductive ClaimActivation where
  | syntheticNonRelease
  deriving DecidableEq, BEq, Repr, Inhabited

def ClaimActivation.tag : ClaimActivation → String
  | .syntheticNonRelease => "synthetic-non-release"

/-! ## Public request (premises only) -/

/-- Raw claim request. Deliberately has no `complete`, `pass`, or
`certificateDigest` field. The request names one selected promoted event; the
full registry may contain cycles for other targets/profiles and is retained for
omission totality. -/
structure GeneratedClaimRequest where
  activation : ClaimActivation
  profileRequest : ProfileCompletionRequest
  judgingClosure : JudgingInputClosureCandidate
  cycleRegistry : CycleRegistryCandidate
  /-- Exactly one selected promoted event from the registry. -/
  selectedPromotedEvent : CycleEventId
  /-- Provenance for every normative claim-bearing field name. Exact set. -/
  fieldProvenance : List ClaimField
  deriving Repr

/-! ## Validated private certificate -/

structure ValidatedGeneratedClaim where
  private mk ::
    request : GeneratedClaimRequest
    profile : ValidatedProfileCompletion
    judging : ValidatedJudgingInputClosure
    registry : ValidatedCycleRegistry
    selected : CycleEventDescriptor
    /-- Derived canonical identity. Never authored by the request. -/
    claimIdentity : ContentDigest
  deriving Repr

def ValidatedGeneratedClaim.identity (v : ValidatedGeneratedClaim) : ContentDigest :=
  v.claimIdentity

def ValidatedGeneratedClaim.activation (v : ValidatedGeneratedClaim) : ClaimActivation :=
  v.request.activation

def ValidatedGeneratedClaim.profileCompletion
    (v : ValidatedGeneratedClaim) : ValidatedProfileCompletion :=
  v.profile

def ValidatedGeneratedClaim.judgingClosure
    (v : ValidatedGeneratedClaim) : ValidatedJudgingInputClosure :=
  v.judging

def ValidatedGeneratedClaim.cycleRegistry
    (v : ValidatedGeneratedClaim) : ValidatedCycleRegistry :=
  v.registry

def ValidatedGeneratedClaim.selectedEvent
    (v : ValidatedGeneratedClaim) : CycleEventDescriptor :=
  v.selected

/-! ## Diagnosed faults -/

inductive GeneratedClaimFault where
  | wrongActivation
  | profileCompletion (fault : ProfileCompletionFault)
  | judgingClosure (fault : JudgingClosureFault)
  | cycleRegistry (fault : CycleRegistryFault)
  | selectedEventMissing
  | selectedEventNotPromoted
  | selectedEventLacksFreezeIntegrity
  | mismatchedTarget
  | mismatchedProfile
  | mismatchedSourceClosure
  | mismatchedJudgingClosureIdentity
  | mismatchedChronologyTarget
  | mismatchedChronologyProfile
  | subjectCandidateMismatch
  | mismatchedJudgingScopeBinding
  | missingClaimRenderer
  | duplicateClaimRenderer
  | mismatchedClaimRendererBinding
  | malformedFieldProvenance
  | duplicateFieldProvenance (name : String)
  | missingFieldProvenance (name : String)
  | extraFieldProvenance (name : String)
  | emptyDenominator
  deriving DecidableEq, BEq, Repr, Inhabited

inductive GeneratedClaimValidationResult (ok : Type) where
  | ok (value : ok)
  | faults (diagnosed : List GeneratedClaimFault)
  deriving Repr

def GeneratedClaimValidationResult.isOk {ok : Type} :
    GeneratedClaimValidationResult ok → Bool
  | .ok _ => true
  | .faults _ => false

def GeneratedClaimValidationResult.hasFault {ok : Type}
    (result : GeneratedClaimValidationResult ok) (f : GeneratedClaimFault) : Bool :=
  match result with
  | .ok _ => false
  | .faults diagnosed => diagnosed.contains f

/-! ## Required provenance field names -/

def requiredClaimFieldNames : List String :=
  [ "target", "profile", "sourceClosure", "subjectTree", "runtime",
    "assurancePolicy", "judgingClosureIdentity", "claimRenderer",
    "catalogDenominator", "discharges", "cycleRegistry", "activation",
    "selectedPromotedEvent" ]

/-! ## Residual obligations derived from discharges -/

inductive ResidualObligationKind where
  | open
  | blocked
  | refuted
  | survivedSearch
  | acceptedByJudgment
  deriving DecidableEq, BEq, Repr, Inhabited

def ResidualObligationKind.tag : ResidualObligationKind → String
  | .open => "open"
  | .blocked => "blocked"
  | .refuted => "refuted"
  | .survivedSearch => "survivedSearch"
  | .acceptedByJudgment => "acceptedByJudgment"

structure ResidualObligation where
  requirement : RequirementId
  kind : ResidualObligationKind
  detail : String
  deriving DecidableEq, BEq, Repr, Inhabited

def residualKindOf : AssuranceState → Option (ResidualObligationKind × String)
  | .open => some (.open, "unresolved")
  | .blocked r => some (.blocked, s!"{r.id.value}:{r.kind}")
  | .refuted r => some (.refuted, r.id.value)
  | .survivedSearch r =>
      some (.survivedSearch,
        s!"generator={r.generatorId};budget={r.budgetDesc};divergences={r.divergenceCount}")
  | .acceptedBy j => some (.acceptedByJudgment, j.id.value)
  | .proved _ => none

def deriveResidualObligations
    (discharges : List RequirementDischargeCandidate) : List ResidualObligation :=
  let rows :=
    discharges.filterMap fun d =>
      match residualKindOf d.assuranceState with
      | none => none
      | some (kind, detail) =>
          some
            { requirement := d.requirement.id
              kind := kind
              detail := detail }
  rows.mergeSort (fun a b =>
    a.requirement.value < b.requirement.value ||
      (a.requirement.value == b.requirement.value && a.kind.tag ≤ b.kind.tag))

/-! ## Canonical encoding of claim structure -/

def encodeAssuranceState : AssuranceState → String
  | .open => encStr "open"
  | .blocked r => encStr "blocked" ++ encStr r.id.value ++ encStr r.kind
  | .refuted r =>
      encStr "refuted" ++ encStr r.id.value ++ encDigest r.artifact
  | .survivedSearch r =>
      encStr "survivedSearch" ++ encStr r.id.value ++ encStr r.generatorId ++
        encDigest r.sourceClosure.digest ++ encDigest r.seeds ++
        encStr r.budgetDesc ++ encDigest r.partitions ++
        encNat r.divergenceCount ++ encStr r.calibration.value
  | .acceptedBy j =>
      encStr "acceptedBy" ++ encStr j.id.value ++ encStr j.authority ++
        encDigest j.scope ++ encDigest j.invalidation
  | .proved p => encStr "proved" ++ encStr p.id.value ++ encStr p.theoremName

def encodeEnvironmentIdentity (e : EnvironmentIdentity) : String :=
  encStr e.id.value ++ encDigest e.digest

def encodeEvidenceIdentity (e : EvidenceIdentity) : String :=
  encStr e.id.value ++ encDigest e.digest

def encodeProductTreeId (t : ProductTreeId) : String :=
  encStr t.revision ++ encDigest t.content ++ encStr (if t.dirty then "dirty" else "clean")

def encodeTargetIdentity (t : TargetIdentity) : String :=
  encStr t.id.value ++ encStr t.repository ++ encDigest t.revision ++
    encDigest t.sourceClosure.digest ++ encStr t.profile.value ++
    encStr (if t.isPromoted then "promoted" else "extractedCandidate")

def encodeRequirementClosureEntry (e : RequirementClosureEntry) : String :=
  let envs :=
    (e.requiredEnvironments.mergeSort (fun a b => a.id.value ≤ b.id.value)).map
      encodeEnvironmentIdentity
  encStr e.requirement.id.value ++ encDigest e.requirement.semantics ++
    encStr e.specification.id.value ++ encDigest e.specification.semantics ++
    encStr e.boundary.id.value ++ encDigest e.boundary.semantics ++
    encStr e.scenario.id.value ++ encDigest e.scenario.digest ++
    encStr e.relation.id.value ++ encDigest e.relation.digest ++
    encStr e.adapter.id.value ++ encDigest e.adapter.digest ++
    encStr e.validator.id.value ++ encDigest e.validator.version ++
    encStr e.calibration.id.value ++ encDigest e.calibration.campaign ++
    encStr e.calibration.faultModel ++ encSeq envs

def encodeSourceDisposition : SourceDisposition → String
  | .required r => encStr "required" ++ encStr r.value
  | .excluded j => encStr "excluded" ++ encStr j.value
  | .ambiguous g => encStr "ambiguous" ++ encStr g
  | .superseded r => encStr "superseded" ++ encStr r.value

def encodeSourceItemDisposition (row : SourceItemDisposition) : String :=
  encStr row.item.value ++ encodeSourceDisposition row.disposition

def encodeDischargeCandidate (d : RequirementDischargeCandidate) : String :=
  let envs :=
    (d.environments.mergeSort (fun a b => a.id.value ≤ b.id.value)).map
      encodeEnvironmentIdentity
  let evs :=
    (d.evidence.mergeSort (fun a b => a.id.value ≤ b.id.value)).map
      encodeEvidenceIdentity
  encStr d.requirement.id.value ++ encDigest d.requirement.semantics ++
    encStr d.specification.id.value ++ encDigest d.specification.semantics ++
    encStr d.boundary.id.value ++ encDigest d.boundary.semantics ++
    encStr d.scenario.id.value ++ encDigest d.scenario.digest ++
    encStr d.relation.id.value ++ encDigest d.relation.digest ++
    encStr d.adapter.id.value ++ encDigest d.adapter.digest ++
    encStr d.validator.id.value ++ encDigest d.validator.version ++
    encStr d.calibration.id.value ++ encDigest d.calibration.campaign ++
    encStr d.calibration.faultModel ++ encodeTargetIdentity d.target ++
    encStr d.profile.value ++ encodeProductTreeId d.subjectTree.id ++
    encDigest d.runtime.id.artifact ++ encodeProductTreeId d.runtime.sourceTree ++
    encSeq envs ++ encSeq evs ++
    encStr (match d.purpose with
      | .semanticCompatibility => "semanticCompatibility"
      | .performance => "performance") ++
    encodeAssuranceState d.assuranceState

def encodeCycleEventDescriptor (d : CycleEventDescriptor) : String :=
  encStr d.id.value ++ encodeTargetIdentity d.target ++ encStr d.profile.value ++
    encDigest d.expectedFrozenJudgingClosureIdentity ++
    encStr d.outcome.tag ++ encStr d.freeze.value ++ encStr d.candidate.value

def encodeAncestryCommitRecord (r : AncestryCommitRecord) : String :=
  let parents :=
    (r.parentsWithinCapture.mergeSort (fun a b => a.value ≤ b.value)).map
      fun p => encStr p.value
  encStr r.commit.value ++ encSeq parents ++ encDigest r.judgingClosureIdentity

def encodeAncestryManifest (m : AncestryManifest) : String :=
  let captured :=
    (m.capturedCommitIds.mergeSort (fun a b => a.value ≤ b.value)).map
      fun c => encStr c.value
  let records :=
    (m.records.mergeSort (fun a b => a.commit.value ≤ b.commit.value)).map
      encodeAncestryCommitRecord
  encStr m.captureIdentity.value ++ encDigest m.extractorIdentity ++
    encodeTargetIdentity m.target ++ encStr m.profile.value ++
    encDigest m.frozenJudgingClosureIdentity ++
    encStr m.freeze.value ++ encStr m.candidate.value ++
    encSeq captured ++ encSeq records

def encodeClaimedChronologyKind : ClaimedChronologyKind → String
  | .retrospectiveFreezeIntegrity => "retrospectiveFreezeIntegrity"
  | .prospective => "prospective"

def encodeBlindFreezeProtocol : Option BlindFreezeProtocolRef → String
  | none => encStr "none"
  | some p => encStr "some" ++ encStr p.id.value ++ encDigest p.digest

def encodeFreezeIntegrityCandidate (fi : FreezeIntegrityCandidate) : String :=
  let snaps :=
    (fi.suppliedSnapshots.mergeSort (fun a b => a.commit.value ≤ b.commit.value)).map
      encodeAncestryCommitRecord
  encStr (encodeClaimedChronologyKind fi.claimedKind) ++
    encodeBlindFreezeProtocol fi.prospectiveProtocol ++
    encodeAncestryManifest fi.manifest ++
    encSeq snaps

def encodeCycleEventEntry (e : CycleEventEntry) : String :=
  encodeCycleEventDescriptor e.descriptor ++
    match e.freezeIntegrity with
    | none => encStr "none"
    | some fi => encStr "some" ++ encodeFreezeIntegrityCandidate fi

def encodeClaimField (f : ClaimField) : String :=
  encStr f.name ++ encodeFieldProvenance f.provenance

/-- Canonical frozen catalog/denotation binding for Phase M0.

Binds the full validated catalog — target, profile, source closure, inventory,
dispositions, requirement entries (requirement/spec/boundary/scenario/relation/
adapter/validator/calibration/environment denotations), accepted judgments, and
assurance policy — into the judging-input `targetProfileDecision` node content.

Discharge evidence and assurance grades are intentionally excluded: they may
be produced after freeze. Catalog semantics may not drift without rebinding.
-/
def encodeFrozenCatalogBinding
    (target : TargetIdentity) (profile : ProfileId)
    (sourceClosure : SourceClosureId) (inventory : List SourceItemId)
    (dispositions : List SourceItemDisposition)
    (requirementEntries : List RequirementClosureEntry)
    (acceptedJudgments : List JudgmentIdentity)
    (policy : ProfileAssurancePolicy) : String :=
  let inv :=
    (inventory.mergeSort (fun a b => a.value ≤ b.value)).map fun id => encStr id.value
  let disp :=
    (dispositions.mergeSort (fun a b => a.item.value ≤ b.item.value)).map
      encodeSourceItemDisposition
  let entries :=
    (requirementEntries.mergeSort
      (fun a b => a.requirement.id.value ≤ b.requirement.id.value)).map
      encodeRequirementClosureEntry
  let judgments :=
    (acceptedJudgments.mergeSort (fun a b => a.id.value ≤ b.id.value)).map fun j =>
      encStr j.id.value ++ encStr j.authority ++ encDigest j.scope ++
        encDigest j.invalidation
  encodeTargetIdentity target ++ encStr profile.value ++
    encDigest sourceClosure.digest ++
    encSeq inv ++ encSeq disp ++ encSeq entries ++ encSeq judgments ++
    encDigest policy.contentId

/-- Convenience over a catalog candidate. -/
def encodeFrozenCatalogBindingOf (c : CatalogClosureCandidate) : String :=
  encodeFrozenCatalogBinding
    c.target c.profile c.sourceClosure c.inventory c.dispositions
    c.requirementEntries c.acceptedJudgments c.assurancePolicy

/-- Convenience over a validated catalog. -/
def encodeFrozenCatalogBindingValidated (c : ValidatedCatalogClosure) : String :=
  encodeFrozenCatalogBinding
    c.target c.profile c.sourceClosure c.inventory c.dispositions
    c.requirementEntries c.acceptedJudgments c.assurancePolicy

def claimRendererBindingPayload : String :=
  encStr "generated-claim-renderer" ++ encNat generatedClaimSchemaVersion

/-- Canonical claim identity over the full validated structure.
Semantic sets are sorted; freeze-integrity payloads are included so two valid
registries with different capture/extractor/ancestry/protocol/snapshots cannot
share an identity. -/
def computeGeneratedClaimIdentity
    (req : GeneratedClaimRequest)
    (profile : ValidatedProfileCompletion)
    (judging : ValidatedJudgingInputClosure)
    (registry : ValidatedCycleRegistry)
    (selected : CycleEventDescriptor) : ContentDigest :=
  let catalog := profile.catalog
  let inventory :=
    (catalog.inventory.mergeSort (fun a b => a.value ≤ b.value)).map
      fun id => encStr id.value
  let dispositions :=
    (catalog.dispositions.mergeSort (fun a b => a.item.value ≤ b.item.value)).map
      encodeSourceItemDisposition
  let entries :=
    (catalog.requirementEntries.mergeSort
      (fun a b => a.requirement.id.value ≤ b.requirement.id.value)).map
      encodeRequirementClosureEntry
  let judgments :=
    (catalog.acceptedJudgments.mergeSort
      (fun a b => a.id.value ≤ b.id.value)).map fun j =>
      encStr j.id.value ++ encStr j.authority ++ encDigest j.scope ++
        encDigest j.invalidation
  let discharges :=
    (profile.discharges.mergeSort
      (fun a b => a.requirement.id.value ≤ b.requirement.id.value)).map
      encodeDischargeCandidate
  let registryEntries :=
    (registry.entries.mergeSort
      (fun a b => a.descriptor.id.value ≤ b.descriptor.id.value)).map
      encodeCycleEventEntry
  let fields :=
    (req.fieldProvenance.mergeSort (fun a b => a.name ≤ b.name)).map encodeClaimField
  let rendererNodes :=
    (judging.nodes.filter (fun n => n.category == .claimRenderer)).mergeSort
      (fun a b => a.id.value ≤ b.id.value)
  let rendererPart :=
    encSeq (rendererNodes.map fun n =>
      encStr n.id.value ++ encDigest n.content ++ encodeFieldProvenance n.provenance)
  digest (
    encNat generatedClaimSchemaVersion ++
    encStr req.activation.tag ++
    encStr selected.id.value ++
    encodeTargetIdentity catalog.target ++
    encStr catalog.profile.value ++
    encDigest catalog.sourceClosure.digest ++
    encodeProductTreeId profile.subjectTree.id ++
    encDigest profile.runtime.id.artifact ++
    encodeProductTreeId profile.runtime.sourceTree ++
    encDigest catalog.assurancePolicy.contentId ++
    encDigest judging.identity ++
    encSeq inventory ++
    encSeq dispositions ++
    encSeq entries ++
    encSeq judgments ++
    encSeq discharges ++
    encSeq registryEntries ++
    encSeq fields ++
    rendererPart)

/-! ## Cross-binding diagnosis -/

private def pushClaimFault
    (faults : List GeneratedClaimFault) (cond : Bool) (f : GeneratedClaimFault) :
    List GeneratedClaimFault :=
  if cond then faults ++ [f] else faults

private def claimRendererNodes (nodes : List JudgingInputNode) : List JudgingInputNode :=
  nodes.filter (fun n => n.category == .claimRenderer)

private def scopeDecisionNodes (nodes : List JudgingInputNode) : List JudgingInputNode :=
  nodes.filter (fun n => n.category == .targetProfileDecision)

private def lookupSelectedDescriptor?
    (registry : ValidatedCycleRegistry) (id : CycleEventId) :
    Option CycleEventDescriptor :=
  registry.importedDescriptors.find? (fun d => d.id == id)

private def selectedEntry?
    (registry : ValidatedCycleRegistry) (id : CycleEventId) :
    Option CycleEventEntry :=
  registry.entries.find? (fun e => e.descriptor.id == id)

private def crossBindDiagnose
    (req : GeneratedClaimRequest)
    (profile : ValidatedProfileCompletion)
    (judging : ValidatedJudgingInputClosure)
    (registry : ValidatedCycleRegistry) :
    List GeneratedClaimFault × Option CycleEventDescriptor :=
  let faults : List GeneratedClaimFault := []
  let faults :=
    pushClaimFault faults (req.activation != .syntheticNonRelease) .wrongActivation
  let catalog := profile.catalog
  let faults :=
    pushClaimFault faults
      (catalog.inventory.isEmpty || catalog.requirementEntries.isEmpty)
      .emptyDenominator
  let faults :=
    pushClaimFault faults
      (catalog.target.sourceClosure != catalog.sourceClosure)
      .mismatchedSourceClosure
  let faults :=
    pushClaimFault faults (!req.fieldProvenance.all ClaimField.hasAdmissibleProvenance)
      .malformedFieldProvenance
  let names := req.fieldProvenance.map (·.name)
  let faults :=
    if listNodup names then faults
    else
      names.foldl (fun acc name =>
        if names.count name > 1 && !(acc.contains (.duplicateFieldProvenance name)) then
          acc ++ [.duplicateFieldProvenance name]
        else acc) faults
  let faults :=
    requiredClaimFieldNames.foldl (fun acc name =>
      if names.contains name then acc else acc ++ [.missingFieldProvenance name]) faults
  let faults :=
    names.foldl (fun acc name =>
      if requiredClaimFieldNames.contains name then acc
      else if acc.contains (.extraFieldProvenance name) then acc
      else acc ++ [.extraFieldProvenance name]) faults
  let renderers := claimRendererNodes judging.nodes
  let faults := pushClaimFault faults renderers.isEmpty .missingClaimRenderer
  let faults := pushClaimFault faults (renderers.length > 1) .duplicateClaimRenderer
  let expectedRenderer := digest claimRendererBindingPayload
  let faults :=
    pushClaimFault faults
      (renderers.any (fun n => n.content != expectedRenderer))
      .mismatchedClaimRendererBinding
  let expectedScope := digest (encodeFrozenCatalogBindingValidated catalog)
  let scopeNodes := scopeDecisionNodes judging.nodes
  let faults :=
    pushClaimFault faults
      (scopeNodes.isEmpty || scopeNodes.any (fun n => n.content != expectedScope))
      .mismatchedJudgingScopeBinding
  match lookupSelectedDescriptor? registry req.selectedPromotedEvent with
  | none => (faults ++ [.selectedEventMissing], none)
  | some selected =>
      if selected.outcome != .promoted then
        -- Primary diagnosis only: non-promoted events must not carry freeze
        -- integrity, so do not also demand selectedEventLacksFreezeIntegrity.
        (faults ++ [.selectedEventNotPromoted], some selected)
      else
        let faults :=
          match selectedEntry? registry selected.id with
          | some { freezeIntegrity := some _, .. } => faults
          | some { freezeIntegrity := none, .. } =>
              faults ++ [.selectedEventLacksFreezeIntegrity]
          | none => faults ++ [.selectedEventMissing]
        let faults :=
          pushClaimFault faults (selected.target != catalog.target)
            .mismatchedChronologyTarget
        let faults :=
          pushClaimFault faults (selected.profile != catalog.profile)
            .mismatchedChronologyProfile
        let faults :=
          pushClaimFault faults (selected.target.id != catalog.target.id) .mismatchedTarget
        let faults :=
          pushClaimFault faults (selected.profile != catalog.profile) .mismatchedProfile
        let faults :=
          pushClaimFault faults
            (selected.expectedFrozenJudgingClosureIdentity != judging.identity)
            .mismatchedJudgingClosureIdentity
        let faults :=
          pushClaimFault faults
            (profile.subjectTree.id.revision != selected.candidate.value)
            .subjectCandidateMismatch
        let faults :=
          match selectedEntry? registry selected.id with
          | some { freezeIntegrity := some fi, .. } =>
              let faults :=
                pushClaimFault faults (fi.manifest.target != catalog.target)
                  .mismatchedChronologyTarget
              let faults :=
                pushClaimFault faults (fi.manifest.profile != catalog.profile)
                  .mismatchedChronologyProfile
              let faults :=
                pushClaimFault faults
                  (fi.manifest.frozenJudgingClosureIdentity != judging.identity)
                  .mismatchedJudgingClosureIdentity
              pushClaimFault faults
                (fi.manifest.candidate.value != profile.subjectTree.id.revision)
                .subjectCandidateMismatch
          | _ => faults
        (faults, some selected)

/-! ## Smart constructor -/

def validateGeneratedClaim (req : GeneratedClaimRequest) :
    GeneratedClaimValidationResult ValidatedGeneratedClaim :=
  match validateProfileFromCatalog req.profileRequest with
  | .faults fs => .faults (fs.map GeneratedClaimFault.profileCompletion)
  | .ok profile =>
      match validateJudgingInputClosure req.judgingClosure with
      | .faults fs => .faults (fs.map GeneratedClaimFault.judgingClosure)
      | .ok judging =>
          match validateCycleRegistry req.cycleRegistry with
          | .faults fs => .faults (fs.map GeneratedClaimFault.cycleRegistry)
          | .ok registry =>
              match crossBindDiagnose req profile judging registry with
              | ([], some selected) =>
                  let identity :=
                    computeGeneratedClaimIdentity
                      req profile judging registry selected
                  .ok ⟨req, profile, judging, registry, selected, identity⟩
              | (diagnosed, _) => .faults diagnosed

/-! ## Provenance labels -/

private def provenanceLabel : FieldProvenance → String
  | .imported s => s!"imported({s.id.value})"
  | .derived c => s!"derived({c.id.value}/{c.verifier.value})"
  | .calibrated c => s!"calibrated({c.id.value})"
  | .judgment j => s!"judgment({j.id.value}/{j.authority})"

/-! ## Renderer — only path to human-facing completion language -/

structure RenderedClaimReport where
  activation : ClaimActivation
  claimIdentity : ContentDigest
  text : String
  residualObligations : List ResidualObligation
  /-- Derived: true only when denominator nonempty, residuals exclude open /
  blocked / refuted, and activation is synthetic. Never an authored Boolean. -/
  syntheticComplete : Bool
  deriving Repr

private def assuranceGradeLabel : AssuranceState → String
  | .open => "open"
  | .blocked r => s!"blocked({r.id.value}/{r.kind})"
  | .refuted r => s!"refuted({r.id.value})"
  | .survivedSearch r =>
      s!"survivedSearch(generator={r.generatorId};budget={r.budgetDesc};divergences={r.divergenceCount})"
  | .acceptedBy j => s!"acceptedByJudgment({j.id.value}/authority={j.authority})"
  | .proved p => s!"proved({p.id.value}/{p.theoremName})"

/-- Render a validated certificate. Never collapses bounded search or judgment
into proof. Never prints a scalar parity percentage. Empty denominator cannot
render complete. Always prints epistemic limitations and per-field provenance. -/
private def textHas (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

private def uniqueSortedStrings (xs : List String) : List String :=
  let sorted := xs.mergeSort (fun a b => a ≤ b)
  sorted.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []

def renderGeneratedClaim (v : ValidatedGeneratedClaim) : RenderedClaimReport :=
  let catalog := v.profile.catalog
  let residuals := deriveResidualObligations v.profile.discharges
  let denomEmpty :=
    catalog.inventory.isEmpty || catalog.requirementEntries.isEmpty
  let unresolved :=
    residuals.any fun r =>
      r.kind == .open || r.kind == .blocked || r.kind == .refuted
  let syntheticComplete :=
    v.request.activation == .syntheticNonRelease && !denomEmpty && !unresolved
  let exclusions :=
    (catalog.dispositions.filterMap fun row =>
      match row.disposition with
      | .excluded j => some s!"{row.item.value}->{j.value}"
      | _ => none).mergeSort (fun a b => a ≤ b)
  let envScope :=
    String.intercalate ","
      (uniqueSortedStrings
        (v.profile.discharges.flatMap (·.environments.map (·.id.value))))
  let grades :=
    String.intercalate "\n"
      (v.profile.discharges.map fun d =>
        s!"  - {d.requirement.id.value}: {assuranceGradeLabel d.assuranceState}")
  let residualText :=
    if residuals.isEmpty then "  (none)"
    else
      String.intercalate "\n"
        (residuals.map fun r => s!"  - {r.requirement.value}: {r.kind.tag} ({r.detail})")
  let searchBounds :=
    String.intercalate "\n"
      (v.profile.discharges.filterMap fun d =>
        match d.assuranceState with
        | .survivedSearch r =>
            some s!"  - {d.requirement.id.value}: budget={r.budgetDesc}; seeds={r.seeds.value}"
        | _ => none)
  let searchSection :=
    if searchBounds.isEmpty then "  (none)" else searchBounds
  let provenanceText :=
    String.intercalate "\n"
      ((v.request.fieldProvenance.mergeSort (fun a b => a.name ≤ b.name)).map fun f =>
        s!"  - {f.name}: {provenanceLabel f.provenance}")
  let cleanTag := if v.profile.subjectTree.id.dirty then "dirty" else "clean"
  let statusLine :=
    if denomEmpty then
      "STATUS: incomplete (empty denominator; complete language forbidden)"
    else if unresolved then
      "STATUS: incomplete (open/blocked/refuted obligations remain)"
    else if syntheticComplete then
      "STATUS: synthetic-complete under declared scope (NON-RELEASE; not a real Tgrad claim)"
    else
      "STATUS: incomplete"
  let text :=
    String.intercalate "\n"
      [ "GENERATED CLAIM REPORT",
        s!"schema_version={generatedClaimSchemaVersion}",
        s!"activation={v.request.activation.tag}",
        s!"claim_identity={v.claimIdentity.value}",
        "EPISTEMIC LIMITATIONS:",
        "  - ContentDigest is a shape token, not a cryptographic digest",
        "  - synthetic one-requirement fixture is unauthenticated",
        "  - no real Tgrad completeness or tinygrad-compatibility claim exists",
        "  - Lean/Python agreement is not authentication",
        "  - field provenance classification is authored premise, not cryptographic field grounding",
        s!"target={catalog.target.id.value}@{catalog.target.revision.value} repo={catalog.target.repository}",
        s!"profile={catalog.profile.value}",
        s!"source_closure={catalog.sourceClosure.digest.value}",
        s!"subject_tree={v.profile.subjectTree.id.revision}/{v.profile.subjectTree.id.content.value}/{cleanTag}",
        s!"runtime_artifact={v.profile.runtime.id.artifact.value}",
        s!"runtime_source_tree={v.profile.runtime.sourceTree.revision}/{v.profile.runtime.sourceTree.content.value}",
        s!"selected_promoted_event={v.selected.id.value}",
        s!"selected_candidate_commit={v.selected.candidate.value}",
        s!"subject_candidate_binding={v.profile.subjectTree.id.revision}=={v.selected.candidate.value}",
        "environment_scope={" ++ envScope ++ "}",
        s!"denominator_inventory={catalog.inventory.length}",
        s!"denominator_requirements={catalog.requirementEntries.length}",
        s!"exclusions=[{String.intercalate "," exclusions}]",
        "tolerances: (none declared in synthetic fixture)",
        "search_bounds:",
        searchSection,
        s!"assurance_policy={catalog.assurancePolicy.contentId.value}",
        s!"judging_closure_identity={v.judging.identity.value}",
        "field_provenance:",
        provenanceText,
        "assurance_grades:",
        grades,
        "residual_obligations:",
        residualText,
        statusLine,
        "NOTE: bounded search and accepted judgment are never printed as proof.",
        "NOTE: no scalar parity percentage is defined or printed." ]
  { activation := v.request.activation
    claimIdentity := v.claimIdentity
    text := text
    residualObligations := residuals
    syntheticComplete := syntheticComplete }

/-! ## Attempt / failure report from raw request

Profile validation rejects open/blocked/refuted under requireProof before the
validated renderer runs. This path always emits a report from the raw request
and typed faults: it may be incomplete and list blockers, but never emits
completion language on failure.
-/

structure GeneratedClaimAttemptReport where
  validationOk : Bool
  faults : List GeneratedClaimFault
  residualObligations : List ResidualObligation
  text : String
  syntheticComplete : Bool
  deriving Repr

private def faultTag : GeneratedClaimFault → String
  | .wrongActivation => "wrongActivation"
  | .profileCompletion _ => "profileCompletion"
  | .judgingClosure _ => "judgingClosure"
  | .cycleRegistry _ => "cycleRegistry"
  | .selectedEventMissing => "selectedEventMissing"
  | .selectedEventNotPromoted => "selectedEventNotPromoted"
  | .selectedEventLacksFreezeIntegrity => "selectedEventLacksFreezeIntegrity"
  | .mismatchedTarget => "mismatchedTarget"
  | .mismatchedProfile => "mismatchedProfile"
  | .mismatchedSourceClosure => "mismatchedSourceClosure"
  | .mismatchedJudgingClosureIdentity => "mismatchedJudgingClosureIdentity"
  | .mismatchedChronologyTarget => "mismatchedChronologyTarget"
  | .mismatchedChronologyProfile => "mismatchedChronologyProfile"
  | .subjectCandidateMismatch => "subjectCandidateMismatch"
  | .mismatchedJudgingScopeBinding => "mismatchedJudgingScopeBinding"
  | .missingClaimRenderer => "missingClaimRenderer"
  | .duplicateClaimRenderer => "duplicateClaimRenderer"
  | .mismatchedClaimRendererBinding => "mismatchedClaimRendererBinding"
  | .malformedFieldProvenance => "malformedFieldProvenance"
  | .duplicateFieldProvenance n => s!"duplicateFieldProvenance:{n}"
  | .missingFieldProvenance n => s!"missingFieldProvenance:{n}"
  | .extraFieldProvenance n => s!"extraFieldProvenance:{n}"
  | .emptyDenominator => "emptyDenominator"

def renderGeneratedClaimAttempt (req : GeneratedClaimRequest) :
    GeneratedClaimAttemptReport :=
  let residuals := deriveResidualObligations req.profileRequest.discharges
  match validateGeneratedClaim req with
  | .ok v =>
      let report := renderGeneratedClaim v
      { validationOk := true
        faults := []
        residualObligations := report.residualObligations
        text := report.text
        syntheticComplete := report.syntheticComplete }
  | .faults fs =>
      let residualText :=
        if residuals.isEmpty then "  (none)"
        else
          String.intercalate "\n"
            (residuals.map fun r =>
              s!"  - {r.requirement.value}: {r.kind.tag} ({r.detail})")
      let faultText :=
        String.intercalate "\n" (fs.map fun f => s!"  - {faultTag f}")
      let text :=
        String.intercalate "\n"
          [ "GENERATED CLAIM ATTEMPT REPORT",
            s!"schema_version={generatedClaimSchemaVersion}",
            s!"activation={req.activation.tag}",
            "EPISTEMIC LIMITATIONS:",
            "  - ContentDigest is a shape token, not a cryptographic digest",
            "  - synthetic one-requirement fixture is unauthenticated",
            "  - no real Tgrad completeness or tinygrad-compatibility claim exists",
            "  - Lean/Python agreement is not authentication",
            "  - field provenance classification is authored premise, not cryptographic field grounding",
            "validation_ok=false",
            "synthetic_complete=false",
            "STATUS: incomplete (validation failed; completion language forbidden)",
            "diagnosed_faults:",
            faultText,
            "residual_obligations_from_raw_discharges:",
            residualText,
            "NOTE: this failure report may be incomplete but never claims completion.",
            "NOTE: bounded search and accepted judgment are never printed as proof.",
            "NOTE: no scalar parity percentage is defined or printed." ]
      { validationOk := false
        faults := fs
        residualObligations := residuals
        text := text
        syntheticComplete := false }

/-! ## Synthetic fixtures (unauthenticated; non-release) -/

def claimSubject : ProductTreeIdentity where
  id := { revision := toyCommitC.value, content := digest "subj-tree", dirty := false }

def claimRuntime : RuntimeIdentity where
  id := { artifact := digest "runtime-dylib" }
  sourceTree := claimSubject.id

def claimDischargeCandidate : RequirementDischargeCandidate :=
  { toyDischargeCandidate with
    subjectTree := claimSubject
    runtime := claimRuntime }

def claimCatalogCandidate : CatalogClosureCandidate :=
  toyCatalogCandidate

def claimProfileRequest : ProfileCompletionRequest where
  catalog := claimCatalogCandidate
  subjectTree := claimSubject
  runtime := claimRuntime
  discharges := [claimDischargeCandidate]

def claimScopeContent : ContentDigest :=
  digest (encodeFrozenCatalogBindingOf claimCatalogCandidate)

def claimRendererContent : ContentDigest :=
  digest claimRendererBindingPayload

/-- Judging-input graph with category-node contents bound to claim scope fields. -/
def claimJudgingNodes : List JudgingInputNode :=
  toyJudgingNodes.map fun n =>
    if n.category == .targetProfileDecision then
      { n with content := claimScopeContent }
    else if n.category == .claimRenderer then
      { n with content := claimRendererContent }
    else n

def claimJudgingCandidate : JudgingInputClosureCandidate where
  schemaVersion := judgingInputClosureSchemaVersion
  nodes := claimJudgingNodes
  roots := toyJudgingRoots
  discoveredInventory := claimJudgingNodes.map (·.id)

def claimJudgingIdentity : ContentDigest :=
  computeJudgingClosureIdentity claimJudgingCandidate

def claimAncestryRecords : List AncestryCommitRecord :=
  [ { commit := toyCommitA, parentsWithinCapture := [],
      judgingClosureIdentity := claimJudgingIdentity }
  , { commit := toyCommitB, parentsWithinCapture := [toyCommitA],
      judgingClosureIdentity := claimJudgingIdentity }
  , { commit := toyCommitC, parentsWithinCapture := [toyCommitB],
      judgingClosureIdentity := claimJudgingIdentity } ]

def claimFreezeIntegrity : FreezeIntegrityCandidate where
  claimedKind := .retrospectiveFreezeIntegrity
  prospectiveProtocol := none
  manifest :=
    { captureIdentity := { value := "capture.claim.v1" }
      extractorIdentity := digest "extractor-claim"
      target := toyTarget
      profile := toyProfile
      frozenJudgingClosureIdentity := claimJudgingIdentity
      freeze := toyCommitA
      candidate := toyCommitC
      capturedCommitIds := [toyCommitA, toyCommitB, toyCommitC]
      records := claimAncestryRecords }
  suppliedSnapshots := claimAncestryRecords

def claimPromotedDescriptor : CycleEventDescriptor where
  id := { value := "cycle.claim.promoted.1" }
  target := toyTarget
  profile := toyProfile
  expectedFrozenJudgingClosureIdentity := claimJudgingIdentity
  outcome := .promoted
  freeze := toyCommitA
  candidate := toyCommitC

def claimRejectedDescriptor : CycleEventDescriptor where
  id := { value := "cycle.claim.rejected.1" }
  target := toyTarget
  profile := toyProfile
  expectedFrozenJudgingClosureIdentity := claimJudgingIdentity
  outcome := .rejected
  freeze := toyCommitA
  candidate := toyCommitC

/-- Foreign-target rejected cycle retained for registry totality. Must not
force the claim's selected-event cross-bind to fail. -/
def foreignRejectedDescriptor : CycleEventDescriptor where
  id := { value := "cycle.foreign.rejected.1" }
  target :=
    { toyTarget with
      id := { value := "target.foreign" }
      profile := { value := "profile.foreign.v1" } }
  profile := { value := "profile.foreign.v1" }
  expectedFrozenJudgingClosureIdentity := digest "foreign-judging-shape"
  outcome := .rejected
  freeze := toyCommitA
  candidate := toyCommitC

def claimCycleRegistry : CycleRegistryCandidate where
  importedDescriptors :=
    [claimPromotedDescriptor, claimRejectedDescriptor, foreignRejectedDescriptor]
  entries :=
    [ { descriptor := claimPromotedDescriptor
        freezeIntegrity := some claimFreezeIntegrity }
    , { descriptor := claimRejectedDescriptor, freezeIntegrity := none }
    , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ]

def claimFieldProvenance : List ClaimField :=
  let imp : FieldProvenance := .imported sampleImported
  let der : FieldProvenance := .derived sampleDerived
  let cal : FieldProvenance := .calibrated sampleCalibration
  let jud : FieldProvenance := .judgment sampleJudgment
  [ { name := "target", provenance := imp },
    { name := "profile", provenance := jud },
    { name := "sourceClosure", provenance := imp },
    { name := "subjectTree", provenance := der },
    { name := "runtime", provenance := der },
    { name := "assurancePolicy", provenance := jud },
    { name := "judgingClosureIdentity", provenance := der },
    { name := "claimRenderer", provenance := der },
    { name := "catalogDenominator", provenance := imp },
    { name := "discharges", provenance := cal },
    { name := "cycleRegistry", provenance := der },
    { name := "activation", provenance := jud },
    { name := "selectedPromotedEvent", provenance := der } ]

def toyGeneratedClaimRequest : GeneratedClaimRequest where
  activation := .syntheticNonRelease
  profileRequest := claimProfileRequest
  judgingClosure := claimJudgingCandidate
  cycleRegistry := claimCycleRegistry
  selectedPromotedEvent := claimPromotedDescriptor.id
  fieldProvenance := claimFieldProvenance

/-! ## Positive executable checks -/

theorem toy_generated_claim_validates :
    (validateGeneratedClaim toyGeneratedClaimRequest).isOk = true := by
  native_decide

theorem toy_generated_claim_identity_is_derived :
    (match validateGeneratedClaim toyGeneratedClaimRequest with
     | .ok v =>
         v.claimIdentity ==
           computeGeneratedClaimIdentity
             toyGeneratedClaimRequest v.profile v.judging v.registry v.selected
     | .faults _ => false) = true := by
  native_decide

theorem toy_renderer_marks_synthetic_complete_without_parity_percent :
    (match validateGeneratedClaim toyGeneratedClaimRequest with
     | .ok v =>
         let report := renderGeneratedClaim v
         report.syntheticComplete &&
         textHas report.text "synthetic-complete" &&
         textHas report.text "NON-RELEASE" &&
         textHas report.text "shape token" &&
         textHas report.text "field_provenance:" &&
         textHas report.text "selected_promoted_event=" &&
         textHas report.text "subject_candidate_binding=" &&
         !(textHas report.text "parity%") &&
         report.residualObligations.isEmpty
     | .faults _ => false) = true := by
  native_decide

/-- Reordering semantic sets must preserve identity. -/
def toyGeneratedClaimRequestReordered : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    fieldProvenance := claimFieldProvenance.reverse
    cycleRegistry :=
      { importedDescriptors :=
          [foreignRejectedDescriptor, claimRejectedDescriptor, claimPromotedDescriptor]
        entries :=
          [ { descriptor := foreignRejectedDescriptor, freezeIntegrity := none }
          , { descriptor := claimRejectedDescriptor, freezeIntegrity := none }
          , { descriptor := claimPromotedDescriptor
              freezeIntegrity := some claimFreezeIntegrity } ] } }

theorem semantic_set_reorder_preserves_claim_identity :
    (match validateGeneratedClaim toyGeneratedClaimRequest,
           validateGeneratedClaim toyGeneratedClaimRequestReordered with
     | .ok a, .ok b => a.claimIdentity == b.claimIdentity
     | _, _ => false) = true := by
  native_decide

/-- Alternate valid global registry entry (foreign rejected cycle) does not
invalidate the selected promoted claim. -/
theorem alternate_valid_global_registry_entry_preserved :
    (validateGeneratedClaim toyGeneratedClaimRequest).isOk = true &&
    (toyGeneratedClaimRequest.cycleRegistry.importedDescriptors.any
      (fun d => d.id == foreignRejectedDescriptor.id)) = true := by
  native_decide

/-! ## Negative executable checks — exact diagnosed reasons -/

def staleJudgingIdentityRegistry : CycleRegistryCandidate :=
  let stale := digest "stale-judging-closure-identity"
  let desc :=
    { claimPromotedDescriptor with expectedFrozenJudgingClosureIdentity := stale }
  let records :=
    claimAncestryRecords.map fun r =>
      { r with judgingClosureIdentity := stale }
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          frozenJudgingClosureIdentity := stale
          records := records }
      suppliedSnapshots := records }
  { importedDescriptors :=
      [desc, claimRejectedDescriptor, foreignRejectedDescriptor]
    entries :=
      [ { descriptor := desc, freezeIntegrity := some fi }
      , { descriptor := claimRejectedDescriptor, freezeIntegrity := none }
      , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] }

def staleIdentityRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with cycleRegistry := staleJudgingIdentityRegistry }

theorem stale_derived_judging_identity_rejected :
    (validateGeneratedClaim staleIdentityRequest).hasFault
      .mismatchedJudgingClosureIdentity = true := by
  native_decide

def missingDischargeClaimRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    profileRequest := { claimProfileRequest with discharges := [] } }

theorem missing_discharge_rejected_by_claim_guard :
    (validateGeneratedClaim missingDischargeClaimRequest).hasFault
      (.profileCompletion (.profile (.missingDischarge toyRequirementId))) = true := by
  native_decide

/-- Catalog/target moves while selected chronology stays on the toy target. -/
def mismatchedChronologyTargetRequest : GeneratedClaimRequest :=
  let otherTarget :=
    { toyTarget with id := { value := "target.other" } }
  let catalog := { claimCatalogCandidate with target := otherTarget }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  -- Keep selected chronology on toyTarget while catalog moves.
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := catalog
        discharges :=
          [{ claimDischargeCandidate with target := otherTarget }] }
    judgingClosure := judging }

theorem changed_target_rejected_by_claim_guard :
    (validateGeneratedClaim mismatchedChronologyTargetRequest).hasFault
      .mismatchedChronologyTarget = true := by
  native_decide

def mismatchedChronologyProfileRequest : GeneratedClaimRequest :=
  let otherProfile : ProfileId := { value := "profile.other.v1" }
  let otherPolicy := uniformPolicy otherProfile 1 .requireProof
  let otherTarget := { toyTarget with profile := otherProfile }
  let catalog :=
    { claimCatalogCandidate with
      target := otherTarget
      profile := otherProfile
      assurancePolicy := otherPolicy }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := catalog
        discharges :=
          [{ claimDischargeCandidate with
              target := otherTarget
              profile := otherProfile }] }
    judgingClosure := judging }

theorem changed_profile_rejected_by_claim_guard :
    (validateGeneratedClaim mismatchedChronologyProfileRequest).hasFault
      .mismatchedChronologyProfile = true := by
  native_decide

def changedSourceClosureRequest : GeneratedClaimRequest :=
  let otherClosure : SourceClosureId := { digest := digest "other-source-closure" }
  let otherTarget := { toyTarget with sourceClosure := otherClosure }
  let catalog :=
    { claimCatalogCandidate with
      target := otherTarget
      sourceClosure := otherClosure }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := catalog
        discharges := [{ claimDischargeCandidate with target := otherTarget }] }
    judgingClosure := judging }

theorem changed_source_closure_rejected_by_claim_guard :
    (validateGeneratedClaim changedSourceClosureRequest).hasFault
      .mismatchedChronologyTarget = true := by
  native_decide

/-- Subject tree revision unrelated to promoted candidate commit. -/
def subjectCandidateMismatchRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        subjectTree := otherSubject
        runtime :=
          { id := { artifact := digest "runtime-dylib" }, sourceTree := otherSubject.id }
        discharges :=
          [{ claimDischargeCandidate with
              subjectTree := otherSubject
              runtime :=
                { id := { artifact := digest "runtime-dylib" }
                  sourceTree := otherSubject.id } }] } }

theorem subject_candidate_mismatch_rejected :
    (validateGeneratedClaim subjectCandidateMismatchRequest).hasFault
      .subjectCandidateMismatch = true := by
  native_decide

/-- Subject content change with candidate still bound: validates, identity moves. -/
def changedSubjectContentRequest : GeneratedClaimRequest :=
  let otherTree : ProductTreeIdentity :=
    { id :=
        { revision := toyCommitC.value
          content := digest "subj-tree-OTHER"
          dirty := false } }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        subjectTree := otherTree
        runtime :=
          { id := { artifact := digest "runtime-dylib" }, sourceTree := otherTree.id }
        discharges :=
          [{ claimDischargeCandidate with
              subjectTree := otherTree
              runtime :=
                { id := { artifact := digest "runtime-dylib" }
                  sourceTree := otherTree.id } }] } }

theorem changed_subject_tree_changes_claim_identity :
    (match validateGeneratedClaim toyGeneratedClaimRequest,
           validateGeneratedClaim changedSubjectContentRequest with
     | .ok a, .ok b => a.claimIdentity != b.claimIdentity
     | _, _ => false) = true := by
  native_decide

def changedRuntimeRequest : GeneratedClaimRequest :=
  let otherRuntime : RuntimeIdentity :=
    { id := { artifact := digest "runtime-OTHER" }, sourceTree := claimSubject.id }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        runtime := otherRuntime
        discharges := [{ claimDischargeCandidate with runtime := otherRuntime }] } }

theorem changed_runtime_changes_claim_identity :
    (match validateGeneratedClaim toyGeneratedClaimRequest,
           validateGeneratedClaim changedRuntimeRequest with
     | .ok a, .ok b => a.claimIdentity != b.claimIdentity
     | _, _ => false) = true := by
  native_decide

/-- Policy changes while judging scope node is unchanged → binding fault. -/
def unchangedClosureChangedPolicyRequest : GeneratedClaimRequest :=
  let otherPolicy := uniformPolicy toyProfile 2 .requireProof
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := { claimCatalogCandidate with assurancePolicy := otherPolicy } } }

theorem unchanged_closure_changed_policy_rejected :
    (validateGeneratedClaim unchangedClosureChangedPolicyRequest).hasFault
      .mismatchedJudgingScopeBinding = true := by
  native_decide

/-- Post-freeze catalog denotation drift: requirement semantics change in both
catalog entry and discharge (kept cross-bound), while judging nodes/closure and
cycle registry stay frozen. Must reject — catalog semantics are freeze-bound. -/
def postFreezeDenotationDriftRequest : GeneratedClaimRequest :=
  let mutatedReq : RequirementIdentity :=
    { toyRequirement with semantics := digest "mutated-after-freeze" }
  let mutatedEntry : RequirementClosureEntry :=
    { toyRequirementEntry with requirement := mutatedReq }
  let mutatedDischarge : RequirementDischargeCandidate :=
    { claimDischargeCandidate with requirement := mutatedReq }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog :=
          { claimCatalogCandidate with requirementEntries := [mutatedEntry] }
        discharges := [mutatedDischarge] } }

theorem post_freeze_catalog_denotation_drift_rejected :
    (validateGeneratedClaim postFreezeDenotationDriftRequest).hasFault
      .mismatchedJudgingScopeBinding = true := by
  native_decide

/-- Same catalog denotation drift rebound into judging + chronology: validates
and changes identity. -/
def postFreezeDenotationReboundRequest : GeneratedClaimRequest :=
  let mutatedReq : RequirementIdentity :=
    { toyRequirement with semantics := digest "mutated-after-freeze" }
  let mutatedEntry : RequirementClosureEntry :=
    { toyRequirementEntry with requirement := mutatedReq }
  let mutatedDischarge : RequirementDischargeCandidate :=
    { claimDischargeCandidate with requirement := mutatedReq }
  let catalog :=
    { claimCatalogCandidate with requirementEntries := [mutatedEntry] }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  let jid := computeJudgingClosureIdentity judging
  let records :=
    claimAncestryRecords.map fun r => { r with judgingClosureIdentity := jid }
  let desc :=
    { claimPromotedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let rejected :=
    { claimRejectedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          frozenJudgingClosureIdentity := jid
          records := records }
      suppliedSnapshots := records }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := catalog
        discharges := [mutatedDischarge] }
    judgingClosure := judging
    cycleRegistry :=
      { importedDescriptors := [desc, rejected, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := desc, freezeIntegrity := some fi }
          , { descriptor := rejected, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem post_freeze_denotation_rebound_changes_identity :
    (match validateGeneratedClaim toyGeneratedClaimRequest,
           validateGeneratedClaim postFreezeDenotationReboundRequest with
     | .ok a, .ok b => a.claimIdentity != b.claimIdentity
     | _, _ => false) = true := by
  native_decide

/-- Discharge evidence may change after freeze without rebinding the catalog. -/
def postFreezeEvidenceOnlyRequest : GeneratedClaimRequest :=
  let otherEvidence : EvidenceIdentity :=
    { id := { value := "ev.toy.POST" }, digest := digest "ev-hash-POST" }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        discharges :=
          [{ claimDischargeCandidate with evidence := [otherEvidence] }] } }

theorem post_freeze_evidence_only_still_validates :
    (validateGeneratedClaim postFreezeEvidenceOnlyRequest).isOk = true := by
  native_decide

/-- Inventory gains an undisposed item; frozen binding rebound so the primary
fault is catalog missingDisposition (not scope-binding drift). -/
def missingDispositionClaimRequest : GeneratedClaimRequest :=
  let catalog :=
    { claimCatalogCandidate with inventory := [toyItem, toyItemOther] }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  let jid := computeJudgingClosureIdentity judging
  let records :=
    claimAncestryRecords.map fun r => { r with judgingClosureIdentity := jid }
  let desc :=
    { claimPromotedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let rejected :=
    { claimRejectedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          frozenJudgingClosureIdentity := jid
          records := records }
      suppliedSnapshots := records }
  { toyGeneratedClaimRequest with
    profileRequest := { claimProfileRequest with catalog := catalog }
    judgingClosure := judging
    cycleRegistry :=
      { importedDescriptors := [desc, rejected, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := desc, freezeIntegrity := some fi }
          , { descriptor := rejected, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem missing_disposition_rejected_by_claim_guard :
    (validateGeneratedClaim missingDispositionClaimRequest).hasFault
      (.profileCompletion (.catalog (.missingDisposition toyItemOther))) = true := by
  native_decide

/-- Validator category removed; dependencies rewired; identities rebound so the
primary fault is judging missingRequiredCategory.validator. -/
def missingValidatorCategoryClaimRequest : GeneratedClaimRequest :=
  let nodes0 :=
    claimJudgingNodes.filter (fun n => n.category != .validator)
  let nodes :=
    nodes0.map fun n =>
      let deps := n.dependencies.filter (fun d => d.value != "n.val")
      if n.id.value == "n.cal" then
        { n with dependencies := [{ value := "n.scn" }] }
      else { n with dependencies := deps }
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  let jid := computeJudgingClosureIdentity judging
  let records :=
    claimAncestryRecords.map fun r => { r with judgingClosureIdentity := jid }
  let desc :=
    { claimPromotedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let rejected :=
    { claimRejectedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          frozenJudgingClosureIdentity := jid
          records := records }
      suppliedSnapshots := records }
  { toyGeneratedClaimRequest with
    judgingClosure := judging
    cycleRegistry :=
      { importedDescriptors := [desc, rejected, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := desc, freezeIntegrity := some fi }
          , { descriptor := rejected, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem missing_validator_category_rejected_by_claim_guard :
    (validateGeneratedClaim missingValidatorCategoryClaimRequest).hasFault
      (.judgingClosure (.missingRequiredCategory .validator)) = true := by
  native_decide

/-- Middle ancestry commit drifts while freeze/candidate endpoints stay frozen. -/
def freezeIntervalDriftClaimRequest : GeneratedClaimRequest :=
  let records :=
    claimAncestryRecords.map fun r =>
      if r.commit == toyCommitB then
        { r with judgingClosureIdentity := digest "MUTATED-IN-CYCLE" }
      else r
  let fi :=
    { claimFreezeIntegrity with
      manifest := { claimFreezeIntegrity.manifest with records := records }
      suppliedSnapshots := records }
  { toyGeneratedClaimRequest with
    cycleRegistry :=
      { importedDescriptors :=
          [claimPromotedDescriptor, claimRejectedDescriptor, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := claimPromotedDescriptor, freezeIntegrity := some fi }
          , { descriptor := claimRejectedDescriptor, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem freeze_interval_judging_closure_drift_rejected_by_claim_guard :
    (validateGeneratedClaim freezeIntervalDriftClaimRequest).hasFault
      (.cycleRegistry
        (.promotedFreezeIntegrityFault claimPromotedDescriptor.id
          (.judgingClosureDrift toyCommitB))) = true := by
  native_decide

/-- Incomplete assurance policy (missing obligation classes) with consistent
rebinding — Lean ProfileAssurancePolicy.wellFormed must reject. -/
def incompletePolicyClaimRequest : GeneratedClaimRequest :=
  let badPolicy : ProfileAssurancePolicy :=
    { profileId := toyProfile
      version := 1
      rules :=
        [{ obligationClass := .requirementDischarge, rule := .requireProof }] }
  let catalog := { claimCatalogCandidate with assurancePolicy := badPolicy }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  let jid := computeJudgingClosureIdentity judging
  let records :=
    claimAncestryRecords.map fun r => { r with judgingClosureIdentity := jid }
  let desc :=
    { claimPromotedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let rejected :=
    { claimRejectedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          frozenJudgingClosureIdentity := jid
          records := records }
      suppliedSnapshots := records }
  { toyGeneratedClaimRequest with
    profileRequest := { claimProfileRequest with catalog := catalog }
    judgingClosure := judging
    cycleRegistry :=
      { importedDescriptors := [desc, rejected, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := desc, freezeIntegrity := some fi }
          , { descriptor := rejected, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem incomplete_assurance_policy_rejected_by_claim_guard :
    (validateGeneratedClaim incompletePolicyClaimRequest).hasFault
      (.profileCompletion (.catalog .malformedPolicy)) = true := by
  native_decide

/-- Empty required environments on the catalog entry (and matching discharge),
with frozen binding rebound — Lean RequirementClosureEntry.wellFormed rejects. -/
def emptyEnvironmentsClaimRequest : GeneratedClaimRequest :=
  let emptyEntry : RequirementClosureEntry :=
    { toyRequirementEntry with requiredEnvironments := [] }
  let emptyDischarge : RequirementDischargeCandidate :=
    { claimDischargeCandidate with environments := [] }
  let catalog :=
    { claimCatalogCandidate with requirementEntries := [emptyEntry] }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  let jid := computeJudgingClosureIdentity judging
  let records :=
    claimAncestryRecords.map fun r => { r with judgingClosureIdentity := jid }
  let desc :=
    { claimPromotedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let rejected :=
    { claimRejectedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          frozenJudgingClosureIdentity := jid
          records := records }
      suppliedSnapshots := records }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := catalog
        discharges := [emptyDischarge] }
    judgingClosure := judging
    cycleRegistry :=
      { importedDescriptors := [desc, rejected, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := desc, freezeIntegrity := some fi }
          , { descriptor := rejected, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem empty_requirement_environments_rejected_by_claim_guard :
    (validateGeneratedClaim emptyEnvironmentsClaimRequest).hasFault
      (.profileCompletion (.catalog .malformedRequirementEntry)) = true := by
  native_decide

/-- Policy + judging scope rebound together: validates; identity changes. -/
def reboundPolicyRequest : GeneratedClaimRequest :=
  let otherPolicy := uniformPolicy toyProfile 2 .requireProof
  let catalog := { claimCatalogCandidate with assurancePolicy := otherPolicy }
  let scope := digest (encodeFrozenCatalogBindingOf catalog)
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then { n with content := scope } else n
  let judging :=
    { claimJudgingCandidate with
      nodes := nodes
      discoveredInventory := nodes.map (·.id) }
  let jid := computeJudgingClosureIdentity judging
  let records :=
    claimAncestryRecords.map fun r => { r with judgingClosureIdentity := jid }
  let desc :=
    { claimPromotedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let rejected :=
    { claimRejectedDescriptor with expectedFrozenJudgingClosureIdentity := jid }
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          frozenJudgingClosureIdentity := jid
          records := records }
      suppliedSnapshots := records }
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with catalog := catalog }
    judgingClosure := judging
    cycleRegistry :=
      { importedDescriptors := [desc, rejected, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := desc, freezeIntegrity := some fi }
          , { descriptor := rejected, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem rebound_policy_changes_claim_identity :
    (match validateGeneratedClaim toyGeneratedClaimRequest,
           validateGeneratedClaim reboundPolicyRequest with
     | .ok a, .ok b => a.claimIdentity != b.claimIdentity
     | _, _ => false) = true := by
  native_decide

def changedProvenanceRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    fieldProvenance :=
      claimFieldProvenance.map fun f =>
        if f.name == "target" then
          { f with
            provenance :=
              .imported
                { id := { value := "import.upstream.closure" }
                  sourceClosure := { digest := digest "src-hash-MUTATED" } } }
        else f }

theorem changed_provenance_changes_claim_identity :
    (match validateGeneratedClaim toyGeneratedClaimRequest,
           validateGeneratedClaim changedProvenanceRequest with
     | .ok a, .ok b => a.claimIdentity != b.claimIdentity
     | _, _ => false) = true := by
  native_decide

def extraProvenanceRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    fieldProvenance :=
      claimFieldProvenance ++
        [{ name := "sneakyExtra", provenance := .imported sampleImported }] }

theorem extra_field_provenance_rejected :
    (validateGeneratedClaim extraProvenanceRequest).hasFault
      (.extraFieldProvenance "sneakyExtra") = true := by
  native_decide

def ancestryCaptureMutationRequest : GeneratedClaimRequest :=
  let fi :=
    { claimFreezeIntegrity with
      manifest :=
        { claimFreezeIntegrity.manifest with
          captureIdentity := { value := "capture.MUTATED" }
          extractorIdentity := digest "extractor-MUTATED" } }
  { toyGeneratedClaimRequest with
    cycleRegistry :=
      { importedDescriptors :=
          [claimPromotedDescriptor, claimRejectedDescriptor, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := claimPromotedDescriptor, freezeIntegrity := some fi }
          , { descriptor := claimRejectedDescriptor, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem ancestry_capture_mutation_changes_claim_identity :
    (match validateGeneratedClaim toyGeneratedClaimRequest,
           validateGeneratedClaim ancestryCaptureMutationRequest with
     | .ok a, .ok b => a.claimIdentity != b.claimIdentity
     | _, _ => false) = true := by
  native_decide

def selectedMissingEventRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    selectedPromotedEvent := { value := "cycle.absent.1" } }

theorem selected_missing_event_rejected :
    (validateGeneratedClaim selectedMissingEventRequest).hasFault
      .selectedEventMissing = true := by
  native_decide

def selectedRejectedEventRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    selectedPromotedEvent := claimRejectedDescriptor.id }

theorem selected_rejected_event_rejected :
    (validateGeneratedClaim selectedRejectedEventRequest).hasFault
      .selectedEventNotPromoted = true := by
  native_decide

def boundedSearchMislabeledRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        discharges :=
          [{ claimDischargeCandidate with
              assuranceState := .survivedSearch sampleSearch }] } }

theorem bounded_search_rejected_under_require_proof_policy :
    (validateGeneratedClaim boundedSearchMislabeledRequest).hasFault
      (.profileCompletion
        (.profile (.dischargeFault toyRequirementId .policyDoesNotAdmit))) = true := by
  native_decide

/-- Under a search-admitting policy, the renderer must not print proof. -/
def searchAdmittingPolicy : ProfileAssurancePolicy :=
  let rules :=
    allObligationClasses.map fun cls =>
      { obligationClass := cls
        rule :=
          if cls == .requirementDischarge then .acceptBoundedSearch
          else .requireProof }
  { profileId := toyProfile, version := 1, rules := rules }

def searchAdmittingCatalog : CatalogClosureCandidate :=
  { claimCatalogCandidate with assurancePolicy := searchAdmittingPolicy }

def searchAdmittingScope : ContentDigest :=
  digest (encodeFrozenCatalogBindingOf searchAdmittingCatalog)

def searchAdmittingJudging : JudgingInputClosureCandidate :=
  let nodes :=
    claimJudgingNodes.map fun n =>
      if n.category == .targetProfileDecision then
        { n with content := searchAdmittingScope }
      else n
  { claimJudgingCandidate with
    nodes := nodes
    discoveredInventory := nodes.map (·.id) }

def searchAdmittingJudgingIdentity : ContentDigest :=
  computeJudgingClosureIdentity searchAdmittingJudging

def searchAdmittingRecords : List AncestryCommitRecord :=
  claimAncestryRecords.map fun r =>
    { r with judgingClosureIdentity := searchAdmittingJudgingIdentity }

def searchAdmittingFreeze : FreezeIntegrityCandidate :=
  { claimFreezeIntegrity with
    manifest :=
      { claimFreezeIntegrity.manifest with
        frozenJudgingClosureIdentity := searchAdmittingJudgingIdentity
        records := searchAdmittingRecords }
    suppliedSnapshots := searchAdmittingRecords }

def searchAdmittingPromoted : CycleEventDescriptor :=
  { claimPromotedDescriptor with
    expectedFrozenJudgingClosureIdentity := searchAdmittingJudgingIdentity }

def searchAdmittingRejected : CycleEventDescriptor :=
  { claimRejectedDescriptor with
    expectedFrozenJudgingClosureIdentity := searchAdmittingJudgingIdentity }

def searchAdmittingRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := searchAdmittingCatalog
        discharges :=
          [{ claimDischargeCandidate with
              assuranceState := .survivedSearch sampleSearch }] }
    judgingClosure := searchAdmittingJudging
    cycleRegistry :=
      { importedDescriptors :=
          [searchAdmittingPromoted, searchAdmittingRejected, foreignRejectedDescriptor]
        entries :=
          [ { descriptor := searchAdmittingPromoted
              freezeIntegrity := some searchAdmittingFreeze }
          , { descriptor := searchAdmittingRejected, freezeIntegrity := none }
          , { descriptor := foreignRejectedDescriptor, freezeIntegrity := none } ] } }

theorem bounded_search_renders_as_search_not_proof :
    (match validateGeneratedClaim searchAdmittingRequest with
     | .ok v =>
         let report := renderGeneratedClaim v
         textHas report.text "survivedSearch" &&
         !(textHas report.text "proved(proof.adeq.demo") &&
         report.residualObligations.any (fun r => r.kind == .survivedSearch) &&
         report.syntheticComplete
     | .faults _ => false) = true := by
  native_decide

def performancePurposeClaimRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        discharges :=
          [{ claimDischargeCandidate with purpose := .performance }] } }

theorem performance_purpose_rejected_by_claim_guard :
    (validateGeneratedClaim performancePurposeClaimRequest).hasFault
      (.profileCompletion
        (.profile
          (.dischargeFault toyRequirementId .performancePurposeRejected))) = true := by
  native_decide

def hiddenBlockerDischargeRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        discharges :=
          [{ claimDischargeCandidate with
              assuranceState := .blocked sampleBlocker }] } }

theorem hidden_blocker_cannot_validate_as_complete :
    (validateGeneratedClaim hiddenBlockerDischargeRequest).hasFault
      (.profileCompletion
        (.profile (.dischargeFault toyRequirementId .policyDoesNotAdmit))) = true := by
  native_decide

theorem failure_report_lists_blocker_without_completion_language :
    (let report := renderGeneratedClaimAttempt hiddenBlockerDischargeRequest
     (!report.validationOk) &&
     (!report.syntheticComplete) &&
     textHas report.text "STATUS: incomplete" &&
     textHas report.text "completion language forbidden" &&
     !(textHas report.text "synthetic-complete") &&
     report.residualObligations.any (fun r => r.kind == .blocked) &&
     textHas report.text "blocked") = true := by
  native_decide

def emptyDenominatorClaimRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    profileRequest :=
      { claimProfileRequest with
        catalog := emptyDenominatorCatalog
        discharges := [] } }

theorem empty_denominator_rejected_by_claim_guard :
    (validateGeneratedClaim emptyDenominatorClaimRequest).hasFault
      (.profileCompletion (.catalog .emptyDenominator)) = true := by
  native_decide

def missingClaimRendererRequest : GeneratedClaimRequest :=
  let nodes :=
    claimJudgingCandidate.nodes.map fun n =>
      if n.category == .claimRenderer then
        { n with category := .toolchain }
      else n
  { toyGeneratedClaimRequest with
    judgingClosure :=
      { claimJudgingCandidate with
        nodes := nodes
        discoveredInventory := nodes.map (·.id) } }

theorem missing_claim_renderer_rejected :
    (validateGeneratedClaim missingClaimRendererRequest).hasFault
      (.judgingClosure (.missingRequiredCategory .claimRenderer)) = true := by
  native_decide

def missingProvenanceRequest : GeneratedClaimRequest :=
  { toyGeneratedClaimRequest with
    fieldProvenance :=
      claimFieldProvenance.filter (fun f => f.name != "runtime") }

theorem missing_field_provenance_rejected :
    (validateGeneratedClaim missingProvenanceRequest).hasFault
      (.missingFieldProvenance "runtime") = true := by
  native_decide

theorem renderer_lists_every_residual_search_obligation :
    (match validateGeneratedClaim searchAdmittingRequest with
     | .ok v =>
         let report := renderGeneratedClaim v
         report.residualObligations.length == 1 &&
         (report.residualObligations.head!).kind == .survivedSearch &&
         textHas report.text "residual_obligations:" &&
         textHas report.text "survivedSearch" &&
         textHas report.text "field_provenance:"
     | .faults _ => false) = true := by
  native_decide

/-- Mechanical handwritten-complete rejection lives in the Python guard.
Lean documents absence of an authored complete field by requiring validation
through premises only; a forged attempt report must not mark complete. -/
theorem attempt_report_never_marks_complete_on_faults :
    (let report := renderGeneratedClaimAttempt subjectCandidateMismatchRequest
     (!report.validationOk) && (!report.syntheticComplete) &&
     !(textHas report.text "synthetic-complete")) = true := by
  native_decide

end Tgrad.Contract

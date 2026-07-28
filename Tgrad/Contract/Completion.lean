import Tgrad.Contract.Identity
import Tgrad.Contract.Assurance

/-! # Tgrad.Contract.Completion — catalog, discharge, and profile candidates

Public candidates carry premises only. Validity is derived by typed validation
that returns diagnosed faults. Validated forms are constructed only inside this
module. This substrate does not claim that Tgrad is complete.
-/

namespace Tgrad.Contract

/-! ## Source dispositions (catalog denominator) -/

inductive SourceDisposition where
  | required (requirement : RequirementId)
  | excluded (judgment : JudgmentId)
  | ambiguous (resolveBy : String)
  | superseded (replacement : SourceItemId)
  deriving DecidableEq, BEq, Repr, Inhabited

def SourceDisposition.wellFormed : SourceDisposition → Bool
  | .required r => r.wellFormed
  | .excluded j => j.wellFormed
  | .ambiguous resolveBy => !resolveBy.trimAscii.isEmpty
  | .superseded r => r.wellFormed

def SourceDisposition.isUnresolved : SourceDisposition → Bool
  | .ambiguous _ => true
  | _ => false

structure SourceItemDisposition where
  item : SourceItemId
  disposition : SourceDisposition
  deriving DecidableEq, BEq, Repr, Inhabited

def SourceItemDisposition.wellFormed (row : SourceItemDisposition) : Bool :=
  row.item.wellFormed && row.disposition.wellFormed

/-! ## Requirement closure entries — full normative binding -/

/-- One required obligation with exact supporting identities and environments.
A scenario identity may name a whole generated manifest/family. -/
structure RequirementClosureEntry where
  requirement : RequirementIdentity
  specification : SpecIdentity
  boundary : BoundaryIdentity
  scenario : ScenarioIdentity
  relation : RelationIdentity
  adapter : AdapterIdentity
  validator : ValidatorIdentity
  calibration : CalibrationIdentity
  requiredEnvironments : List EnvironmentIdentity
  deriving DecidableEq, BEq, Repr, Inhabited

def RequirementClosureEntry.wellFormed (e : RequirementClosureEntry) : Bool :=
  e.requirement.wellFormed &&
  e.specification.wellFormed &&
  e.boundary.wellFormed &&
  e.scenario.wellFormed &&
  e.relation.wellFormed &&
  e.adapter.wellFormed &&
  e.validator.wellFormed &&
  e.calibration.wellFormed &&
  !e.requiredEnvironments.isEmpty &&
  e.requiredEnvironments.all EnvironmentIdentity.wellFormed &&
  listNodup (e.requiredEnvironments.map (·.id))

/-! ## Public candidates — premises only, no pass/valid/complete Booleans -/

structure CatalogClosureCandidate where
  target : TargetIdentity
  profile : ProfileId
  sourceClosure : SourceClosureId
  inventory : List SourceItemId
  dispositions : List SourceItemDisposition
  /-- Exact required closures; requirement IDs must be unique. -/
  requirementEntries : List RequirementClosureEntry
  /-- Single bound assurance policy for every requirement discharge. -/
  assurancePolicy : ProfileAssurancePolicy
  /-- Accepted exclusion judgments; must exactly match referenced exclusions. -/
  acceptedJudgments : List JudgmentIdentity
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Discharge premises. Policy and obligation class are not authorable here:
validators derive the catalog-bound policy and always evaluate the fixed
`requirementDischarge` class. -/
structure RequirementDischargeCandidate where
  requirement : RequirementIdentity
  specification : SpecIdentity
  boundary : BoundaryIdentity
  scenario : ScenarioIdentity
  relation : RelationIdentity
  adapter : AdapterIdentity
  validator : ValidatorIdentity
  calibration : CalibrationIdentity
  target : TargetIdentity
  profile : ProfileId
  subjectTree : ProductTreeIdentity
  runtime : RuntimeIdentity
  environments : List EnvironmentIdentity
  evidence : List EvidenceIdentity
  /-- Purpose axis, distinct from obligation-class policy. -/
  purpose : CertificatePurpose
  assuranceState : AssuranceState
  deriving DecidableEq, BEq, Repr, Inhabited

/-! ## Validated (private construction) forms -/

structure ValidatedCatalogClosure where
  private mk ::
    candidate : CatalogClosureCandidate
  deriving Repr

/-- Catalog-contextual discharge certificate. Policy is retained from the
validated catalog for reporting; it is never authored by the candidate. -/
structure ValidatedRequirementDischarge where
  private mk ::
    candidate : RequirementDischargeCandidate
    assurancePolicy : ProfileAssurancePolicy
  deriving Repr

structure ProfileCompletionCandidate where
  catalog : ValidatedCatalogClosure
  subjectTree : ProductTreeIdentity
  runtime : RuntimeIdentity
  discharges : List RequirementDischargeCandidate
  deriving Repr

structure ValidatedProfileCompletion where
  private mk ::
    candidate : ProfileCompletionCandidate
  deriving Repr

structure ProfileCompletionRequest where
  catalog : CatalogClosureCandidate
  subjectTree : ProductTreeIdentity
  runtime : RuntimeIdentity
  discharges : List RequirementDischargeCandidate
  deriving Repr

def ValidatedCatalogClosure.requirementEntries
    (v : ValidatedCatalogClosure) : List RequirementClosureEntry :=
  v.candidate.requirementEntries

def ValidatedCatalogClosure.requiredRequirementIds
    (v : ValidatedCatalogClosure) : List RequirementId :=
  v.candidate.requirementEntries.map (·.requirement.id)

def ValidatedCatalogClosure.assurancePolicy
    (v : ValidatedCatalogClosure) : ProfileAssurancePolicy :=
  v.candidate.assurancePolicy

def ValidatedCatalogClosure.target (v : ValidatedCatalogClosure) : TargetIdentity :=
  v.candidate.target

def ValidatedCatalogClosure.profile (v : ValidatedCatalogClosure) : ProfileId :=
  v.candidate.profile

def ValidatedCatalogClosure.sourceClosure
    (v : ValidatedCatalogClosure) : SourceClosureId :=
  v.candidate.sourceClosure

def ValidatedCatalogClosure.inventory
    (v : ValidatedCatalogClosure) : List SourceItemId :=
  v.candidate.inventory

def ValidatedCatalogClosure.dispositions
    (v : ValidatedCatalogClosure) : List SourceItemDisposition :=
  v.candidate.dispositions

def ValidatedCatalogClosure.acceptedJudgments
    (v : ValidatedCatalogClosure) : List JudgmentIdentity :=
  v.candidate.acceptedJudgments

def ValidatedCatalogClosure.lookupEntry?
    (v : ValidatedCatalogClosure) (id : RequirementId) : Option RequirementClosureEntry :=
  v.candidate.requirementEntries.find? (fun e => e.requirement.id == id)

def ValidatedRequirementDischarge.requirement
    (v : ValidatedRequirementDischarge) : RequirementId :=
  v.candidate.requirement.id

def ValidatedRequirementDischarge.assuranceState
    (v : ValidatedRequirementDischarge) : AssuranceState :=
  v.candidate.assuranceState

def ValidatedProfileCompletion.catalog
    (v : ValidatedProfileCompletion) : ValidatedCatalogClosure :=
  v.candidate.catalog

def ValidatedProfileCompletion.subjectTree
    (v : ValidatedProfileCompletion) : ProductTreeIdentity :=
  v.candidate.subjectTree

def ValidatedProfileCompletion.runtime
    (v : ValidatedProfileCompletion) : RuntimeIdentity :=
  v.candidate.runtime

def ValidatedProfileCompletion.discharges
    (v : ValidatedProfileCompletion) : List RequirementDischargeCandidate :=
  v.candidate.discharges

/-! ## Typed diagnosed faults -/

inductive CatalogFault where
  | malformedTarget
  | malformedProfile
  | malformedSourceClosure
  | unpromotedTarget
  | emptyDenominator
  | malformedRequirementEntry
  | duplicateRequirement (id : RequirementId)
  | duplicateInventoryItem (id : SourceItemId)
  | missingDisposition (item : SourceItemId)
  | extraDisposition (item : SourceItemId)
  | duplicateDisposition (item : SourceItemId)
  | unresolvedApplicable (item : SourceItemId)
  | mismatchedTargetProfile
  | mismatchedSourceClosure
  | malformedPolicy
  | mismatchedPolicyProfile
  | requiredNotInDenominator (id : RequirementId)
  | requiredDispositionOmittedFromDenominator (id : RequirementId)
  | malformedDisposition
  | malformedJudgment
  | duplicateJudgment (id : JudgmentId)
  | unacceptedExclusion (id : JudgmentId)
  | unreferencedJudgment (id : JudgmentId)
  | supersededNotInInventory (item : SourceItemId) (replacement : SourceItemId)
  | supersededSelf (item : SourceItemId)
  | supersededCycle (witness : SourceItemId)
  deriving DecidableEq, BEq, Repr, Inhabited

inductive DischargeFault where
  | malformedRequirement
  | malformedSpecification
  | malformedBoundary
  | malformedScenario
  | malformedRelation
  | malformedAdapter
  | malformedValidator
  | malformedCalibration
  | malformedTarget
  | malformedProfile
  | malformedSubjectTree
  | dirtySubjectTree
  | malformedRuntime
  | dirtyRuntimeTree
  | emptyEnvironments
  | emptyEvidence
  | malformedEnvironment
  | duplicateEnvironment
  | malformedEvidence
  | duplicateEvidence
  | unpromotedTarget
  | mismatchedTargetProfile
  | runtimeTreeMismatch
  | performancePurposeRejected
  | assuranceNotWellFormed
  | unknownRequirement
  | mismatchedCatalogTarget
  | mismatchedCatalogProfile
  | mismatchedRequirementIdentity
  | mismatchedSpecification
  | mismatchedBoundary
  | mismatchedScenario
  | mismatchedRelation
  | mismatchedAdapter
  | mismatchedValidator
  | mismatchedCalibration
  | mismatchedEnvironments
  | policyDoesNotAdmit
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ProfileFault where
  | malformedSubjectTree
  | dirtySubjectTree
  | malformedRuntime
  | dirtyRuntimeTree
  | runtimeTreeMismatch
  | emptyDischarges
  | missingDischarge (id : RequirementId)
  | extraDischarge (id : RequirementId)
  | duplicateDischarge (id : RequirementId)
  | mismatchedTarget
  | mismatchedProfile
  | mismatchedSubject
  | mismatchedRuntime
  | dischargeFault (id : RequirementId) (fault : DischargeFault)
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ProfileCompletionFault where
  | catalog (fault : CatalogFault)
  | profile (fault : ProfileFault)
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ValidationResult (fault : Type) (ok : Type) where
  | ok (value : ok)
  | faults (diagnosed : List fault)
  deriving Repr

def ValidationResult.isOk {fault ok : Type} : ValidationResult fault ok → Bool
  | .ok _ => true
  | .faults _ => false

def ValidationResult.hasFault {fault ok : Type} [BEq fault]
    (result : ValidationResult fault ok) (f : fault) : Bool :=
  match result with
  | .ok _ => false
  | .faults diagnosed => diagnosed.contains f

/-! ## Supersession graph -/

private def supersededEdges
    (dispositions : List SourceItemDisposition) : List (SourceItemId × SourceItemId) :=
  dispositions.filterMap fun row =>
    match row.disposition with
    | .superseded replacement => some (row.item, replacement)
    | _ => none

/-- Path-sensitive cycle check from `start` through supersession edges. -/
private def supersededCycleFrom
    (edges : List (SourceItemId × SourceItemId)) (start : SourceItemId) : Bool :=
  let rec go (current : SourceItemId) (path : List SourceItemId) (fuel : Nat) : Bool :=
    match fuel with
    | 0 => false
    | fuel + 1 =>
      if path.contains current then true
      else
        let successors :=
          edges.filterMap fun (src, dst) => if src == current then some dst else none
        successors.any fun next => go next (current :: path) fuel
  go start [] (edges.length + 1)

private def supersededCycleWitnesses
    (edges : List (SourceItemId × SourceItemId)) : List SourceItemId :=
  let starts := edges.map (·.1)
  starts.filterMap fun start =>
    if supersededCycleFrom edges start then some start else none

/-! ## Catalog validation -/

private def catalogDiagnose (c : CatalogClosureCandidate) : List CatalogFault :=
  let push (faults : List CatalogFault) (cond : Bool) (f : CatalogFault) : List CatalogFault :=
    if cond then faults ++ [f] else faults
  let faults : List CatalogFault := []
  let faults := push faults (!c.target.wellFormed) .malformedTarget
  let faults := push faults (!c.profile.wellFormed) .malformedProfile
  let faults := push faults (!c.sourceClosure.wellFormed) .malformedSourceClosure
  let faults := push faults (!c.target.isPromoted) .unpromotedTarget
  let faults := push faults (c.target.profile != c.profile) .mismatchedTargetProfile
  let faults := push faults (c.target.sourceClosure != c.sourceClosure) .mismatchedSourceClosure
  let faults := push faults (!c.assurancePolicy.wellFormed) .malformedPolicy
  let faults :=
    push faults (c.assurancePolicy.profileId != c.profile) .mismatchedPolicyProfile
  let faults :=
    push faults (c.inventory.isEmpty || c.requirementEntries.isEmpty) .emptyDenominator
  let faults :=
    push faults (!c.requirementEntries.all RequirementClosureEntry.wellFormed)
      .malformedRequirementEntry
  let requiredIds := c.requirementEntries.map (·.requirement.id)
  let faults :=
    if listNodup requiredIds then faults
    else
      requiredIds.foldl (fun acc id =>
        if requiredIds.count id > 1 && !(acc.contains (.duplicateRequirement id)) then
          acc ++ [.duplicateRequirement id]
        else acc) faults
  let faults :=
    if listNodup c.inventory then faults
    else
      c.inventory.foldl (fun acc id =>
        if c.inventory.count id > 1 && !(acc.contains (.duplicateInventoryItem id)) then
          acc ++ [.duplicateInventoryItem id]
        else acc) faults
  let faults :=
    push faults (!c.dispositions.all SourceItemDisposition.wellFormed) .malformedDisposition
  let faults :=
    push faults (!c.acceptedJudgments.all JudgmentIdentity.wellFormed) .malformedJudgment
  let acceptedIds := c.acceptedJudgments.map (·.id)
  let faults :=
    if listNodup acceptedIds then faults
    else
      acceptedIds.foldl (fun acc id =>
        if acceptedIds.count id > 1 && !(acc.contains (.duplicateJudgment id)) then
          acc ++ [.duplicateJudgment id]
        else acc) faults
  let dispositionItems := c.dispositions.map (·.item)
  let faults :=
    if listNodup dispositionItems then faults
    else
      dispositionItems.foldl (fun acc id =>
        if dispositionItems.count id > 1 && !(acc.contains (.duplicateDisposition id)) then
          acc ++ [.duplicateDisposition id]
        else acc) faults
  let faults :=
    c.inventory.foldl (fun acc item =>
      if dispositionItems.contains item then acc
      else acc ++ [.missingDisposition item]) faults
  let faults :=
    dispositionItems.foldl (fun acc item =>
      if c.inventory.contains item then acc
      else acc ++ [.extraDisposition item]) faults
  let faults :=
    c.dispositions.foldl (fun acc row =>
      if row.disposition.isUnresolved then acc ++ [.unresolvedApplicable row.item]
      else acc) faults
  let faults :=
    c.dispositions.foldl (fun acc row =>
      match row.disposition with
      | .superseded replacement =>
          let acc :=
            if !(c.inventory.contains replacement) then
              acc ++ [.supersededNotInInventory row.item replacement]
            else acc
          if replacement == row.item then acc ++ [.supersededSelf row.item] else acc
      | _ => acc) faults
  let edges := supersededEdges c.dispositions
  let faults :=
    (supersededCycleWitnesses edges).foldl (fun acc witness =>
      if acc.contains (.supersededCycle witness) then acc
      else acc ++ [.supersededCycle witness]) faults
  let requiredFromDisp :=
    c.dispositions.filterMap fun row =>
      match row.disposition with
      | .required id => some id
      | _ => none
  let faults :=
    requiredIds.foldl (fun acc id =>
      if requiredFromDisp.contains id then acc
      else acc ++ [.requiredNotInDenominator id]) faults
  let faults :=
    requiredFromDisp.foldl (fun acc id =>
      if requiredIds.contains id then acc
      else acc ++ [.requiredDispositionOmittedFromDenominator id]) faults
  let exclusionIds :=
    c.dispositions.filterMap fun row =>
      match row.disposition with
      | .excluded id => some id
      | _ => none
  let faults :=
    exclusionIds.foldl (fun acc id =>
      if acceptedIds.contains id then acc
      else acc ++ [.unacceptedExclusion id]) faults
  acceptedIds.foldl (fun acc id =>
    if exclusionIds.contains id then acc
    else acc ++ [.unreferencedJudgment id]) faults

def validateCatalogClosure (c : CatalogClosureCandidate) :
    ValidationResult CatalogFault ValidatedCatalogClosure :=
  match catalogDiagnose c with
  | [] => .ok ⟨c⟩
  | diagnosed => .faults diagnosed

/-! ## Discharge validation

Premise/structural checks never mint `ValidatedRequirementDischarge`.
Successful validation requires a `ValidatedCatalogClosure` and matches the
catalog target/profile, exact assurance policy, and complete closure entry.
-/

/-- Candidate-only structural/premise diagnosis. Does not return a validated
discharge and does not consult catalog policy or closure entries. -/
def diagnoseRequirementDischargePremises
    (c : RequirementDischargeCandidate) : List DischargeFault :=
  let push (faults : List DischargeFault) (cond : Bool) (f : DischargeFault) : List DischargeFault :=
    if cond then faults ++ [f] else faults
  let faults : List DischargeFault := []
  let faults := push faults (!c.requirement.wellFormed) .malformedRequirement
  let faults := push faults (!c.specification.wellFormed) .malformedSpecification
  let faults := push faults (!c.boundary.wellFormed) .malformedBoundary
  let faults := push faults (!c.scenario.wellFormed) .malformedScenario
  let faults := push faults (!c.relation.wellFormed) .malformedRelation
  let faults := push faults (!c.adapter.wellFormed) .malformedAdapter
  let faults := push faults (!c.validator.wellFormed) .malformedValidator
  let faults := push faults (!c.calibration.wellFormed) .malformedCalibration
  let faults := push faults (!c.target.wellFormed) .malformedTarget
  let faults := push faults (!c.profile.wellFormed) .malformedProfile
  let faults := push faults (!c.subjectTree.wellFormed) .malformedSubjectTree
  let faults := push faults (!c.subjectTree.isClean) .dirtySubjectTree
  let faults := push faults (!c.runtime.wellFormed) .malformedRuntime
  let faults := push faults (!c.runtime.sourceTree.isClean) .dirtyRuntimeTree
  let faults := push faults c.environments.isEmpty .emptyEnvironments
  let faults := push faults c.evidence.isEmpty .emptyEvidence
  let faults :=
    push faults (!c.environments.all EnvironmentIdentity.wellFormed) .malformedEnvironment
  let faults := push faults (!c.evidence.all EvidenceIdentity.wellFormed) .malformedEvidence
  let faults :=
    push faults (!listNodup (c.environments.map (·.id))) .duplicateEnvironment
  let faults := push faults (!listNodup (c.evidence.map (·.id))) .duplicateEvidence
  let faults := push faults (!c.target.isPromoted) .unpromotedTarget
  let faults := push faults (c.target.profile != c.profile) .mismatchedTargetProfile
  let faults := push faults (c.runtime.sourceTree != c.subjectTree.id) .runtimeTreeMismatch
  let faults :=
    push faults (c.purpose != .semanticCompatibility) .performancePurposeRejected
  push faults (!c.assuranceState.wellFormed) .assuranceNotWellFormed

def RequirementDischargeCandidate.premisesWellFormed
    (c : RequirementDischargeCandidate) : Bool :=
  (diagnoseRequirementDischargePremises c).isEmpty

private def dischargeDiagnose
    (catalog : ValidatedCatalogClosure) (c : RequirementDischargeCandidate) :
    List DischargeFault :=
  let push (faults : List DischargeFault) (cond : Bool) (f : DischargeFault) : List DischargeFault :=
    if cond then faults ++ [f] else faults
  let faults := diagnoseRequirementDischargePremises c
  let faults := push faults (c.target != catalog.target) .mismatchedCatalogTarget
  let faults := push faults (c.profile != catalog.profile) .mismatchedCatalogProfile
  let policy := catalog.assurancePolicy
  match catalog.lookupEntry? c.requirement.id with
  | none => faults ++ [.unknownRequirement]
  | some entry =>
      let faults :=
        push faults (c.requirement != entry.requirement) .mismatchedRequirementIdentity
      let faults :=
        push faults (c.specification != entry.specification) .mismatchedSpecification
      let faults := push faults (c.boundary != entry.boundary) .mismatchedBoundary
      let faults := push faults (c.scenario != entry.scenario) .mismatchedScenario
      let faults := push faults (c.relation != entry.relation) .mismatchedRelation
      let faults := push faults (c.adapter != entry.adapter) .mismatchedAdapter
      let faults := push faults (c.validator != entry.validator) .mismatchedValidator
      let faults :=
        push faults (c.calibration != entry.calibration) .mismatchedCalibration
      let faults :=
        push faults (!listSetEq c.environments entry.requiredEnvironments)
          .mismatchedEnvironments
      -- Fixed obligation class against the catalog-bound policy only.
      push faults (!(policy.meets .requirementDischarge c.assuranceState))
        .policyDoesNotAdmit

/-- Catalog-contextual discharge validation. This is the only path that may
construct `ValidatedRequirementDischarge`. -/
def validateRequirementDischarge
    (catalog : ValidatedCatalogClosure) (c : RequirementDischargeCandidate) :
    ValidationResult DischargeFault ValidatedRequirementDischarge :=
  match dischargeDiagnose catalog c with
  | [] => .ok ⟨c, catalog.assurancePolicy⟩
  | diagnosed => .faults diagnosed

/-! ## Profile completion validation -/

private def profileDiagnose (c : ProfileCompletionCandidate) : List ProfileFault :=
  let push (faults : List ProfileFault) (cond : Bool) (f : ProfileFault) : List ProfileFault :=
    if cond then faults ++ [f] else faults
  let faults : List ProfileFault := []
  let faults := push faults (!c.subjectTree.wellFormed) .malformedSubjectTree
  let faults := push faults (!c.subjectTree.isClean) .dirtySubjectTree
  let faults := push faults (!c.runtime.wellFormed) .malformedRuntime
  let faults := push faults (!c.runtime.sourceTree.isClean) .dirtyRuntimeTree
  let faults := push faults (c.runtime.sourceTree != c.subjectTree.id) .runtimeTreeMismatch
  let faults := push faults c.discharges.isEmpty .emptyDischarges
  let requiredIds := c.catalog.requiredRequirementIds
  let dischargedIds := c.discharges.map (·.requirement.id)
  let faults :=
    if listNodup dischargedIds then faults
    else
      dischargedIds.foldl (fun acc id =>
        if dischargedIds.count id > 1 && !(acc.contains (.duplicateDischarge id)) then
          acc ++ [.duplicateDischarge id]
        else acc) faults
  let faults :=
    requiredIds.foldl (fun acc id =>
      if dischargedIds.contains id then acc else acc ++ [.missingDischarge id]) faults
  let faults :=
    dischargedIds.foldl (fun acc id =>
      if requiredIds.contains id then acc else acc ++ [.extraDischarge id]) faults
  let faults :=
    c.discharges.foldl (fun acc d =>
      let acc :=
        if d.target != c.catalog.target && !(acc.contains .mismatchedTarget) then
          acc ++ [.mismatchedTarget]
        else acc
      let acc :=
        if d.profile != c.catalog.profile && !(acc.contains .mismatchedProfile) then
          acc ++ [.mismatchedProfile]
        else acc
      let acc :=
        if d.subjectTree != c.subjectTree && !(acc.contains .mismatchedSubject) then
          acc ++ [.mismatchedSubject]
        else acc
      if d.runtime != c.runtime && !(acc.contains .mismatchedRuntime) then
        acc ++ [.mismatchedRuntime]
      else acc) faults
  -- Same contextual discharge validator as the public API; no semantic drift.
  c.discharges.foldl (fun acc d =>
    match validateRequirementDischarge c.catalog d with
    | .ok _ => acc
    | .faults dfs =>
        dfs.foldl (fun acc f => acc ++ [.dischargeFault d.requirement.id f]) acc) faults

def validateProfileCompletion (c : ProfileCompletionCandidate) :
    ValidationResult ProfileFault ValidatedProfileCompletion :=
  match profileDiagnose c with
  | [] => .ok ⟨c⟩
  | diagnosed => .faults diagnosed

def validateProfileFromCatalog (req : ProfileCompletionRequest) :
    ValidationResult ProfileCompletionFault ValidatedProfileCompletion :=
  match validateCatalogClosure req.catalog with
  | .faults fs => .faults (fs.map ProfileCompletionFault.catalog)
  | .ok catalog =>
      match validateProfileCompletion
          { catalog := catalog,
            subjectTree := req.subjectTree,
            runtime := req.runtime,
            discharges := req.discharges } with
      | .ok v => .ok v
      | .faults fs => .faults (fs.map ProfileCompletionFault.profile)

/-! ## Toy fixtures and executable checks -/

def toyProfile : ProfileId := { value := "profile.toy.v1" }
def toySourceClosure : SourceClosureId := { digest := digest "toy-source-closure" }
def toyTarget : TargetIdentity where
  id := { value := "target.toy" }
  repository := "github.com/tinygrad/tinygrad"
  revision := digest "19c4d736"
  sourceClosure := toySourceClosure
  profile := toyProfile
  disposition := .promoted

def toyRequirementId : RequirementId := { value := "REQ-TOY-IMPORT" }
def toyRequirement : RequirementIdentity where
  id := toyRequirementId
  semantics := digest "req-sem"
def toyItem : SourceItemId := { value := "src.helpers" }
def toyItemOther : SourceItemId := { value := "src.other" }
def toyItemThird : SourceItemId := { value := "src.third" }

def toySpec : SpecIdentity where
  id := { value := "SPEC-TOY" }
  semantics := digest "spec-sem"

def toyBoundary : BoundaryIdentity where
  id := { value := "BND-TOY" }
  semantics := digest "bnd-sem"

def toyScenario : ScenarioIdentity where
  id := { value := "SCN-TOY" }
  digest := digest "scn-hash"

def toyRelation : RelationIdentity where
  id := { value := "REL-TOY" }
  digest := digest "rel-hash"

def toyAdapter : AdapterIdentity where
  id := { value := "ADP-TOY" }
  digest := digest "adp-hash"

def toyValidator : ValidatorIdentity where
  id := { value := "VAL-TOY" }
  version := digest "val-v1"

def toyCalibration : CalibrationIdentity where
  id := { value := "CAL-TOY" }
  campaign := digest "cal-hash"
  faultModel := "missing-public-name"

def toyEnv : EnvironmentIdentity where
  id := { value := "env.macos-metal" }
  digest := digest "env-hash"

def toyEnvOther : EnvironmentIdentity where
  id := { value := "env.cpu-only" }
  digest := digest "env-cpu-hash"

def toyEvidence : EvidenceIdentity where
  id := { value := "ev.toy.1" }
  digest := digest "ev-hash"

def toyAssurancePolicy : ProfileAssurancePolicy :=
  uniformPolicy toyProfile 1 .requireProof

def toyRequirementEntry : RequirementClosureEntry where
  requirement := toyRequirement
  specification := toySpec
  boundary := toyBoundary
  scenario := toyScenario
  relation := toyRelation
  adapter := toyAdapter
  validator := toyValidator
  calibration := toyCalibration
  requiredEnvironments := [toyEnv]

def toyCatalogCandidate : CatalogClosureCandidate where
  target := toyTarget
  profile := toyProfile
  sourceClosure := toySourceClosure
  inventory := [toyItem]
  dispositions :=
    [{ item := toyItem, disposition := .required toyRequirementId }]
  requirementEntries := [toyRequirementEntry]
  assurancePolicy := toyAssurancePolicy
  acceptedJudgments := []

def toySubject : ProductTreeIdentity where
  id := { revision := "subj-rev", content := digest "subj-tree", dirty := false }

def toyRuntime : RuntimeIdentity where
  id := { artifact := digest "runtime-dylib" }
  sourceTree := toySubject.id

def toyDischargeCandidate : RequirementDischargeCandidate where
  requirement := toyRequirement
  specification := toySpec
  boundary := toyBoundary
  scenario := toyScenario
  relation := toyRelation
  adapter := toyAdapter
  validator := toyValidator
  calibration := toyCalibration
  target := toyTarget
  profile := toyProfile
  subjectTree := toySubject
  runtime := toyRuntime
  environments := [toyEnv]
  evidence := [toyEvidence]
  purpose := .semanticCompatibility
  assuranceState := .proved sampleProof

def toyProfileRequest : ProfileCompletionRequest where
  catalog := toyCatalogCandidate
  subjectTree := toySubject
  runtime := toyRuntime
  discharges := [toyDischargeCandidate]

theorem toy_catalog_validates :
    (validateCatalogClosure toyCatalogCandidate).isOk = true := by
  native_decide

theorem toy_contextual_discharge_validates :
    (match validateCatalogClosure toyCatalogCandidate with
     | .ok catalog => (validateRequirementDischarge catalog toyDischargeCandidate).isOk
     | .faults _ => false) = true := by
  native_decide

theorem toy_discharge_premises_are_structurally_well_formed :
    toyDischargeCandidate.premisesWellFormed = true := by
  native_decide

theorem toy_profile_validates :
    (validateProfileFromCatalog toyProfileRequest).isOk = true := by
  native_decide

theorem dirty_tree_is_structurally_well_formed :
    let dirty : ProductTreeId :=
      { revision := "subj-rev", content := digest "subj-tree", dirty := true }
    dirty.wellFormed = true && dirty.isClean = false && dirty.isEligible = false := by
  native_decide

def unpromotedCatalog : CatalogClosureCandidate :=
  { toyCatalogCandidate with
    target := { toyTarget with disposition := .extractedCandidate } }

theorem unpromoted_target_rejected :
    (validateCatalogClosure unpromotedCatalog).hasFault .unpromotedTarget = true := by
  native_decide

def emptyDenominatorCatalog : CatalogClosureCandidate :=
  { toyCatalogCandidate with
    inventory := []
    requirementEntries := []
    dispositions := [] }

theorem empty_denominator_rejected :
    (validateCatalogClosure emptyDenominatorCatalog).hasFault .emptyDenominator = true := by
  native_decide

def duplicateRequirementCatalog : CatalogClosureCandidate :=
  { toyCatalogCandidate with
    requirementEntries :=
      [toyRequirementEntry,
       { toyRequirementEntry with
         requirement := { toyRequirement with semantics := digest "other-sem" } }] }

theorem duplicate_requirement_rejected :
    (validateCatalogClosure duplicateRequirementCatalog).hasFault
      (.duplicateRequirement toyRequirementId) = true := by
  native_decide

def omittedDenominatorCatalog : CatalogClosureCandidate where
  target := toyTarget
  profile := toyProfile
  sourceClosure := toySourceClosure
  inventory := [toyItem, toyItemOther]
  dispositions :=
    [ { item := toyItem, disposition := .required toyRequirementId },
      { item := toyItemOther,
        disposition := .required { value := "REQ-OMITTED" } } ]
  requirementEntries := [toyRequirementEntry]
  assurancePolicy := toyAssurancePolicy
  acceptedJudgments := []

theorem required_disposition_omitted_from_denominator_rejected :
    (validateCatalogClosure omittedDenominatorCatalog).hasFault
      (.requiredDispositionOmittedFromDenominator { value := "REQ-OMITTED" }) = true := by
  native_decide

def missingDispositionCatalog : CatalogClosureCandidate :=
  { toyCatalogCandidate with dispositions := [] }

theorem missing_disposition_rejected :
    (validateCatalogClosure missingDispositionCatalog).hasFault
      (.missingDisposition toyItem) = true := by
  native_decide

def unresolvedCatalog : CatalogClosureCandidate :=
  { toyCatalogCandidate with
    dispositions :=
      [{ item := toyItem, disposition := .ambiguous "GAP-TOY" }] }

theorem unresolved_applicable_rejected :
    (validateCatalogClosure unresolvedCatalog).hasFault
      (.unresolvedApplicable toyItem) = true := by
  native_decide

def mismatchedProfileCatalog : CatalogClosureCandidate :=
  { toyCatalogCandidate with profile := { value := "profile.other" } }

theorem mismatched_target_profile_rejected :
    (validateCatalogClosure mismatchedProfileCatalog).hasFault
      .mismatchedTargetProfile = true := by
  native_decide

def toyExclusionJudgment : JudgmentIdentity :=
  { id := { value := "judgment.exclude.demo" },
    authority := "owner",
    scope := digest "excl-scope",
    invalidation := digest "excl-inv" }

def unacceptedExclusionCatalog : CatalogClosureCandidate where
  target := toyTarget
  profile := toyProfile
  sourceClosure := toySourceClosure
  inventory := [toyItem, toyItemOther]
  dispositions :=
    [ { item := toyItem, disposition := .required toyRequirementId },
      { item := toyItemOther,
        disposition := .excluded toyExclusionJudgment.id } ]
  requirementEntries := [toyRequirementEntry]
  assurancePolicy := toyAssurancePolicy
  acceptedJudgments := []

theorem unaccepted_exclusion_rejected :
    (validateCatalogClosure unacceptedExclusionCatalog).hasFault
      (.unacceptedExclusion toyExclusionJudgment.id) = true := by
  native_decide

def unreferencedJudgmentCatalog : CatalogClosureCandidate :=
  { toyCatalogCandidate with acceptedJudgments := [toyExclusionJudgment] }

theorem unreferenced_judgment_rejected :
    (validateCatalogClosure unreferencedJudgmentCatalog).hasFault
      (.unreferencedJudgment toyExclusionJudgment.id) = true := by
  native_decide

def supersededSelfCatalog : CatalogClosureCandidate where
  target := toyTarget
  profile := toyProfile
  sourceClosure := toySourceClosure
  inventory := [toyItem, toyItemOther]
  dispositions :=
    [ { item := toyItem, disposition := .required toyRequirementId },
      { item := toyItemOther, disposition := .superseded toyItemOther } ]
  requirementEntries := [toyRequirementEntry]
  assurancePolicy := toyAssurancePolicy
  acceptedJudgments := []

theorem superseded_self_rejected :
    (validateCatalogClosure supersededSelfCatalog).hasFault
      (.supersededSelf toyItemOther) = true := by
  native_decide

def supersededMissingCatalog : CatalogClosureCandidate where
  target := toyTarget
  profile := toyProfile
  sourceClosure := toySourceClosure
  inventory := [toyItem, toyItemOther]
  dispositions :=
    [ { item := toyItem, disposition := .required toyRequirementId },
      { item := toyItemOther,
        disposition := .superseded { value := "src.missing" } } ]
  requirementEntries := [toyRequirementEntry]
  assurancePolicy := toyAssurancePolicy
  acceptedJudgments := []

theorem superseded_missing_rejected :
    (validateCatalogClosure supersededMissingCatalog).hasFault
      (.supersededNotInInventory toyItemOther { value := "src.missing" }) = true := by
  native_decide

/-- Two-cycle supersession has no normative endpoint. -/
def supersededCycleCatalog : CatalogClosureCandidate where
  target := toyTarget
  profile := toyProfile
  sourceClosure := toySourceClosure
  inventory := [toyItem, toyItemOther, toyItemThird]
  dispositions :=
    [ { item := toyItem, disposition := .required toyRequirementId },
      { item := toyItemOther, disposition := .superseded toyItemThird },
      { item := toyItemThird, disposition := .superseded toyItemOther } ]
  requirementEntries := [toyRequirementEntry]
  assurancePolicy := toyAssurancePolicy
  acceptedJudgments := []

theorem superseded_cycle_rejected :
    (validateCatalogClosure supersededCycleCatalog).hasFault
      (.supersededCycle toyItemOther) = true ∨
    (validateCatalogClosure supersededCycleCatalog).hasFault
      (.supersededCycle toyItemThird) = true := by
  native_decide

def performancePurposeDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with purpose := .performance }

theorem performance_purpose_rejected_for_requirement_discharge :
    (match validateCatalogClosure toyCatalogCandidate with
     | .ok catalog =>
         (validateRequirementDischarge catalog performancePurposeDischarge).hasFault
           .performancePurposeRejected
     | .faults _ => false) = true := by
  native_decide

/-- Search that would pass under a permissive same-profile policy is rejected
when validated against the catalog-bound requireProof policy. -/
def permissiveSearchDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with assuranceState := .survivedSearch sampleSearch }

def permissiveSearchRequest : ProfileCompletionRequest :=
  { toyProfileRequest with discharges := [permissiveSearchDischarge] }

theorem permissive_same_profile_search_rejected_by_contextual_discharge :
    (match validateCatalogClosure toyCatalogCandidate with
     | .ok catalog =>
         (validateRequirementDischarge catalog permissiveSearchDischarge).hasFault
           .policyDoesNotAdmit
     | .faults _ => false) = true := by
  native_decide

theorem permissive_same_profile_search_rejected_in_profile_completion :
    (validateProfileFromCatalog permissiveSearchRequest).hasFault
      (ProfileCompletionFault.profile
        (.dischargeFault toyRequirementId .policyDoesNotAdmit)) = true := by
  native_decide

/-- Premises alone may look fine; they must not mint a validated discharge. -/
theorem permissive_search_premises_can_be_structurally_ok :
    permissiveSearchDischarge.premisesWellFormed = true := by
  native_decide

def missingDischargeRequest : ProfileCompletionRequest :=
  { toyProfileRequest with discharges := [] }

theorem missing_discharge_rejected :
    (validateProfileFromCatalog missingDischargeRequest).hasFault
      (ProfileCompletionFault.profile (.missingDischarge toyRequirementId)) = true ∨
    (validateProfileFromCatalog missingDischargeRequest).hasFault
      (ProfileCompletionFault.profile .emptyDischarges) = true := by
  native_decide

def duplicateDischargeRequest : ProfileCompletionRequest :=
  { toyProfileRequest with
    discharges := [toyDischargeCandidate, toyDischargeCandidate] }

theorem duplicate_discharge_rejected :
    (validateProfileFromCatalog duplicateDischargeRequest).hasFault
      (ProfileCompletionFault.profile (.duplicateDischarge toyRequirementId)) = true := by
  native_decide

theorem certificate_cannot_choose_smaller_set :
    (validateProfileFromCatalog missingDischargeRequest).hasFault
      (ProfileCompletionFault.profile (.missingDischarge toyRequirementId)) = true ∨
    (validateProfileFromCatalog missingDischargeRequest).hasFault
      (ProfileCompletionFault.profile .emptyDischarges) = true := by
  native_decide

def otherSubject : ProductTreeIdentity where
  id := { revision := "other-rev", content := digest "other-tree", dirty := false }

def mismatchedSubjectDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with
    subjectTree := otherSubject
    runtime := { id := { artifact := digest "runtime-dylib" }, sourceTree := otherSubject.id } }

def mismatchedSubjectRequest : ProfileCompletionRequest :=
  { toyProfileRequest with discharges := [mismatchedSubjectDischarge] }

theorem mismatched_subject_rejected :
    (validateProfileFromCatalog mismatchedSubjectRequest).hasFault
      (ProfileCompletionFault.profile .mismatchedSubject) = true := by
  native_decide

def mismatchedRuntimeDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with
    runtime :=
      { id := { artifact := digest "other-runtime" }, sourceTree := toySubject.id } }

def mismatchedRuntimeRequest : ProfileCompletionRequest :=
  { toyProfileRequest with discharges := [mismatchedRuntimeDischarge] }

theorem mismatched_runtime_rejected :
    (validateProfileFromCatalog mismatchedRuntimeRequest).hasFault
      (ProfileCompletionFault.profile .mismatchedRuntime) = true := by
  native_decide

def mismatchedRequirementDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with
    requirement := { toyRequirement with semantics := digest "altered-sem" } }

def mismatchedRequirementRequest : ProfileCompletionRequest :=
  { toyProfileRequest with discharges := [mismatchedRequirementDischarge] }

theorem mismatched_requirement_identity_rejected :
    (validateProfileFromCatalog mismatchedRequirementRequest).hasFault
      (ProfileCompletionFault.profile
        (.dischargeFault toyRequirementId .mismatchedRequirementIdentity)) = true := by
  native_decide

def mismatchedScenarioDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with
    scenario := { id := { value := "SCN-EASY" }, digest := digest "easy-scn" } }

def mismatchedScenarioRequest : ProfileCompletionRequest :=
  { toyProfileRequest with discharges := [mismatchedScenarioDischarge] }

theorem mismatched_scenario_rejected :
    (validateProfileFromCatalog mismatchedScenarioRequest).hasFault
      (ProfileCompletionFault.profile
        (.dischargeFault toyRequirementId .mismatchedScenario)) = true := by
  native_decide

/-- Omitting a required environment (or substituting an easier set) fails. -/
def omittedEnvironmentDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with environments := [toyEnvOther] }

def omittedEnvironmentRequest : ProfileCompletionRequest :=
  { toyProfileRequest with discharges := [omittedEnvironmentDischarge] }

theorem omitted_environment_rejected :
    (validateProfileFromCatalog omittedEnvironmentRequest).hasFault
      (ProfileCompletionFault.profile
        (.dischargeFault toyRequirementId .mismatchedEnvironments)) = true := by
  native_decide

def dirtySubjectDischarge : RequirementDischargeCandidate :=
  { toyDischargeCandidate with
    subjectTree :=
      { id := { revision := "subj-rev", content := digest "subj-tree", dirty := true } }
    runtime :=
      { id := { artifact := digest "runtime-dylib" },
        sourceTree :=
          { revision := "subj-rev", content := digest "subj-tree", dirty := true } } }

theorem dirty_subject_rejected_by_discharge :
    (match validateCatalogClosure toyCatalogCandidate with
     | .ok catalog =>
         (validateRequirementDischarge catalog dirtySubjectDischarge).hasFault
           .dirtySubjectTree
     | .faults _ => false) = true := by
  native_decide

theorem chained_validation_surfaces_catalog_fault :
    (validateProfileFromCatalog
      { toyProfileRequest with catalog := unpromotedCatalog }).hasFault
      (ProfileCompletionFault.catalog .unpromotedTarget) = true := by
  native_decide

end Tgrad.Contract

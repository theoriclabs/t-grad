/-! # Tgrad.Contract.Identity — typed identities and content digests

Certificate boundaries carry these typed identities rather than bare
interchangeable `String` fields. Digests are content-address tokens; later
packets may cryptographically bind them. This module authenticates shape and
distinctness only—it does not claim completion or behavioral truth.
-/

namespace Tgrad.Contract

/-- Opaque content-address token. Distinct types prevent silent mixing. -/
structure ContentDigest where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

def ContentDigest.wellFormed (d : ContentDigest) : Bool :=
  !d.value.trimAscii.isEmpty

def digest (value : String) : ContentDigest := { value := value }

structure TargetId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure SourceClosureId where
  digest : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProfileId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RequirementId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure SpecId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure BoundaryId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ScenarioId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RelationId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ValidatorId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure CalibrationId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure EnvironmentId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProductTreeId where
  revision : String
  content : ContentDigest
  dirty : Bool
  deriving DecidableEq, BEq, Repr, Inhabited

structure RuntimeId where
  artifact : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

structure EvidenceId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure JudgmentId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProofId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure SourceItemId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure BlockerId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure CounterexampleId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure SearchCertificateId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure DerivedComputationId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ImportedSourceId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure AdapterId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Node identity inside a judging-input dependency graph. -/
structure JudgingNodeId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Exact Git commit identity used by freeze-integrity chronology. -/
structure CommitId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Promotion / rejection / abandonment cycle event identity. -/
structure CycleEventId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Blind/freeze protocol identity required for prospective chronology claims. -/
structure BlindFreezeProtocolId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Extractor/capture identity for an imported ancestry manifest. -/
structure AncestryCaptureId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

def TargetId.wellFormed (id : TargetId) : Bool := !id.value.trimAscii.isEmpty
def SourceClosureId.wellFormed (id : SourceClosureId) : Bool := id.digest.wellFormed
def ProfileId.wellFormed (id : ProfileId) : Bool := !id.value.trimAscii.isEmpty
def RequirementId.wellFormed (id : RequirementId) : Bool := !id.value.trimAscii.isEmpty
def SpecId.wellFormed (id : SpecId) : Bool := !id.value.trimAscii.isEmpty
def BoundaryId.wellFormed (id : BoundaryId) : Bool := !id.value.trimAscii.isEmpty
def ScenarioId.wellFormed (id : ScenarioId) : Bool := !id.value.trimAscii.isEmpty
def RelationId.wellFormed (id : RelationId) : Bool := !id.value.trimAscii.isEmpty
def ValidatorId.wellFormed (id : ValidatorId) : Bool := !id.value.trimAscii.isEmpty
def CalibrationId.wellFormed (id : CalibrationId) : Bool := !id.value.trimAscii.isEmpty
def EnvironmentId.wellFormed (id : EnvironmentId) : Bool := !id.value.trimAscii.isEmpty
/-- Structural validity only. Dirtiness is a separate eligibility concern. -/
def ProductTreeId.wellFormed (id : ProductTreeId) : Bool :=
  !id.revision.trimAscii.isEmpty && id.content.wellFormed
def ProductTreeId.isClean (id : ProductTreeId) : Bool := !id.dirty
def ProductTreeId.isEligible (id : ProductTreeId) : Bool :=
  id.wellFormed && id.isClean
def RuntimeId.wellFormed (id : RuntimeId) : Bool := id.artifact.wellFormed
def EvidenceId.wellFormed (id : EvidenceId) : Bool := !id.value.trimAscii.isEmpty
def JudgmentId.wellFormed (id : JudgmentId) : Bool := !id.value.trimAscii.isEmpty
def ProofId.wellFormed (id : ProofId) : Bool := !id.value.trimAscii.isEmpty
def SourceItemId.wellFormed (id : SourceItemId) : Bool := !id.value.trimAscii.isEmpty
def BlockerId.wellFormed (id : BlockerId) : Bool := !id.value.trimAscii.isEmpty
def CounterexampleId.wellFormed (id : CounterexampleId) : Bool :=
  !id.value.trimAscii.isEmpty
def SearchCertificateId.wellFormed (id : SearchCertificateId) : Bool :=
  !id.value.trimAscii.isEmpty
def DerivedComputationId.wellFormed (id : DerivedComputationId) : Bool :=
  !id.value.trimAscii.isEmpty
def ImportedSourceId.wellFormed (id : ImportedSourceId) : Bool :=
  !id.value.trimAscii.isEmpty
def AdapterId.wellFormed (id : AdapterId) : Bool := !id.value.trimAscii.isEmpty
def JudgingNodeId.wellFormed (id : JudgingNodeId) : Bool := !id.value.trimAscii.isEmpty
def CommitId.wellFormed (id : CommitId) : Bool := !id.value.trimAscii.isEmpty
def CycleEventId.wellFormed (id : CycleEventId) : Bool := !id.value.trimAscii.isEmpty
def BlindFreezeProtocolId.wellFormed (id : BlindFreezeProtocolId) : Bool :=
  !id.value.trimAscii.isEmpty
def AncestryCaptureId.wellFormed (id : AncestryCaptureId) : Bool :=
  !id.value.trimAscii.isEmpty

inductive TargetDisposition where
  | extractedCandidate
  | promoted
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Promoted-or-candidate target bound to a source closure and profile scope. -/
structure TargetIdentity where
  id : TargetId
  repository : String
  revision : ContentDigest
  sourceClosure : SourceClosureId
  profile : ProfileId
  disposition : TargetDisposition
  deriving DecidableEq, BEq, Repr, Inhabited

def TargetIdentity.wellFormed (t : TargetIdentity) : Bool :=
  t.id.wellFormed &&
  !t.repository.trimAscii.isEmpty &&
  t.revision.wellFormed &&
  t.sourceClosure.wellFormed &&
  t.profile.wellFormed

def TargetIdentity.isPromoted (t : TargetIdentity) : Bool :=
  t.disposition == .promoted

structure JudgmentIdentity where
  id : JudgmentId
  authority : String
  scope : ContentDigest
  invalidation : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def JudgmentIdentity.wellFormed (j : JudgmentIdentity) : Bool :=
  j.id.wellFormed &&
  !j.authority.trimAscii.isEmpty &&
  j.scope.wellFormed &&
  j.invalidation.wellFormed

structure ProofIdentity where
  id : ProofId
  theoremName : String
  deriving DecidableEq, BEq, Repr, Inhabited

def ProofIdentity.wellFormed (p : ProofIdentity) : Bool :=
  p.id.wellFormed && !p.theoremName.trimAscii.isEmpty

structure ProductTreeIdentity where
  id : ProductTreeId
  deriving DecidableEq, BEq, Repr, Inhabited

def ProductTreeIdentity.wellFormed (t : ProductTreeIdentity) : Bool :=
  t.id.wellFormed

def ProductTreeIdentity.isClean (t : ProductTreeIdentity) : Bool :=
  t.id.isClean

def ProductTreeIdentity.isEligible (t : ProductTreeIdentity) : Bool :=
  t.id.isEligible

structure RuntimeIdentity where
  id : RuntimeId
  sourceTree : ProductTreeId
  deriving DecidableEq, BEq, Repr, Inhabited

def RuntimeIdentity.wellFormed (r : RuntimeIdentity) : Bool :=
  r.id.wellFormed && r.sourceTree.wellFormed

structure EnvironmentIdentity where
  id : EnvironmentId
  digest : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def EnvironmentIdentity.wellFormed (e : EnvironmentIdentity) : Bool :=
  e.id.wellFormed && e.digest.wellFormed

structure EvidenceIdentity where
  id : EvidenceId
  digest : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def EvidenceIdentity.wellFormed (e : EvidenceIdentity) : Bool :=
  e.id.wellFormed && e.digest.wellFormed

structure SpecIdentity where
  id : SpecId
  semantics : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def SpecIdentity.wellFormed (s : SpecIdentity) : Bool :=
  s.id.wellFormed && s.semantics.wellFormed

structure BoundaryIdentity where
  id : BoundaryId
  semantics : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def BoundaryIdentity.wellFormed (b : BoundaryIdentity) : Bool :=
  b.id.wellFormed && b.semantics.wellFormed

structure RequirementIdentity where
  id : RequirementId
  semantics : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def RequirementIdentity.wellFormed (r : RequirementIdentity) : Bool :=
  r.id.wellFormed && r.semantics.wellFormed

structure ScenarioIdentity where
  id : ScenarioId
  digest : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def ScenarioIdentity.wellFormed (s : ScenarioIdentity) : Bool :=
  s.id.wellFormed && s.digest.wellFormed

structure RelationIdentity where
  id : RelationId
  digest : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def RelationIdentity.wellFormed (r : RelationIdentity) : Bool :=
  r.id.wellFormed && r.digest.wellFormed

structure AdapterIdentity where
  id : AdapterId
  digest : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def AdapterIdentity.wellFormed (a : AdapterIdentity) : Bool :=
  a.id.wellFormed && a.digest.wellFormed

structure ValidatorIdentity where
  id : ValidatorId
  version : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def ValidatorIdentity.wellFormed (v : ValidatorIdentity) : Bool :=
  v.id.wellFormed && v.version.wellFormed

structure CalibrationIdentity where
  id : CalibrationId
  campaign : ContentDigest
  faultModel : String
  deriving DecidableEq, BEq, Repr, Inhabited

def CalibrationIdentity.wellFormed (c : CalibrationIdentity) : Bool :=
  c.id.wellFormed && c.campaign.wellFormed && !c.faultModel.trimAscii.isEmpty

/-- Certificate/evidence purpose axis, distinct from obligation-class policy. -/
inductive CertificatePurpose where
  | semanticCompatibility
  | performance
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Exact set equality for identity lists used at certificate boundaries. -/
def listNodup {α : Type} [DecidableEq α] (as : List α) : Bool :=
  decide as.Nodup

def listSetEq {α : Type} [BEq α] [DecidableEq α] (as bs : List α) : Bool :=
  as.length == bs.length &&
  listNodup as &&
  listNodup bs &&
  as.all bs.contains &&
  bs.all as.contains

end Tgrad.Contract

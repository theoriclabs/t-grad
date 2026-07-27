/-! # Tgrad.Requirements.World — the problem world around Tgrad

This module deliberately imports no Tgrad product module.  It names the
environmental domains and shared phenomena in which compatibility matters.
Concrete product symbols belong in `Tgrad.Conformance`, not here.
-/

namespace Tgrad.Requirements

structure DomainId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure PhenomenonId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure AssumptionId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RequirementId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure FrameId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ProfileId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure SourceId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

def DomainId.valid (id : DomainId) : Bool := !id.value.trimAscii.isEmpty
def PhenomenonId.valid (id : PhenomenonId) : Bool := !id.value.trimAscii.isEmpty
def AssumptionId.valid (id : AssumptionId) : Bool := !id.value.trimAscii.isEmpty
def RequirementId.valid (id : RequirementId) : Bool := !id.value.trimAscii.isEmpty
def FrameId.valid (id : FrameId) : Bool := !id.value.trimAscii.isEmpty
def ProfileId.valid (id : ProfileId) : Bool := !id.value.trimAscii.isEmpty
def SourceId.valid (id : SourceId) : Bool := !id.value.trimAscii.isEmpty

/-- Jackson's lexical, biddable, and causal problem-domain distinction. -/
inductive DomainKind where
  | lexical
  | biddable
  | causal
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Which side can directly cause a shared phenomenon. -/
inductive Controller where
  | environment
  | machine
  deriving DecidableEq, BEq, Repr, Inhabited

structure WorldDomain where
  id : DomainId
  name : String
  kind : DomainKind
  description : String
  deriving DecidableEq, BEq, Repr, Inhabited

def WorldDomain.wellFormed (domain : WorldDomain) : Bool :=
  domain.id.valid &&
  !domain.name.trimAscii.isEmpty &&
  !domain.description.trimAscii.isEmpty

inductive PhenomenonKind where
  | moduleRequest
  | moduleResolution
  | publicName
  | call
  | argument
  | tensorValue
  | shape
  | dtype
  | exception
  | effect
  | storageIdentity
  | lifetime
  | realization
  | deviceEvent
  | timingSample
  deriving DecidableEq, BEq, Repr, Inhabited

/-- A phenomenon visible at a boundary between Tgrad and its problem world. -/
structure Phenomenon where
  id : PhenomenonId
  domain : DomainId
  kind : PhenomenonKind
  controlledBy : Controller
  description : String
  deriving DecidableEq, BEq, Repr, Inhabited

def Phenomenon.wellFormed (phenomenon : Phenomenon) : Bool :=
  phenomenon.id.valid &&
  phenomenon.domain.valid &&
  !phenomenon.description.trimAscii.isEmpty

inductive SourceKind where
  | upstreamSource
  | upstreamDocumentation
  | upstreamTest
  | upstreamRuntime
  | mathematicalDefinition
  | externalStandard
  | projectDecision
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Provenance for a fact or interpretation; not evidence that Tgrad conforms. -/
structure SourceRef where
  id : SourceId
  kind : SourceKind
  revision : String
  locator : String
  deriving DecidableEq, BEq, Repr, Inhabited

def SourceRef.wellFormed (source : SourceRef) : Bool :=
  source.id.valid &&
  !source.revision.trimAscii.isEmpty &&
  !source.locator.trimAscii.isEmpty

/-- An indicative statement about the problem world. -/
structure Assumption where
  id : AssumptionId
  domains : List DomainId
  statement : String
  provenance : List SourceRef
  deriving DecidableEq, BEq, Repr, Inhabited

def Assumption.wellFormed (assumption : Assumption) : Bool :=
  assumption.id.valid &&
  !assumption.domains.isEmpty &&
  assumption.domains.all DomainId.valid &&
  !assumption.statement.trimAscii.isEmpty &&
  !assumption.provenance.isEmpty &&
  assumption.provenance.all SourceRef.wellFormed

/-- A declared slice of the otherwise unbounded compatibility problem. -/
structure CompatibilityProfile where
  id : ProfileId
  upstreamRevision : String
  includedFrames : List FrameId
  environments : List String
  description : String
  deriving DecidableEq, BEq, Repr, Inhabited

def CompatibilityProfile.wellFormed (profile : CompatibilityProfile) : Bool :=
  profile.id.valid &&
  !profile.upstreamRevision.trimAscii.isEmpty &&
  !profile.includedFrames.isEmpty &&
  profile.includedFrames.all FrameId.valid &&
  !profile.environments.isEmpty &&
  profile.environments.all (fun environment => !environment.trimAscii.isEmpty) &&
  !profile.description.trimAscii.isEmpty

end Tgrad.Requirements

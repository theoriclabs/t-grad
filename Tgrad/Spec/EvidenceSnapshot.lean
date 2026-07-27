import Tgrad.Spec.Evolution

/-! # Tgrad.Spec.EvidenceSnapshot — evidence as a promoted closure

Gate output is not release evidence merely because JSON files exist.  This
module records the ontology implemented by `scripts/evidence/candidate.py`:
one clean source identity, an immutable ordered gate inventory, a total run,
a same-source prepared-runtime prerequisite, and a durable referent for every
hash claim. Candidate state is derived from observations; `promoted` is a
separate type, not an authored state label. -/

namespace Tgrad.Spec

structure Sha256 where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

private def lowercaseHex : List Char :=
  "0123456789abcdef".toList

def Sha256.wellFormed (digest : Sha256) : Bool :=
  digest.value.length == 64 && digest.value.toList.all lowercaseHex.contains

/-- Independent pins for the reviewed machine-readable release denominator.
Changing either JSON contract must deliberately change these constants too. -/
def evidenceReleaseInventoryDigest : Sha256 :=
  { value := "0a44ea4eaeca5169e79a68e9d9aec4c7e6c7597d90498dc680c86282576e3b9a" }

def evidenceHashContractDigest : Sha256 :=
  { value := "b99286b4a8eed0908b03525a9aed85dfa040381ce7594e3608cf6753d58dd4af" }

structure EvidenceSource where
  repository : String
  commit : String
  tree : String
  cleanObservation : Sha256
  deriving DecidableEq, BEq, Repr, Inhabited

def EvidenceSource.wellFormed (source : EvidenceSource) : Bool :=
  !source.repository.isEmpty &&
  !source.commit.isEmpty &&
  !source.tree.isEmpty &&
  source.cleanObservation.wellFormed

structure EvidenceGateDefinition where
  id : Sha256
  name : String
  writerPath : String
  writerDigest : Sha256
  dependsOn : List Sha256
  deriving DecidableEq, BEq, Repr, Inhabited

def EvidenceGateDefinition.wellFormed (gate : EvidenceGateDefinition) : Bool :=
  gate.id.wellFormed && !gate.name.isEmpty && !gate.writerPath.isEmpty &&
  gate.writerDigest.wellFormed && gate.dependsOn.all Sha256.wellFormed &&
  gate.dependsOn.eraseDups.length == gate.dependsOn.length

structure EvidenceEnvironment where
  host : String
  platform : String
  manifestDigest : Sha256
  deriving DecidableEq, BEq, Repr, Inhabited

def EvidenceEnvironment.wellFormed (environment : EvidenceEnvironment) : Bool :=
  !environment.host.isEmpty && !environment.platform.isEmpty &&
  environment.manifestDigest.wellFormed

structure RetainedArtifact where
  locator : String
  digest : Sha256
  deriving DecidableEq, BEq, Repr, Inhabited

def RetainedArtifact.wellFormed (artifact : RetainedArtifact) : Bool :=
  !artifact.locator.isEmpty && artifact.digest.wellFormed

structure ReviewedPerformanceDecision where
  digest : Sha256
  statistic : String
  observed : String
  threshold : String
  accepted : Bool
  deriving Repr, Inhabited

def ReviewedPerformanceDecision.wellFormed
    (decision : ReviewedPerformanceDecision) : Bool :=
  decision.digest.wellFormed && !decision.statistic.isEmpty &&
  !decision.observed.isEmpty && !decision.threshold.isEmpty && decision.accepted

structure PromotedPerformancePrerequisite where
  certificate : RetainedArtifact
  source : EvidenceSource
  boundaryId : String
  independentRuns : Nat
  logicalSessions : Nat
  evaluationArtifacts : List RetainedArtifact
  varianceModelDigest : Sha256
  decision : ReviewedPerformanceDecision
  deriving Repr, Inhabited

def PromotedPerformancePrerequisite.wellFormed
    (expectedSource : EvidenceSource)
    (prerequisite : PromotedPerformancePrerequisite) : Bool :=
  prerequisite.certificate.wellFormed && prerequisite.source == expectedSource &&
  prerequisite.boundaryId == "prepared_runtime" && prerequisite.independentRuns >= 3 &&
  prerequisite.logicalSessions >= 3 * prerequisite.independentRuns &&
  !prerequisite.evaluationArtifacts.isEmpty &&
  prerequisite.evaluationArtifacts.all RetainedArtifact.wellFormed &&
  prerequisite.varianceModelDigest.wellFormed && prerequisite.decision.wellFormed

inductive EvidenceGateStatus where
  | pass
  | red
  | blocked
  deriving DecidableEq, BEq, Repr, Inhabited

structure EvidenceGateOutcome where
  gateId : Sha256
  status : EvidenceGateStatus
  returnCode : Int
  processLog : RetainedArtifact
  evidenceDocument : Option RetainedArtifact
  blockedBy : List String
  deriving DecidableEq, BEq, Repr, Inhabited

structure ArtifactObservation where
  producerId : Sha256
  phase : String
  observed : List RetainedArtifact
  observationDigest : Sha256
  deriving Repr, Inhabited

def ArtifactObservation.wellFormed (observation : ArtifactObservation) : Bool :=
  observation.producerId.wellFormed &&
  (observation.phase == "before" || observation.phase == "after") &&
  observation.observed.all RetainedArtifact.wellFormed &&
  observation.observationDigest.wellFormed

structure ProducedArtifactRecord where
  producerId : Sha256
  beforeObservation : ArtifactObservation
  afterObservation : ArtifactObservation
  changedArtifacts : List RetainedArtifact
  recordDigest : Sha256
  deriving Repr, Inhabited

def ProducedArtifactRecord.wellFormed (record : ProducedArtifactRecord) : Bool :=
  record.producerId.wellFormed && record.beforeObservation.wellFormed &&
  record.afterObservation.wellFormed &&
  record.beforeObservation.producerId == record.producerId &&
  record.afterObservation.producerId == record.producerId &&
  record.beforeObservation.phase == "before" &&
  record.afterObservation.phase == "after" &&
  record.changedArtifacts.all RetainedArtifact.wellFormed && record.recordDigest.wellFormed

def EvidenceGateOutcome.passed (outcome : EvidenceGateOutcome) : Bool :=
  outcome.status == .pass && outcome.returnCode == 0 && outcome.blockedBy.isEmpty &&
  outcome.processLog.wellFormed &&
  match outcome.evidenceDocument with
  | some document => document.wellFormed
  | none => false

def EvidenceGateOutcome.completed (outcome : EvidenceGateOutcome) : Bool :=
  outcome.processLog.wellFormed &&
  match outcome.status with
  | .pass => outcome.passed
  | .red => outcome.blockedBy.isEmpty &&
      match outcome.evidenceDocument with
      | some document => document.wellFormed
      | none => true
  | .blocked => outcome.returnCode != 0 && !outcome.blockedBy.isEmpty &&
      outcome.evidenceDocument.isNone

structure EvidenceHashClaim where
  gateId : Sha256
  key : String
  digest : Sha256
  deriving DecidableEq, BEq, Repr, Inhabited

def EvidenceHashClaim.identity (claim : EvidenceHashClaim) : Sha256 × String :=
  (claim.gateId, claim.key)

structure SourceBlobReferent where
  repository : String
  commit : String
  tree : String
  path : String
  digest : Sha256
  deriving DecidableEq, BEq, Repr, Inhabited

structure ArtifactBlobReferent where
  contentAddress : String
  digest : Sha256
  retentionPolicyId : String
  deriving DecidableEq, BEq, Repr, Inhabited

inductive DurableReferent where
  | sourceBlob (value : SourceBlobReferent)
  | retainedArtifact (value : ArtifactBlobReferent)
  deriving DecidableEq, BEq, Repr, Inhabited

structure EvidenceHashReferent where
  claim : EvidenceHashClaim
  referent : DurableReferent
  deriving DecidableEq, BEq, Repr, Inhabited

def DurableReferent.resolves
    (source : EvidenceSource) (claim : EvidenceHashClaim)
    (referent : DurableReferent) : Bool :=
  match referent with
  | .sourceBlob blob =>
      !blob.path.isEmpty && blob.digest.wellFormed &&
      blob.repository == source.repository && blob.commit == source.commit &&
      blob.tree == source.tree && blob.digest == claim.digest
  | .retainedArtifact artifact =>
      !artifact.contentAddress.isEmpty && !artifact.retentionPolicyId.isEmpty &&
      artifact.digest.wellFormed && artifact.digest == claim.digest

inductive EvidenceCandidateState where
  | collecting
  | completeRed
  | completeGreen
  | invalid
  deriving DecidableEq, BEq, Repr, Inhabited

structure EvidenceCandidate where
  source : EvidenceSource
  environment : EvidenceEnvironment
  releaseInventoryDigest : Sha256
  hashContractDigest : Sha256
  performancePrerequisite : Option PromotedPerformancePrerequisite
  gates : List EvidenceGateDefinition
  outcomes : List EvidenceGateOutcome
  productionRecords : List ProducedArtifactRecord
  claims : List EvidenceHashClaim
  referents : List EvidenceHashReferent
  deriving Repr, Inhabited

private def dependencyOrderWellFormed : List EvidenceGateDefinition → List Sha256 → Bool
  | [], _ => true
  | gate :: rest, seen =>
      gate.dependsOn.all seen.contains &&
      dependencyOrderWellFormed rest (seen ++ [gate.id])

private def gateInventoryWellFormed (gates : List EvidenceGateDefinition) : Bool :=
  !gates.isEmpty && gates.all EvidenceGateDefinition.wellFormed &&
  (gates.map (·.id)).eraseDups.length == gates.length &&
  (gates.map (·.name)).eraseDups.length == gates.length &&
  dependencyOrderWellFormed gates []

private def orderedPrefix (observed required : List Sha256) : Bool :=
  observed.length < required.length && required.take observed.length == observed

private def exactClaimCover (candidate : EvidenceCandidate) : Bool :=
  let claimIds := candidate.claims.map EvidenceHashClaim.identity
  let gateIds := candidate.gates.map (·.id)
  !candidate.claims.isEmpty &&
  claimIds.eraseDups.length == claimIds.length &&
  candidate.referents.length == candidate.claims.length &&
  candidate.referents.all (fun referent => candidate.claims.contains referent.claim) &&
  gateIds.all (fun gateId => candidate.claims.any (fun claim => claim.gateId == gateId)) &&
  candidate.claims.all fun claim =>
    gateIds.contains claim.gateId && !claim.key.isEmpty && claim.digest.wellFormed &&
      candidate.referents.countP (fun referent => referent.claim == claim) == 1 &&
      candidate.referents.all fun referent =>
        referent.claim != claim || referent.referent.resolves candidate.source claim

private def completedRun (candidate : EvidenceCandidate) : Bool :=
  let gateIds := candidate.gates.map (·.id)
  let producerIds := candidate.productionRecords.map (·.producerId)
  candidate.source.wellFormed && candidate.environment.wellFormed &&
  candidate.releaseInventoryDigest == evidenceReleaseInventoryDigest &&
  candidate.hashContractDigest == evidenceHashContractDigest &&
  gateInventoryWellFormed candidate.gates &&
  candidate.outcomes.map (·.gateId) == gateIds &&
  candidate.outcomes.all EvidenceGateOutcome.completed &&
  candidate.productionRecords.length == candidate.gates.length + 1 &&
  producerIds.eraseDups.length == producerIds.length &&
  gateIds.all producerIds.contains &&
  candidate.productionRecords.all ProducedArtifactRecord.wellFormed

private def releasePrerequisitesWellFormed (candidate : EvidenceCandidate) : Bool :=
  match candidate.performancePrerequisite with
  | some prerequisite => prerequisite.wellFormed candidate.source
  | none => false

def EvidenceCandidate.state (candidate : EvidenceCandidate) : EvidenceCandidateState :=
  let observed := candidate.outcomes.map (·.gateId)
  let required := candidate.gates.map (·.id)
  if !candidate.source.wellFormed || !candidate.environment.wellFormed ||
      candidate.releaseInventoryDigest != evidenceReleaseInventoryDigest ||
      candidate.hashContractDigest != evidenceHashContractDigest ||
      !gateInventoryWellFormed candidate.gates then
    .invalid
  else if orderedPrefix observed required then
    .collecting
  else if completedRun candidate && candidate.outcomes.all EvidenceGateOutcome.passed &&
      exactClaimCover candidate && releasePrerequisitesWellFormed candidate then
    .completeGreen
  else if completedRun candidate && (candidate.outcomes.any (fun outcome => !outcome.passed) ||
      !exactClaimCover candidate || !releasePrerequisitesWellFormed candidate) then
    .completeRed
  else
    .invalid

structure PromotedEvidence where
  candidate : EvidenceCandidate
  snapshotDigest : Sha256
  certificate : Evolution.PromotionCertificate
  deriving Repr, Inhabited

private def promotionCertificateWellFormed
    (source : EvidenceSource) (certificate : Evolution.PromotionCertificate) : Bool :=
  !certificate.growthCase.isEmpty && !certificate.candidate.value.isEmpty &&
  !certificate.checkRuns.isEmpty &&
  certificate.checkRuns.all (fun run => !run.value.isEmpty) &&
  certificate.checkRuns.eraseDups.length == certificate.checkRuns.length &&
  !certificate.requiredObligations.isEmpty && !certificate.acceptedBy.isEmpty &&
  certificate.acceptedBy.all (fun actor => !actor.isEmpty) &&
  certificate.target.promotable && certificate.target.commit == source.commit &&
  certificate.target.tree == source.tree

def promoteEvidence
    (candidate : EvidenceCandidate) (snapshotDigest : Sha256)
    (certificate : Evolution.PromotionCertificate) : Option PromotedEvidence :=
  if candidate.state == .completeGreen && snapshotDigest.wellFormed &&
      promotionCertificateWellFormed candidate.source certificate then
    some { candidate, snapshotDigest, certificate }
  else
    none

def PromotedEvidence.wellFormed (evidence : PromotedEvidence) : Bool :=
  evidence.candidate.state == .completeGreen && evidence.snapshotDigest.wellFormed &&
  promotionCertificateWellFormed evidence.candidate.source evidence.certificate

theorem evidence_contract_digests_are_well_formed :
    evidenceReleaseInventoryDigest.wellFormed && evidenceHashContractDigest.wellFormed := by
  native_decide

theorem default_candidate_is_invalid :
    (default : EvidenceCandidate).state = .invalid := by
  native_decide

theorem default_candidate_cannot_promote :
    promoteEvidence (default : EvidenceCandidate) (default : Sha256)
      (default : Evolution.PromotionCertificate) = none := by
  native_decide

end Tgrad.Spec

import Tgrad.Spec.Growth
import Tgrad.Spec.LiveConditions

/-! # Tgrad.Spec.Evolution — durable work on the codebase

`Growth.Case` says why the repository should change. This module models the
state-changing protocol itself:

`intent -> leased attempt -> immutable candidate -> exact-tree checks -> promotion`

The protocol deliberately records events instead of mutating a `WorkItem` from
"planned" to "done". Intent, execution, candidate state, observations, and
acceptance are different facts with different authorities. In particular, a
check is evidence only for the exact candidate tree named by the check.
-/

namespace Tgrad.Spec.Evolution

structure RevisionRef where
  commit : String
  tree : String
  dirty : Bool
  deriving DecidableEq, BEq, Repr, Inhabited

def RevisionRef.promotable (revision : RevisionRef) : Bool :=
  !revision.commit.isEmpty && !revision.tree.isEmpty && !revision.dirty

structure AttemptId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure CandidateId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure CheckRunId where
  value : String
  deriving DecidableEq, BEq, Repr, Inhabited

instance : ToString AttemptId where toString id := id.value
instance : ToString CandidateId where toString id := id.value
instance : ToString CheckRunId where toString id := id.value

inductive EffectKind where
  | add
  | modify
  | delete
  | publish
  deriving DecidableEq, BEq, Repr, Inhabited

/-- A declared mutation budget. Targets are repository-relative paths or a
named external publication surface. This first version uses exact scopes;
expanding directory scopes into concrete diffs is an explicit next step. -/
structure Effect where
  kind : EffectKind
  target : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure Lease where
  token : String
  resources : List Resource
  validThroughEpoch : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure Attempt where
  id : AttemptId
  intent : Growth.EvolutionWorkId
  actor : String
  base : RevisionRef
  authorizedEffects : List Effect
  lease : Lease
  deriving Repr, Inhabited

structure Candidate where
  id : CandidateId
  attempt : AttemptId
  tree : String
  observedEffects : List Effect
  summary : String
  deriving Repr, Inhabited

inductive CheckOutcome where
  | passed
  | failed (reason : String)
  | blocked (reason : String)
  deriving DecidableEq, BEq, Repr, Inhabited

structure CheckRun where
  id : CheckRunId
  candidate : CandidateId
  tree : String
  validator : Runtime.WorkId
  obligation : Growth.ObligationKind
  outcome : CheckOutcome
  command : String
  artifactDigest : String
  deriving Repr, Inhabited

structure PromotionCertificate where
  growthCase : String
  candidate : CandidateId
  checkRuns : List CheckRunId
  requiredObligations : List Growth.ObligationKind
  acceptedBy : List String
  target : RevisionRef
  residualRisks : List String
  deriving Repr, Inhabited

structure PromotionRevocation where
  candidate : CandidateId
  reason : String
  discoveredBy : String
  deriving Repr, Inhabited

inductive Event where
  | attemptStarted (attempt : Attempt)
  | candidateProduced (candidate : Candidate)
  | checkRecorded (run : CheckRun)
  | promoted (certificate : PromotionCertificate)
  | promotionRevoked (revocation : PromotionRevocation)
  | attemptAbandoned (attempt : AttemptId) (reason : String)
  deriving Repr, Inhabited

structure State where
  epoch : Nat := 0
  attempts : List Attempt := []
  candidates : List Candidate := []
  checks : List CheckRun := []
  promotions : List PromotionCertificate := []
  revokedPromotions : List PromotionRevocation := []
  abandoned : List AttemptId := []
  deriving Repr, Inhabited

inductive TransitionError where
  | duplicateAttempt
  | unknownIntent
  | malformedAttempt
  | resourceCapacityExceeded
  | overlappingEffects
  | unknownAttempt
  | closedAttempt
  | expiredLease
  | duplicateCandidate
  | malformedCandidate
  | unauthorizedEffect
  | duplicateCheck
  | unknownCandidate
  | checkTreeMismatch
  | inadmissibleValidator
  | malformedCheck
  | duplicatePromotion
  | unknownPromotion
  | duplicateRevocation
  | malformedRevocation
  | incompleteCertificate
  | missingRequiredObligation
  | promotionTreeMismatch
  deriving DecidableEq, BEq, Repr, Inhabited

def State.attemptFor? (state : State) (id : AttemptId) : Option Attempt :=
  state.attempts.find? (fun attempt => attempt.id == id)

def State.candidateFor? (state : State) (id : CandidateId) : Option Candidate :=
  state.candidates.find? (fun candidate => candidate.id == id)

def State.checkFor? (state : State) (id : CheckRunId) : Option CheckRun :=
  state.checks.find? (fun run => run.id == id)

def State.attemptWasPromoted (state : State) (id : AttemptId) : Bool :=
  state.promotions.any (fun certificate =>
    match state.candidateFor? certificate.candidate with
    | some candidate => candidate.attempt == id
    | none => false)

def State.activePromotions (state : State) : List PromotionCertificate :=
  state.promotions.filter (fun certificate =>
    !state.revokedPromotions.any (fun revocation =>
      revocation.candidate == certificate.candidate))

def State.attemptClosed (state : State) (id : AttemptId) : Bool :=
  state.abandoned.contains id || state.attemptWasPromoted id

def State.activeAttempts (state : State) : List Attempt :=
  state.attempts.filter (fun attempt =>
    !state.attemptClosed attempt.id && state.epoch <= attempt.lease.validThroughEpoch)

def effectsOverlap (left right : Effect) : Bool :=
  left.target == right.target

def attemptsOverlap (left right : Attempt) : Bool :=
  left.authorizedEffects.any (fun leftEffect =>
    right.authorizedEffects.any (effectsOverlap leftEffect))

def leaseWellFormed (state : State) (lease : Lease) : Bool :=
  !lease.token.isEmpty &&
  !lease.resources.isEmpty &&
  lease.resources.eraseDups.length == lease.resources.length &&
  lease.resources.all (fun resource => (policyFor? resource).isSome) &&
  state.epoch < lease.validThroughEpoch

def resourceCapacityAvailable (state : State) (resource : Resource) : Bool :=
  match policyFor? resource with
  | none => false
  | some policy =>
      let inUse := (state.activeAttempts.filter (fun attempt =>
        attempt.lease.resources.contains resource)).length
      inUse < policy.authoringCapacity

def attemptWellFormed (state : State) (attempt : Attempt) : Bool :=
  !attempt.id.value.isEmpty &&
  !attempt.actor.isEmpty &&
  attempt.base.promotable &&
  !attempt.authorizedEffects.isEmpty &&
  attempt.authorizedEffects.all (fun effect => !effect.target.isEmpty) &&
  leaseWellFormed state attempt.lease

def candidateWellFormed (candidate : Candidate) : Bool :=
  !candidate.id.value.isEmpty && !candidate.tree.isEmpty &&
  !candidate.summary.isEmpty && !candidate.observedEffects.isEmpty

def validatorIsAdmissible (id : Runtime.WorkId) : Bool :=
  match Runtime.workUnitFor? id with
  | none => false
  | some unit =>
      let reflexive := unit.realm == .verification || unit.realm == .specification
      let executable :=
        unit.isState .loadBearing || unit.isState .bounded
      reflexive && executable

def CheckOutcome.passed? : CheckOutcome -> Bool
  | .passed => true
  | _ => false

private def tick (state : State) : State :=
  { state with epoch := state.epoch + 1 }

/-- Apply one repository-evolution event. The known-intent set comes from the
checked work graph, avoiding an import cycle from this protocol back to the
current roadmap. -/
def applyEvent
    (knownIntents : List Growth.EvolutionWorkId) (state : State) (event : Event) :
    Except TransitionError State :=
  match event with
  | .attemptStarted attempt =>
      if state.attempts.any (fun existing => existing.id == attempt.id) then
        .error .duplicateAttempt
      else if !knownIntents.contains attempt.intent then
        .error .unknownIntent
      else if !attemptWellFormed state attempt then
        .error .malformedAttempt
      else if !attempt.lease.resources.all (resourceCapacityAvailable state) then
        .error .resourceCapacityExceeded
      else if state.activeAttempts.any (attemptsOverlap attempt) then
        .error .overlappingEffects
      else
        .ok (tick { state with attempts := state.attempts ++ [attempt] })
  | .candidateProduced candidate =>
      if state.candidates.any (fun existing => existing.id == candidate.id) then
        .error .duplicateCandidate
      else match state.attemptFor? candidate.attempt with
        | none => .error .unknownAttempt
        | some attempt =>
            if state.attemptClosed attempt.id then
              .error .closedAttempt
            else if state.epoch > attempt.lease.validThroughEpoch then
              .error .expiredLease
            else if !candidateWellFormed candidate || candidate.tree == attempt.base.tree then
              .error .malformedCandidate
            else if !candidate.observedEffects.all attempt.authorizedEffects.contains then
              .error .unauthorizedEffect
            else
              .ok (tick { state with candidates := state.candidates ++ [candidate] })
  | .checkRecorded run =>
      if state.checks.any (fun existing => existing.id == run.id) then
        .error .duplicateCheck
      else match state.candidateFor? run.candidate with
        | none => .error .unknownCandidate
        | some candidate =>
            if run.tree != candidate.tree then
              .error .checkTreeMismatch
            else if !validatorIsAdmissible run.validator then
              .error .inadmissibleValidator
            else if run.id.value.isEmpty || run.command.isEmpty || run.artifactDigest.isEmpty then
              .error .malformedCheck
            else
              .ok (tick { state with checks := state.checks ++ [run] })
  | .promoted certificate =>
      if state.promotions.any (fun existing => existing.candidate == certificate.candidate) then
        .error .duplicatePromotion
      else match state.candidateFor? certificate.candidate with
        | none => .error .unknownCandidate
        | some candidate =>
            let selected := certificate.checkRuns.filterMap state.checkFor?
            let checksComplete :=
              !certificate.growthCase.isEmpty &&
              !certificate.checkRuns.isEmpty &&
              certificate.checkRuns.eraseDups.length == certificate.checkRuns.length &&
              selected.length == certificate.checkRuns.length &&
              !certificate.requiredObligations.isEmpty &&
              !certificate.acceptedBy.isEmpty &&
              certificate.acceptedBy.all (fun actor => !actor.isEmpty) &&
              selected.all (fun run =>
                run.candidate == candidate.id && run.tree == candidate.tree && run.outcome.passed?)
            if !checksComplete then
              .error .incompleteCertificate
            else if !certificate.requiredObligations.all (fun obligation =>
              selected.any (fun run => run.obligation == obligation)) then
              .error .missingRequiredObligation
            else if !certificate.target.promotable || certificate.target.tree != candidate.tree then
              .error .promotionTreeMismatch
            else
              .ok (tick { state with promotions := state.promotions ++ [certificate] })
  | .promotionRevoked revocation =>
      if !state.promotions.any (fun certificate =>
          certificate.candidate == revocation.candidate) then
        .error .unknownPromotion
      else if state.revokedPromotions.any (fun existing =>
          existing.candidate == revocation.candidate) then
        .error .duplicateRevocation
      else if revocation.reason.isEmpty || revocation.discoveredBy.isEmpty then
        .error .malformedRevocation
      else
        .ok (tick { state with
          revokedPromotions := state.revokedPromotions ++ [revocation] })
  | .attemptAbandoned attemptId reason =>
      match state.attemptFor? attemptId with
      | none => .error .unknownAttempt
      | some _ =>
          if state.attemptClosed attemptId then .error .closedAttempt
          else if reason.isEmpty then .error .incompleteCertificate
          else .ok (tick { state with abandoned := state.abandoned ++ [attemptId] })

def replayFrom
    (knownIntents : List Growth.EvolutionWorkId) : State -> List Event ->
    Except TransitionError State
  | state, [] => .ok state
  | state, event :: rest =>
      match applyEvent knownIntents state event with
      | .error error => .error error
      | .ok next => replayFrom knownIntents next rest

def replay (knownIntents : List Growth.EvolutionWorkId) (events : List Event) :
    Except TransitionError State :=
  replayFrom knownIntents {} events

/-! Executable negative cases: stale checks and obligation substitution must
remain impossible even when all strings are otherwise plausible. -/

private def sampleIntent : Growth.EvolutionWorkId :=
  Growth.evolutionWorkId "sample.intent"

private def sampleEffect : Effect := { kind := .modify, target := "Tgrad/Sample.lean" }

private def sampleAttempt : Attempt :=
  { id := { value := "attempt-1" }, intent := sampleIntent, actor := "agent",
    base := { commit := "base-commit", tree := "base-tree", dirty := false },
    authorizedEffects := [sampleEffect],
    lease := { token := "lease-1", resources := [.sourceTree], validThroughEpoch := 20 } }

private def sampleCandidate : Candidate :=
  { id := { value := "candidate-1" }, attempt := sampleAttempt.id,
    tree := "candidate-tree", observedEffects := [sampleEffect], summary := "typed change" }

private def stateWithAttempt : State :=
  match replay [sampleIntent] [.attemptStarted sampleAttempt] with
  | .ok state => state
  | .error _ => {}

private def overlappingAttempt : Attempt :=
  { sampleAttempt with
    id := { value := "attempt-2" }, actor := "other-agent",
    lease := { token := "lease-2", resources := [.sourceTree], validThroughEpoch := 20 } }

theorem overlapping_active_effects_are_rejected :
    (match applyEvent [sampleIntent] stateWithAttempt (.attemptStarted overlappingAttempt) with
     | .error .overlappingEffects => true
     | _ => false) = true := by
  native_decide

private def stateWithCandidate : State :=
  match replay [sampleIntent]
    [.attemptStarted sampleAttempt, .candidateProduced sampleCandidate] with
  | .ok state => state
  | .error _ => {}

private def staleCheck : CheckRun :=
  { id := { value := "check-stale" }, candidate := sampleCandidate.id,
    tree := "different-tree", validator := Runtime.workId "verify.unit-tests",
    obligation := .unitRegression, outcome := .passed,
    command := "lake build tgrad-tests", artifactDigest := "sha256:stale" }

theorem stale_tree_check_is_rejected :
    (match applyEvent [sampleIntent] stateWithCandidate (.checkRecorded staleCheck) with
     | .error .checkTreeMismatch => true
     | _ => false) = true := by
  native_decide

private def regressionCheck : CheckRun :=
  { id := { value := "check-regression" }, candidate := sampleCandidate.id,
    tree := sampleCandidate.tree, validator := Runtime.workId "verify.unit-tests",
    obligation := .unitRegression, outcome := .passed,
    command := "lake build tgrad-tests", artifactDigest := "sha256:regression" }

private def stateWithRegressionCheck : State :=
  match applyEvent [sampleIntent] stateWithCandidate (.checkRecorded regressionCheck) with
  | .ok state => state
  | .error _ => {}

private def insufficientCertificate : PromotionCertificate :=
  { growthCase := "sample-growth", candidate := sampleCandidate.id,
    checkRuns := [regressionCheck.id], requiredObligations := [.numerical],
    acceptedBy := ["integrator"],
    target := { commit := "candidate-commit", tree := sampleCandidate.tree, dirty := false },
    residualRisks := [] }

theorem one_obligation_cannot_substitute_for_another :
    (match applyEvent [sampleIntent] stateWithRegressionCheck
      (.promoted insufficientCertificate) with
     | .error .missingRequiredObligation => true
     | _ => false) = true := by
  native_decide

private def sufficientCertificate : PromotionCertificate :=
  { insufficientCertificate with requiredObligations := [.unitRegression] }

theorem exact_tree_complete_certificate_is_promotable :
    (match applyEvent [sampleIntent] stateWithRegressionCheck
      (.promoted sufficientCertificate) with
     | .ok state => state.promotions.length == 1
     | .error _ => false) = true := by
  native_decide

private def stateWithPromotion : State :=
  match applyEvent [sampleIntent] stateWithRegressionCheck
      (.promoted sufficientCertificate) with
  | .ok state => state
  | .error _ => {}

theorem a_falsified_promotion_can_be_revoked_without_erasing_history :
    (match applyEvent [sampleIntent] stateWithPromotion
      (.promotionRevoked
        { candidate := sampleCandidate.id,
          reason := "adversarial replay check failed",
          discoveredBy := "independent reviewer" }) with
     | .ok state =>
         state.promotions.length == 1 &&
         state.revokedPromotions.length == 1 &&
         state.activePromotions.isEmpty
     | .error _ => false) = true := by
  native_decide

end Tgrad.Spec.Evolution

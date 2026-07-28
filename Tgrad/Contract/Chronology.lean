import Tgrad.Contract.Identity
import Tgrad.Contract.Assurance

/-! # Tgrad.Contract.Chronology — judging-input closure and freeze integrity

Packet `mechanics.cycle-chronology-v1`.

This module is freeze-integrity evidence only. It does not discharge semantic
requirements, prove behavioral compatibility, or authenticate cryptographic
digests. `ContentDigest` remains a shape token: canonical closure identity is a
deterministic length-prefixed encoding over validated structure, not a
cryptographic hash bound to repository objects.

Residual trust assumption: the imported ancestry capture
(`capturedCommitIds`, records, snapshots, cycle descriptors, target/profile
scope, and frozen closure identity) is unauthenticated until a grounded
extractor exists. Schema equality across these objects is binding, not
extractor authentication of Git or artifact contents.

Roles:

* Lean validates typed judging-input graphs, ancestry manifests, and cycle
  registries, returning diagnosed faults. Validated forms have private
  constructors; candidates carry premises only and cannot author a pass Boolean
  or a conclusion digest.
* An independent Python checker (`scripts/contract/check_cycle_chronology.py`)
  exercises full-history Git ancestry mechanics on a synthetic temporary
  repository. Agreement between the Lean schema checks and the Python Git
  exercise is not authentication of either artifact.
-/

namespace Tgrad.Contract

/-! ## Schema version and judging-input categories -/

/-- Schema version bound into every canonical judging-input closure identity. -/
def judgingInputClosureSchemaVersion : Nat := 1

/-- Required judging-input categories. Every validated closure must contain at
least one node in each category. -/
inductive JudgingInputCategory where
  | targetProfileDecision
  | requirementDenotation
  | boundaryDenotation
  | adapter
  | relation
  | scenarioGenerator
  | validator
  | calibrationPolicy
  | importedHelper
  | environmentPolicy
  | toolchain
  | claimRenderer
  deriving DecidableEq, BEq, Repr, Inhabited

def JudgingInputCategory.tag : JudgingInputCategory → String
  | .targetProfileDecision => "targetProfileDecision"
  | .requirementDenotation => "requirementDenotation"
  | .boundaryDenotation => "boundaryDenotation"
  | .adapter => "adapter"
  | .relation => "relation"
  | .scenarioGenerator => "scenarioGenerator"
  | .validator => "validator"
  | .calibrationPolicy => "calibrationPolicy"
  | .importedHelper => "importedHelper"
  | .environmentPolicy => "environmentPolicy"
  | .toolchain => "toolchain"
  | .claimRenderer => "claimRenderer"

def allRequiredJudgingCategories : List JudgingInputCategory :=
  [.targetProfileDecision, .requirementDenotation, .boundaryDenotation,
   .adapter, .relation, .scenarioGenerator, .validator, .calibrationPolicy,
   .importedHelper, .environmentPolicy, .toolchain, .claimRenderer]

/-! ## Length-prefixed canonical encoding

Delimiter concatenation of arbitrary strings is non-injective. Every payload
field is length-prefixed so distinct structures cannot alias.
-/

def encStr (s : String) : String := s!"{s.length}:{s}"

def encNat (n : Nat) : String := encStr (toString n)

def encDigest (d : ContentDigest) : String := encStr d.value

/-- Length-prefixed sequence of already-encoded parts. -/
def encSeq (parts : List String) : String :=
  encNat parts.length ++ String.join parts

/-- Complete provenance payload. Changing any referenced identity changes the
canonical encoding. -/
def encodeFieldProvenance : FieldProvenance → String
  | .imported s =>
      encStr "imported" ++ encStr s.id.value ++ encDigest s.sourceClosure.digest
  | .derived c =>
      encStr "derived" ++ encStr c.id.value ++ encStr c.verifier.value ++
        encDigest c.inputClosure
  | .calibrated c =>
      encStr "calibrated" ++ encStr c.id.value ++ encDigest c.campaign ++
        encStr c.faultModel
  | .judgment j =>
      encStr "judgment" ++ encStr j.id.value ++ encStr j.authority ++
        encDigest j.scope ++ encDigest j.invalidation

/-! ## Judging-input nodes and candidates (premises only) -/

structure JudgingInputNode where
  id : JudgingNodeId
  category : JudgingInputCategory
  content : ContentDigest
  dependencies : List JudgingNodeId
  provenance : FieldProvenance
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Public candidate. Deliberately has no authored closure digest field: the
canonical identity is derived only after validation succeeds. -/
structure JudgingInputClosureCandidate where
  schemaVersion : Nat
  nodes : List JudgingInputNode
  roots : List JudgingNodeId
  /-- Imported discovered-input inventory. Must set-equal declared node IDs. -/
  discoveredInventory : List JudgingNodeId
  deriving DecidableEq, BEq, Repr, Inhabited

structure ValidatedJudgingInputClosure where
  private mk ::
    candidate : JudgingInputClosureCandidate
    /-- Derived canonical identity. Never authored by the candidate. -/
    closureIdentity : ContentDigest
  deriving Repr

def ValidatedJudgingInputClosure.nodes
    (v : ValidatedJudgingInputClosure) : List JudgingInputNode :=
  v.candidate.nodes

def ValidatedJudgingInputClosure.roots
    (v : ValidatedJudgingInputClosure) : List JudgingNodeId :=
  v.candidate.roots

def ValidatedJudgingInputClosure.identity
    (v : ValidatedJudgingInputClosure) : ContentDigest :=
  v.closureIdentity

/-! ## Chronology mode: retrospective vs prospective -/

structure BlindFreezeProtocolRef where
  id : BlindFreezeProtocolId
  digest : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def BlindFreezeProtocolRef.wellFormed (p : BlindFreezeProtocolRef) : Bool :=
  p.id.wellFormed && p.digest.wellFormed

/-- Claimed chronology kind. Prospective claims require a nonempty protocol. -/
inductive ClaimedChronologyKind where
  | retrospectiveFreezeIntegrity
  | prospective
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Derived kind retained on validated freeze integrity. -/
inductive ChronologyKind where
  | retrospectiveFreezeIntegrity
  | prospective (protocol : BlindFreezeProtocolRef)
  deriving DecidableEq, BEq, Repr, Inhabited

/-! ## Ancestry capture and freeze-integrity candidates

Interval convention: `freeze` is the interval root (`parentsWithinCapture = []`).
`capturedCommitIds` is an imported inventory distinct from records/snapshots;
omission is detected against that inventory, not only a rewritten parent graph.
-/

structure AncestryCommitRecord where
  commit : CommitId
  /-- Parents that lie inside the captured interval. -/
  parentsWithinCapture : List CommitId
  judgingClosureIdentity : ContentDigest
  deriving DecidableEq, BEq, Repr, Inhabited

def AncestryCommitRecord.wellFormed (r : AncestryCommitRecord) : Bool :=
  r.commit.wellFormed &&
  r.parentsWithinCapture.all CommitId.wellFormed &&
  listNodup r.parentsWithinCapture &&
  r.judgingClosureIdentity.wellFormed

/-- Imported full-ancestry manifest between freeze and candidate.

Binds the same full target/profile scope and expected frozen judging-input
closure identity that cycle descriptors carry. Schema binding only: the
imported capture remains unauthenticated until a grounded extractor exists.
-/
structure AncestryManifest where
  captureIdentity : AncestryCaptureId
  /-- Extractor/capture method identity (shape token, not cryptographic auth). -/
  extractorIdentity : ContentDigest
  /-- Full promoted-or-candidate target identity for this chronology scope. -/
  target : TargetIdentity
  profile : ProfileId
  /-- Expected frozen judging-input closure identity for the interval. -/
  frozenJudgingClosureIdentity : ContentDigest
  freeze : CommitId
  candidate : CommitId
  /-- Imported captured commit inventory; distinct from records/snapshots. -/
  capturedCommitIds : List CommitId
  records : List AncestryCommitRecord
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Premises for freeze-integrity validation. Supplied snapshots are a second
imported view that must exactly match the manifest records. -/
structure FreezeIntegrityCandidate where
  claimedKind : ClaimedChronologyKind
  prospectiveProtocol : Option BlindFreezeProtocolRef
  manifest : AncestryManifest
  suppliedSnapshots : List AncestryCommitRecord
  deriving DecidableEq, BEq, Repr, Inhabited

structure ValidatedFreezeIntegrity where
  private mk ::
    candidate : FreezeIntegrityCandidate
    kind : ChronologyKind
  deriving Repr

/-! ## Cycle events and registry

Imported descriptors bind event ID, full `TargetIdentity`, profile, expected
frozen judging-input closure identity, outcome, and freeze/candidate commits.
Registry entries may attach evidence but may not rewrite those facts. This is
schema binding across objects, not extractor authentication.
-/

inductive CycleOutcome where
  | promoted
  | rejected
  | abandoned
  deriving DecidableEq, BEq, Repr, Inhabited

def CycleOutcome.tag : CycleOutcome → String
  | .promoted => "promoted"
  | .rejected => "rejected"
  | .abandoned => "abandoned"

structure CycleEventDescriptor where
  id : CycleEventId
  target : TargetIdentity
  profile : ProfileId
  expectedFrozenJudgingClosureIdentity : ContentDigest
  outcome : CycleOutcome
  freeze : CommitId
  candidate : CommitId
  deriving DecidableEq, BEq, Repr, Inhabited

def CycleEventDescriptor.wellFormed (d : CycleEventDescriptor) : Bool :=
  d.id.wellFormed &&
  d.target.wellFormed &&
  d.profile.wellFormed &&
  d.expectedFrozenJudgingClosureIdentity.wellFormed &&
  d.freeze.wellFormed &&
  d.candidate.wellFormed &&
  d.target.profile == d.profile

/-- Registry entry: descriptor facts are imported; evidence is attached only. -/
structure CycleEventEntry where
  descriptor : CycleEventDescriptor
  /-- Allowed only when `descriptor.outcome = promoted`. -/
  freezeIntegrity : Option FreezeIntegrityCandidate
  deriving DecidableEq, BEq, Repr, Inhabited

structure CycleRegistryCandidate where
  importedDescriptors : List CycleEventDescriptor
  entries : List CycleEventEntry
  deriving DecidableEq, BEq, Repr, Inhabited

structure ValidatedCycleRegistry where
  private mk ::
    candidate : CycleRegistryCandidate
  deriving Repr

def ValidatedCycleRegistry.importedDescriptors
    (v : ValidatedCycleRegistry) : List CycleEventDescriptor :=
  v.candidate.importedDescriptors

def ValidatedCycleRegistry.entries
    (v : ValidatedCycleRegistry) : List CycleEventEntry :=
  v.candidate.entries

/-! ## Typed diagnosed faults -/

inductive JudgingClosureFault where
  | wrongSchemaVersion
  | emptyNodes
  | emptyRoots
  | malformedNode
  | duplicateNode (id : JudgingNodeId)
  | duplicateDependency (node : JudgingNodeId) (dep : JudgingNodeId)
  | unresolvedDependency (node : JudgingNodeId) (dep : JudgingNodeId)
  | malformedRoot (id : JudgingNodeId)
  | duplicateRoot (id : JudgingNodeId)
  | unresolvedRoot (id : JudgingNodeId)
  | unreachableNode (id : JudgingNodeId)
  | dependencyCycle (witness : JudgingNodeId)
  | inadmissibleProvenance (id : JudgingNodeId)
  | missingRequiredCategory (category : JudgingInputCategory)
  | discoveredInventoryMismatch
  | undeclaredDiscoveredInput (id : JudgingNodeId)
  | undiscoveredDeclaredNode (id : JudgingNodeId)
  | duplicateDiscoveredInput (id : JudgingNodeId)
  deriving DecidableEq, BEq, Repr, Inhabited

inductive FreezeIntegrityFault where
  | malformedCaptureIdentity
  | malformedExtractorIdentity
  | malformedTarget
  | unpromotedTarget
  | mismatchedTargetProfile
  | malformedFrozenClosureIdentity
  | malformedFreeze
  | malformedCandidate
  | emptyAncestry
  | emptyCapturedInventory
  | malformedCapturedCommitId (id : CommitId)
  | malformedCommitRecord
  | duplicateCommit (id : CommitId)
  | duplicateCapturedCommit (id : CommitId)
  | captureInventoryMismatch
  | unresolvedParent (commit : CommitId) (parent : CommitId)
  | freezeNotInAncestry
  | candidateNotInAncestry
  | freezeNotIntervalRoot
  | freezeNotAncestor
  | omittedAncestryCommit (id : CommitId)
  | snapshotInventoryMismatch
  | freezeClosureMismatch
  | judgingClosureDrift (id : CommitId)
  | prospectiveWithoutProtocol
  | malformedProspectiveProtocol
  | retrospectiveCarriesProtocol
  deriving DecidableEq, BEq, Repr, Inhabited

inductive CycleRegistryFault where
  | emptyInventory
  | malformedDescriptor (id : CycleEventId)
  | unpromotedDescriptorTarget (id : CycleEventId)
  | duplicateInventoryId (id : CycleEventId)
  | duplicateEntryId (id : CycleEventId)
  | descriptorEntryMismatch
  | omittedInventoryEvent (id : CycleEventId)
  | extraRegistryEvent (id : CycleEventId)
  | promotedMissingFreezeIntegrity (id : CycleEventId)
  | promotedTargetMismatch (id : CycleEventId)
  | promotedProfileMismatch (id : CycleEventId)
  | promotedClosureMismatch (id : CycleEventId)
  | promotedCommitMismatch (id : CycleEventId)
  | nonPromotedCarriesChronology (id : CycleEventId)
  | promotedFreezeIntegrityFault (id : CycleEventId) (fault : FreezeIntegrityFault)
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ChronologyValidationResult (fault : Type) (ok : Type) where
  | ok (value : ok)
  | faults (diagnosed : List fault)
  deriving Repr

def ChronologyValidationResult.isOk {fault ok : Type} :
    ChronologyValidationResult fault ok → Bool
  | .ok _ => true
  | .faults _ => false

def ChronologyValidationResult.hasFault {fault ok : Type} [BEq fault]
    (result : ChronologyValidationResult fault ok) (f : fault) : Bool :=
  match result with
  | .ok _ => false
  | .faults diagnosed => diagnosed.contains f

/-! ## Graph helpers -/

private def pushFault {α : Type} (faults : List α) (cond : Bool) (f : α) : List α :=
  if cond then faults ++ [f] else faults

private def nodeIds (nodes : List JudgingInputNode) : List JudgingNodeId :=
  nodes.map (·.id)

private def lookupNode?
    (nodes : List JudgingInputNode) (id : JudgingNodeId) : Option JudgingInputNode :=
  nodes.find? (fun n => n.id == id)

private def dependencyEdgeCount (nodes : List JudgingInputNode) : Nat :=
  nodes.foldl (fun acc n => acc + n.dependencies.length) 0

/-- Public edge-count used by dense-DAG fuel checks. -/
def judgingDependencyEdgeCount (nodes : List JudgingInputNode) : Nat :=
  dependencyEdgeCount nodes

/-- Reachability from roots following dependency edges.

Fuel covers frontier processing on dense DAGs: nodes + dependency edges + roots.
-/
private def reachableFromRoots
    (nodes : List JudgingInputNode) (roots : List JudgingNodeId) : List JudgingNodeId :=
  let fuel := nodes.length + dependencyEdgeCount nodes + roots.length + 1
  let rec go (frontier : List JudgingNodeId) (seen : List JudgingNodeId) (fuel : Nat) :
      List JudgingNodeId :=
    match fuel with
    | 0 => seen
    | fuel + 1 =>
      match frontier with
      | [] => seen
      | id :: rest =>
          if seen.contains id then go rest seen fuel
          else
            let deps :=
              match lookupNode? nodes id with
              | some n => n.dependencies
              | none => []
            go (rest ++ deps) (seen ++ [id]) fuel
  go roots [] fuel

/-- Path-sensitive dependency cycle detection. -/
private def dependencyCycleFrom
    (nodes : List JudgingInputNode) (start : JudgingNodeId) : Bool :=
  let rec go (current : JudgingNodeId) (path : List JudgingNodeId) (fuel : Nat) : Bool :=
    match fuel with
    | 0 => false
    | fuel + 1 =>
      if path.contains current then true
      else
        match lookupNode? nodes current with
        | none => false
        | some n =>
            n.dependencies.any fun next => go next (current :: path) fuel
  go start [] (nodes.length + dependencyEdgeCount nodes + 1)

private def dependencyCycleWitnesses (nodes : List JudgingInputNode) : List JudgingNodeId :=
  (nodeIds nodes).filterMap fun start =>
    if dependencyCycleFrom nodes start then some start else none

def encodeJudgingInputNode (n : JudgingInputNode) : String :=
  let deps :=
    (n.dependencies.mergeSort (fun a b => a.value ≤ b.value)).map fun d => encStr d.value
  encStr n.id.value ++ encStr n.category.tag ++ encDigest n.content ++
    encSeq deps ++ encodeFieldProvenance n.provenance

/-- Canonical closure identity from validated structure. Shape token only;
includes the complete provenance payload under length-prefixed encoding. -/
def computeJudgingClosureIdentity (c : JudgingInputClosureCandidate) : ContentDigest :=
  let sortedNodes :=
    c.nodes.mergeSort (fun a b => a.id.value ≤ b.id.value)
  let nodePart := encSeq (sortedNodes.map encodeJudgingInputNode)
  let edgePart :=
    encSeq
      (sortedNodes.flatMap fun n =>
        (n.dependencies.mergeSort (fun a b => a.value ≤ b.value)).map fun d =>
          encStr n.id.value ++ encStr d.value)
  let rootPart :=
    encSeq
      ((c.roots.mergeSort (fun a b => a.value ≤ b.value)).map fun r => encStr r.value)
  digest (encNat c.schemaVersion ++ nodePart ++ edgePart ++ rootPart)

/-! ## Judging-input closure validation -/

private def judgingClosureDiagnose (c : JudgingInputClosureCandidate) :
    List JudgingClosureFault :=
  let faults : List JudgingClosureFault := []
  let faults :=
    pushFault faults (c.schemaVersion != judgingInputClosureSchemaVersion)
      .wrongSchemaVersion
  let faults := pushFault faults c.nodes.isEmpty .emptyNodes
  let faults := pushFault faults c.roots.isEmpty .emptyRoots
  let ids := nodeIds c.nodes
  let faults :=
    c.nodes.foldl (fun acc n =>
      let acc :=
        pushFault acc (!(n.id.wellFormed && n.content.wellFormed)) .malformedNode
      let acc :=
        pushFault acc (!ClaimField.hasAdmissibleProvenance
          { name := n.id.value, provenance := n.provenance })
          (.inadmissibleProvenance n.id)
      let acc :=
        if listNodup n.dependencies then acc
        else
          n.dependencies.foldl (fun acc d =>
            if n.dependencies.count d > 1 &&
                !(acc.contains (.duplicateDependency n.id d)) then
              acc ++ [.duplicateDependency n.id d]
            else acc) acc
      n.dependencies.foldl (fun acc d =>
        if ids.contains d then acc
        else acc ++ [.unresolvedDependency n.id d]) acc) faults
  let faults :=
    if listNodup ids then faults
    else
      ids.foldl (fun acc id =>
        if ids.count id > 1 && !(acc.contains (.duplicateNode id)) then
          acc ++ [.duplicateNode id]
        else acc) faults
  let faults :=
    if listNodup c.roots then faults
    else
      c.roots.foldl (fun acc id =>
        if c.roots.count id > 1 && !(acc.contains (.duplicateRoot id)) then
          acc ++ [.duplicateRoot id]
        else acc) faults
  let faults :=
    c.roots.foldl (fun acc id =>
      let acc := pushFault acc (!id.wellFormed) (.malformedRoot id)
      if ids.contains id then acc else acc ++ [.unresolvedRoot id]) faults
  let reachable := reachableFromRoots c.nodes c.roots
  let faults :=
    ids.foldl (fun acc id =>
      if reachable.contains id then acc else acc ++ [.unreachableNode id]) faults
  let faults :=
    (dependencyCycleWitnesses c.nodes).foldl (fun acc w =>
      if acc.contains (.dependencyCycle w) then acc
      else acc ++ [.dependencyCycle w]) faults
  let presentCategories := c.nodes.map (·.category)
  let faults :=
    allRequiredJudgingCategories.foldl (fun acc cat =>
      if presentCategories.contains cat then acc
      else acc ++ [.missingRequiredCategory cat]) faults
  let faults :=
    if listNodup c.discoveredInventory then faults
    else
      c.discoveredInventory.foldl (fun acc id =>
        if c.discoveredInventory.count id > 1 &&
            !(acc.contains (.duplicateDiscoveredInput id)) then
          acc ++ [.duplicateDiscoveredInput id]
        else acc) faults
  let faults :=
    c.discoveredInventory.foldl (fun acc id =>
      if ids.contains id then acc else acc ++ [.undeclaredDiscoveredInput id]) faults
  let faults :=
    ids.foldl (fun acc id =>
      if c.discoveredInventory.contains id then acc
      else acc ++ [.undiscoveredDeclaredNode id]) faults
  pushFault faults (!listSetEq ids c.discoveredInventory) .discoveredInventoryMismatch

def validateJudgingInputClosure (c : JudgingInputClosureCandidate) :
    ChronologyValidationResult JudgingClosureFault ValidatedJudgingInputClosure :=
  match judgingClosureDiagnose c with
  | [] => .ok ⟨c, computeJudgingClosureIdentity c⟩
  | diagnosed => .faults diagnosed

/-! ## Ancestry / freeze-integrity validation -/

private def parentEdgeCount
    (parentMap : List (CommitId × List CommitId)) : Nat :=
  parentMap.foldl (fun acc (_, ps) => acc + ps.length) 0

/-- Walk parentsWithinCapture from `start`. Fuel = commits + parent edges. -/
private def walkParents
    (parentMap : List (CommitId × List CommitId)) (start : CommitId) : List CommitId :=
  let fuel := parentMap.length + parentEdgeCount parentMap + 1
  let rec go (frontier : List CommitId) (seen : List CommitId) (fuel : Nat) :
      List CommitId :=
    match fuel with
    | 0 => seen
    | fuel + 1 =>
      match frontier with
      | [] => seen
      | id :: rest =>
          if seen.contains id then go rest seen fuel
          else
            let parents :=
              match parentMap.find? (fun (c, _) => c == id) with
              | some (_, ps) => ps
              | none => []
            go (rest ++ parents) (seen ++ [id]) fuel
  go [start] [] fuel

private def freezeIntegrityDiagnose (c : FreezeIntegrityCandidate) :
    List FreezeIntegrityFault :=
  let m := c.manifest
  let faults : List FreezeIntegrityFault := []
  let faults := pushFault faults (!m.captureIdentity.wellFormed) .malformedCaptureIdentity
  let faults := pushFault faults (!m.extractorIdentity.wellFormed) .malformedExtractorIdentity
  let faults := pushFault faults (!m.target.wellFormed) .malformedTarget
  let faults := pushFault faults (!m.target.isPromoted) .unpromotedTarget
  let faults := pushFault faults (!m.profile.wellFormed) .mismatchedTargetProfile
  let faults :=
    pushFault faults (m.target.profile != m.profile) .mismatchedTargetProfile
  let faults :=
    pushFault faults (!m.frozenJudgingClosureIdentity.wellFormed)
      .malformedFrozenClosureIdentity
  let faults := pushFault faults (!m.freeze.wellFormed) .malformedFreeze
  let faults := pushFault faults (!m.candidate.wellFormed) .malformedCandidate
  let faults := pushFault faults m.capturedCommitIds.isEmpty .emptyCapturedInventory
  let faults :=
    m.capturedCommitIds.foldl (fun acc id =>
      if id.wellFormed then acc else acc ++ [.malformedCapturedCommitId id]) faults
  let faults := pushFault faults m.records.isEmpty .emptyAncestry
  let faults :=
    pushFault faults (!m.records.all AncestryCommitRecord.wellFormed) .malformedCommitRecord
  let faults :=
    pushFault faults (!c.suppliedSnapshots.all AncestryCommitRecord.wellFormed)
      .malformedCommitRecord
  let recordIds := m.records.map (·.commit)
  let faults :=
    if listNodup m.capturedCommitIds then faults
    else
      m.capturedCommitIds.foldl (fun acc id =>
        if m.capturedCommitIds.count id > 1 &&
            !(acc.contains (.duplicateCapturedCommit id)) then
          acc ++ [.duplicateCapturedCommit id]
        else acc) faults
  let faults :=
    if listNodup recordIds then faults
    else
      recordIds.foldl (fun acc id =>
        if recordIds.count id > 1 && !(acc.contains (.duplicateCommit id)) then
          acc ++ [.duplicateCommit id]
        else acc) faults
  -- Omission / rewrite detected against the imported captured inventory.
  let faults :=
    pushFault faults (!listSetEq m.capturedCommitIds recordIds) .captureInventoryMismatch
  let faults :=
    m.capturedCommitIds.foldl (fun acc id =>
      if recordIds.contains id then acc else acc ++ [.omittedAncestryCommit id]) faults
  let faults :=
    pushFault faults (!(m.capturedCommitIds.contains m.freeze)) .freezeNotInAncestry
  let faults :=
    pushFault faults (!(m.capturedCommitIds.contains m.candidate)) .candidateNotInAncestry
  -- Parent edges are parentsWithinCapture: must resolve inside the capture.
  let faults :=
    m.records.foldl (fun acc r =>
      r.parentsWithinCapture.foldl (fun acc p =>
        if m.capturedCommitIds.contains p then acc
        else acc ++ [.unresolvedParent r.commit p]) acc) faults
  -- Freeze is the interval root.
  let faults :=
    match m.records.find? (fun r => r.commit == m.freeze) with
    | none => faults
    | some fr =>
        pushFault faults (!fr.parentsWithinCapture.isEmpty) .freezeNotIntervalRoot
  let parentMap := m.records.map fun r => (r.commit, r.parentsWithinCapture)
  let reachableFromCandidate := walkParents parentMap m.candidate
  let faults :=
    pushFault faults (!(reachableFromCandidate.contains m.freeze)) .freezeNotAncestor
  -- Interval membership: reachable set must equal the imported capture.
  let faults :=
    reachableFromCandidate.foldl (fun acc id =>
      if m.capturedCommitIds.contains id then acc
      else acc ++ [.omittedAncestryCommit id]) faults
  let faults :=
    m.capturedCommitIds.foldl (fun acc id =>
      if reachableFromCandidate.contains id then acc
      else acc ++ [.omittedAncestryCommit id]) faults
  let faults :=
    pushFault faults (!listSetEq m.records c.suppliedSnapshots) .snapshotInventoryMismatch
  -- Freeze record must equal the manifest's declared frozen closure identity.
  let faults :=
    match m.records.find? (fun r => r.commit == m.freeze) with
    | none => faults
    | some fr =>
        pushFault faults
          (fr.judgingClosureIdentity != m.frozenJudgingClosureIdentity)
          .freezeClosureMismatch
  let faults :=
    m.records.foldl (fun acc r =>
      if r.judgingClosureIdentity == m.frozenJudgingClosureIdentity then acc
      else acc ++ [.judgingClosureDrift r.commit]) faults
  match c.claimedKind with
  | .retrospectiveFreezeIntegrity =>
      pushFault faults c.prospectiveProtocol.isSome .retrospectiveCarriesProtocol
  | .prospective =>
      match c.prospectiveProtocol with
      | none => faults ++ [.prospectiveWithoutProtocol]
      | some p =>
          pushFault faults (!p.wellFormed) .malformedProspectiveProtocol

def validateFreezeIntegrity (c : FreezeIntegrityCandidate) :
    ChronologyValidationResult FreezeIntegrityFault ValidatedFreezeIntegrity :=
  match freezeIntegrityDiagnose c with
  | [] =>
      let kind : ChronologyKind :=
        match c.claimedKind, c.prospectiveProtocol with
        | .prospective, some p => .prospective p
        | _, _ => .retrospectiveFreezeIntegrity
      .ok ⟨c, kind⟩
  | diagnosed => .faults diagnosed

/-! ## Cycle registry validation -/

private def cycleRegistryDiagnose (c : CycleRegistryCandidate) :
    List CycleRegistryFault :=
  let faults : List CycleRegistryFault := []
  let faults := pushFault faults c.importedDescriptors.isEmpty .emptyInventory
  let invIds := c.importedDescriptors.map (·.id)
  let faults :=
    c.importedDescriptors.foldl (fun acc d =>
      let acc := pushFault acc (!d.wellFormed) (.malformedDescriptor d.id)
      pushFault acc (d.wellFormed && !d.target.isPromoted)
        (.unpromotedDescriptorTarget d.id)) faults
  let faults :=
    if listNodup invIds then faults
    else
      invIds.foldl (fun acc id =>
        if invIds.count id > 1 && !(acc.contains (.duplicateInventoryId id)) then
          acc ++ [.duplicateInventoryId id]
        else acc) faults
  let entryIds := c.entries.map (·.descriptor.id)
  let entryDescriptors := c.entries.map (·.descriptor)
  let faults :=
    if listNodup entryIds then faults
    else
      entryIds.foldl (fun acc id =>
        if entryIds.count id > 1 && !(acc.contains (.duplicateEntryId id)) then
          acc ++ [.duplicateEntryId id]
        else acc) faults
  let faults :=
    pushFault faults (!listSetEq c.importedDescriptors entryDescriptors)
      .descriptorEntryMismatch
  let faults :=
    c.importedDescriptors.foldl (fun acc d =>
      if entryDescriptors.contains d then acc
      else acc ++ [.omittedInventoryEvent d.id]) faults
  let faults :=
    entryDescriptors.foldl (fun acc d =>
      if c.importedDescriptors.contains d then acc
      else acc ++ [.extraRegistryEvent d.id]) faults
  c.entries.foldl (fun acc e =>
    match e.descriptor.outcome with
    | .promoted =>
        match e.freezeIntegrity with
        | none => acc ++ [.promotedMissingFreezeIntegrity e.descriptor.id]
        | some fi =>
            let acc :=
              pushFault acc (fi.manifest.target != e.descriptor.target)
                (.promotedTargetMismatch e.descriptor.id)
            let acc :=
              pushFault acc (fi.manifest.profile != e.descriptor.profile)
                (.promotedProfileMismatch e.descriptor.id)
            let acc :=
              pushFault acc
                (fi.manifest.frozenJudgingClosureIdentity !=
                  e.descriptor.expectedFrozenJudgingClosureIdentity)
                (.promotedClosureMismatch e.descriptor.id)
            let acc :=
              if fi.manifest.freeze == e.descriptor.freeze &&
                  fi.manifest.candidate == e.descriptor.candidate then acc
              else acc ++ [.promotedCommitMismatch e.descriptor.id]
            match validateFreezeIntegrity fi with
            | .ok _ => acc
            | .faults fs =>
                fs.foldl (fun acc f =>
                  acc ++ [.promotedFreezeIntegrityFault e.descriptor.id f]) acc
    | .rejected | .abandoned =>
        match e.freezeIntegrity with
        | none => acc
        | some _ => acc ++ [.nonPromotedCarriesChronology e.descriptor.id]) faults

def validateCycleRegistry (c : CycleRegistryCandidate) :
    ChronologyValidationResult CycleRegistryFault ValidatedCycleRegistry :=
  match cycleRegistryDiagnose c with
  | [] => .ok ⟨c⟩
  | diagnosed => .faults diagnosed

/-! ## Toy fixtures -/

private def toyProv : FieldProvenance := .imported sampleImported

private def toyNode
    (id : String) (cat : JudgingInputCategory) (content : String)
    (deps : List String) : JudgingInputNode :=
  { id := { value := id },
    category := cat,
    content := digest content,
    dependencies := deps.map fun d => { value := d },
    provenance := toyProv }

/-- Minimal closed judging-input graph covering every required category. -/
def toyJudgingNodes : List JudgingInputNode :=
  [ toyNode "n.claimRenderer" .claimRenderer "cr" ["n.toolchain"]
  , toyNode "n.toolchain" .toolchain "tc" ["n.envPolicy"]
  , toyNode "n.envPolicy" .environmentPolicy "ep" ["n.helper"]
  , toyNode "n.helper" .importedHelper "ih" ["n.cal"]
  , toyNode "n.cal" .calibrationPolicy "cp" ["n.val"]
  , toyNode "n.val" .validator "vl" ["n.scn"]
  , toyNode "n.scn" .scenarioGenerator "sg" ["n.rel"]
  , toyNode "n.rel" .relation "rl" ["n.adp"]
  , toyNode "n.adp" .adapter "ad" ["n.bnd"]
  , toyNode "n.bnd" .boundaryDenotation "bd" ["n.req"]
  , toyNode "n.req" .requirementDenotation "rd" ["n.tgt"]
  , toyNode "n.tgt" .targetProfileDecision "tp" [] ]

def toyJudgingRoots : List JudgingNodeId := [{ value := "n.claimRenderer" }]

def toyJudgingCandidate : JudgingInputClosureCandidate where
  schemaVersion := judgingInputClosureSchemaVersion
  nodes := toyJudgingNodes
  roots := toyJudgingRoots
  discoveredInventory := toyJudgingNodes.map (·.id)

def toyJudgingClosureIdentity : ContentDigest :=
  computeJudgingClosureIdentity toyJudgingCandidate

/-- Dense DAG: each node depends on all later nodes. Edge count ≫ node count;
fuel must include dependency edges or reachability fails. -/
def denseJudgingCandidate : JudgingInputClosureCandidate :=
  let nodes :=
    [ toyNode "n.d0" .claimRenderer "0"
        ["n.d1", "n.d2", "n.d3", "n.d4", "n.d5", "n.d6", "n.d7", "n.d8", "n.d9",
         "n.d10", "n.d11"]
    , toyNode "n.d1" .toolchain "1"
        ["n.d2", "n.d3", "n.d4", "n.d5", "n.d6", "n.d7", "n.d8", "n.d9", "n.d10",
         "n.d11"]
    , toyNode "n.d2" .environmentPolicy "2"
        ["n.d3", "n.d4", "n.d5", "n.d6", "n.d7", "n.d8", "n.d9", "n.d10", "n.d11"]
    , toyNode "n.d3" .importedHelper "3"
        ["n.d4", "n.d5", "n.d6", "n.d7", "n.d8", "n.d9", "n.d10", "n.d11"]
    , toyNode "n.d4" .calibrationPolicy "4"
        ["n.d5", "n.d6", "n.d7", "n.d8", "n.d9", "n.d10", "n.d11"]
    , toyNode "n.d5" .validator "5"
        ["n.d6", "n.d7", "n.d8", "n.d9", "n.d10", "n.d11"]
    , toyNode "n.d6" .scenarioGenerator "6"
        ["n.d7", "n.d8", "n.d9", "n.d10", "n.d11"]
    , toyNode "n.d7" .relation "7" ["n.d8", "n.d9", "n.d10", "n.d11"]
    , toyNode "n.d8" .adapter "8" ["n.d9", "n.d10", "n.d11"]
    , toyNode "n.d9" .boundaryDenotation "9" ["n.d10", "n.d11"]
    , toyNode "n.d10" .requirementDenotation "10" ["n.d11"]
    , toyNode "n.d11" .targetProfileDecision "11" [] ]
  { schemaVersion := judgingInputClosureSchemaVersion
    nodes := nodes
    roots := [{ value := "n.d0" }]
    discoveredInventory := nodes.map (·.id) }

def toyCommitA : CommitId := { value := "commit-a-freeze" }
def toyCommitB : CommitId := { value := "commit-b-middle" }
def toyCommitC : CommitId := { value := "commit-c-candidate" }

def toyProfileId : ProfileId := { value := "profile.toy.chronology" }

def toyChronologyTarget : TargetIdentity where
  id := { value := "target.toy.chronology" }
  repository := "github.com/tinygrad/tinygrad"
  revision := digest "19c4d736-chronology"
  sourceClosure := { digest := digest "toy-chrono-source-closure" }
  profile := toyProfileId
  disposition := .promoted

def toyAncestryStable : List AncestryCommitRecord :=
  [ { commit := toyCommitA, parentsWithinCapture := [],
      judgingClosureIdentity := toyJudgingClosureIdentity }
  , { commit := toyCommitB, parentsWithinCapture := [toyCommitA],
      judgingClosureIdentity := toyJudgingClosureIdentity }
  , { commit := toyCommitC, parentsWithinCapture := [toyCommitB],
      judgingClosureIdentity := toyJudgingClosureIdentity } ]

def toyCapturedIds : List CommitId := [toyCommitA, toyCommitB, toyCommitC]

def toyAncestryManifest : AncestryManifest where
  captureIdentity := { value := "capture.toy.v1" }
  extractorIdentity := digest "extractor-toy"
  target := toyChronologyTarget
  profile := toyProfileId
  frozenJudgingClosureIdentity := toyJudgingClosureIdentity
  freeze := toyCommitA
  candidate := toyCommitC
  capturedCommitIds := toyCapturedIds
  records := toyAncestryStable

def toyFreezeIntegrity : FreezeIntegrityCandidate where
  claimedKind := .retrospectiveFreezeIntegrity
  prospectiveProtocol := none
  manifest := toyAncestryManifest
  suppliedSnapshots := toyAncestryStable

def toyProtocol : BlindFreezeProtocolRef where
  id := { value := "protocol.blind-freeze.v1" }
  digest := digest "protocol-hash"

def toyProspectiveFreezeIntegrity : FreezeIntegrityCandidate :=
  { toyFreezeIntegrity with
    claimedKind := .prospective
    prospectiveProtocol := some toyProtocol }

def toyPromotedDescriptor : CycleEventDescriptor where
  id := { value := "cycle.promoted.1" }
  target := toyChronologyTarget
  profile := toyProfileId
  expectedFrozenJudgingClosureIdentity := toyJudgingClosureIdentity
  outcome := .promoted
  freeze := toyCommitA
  candidate := toyCommitC

def toyRejectedDescriptor : CycleEventDescriptor where
  id := { value := "cycle.rejected.1" }
  target := toyChronologyTarget
  profile := toyProfileId
  expectedFrozenJudgingClosureIdentity := toyJudgingClosureIdentity
  outcome := .rejected
  freeze := toyCommitA
  candidate := toyCommitC

def toyAbandonedDescriptor : CycleEventDescriptor where
  id := { value := "cycle.abandoned.1" }
  target := toyChronologyTarget
  profile := toyProfileId
  expectedFrozenJudgingClosureIdentity := toyJudgingClosureIdentity
  outcome := .abandoned
  freeze := toyCommitA
  candidate := toyCommitC

def toyCycleRegistry : CycleRegistryCandidate where
  importedDescriptors :=
    [toyPromotedDescriptor, toyRejectedDescriptor, toyAbandonedDescriptor]
  entries :=
    [ { descriptor := toyPromotedDescriptor
      , freezeIntegrity := some toyFreezeIntegrity }
    , { descriptor := toyRejectedDescriptor, freezeIntegrity := none }
    , { descriptor := toyAbandonedDescriptor, freezeIntegrity := none } ]

/-! ## Positive executable checks -/

theorem toy_judging_closure_validates :
    (validateJudgingInputClosure toyJudgingCandidate).isOk = true := by
  native_decide

theorem toy_judging_closure_identity_is_derived :
    (match validateJudgingInputClosure toyJudgingCandidate with
     | .ok v => v.closureIdentity == toyJudgingClosureIdentity
     | .faults _ => false) = true := by
  native_decide

theorem dense_dag_judging_closure_validates :
    (validateJudgingInputClosure denseJudgingCandidate).isOk = true := by
  native_decide

theorem dense_dag_has_many_shared_edges :
    (judgingDependencyEdgeCount denseJudgingCandidate.nodes >
      denseJudgingCandidate.nodes.length) = true := by
  native_decide

theorem toy_freeze_integrity_validates :
    (validateFreezeIntegrity toyFreezeIntegrity).isOk = true := by
  native_decide

theorem toy_prospective_freeze_integrity_validates :
    (validateFreezeIntegrity toyProspectiveFreezeIntegrity).isOk = true := by
  native_decide

theorem toy_cycle_registry_validates :
    (validateCycleRegistry toyCycleRegistry).isOk = true := by
  native_decide

theorem git_ancestry_alone_yields_retrospective :
    (match validateFreezeIntegrity toyFreezeIntegrity with
     | .ok v =>
         match v.kind with
         | .retrospectiveFreezeIntegrity => true
         | .prospective _ => false
     | .faults _ => false) = true := by
  native_decide

/-! ## Provenance and encoding injectivity checks -/

def provenanceMutatedImported : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with
    nodes :=
      toyJudgingNodes.map fun n =>
        { n with
          provenance :=
            .imported
              { id := { value := "import.upstream.closure" },
                sourceClosure := { digest := digest "src-hash-MUTATED" } } } }

theorem provenance_mutation_changes_closure_identity :
    (computeJudgingClosureIdentity provenanceMutatedImported !=
      toyJudgingClosureIdentity) = true := by
  native_decide

def provenanceMutatedDerived : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with
    nodes :=
      toyJudgingNodes.map fun n =>
        { n with
          provenance :=
            .derived
              { id := { value := "derived.status" },
                verifier := { value := "verifier.pilot-status" },
                inputClosure := digest "in-hash-MUTATED" } } }

def provenanceMutatedCalibrated : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with
    nodes :=
      toyJudgingNodes.map fun n =>
        { n with
          provenance :=
            .calibrated
              { id := { value := "cal.helpers" },
                campaign := digest "cal-hash",
                faultModel := "missing-public-name-MUTATED" } } }

def provenanceMutatedJudgment : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with
    nodes :=
      toyJudgingNodes.map fun n =>
        { n with
          provenance :=
            .judgment
              { id := { value := "judgment.scope.demo" },
                authority := "owner-MUTATED",
                scope := digest "scope-hash",
                invalidation := digest "inv-hash" } } }

theorem provenance_payload_mutations_all_change_identity :
    (computeJudgingClosureIdentity provenanceMutatedDerived !=
      toyJudgingClosureIdentity) &&
    (computeJudgingClosureIdentity provenanceMutatedCalibrated !=
      toyJudgingClosureIdentity) &&
    (computeJudgingClosureIdentity provenanceMutatedJudgment !=
      toyJudgingClosureIdentity) = true := by
  native_decide

/-- Delimiter-heavy IDs/content that would alias under naive concatenation. -/
def delimiterAliasLeft : JudgingInputClosureCandidate :=
  let nodes :=
    [ toyNode "a:b" .claimRenderer "c" ["x"]
    , toyNode "x" .toolchain "t" ["y"]
    , toyNode "y" .environmentPolicy "e" ["h"]
    , toyNode "h" .importedHelper "i" ["k"]
    , toyNode "k" .calibrationPolicy "p" ["v"]
    , toyNode "v" .validator "l" ["s"]
    , toyNode "s" .scenarioGenerator "g" ["r"]
    , toyNode "r" .relation "q" ["d"]
    , toyNode "d" .adapter "a" ["b"]
    , toyNode "b" .boundaryDenotation "n" ["q2"]
    , toyNode "q2" .requirementDenotation "m" ["t2"]
    , toyNode "t2" .targetProfileDecision "u" [] ]
  { schemaVersion := judgingInputClosureSchemaVersion
    nodes := nodes
    roots := [{ value := "a:b" }]
    discoveredInventory := nodes.map (·.id) }

def delimiterAliasRight : JudgingInputClosureCandidate :=
  let nodes :=
    [ toyNode "a" .claimRenderer "b:c" ["x"]
    , toyNode "x" .toolchain "t" ["y"]
    , toyNode "y" .environmentPolicy "e" ["h"]
    , toyNode "h" .importedHelper "i" ["k"]
    , toyNode "k" .calibrationPolicy "p" ["v"]
    , toyNode "v" .validator "l" ["s"]
    , toyNode "s" .scenarioGenerator "g" ["r"]
    , toyNode "r" .relation "q" ["d"]
    , toyNode "d" .adapter "a2" ["b"]
    , toyNode "b" .boundaryDenotation "n" ["q2"]
    , toyNode "q2" .requirementDenotation "m" ["t2"]
    , toyNode "t2" .targetProfileDecision "u" [] ]
  { schemaVersion := judgingInputClosureSchemaVersion
    nodes := nodes
    roots := [{ value := "a" }]
    discoveredInventory := nodes.map (·.id) }

theorem delimiter_heavy_structures_do_not_alias :
    (computeJudgingClosureIdentity delimiterAliasLeft !=
      computeJudgingClosureIdentity delimiterAliasRight) = true := by
  native_decide

/-! ## Negative executable checks -/

def missingCategoryCandidate : JudgingInputClosureCandidate :=
  let nodes := toyJudgingNodes.filter (fun n => n.category != .claimRenderer)
  { schemaVersion := judgingInputClosureSchemaVersion
    nodes := nodes
    roots := [{ value := "n.toolchain" }]
    discoveredInventory := nodes.map (·.id) }

theorem missing_category_rejected :
    (validateJudgingInputClosure missingCategoryCandidate).hasFault
      (.missingRequiredCategory .claimRenderer) = true := by
  native_decide

def missingDependencyCandidate : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with
    nodes :=
      toyJudgingNodes.map fun n =>
        if n.id.value == "n.req" then
          { n with dependencies := [{ value := "n.missing" }] }
        else n }

theorem missing_dependency_rejected :
    (validateJudgingInputClosure missingDependencyCandidate).hasFault
      (.unresolvedDependency { value := "n.req" } { value := "n.missing" }) = true := by
  native_decide

def missingNodeViaRootCandidate : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with roots := [{ value := "n.absent" }] }

theorem missing_root_node_rejected :
    (validateJudgingInputClosure missingNodeViaRootCandidate).hasFault
      (.unresolvedRoot { value := "n.absent" }) = true := by
  native_decide

def undeclaredDiscoveredCandidate : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with
    discoveredInventory :=
      toyJudgingCandidate.discoveredInventory ++ [{ value := "n.relocated" }] }

theorem undeclared_discovered_input_rejected :
    (validateJudgingInputClosure undeclaredDiscoveredCandidate).hasFault
      (.undeclaredDiscoveredInput { value := "n.relocated" }) = true := by
  native_decide

def unreachableNodeCandidate : JudgingInputClosureCandidate :=
  { toyJudgingCandidate with
    nodes :=
      toyJudgingNodes ++
        [toyNode "n.orphan" .claimRenderer "orphan" []]
    discoveredInventory :=
      toyJudgingCandidate.discoveredInventory ++ [{ value := "n.orphan" }] }

theorem unreachable_node_rejected :
    (validateJudgingInputClosure unreachableNodeCandidate).hasFault
      (.unreachableNode { value := "n.orphan" }) = true := by
  native_decide

def dependencyCycleCandidate : JudgingInputClosureCandidate :=
  let cycled :=
    toyJudgingNodes.map fun n =>
      if n.id.value == "n.tgt" then
        { n with dependencies := [{ value := "n.req" }] }
      else n
  { toyJudgingCandidate with
    nodes := cycled
    discoveredInventory := cycled.map (·.id) }

theorem dependency_cycle_rejected :
    (validateJudgingInputClosure dependencyCycleCandidate).hasFault
      (.dependencyCycle { value := "n.tgt" }) = true ∨
    (validateJudgingInputClosure dependencyCycleCandidate).hasFault
      (.dependencyCycle { value := "n.req" }) = true := by
  native_decide

/-- Parent-graph rewrite that skips an intermediate commit while the imported
captured inventory still names it. -/
def omittedAncestry : FreezeIntegrityCandidate :=
  let records :=
    [ { commit := toyCommitA, parentsWithinCapture := [],
        judgingClosureIdentity := toyJudgingClosureIdentity }
    , { commit := toyCommitC, parentsWithinCapture := [toyCommitA],
        judgingClosureIdentity := toyJudgingClosureIdentity } ]
  { toyFreezeIntegrity with
    manifest :=
      { toyAncestryManifest with
        capturedCommitIds := [toyCommitA, toyCommitB, toyCommitC]
        records := records }
    suppliedSnapshots := records }

theorem omitted_ancestry_commit_rejected :
    (validateFreezeIntegrity omittedAncestry).hasFault
      (.omittedAncestryCommit toyCommitB) = true ∨
    (validateFreezeIntegrity omittedAncestry).hasFault
      .captureInventoryMismatch = true := by
  native_decide

/-- Intermediate mutation then reversion: endpoints match, middle differs. -/
def mutationReversionAncestry : FreezeIntegrityCandidate :=
  let mutated := digest "mutated-judging-closure"
  let records :=
    [ { commit := toyCommitA, parentsWithinCapture := [],
        judgingClosureIdentity := toyJudgingClosureIdentity }
    , { commit := toyCommitB, parentsWithinCapture := [toyCommitA],
        judgingClosureIdentity := mutated }
    , { commit := toyCommitC, parentsWithinCapture := [toyCommitB],
        judgingClosureIdentity := toyJudgingClosureIdentity } ]
  { toyFreezeIntegrity with
    manifest := { toyAncestryManifest with records := records }
    suppliedSnapshots := records }

theorem intermediate_mutation_reversion_rejected :
    (validateFreezeIntegrity mutationReversionAncestry).hasFault
      (.judgingClosureDrift toyCommitB) = true := by
  native_decide

def omittedFailedCycle : CycleRegistryCandidate :=
  { importedDescriptors := [toyPromotedDescriptor, toyRejectedDescriptor]
    entries :=
      [ { descriptor := toyPromotedDescriptor
        , freezeIntegrity := some toyFreezeIntegrity } ] }

theorem omitted_failed_cycle_rejected :
    (validateCycleRegistry omittedFailedCycle).hasFault
      (.omittedInventoryEvent { value := "cycle.rejected.1" }) = true ∨
    (validateCycleRegistry omittedFailedCycle).hasFault
      .descriptorEntryMismatch = true := by
  native_decide

def duplicateCycle : CycleRegistryCandidate :=
  { importedDescriptors := [toyPromotedDescriptor, toyPromotedDescriptor]
    entries :=
      [ { descriptor := toyPromotedDescriptor
        , freezeIntegrity := some toyFreezeIntegrity }
      , { descriptor := toyPromotedDescriptor
        , freezeIntegrity := some toyFreezeIntegrity } ] }

theorem duplicate_cycle_rejected :
    (validateCycleRegistry duplicateCycle).hasFault
      (.duplicateEntryId { value := "cycle.promoted.1" }) = true ∨
    (validateCycleRegistry duplicateCycle).hasFault
      (.duplicateInventoryId { value := "cycle.promoted.1" }) = true := by
  native_decide

def prospectiveWithoutProtocol : FreezeIntegrityCandidate :=
  { toyFreezeIntegrity with
    claimedKind := .prospective
    prospectiveProtocol := none }

theorem prospective_without_protocol_rejected :
    (validateFreezeIntegrity prospectiveWithoutProtocol).hasFault
      .prospectiveWithoutProtocol = true := by
  native_decide

/-- Relabeled outcome: imported descriptor says rejected; entry claims promoted. -/
def relabeledOutcomeCycle : CycleRegistryCandidate :=
  let relabeled := { toyRejectedDescriptor with outcome := .promoted }
  { importedDescriptors := [toyRejectedDescriptor]
    entries :=
      [ { descriptor := relabeled
        , freezeIntegrity := some toyFreezeIntegrity } ] }

theorem relabeled_outcome_rejected :
    (validateCycleRegistry relabeledOutcomeCycle).hasFault
      .descriptorEntryMismatch = true ∨
    (validateCycleRegistry relabeledOutcomeCycle).hasFault
      (.extraRegistryEvent { value := "cycle.rejected.1" }) = true ∨
    (validateCycleRegistry relabeledOutcomeCycle).hasFault
      (.omittedInventoryEvent { value := "cycle.rejected.1" }) = true := by
  native_decide

/-- Promoted entry attaches a valid but unrelated freeze/candidate chronology. -/
def reboundCommitCycle : CycleRegistryCandidate :=
  let otherFreeze : CommitId := { value := "commit-other-freeze" }
  let otherCand : CommitId := { value := "commit-other-cand" }
  let otherRecords : List AncestryCommitRecord :=
    [ { commit := otherFreeze, parentsWithinCapture := [],
        judgingClosureIdentity := toyJudgingClosureIdentity }
    , { commit := otherCand, parentsWithinCapture := [otherFreeze],
        judgingClosureIdentity := toyJudgingClosureIdentity } ]
  let unrelated : FreezeIntegrityCandidate :=
    { toyFreezeIntegrity with
      manifest :=
        { toyAncestryManifest with
          freeze := otherFreeze
          candidate := otherCand
          capturedCommitIds := [otherFreeze, otherCand]
          records := otherRecords }
      suppliedSnapshots := otherRecords }
  { importedDescriptors := [toyPromotedDescriptor]
    entries :=
      [ { descriptor := toyPromotedDescriptor
        , freezeIntegrity := some unrelated } ] }

theorem rebound_unrelated_commits_rejected :
    (validateCycleRegistry reboundCommitCycle).hasFault
      (.promotedCommitMismatch { value := "cycle.promoted.1" }) = true := by
  native_decide

def nonPromotedCarriesChronologyCycle : CycleRegistryCandidate :=
  { importedDescriptors := [toyRejectedDescriptor]
    entries :=
      [ { descriptor := toyRejectedDescriptor
        , freezeIntegrity := some toyFreezeIntegrity } ] }

theorem non_promoted_carrying_chronology_rejected :
    (validateCycleRegistry nonPromotedCarriesChronologyCycle).hasFault
      (.nonPromotedCarriesChronology { value := "cycle.rejected.1" }) = true := by
  native_decide

/-- Same commits, but chronology target has a different revision/source closure. -/
def substitutedTargetRevisionCycle : CycleRegistryCandidate :=
  let otherTarget : TargetIdentity :=
    { toyChronologyTarget with
      revision := digest "other-revision"
      sourceClosure := { digest := digest "other-source-closure" } }
  let substituted : FreezeIntegrityCandidate :=
    { toyFreezeIntegrity with
      manifest := { toyAncestryManifest with target := otherTarget } }
  { importedDescriptors := [toyPromotedDescriptor]
    entries :=
      [ { descriptor := toyPromotedDescriptor
        , freezeIntegrity := some substituted } ] }

theorem substituted_target_revision_rejected :
    (validateCycleRegistry substitutedTargetRevisionCycle).hasFault
      (.promotedTargetMismatch { value := "cycle.promoted.1" }) = true := by
  native_decide

/-- Same commits, but chronology uses a different profile scope. -/
def substitutedProfileCycle : CycleRegistryCandidate :=
  let otherProfile : ProfileId := { value := "profile.other.chronology" }
  let otherTarget : TargetIdentity :=
    { toyChronologyTarget with profile := otherProfile }
  let substituted : FreezeIntegrityCandidate :=
    { toyFreezeIntegrity with
      manifest :=
        { toyAncestryManifest with
          target := otherTarget
          profile := otherProfile } }
  { importedDescriptors := [toyPromotedDescriptor]
    entries :=
      [ { descriptor := toyPromotedDescriptor
        , freezeIntegrity := some substituted } ] }

theorem substituted_profile_rejected :
    (validateCycleRegistry substitutedProfileCycle).hasFault
      (.promotedProfileMismatch { value := "cycle.promoted.1" }) = true := by
  native_decide

/-- Stable arbitrary closure token on all records; commits match descriptor. -/
def arbitraryStableClosureTokenCycle : CycleRegistryCandidate :=
  let arbitrary := digest "arbitrary-stable-closure-token"
  let records : List AncestryCommitRecord :=
    [ { commit := toyCommitA, parentsWithinCapture := [],
        judgingClosureIdentity := arbitrary }
    , { commit := toyCommitB, parentsWithinCapture := [toyCommitA],
        judgingClosureIdentity := arbitrary }
    , { commit := toyCommitC, parentsWithinCapture := [toyCommitB],
        judgingClosureIdentity := arbitrary } ]
  let substituted : FreezeIntegrityCandidate :=
    { toyFreezeIntegrity with
      manifest :=
        { toyAncestryManifest with
          frozenJudgingClosureIdentity := arbitrary
          records := records }
      suppliedSnapshots := records }
  { importedDescriptors := [toyPromotedDescriptor]
    entries :=
      [ { descriptor := toyPromotedDescriptor
        , freezeIntegrity := some substituted } ] }

theorem arbitrary_stable_closure_token_rejected :
    (validateCycleRegistry arbitraryStableClosureTokenCycle).hasFault
      (.promotedClosureMismatch { value := "cycle.promoted.1" }) = true := by
  native_decide

def emptyCapturedInventoryAncestry : FreezeIntegrityCandidate :=
  { toyFreezeIntegrity with
    manifest :=
      { toyAncestryManifest with
        capturedCommitIds := []
        records := [] }
    suppliedSnapshots := [] }

theorem empty_captured_inventory_rejected :
    (validateFreezeIntegrity emptyCapturedInventoryAncestry).hasFault
      .emptyCapturedInventory = true := by
  native_decide

def malformedCapturedCommitIdAncestry : FreezeIntegrityCandidate :=
  let bad : CommitId := { value := "   " }
  { toyFreezeIntegrity with
    manifest :=
      { toyAncestryManifest with
        capturedCommitIds := [toyCommitA, bad, toyCommitC] } }

theorem malformed_captured_commit_id_rejected :
    (validateFreezeIntegrity malformedCapturedCommitIdAncestry).hasFault
      (.malformedCapturedCommitId { value := "   " }) = true := by
  native_decide

end Tgrad.Contract

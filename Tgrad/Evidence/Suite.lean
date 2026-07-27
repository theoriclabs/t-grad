/-! # Tgrad.Evidence.Suite — non-normative suite execution facts

These records describe what a revision-bound test observer saw.  They are
deliberately not `Observation`s: a suite case is not a requirement, and no
case-to-requirement witness map was frozen before the current Tgrad run.

This module therefore imports no requirements or product module and exposes no
conversion to conformance, coverage, promotion, or work state.  A later,
reviewed witness layer may relate exact case descriptors to interpreted world
requirements without changing these facts.
-/

namespace Tgrad.Evidence.Suite

inductive SubjectKind where
  | upstream
  | tgrad
  deriving DecidableEq, BEq, Repr, Inhabited

inductive SuitePhase where
  | collection
  | execution
  | setupTeardown
  | environment
  | oracle
  | mixed
  | harness
  | noTests
  deriving DecidableEq, BEq, Repr, Inhabited

inductive FileOutcome where
  | passed
  | blockedProductSurface
  | blockedEnvironment
  | nonconforming
  | unobservedEnvironment
  | unobservedUpstream
  | mixed
  | collectionMismatch
  | collectionError
  | timeout
  | empty
  | verifierError
  deriving DecidableEq, BEq, Repr, Inhabited

structure TestCounts where
  collected : Nat
  passed : Nat
  failed : Nat
  skipped : Nat
  xfailed : Nat
  xpassed : Nat
  errors : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure SuiteAggregate where
  files : Nat
  passed : Nat
  failed : Nat
  skipped : Nat
  xfailed : Nat
  xpassed : Nat
  errors : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure FileFact where
  path : String
  phase : SuitePhase
  outcome : FileOutcome
  reasonCodes : List String
  counts : TestCounts
  nodeIdCount : Nat
  caseCount : Nat
  nodeIdManifestHash : String
  collectionCaseManifestHash : String
  terminalCaseManifestHash : String
  caseOutcomeManifestHash : String
  sourceHash : String
  reportHash : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure OracleCaseFacts where
  upstreamEligible : Nat
  upstreamUnobserved : Nat
  subjectMatched : Nat
  subjectPassed : Nat
  subjectNonpassing : Nat
  subjectMissing : Nat
  subjectDescriptorMismatches : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure UpstreamBaselineBinding where
  artifactHash : String
  resultId : String
  runArtifactId : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure SuiteRunFact where
  schemaVersion : Nat
  subjectKind : SubjectKind
  evidencePath : String
  evidenceHash : String
  resultId : String
  runArtifactId : String
  scenarioId : String
  upstreamRevision : String
  upstreamTree : String
  subjectRevision : String
  subjectTree : String
  subjectDirty : Bool
  verifierHash : String
  reporterHash : String
  contractHash : String
  environmentHash : String
  environmentFactsHash : String
  backend : String
  hardwareIdentityHash : String
  adapterHash : Option String
  runtimeArtifactHash : Option String
  productSourcesHash : Option String
  upstreamBaseline : Option UpstreamBaselineBinding
  aggregate : SuiteAggregate
  files : List FileFact
  oracleCases : Option OracleCaseFacts
  deriving DecidableEq, BEq, Repr, Inhabited

def FileFact.wellFormed (fact : FileFact) : Bool :=
  !fact.path.trimAscii.isEmpty &&
  !fact.nodeIdManifestHash.trimAscii.isEmpty &&
  !fact.collectionCaseManifestHash.trimAscii.isEmpty &&
  !fact.terminalCaseManifestHash.trimAscii.isEmpty &&
  !fact.caseOutcomeManifestHash.trimAscii.isEmpty &&
  !fact.sourceHash.trimAscii.isEmpty &&
  !fact.reportHash.trimAscii.isEmpty

def SuiteRunFact.wellFormed (run : SuiteRunFact) : Bool :=
  run.schemaVersion > 0 &&
  !run.evidencePath.trimAscii.isEmpty &&
  !run.evidenceHash.trimAscii.isEmpty &&
  !run.resultId.trimAscii.isEmpty &&
  !run.runArtifactId.trimAscii.isEmpty &&
  !run.scenarioId.trimAscii.isEmpty &&
  !run.upstreamRevision.trimAscii.isEmpty &&
  !run.upstreamTree.trimAscii.isEmpty &&
  !run.subjectRevision.trimAscii.isEmpty &&
  !run.subjectTree.trimAscii.isEmpty &&
  !run.verifierHash.trimAscii.isEmpty &&
  !run.reporterHash.trimAscii.isEmpty &&
  !run.contractHash.trimAscii.isEmpty &&
  !run.environmentHash.trimAscii.isEmpty &&
  !run.environmentFactsHash.trimAscii.isEmpty &&
  !run.backend.trimAscii.isEmpty &&
  !run.hardwareIdentityHash.trimAscii.isEmpty &&
  run.aggregate.files == run.files.length &&
  run.files.all FileFact.wellFormed &&
  (run.files.map FileFact.path).Nodup

end Tgrad.Evidence.Suite

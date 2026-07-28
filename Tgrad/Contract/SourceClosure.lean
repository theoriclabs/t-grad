import Tgrad.Contract.Identity

/-! # Tgrad.Contract.SourceClosure — typed imported source boundary

This module validates the representation imported from the canonical
source-closure JSON.  It does not authenticate the foreign bytes: Python must
re-enumerate the pinned Git tree, read every blob from the object database, and
byte-compare regenerated JSON before this Lean projection is checked.

The validated boundary is private.  Nothing here constructs target promotion,
catalog closure, requirement discharge, completion, runtime parity, or build
attestation.
-/

namespace Tgrad.Contract

def sourceClosureSchema : String := "tgrad.source-closure.v1"
def sourceClosureRepository : String := "github.com/tinygrad/tinygrad"
def sourceClosureRevision : String := "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
def sourceClosureTree : String := "855cca3b00c38841a6d3a043284f3a2ca696d4b0"
def sourceClosureObjectFormat : String := "sha1"
def sourceClosureExtractorPolicy : String := "tgrad.source-closure.extractor-policy.v1"
def sourceClosureCanonicalizer : String := "tgrad.canonical-json.sorted-compact-utf8.v1"
def sourceClosureParserPolicy : String := "cpython-ast.structural-signature.v1"
def sourceClosurePythonImplementation : String := "cpython"
def sourceClosurePythonMin : String := "3.12"
def sourceClosurePythonMax : String := "3.14"
def sourceClosureParserGrammar : String := "3.12"
def sourceClosureApiTestPolicy : String :=
  "tinygrad.api-surface-tests.null-unit-backend.non-init.v1"

/-- Exact canonical source-closure identity expected by this contract. -/
def expectedSourceClosureSha256 : String :=
  "ae93a447ecd98b7bcb9abd3c282e46c56a1cf313b13648253c276a91a5eb1c73"

/-- Exact local extractor-source bundle expected by the pinned artifact. -/
def expectedExtractorSourceBundleSha256 : String :=
  "71fe03337e22fbf91c30b0354143af05101647c19cf3825e63c7e0ec026d0052"

inductive SourceCategoryId where
  | apiSurfaceTests
  | backends
  | dtypes
  | extractor
  | ops
  | tensorApi
  | tinygradPython
  | upstreamTests
  deriving DecidableEq, BEq, Repr, Inhabited

def SourceCategoryId.tag : SourceCategoryId → String
  | .apiSurfaceTests => "api_surface_tests"
  | .backends => "backends"
  | .dtypes => "dtypes"
  | .extractor => "extractor"
  | .ops => "ops"
  | .tensorApi => "tensor_api"
  | .tinygradPython => "tinygrad_python"
  | .upstreamTests => "upstream_tests"

def allSourceCategoryIds : List SourceCategoryId :=
  [.apiSurfaceTests, .backends, .dtypes, .extractor, .ops, .tensorApi,
   .tinygradPython, .upstreamTests]

inductive ExtractionStatus where
  | complete
  | notExtracted
  deriving DecidableEq, BEq, Repr, Inhabited

inductive SourceKind where
  | foreignGit
  | localExtractor
  deriving DecidableEq, BEq, Repr, Inhabited

inductive ExtractionLimit where
  | backendExecution
  | catalogClosure
  | docsAnchors
  | officialWorkloads
  | publicExportSemantics
  | pytestNodeIds
  | requirementInterpretation
  | requirementRows590
  | runtimeBuildAttestation
  | runtimeParity
  | runtimeResolvedTensorBehavior
  | scenarioAdequacy
  | targetPromotion
  deriving DecidableEq, BEq, Repr, Inhabited

def allExtractionLimits : List ExtractionLimit :=
  [.backendExecution, .catalogClosure, .docsAnchors, .officialWorkloads,
   .publicExportSemantics, .pytestNodeIds, .requirementInterpretation,
   .requirementRows590, .runtimeBuildAttestation, .runtimeParity,
   .runtimeResolvedTensorBehavior, .scenarioAdequacy, .targetPromotion]

structure RawTargetIdentity where
  repository : String
  revision : String
  tree : String
  objectFormat : String
  disposition : TargetDisposition
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawRepositoryInventory where
  entryCount : Nat
  inventorySha256 : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawExtractorSourceIdentity where
  path : String
  byteSize : Nat
  sha256 : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawExtractorIdentity where
  policyId : String
  canonicalizerId : String
  parserPolicyId : String
  parserGrammarFeature : String
  pythonImplementation : String
  pythonMajorMinorMin : String
  pythonMajorMinorMax : String
  sourceFiles : List RawExtractorSourceIdentity
  sourceBundleSha256 : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawSourceFileIdentity where
  path : String
  mode : String
  blobOid : String
  byteSize : Nat
  sha256 : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawCategoryInventory where
  id : SourceCategoryId
  status : ExtractionStatus
  sourceKind : SourceKind
  paths : List String
  fileCount : Nat
  inventorySha256 : String
  deriving DecidableEq, BEq, Repr, Inhabited

inductive TensorDeclarationKind where
  | method
  | property
  deriving DecidableEq, BEq, Repr, Inhabited

inductive TensorSourceScopeStatus where
  | selectedClass
  | explicitNoClass
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawTensorSourceScope where
  source : String
  declaringClass : Option String
  status : TensorSourceScopeStatus
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawTensorDeclaration where
  source : String
  declaringClass : String
  kind : TensorDeclarationKind
  name : String
  structuralSignature : String
  signatureSha256 : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawTensorApiInventory where
  sources : List String
  sourceScopes : List RawTensorSourceScope
  declarations : List RawTensorDeclaration
  directMethods : List String
  methodNames : List String
  propertyNames : List String
  declarationCount : Nat
  directMethodCount : Nat
  methodCount : Nat
  propertyCount : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawNamedDeclaration where
  name : String
  source : String
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawOpsInventory where
  source : String
  declarations : List RawNamedDeclaration
  count : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawDtypeInventory where
  source : String
  declarations : List RawNamedDeclaration
  count : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawBackendInventory where
  declarations : List RawNamedDeclaration
  count : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

structure RawTestInventory where
  policyId : String
  groups : List String
  sources : List String
  count : Nat
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Public imported premises.  Generated Lean constructs only this raw form. -/
structure SourceClosureCandidate where
  schema : String
  target : RawTargetIdentity
  repository : RawRepositoryInventory
  extractor : RawExtractorIdentity
  files : List RawSourceFileIdentity
  categories : List RawCategoryInventory
  tensorApi : RawTensorApiInventory
  ops : RawOpsInventory
  dtypes : RawDtypeInventory
  backends : RawBackendInventory
  tests : RawTestInventory
  limits : List ExtractionLimit
  closureSha256 : String
  deriving DecidableEq, BEq, Repr, Inhabited

/-! ## Shape smart constructors -/

structure Sha256Value where
  private mk ::
  value : String
  deriving DecidableEq, BEq, Repr

structure GitObjectId where
  private mk ::
  value : String
  deriving DecidableEq, BEq, Repr

structure RepositoryPath where
  private mk ::
  value : String
  deriving DecidableEq, BEq, Repr

private def lowerHexChars : String := "0123456789abcdef"

def isLowerHexOfLength (length : Nat) (value : String) : Bool :=
  value.length == length && value.toList.all fun character =>
    lowerHexChars.toList.contains character

def validSha256 (value : String) : Bool := isLowerHexOfLength 64 value

def validGitOid (objectFormat value : String) : Bool :=
  if objectFormat == "sha1" then isLowerHexOfLength 40 value
  else if objectFormat == "sha256" then isLowerHexOfLength 64 value
  else false

def validRepositoryPath (value : String) : Bool :=
  !value.isEmpty &&
  !(value.startsWith "/") &&
  !(value.contains '\\') &&
  (value.splitOn "/").all fun part =>
    !part.isEmpty && part != "." && part != ".."

def Sha256Value.ofString? (value : String) : Option Sha256Value :=
  if validSha256 value then some ⟨value⟩ else none

def GitObjectId.ofString? (objectFormat value : String) : Option GitObjectId :=
  if validGitOid objectFormat value then some ⟨value⟩ else none

def RepositoryPath.ofString? (value : String) : Option RepositoryPath :=
  if validRepositoryPath value then some ⟨value⟩ else none

/-! ## Validation -/

inductive SourceClosureFault where
  | wrongSchema
  | wrongTarget
  | malformedRepositoryInventory
  | malformedExtractorIdentity
  | staleExtractorIdentity
  | malformedFileIdentity
  | duplicateFilePath
  | malformedCategory
  | wrongCategorySet
  | categoryCrossReference
  | categoryPredicateMismatch
  | conflatedTestCategories
  | malformedTensorApi
  | tensorSourceTrap
  | malformedOps
  | opsSourceTrap
  | malformedDtypes
  | malformedBackends
  | malformedTests
  | wrongLimitSet
  | malformedClosureDigest
  | wrongClosureDigest
  deriving DecidableEq, BEq, Repr, Inhabited

inductive SourceClosureValidationResult (ok : Type) where
  | ok (value : ok)
  | faults (diagnosed : List SourceClosureFault)
  deriving Repr

def SourceClosureValidationResult.isOk {ok : Type} :
    SourceClosureValidationResult ok → Bool
  | .ok _ => true
  | .faults _ => false

def SourceClosureValidationResult.hasFault {ok : Type}
    (result : SourceClosureValidationResult ok) (fault : SourceClosureFault) : Bool :=
  match result with
  | .ok _ => false
  | .faults diagnosed => diagnosed.contains fault

structure ValidatedSourceClosure where
  private mk ::
  candidate : SourceClosureCandidate
  deriving Repr

private def pushFault
    (faults : List SourceClosureFault) (condition : Bool) (fault : SourceClosureFault) :
    List SourceClosureFault :=
  if condition then faults ++ [fault] else faults

private def category? (candidate : SourceClosureCandidate) (id : SourceCategoryId) :
    Option RawCategoryInventory :=
  candidate.categories.find? fun category => category.id == id

private def categoryPaths (candidate : SourceClosureCandidate) (id : SourceCategoryId) :
    List String :=
  match category? candidate id with
  | some category => category.paths
  | none => []

private def pathInFiles (candidate : SourceClosureCandidate) (path : String) : Bool :=
  candidate.files.any fun file => file.path == path

private def pathInExtractor (candidate : SourceClosureCandidate) (path : String) : Bool :=
  candidate.extractor.sourceFiles.any fun file => file.path == path

private def validMode (mode : String) : Bool :=
  mode.length == 6 && mode.toList.all ("01234567".contains ·)

private def fileIdentityWellFormed
    (objectFormat : String) (file : RawSourceFileIdentity) : Bool :=
  validRepositoryPath file.path && validMode file.mode &&
  validGitOid objectFormat file.blobOid && validSha256 file.sha256

private def extractorSourceWellFormed (file : RawExtractorSourceIdentity) : Bool :=
  validRepositoryPath file.path && validSha256 file.sha256

private def categoryWellFormed (category : RawCategoryInventory) : Bool :=
  category.status == .complete &&
  category.fileCount == category.paths.length &&
  validSha256 category.inventorySha256 &&
  category.paths.all validRepositoryPath && listNodup category.paths

private def categoryCrossReferences (candidate : SourceClosureCandidate) : Bool :=
  candidate.categories.all fun category =>
    if category.sourceKind == .foreignGit then
      category.paths.all (pathInFiles candidate)
    else
      category.paths.all (pathInExtractor candidate)

private def isTinygradPython (path : String) : Bool :=
  path.startsWith "tinygrad/" && path.endsWith ".py"

private def isUpstreamTest (path : String) : Bool :=
  path.startsWith "test/" && path.endsWith ".py"

private def isApiSurfaceTest (path : String) : Bool :=
  path.endsWith ".py" && !(path.endsWith "/__init__.py") &&
  (path.startsWith "test/null/" || path.startsWith "test/unit/" ||
   path.startsWith "test/backend/")

private def isTensorSource (path : String) : Bool :=
  path == "tinygrad/tensor.py" ||
  (path.startsWith "tinygrad/mixin/" && path.endsWith ".py" &&
   !(path.endsWith "/__init__.py"))

private def isBackendSource (path : String) : Bool :=
  path.startsWith "tinygrad/runtime/ops_" && path.endsWith ".py" &&
  ((path.drop "tinygrad/runtime/ops_".length).toString.splitOn "/").length == 1

private def categoryPredicatesMatch (candidate : SourceClosureCandidate) : Bool :=
  let allPaths := candidate.files.map (·.path)
  listSetEq (categoryPaths candidate .tinygradPython)
      (allPaths.filter isTinygradPython) &&
  listSetEq (categoryPaths candidate .upstreamTests)
      (allPaths.filter isUpstreamTest) &&
  listSetEq (categoryPaths candidate .apiSurfaceTests)
      (allPaths.filter isApiSurfaceTest) &&
  listSetEq (categoryPaths candidate .tensorApi)
      (allPaths.filter isTensorSource) &&
  listSetEq (categoryPaths candidate .backends)
      (allPaths.filter isBackendSource) &&
  listSetEq (categoryPaths candidate .ops) ["tinygrad/uop/__init__.py"] &&
  listSetEq (categoryPaths candidate .dtypes) ["tinygrad/dtype.py"] &&
  listSetEq (categoryPaths candidate .extractor)
      ["scripts/contract/generate_source_closure.py", "scripts/parity/ensure_oracle.py",
       "scripts/parity/extract_upstream.py",
       "scripts/parity/upstream_target.py"]

private def testCategoriesDistinct (candidate : SourceClosureCandidate) : Bool :=
  let allTests := categoryPaths candidate .upstreamTests
  let apiTests := categoryPaths candidate .apiSurfaceTests
  allTests.length == 331 && apiTests.length == 138 &&
  allTests != apiTests && apiTests.all allTests.contains &&
  match category? candidate .upstreamTests, category? candidate .apiSurfaceTests with
  | some allCategory, some apiCategory =>
      allCategory.inventorySha256 != apiCategory.inventorySha256
  | _, _ => false

private def expectedTensorScopes : List RawTensorSourceScope :=
  [{ source := "tinygrad/mixin/creation.py", declaringClass := some "CreationMixin",
     status := .selectedClass },
   { source := "tinygrad/mixin/dtype.py", declaringClass := some "DTypeMixin",
     status := .selectedClass },
   { source := "tinygrad/mixin/elementwise.py", declaringClass := some "ElementwiseMixin",
     status := .selectedClass },
   { source := "tinygrad/mixin/gradient.py", declaringClass := none,
     status := .explicitNoClass },
   { source := "tinygrad/mixin/movement.py", declaringClass := some "MovementMixin",
     status := .selectedClass },
   { source := "tinygrad/mixin/op.py", declaringClass := some "OpMixin",
     status := .selectedClass },
   { source := "tinygrad/mixin/rand.py", declaringClass := some "RandMixin",
     status := .selectedClass },
   { source := "tinygrad/mixin/reduce.py", declaringClass := some "ReduceMixin",
     status := .selectedClass },
   { source := "tinygrad/tensor.py", declaringClass := some "Tensor",
     status := .selectedClass }]

private def tensorDeclarationKey
    (declaration : RawTensorDeclaration) : TensorDeclarationKind × String × String × String :=
  (declaration.kind, declaration.name, declaration.source, declaration.declaringClass)

private def declarationScopeMatches
    (scopes : List RawTensorSourceScope) (declaration : RawTensorDeclaration) : Bool :=
  match scopes.find? fun scope => scope.source == declaration.source with
  | some scope =>
      scope.status == .selectedClass &&
      scope.declaringClass == some declaration.declaringClass
  | none => false

private def tensorApiWellFormed (candidate : SourceClosureCandidate) : Bool :=
  let tensor := candidate.tensorApi
  let derivedMethodNames :=
    ((tensor.declarations.filter fun declaration => declaration.kind == .method).map
      (·.name)).eraseDups
  let derivedPropertyNames :=
    ((tensor.declarations.filter fun declaration => declaration.kind == .property).map
      (·.name)).eraseDups
  let derivedDirectMethods :=
    ((tensor.declarations.filter fun declaration =>
      declaration.kind == .method && declaration.source == "tinygrad/tensor.py" &&
      declaration.declaringClass == "Tensor").map (·.name)).eraseDups
  listSetEq tensor.sources (categoryPaths candidate .tensorApi) &&
  listSetEq tensor.sourceScopes expectedTensorScopes &&
  listNodup (tensor.sourceScopes.map (·.source)) &&
  listSetEq (tensor.sourceScopes.map (·.source)) tensor.sources &&
  tensor.declarations.all (fun declaration =>
    !declaration.name.isEmpty && !declaration.declaringClass.isEmpty &&
    !declaration.structuralSignature.isEmpty && validSha256 declaration.signatureSha256 &&
    declarationScopeMatches tensor.sourceScopes declaration) &&
  listNodup (tensor.declarations.map tensorDeclarationKey) &&
  listNodup tensor.methodNames && listSetEq tensor.methodNames derivedMethodNames &&
  listNodup tensor.propertyNames && listSetEq tensor.propertyNames derivedPropertyNames &&
  listNodup (tensor.methodNames ++ tensor.propertyNames) &&
  listNodup tensor.directMethods && listSetEq tensor.directMethods derivedDirectMethods &&
  tensor.declarationCount == tensor.declarations.length &&
  tensor.directMethodCount == tensor.directMethods.length &&
  tensor.methodCount == tensor.methodNames.length &&
  tensor.propertyCount == tensor.propertyNames.length

private def tensorSourceTrapPasses (candidate : SourceClosureCandidate) : Bool :=
  candidate.tensorApi.sources.contains "tinygrad/tensor.py" &&
  candidate.tensorApi.sources.length > 1 &&
  candidate.tensorApi.declarationCount == 307 &&
  candidate.tensorApi.directMethodCount == 47 &&
  candidate.tensorApi.methodCount == 295 &&
  candidate.tensorApi.propertyCount == 5

private def namedDeclarationsWellFormed
    (declarations : List RawNamedDeclaration) : Bool :=
  declarations.all (fun declaration =>
      !declaration.name.isEmpty && validRepositoryPath declaration.source) &&
  listNodup (declarations.map (·.name))

private def opsWellFormed (candidate : SourceClosureCandidate) : Bool :=
  let ops := candidate.ops
  namedDeclarationsWellFormed ops.declarations &&
  ops.count == ops.declarations.length && ops.count == 82 &&
  ops.declarations.all fun declaration => declaration.source == ops.source

private def dtypesWellFormed (candidate : SourceClosureCandidate) : Bool :=
  let dtypes := candidate.dtypes
  namedDeclarationsWellFormed dtypes.declarations &&
  dtypes.count == dtypes.declarations.length && dtypes.count == 52 &&
  dtypes.declarations.all fun declaration => declaration.source == dtypes.source

private def backendsWellFormed (candidate : SourceClosureCandidate) : Bool :=
  let backends := candidate.backends
  namedDeclarationsWellFormed backends.declarations &&
  backends.count == backends.declarations.length && backends.count == 16 &&
  listSetEq (backends.declarations.map (·.source))
    (categoryPaths candidate .backends)

private def testsWellFormed (candidate : SourceClosureCandidate) : Bool :=
  candidate.tests.policyId == sourceClosureApiTestPolicy &&
  listSetEq candidate.tests.groups ["backend", "null", "unit"] &&
  listSetEq candidate.tests.sources (categoryPaths candidate .apiSurfaceTests) &&
  candidate.tests.count == candidate.tests.sources.length &&
  candidate.tests.count == 138

private def sourceClosureDiagnose (candidate : SourceClosureCandidate) :
    List SourceClosureFault :=
  let faults : List SourceClosureFault := []
  let faults := pushFault faults (candidate.schema != sourceClosureSchema) .wrongSchema
  let expectedTarget : RawTargetIdentity :=
    { repository := sourceClosureRepository
      revision := sourceClosureRevision
      tree := sourceClosureTree
      objectFormat := sourceClosureObjectFormat
      disposition := .extractedCandidate }
  let faults := pushFault faults (candidate.target != expectedTarget) .wrongTarget
  let faults := pushFault faults
    (candidate.repository.entryCount != candidate.files.length ||
     !validSha256 candidate.repository.inventorySha256)
    .malformedRepositoryInventory
  let faults := pushFault faults
    (!candidate.extractor.sourceFiles.all extractorSourceWellFormed ||
     !listNodup (candidate.extractor.sourceFiles.map (·.path)) ||
     !validSha256 candidate.extractor.sourceBundleSha256)
    .malformedExtractorIdentity
  let expectedExtractorPaths :=
    ["scripts/contract/generate_source_closure.py", "scripts/parity/ensure_oracle.py",
     "scripts/parity/extract_upstream.py",
     "scripts/parity/upstream_target.py"]
  let faults := pushFault faults
    (candidate.extractor.policyId != sourceClosureExtractorPolicy ||
     candidate.extractor.canonicalizerId != sourceClosureCanonicalizer ||
     candidate.extractor.parserPolicyId != sourceClosureParserPolicy ||
     candidate.extractor.parserGrammarFeature != sourceClosureParserGrammar ||
     candidate.extractor.pythonImplementation != sourceClosurePythonImplementation ||
     candidate.extractor.pythonMajorMinorMin != sourceClosurePythonMin ||
     candidate.extractor.pythonMajorMinorMax != sourceClosurePythonMax ||
     !listSetEq (candidate.extractor.sourceFiles.map (·.path)) expectedExtractorPaths ||
     candidate.extractor.sourceBundleSha256 != expectedExtractorSourceBundleSha256)
    .staleExtractorIdentity
  let faults := pushFault faults
    (!candidate.files.all (fileIdentityWellFormed candidate.target.objectFormat))
    .malformedFileIdentity
  let faults := pushFault faults
    (!listNodup (candidate.files.map (·.path))) .duplicateFilePath
  let faults := pushFault faults
    (!candidate.categories.all categoryWellFormed ||
     candidate.categories.any fun category =>
       if category.id == .extractor then category.sourceKind != .localExtractor
       else category.sourceKind != .foreignGit)
    .malformedCategory
  let faults := pushFault faults
    (!listSetEq (candidate.categories.map (·.id)) allSourceCategoryIds)
    .wrongCategorySet
  let faults := pushFault faults (!categoryCrossReferences candidate) .categoryCrossReference
  let faults := pushFault faults (!categoryPredicatesMatch candidate) .categoryPredicateMismatch
  let faults := pushFault faults (!testCategoriesDistinct candidate) .conflatedTestCategories
  let faults := pushFault faults (!tensorApiWellFormed candidate) .malformedTensorApi
  let faults := pushFault faults (!tensorSourceTrapPasses candidate) .tensorSourceTrap
  let faults := pushFault faults (!opsWellFormed candidate) .malformedOps
  let faults := pushFault faults
    (candidate.ops.source != "tinygrad/uop/__init__.py" ||
     !listSetEq (categoryPaths candidate .ops) [candidate.ops.source])
    .opsSourceTrap
  let faults := pushFault faults
    (!dtypesWellFormed candidate || candidate.dtypes.source != "tinygrad/dtype.py" ||
     !listSetEq (categoryPaths candidate .dtypes) [candidate.dtypes.source])
    .malformedDtypes
  let faults := pushFault faults (!backendsWellFormed candidate) .malformedBackends
  let faults := pushFault faults (!testsWellFormed candidate) .malformedTests
  let faults := pushFault faults
    (!listSetEq candidate.limits allExtractionLimits) .wrongLimitSet
  let faults := pushFault faults (!validSha256 candidate.closureSha256) .malformedClosureDigest
  pushFault faults
    (candidate.closureSha256 != expectedSourceClosureSha256) .wrongClosureDigest

def validateSourceClosure (candidate : SourceClosureCandidate) :
    SourceClosureValidationResult ValidatedSourceClosure :=
  match sourceClosureDiagnose candidate with
  | [] => .ok ⟨candidate⟩
  | faults => .faults faults

/-! ## Read-only accessors.  The sole SourceClosureId is the closure SHA-256. -/

def ValidatedSourceClosure.target (closure : ValidatedSourceClosure) : RawTargetIdentity :=
  closure.candidate.target

def ValidatedSourceClosure.repository
    (closure : ValidatedSourceClosure) : RawRepositoryInventory :=
  closure.candidate.repository

def ValidatedSourceClosure.extractor
    (closure : ValidatedSourceClosure) : RawExtractorIdentity :=
  closure.candidate.extractor

def ValidatedSourceClosure.files
    (closure : ValidatedSourceClosure) : List RawSourceFileIdentity :=
  closure.candidate.files

def ValidatedSourceClosure.categories
    (closure : ValidatedSourceClosure) : List RawCategoryInventory :=
  closure.candidate.categories

def ValidatedSourceClosure.tensorApi
    (closure : ValidatedSourceClosure) : RawTensorApiInventory :=
  closure.candidate.tensorApi

def ValidatedSourceClosure.ops
    (closure : ValidatedSourceClosure) : RawOpsInventory :=
  closure.candidate.ops

def ValidatedSourceClosure.dtypes
    (closure : ValidatedSourceClosure) : RawDtypeInventory :=
  closure.candidate.dtypes

def ValidatedSourceClosure.backends
    (closure : ValidatedSourceClosure) : RawBackendInventory :=
  closure.candidate.backends

def ValidatedSourceClosure.tests
    (closure : ValidatedSourceClosure) : RawTestInventory :=
  closure.candidate.tests

def ValidatedSourceClosure.limits
    (closure : ValidatedSourceClosure) : List ExtractionLimit :=
  closure.candidate.limits

def ValidatedSourceClosure.closureSha256 (closure : ValidatedSourceClosure) : String :=
  closure.candidate.closureSha256

def ValidatedSourceClosure.sourceClosureId
    (closure : ValidatedSourceClosure) : SourceClosureId :=
  { digest := digest closure.candidate.closureSha256 }

end Tgrad.Contract

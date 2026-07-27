import Tgrad.Requirements.Requirements

/-! # Tgrad.Requirements.Pilot — three world-facing pilot requirements

These requirements mention Python imports, observable tensor behavior, and
storage lifetime.  They intentionally do not mention UOp, FFI exports,
renderers, kernels, or product files.
-/

namespace Tgrad.Requirements.Pilot

open Tgrad.Requirements

private def revision := "19c4d736f2bc8e26d21f08b28ffd6298408da00f"

private def upstreamManifest : SourceRef :=
  { id := ⟨"SRC-UPSTREAM-MANIFEST"⟩
    kind := .upstreamSource
    revision
    locator := "fixtures/parity/upstream_19c4d736f2bc.json#content_sha256=439370c2308bae30756329b6dfd96c0e004ce7a4aa78f155b4ee689554973874" }

private def oracleClassification : SourceRef :=
  { id := ⟨"SRC-ORACLE-CLASSIFICATION"⟩
    kind := .projectDecision
    revision
    locator := "fixtures/parity/oracle_classification.json" }

private def substitutionPolicy : SourceRef :=
  { id := ⟨"SRC-STRICT-SUBSTITUTION-POLICY"⟩
    kind := .projectDecision
    revision := "fde45f2"
    locator := "scripts/parity/shim/run_pytest.py" }

def pythonProgram : WorldDomain :=
  { id := ⟨"DOMAIN-PYTHON-PROGRAM"⟩
    name := "Python client program"
    kind := .biddable
    description := "A client written against the declared tinygrad profile." }

def pythonImportSystem : WorldDomain :=
  { id := ⟨"DOMAIN-PYTHON-IMPORT"⟩
    name := "Python import system"
    kind := .causal
    description := "Python module lookup, loading, caching, and attribute exposure." }

def upstreamContract : WorldDomain :=
  { id := ⟨"DOMAIN-UPSTREAM-CONTRACT"⟩
    name := "Pinned tinygrad contract"
    kind := .lexical
    description := "The interpreted public behavior at the pinned upstream revision." }

def tensorValues : WorldDomain :=
  { id := ⟨"DOMAIN-TENSOR-VALUES"⟩
    name := "Tensor mathematics"
    kind := .lexical
    description := "Tensor values, shapes, dtypes, broadcasting, and indexing maps." }

def storageWorld : WorldDomain :=
  { id := ⟨"DOMAIN-STORAGE-LIFETIME"⟩
    name := "Tensor storage and lifetime"
    kind := .causal
    description := "Storage identity, Python reachability, view validity, and readback." }

def moduleRequest : Phenomenon :=
  { id := ⟨"PHEN-MODULE-REQUEST"⟩
    domain := pythonImportSystem.id
    kind := .moduleRequest
    controlledBy := .environment
    description := "A client requests a tinygrad module by import name." }

def moduleResolution : Phenomenon :=
  { id := ⟨"PHEN-MODULE-RESOLUTION"⟩
    domain := pythonImportSystem.id
    kind := .moduleResolution
    controlledBy := .machine
    description := "The replacement resolves or rejects the requested module." }

def publicNames : Phenomenon :=
  { id := ⟨"PHEN-PUBLIC-NAMES"⟩
    domain := upstreamContract.id
    kind := .publicName
    controlledBy := .machine
    description := "Names exposed by a resolved compatibility module." }

def tensorCall : Phenomenon :=
  { id := ⟨"PHEN-TENSOR-CALL"⟩
    domain := pythonProgram.id
    kind := .call
    controlledBy := .environment
    description := "A client invokes a declared Tensor operation." }

def tensorArguments : Phenomenon :=
  { id := ⟨"PHEN-TENSOR-ARGUMENTS"⟩
    domain := pythonProgram.id
    kind := .argument
    controlledBy := .environment
    description := "Tensor values, shapes, dtypes, and operation arguments supplied by the client." }

def returnedValue : Phenomenon :=
  { id := ⟨"PHEN-RETURNED-VALUE"⟩
    domain := tensorValues.id
    kind := .tensorValue
    controlledBy := .machine
    description := "The realized tensor value returned to the client." }

def returnedShape : Phenomenon :=
  { id := ⟨"PHEN-RETURNED-SHAPE"⟩
    domain := tensorValues.id
    kind := .shape
    controlledBy := .machine
    description := "The shape observable on the returned Tensor." }

def returnedDtype : Phenomenon :=
  { id := ⟨"PHEN-RETURNED-DTYPE"⟩
    domain := tensorValues.id
    kind := .dtype
    controlledBy := .machine
    description := "The dtype observable on the returned Tensor." }

def raisedException : Phenomenon :=
  { id := ⟨"PHEN-RAISED-EXCEPTION"⟩
    domain := pythonProgram.id
    kind := .exception
    controlledBy := .machine
    description := "The exception class and relevant public message raised to the client." }

def realizationEffects : Phenomenon :=
  { id := ⟨"PHEN-REALIZATION-EFFECTS"⟩
    domain := storageWorld.id
    kind := .realization
    controlledBy := .machine
    description := "Observable realization, synchronization, and materialization effects." }

def baseRelease : Phenomenon :=
  { id := ⟨"PHEN-BASE-REFERENCE-RELEASE"⟩
    domain := storageWorld.id
    kind := .lifetime
    controlledBy := .environment
    description := "The client releases the original Python base Tensor reference." }

def viewValidity : Phenomenon :=
  { id := ⟨"PHEN-VIEW-STORAGE-VALIDITY"⟩
    domain := storageWorld.id
    kind := .storageIdentity
    controlledBy := .machine
    description := "The surviving view continues to designate valid storage and index mapping." }

def pinnedRevision : Assumption :=
  { id := ⟨"ASM-PINNED-UPSTREAM"⟩
    domains := [upstreamContract.id]
    statement := "Compatibility observations use tinygrad revision 19c4d736f2bc8e26d21f08b28ffd6298408da00f."
    provenance := [upstreamManifest] }

def strictSubstitution : Assumption :=
  { id := ⟨"ASM-NO-UPSTREAM-FALLBACK"⟩
    domains := [pythonImportSystem.id]
    statement := "The Tgrad substitution does not resolve missing modules or names from an installed upstream tinygrad package."
    provenance := [substitutionPolicy] }

def deterministicPilotInputs : Assumption :=
  { id := ⟨"ASM-DETERMINISTIC-PILOT-INPUTS"⟩
    domains := [tensorValues.id, storageWorld.id]
    statement := "Pilot tensor scenarios use finite deterministic inputs and no concurrent mutation."
    provenance := [oracleClassification] }

def publicMetalPilot : CompatibilityProfile :=
  { id := ⟨"PROFILE-PUBLIC-METAL-PILOT"⟩
    upstreamRevision := revision
    includedFrames :=
      [⟨"FRAME-PYTHON-SUBSTITUTION"⟩,
       ⟨"FRAME-TENSOR-TRANSFORMATION"⟩,
       ⟨"FRAME-VIEW-WORKPIECE"⟩]
    environments := ["CPython on macOS", "Metal-capable Apple device"]
    description := "The three-requirement pilot slice; not a claim of full public-API parity." }

def importHelpers : Requirement :=
  { id := ⟨"REQ-PY-IMPORT-HELPERS"⟩
    frame := ⟨"FRAME-PYTHON-SUBSTITUTION"⟩
    profiles := [publicMetalPilot.id]
    monitored := [moduleRequest.id]
    controlled := [moduleResolution.id, publicNames.id, raisedException.id]
    assumptions := [pinnedRevision.id, strictSubstitution.id]
    relation := .all
      [.importResolvesWithoutFallback "Tgrad strict substitution",
       .publicNamesContain ["Context", "getenv", "DEV"],
       .sameException]
    provenance := [upstreamManifest, oracleClassification, substitutionPolicy]
    statement := "A request for tinygrad.helpers resolves entirely to the Tgrad substitution, exposes the pilot's prerequisite names, and never silently falls back to upstream tinygrad." }

def broadcastAdd : Requirement :=
  { id := ⟨"REQ-TENSOR-ADD-BROADCAST"⟩
    frame := ⟨"FRAME-TENSOR-TRANSFORMATION"⟩
    profiles := [publicMetalPilot.id]
    monitored := [tensorCall.id, tensorArguments.id]
    controlled :=
      [returnedValue.id, returnedShape.id, returnedDtype.id,
       raisedException.id, realizationEffects.id]
    assumptions := [pinnedRevision.id, deterministicPilotInputs.id]
    relation := .all
      [.exactTensorValue, .sameShape, .sameDtype, .sameException, .sameEffects]
    provenance := [upstreamManifest, oracleClassification]
    statement := "For deterministic bf16 pilot inputs with legal two-dimensional broadcasting, Tensor addition has the pinned upstream value, shape, dtype, exception, and realization observations." }

def viewReadbackLifetime : Requirement :=
  { id := ⟨"REQ-VIEW-READBACK-TRANSPOSE"⟩
    frame := ⟨"FRAME-VIEW-WORKPIECE"⟩
    profiles := [publicMetalPilot.id]
    monitored := [tensorCall.id, tensorArguments.id, baseRelease.id]
    controlled :=
      [returnedValue.id, returnedShape.id, raisedException.id,
       realizationEffects.id, viewValidity.id]
    assumptions := [pinnedRevision.id, deterministicPilotInputs.id]
    relation := .all
      [.exactTensorValue, .sameShape, .sameException,
       .sameEffects, .sameStorageAndLifetime]
    provenance := [upstreamManifest, oracleClassification]
    statement := "A transposed or sliced view reads back through its declared index mapping and remains valid after the original Python base reference is released." }

def substitutionFrame : ProblemFrame :=
  { id := ⟨"FRAME-PYTHON-SUBSTITUTION"⟩
    kind := .substitution
    domains := [pythonProgram.id, pythonImportSystem.id, upstreamContract.id]
    sharedPhenomena :=
      [moduleRequest.id, moduleResolution.id, publicNames.id, raisedException.id]
    requirementIds := [importHelpers.id]
    description := "Substitute Tgrad at Python's module boundary without upstream fallback." }

def transformationFrame : ProblemFrame :=
  { id := ⟨"FRAME-TENSOR-TRANSFORMATION"⟩
    kind := .transformation
    domains := [pythonProgram.id, upstreamContract.id, tensorValues.id, storageWorld.id]
    sharedPhenomena :=
      [tensorCall.id, tensorArguments.id, returnedValue.id, returnedShape.id,
       returnedDtype.id, raisedException.id, realizationEffects.id]
    requirementIds := [broadcastAdd.id]
    description := "Preserve observable tensor behavior for a bounded broadcast-add slice." }

def viewFrame : ProblemFrame :=
  { id := ⟨"FRAME-VIEW-WORKPIECE"⟩
    kind := .workpiece
    domains := [pythonProgram.id, tensorValues.id, storageWorld.id]
    sharedPhenomena :=
      [tensorCall.id, tensorArguments.id, returnedValue.id, returnedShape.id,
       raisedException.id, realizationEffects.id, baseRelease.id, viewValidity.id]
    requirementIds := [viewReadbackLifetime.id]
    description := "Preserve view indexing and storage validity across Python lifetime events." }

def helpersObstacle : Obstacle :=
  { id := "OBS-MISSING-HELPERS-SURFACE"
    kind := .missingImplementation
    obstructs := [importHelpers.id, broadcastAdd.id, viewReadbackLifetime.id]
    responsibility := .compatibilityBoundary
    condition := "Applicable upstream tests cannot collect because the strict substitution does not provide tinygrad.helpers."
    resolveBy := "Provide only the reviewed prerequisite surface, retain the no-fallback invariant, and re-run collection before interpreting operation results." }

def catalog : RequirementCatalog :=
  { domains :=
      [pythonProgram, pythonImportSystem, upstreamContract, tensorValues, storageWorld]
    phenomena :=
      [moduleRequest, moduleResolution, publicNames, tensorCall, tensorArguments,
       returnedValue, returnedShape, returnedDtype, raisedException,
       realizationEffects, baseRelease, viewValidity]
    assumptions := [pinnedRevision, strictSubstitution, deterministicPilotInputs]
    profiles := [publicMetalPilot]
    requirements := [importHelpers, broadcastAdd, viewReadbackLifetime]
    frames := [substitutionFrame, transformationFrame, viewFrame]
    obstacles := [helpersObstacle] }

/-- Internal consistency only: this theorem is not compatibility evidence. -/
theorem catalog_is_structurally_well_formed : catalog.wellFormed = true := by
  native_decide

end Tgrad.Requirements.Pilot

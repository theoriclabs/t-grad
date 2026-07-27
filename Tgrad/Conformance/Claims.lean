import Tgrad.Requirements.Pilot
import Tgrad.Specification.Pilot
import Tgrad.PythonFFI

/-! # Tgrad.Conformance.Claims — mapping specifications to product candidates

A symbol mapping establishes that there is code worth observing.  It does not
establish conformance.  Behavioral claims are promoted only through
revision-bound observations in `Tgrad.Evidence`.
-/

namespace Tgrad.Conformance

open Tgrad.Requirements
open Tgrad.Requirements.Pilot
open Tgrad.Specification
open Tgrad.Specification.Pilot

structure ImplementationRef where
  id : String
  symbols : List String
  files : List String
  deriving DecidableEq, BEq, Repr, Inhabited

def ImplementationRef.wellFormed (implementation : ImplementationRef) : Bool :=
  !implementation.id.trimAscii.isEmpty &&
  !implementation.symbols.isEmpty &&
  implementation.symbols.all (fun symbol => !symbol.trimAscii.isEmpty) &&
  !implementation.files.isEmpty &&
  implementation.files.all (fun file => !file.trimAscii.isEmpty)

structure CandidateMapping where
  requirement : RequirementId
  specification : SpecId
  implementation : ImplementationRef
  rationale : String
  deriving DecidableEq, BEq, Repr, Inhabited

def CandidateMapping.wellFormed (mapping : CandidateMapping) : Bool :=
  mapping.requirement.valid &&
  mapping.specification.valid &&
  mapping.implementation.wellFormed &&
  !mapping.rationale.trimAscii.isEmpty

def broadcastAddCandidate : CandidateMapping :=
  { requirement := broadcastAdd.id
    specification := addBoundary.id
    implementation :=
      { id := "IMPL-GRAPH-BROADCAST-ADD"
        symbols :=
          ["Tgrad.Tensor.expand", "Tgrad.PythonFFI.tensorBinop",
           "Tgrad.PythonFFI.realizeGraph"]
        files :=
          ["Tgrad/Tensor.lean", "Tgrad/PythonFFI.lean",
           "Tgrad/Renderer/Elementwise.lean", "python/tgrad.py"] }
    rationale := "The product has a graph constructor, broadcast view mapping, and graph-indexed realization route for the pilot operation." }

def viewReadbackCandidate : CandidateMapping :=
  { requirement := viewReadbackLifetime.id
    specification := viewBoundary.id
    implementation :=
      { id := "IMPL-VIEW-MATERIALIZATION"
        symbols :=
          ["Tgrad.Tensor.transpose", "Tgrad.Pipeline.materializeView",
           "Tgrad.PythonFFI.realizeGraph"]
        files :=
          ["Tgrad/Tensor.lean", "Tgrad/Pipeline.lean",
           "Tgrad/PythonFFI.lean", "python/tgrad.py"] }
    rationale := "The product records views and has a materialization route; observation must still establish value and lifetime behavior." }

def pilotCandidates : List CandidateMapping :=
  [broadcastAddCandidate, viewReadbackCandidate]

def candidateFor (requirement : RequirementId) : Option CandidateMapping :=
  pilotCandidates.find? (fun candidate => candidate.requirement == requirement)

/-- Internal consistency only; mappings are not conformance evidence. -/
theorem pilot_candidate_mappings_are_well_formed :
    pilotCandidates.all CandidateMapping.wellFormed = true := by
  native_decide

/- Compile-time product pins.  Renaming or removing a mapped implementation
symbol breaks the specification build and forces reconciliation. -/
#check Tgrad.Tensor.expand
#check Tgrad.Tensor.transpose
#check Tgrad.Pipeline.materializeView
#check Tgrad.PythonFFI.tensorBinop
#check Tgrad.PythonFFI.realizeGraph

end Tgrad.Conformance

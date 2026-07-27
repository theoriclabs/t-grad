import Tgrad.Spec.Epistemic

/-! # Tgrad.Spec.Architecture — components and channels

The component boundary is the unit of ownership and impact analysis. File paths
remain in work items because merge conflicts happen at files; components answer
the architectural question “which part of the system is changing?”
-/

namespace Tgrad.Spec

inductive Component where
  | pythonAuthoring
  | cTrampoline
  | leanFfi
  | tensorIr
  | viewAlgebra
  | rewriteEngine
  | scheduler
  | renderer
  | metalRuntime
  | gateHarness
  | evidenceStore
  | specification
  deriving DecidableEq, BEq, Repr, Inhabited

inductive Payload where
  | tensorRequest
  | rawPointer
  | uopGraph
  | view
  | indexExpr
  | kernelDecl
  | metalSource
  | dispatchResult
  | evidence
  | workState
  deriving DecidableEq, BEq, Repr, Inhabited

structure Channel where
  name : String
  source : Component
  target : Component
  payload : Payload
  contract : Epistemic String
  deriving Repr, Inhabited

def productChannels : List Channel :=
  [ { name := "python-to-c", source := .pythonAuthoring,
      target := .cTrampoline, payload := .tensorRequest,
      contract := .confirmed "ctypes call with typed argtypes/restype"
        "python/tgrad.py bindings checked against c/tgrad_python.c" },
    { name := "c-to-lean", source := .cTrampoline,
      target := .leanFfi, payload := .rawPointer,
      contract := .confirmed "C trampolines call Lean @[export] symbols"
        "generated Lean C signatures compared during FFI audit" },
    { name := "tensor-to-view", source := .tensorIr,
      target := .viewAlgebra, payload := .uopGraph,
      contract := .confirmed "movement chain lowers to strides/offset View"
        "f679bf7; 20 Lean assertions and 13 numpy differential cases" },
    { name := "view-to-renderer", source := .viewAlgebra,
      target := .renderer, payload := .indexExpr,
      contract := .confirmed "View.indexOf supplies scalar matmul load indices"
        "f679bf7; multi-axis slice and axis-0 expand regressions pass" },
    { name := "renderer-to-runtime", source := .renderer,
      target := .metalRuntime, payload := .metalSource,
      contract := .confirmed "runtime compiles renderKernel output"
        "f679bf7; fixtures hidden and 64x64 still byte-matched" },
    { name := "runtime-to-evidence", source := .metalRuntime,
      target := .evidenceStore, payload := .evidence,
      contract := .tentative "gates write JSON evidence"
        "gate scripts inspected"
        "regenerate evidence at HEAD and verify every recorded hash" } ]

def channelsFrom (component : Component) : List Channel :=
  productChannels.filter (fun channel => channel.source == component)

example : (channelsFrom .renderer).length = 1 := by native_decide

end Tgrad.Spec

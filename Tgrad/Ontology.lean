import Tgrad.Dtype
import Tgrad.Shape
import Tgrad.UOp
import Tgrad.UPat
import Tgrad.GraphRewrite
import Tgrad.Schedule.View
import Tgrad.Schedule.Rangeify
import Tgrad.Renderer.Metal
import Tgrad.Codegen.Opt.Heuristic
import Tgrad.Tensor
import Tgrad.Pipeline

/-! # Tgrad.Ontology — stable vocabulary of the product

This module describes the kinds of thing that exist in Tgrad and the maps
between them. It deliberately does **not** contain the current bug list,
roadmap, or maturity judgments. Those facts change as work lands and belong in
`Tgrad.Spec`, where they carry evidence and upgrade paths.

The separation is load-bearing. An earlier version encoded fixed bugs as
constructors of `Ontology.Gap`; after the bugs were repaired, the constructors
still compiled and the document silently lied. Ontology should change when the
domain vocabulary changes, not whenever a finding is remediated.
-/

namespace Tgrad
namespace Ontology

/-- Stable carrier sorts in the current product. -/
inductive Sort_ where
  | dtype
  | shape
  | view
  | uop
  | upat
  | indexExpr
  | kernelDecl
  | metalSource
  | launchPlan
  | program
  | buffer
  | hostBytes
  | tensor
  deriving DecidableEq, BEq, Repr, Inhabited

/-- Whether failure is visible in a function's result type. -/
inductive Totality where
  | total
  | typedPartial
  | terminationUnproven
  | panicsToDefault
  deriving DecidableEq, BEq, Repr, Inhabited

/-- A typed edge in the product's compiler/runtime graph. -/
structure Morphism where
  name   : String
  sources : List Sort_
  targets : List Sort_
  totality : Totality
  deriving Repr, Inhabited

/-- The currently observed path for a view-aware matmul.

This is architectural vocabulary, not a claim that every edge is correct.
Correctness and operational confidence live in `Tgrad.Spec.Findings` and
`Tgrad.Spec.LiveConditions`. -/
def viewMatmulPath : List Morphism :=
  [ { name := "Tensor.uop", sources := [.tensor], targets := [.uop],
      totality := .total },
    { name := "Schedule.viewOfUOp", sources := [.uop], targets := [.view],
      totality := .typedPartial },
    { name := "Schedule.View.indexOf", sources := [.view, .uop],
      targets := [.indexExpr],
      totality := .total },
    { name := "scalarMatmulKernelDeclWithIdx",
      sources := [.shape, .indexExpr, .indexExpr], targets := [.kernelDecl],
      totality := .total },
    { name := "Renderer.Metal.renderKernel", sources := [.kernelDecl],
      targets := [.metalSource], totality := .terminationUnproven },
    { name := "metalCompile", sources := [.metalSource], targets := [.program],
      totality := .typedPartial },
    { name := "metalDispatch",
      sources := [.program, .buffer, .launchPlan], targets := [.buffer],
      totality := .typedPartial } ]

/-- The promoted view-readback path. Unlike `viewMatmulPath`, rangeify's
indexed result directly governs the copy kernel's source addressing. -/
def viewMaterializationPath : List Morphism :=
  [ { name := "Tensor.uop", sources := [.tensor], targets := [.uop],
      totality := .total },
    { name := "Schedule.viewOfUOp", sources := [.uop], targets := [.view],
      totality := .typedPartial },
    { name := "Schedule.Rangeify.rangeify", sources := [.uop], targets := [.uop],
      totality := .terminationUnproven },
    { name := "extract INDEX(BUFFER,index)", sources := [.uop],
      targets := [.buffer, .indexExpr], totality := .typedPartial },
    { name := "Pipeline.materializeViewKernelDecl",
      sources := [.view, .indexExpr], targets := [.kernelDecl],
      totality := .total },
    { name := "Renderer.Metal.renderKernel", sources := [.kernelDecl],
      targets := [.metalSource], totality := .terminationUnproven },
    { name := "metalCompile", sources := [.metalSource], targets := [.program],
      totality := .typedPartial },
    { name := "metalAlloc", sources := [.view, .dtype], targets := [.buffer],
      totality := .typedPartial },
    { name := "metalDispatch",
      sources := [.program, .buffer, .buffer, .launchPlan], targets := [.buffer],
      totality := .typedPartial },
    { name := "metalBufferReadBytes", sources := [.buffer], targets := [.hostBytes],
      totality := .typedPartial } ]

/-- The rewrite engine is real but remains a distinct path from runtime
lowering. The relationship between these paths is a mutable finding, not an
ontological constructor. -/
def rewritePath : List Morphism :=
  [ { name := "UOp.ofParsed", sources := [.uop], targets := [.uop],
      totality := .typedPartial },
    { name := "matchPat", sources := [.upat, .uop], targets := [.uop],
      totality := .typedPartial },
    { name := "graphRewriteBottomUp", sources := [.uop, .upat], targets := [.uop],
      totality := .panicsToDefault },
    { name := "toRecords", sources := [.uop], targets := [.uop],
      totality := .total } ]

def pathSorts (path : List Morphism) : List Sort_ :=
  (path.flatMap (·.sources) ++ path.flatMap (·.targets)).eraseDups

example : (pathSorts viewMatmulPath).contains Sort_.view = true := by native_decide
example : (pathSorts viewMatmulPath).contains Sort_.upat = false := by native_decide
example : (pathSorts rewritePath).contains Sort_.upat = true := by native_decide
example : (pathSorts viewMaterializationPath).contains Sort_.hostBytes = true := by
  native_decide
example :
    (viewMaterializationPath.find? (fun step =>
      step.name == "Pipeline.materializeViewKernelDecl")).map
      (fun step => step.sources == [.view, .indexExpr]) = some true := by
  native_decide

/-! ## Pins against product code

These checks make vocabulary drift visible. They are intentionally shallow:
epistemic claims about what these functions *do* belong in the spec layer.
-/

#check (Dtype : Type)
#check (Shape : Type)
#check (UOp : Type)
#check (UPat : Type)
#check (Slice : Type)
#check (Tensor : Type)
#check (Schedule.View : Type)
#check (Renderer.Metal.KernelDecl : Type)
#check (Codegen.GpuDims : Type)
#check (Runtime.BufferHandle : Type)

#check (UOp.kind : UOp -> UOpKind)
#check (UOp.dtypeOf : UOp -> Dtype)
#check (UOp.children : UOp -> List UOp)
#check (Schedule.viewOfUOp : UOp -> Option Schedule.View)
#check (Schedule.View.indexOf : Schedule.View -> List UOp -> UOp)
#check (Schedule.Rangeify.rangeify : UOp -> UOp)
#check (Pipeline.materializeViewKernelDecl :
  Schedule.View -> UOp -> Renderer.Metal.KernelDecl)
#check (Pipeline.materializeView : Tensor -> IO (Except PipelineError Tensor))
#check (Renderer.Metal.renderKernel : Renderer.Metal.KernelDecl -> String)
#check (Tensor.shape : Tensor -> Shape)

/-- A concrete pin that the repaired scheduler is not the old identity pass. -/
example :
    ((Schedule.Rangeify.rangeify
      (.permute (.buffer 0 [4, 8] .bfloat16_) [1, 0])).beq
      (.permute (.buffer 0 [4, 8] .bfloat16_) [1, 0])) = false := by
  native_decide

end Ontology
end Tgrad

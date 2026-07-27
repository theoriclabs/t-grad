import Tgrad.Dtype
import Tgrad.Shape
import Tgrad.UOp
import Tgrad.UPat
import Tgrad.GraphRewrite
import Tgrad.Schedule.Indexing
import Tgrad.Schedule.View
import Tgrad.Schedule.Rangeify
import Tgrad.Renderer.Metal
import Tgrad.Codegen.Opt.Heuristic
import Tgrad.Tensor

/-! # Tgrad.Ontology — the ontology of Tgrad, written in Lean

  This module is **documentation that type-checks**. It states what
  kinds of things exist in Tgrad, how they map into each other, and
  which of those maps are total. Every claim about a real definition
  is pinned with `#check`, so if the underlying code changes shape,
  this file stops compiling instead of quietly becoming a lie.

  An *ontology* here means three things:

  1. **Sorts** — the kinds of thing that exist.
  2. **Morphisms** — the maps between sorts that the compiler walks.
  3. **Invariants** — the facts a sort guarantees about itself.

  A claim frequently made about this project is that tinygrad's
  ontology was translated "into a much stronger type system". Section
  4 grades that claim per sort rather than in aggregate, because the
  answer differs sharply between layers: the IR layer is genuinely
  stronger than tinygrad's, and the renderer layer is weaker.
-/

namespace Tgrad
namespace Ontology

/-! ## 1. Sorts

  Tgrad's universe has nine sorts. `Sort_` is a first-class value so
  that the ontology can be queried, folded over, and tested. -/

inductive Sort_ where
  /-- Scalar element types. Carrier: `Tgrad.Dtype`, 15 constructors. -/
  | dtype
  /-- Tensor extents. Carrier: `Tgrad.Shape := List Nat`, and the
      symbolic lift `Tgrad.SintShape := List Sint`. -/
  | shape
  /-- The compute IR. Carrier: `Tgrad.UOp`, 21 constructors. -/
  | uop
  /-- Patterns over the IR. Carrier: `Tgrad.UPat`. -/
  | upat
  /-- Index arithmetic derived from movement ops. Carrier: `UOp`
      again — indices are not a separate sort, which is itself an
      ontological choice (see `Gap.indexIsNotASort`). -/
  | index
  /-- The Metal AST. Carrier: `Renderer.Metal.KernelDecl`. -/
  | kernel
  /-- Launch geometry. Carrier: `Codegen.GpuDims`, `DispatchPlan`. -/
  | plan
  /-- Device memory. Carrier: `Runtime.BufferHandle`. -/
  | buffer
  /-- The user-facing value. Carrier: `Tgrad.Tensor`. -/
  | tensor
  deriving DecidableEq, Repr, Inhabited

/-- How total a map is. This is the ontology's most load-bearing
    annotation: it records whether a morphism can fail, and if so,
    whether the failure is *visible* to the caller. -/
inductive Totality where
  /-- Total function. Cannot fail. -/
  | total
  /-- Failure is in the return type (`Option`/`Except`). Honest. -/
  | typedPartial
  /-- `partial def` — no termination proof, but total on the nose. -/
  | unproven
  /-- Falls through to `panic!`, which in Lean 4 does **not** abort:
      it returns `default`. Failure is therefore *invisible* at the
      call site and surfaces later as corrupt output. -/
  | panicsToDefault
  deriving DecidableEq, Repr, Inhabited

structure Morphism where
  name   : String
  source : Sort_
  target : Sort_
  tot    : Totality
  deriving Repr, Inhabited

/-! ## 2. The morphisms that actually run

  This is the `a @ b` path, in order. Compare it against §3: the
  rewrite engine does not appear here. -/

def runtimePath : List Morphism :=
  [ { name := "Tensor.shape",          source := .tensor, target := .shape,  tot := .panicsToDefault },
    { name := "Tensor.buffer",         source := .tensor, target := .buffer, tot := .panicsToDefault },
    { name := "viewIndexUOpForA/B",    source := .tensor, target := .index,  tot := .panicsToDefault },
    { name := "pickDispatchPlan",      source := .shape,  target := .plan,   tot := .typedPartial },
    { name := "matmulKernelDeclFor",   source := .plan,   target := .kernel, tot := .total },
    { name := "renderKernel",          source := .kernel, target := .uop,    tot := .unproven },
    { name := "metalCompile+dispatch", source := .kernel, target := .buffer, tot := .typedPartial } ]

/-- The ontology's headline weakness, as a computed fact rather than
    a claim: the majority of the runtime path is not honestly partial.
    Three of seven steps report failure by returning `default`. -/
def panickingSteps : List String :=
  (runtimePath.filter (fun m => m.tot == .panicsToDefault)).map (·.name)

example : panickingSteps.length = 3 := by native_decide

/-! ## 3. The rewrite ontology — real, and disconnected

  `UPat`/`matchPat`/`graphRewriteBottomUp` is a faithful port of
  tinygrad's matcher contract, and it genuinely reduces its fixture
  from 45 nodes to 23. It has exactly one caller in the repository:
  the `tgrad reduce-symbolic-dag` CLI subcommand.

  Nothing on `runtimePath` invokes it. The two halves of the project
  are disjoint programs that share a `UOp` type. -/

def rewritePath : List Morphism :=
  [ { name := "UOp.ofParsed",         source := .uop,  target := .uop,  tot := .typedPartial },
    { name := "matchPat",             source := .upat, target := .uop,  tot := .typedPartial },
    { name := "graphRewriteBottomUp", source := .uop,  target := .uop,  tot := .panicsToDefault },
    { name := "toRecords",            source := .uop,  target := .uop,  tot := .total } ]

/-- Morphisms reachable from `Tensor`. `.upat` is not among them. -/
def sortsOnRuntimePath : List Sort_ :=
  (runtimePath.map (·.source) ++ runtimePath.map (·.target)).eraseDups

example : ¬ (Sort_.upat ∈ sortsOnRuntimePath) := by native_decide

/-! ## 4. Where the types are genuinely stronger than tinygrad

  Graded per sort. `strongerThanPython` means the Lean type rules out
  a class of error that tinygrad's Python representation permits. -/

inductive Strength where
  /-- The type rules out states tinygrad allows. A real improvement. -/
  | stronger
  /-- Same expressive power, different syntax. -/
  | parity
  /-- The Lean type is *weaker*: semantics live in `String` payloads
      that no invariant constrains. -/
  | weaker
  deriving DecidableEq, Repr, Inhabited

def strengthOf : Sort_ → Strength
  -- 15 closed constructors vs tinygrad's `DType` dataclass. Genuine:
  -- exhaustive matching is enforced, and `Spec.classify` breaks the
  -- build when a constructor is added.
  | .dtype  => .stronger
  -- `UOp` has typed per-op payloads instead of tinygrad's untyped
  -- `arg: Any`. This is the single strongest result in the project
  -- and it is real.
  | .uop    => .stronger
  -- `UPat` mirrors tinygrad's matcher faithfully.
  | .upat   => .parity
  -- `Shape := List Nat` carries no rank or positivity. The validated
  -- algebra (`reshapeShape`, `permuteShape`, `expandShape`) exists
  -- but no `Tensor` method calls it.
  | .shape  => .parity
  -- Indices are `UOp`s built by a 6-arm pattern table matching
  -- literal axis lists such as `[1, 0]`. Not a sort of its own.
  | .index  => .weaker
  -- `KernelDecl`'s statements carry their semantics as `String`:
  -- `.declInt "alu0" "((lidx0>>4)<<8)"`. Nothing relates
  -- `.declAccArray "acc0" 32` to `.accStore "acc0" 99`. tinygrad's
  -- renderer at least consumes a typed UOp graph here, so this is
  -- strictly weaker than the thing it replaced.
  | .kernel => .weaker
  -- `GpuDims` is a typed record; `DispatchPlan.useTc` is read by
  -- nothing outside its own theorem.
  | .plan   => .parity
  -- `BufferHandle` is a raw `UInt64` with no ownership or liveness
  -- in the type. Python views share it with no parent reference.
  | .buffer => .weaker
  -- `Tensor` is a `UOp` plus a handle, with no shape-buffer
  -- agreement invariant: `sliceShapeNat` uses `List.zip`, so a
  -- rank-2 tensor with one slice yields a rank-1 shape while the
  -- Python side computes rank-2.
  | .tensor => .weaker

/-- The honest summary of "tinygrad's ontology in a stronger type
    system": true for the IR core, false for everything downstream
    of it. -/
def strongerSorts : List Sort_ :=
  [Sort_.dtype, .shape, .uop, .upat, .index, .kernel, .plan, .buffer, .tensor].filter
    (fun s => strengthOf s == .stronger)

def weakerSorts : List Sort_ :=
  [Sort_.dtype, .shape, .uop, .upat, .index, .kernel, .plan, .buffer, .tensor].filter
    (fun s => strengthOf s == .weaker)

example : strongerSorts.length = 2 := by native_decide
example : weakerSorts.length  = 4 := by native_decide

/-! ## 5. Gaps — ontological commitments the code does not honour

  Each constructor names a place where the ontology as designed and
  the ontology as implemented disagree. These are the concrete
  targets for any expansion work. -/

inductive Gap where
  /-- `Schedule.Rangeify.rangeify : UOp → UOp` is `fun u => u`. The
      movement algebra that does real work (`rangeifyChain`) is only
      reachable from a CLI subcommand over a fixture. -/
  | rangeifyIsIdentity
  /-- `viewIndexUOpForA` matches `.slice _ (slc :: _)` and discards
      every slice after the first, so `a[0:8, 4:12] @ b` silently
      reads column 0. Wrong numbers, exit 0. -/
  | sliceDropsTrailingAxes
  /-- The `.expand` arm assumes the broadcast singleton is on axis 1.
      A `(1,N) -> (K,N)` broadcast emits `Ridx0 * N` where the
      correct index is K-invariant. -/
  | expandAssumesAxis1
  /-- Indices are `UOp` rather than their own sort, so nothing
      distinguishes an index expression from a value expression, and
      `renderIndexExpr` must `panic!` on the difference. -/
  | indexIsNotASort
  /-- `UOp.beq` has no `.reduce` arm and falls through to `true`, so
      `reduce .add body axes == reduce .mul body axes`. It is the
      interning key for `findIdx`. Unreachable only because
      `UOp.ofParsed` cannot build a REDUCE. -/
  | beqIgnoresReduceOp
  /-- Const-folding uses Lean's `Int.ediv`/`emod` (Euclidean) while
      tinygrad uses Python floor semantics and the renderer emits C
      truncating `/`. Three meanings for one `UOp` kind. -/
  | threeDivisionSemantics
  /-- `Tensor.reshape/permute/expand/slice` push unvalidated nodes;
      `reshapeShape`/`permuteShape`/`expandShape` — the only
      validating functions — are called solely by JSON emitters. -/
  | movementValidationIsDead
  /-- The production kernel for every sentinel shape is read from
      `fixtures/codegen/*.msl` by `IO.FS.readFile`, so `renderKernel`
      is not on the benchmarked path at all. -/
  | productionPathIsFileReplay
  deriving DecidableEq, Repr, Inhabited

/-- Gaps that produce a *wrong answer* rather than a crash or a
    missing feature. These are the ones that matter most, because no
    gate can catch them by construction: the output is plausible. -/
def silentlyWrong : List Gap :=
  [.sliceDropsTrailingAxes, .expandAssumesAxis1, .threeDivisionSemantics]

/-! ## 6. Pins against the real code

  If any signature below changes, this module stops compiling. That
  is the entire point of writing the ontology in Lean rather than in
  Markdown. -/

section Pins

-- Sort carriers exist with the arities claimed above.
#check (Dtype : Type)
#check (Shape : Type)
#check (Sint : Type)
#check (UOp : Type)
#check (UPat : Type)
#check (Slice : Type)
#check (Tensor : Type)
#check (Renderer.Metal.KernelDecl : Type)
#check (Renderer.Metal.Stmt : Type)
#check (Renderer.Metal.ShapeSentinel : Type)
#check (Codegen.GpuDims : Type)
#check (Runtime.BufferHandle : Type)

-- The IR projections that make `UOp` a usable sort.
#check (UOp.kind    : UOp → UOpKind)
#check (UOp.dtypeOf : UOp → Dtype)
#check (UOp.children : UOp → List UOp)

-- The renderer contract, as advertised in EXPERIMENT_RESULT.md.
#check (Renderer.Metal.renderKernel : Renderer.Metal.KernelDecl → String)

-- Movement validation exists (and is dead — see `Gap`).
#check (reshapeShape : SintShape → SintShape → Option SintShape)
#check (permuteShape : SintShape → List Nat → Option SintShape)

-- `Tensor` shape computation, including the `List.zip` truncation.
#check (Tensor.shape : Tensor → Shape)
#check (sliceShapeNat : List Nat → List Slice → List Nat)

-- `rangeify` was `fun u => u`. It now collapses movement chains into
-- `.index` nodes via `Schedule.View`. This file previously carried
-- `example (u : UOp) : Rangeify.rangeify u = u := rfl`, which stopped
-- compiling the moment the behaviour changed — the pin doing its job.
#check (Rangeify.rangeify : UOp → UOp)
#check (Schedule.Rangeify.rangeify : UOp → UOp)

-- The `View` sort that closed the gap.
#check (Schedule.View : Type)
#check (Schedule.View.indexOf : Schedule.View → List UOp → UOp)
#check (Schedule.viewOfUOp : UOp → Option Schedule.View)

-- Rangeify is no longer the identity: a transposed buffer becomes an
-- indexed load. Decided, not asserted. (`UOp` has no `DecidableEq`
-- instance, so this goes through `UOp.beq`.)
example :
    ((Rangeify.rangeify (.permute (.buffer 0 [4, 8] .bfloat16_) [1, 0])).beq
      (.permute (.buffer 0 [4, 8] .bfloat16_) [1, 0])) = false := by native_decide

-- The movement node is gone from the output.
example :
    UOp.countMovementNodes
      (Rangeify.rangeify (.permute (.buffer 0 [4, 8] .bfloat16_) [1, 0])) = 0 := by
  native_decide

end Pins

/-! ## 7. What a complete ontology would add

  Ordered by how much of the current gap set each item closes.

  1. **A `View` sort.** tinygrad's `ShapeTracker`/`View` is the thing
     Tgrad is missing, and its absence is why `viewIndexUOpForA` is a
     pattern table. A `View` with `shape`, `strides`, `offset`, and
     `mask`, plus a `View.compose` that is closed under the movement
     ops, replaces all six arms and closes
     `sliceDropsTrailingAxes`, `expandAssumesAxis1`, and
     `indexIsNotASort` at once.

  2. **Indices as their own sort.** `IndexExpr` distinct from `UOp`
     makes `renderIndexExpr` total and removes its `panic!`.

  3. **A shape–buffer invariant on `Tensor`.** Carrying a proof that
     `numel shape * dtype.sizeBytes ≤ buffer.size` turns the current
     class of out-of-bounds GPU reads into a compile error.

  4. **A typed expression sort in `KernelDecl`.** Replacing the
     `String` payloads with an expression tree is what would make
     "Metal rendering is a pure function over typed declarations"
     true in the sense the phrase implies.

  5. **Wiring the rewrite engine into `runtimePath`.** The engine
     exists and works. Until a lowering path calls it, the IR sort
     and the kernel sort are connected only by transcription. -/

end Ontology
end Tgrad

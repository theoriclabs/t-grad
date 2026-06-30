import Tgrad.UOp
import Tgrad.Dtype

/-! # Tgrad.Model.Spec — S0 product specification (capability lattice)

  Companion to `Tgrad.Model.Impact` (S1, the change calculus). This is
  the S0 layer from `docs/TGRAD_PRODUCTION_MODEL.md`: the product modeled
  as a *maturity assignment over a capability space*, plus the closure
  frontier between the current product and "full tinygrad".

  ## Experiment 0 — grounding by participation

  The op- and dtype-axes are tied to the REAL `Tgrad.UOp` / `Tgrad.Dtype`
  by the TOTAL functions `classify` and `dtypeClassOf`, written with
  **no wildcard**. Consequences, both of which are falsifiable by build:

  * Adding / renaming a real `UOp` constructor or `Dtype` makes this
    module fail to compile until it is reconciled — the spec cannot
    silently drift from the product it describes.
  * A `_ => …` catch-all in either function would OPT OUT of grounding
    (the spec would keep compiling against a changed product) and void
    the experiment. So: never add one here.

  This is the property that distinguishes a *grounded* Lean spec from
  prose-or-JSON-in-Lean. See `docs/TGRAD_PRODUCTION_MODEL.md` §6.
-/

namespace Tgrad
namespace Model

/-! ## Maturity ladder (the inter-cycle ratchet) -/

/-- How far a capability has been carried. Production only ever raises
    this; it is the per-capability form of TGrad's gate ladder and the
    factory's `QualityLevel`. -/
inductive Maturity
  | absent      -- not representable
  | modeled     -- typed / spec-only (Ring 3): compiles, not on the runtime path
  | wired       -- reachable on the runtime path
  | verified    -- gated by an oracle (byte-equal / numerical-diff / proof)
  | optimized   -- gated by a perf bound
  deriving BEq, Repr, Inhabited, DecidableEq

def Maturity.rank : Maturity → Nat
  | .absent => 0 | .modeled => 1 | .wired => 2 | .verified => 3 | .optimized => 4

def Maturity.toStr : Maturity → String
  | .absent => "absent" | .modeled => "modeled" | .wired => "wired"
  | .verified => "verified" | .optimized => "optimized"

/-- Bool strictly-below test, usable in `List.filter`. -/
def Maturity.below (a b : Maturity) : Bool := decide (a.rank < b.rank)

/-! ## The op axis — GROUNDED against `Tgrad.UOp` -/

/-- Capability families over the op ontology. Each real `UOp` constructor
    maps to exactly one of these via `classify`. -/
inductive OpClass
  | leafIO         -- param/const/vconst/defineVar/special/buffer/var
  | movementView   -- permute/reshape/expand/slice
  | memoryAccess   -- index/load/store
  | aluBinary      -- binop
  | aluConvert     -- cast/gep
  | reduction      -- reduce
  | loopStructure  -- range/endR/sink
  | tensorCore     -- wmma
  deriving BEq, Repr, Inhabited, DecidableEq

def OpClass.toStr : OpClass → String
  | .leafIO => "leaf-io" | .movementView => "movement-view"
  | .memoryAccess => "memory-access" | .aluBinary => "alu-binary"
  | .aluConvert => "alu-convert" | .reduction => "reduction"
  | .loopStructure => "loop-structure" | .tensorCore => "tensor-core"

/-- GROUNDING HOOK. Total over the real `Tgrad.UOp`, no wildcard.
    Add or rename a `UOp` constructor and this stops compiling. -/
def classify : UOp → OpClass
  | .param ..     => .leafIO
  | .const ..     => .leafIO
  | .vconst ..    => .leafIO
  | .defineVar .. => .leafIO
  | .special ..   => .leafIO
  | .buffer ..    => .leafIO
  | .var ..       => .leafIO
  | .permute ..   => .movementView
  | .reshape ..   => .movementView
  | .expand ..    => .movementView
  | .slice ..     => .movementView
  | .index ..     => .memoryAccess
  | .load ..      => .memoryAccess
  | .store ..     => .memoryAccess
  | .binop ..     => .aluBinary
  | .cast ..      => .aluConvert
  | .gep ..       => .aluConvert
  | .reduce ..    => .reduction
  | .range ..     => .loopStructure
  | .endR ..      => .loopStructure
  | .sink ..      => .loopStructure
  | .wmma ..      => .tensorCore

/-! ## The dtype axis — GROUNDED against `Tgrad.Dtype` -/

inductive DtypeClass
  | boolean | weak | integer | lowFloat | fullFloat | voidLike
  deriving BEq, Repr, Inhabited, DecidableEq

def DtypeClass.toStr : DtypeClass → String
  | .boolean => "boolean" | .weak => "weak" | .integer => "integer"
  | .lowFloat => "low-float" | .fullFloat => "full-float" | .voidLike => "void"

/-- GROUNDING HOOK. Total over the real `Tgrad.Dtype`, no wildcard. -/
def dtypeClassOf : Dtype → DtypeClass
  | .bool_     => .boolean
  | .weakint_  => .weak
  | .int8_  | .uint8_  | .int16_ | .uint16_
  | .int32_ | .uint32_ | .int64_ | .uint64_ => .integer
  | .float16_ | .bfloat16_ => .lowFloat
  | .float32_ | .float64_  => .fullFloat
  | .void_     => .voidLike

/-! ## The remaining axes (authored judgment, not grounded) -/

inductive BackendClass | metal | cuda | rocm | cpu | webgpu
  deriving BEq, Repr, Inhabited, DecidableEq
inductive ExecMode | singleKernel | multiKernelInfer | training
  deriving BEq, Repr, Inhabited, DecidableEq
inductive SchedMode | fixedBeam0 | beamSearch
  deriving BEq, Repr, Inhabited, DecidableEq

def BackendClass.toStr : BackendClass → String
  | .metal => "metal" | .cuda => "cuda" | .rocm => "rocm" | .cpu => "cpu" | .webgpu => "webgpu"
def ExecMode.toStr : ExecMode → String
  | .singleKernel => "single-kernel" | .multiKernelInfer => "multi-kernel-infer" | .training => "training"
def SchedMode.toStr : SchedMode → String
  | .fixedBeam0 => "fixed-beam0" | .beamSearch => "beam-search"

/-! ## Capabilities and the spec -/

inductive Capability
  | op      (k : OpClass)
  | dtype   (d : DtypeClass)
  | backend (b : BackendClass)
  | exec    (m : ExecMode)
  | sched   (s : SchedMode)
  deriving BEq, Repr, Inhabited, DecidableEq

def Capability.toStr : Capability → String
  | .op k      => "op:" ++ k.toStr
  | .dtype d   => "dtype:" ++ d.toStr
  | .backend b => "backend:" ++ b.toStr
  | .exec m    => "exec:" ++ m.toStr
  | .sched s   => "sched:" ++ s.toStr

/-- Enumerations of each axis (reused by `allCapabilities` and by the
    S0→S1 bridge in `Model.Plan`). -/
def allOpClasses : List OpClass :=
  [.leafIO, .movementView, .memoryAccess, .aluBinary, .aluConvert,
   .reduction, .loopStructure, .tensorCore]
def allDtypeClasses : List DtypeClass := [.boolean, .weak, .integer, .lowFloat, .fullFloat]
def allBackends : List BackendClass := [.metal, .cuda, .rocm, .cpu, .webgpu]
def allExecModes : List ExecMode := [.singleKernel, .multiKernelInfer, .training]
def allSchedModes : List SchedMode := [.fixedBeam0, .beamSearch]

/-- The finite capability universe used for the frontier. `voidLike`
    is excluded — it is a sentinel dtype, not a tensor capability. -/
def allCapabilities : List Capability :=
  (allOpClasses.map Capability.op) ++
  (allDtypeClasses.map Capability.dtype) ++
  (allBackends.map Capability.backend) ++
  (allExecModes.map Capability.exec) ++
  (allSchedModes.map Capability.sched)

/-- Current product maturity (authored judgment; the S0 reconcile PF will
    eventually derive the grounded parts from source). Ops are `wired`
    (used by the matmul hot path) but not `verified` as general ops;
    `tensorCore` is the optimized perf path; `reduction` is `modeled`
    only (the `reduce` constructor exists but no runtime reduction). -/
def current : Capability → Maturity
  | .op .leafIO        => .wired
  | .op .movementView  => .wired
  | .op .memoryAccess  => .wired
  | .op .aluBinary     => .wired
  | .op .aluConvert    => .wired
  | .op .loopStructure => .wired
  | .op .reduction     => .modeled
  | .op .tensorCore    => .optimized
  | .dtype .lowFloat   => .optimized
  | .dtype .fullFloat  => .modeled
  | .dtype .integer    => .absent
  | .dtype .boolean    => .absent
  | .dtype .weak       => .modeled
  | .dtype .voidLike   => .absent
  | .backend .metal    => .optimized
  | .backend .cuda     => .absent
  | .backend .rocm     => .absent
  | .backend .cpu      => .absent
  | .backend .webgpu   => .absent
  | .exec .singleKernel     => .verified
  | .exec .multiKernelInfer => .absent
  | .exec .training         => .absent
  | .sched .fixedBeam0 => .verified
  | .sched .beamSearch => .absent

/-- "Full tinygrad" target: every capability carried to at least
    `verified` (correct + supported). `optimized` is a stretch rung. -/
def target : Capability → Maturity := fun _ => .verified

/-- The production frontier: capabilities still below target. Each
    becomes (after S1 decomposition) one or more factory work orders. -/
def frontier : List Capability :=
  allCapabilities.filter (fun c => Maturity.below (current c) (target c))

def frontierReport : String :=
  let lines := frontier.map (fun c =>
    "- " ++ c.toStr ++ " : " ++ (current c).toStr ++ " -> " ++ (target c).toStr)
  "frontier (" ++ toString frontier.length ++ " capabilities below target):\n"
    ++ String.intercalate "\n" lines

/-! ## Grounding smoke (classify / dtypeClassOf are executable) -/

#guard classify (.binop .add (.const .int32_ (.i 0)) (.const .int32_ (.i 0)) .int32_) == .aluBinary
#guard classify (.permute (.buffer 0 [4, 4] .bfloat16_) [1, 0]) == .movementView
#guard classify (.wmma "w"
        (.const .bfloat16_ (.i 0)) (.const .bfloat16_ (.i 0)) (.const .float32_ (.i 0))
        .bfloat16_ .float32_ { M := 8, N := 8, K := 8 }) == .tensorCore
#guard dtypeClassOf .bfloat16_ == .lowFloat
#guard dtypeClassOf .int32_ == .integer

/-! ## Queryable facts (the spec answers structured questions) -/

/-- The frontier is non-empty: there is production work to do. -/
theorem frontier_nonempty : frontier.isEmpty = false := by native_decide

/-- The bf16 Metal matmul (tensor-core path) is already at/above target,
    so it is NOT on the frontier. -/
theorem matmul_done :
    Maturity.below (current (.op .tensorCore)) (target (.op .tensorCore)) = false := by
  native_decide

/-- Training is a pending capability (qualitatively the hardest: needs a
    backward transform, not just more ops). -/
theorem training_pending : frontier.contains (.exec .training) = true := by native_decide

/-- A second backend is pending (forces the renderer/runtime abstraction). -/
theorem cuda_pending : frontier.contains (.backend .cuda) = true := by native_decide

/-- Schedule search is pending (its oracle cannot be byte-equality). -/
theorem beam_search_pending : frontier.contains (.sched .beamSearch) = true := by native_decide

end Model
end Tgrad

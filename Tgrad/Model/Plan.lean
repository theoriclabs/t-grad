import Tgrad.Model.Spec
import Tgrad.Model.Impact

/-! # Tgrad.Model.Plan — the S0 → S1 bridge

  Connects the product specification (`Spec`, S0: where are we / where to)
  to the change calculus (`Impact`, S1: how, in what order). The thesis
  of `docs/TGRAD_PRODUCTION_MODEL.md` is "spec diff drives work orders";
  this module is that arrow, executable:

      frontier  ──capabilityToChange──▶  DesiredChange  ──impactOf──▶  plan

  Two seams are surfaced honestly rather than hidden, because the two
  modules were authored independently (Spec here, Impact by a sibling
  agent) and wiring them exposes real gaps:

  * **Vocabulary gap.** `Impact.ChangeSubject` has no subject for the op
    axis (elementwise / reduction). Every op-class frontier capability is
    therefore unmappable today — proved by `op_axis_has_no_change_subject`.
  * **Tunnel under-detection (fixed here).** `requiresModeledSpecPromotion`
    flags the graph-realize / rangeify / schedule-search promotions but NOT
    the renderer/runtime backend abstraction, which is also a structural
    tunnel. The bridge defines a complete `changeIsTunnel` (below) so
    backends are sequenced as tunnels, and proves it (`cuda_is_tunnel`).
-/

namespace Tgrad
namespace Model

/-- Map a frontier capability to a request in the S1 change language, if
    one exists. `none` marks an open vocabulary gap (see module header).
    This is authored judgment (S1 side), not a grounding hook. -/
def capabilityToChange : Capability → Option DesiredChange
  | .dtype .lowFloat  => some { verb := .add, subject := .runtimeDtype .float16_,
                                constraints := [.keepPythonThin, .requireRuntimeExecution] }
  | .dtype .fullFloat => some { verb := .add, subject := .runtimeDtype .float32_,
                                constraints := [.keepPythonThin] }
  | .dtype .integer   => some { verb := .add, subject := .runtimeDtype .int8_,
                                constraints := [.keepPythonThin] }
  | .dtype .boolean   => some { verb := .add, subject := .runtimeDtype .bool_, constraints := [] }
  | .dtype .weak      => none   -- weakint is an internal lattice element, not a runtime dtype
  | .dtype .voidLike  => none
  | .backend .metal   => none   -- already optimized; not on the frontier
  | .backend .cuda    => some { verb := .add, subject := .backend .cuda,
                                constraints := [.preferPureLeanModel, .performanceRelevant] }
  | .backend .rocm    => some { verb := .add, subject := .backend .rocm, constraints := [.preferPureLeanModel] }
  | .backend .cpu     => some { verb := .add, subject := .backend .cpu, constraints := [.preferPureLeanModel] }
  | .backend .webgpu  => some { verb := .add, subject := .backend .webgpu, constraints := [.preferPureLeanModel] }
  | .exec .singleKernel     => none
  | .exec .multiKernelInfer => some { verb := .growToward, subject := .workload .transformerInference,
                                      constraints := [.preferPureLeanModel, .requireRuntimeExecution] }
  | .exec .training         => some { verb := .growToward, subject := .workload .fullInference,
                                      constraints := [.preferPureLeanModel, .requireRuntimeExecution] }
  | .sched .fixedBeam0 => none
  | .sched .beamSearch => some { verb := .add, subject := .scheduleSearch, constraints := [.performanceRelevant] }
  | .op _ => none   -- FINDING: the S1 change language has no op-vocabulary subject yet

/-- Structural abstractions whose introduction is a "tunnel": a spine
    rebuild that must precede the additive work depending on it. Broader
    than `requiresModeledSpecPromotion`, which misses the renderer/runtime
    backend abstractions. (Refinement for later: only the FIRST backend
    forces `rendererTypeclass`; subsequent backends are then leaves.) -/
def structuralBlockers : List MissingAbstraction :=
  [.generalGraphRealize, .generalRangeify, .scheduleSearchState,
   .rendererTypeclass, .backendDescriptor, .runtimeBackend]

/-- Complete tunnel test: a modeled-spec promotion, or a change that
    blocks on any structural abstraction. -/
def changeIsTunnel (ch : DesiredChange) : Bool :=
  requiresModeledSpecPromotion ch || (blockersOf ch).any (fun b => structuralBlockers.contains b)

/-- One planned production step: a frontier capability, its S1 change (if
    expressible), the change's blast-radius size + blockers, and whether
    it is a structural "tunnel" (Ring-3 promotion). -/
structure PlanItem where
  capability        : Capability
  change            : Option DesiredChange
  impactCount       : Nat
  blockers          : List MissingAbstraction
  requiresPromotion : Bool
  deriving Repr, Inhabited

def planItemFor (c : Capability) : PlanItem :=
  match capabilityToChange c with
  | some ch => { capability := c, change := some ch,
                 impactCount := (impactOf ch).length, blockers := blockersOf ch,
                 requiresPromotion := changeIsTunnel ch }
  | none    => { capability := c, change := none, impactCount := 0,
                 blockers := [], requiresPromotion := false }

/-- The full plan: one item per frontier capability. -/
def productionPlan : List PlanItem := frontier.map planItemFor

def actionableItems : List PlanItem := productionPlan.filter (fun i => i.change.isSome)
def tunnelItems     : List PlanItem := productionPlan.filter (fun i => i.requiresPromotion)
def leafItems       : List PlanItem :=
  productionPlan.filter (fun i => i.change.isSome && !i.requiresPromotion)
def unmappedCapabilities : List Capability :=
  (productionPlan.filter (fun i => i.change.isNone)).map (fun i => i.capability)

/-- Coarse production ordering: structural tunnels first (they rebuild the
    spine and gate on "nothing regresses"), then additive leaves. Unmapped
    capabilities are reported separately as open vocabulary work. -/
def orderedPlan : List PlanItem := tunnelItems ++ leafItems

/-! ## Report surface (the model displays its own plan) -/

def planItemLine (i : PlanItem) : String :=
  let chg := match i.change with
    | some c => c.toStr
    | none   => "(no S1 change subject — open vocabulary gap)"
  let tun := if i.requiresPromotion then "  [TUNNEL]" else ""
  "  - " ++ i.capability.toStr ++ "  =>  " ++ chg ++ tun

def planReport : String :=
  let tunnels := tunnelItems.map planItemLine
  let leaves  := leafItems.map planItemLine
  let gaps    := unmappedCapabilities.map (fun c => "  - " ++ c.toStr)
  "PRODUCTION PLAN to full TGrad (" ++ toString frontier.length ++ " frontier capabilities)\n\n"
    ++ "Tunnels — structural promotions, sequence first:\n" ++ String.intercalate "\n" tunnels ++ "\n\n"
    ++ "Leaves — additive:\n" ++ String.intercalate "\n" leaves ++ "\n\n"
    ++ "Unmapped — S1 change language gaps:\n" ++ String.intercalate "\n" gaps

#eval IO.println planReport

/-! ## Queryable facts about the plan -/

/-- The bridge produces actionable work — the spec diff is non-vacuous. -/
theorem plan_has_actionable_items : actionableItems.isEmpty = false := by native_decide

/-- FINDING: the S1 change language has no subject for the op axis, so
    every op-class capability is unmappable. Extending `ChangeSubject` to
    cover elementwise/reduction ops is open work this surfaces. -/
theorem op_axis_has_no_change_subject :
    allOpClasses.all (fun k => (capabilityToChange (.op k)).isNone) = true := by native_decide

/-- Training is a structural tunnel — it forces a modeled-spec promotion. -/
theorem training_is_tunnel :
    (planItemFor (.exec .training)).requiresPromotion = true := by native_decide

/-- Schedule search is a tunnel (its oracle cannot be byte-equality). -/
theorem beam_search_is_tunnel :
    (planItemFor (.sched .beamSearch)).requiresPromotion = true := by native_decide

/-- A new backend is a tunnel: it blocks on the renderer/runtime backend
    abstraction. The bridge surfaces this; `requiresModeledSpecPromotion`
    alone would have mis-sequenced it as an additive leaf. -/
theorem cuda_is_tunnel :
    (planItemFor (.backend .cuda)).requiresPromotion = true := by native_decide

/-- Adding a runtime dtype is additive, NOT a tunnel. The plan
    distinguishes additive from structural — the distinction a planner
    needs to sequence tunnels before leaves. -/
theorem dtype_is_additive :
    (planItemFor (.dtype .integer)).requiresPromotion = false := by native_decide

end Model
end Tgrad

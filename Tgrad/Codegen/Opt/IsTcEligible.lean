import Tgrad.Codegen.Opt.Tc

/-! # Tgrad.Codegen.Opt.IsTcEligible — TC eligibility predicate

  Lift from theograd_phases/08_wmma_matching/Demo.lean. The predicate
  half of `_apply_tc_opt` (the rewrite half is `Opt.Apply`).
-/
namespace Tgrad

namespace Codegen

namespace Opt

structure TcQuery where
  m        : Nat
  n        : Nat
  k        : Nat
  dtypeIn  : Dtype
  dtypeOut : Dtype
  deriving Repr, Inhabited, DecidableEq

/-- `isTcEligible inv q` finds the first TC tile in `inv` whose
    `(dtypeIn, dtypeOut)` matches `q` and whose `dims=(d0,d1,d2)`
    divides `(q.m, q.n, q.k)` respectively. Mirrors the inner loop of
    `_apply_tc_opt` (postrange.py:225-235). -/
def isTcEligible (inv : List TensorCore) (q : TcQuery) : Option TensorCore :=
  inv.find? (fun tc =>
    let dimsOk : Bool := match tc.dims with
      | [d0, d1, d2] =>
          d0 != 0 && d1 != 0 && d2 != 0 &&
          q.m % d0 == 0 && q.n % d1 == 0 && q.k % d2 == 0
      | _ => false
    tc.dtypeIn == q.dtypeIn && tc.dtypeOut == q.dtypeOut && dimsOk)

-- ============================================================================
-- Static BL spot-checks (G3 deliverable per phase 08 v2 — `decide`-proved).
-- ============================================================================

/-- The (M=64, N=64, K=64, bf16→f32) query matches Metal's 8×8×8 tile. -/
theorem tc_eligible_64x64_bf16_f32 :
    (isTcEligible metalTensorCores
      { m := 64, n := 64, k := 64, dtypeIn := .bfloat16_, dtypeOut := .float32_ }).isSome
      = true := by
  decide

/-- The (M=4, N=4, K=4, bf16→f32) query is ineligible: 4 is not a
    multiple of 8. -/
theorem tc_ineligible_4x4_bf16_f32 :
    (isTcEligible metalTensorCores
      { m := 4, n := 4, k := 4, dtypeIn := .bfloat16_, dtypeOut := .float32_ }).isSome
      = false := by
  decide

/-- The (M=64, N=64, K=64, int32→int32) query is ineligible: Metal's
    inventory has no int32 TC tile. -/
theorem tc_ineligible_int32 :
    (isTcEligible metalTensorCores
      { m := 64, n := 64, k := 64, dtypeIn := .int32_, dtypeOut := .int32_ }).isSome
      = false := by
  decide

end Opt

end Codegen

end Tgrad

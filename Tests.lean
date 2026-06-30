import Tgrad

/-! # Tgrad layer-2 tests entry

  Walks each module's unit fixtures in one Lean executable. Replaces
  the 17 per-phase Mains with one orchestrator. See `01_design.md` §8.

  At L0: smoke-checks the lifted modules load.
  At L1: smoke-checks Dtype.lub + Shape.numel + UOp.binop + UPat.add
         construct and project. Behaviour is byte-diffed by the gate
         script — these prints are noise filters, not the predicate.
-/

open Tgrad

def main : IO Unit := do
  -- L0 smoke (preserved verbatim — the gate's stdout grep).
  let d := Dtype.bfloat16_
  IO.println s!"  ✓ Dtype: {d.toStr}, {d.sizeBytes} bytes"
  let s : SintShape := [.nat 64, .nat 64]
  IO.println s!"  ✓ SintShape: numel={repr (sintNumel s)}"
  let u : UOp := .const .float32_ (.f 0.0)
  let _ := u
  IO.println "  ✓ UOp.const constructed"
  -- L1 smoke.
  IO.println s!"  ✓ Dtype.lub bf16 f32 = {(Dtype.lub .bfloat16_ .float32_).toStr}"
  let cs : Shape := [2, 3, 4]
  IO.println s!"  ✓ Shape.numel [2,3,4] = {numel cs}"
  let bin : UOp := .binop .add (.const .int32_ (.i 1)) (.const .int32_ (.i 2)) .int32_
  IO.println s!"  ✓ UOp.binop kind = {bin.kind.toStr}"
  let pat := UPat.add (UPat.var "x") (UPat.constInt 0)
  let _ := pat
  IO.println "  ✓ UPat.add (var x) (const 0) constructed"
  IO.println s!"  ✓ Rules.Symbolic.ruleSet has {Rules.Symbolic.ruleSet.length} rules"
  IO.println "tgrad-tests: scaffold layer ✓"

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

/-- Minimal assertion harness. Returns 1 on failure so callers can sum. -/
def check (name : String) (cond : Bool) : IO Nat := do
  if cond then
    IO.println s!"  ✓ {name}"
    pure 0
  else
    IO.eprintln s!"  ✗ FAIL {name}"
    pure 1

/-- Render an index UOp to its C-ish string, for comparison. -/
private def idxStr (u : UOp) : String := UOp.renderIndexExpr u

/-- Tests for `Schedule.View` — the movement-op algebra that replaced
    `Pipeline.viewIndexUOpForA/B`'s hand-enumerated pattern table.

    The two `BUG` cases are regressions for silent wrong-answer bugs
    in that table: it discarded every slice axis after the first, and
    it assumed a broadcast axis was always axis 1. -/
def runViewTests : IO Nat := do
  let mut f := 0
  let gid : UOp := .var "gidx0" .int32_
  let rid : UOp := .var "Ridx0" .int32_
  let gid1 : UOp := .var "gidx1" .int32_

  -- Contiguous row-major.
  let vc := Schedule.View.contiguous [64, 32]
  f := f + (← check "View.contiguous [64,32] strides = [32,1]"
    (vc.strides == [32, 1]))
  f := f + (← check "View.indexOf contiguous = (gidx0*32)+Ridx0"
    (idxStr (Schedule.View.indexOf vc [gid, rid]) == "((gidx0*32)+Ridx0)"))

  -- Transpose: strides swap, and the emitted form matches what the
  -- old hand-written arm produced.
  match (Schedule.View.contiguous [8, 4]).permute [1, 0] with
  | none   => f := f + (← check "View.permute [1,0] succeeds" false)
  | some v =>
    f := f + (← check "View.permute [1,0] shape = [4,8]" (v.shape == [4, 8]))
    f := f + (← check "View.permute [1,0] strides = [1,4]" (v.strides == [1, 4]))
    f := f + (← check "View.indexOf transpose = (Ridx0*4)+gidx0"
      (idxStr (Schedule.View.indexOf v [gid, rid]) == "((Ridx0*4)+gidx0)"))

  -- Duplicate axes are not a permutation; the old `permuteShape`
  -- accepted [0,0].
  f := f + (← check "View.permute rejects duplicate axes [0,0]"
    (((Schedule.View.contiguous [2, 3]).permute [0, 0]).isNone))

  -- BUG 1: multi-axis slice. Every sliced axis must reach the offset.
  match (Schedule.View.contiguous [64, 64]).slice
          [{ start := 0, stop := 8, step := 1 },
           { start := 4, stop := 12, step := 1 }] with
  | none   => f := f + (← check "View.slice 2-axis succeeds" false)
  | some v =>
    f := f + (← check "BUG1 slice a[0:8,4:12] shape = [8,8]" (v.shape == [8, 8]))
    f := f + (← check "BUG1 slice a[0:8,4:12] offset = 4 (was dropped)"
      (v.offset == 4))

  -- Strided slice folds the step into the stride.
  match (Schedule.View.contiguous [64, 64]).slice
          [{ start := 2, stop := 64, step := 2 }] with
  | none   => f := f + (← check "View.slice strided succeeds" false)
  | some v =>
    f := f + (← check "slice a[2::2] strides = [128,1]" (v.strides == [128, 1]))
    f := f + (← check "slice a[2::2] offset = 128" (v.offset == 128))

  -- BUG 2: broadcast on axis 0 must give stride 0 on axis 0.
  match (Schedule.View.contiguous [1, 64]).expand [64, 64] with
  | none   => f := f + (← check "View.expand axis-0 succeeds" false)
  | some v =>
    f := f + (← check "BUG2 expand (1,64)->(64,64) strides = [0,1]"
      (v.strides == [0, 1]))
    f := f + (← check "BUG2 expand axis-0 index drops the K term"
      (idxStr (Schedule.View.indexOf v [rid, gid1]) == "gidx1"))

  -- Broadcast on axis 1, the case the old arm did handle.
  match (Schedule.View.contiguous [64, 1]).expand [64, 64] with
  | none   => f := f + (← check "View.expand axis-1 succeeds" false)
  | some v =>
    f := f + (← check "expand (64,1)->(64,64) strides = [1,0]"
      (v.strides == [1, 0]))

  -- Expand may not resize a non-singleton axis.
  f := f + (← check "View.expand rejects 2 -> 4"
    (((Schedule.View.contiguous [2, 4]).expand [4, 4]).isNone))

  -- Reshape is sound only on a contiguous view.
  f := f + (← check "View.reshape [4,8] -> [32] ok"
    (((Schedule.View.contiguous [4, 8]).reshape [32]).isSome))
  f := f + (← check "View.reshape rejects numel change"
    (((Schedule.View.contiguous [4, 8]).reshape [33]).isNone))
  match (Schedule.View.contiguous [4, 8]).permute [1, 0] with
  | none   => f := f + (← check "permute for reshape test" false)
  | some t =>
    f := f + (← check "View.reshape rejects non-contiguous (transposed)"
      ((t.reshape [32]).isNone))

  -- Chained movement ops compose.
  let chain : UOp :=
    .slice (.permute (.buffer 0 [8, 16] .bfloat16_) [1, 0])
           [{ start := 2, stop := 10, step := 1 }]
  f := f + (← check "viewOfUOp composes permute∘slice"
    ((Schedule.viewOfUOp chain).isSome))

  -- Regression: rangeify must not be the identity, and must remove
  -- movement nodes.
  let mv : UOp := .permute (.buffer 0 [4, 8] .bfloat16_) [1, 0]
  f := f + (← check "rangeify is not the identity"
    (!((Schedule.Rangeify.rangeify mv).beq mv)))
  f := f + (← check "rangeify eliminates movement nodes"
    (UOp.countMovementNodes (Schedule.Rangeify.rangeify mv) == 0))

  pure f

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
  -- Assertions. Everything above this line only prints; a failure
  -- there exits 0 and the gate greps stdout. Everything below fails
  -- the process.
  let failures ← runViewTests
  if failures != 0 then
    IO.eprintln s!"tgrad-tests: {failures} assertion(s) FAILED"
    IO.Process.exit 1
  IO.println "tgrad-tests: assertions ✓"

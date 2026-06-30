import Tgrad.GraphRewrite
import Tgrad.Schedule.Rangeify

/-! # Tgrad.Codegen.Linearize — DFS post-order toposort

  Lift from theograd_phases/07_linearization/Demo.lean.

  v1 narrow scope: DFS post-order with structural-identity dedup,
  no priority heap. Matches `UOp.toposort()` semantics on the
  phase-07 fixture (no RANGE/END/LOAD/STORE). The full linearizer
  with priority weighting (PARAM=-20, RANGE=5, END=-5, LOAD=-1,
  STORE=1) is L9+.

  The serialised emit shape mirrors `index_out_expected.json` from
  phase-04 — `{records, order}` — except `order` is the identity
  list since `linearize` already returns the post-order.
-/
namespace Tgrad

namespace Linearize

private abbrev WalkSt := Array UOp × Array (UOp × Nat)

private def findIdx (seen : Array (UOp × Nat)) (u : UOp) : Option Nat :=
  seen.foldl
    (fun acc (entry : UOp × Nat) =>
      match acc with
      | some _ => acc
      | none   => if entry.1.beq u then some entry.2 else none)
    none

mutual
partial def visitUOp (u : UOp) (st : WalkSt) : WalkSt :=
  match findIdx st.2 u with
  | some _ => st
  | none   =>
      let st1 := visitUOps u.children st
      let idx := st1.1.size
      (st1.1.push u, st1.2.push (u, idx))

partial def visitUOps (us : List UOp) (st : WalkSt) : WalkSt :=
  match us with
  | []      => st
  | u :: ts =>
      let st1 := visitUOp u st
      visitUOps ts st1
end

/-- DFS post-order toposort. Returns each unique UOp exactly once, in
    the order it's first completed (after all its sources have been
    visited). -/
def linearize (sink : UOp) : List UOp :=
  let (lst, _) := visitUOp sink (#[], #[])
  lst.toList

-- ============================================================================
-- JSON serialisation — `{records, order}` matching phase-07's
-- linear_out_expected.json layout. Reuses the L2 indent-shift trick.
-- ============================================================================

private def linearOrderJson (n : Nat) : String :=
  if n == 0 then "[]"
  else
    let xs := (List.range n).map toString
    "[\n    " ++ String.intercalate ",\n    " xs ++ "\n  ]"

/-- Convert a linearized UOp list to (records, order) and serialise. -/
def linearToJson (uops : List UOp) : String :=
  let (recs, _idxs) := Rangeify.toRecordsMulti uops
  let recsJson := recordsToJson recs
  let shifted  := recsJson.replace "\n" "\n  "
  "{\n  \"records\": " ++ shifted ++
  ",\n  \"order\": " ++ linearOrderJson recs.length ++ "\n}"

end Linearize

end Tgrad

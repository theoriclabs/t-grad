/-! # Tgrad.Schedule.Memory — greedy interval-graph coloring

  Lift from theograd_phases/05_memory_planning/Demo.lean.

  v1 narrow scope: pin the greedy algorithm against a hand-built
  5-buffer fixture. The full tinygrad-implicit allocator depends on
  phases 04-10 + the MetalCompiler — out of L2.
-/
namespace Tgrad

namespace Memory

/-- An Interval — (buf, first-use time, last-use time, size in bytes).
    `first` is inclusive (write); `last` is exclusive (no-longer-needed).
    Matches Python's half-open interval convention. -/
structure Interval where
  buf   : Nat
  first : Nat
  last  : Nat
  size  : Nat
  deriving Repr, BEq, Inhabited

structure Slot where
  physId  : Nat
  sizeMax : Nat
  endTime : Nat
  deriving Repr, Inhabited

structure Assignment where
  logical  : Nat
  physical : Nat
  deriving Repr, BEq, Inhabited

structure AssignmentTable where
  assignments  : List Assignment
  physicalCount : Nat
  peakBytes    : Nat
  deriving Repr, Inhabited

-- ============================================================================
-- Greedy first-fit on sorted intervals.
-- ============================================================================

/-- Order intervals by (first, last, buf) ascending — matches Python's
    `sorted(items, key=lambda x: (x.first, x.last, x.buf))`. -/
def sortIntervals (ivs : List Interval) : List Interval :=
  ivs.mergeSort (fun a b =>
    if a.first < b.first then true
    else if a.first > b.first then false
    else if a.last < b.last then true
    else if a.last > b.last then false
    else a.buf ≤ b.buf)

/-- Try to fit `it` into the first usable slot in `slots`. If found,
    return updated slots + the chosen physId. If not, append a new
    slot + return its physId. -/
def fitInterval (slots : List Slot) (it : Interval) : List Slot × Nat :=
  let rec loop (acc : List Slot) : List Slot → List Slot × Nat
    | []        =>
        let newId := acc.length
        let newSlot : Slot :=
          { physId := newId, sizeMax := it.size, endTime := it.last }
        (acc ++ [newSlot], newId)
    | s :: rest =>
        if it.first ≥ s.endTime ∧ it.size ≤ s.sizeMax then
          let updated : Slot := { s with endTime := it.last }
          (acc ++ [updated] ++ rest, s.physId)
        else
          loop (acc ++ [s]) rest
  loop [] slots

/-- Greedy assign: sort by first-use, fit each interval into the first
    slot it can reuse, else open a new slot. -/
def greedyAssign (ivs : List Interval) : AssignmentTable :=
  let sorted := sortIntervals ivs
  let result := sorted.foldl
    (fun (acc : List Slot × List Assignment) it =>
      let (slots, asgns) := acc
      let (slots', physId) := fitInterval slots it
      (slots', asgns ++ [{ logical := it.buf, physical := physId }]))
    ([], [])
  let (slots, asgns) := result
  let asgnsByLogical := asgns.mergeSort (fun a b => a.logical ≤ b.logical)
  let peak := slots.foldl (fun acc s => acc + s.sizeMax) 0
  { assignments  := asgnsByLogical,
    physicalCount := slots.length,
    peakBytes    := peak }

-- ============================================================================
-- Validation: no two overlapping intervals share a physical slot.
-- ============================================================================

/-- For each pair of intervals, if they're assigned the same physical
    slot, their live-ranges must NOT overlap. -/
def validateAssignment (ivs : List Interval) (table : AssignmentTable) : Bool :=
  let physOf : Nat → Option Nat := fun b =>
    (table.assignments.find? (fun a => a.logical == b)).map Assignment.physical
  let pairs := ivs.flatMap (fun a => ivs.map (fun b => (a, b)))
  pairs.all (fun pair =>
    let (a, b) := pair
    if a.buf == b.buf then true
    else match physOf a.buf, physOf b.buf with
      | some pa, some pb =>
          if pa == pb then a.last ≤ b.first ∨ b.last ≤ a.first else true
      | _, _ => true)

-- ============================================================================
-- JSON I/O — matches phase-05's assignment_expected.json layout.
-- ============================================================================

def Assignment.toJson (a : Assignment) : String :=
  "    {\n" ++
  "      \"logical\": "  ++ toString a.logical  ++ ",\n" ++
  "      \"physical\": " ++ toString a.physical ++ "\n    }"

def tableToJson (t : AssignmentTable) : String :=
  let asJson :=
    if t.assignments.isEmpty then "[]"
    else "[\n" ++ String.intercalate ",\n" (t.assignments.map Assignment.toJson) ++ "\n  ]"
  "{\n" ++
  "  \"assignment\": " ++ asJson ++ ",\n" ++
  "  \"physical_count\": " ++ toString t.physicalCount ++ ",\n" ++
  "  \"peak_bytes\": " ++ toString t.peakBytes ++ "\n}"

end Memory

end Tgrad

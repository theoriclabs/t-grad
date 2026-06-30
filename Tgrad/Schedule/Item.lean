import Tgrad.Schedule.Linear

/-! # Tgrad.Schedule.Item — per-kind ScheduleItem variants

  Lift + tighten from theograd_phases/06_scheduler/Demo.lean.

  Phase 06's v2 had a flat `ScheduleItem` structure with
  `functionName : Option String`, plus a runtime predicate
  `allSinksHaveName` that asserted SINK items always carry `some _`.

  Tgrad lifts the contract into the type system: `ScheduleItem` is a
  sum of per-kind structures. `SinkItem.functionName : String` is
  required by construction, so a SINK without a function name is
  impossible to express. `allSinksHaveName` becomes a structural
  theorem (provable by case analysis) rather than a runtime check.

  L1 gate-style negative test (in `scripts/gates/L2.sh`):
    def bad : ScheduleItem := .sink { bufferCount := 4, totalSrcCount := 4 }
  must fail to typecheck because SinkItem requires a `functionName : String`.
-/
namespace Tgrad

namespace Item

open Linear

/-- The required fields for a SINK kernel: function name + buffer
    counts. `functionName : String` is non-optional — the L2 invariant
    "every SINK item carries a function name" is now structural. -/
structure SinkItem where
  functionName  : String
  bufferCount   : Nat
  totalSrcCount : Nat
  deriving Repr, Inhabited

/-- COPY items have no function name (always null in the captured
    fixtures), just buffer counts. -/
structure CopyItem where
  bufferCount   : Nat
  totalSrcCount : Nat
  deriving Repr, Inhabited

/-- OtherItem absorbs unknown kinds so the captured fixture decodes
    losslessly. The raw kind string is kept for round-trip emit. -/
structure OtherItem where
  kindStr       : String
  bufferCount   : Nat
  totalSrcCount : Nat
  deriving Repr, Inhabited

/-- The typed schedule-item sum. A `.sink` payload is a `SinkItem`
    (function name required); `.copy` is a `CopyItem`; `.other`
    absorbs anything else. -/
inductive ScheduleItem where
  | sink  (s : SinkItem)
  | copy  (c : CopyItem)
  | other (o : OtherItem)
  deriving Repr, Inhabited

def ScheduleItem.kind : ScheduleItem → KernelKind
  | .sink _  => .sink_
  | .copy _  => .copy_
  | .other _ => .other_

def ScheduleItem.bufferCount : ScheduleItem → Nat
  | .sink s  => s.bufferCount
  | .copy c  => c.bufferCount
  | .other o => o.bufferCount

def ScheduleItem.totalSrcCount : ScheduleItem → Nat
  | .sink s  => s.totalSrcCount
  | .copy c  => c.totalSrcCount
  | .other o => o.totalSrcCount

/-- Helper for the structural theorem: does this item carry a function
    name? Only `.sink` does (by construction). -/
def ScheduleItem.hasFunctionName : ScheduleItem → Bool
  | .sink _ => true
  | _       => false

/-- Is this item a SINK? Used in the structural theorem statement. -/
def ScheduleItem.isSink : ScheduleItem → Bool
  | .sink _ => true
  | _       => false

abbrev DetailedSchedule := List ScheduleItem

-- ============================================================================
-- The L2 invariant — structural rather than runtime.
-- ============================================================================

/-- Every SINK item carries a function name. Trivially true by the
    per-kind variant: a `.sink` is `SinkItem` which has `functionName
    : String`, so `hasFunctionName` matches `isSink` identically. -/
theorem allSinksHaveName (d : DetailedSchedule) :
    d.all (fun it => !it.isSink || it.hasFunctionName) = true := by
  induction d with
  | nil       => rfl
  | cons h t ih =>
      cases h <;>
        simp [ScheduleItem.isSink, ScheduleItem.hasFunctionName] <;>
        exact ih

-- ============================================================================
-- v2-style summary on the typed DetailedSchedule.
-- ============================================================================

def summarizeDetailed (d : DetailedSchedule) : SummaryStats :=
  { kernelCount := d.length,
    callFirstKinds := d.map (fun it => it.kind.toStr) }

-- ============================================================================
-- JSON I/O — matches phase-06's sched_fused_detail.json layout.
-- ============================================================================

private def fnameJson (f : Option String) : String :=
  match f with
  | none   => "null"
  | some s => "\"" ++ s ++ "\""

def ScheduleItem.toJson (it : ScheduleItem) : String :=
  let kindStr := it.kind.toStr
  -- For SINK: function_name is the SinkItem.functionName (always present).
  -- For COPY / OTHER: function_name is always null.
  let fname :=
    match it with
    | .sink s => fnameJson (some s.functionName)
    | _       => fnameJson none
  "    {\n" ++
  "      \"kind\": \""          ++ kindStr                       ++ "\",\n" ++
  "      \"function_name\": "   ++ fname                          ++ ",\n" ++
  "      \"buffer_count\": "    ++ toString it.bufferCount       ++ ",\n" ++
  "      \"total_src_count\": " ++ toString it.totalSrcCount     ++ "\n    }"

def detailedToJson (d : DetailedSchedule) : String :=
  let itemsStr :=
    if d.isEmpty then "[]"
    else "[\n" ++ String.intercalate ",\n" (d.map ScheduleItem.toJson) ++ "\n  ]"
  "{\n  \"kernel_count\": " ++ toString d.length ++
  ",\n  \"items\": " ++ itemsStr ++ "\n}"

end Item

end Tgrad

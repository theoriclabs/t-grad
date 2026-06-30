/-! # Tgrad.Schedule.Linear — KernelKind + summary statistics

  Lift from theograd_phases/06_scheduler/Demo.lean — the v1 layer
  (KernelKind + Schedule + SummaryStats). The richer per-CALL detail
  (ScheduleItem + per-kind variants) lives in `Schedule/Item.lean`.
-/
namespace Tgrad

namespace Linear

/-- KernelKind — the top-level op of a CALL in tinygrad's schedule.
    `other_` absorbs anything else so captured fixtures decode losslessly. -/
inductive KernelKind where
  | copy_
  | sink_
  | other_
  deriving BEq, Repr, Inhabited, DecidableEq

def KernelKind.toStr : KernelKind → String
  | .copy_  => "COPY"
  | .sink_  => "SINK"
  | .other_ => "OTHER"

def KernelKind.ofString : String → KernelKind
  | "COPY"  => .copy_
  | "SINK"  => .sink_
  | _       => .other_

abbrev Schedule := List KernelKind

-- ============================================================================
-- Summary statistics.
-- ============================================================================

structure SummaryStats where
  kernelCount : Nat
  callFirstKinds : List String
  deriving Repr, Inhabited

def summarize (s : Schedule) : SummaryStats :=
  { kernelCount := s.length,
    callFirstKinds := s.map KernelKind.toStr }

/-- `summarize` is length-preserving by construction. -/
theorem summarize_len_preserved (s : Schedule) :
    (summarize s).kernelCount = s.length := rfl

/-- A fused fixture has exactly one SINK kernel; everything else is COPY. -/
def isFusedFixture (s : Schedule) : Bool :=
  let sinks := s.filter (fun k => k == .sink_)
  let copies := s.filter (fun k => k == .copy_)
  sinks.length == 1 ∧ copies.length == s.length - 1

/-- An unfused fixture has ≥ 2 SINK kernels. -/
def isUnfusedFixture (s : Schedule) : Bool :=
  let sinks := s.filter (fun k => k == .sink_)
  sinks.length ≥ 2

end Linear

end Tgrad

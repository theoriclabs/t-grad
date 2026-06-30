import Tgrad.Codegen.Opt.IsTcEligible

/-! # Tgrad.Codegen.Opt.Apply — applyTcOpt (capture-and-replay)

  Lift from theograd_phases/09_codegen_opt/Demo.lean (G4 deliverable).

  At L3, `applyTcOpt` is a capture-lookup function: it returns the
  pinned post-TC sha8 for the bf16 64×64 sentinel and `none` for
  every other input. The algebraic rewrite arrives at L8 per the
  brief.
-/
namespace Tgrad

namespace Codegen

namespace Opt

/-- OptOps — the 9 ctors mirroring `tinygrad.codegen.opt.OptOps`. -/
inductive OptOps where
  | tc_
  | upcast_
  | unroll_
  | local_
  | group_
  | grouptop_
  | nolocals_
  | padto_
  | swap_
  deriving BEq, Repr, Inhabited, DecidableEq

def OptOps.toStr : OptOps → String
  | .tc_       => "TC"
  | .upcast_   => "UPCAST"
  | .unroll_   => "UNROLL"
  | .local_    => "LOCAL"
  | .group_    => "GROUP"
  | .grouptop_ => "GROUPTOP"
  | .nolocals_ => "NOLOCALS"
  | .padto_    => "PADTO"
  | .swap_     => "SWAP"

inductive OptArg where
  | none
  | int_   (n : Int)
  | pair   (a b : Int)
  | triple (a b c : Int)
  deriving Repr, Inhabited

structure Opt where
  op   : OptOps
  axis : Option Nat
  arg  : OptArg
  deriving Repr, Inhabited

-- ============================================================================
-- Capture-and-replay table for the TC rewrite.
-- ============================================================================

/-- Captured sha8 of tinygrad's MetalRenderer output for the bf16
    64×64 matmul under USE_TC=1, BEAM=0, NOOPT=0. -/
def expectedPostTcMatmulSha8 : UInt32 := 0x820a2f5e

/-- Captured pre-TC sentinel — encodes the (M, N, K, dtype) signature
    of the bf16 64×64 matmul input. -/
def preTcMatmulSentinel : UInt32 := 0xb6f16064

/-- The captured Opt that fires TC on the bf16 64×64 matmul. -/
def capturedTcOpt : Opt :=
  { op := .tc_, axis := some 0, arg := .triple (-1) 0 1 }

/-- `applyTcOpt opt preSinkSha` — capture-and-replay of `apply_opt(TC)`.
    Returns the captured post-TC sha when given the captured Opt + the
    captured pre-sentinel; `none` otherwise. -/
def applyTcOpt (opt : Opt) (preSinkSha : UInt32) : Option UInt32 :=
  if opt.op == capturedTcOpt.op
     ∧ opt.axis == capturedTcOpt.axis
     ∧ preSinkSha == preTcMatmulSentinel
  then some expectedPostTcMatmulSha8
  else none

-- ============================================================================
-- BL spot-checks.
-- ============================================================================

theorem applyTcOpt_captured_matches :
    applyTcOpt capturedTcOpt preTcMatmulSentinel = some expectedPostTcMatmulSha8 := by
  rfl

theorem applyTcOpt_wrong_opt_rejected :
    applyTcOpt { op := .upcast_, axis := some 0, arg := .int_ 4 } preTcMatmulSentinel = none := by
  rfl

theorem applyTcOpt_wrong_sentinel_rejected :
    applyTcOpt capturedTcOpt 0xdeadbeef = none := by
  rfl

end Opt

end Codegen

end Tgrad

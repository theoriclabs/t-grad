/-! # Tgrad.Codegen.GpuDims — dispatch grid + threadgroup dims

  Captures the (grid, threadgroup) pair tinygrad's MetalProgram dispatch
  takes. Used by L4 (Runtime.MetalProgram) to translate a kernel
  signature into a Metal dispatch.

  Each matmul capture (theograd_phases/fixtures/matmul_captures/*.json)
  records the captured (grid, threadgroup, threads_total) per kernel —
  GpuDims is the typed Lean view.
-/
namespace Tgrad

namespace Codegen

/-- 3D dispatch dims (the per-dim sizes Metal takes). -/
structure Dim3 where
  x : Nat
  y : Nat
  z : Nat
  deriving Repr, Inhabited, DecidableEq

structure GpuDims where
  grid        : Dim3
  threadgroup : Dim3
  threadsTotal : Nat
  deriving Repr, Inhabited, DecidableEq

/-- The captured 64×64 bf16 matmul's dims. Pinned to phase-11's
    captured kernel:  grid (8, 8, 1) × threadgroup (32, 1, 1) =
    threads_total 2048. -/
def matmul64x64Dims : GpuDims :=
  { grid := { x := 8, y := 8, z := 1 },
    threadgroup := { x := 32, y := 1, z := 1 },
    threadsTotal := 2048 }

end Codegen

end Tgrad

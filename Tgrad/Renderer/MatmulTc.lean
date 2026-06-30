import Tgrad.Renderer.Metal
import Tgrad.Codegen.Opt.Heuristic

/-! # Tgrad.Renderer.MatmulTc — L13.F TC matmul KernelDecl for non-sentinel shapes

  Non-sentinel TC-eligible shapes must route through a Lean-owned
  WMMA path (NOT the scalar fallback, NOT the captured-MSL replay).
  `tcMatmulKernelDecl` is the pure function on `(M, K, N, plan)`
  that returns the `KernelDecl` for a WMMA-based matmul kernel. The
  L13.F gate (`scripts/gates/L13_F.sh`) is the predicate this
  module is built against.

  Eligibility (matches §3 scope):
    M ≥ 128 ∧ K ≥ 128 ∧ N ≥ 128 ∧ M % 32 = 0 ∧ K % 8 = 0 ∧ N % 128 = 0
    ∧ (M, K, N) ∉ L11 sentinel set.

  Kernel structure (per `Stmt.tcMatmulBody`):
    * dispatch grid = (M / 8, N / 8, 1), threadgroup = (32, 1, 1)
    * each simdgroup (32 threads, 1 warp) computes one 8×8 output
      tile via cooperative `simdgroup_load` + `simdgroup_multiply_accumulate`
    * K-loop iterates K / 8 times
    * final `simdgroup_store` writes the bf16-cast output back to data0

  The generated source contains `simdgroup_multiply_accumulate`
  (L13.F gate's Layer C1/C3 check).
-/
namespace Tgrad

namespace Renderer

namespace Metal

inductive CodegenError where
  | notTcEligible (M K N : Nat) (reason : String)
  deriving Repr, Inhabited

/-- L13.F: TC matmul KernelDecl generator. Pure function on
    `(M, K, N)`. Returns `Except CodegenError KernelDecl`.

    No `IO` in the signature (Layer D1 check). The generated body's
    sole Stmt is `Stmt.tcMatmulBody M K N`, which renders the full
    WMMA matmul body. -/
def tcMatmulKernelDecl (M K N : Nat) : Except CodegenError KernelDecl :=
  if M < 128 ∨ K < 128 ∨ N < 128 then
    .error (.notTcEligible M K N "dim < 128")
  else if M % 32 != 0 ∨ K % 8 != 0 ∨ N % 128 != 0 then
    .error (.notTcEligible M K N "alignment")
  else
    .ok {
      name     := s!"matmul_tc_{M}x{K}x{N}",
      wmmaArgs := [],
      args     := [
        .buffer { qualifier := "device", baseType := "bfloat", name := "data0" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data1" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data2" },
        .attr   { baseType := "uint3", name := "gid",
                  attrStr := "[[threadgroup_position_in_grid]]" },
        .attr   { baseType := "uint3", name := "lid",
                  attrStr := "[[thread_position_in_threadgroup]]" },
      ],
      body     := [.tcMatmulBody M K N],
      trailingNewline := false
    }

/-- L13.F.STRICT.B: manual-load TC matmul KernelDecl generator. Pure
    function on `(M, K, N)`.

    This emits the tinygrad-shaped 32M × 128N / 4-warp threadgroup
    kernel. The body uses explicit bfloat fragment loads into the
    WMMA prelude's `thread_elements` path rather than Metal's
    cooperative `simdgroup_load`. The leading threadgroup declaration
    and barrier exercise the L13.F.STRICT.A grammar and make the route
    structurally distinguishable in gate checks. -/
def tcMatmulKernelDeclManualLoad (M K N : Nat) : Except CodegenError KernelDecl :=
  if M < 128 ∨ K < 128 ∨ N < 128 then
    .error (.notTcEligible M K N "dim < 128")
  else if M % 32 != 0 ∨ K % 8 != 0 ∨ N % 128 != 0 then
    .error (.notTcEligible M K N "alignment")
  else
    .ok {
      name     := s!"matmul_tc_manual_{M}x{K}x{N}",
      wmmaArgs := [WmmaArg.bf16Float],
      args     := [
        .buffer { qualifier := "device", baseType := "bfloat", name := "data0" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data1" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data2" },
        .attr   { baseType := "uint3", name := "gid",
                  attrStr := "[[threadgroup_position_in_grid]]" },
        .attr   { baseType := "uint3", name := "lid",
                  attrStr := "[[thread_position_in_threadgroup]]" },
      ],
      body     := [
        .threadgroupDecl "bfloat" "tg_a" 256,
        .threadgroupDecl "bfloat" "tg_b" 1024,
        .tcManualLoadMatmulBody M K N
      ],
      trailingNewline := false
    }

end Metal

end Renderer

end Tgrad

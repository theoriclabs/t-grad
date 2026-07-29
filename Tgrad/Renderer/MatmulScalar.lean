import Tgrad.Renderer.Elementwise

/-! # Tgrad.Renderer.MatmulScalar — L13.B scalar matmul KernelDecl

  For shapes below the TC tile multiplier (M, K, or N < 8), the WMMA
  `simdgroup_multiply_accumulate` primitive doesn't apply. L13.B
  emits a naive O(MNK) scalar matmul kernel parameterised by (M, K, N)
  — one thread per output element, no shared memory, no TC.

  This is a pure algebraic template that handles arbitrary
  `(M, K, N)` outside the generated tensor-core domain. Correctness only —
  no perf parity is required for the L13.B sub-gate per the
  manifest's `bucket == "below_tc_tile"` scope.

  Rendered shape (for M=K=N=4):
  ```
  #include <metal_stdlib>
  using namespace metal;
  kernel void matmul_scalar_4x4x4(device ushort* data0, device ushort* data1, device ushort* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    int gidx0 = gid.x;
    int gidx1 = gid.y;
    float acc = 0.0f;
    for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
      float val0 = as_type<float>(((uint)(*(data1+(gidx0*4+Ridx0)))) << 16);
      float val1 = as_type<float>(((uint)(*(data2+(Ridx0*4+gidx1)))) << 16);
      acc = (acc + val0 * val1);
    }
    // Explicit round-to-nearest-even bf16 packing to ushort follows.
  }
  ```

  Dispatch: grid=(M, N, 1), threadgroup=(1, 1, 1). One thread per
  output element. The kernel name embeds (M, K, N) so the cache
  keys by shape.
-/
namespace Tgrad

namespace Renderer

namespace Metal

/-- L14.B.2.c: parametric scalar matmul kernel. The A and B load
    index UOps drive the inner-loop pointer-arith via
    `UOp.renderIndexExpr`. The kernel suffix encodes (M, K, N) plus
    a `tag` string so different access patterns produce distinct
    kernel names (compile cache keys by name). -/
def scalarMatmulKernelDeclWithIdx
    (M K N : Nat) (aIdx bIdx : Tgrad.UOp) (tag : String) : KernelDecl :=
  { name     := s!"matmul_scalar_{tag}_{M}x{K}x{N}",
    wmmaArgs := [],
    args     := [
      .buffer { qualifier := "device", baseType := "ushort", name := "data0" },
      .buffer { qualifier := "device", baseType := "ushort", name := "data1" },
      .buffer { qualifier := "device", baseType := "ushort", name := "data2" },
      .attr   { baseType := "uint3", name := "gid",
                attrStr := "[[threadgroup_position_in_grid]]" },
      .attr   { baseType := "uint3", name := "lid",
                attrStr := "[[thread_position_in_threadgroup]]" },
    ],
    body     := [
      .declInt "gidx0" "gid.x" none,
      .declInt "gidx1" "gid.y" none,
      .declFloat "acc" "0.0f",
      .forLoop "Ridx0" K [
        loadScalarStmt "val0" "data1" aIdx .bfloat16_ "ushort",
        loadScalarStmt "val1" "data2" bIdx .bfloat16_ "ushort",
        .assign "acc"
          "(acc + val0 * val1)"
      ] ] ++
      -- Output store is row-major (always — the output is what the caller
      -- observes via numpy reshape) and uses portable bf16 RNE packing.
      storeScalarStmts "result" .bfloat16_ "ushort" "data0"
        (.binop .add
          (.binop .mul (.var "gidx0" .int32_)
                       (.const .int32_ (.i (Int.ofNat N))) .int32_)
          (.var "gidx1" .int32_) .int32_)
        "acc",
    trailingNewline := false }

/-- L14.B.2.b row-major default for A: `gidx0*K + Ridx0`. -/
def rowMajorAIdx (K : Nat) : Tgrad.UOp :=
  .binop .add
    (.binop .mul (.var "gidx0" .int32_)
                 (.const .int32_ (.i (Int.ofNat K))) .int32_)
    (.var "Ridx0" .int32_) .int32_

/-- L14.B.2.b row-major default for B: `Ridx0*N + gidx1`. -/
def rowMajorBIdx (N : Nat) : Tgrad.UOp :=
  .binop .add
    (.binop .mul (.var "Ridx0" .int32_)
                 (.const .int32_ (.i (Int.ofNat N))) .int32_)
    (.var "gidx1" .int32_) .int32_

/-- The L13.B scalar matmul kernel for arbitrary `(M, K, N)`. No
    WMMA prelude — one thread per output element, scalar inner-loop
    over K. Correctness-only path; not perf-optimised.

    L14.B.2.c: this remains the row-major entry point used by
    `tgrad_matmul_small_lean` (PythonFFI's L13.B/C/D path). The view
    path is `scalarMatmulKernelDeclWithIdx` driven from
    `Pipeline.realizeView`. -/
def scalarMatmulKernelDecl (M K N : Nat) : KernelDecl :=
  -- L14.B.2.c: row-major (BUFFER) access pattern. Tag "rm" makes the
  -- kernel name `matmul_scalar_rm_<M>x<K>x<N>` — a new prefix vs the
  -- pre-L14.B.2.c naming. Python callers reference the kernel name
  -- via `s!"matmul_scalar_{M}x{K}x{N}"` in `tgrad_matmul_small_lean`,
  -- so we keep BACKWARDS-COMPATIBLE naming for this default entry.
  { (scalarMatmulKernelDeclWithIdx M K N (rowMajorAIdx K) (rowMajorBIdx N) "rm") with
    name := s!"matmul_scalar_{M}x{K}x{N}" }

/-- Scalar matmul is the correctness fallback when native WMMA bf16 fails to
    compile, so its rendered source must not itself depend on native `bfloat`.
    This checks the full emitted kernel, including both indexed loads and the
    output conversion. -/
theorem bf16_scalar_matmul_render_is_portable :
    portableBf16ScalarSource
      (renderKernel (scalarMatmulKernelDecl 4 4 4))
      ["data0", "data1", "data2"] true = true := by
  native_decide

end Metal

end Renderer

end Tgrad

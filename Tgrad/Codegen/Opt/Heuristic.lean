import Tgrad.Codegen.Opt.Apply
import Tgrad.Codegen.GpuDims
import Tgrad.Renderer.Metal

/-! # Tgrad.Codegen.Opt.Heuristic — v1 opt-picking + dispatch-dim heuristic

  Lift from theograd_phases/09_codegen_opt/Demo.lean. The full
  `apply_opts` semantics + BEAM are out of v1 scope; this provides
  the wiring point used by phase 11 (realize_coordinator).

  L13.A scope (this commit): `pickDispatchPlan` ports the BEAM=0
  dispatch-dim selection for the matmul case in
  `tinygrad/codegen/opt/heuristic.py`. The function is exhaustive over
  the 11 L11 sentinel shapes:

  * `(64, 64, 64)`         : the L5.a sentinel — generated with two warps;
                              uses `(grid=(2,1,1), tg=(32,2,1))`.
  * `(M, K, N) ≥ 1024`     : the production formula
                              `grid=(N/128, M/32, 1)`, `tg=(32, 4, 1)`.
  * else                    : `none` — L13.B/C will fill in.

  The cross-check theorem `pickDispatchPlan_matches_capture` below
  proves (by `decide`) that this function agrees with the captured
  `Pipeline.dispatchDimsFor` table for all 11 sentinels. `Pipeline.lean`
  then *delegates* to `pickDispatchPlan` instead of duplicating the
  table, so a future drift between formula and captures is impossible
  (the theorem fails to type-check).
-/
namespace Tgrad

namespace Codegen

namespace Opt

structure ShapeDtypeContext where
  shape : List Nat   -- typical: (M, N, K) for matmul
  dtypeInName  : String
  dtypeOutName : String
  deriving Repr, Inhabited

/-- v1 heuristic: emit `[OptOps.TC]` when (dtypes match Metal's
    bf16/f32 pair) AND (all dims are multiples of 8). -/
def chooseOpts (ctx : ShapeDtypeContext) : List OptOps :=
  let dtypesOk := ctx.dtypeInName == "bfloat16" ∧ ctx.dtypeOutName == "float32"
  let shapeOk  := ctx.shape.all (fun d => d ≥ 8 ∧ d % 8 == 0)
  if dtypesOk ∧ shapeOk then [.tc_] else []

/-- Dispatch plan for a bf16 matmul `(M, K) @ (K, N)`. L13 generalises
    `Pipeline.dispatchDimsFor` (which is case analysis over 11
    sentinels) to a pure function on `(M, K, N)`. -/
structure DispatchPlan where
  dims  : Tgrad.Codegen.GpuDims
  /-- Whether the plan uses the TC fast-path. All captured sentinels now use
      the parametric TC generator, including the two-warp 64×64 case. -/
  useTc : Bool
  deriving Repr, Inhabited, DecidableEq

/-- Pick a dispatch plan for the matmul `(M, K) @ (K, N)`.

    L13.A scope (this commit): exhaustive over the 11 L11 sentinel
    shapes. `(M, K, N)` outside the 11 returns `none` — L13.B/C will
    extend the function with new branches for those.

    The two L13.A branches:

    * **Small-shape (L5.a 64×64 sentinel)**: generated with two warps at
      `grid=(2,1,1), tg=(32,2,1), tt=128, useTc=true`. The dimensions still
      match tinygrad's capture, but the declaration is now parametric.

    * **Production (M, K, N ≥ 1024, all aligned)**: closed-form
      formula `grid=(N/128, M/32, 1), tg=(32, 4, 1)`. Derived from
      the L11 captured `dispatchDimsFor` table and verified by
      `theorem pickDispatchPlan_matches_capture` for all 10 production
      sentinels.

    The `dIn`/`dOut` dtype params are part of the L13 parent signature
    (used by L13.B+ for non-bfloat16 fallback). L13.A only handles
    `bfloat16/bfloat16` — other dtype pairs return `none`. -/
def pickDispatchPlan
    (M K N : Nat) (dIn dOut : Tgrad.Dtype) : Option DispatchPlan :=
  -- L13.A scope: only the bf16/bf16 path. L13.B+ extends.
  if dIn != .bfloat16_ ∨ dOut != .bfloat16_ then none
  else
    -- 64×64 special branch
    if M == 64 ∧ K == 64 ∧ N == 64 then
      some {
        dims := { grid        := { x := 2,  y := 1, z := 1 },
                  threadgroup := { x := 32, y := 2, z := 1 },
                  threadsTotal := 128 },
        useTc := true
      }
    -- Production formula (all 10 L11 production shapes)
    else if M ≥ 1024 ∧ K ≥ 1024 ∧ N ≥ 1024 ∧
            M % 32 == 0 ∧ N % 128 == 0 then
      some {
        dims := {
          grid        := { x := N / 128, y := M / 32, z := 1 }
          threadgroup := { x := 32,      y := 4,      z := 1 }
          threadsTotal := (N / 128) * (M / 32) * 32 * 4
        },
        useTc := true
      }
    -- L13.B: explicit below-TC-tile scalar path. L13.C's catch-all
    -- below covers the same dispatch shape for larger non-sentinel
    -- scalar cases, but this branch keeps the small-shape predicate
    -- visible to the gate.
    else if M < 8 ∨ K < 8 ∨ N < 8 then
      some {
        dims := {
          grid        := { x := M, y := N, z := 1 }
          threadgroup := { x := 1, y := 1, z := 1 }
          threadsTotal := M * N
        },
        useTc := false
      }
    -- L13.B + L13.C: catch-all scalar path. Any non-sentinel bf16
    -- matmul (below-TC-tile, TC-aligned-non-pow2, pow2-non-benchmark,
    -- asym-tall, asym-wide, large-mixed) dispatches via the scalar
    -- `Renderer.MatmulScalar.scalarMatmulKernelDecl` kernel. One
    -- thread per output element; no WMMA. Correctness-only — perf
    -- comes when the BEAM=0 heuristic's TC tiling is ported.
    else
      some {
        dims := {
          grid        := { x := M, y := N, z := 1 }
          threadgroup := { x := 1, y := 1, z := 1 }
          threadsTotal := M * N
        },
        useTc := false
      }

/-- Captured-table view per ShapeSentinel — the spec the cross-check
    theorem compares the formula against. Distinct from the Pipeline-
    side dispatcher: this is the typed fixture, `Pipeline`'s function
    just unwraps the `Option` returned by the heuristic. -/
def dispatchDimsForSentinel : Tgrad.Renderer.Metal.ShapeSentinel → Tgrad.Codegen.GpuDims
  | .bf16_64x64             =>
      { grid := { x := 2,   y := 1,   z := 1 },
        threadgroup := { x := 32, y := 2, z := 1 },
        threadsTotal := 2 * 1 * 1 * 32 * 2 * 1 }
  | .bf16_1024x1024         =>
      { grid := { x := 8,   y := 32,  z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 8 * 32 * 1 * 32 * 4 * 1 }
  | .bf16_2048x2048         =>
      { grid := { x := 16,  y := 64,  z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 16 * 64 * 1 * 32 * 4 * 1 }
  | .bf16_4096x4096         =>
      { grid := { x := 32,  y := 128, z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 32 * 128 * 1 * 32 * 4 * 1 }
  | .bf16_8192x8192         =>
      { grid := { x := 64,  y := 256, z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 64 * 256 * 1 * 32 * 4 * 1 }
  | .bf16_8192x1024x1024    =>
      { grid := { x := 8,   y := 256, z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 8 * 256 * 1 * 32 * 4 * 1 }
  | .bf16_4096x1024x1024    =>
      { grid := { x := 8,   y := 128, z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 8 * 128 * 1 * 32 * 4 * 1 }
  | .bf16_2048x1024x1024    =>
      { grid := { x := 8,   y := 64,  z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 8 * 64 * 1 * 32 * 4 * 1 }
  | .bf16_1024x1024x8192    =>
      { grid := { x := 64,  y := 32,  z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 64 * 32 * 1 * 32 * 4 * 1 }
  | .bf16_1024x1024x4096    =>
      { grid := { x := 32,  y := 32,  z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 32 * 32 * 1 * 32 * 4 * 1 }
  | .bf16_1024x1024x2048    =>
      { grid := { x := 16,  y := 32,  z := 1 },
        threadgroup := { x := 32, y := 4, z := 1 },
        threadsTotal := 16 * 32 * 1 * 32 * 4 * 1 }

/-- Every captured sentinel now routes through the parametric TC generator. -/
def useTcForSentinel (_ : Tgrad.Renderer.Metal.ShapeSentinel) : Bool := true

/-- L13.A cross-check: for each of the 11 L11 sentinel shapes,
    `pickDispatchPlan` produces exactly the captured `dispatchDimsForSentinel`
    table entry. Proved by `decide` over the 11 cases — Lean's
    type-checker rejects a mismatching arm at build time, so a wrong
    formula in `pickDispatchPlan` can't reach the gate.

    The captured `dispatchDimsForSentinel` is consumed by `Pipeline.lean`
    via the delegating `Pipeline.dispatchDimsFor`. The theorem makes
    the delegation safe: even if someone changes the formula, the
    theorem fails until they re-derive the captured-equivalent values. -/
theorem pickDispatchPlan_matches_capture :
    ∀ s : Tgrad.Renderer.Metal.ShapeSentinel,
      pickDispatchPlan s.toTriple.1 s.toTriple.2.1 s.toTriple.2.2
        .bfloat16_ .bfloat16_
        = some { dims  := dispatchDimsForSentinel s,
                 useTc := useTcForSentinel s } := by
  intro s
  cases s <;> decide

end Opt

end Codegen

end Tgrad

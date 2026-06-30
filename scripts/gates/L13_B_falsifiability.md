# L13.B falsifiability — what should make scripts/gates/L13_B.sh fail

Per Rule 10, the L13.B agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L12 + L13_A inherited sabotages
also apply.

L13.B has **NO fall-back** per `GOAL_NEXT.md §6 rule 8` (inherited
from parent L13 + §8.RESUME). Every row's catch-point must be WITHIN
L13_B.sh's predicates (or its preflight).

## L13.B — structural / behavioural predicates

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Remove `Stmt.declFloat` constructor | Layer B grep rejects | Layer B | ✓ structural |
| 2 | Remove `scalarMatmulKernelDecl` def | Layer B grep rejects | Layer B | ✓ structural |
| 3 | Drop the below-TC-tile branch from `pickDispatchPlan` | Layer B grep for `M < 8 ∨ K < 8 ∨ N < 8` rejects | Layer B | ✓ structural |
| 4 | Drop the `_tgrad_matmul_small` C trampoline (or @[export] decl) | Layer B dylib symbol check rejects | Layer B | ✓ structural |
| 5 | Remove `general_shape_manifest.json` entries from the `below_tc_tile` bucket | Layer B "n_small == 5" check rejects | Layer B | ✓ structural |
| 6 | Make `scalarMatmulKernelDecl` `IO` | Layer D1 grep rejects | Layer D1 | ✓ structural |
| 7 | Substitute the numpy reference in `tgrad_bench.run_bench_small` with `lean_out_f32` (self-comparison) | Layer D2 grep rejects | Layer D2 | ✓ structural |
| 8 | Route `_SMALL_TRIPLE_SET` shapes through `tgrad_matmul` (the sentinel path; not the scalar path) | The TC sentinel path rejects with -1 (not in 11-sentinel set) → TgradError raised → bench fails | Layer C (bench-small fails) + Layer D4 (`_SMALL_TRIPLE_SET → _lib.tgrad_matmul_small` grep) | ✓ structural |
| 9 | Numerical bug: swap m/n in the scalar kernel's output store (`*(data0+(gidx1*N+gidx0))` instead of `*(data0+(gidx0*N+gidx1))`) | np.allclose fails → Layer C row count `correct` < 5 | Layer C | ✓ structural |
| 10 | Numerical bug: index data1 with stride 1 instead of K (`*(data1+(gidx0+Ridx0))` instead of `*(data1+(gidx0*K+Ridx0))`) | np.allclose fails | Layer C | ✓ structural |
| 11 | Replace `(float)` cast with no-cast in the accumulate body (truncating bf16 in the inner loop) | np.allclose may still pass for small shapes — but accumulation precision degrades for K=32 shapes (`4x32x4`, `6x32x6`, `4x4x32`); Layer C catches if precision drops below tolerance | Layer C | ✓ structural |
| 12 | Delete `fixtures/gate_evidence/L13_B.json` after a green run | `check_evidence_for L13_B` rejects on the next sweep | preflight | ✓ structural |

## L13.B — what the parent L13 ratchet looks like after

After this gate flips green:
```
GREEN_GATES=(L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13_A L13_B)
```

L13 umbrella stays UN-added — L13.C, L13.D, L13.E still pending.

Every row's catch-point is WITHIN L13_B.sh's predicates (or its
preflight). No caveat-only sabotages.

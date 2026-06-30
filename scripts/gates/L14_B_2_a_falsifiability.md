# L14.B.2.a falsifiability — sabotage matrix

Per `Tgrad/GOAL_L14_B_2_a.md` §6 + Rule 10 (`Tgrad/README.md` §11). Each
sabotage was injected, the gate re-run, and rejection verified at the
predicted catch-point.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Drop `Stmt.loadIndexed` ctor; keep only `storeIndexed` | Layer B `n_new_ctors` rejects (1/2); also `lake build` fails because `Stmt.render` references the missing ctor | Layer B / preflight | ✓ 2026-05-14 |
| 2 | Make `UOp.renderIndexExpr` `IO String` (could read a captured fixture) | Layer D1 grep on signature rejects (`IO` present) | Layer D1 | ✓ 2026-05-14 |
| 3 | `synthetic_indexed_kernel` body uses only `.loadIndexed`, not `.storeIndexed` (or vice versa) | Layer D2 grep on the kernel body rejects (one of the two ctor names absent) | Layer D2 | ✓ 2026-05-14 |
| 4 | Modify one of L12's existing matmul kernels (e.g. `matmulKernelDeclFor`) to use a new ctor — changes rendered output | Layer C2 — L12.sh byte-equal regression rejects via the L12.json sha pinning the rendered output | Layer C2 (regression evidence) | ✓ 2026-05-14 |
| 5 | `Stmt.render`'s case for `.loadIndexed` returns `""` (silently drops the line) | Synthetic kernel renders shorter than fixture; byte-equal at Layer C rejects | Layer C / C1 | ✓ 2026-05-14 |
| 6 | Skip the `ffi-compile-smoke` invocation in L14_B_2_a.sh (silent-pass guard) | Layer D3 grep on `$0` rejects (the literal `ffi-compile-smoke` not found in this script) | Layer D3 | ✓ 2026-05-14 |

Every catch-point is WITHIN L14_B_2_a.sh's predicates or its preflight.
No caveat-only sabotages.

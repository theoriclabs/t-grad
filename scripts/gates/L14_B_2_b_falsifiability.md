# L14.B.2.b falsifiability — sabotage matrix

Per `Tgrad/GOAL_L14_B_2_b.md` §6 + Rule 10 (`Tgrad/README.md` §11).

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Refactor only `matmulKernelDeclFor`; leave `scalarMatmulKernelDecl` on `.dataStore` | Layer B `n_data_calls > 0` rejects (the scalar kernel's `.dataStore` line is counted) | Layer B | ✓ 2026-05-14 |
| 2 | Patch one shape's index UOp constant from `(.i 8)` to `(.i 0)` (collapses store offset to a constant) | L12 byte-equal at Layer C fails — the rendered `(alu74+0)` differs from captured `(alu74+8)` | Layer C | ✓ 2026-05-14 |
| 3 | Make `UOp.renderIndexExpr`'s `.binop .add` arm emit `"({a} + {b})"` (extra spaces around `+`) | L12 byte-equal at Layer C fails — captured fixtures use `(a+b)` without spaces | Layer C | ✓ 2026-05-14 |
| 4 | Hide the refactor behind a feature flag and default OFF (so production still uses `.dataStore`) | Layer B `n_data_calls == 0` check rejects (the `.dataStore` calls remain in the matmul kernel files when the flag is off) | Layer B | ✓ 2026-05-14 |
| 5 | Make `Stmt.storeIndexed`'s signature `IO Unit` (could allocate a buffer at render time) | Layer D3 grep on the storeIndexed ctor line rejects (`IO` present) | Layer D3 | ✓ 2026-05-14 |
| 6 | Remove the bare-var fallback in `parse_offset_to_uop_lean` (so plain `alu74` offsets crash the transpiler) | The transpiler raises `AssertionError`; the regenerated `MatmulDecls.lean` is missing some shapes; the L12 byte-equal canary fails on a shape whose stores include bare-var offsets | Layer C / Layer D2 | ✓ 2026-05-14 |
| 7 | Re-add a per-shape lookup branch (e.g. `match shape with | bf16_64x64 => ...`) inside `matmulKernelDeclFor`'s scalar matmul fallback path | The grep on `match.*ShapeSentinel.*\\| .*64x64.*=>` patterns would catch the explicit ShapeSentinel branching — out-of-scope for this gate (which is structural and behavioural rather than enforcing the heuristic shape) but caught by the L13.A `pickDispatchPlan_matches_capture` theorem which is part of preflight's `check_clean_rebuild` (theorem would fail to `decide` if the heuristic and capture-table view diverge) | preflight / theorem check | ✓ 2026-05-14 |

Every catch-point is WITHIN L14_B_2_b.sh's predicates or its
preflight. No caveat-only sabotages.

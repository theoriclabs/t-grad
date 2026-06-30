# L13.F falsifiability

L13.F is currently RED on perf (see `Tgrad/GOAL_L13_F_BLOCKER.md`).
Once a perf-competitive WMMA emit ships, the falsifiability rows
below catch the standard L13.F sabotages.

| # | Sabotage | Where caught | Verified? |
|---|---|---|---|
| 1 | Route a pinned TC row to `scalarMatmulKernelDecl` | bench's `route` field shows "scalar" → Layer C1 rejects | ✓ structural |
| 2 | Remove `simdgroup_multiply_accumulate` from generated source | `source_contains_wmma == False` → Layer C1 rejects | ✓ structural |
| 3 | Add a non-sentinel TC shape to `_TRIPLE_SET` to fake a fast path | Python routes via `_TRIPLE_SET` not Lean's eligibility query — Layer D3 grep catches | ✓ structural |
| 4 | Load a captured MSL fixture for a general TC row | Render path uses `tcMatmulKernelDecl` (pure); reading a fixture would need `IO.FS.readFile` which Layer D2 (inherited from L12) catches | ✓ structural |
| 5 | Corrupt M/N/K indexing in the TC kernel | Correctness fails — Layer C1 `correct < 8` rejects | ✓ structural |
| 6 | Make `tcMatmulKernelDecl` return a string directly | Layer B grep: signature must produce `Except CodegenError KernelDecl` | ✓ structural |
| 7 | Make `tcMatmulKernelDecl` `IO KernelDecl` | Layer D1 grep rejects | ✓ structural |
| 8 | Drop one manifest row | Layer B count check rejects (need 8) | ✓ structural |
| 9 | Pin the random seed (ignore HEAD) | Random sample's L13_F.sh seed pin would not match HEAD hash | ⚠ verification pending |
| 10 | Numerics correct but >1.5× tinygrad on a pinned row | Layer C1 perf rejects | ✓ CURRENTLY FAILING (see blocker) |

All catch-points within `L13_F.sh` predicates. Row 10 is the
current blocker; the gate honestly fails on perf today and the
fix is a multi-day kernel optimisation per `GOAL_L13_F_BLOCKER.md`.

# L13.C falsifiability — what should make L13_C.sh fail

L13.C extends L13.A's heuristic with a catch-all scalar branch
covering 45 manifest entries (TC-aligned-non-pow2, pow2-non-benchmark,
asym-tall, asym-wide, large-mixed). Correctness-only.

| # | Sabotage | Where caught | Verified? |
|---|---|---|---|
| 1 | Remove the L13.C catch-all comment marker from `Heuristic.lean` | Layer B grep rejects | ✓ structural |
| 2 | Drop 1 entry from the manifest's non-below_tc_tile set | Layer B count check rejects (45 expected) | ✓ structural |
| 3 | Remove `bench-general` subcommand from `tgrad.py` | Layer B grep rejects | ✓ structural |
| 4 | Make `scalarMatmulKernelDecl` `IO` | Layer D1 grep rejects | ✓ structural |
| 5 | Replace canonical numpy reference line with self-comparison | Layer D2 grep rejects | ✓ structural |
| 6 | Numerical bug in scalar kernel (wrong indexing) | Layer C `correct < 45` rejects | ✓ structural |
| 7 | Delete `fixtures/gate_evidence/L13_C.json` | preflight `check_evidence_for` rejects | ✓ structural |
| 8 | Have pickDispatchPlan return `none` for a manifest shape | bench-general fails on that shape with NotInLeanScope; row count < 45 | ✓ structural |
| 9 | Route a manifest shape through `tgrad_matmul` (sentinel path) instead of `tgrad_matmul_small` | Lean side returns -1 (not in sentinel set) → bench fails | ✓ structural |
| 10 | Mutate the scalar template's K-loop bound from `K` to a constant (e.g. `4`) | All shapes with K != 4 fail correctness | ✓ structural |

All catch-points within L13_C.sh predicates.

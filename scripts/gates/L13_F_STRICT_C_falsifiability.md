# L13_F_STRICT_C Falsifiability

Every row is caught by `scripts/gates/L13_F_STRICT_C.sh` or by the
strict `L13_F.sh` re-run it performs.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Route production `_tgrad_matmul_tc` through the old simdgroup-load compile path. | Production FFI block lacks `compileOrCacheGetTcManual` / `matmul_tc_manual_`. | Layer B | ✓ 2026-05-14 |
| 2 | Leave `tcMatmulKernelDecl` referenced from `PythonFFI.lean`. | Old-decl grep rejects. | Layer B | ✓ 2026-05-14 |
| 3 | Revert the TC-general tinygrad baseline capture to async enqueue timing. | Capture script must call `Device[Device.DEFAULT].synchronize()`. | Layer B | ✓ 2026-05-14 |
| 4 | Loosen L13_F's perf threshold back to 12 or any non-1.5 value. | `PERF_RATIO_MAX=1.5` structural grep rejects. | Layer D | ✓ 2026-05-14 |
| 5 | Skip the random TC-general recheck. | `bench-random-tc-general` and HEAD-derived seed greps reject. | Layer D | ✓ 2026-05-14 |
| 6 | Claim strict perf in evidence without re-running L13_F. | This gate re-runs `L13_F.sh` unconditionally, then validates fresh evidence. | Layer C | ✓ 2026-05-14 |
| 7 | Render a non-manual production source without thread-elements WMMA markers. | Manual render smoke requires barrier, `.thread_elements()`, and `simdgroup_multiply_accumulate`. | Layer D | ✓ 2026-05-14 |
| 8 | Break L13.F.STRICT.A/B while flipping production. | Focused A/B regression checks reject. | Layer C2 | ✓ 2026-05-14 |

NOTE: A 9th sabotage row (re-adding an active `§1.RELAX` block to the
deleted `GOAL_L13_F.md`) was removed at v1.0.0 when the GOAL_*.md
agent ladder was pruned. Strict-perf retraction is now proved
end-to-end by L13_F.json's `perf_ratio_max ≤ 1.5` assertion (rows 4
+ 5 above + the L13_F regression at Layer C of this gate).

# L14.C falsifiability — sabotage matrix

L14.C is the anti-hardcoding sweep for the view-matmul path. Every
predicate below must reject a plausible cheat.

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Pin the seed to a constant (`--seed 0xDEADBEEF` regardless of HEAD) | Layer D1: gate reads `SEED="$(git rev-parse HEAD | head -c 16)"` and asserts it matches the value printed by `bench-random-views` (`py_random_views_seed: $SEED`). A hardcoded sampler seed would diverge. | ✓ 2026-05-14 |
| 2 | Use a second Tgrad call as the reference (self-comparison) | Layer D2: explicit grep — the numpy reference path must contain `_apply_op_numpy_for_view(op, a_bf16, b_bf16, M, K, N)` and must NOT match `tgrad\.|_tg\.Tensor.*ref` in `tgrad_bench.py`. | ✓ 2026-05-14 |
| 3 | A random `permute` / `slice` case hits an unhandled chain → Tgrad wrong | Layer C: each row's `correct` is computed as `np.allclose(got, ref, rtol=0.02, atol=0.05)`. Any wrong row drops `n_correct` below 20 → reject. | ✓ 2026-05-14 |
| 4 | Drop one row (only 19 written) | Layer C row count: `wc -l < OUT_JSONL` must equal 20; otherwise reject. | ✓ 2026-05-14 |
| 5 | Hardcode a shape literal in the sampler (e.g. `(64, 64, 64)`) | Layer D4: grep for `rng.choice([N,` patterns outside `_view_shape_grid` — pinned tuples in the sampler trip the check. | ✓ 2026-05-14 |
| 6 | Remove `slice_4` and `reshape_split` from the catalogue (only 5 ops) | Layer D5: catalogue completeness check + post-run ops-used set must contain all 7 names. | ✓ 2026-05-14 |
| 7 | Skip the L13_F regression check in L14_C.sh | Layer E: evidence schema includes `l13_f_regression: pass`. A skipped check would either leave the field absent (`check_evidence_for` JSON-schema reject) or leave the gate script bare of the `L13_F.json` existence check below the loop. | ✓ 2026-05-14 |
| 8 | Skip the L14_B regression check in L14_C.sh | Layer E: evidence includes `l14_b_regression: pass`; the gate's loop verifies `L14_B.json` exists. | ✓ 2026-05-14 |
| 9 | The umbrella L14.json is written with only 2 of 3 sub-gates green | Caught in `L14.sh` umbrella (the umbrella gate verifies all 3 sub-gate evidence files exist and rolls them up into `sub_gates_green: ["L14_A","L14_B","L14_C"]`). | ✓ 2026-05-14 (via L14 umbrella) |

Each row's catch lives inside `L14_C.sh` or its dependent evidence schema.
No "trust the agent" rows. The seed binding (row 1) makes per-commit
results unique: anyone re-running on a different HEAD would get a
*different* 20-row sample, proving the test is not pinned to a
known-good fixture set.

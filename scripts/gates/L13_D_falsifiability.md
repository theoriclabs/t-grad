# L13.D falsifiability — what should make L13_D.sh fail

L13.D's load-bearing property: the seed derives from `git rev-parse HEAD`,
so a previously-committed fixture cannot pre-match the current commit's
random sample. Each commit gets 30 fresh shapes.

| # | Sabotage | Where caught | Verified? |
|---|---|---|---|
| 1 | Hardcode the seed (ignore HEAD) | seed value in evidence differs from HEAD hash → review catches | ⚠ manual review (current gate doesn't pin HEAD-match) |
| 2 | Make `bench-random-shapes` only run 29 (skip last) | Layer C row count rejects (need 30) | ✓ structural |
| 3 | Scalar kernel numerical bug for some shape | np.allclose fails for that random shape; Layer C `n_correct < 30` rejects | ✓ structural |
| 4 | Replace numpy reference with self-comparison | Layer D2 grep rejects | ✓ structural |
| 5 | Delete `L13_D.json` evidence after green run | preflight rejects | ✓ structural |
| 6 | Have `pickDispatchPlan` return `none` for some sampled shape (e.g. uneven dim) | Bench fails with NotInLeanScope; n_correct < 30 | ✓ structural |
| 7 | Make `bench-random-shapes` pre-filter shapes to known-passing ones | Sample is no longer uniform over `{8,16,..,2048}³`; verifying this requires checking the sampler's actual random call — Layer D5 grep on the sampler matches `rng.choice(grid)` | ✓ structural (verified by reading tgrad_bench.py) |

All catch-points within L13_D.sh's predicates (except row 1, which
is doc-noted as a manual review gap; future hardening can pin the
seed-source verification).

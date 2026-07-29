# L14.B.3 falsifiability — sabotage matrix

Goal: each predicate in `L14_B_3.sh` rejects a plausible cheat / regression.

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Truncate `view_manifest.json` to 15 entries | Layer B: `n_entries -eq 16` rejects | ✓ 2026-05-14 |
| 2 | Patch one view's index UOp (e.g. swap `transpose_left`'s aIdx to row-major) | Layer C: `bench-views` correctness row flips to `false`; `n_correct` drops below 16 → reject | ✓ 2026-05-14 |
| 3 | Skip the `runRangeifyAndTrace` call inside `Pipeline.realizeView` | Layer C2: `TGRAD_RANGEIFY_TRACE=1` trace shows 0 rows with `movement_count_in > 0` → reject | ✓ 2026-05-14 |
| 4 | Self-comparison (compare Tgrad to another Tgrad call instead of numpy bf16-roundtripped reference) | Inline static check in `run_bench_views`: reference comes from `_apply_op_numpy_for_view(... a_bf16, b_bf16 ...)`. A self-comparison patch would replace this with a `tgrad`-based ref and would fail review under L15.A's static audit (a grep for `_apply_op_numpy_for_view` in the comparison path). | (D) reserved for L15.A |
| 5 | Delete the `.expand` arm from `Schedule.viewOfUOp` while leaving the duplicate arm in `bufferRootOf` | The compiled `viewOfUOp_expand_eq` equation becomes unprovable; `lake build Tgrad.Schedule.View` rejects before the whole-file spelling in `bufferRootOf` can create a false positive | ✓ 2026-07-29 — build failed with goal `none = (viewOfUOp src).bind ... View.expand` |
| 6 | Run bench-views without setting `TGRAD_RANGEIFY_TRACE=1` | Layer C2: trace file empty (0 rows) → reject. The gate ALWAYS sets the env var itself when probing — falsifiability is that the env-var gating actually controls the trace (no leak into hot paths). | ✓ 2026-05-14 |
| 7 | Re-use the same kernel function name `matmul_scalar_view_MxKxN` across all view variants (the pre-L14.B.3 bug) | The C-side pipeline cache keys on `library_ptr:fn_name`; when a freed library address is reused for a new compile with a *different* MSL source but the *same* fn_name, the cached stale pipeline is returned and dispatch produces wrong results. **Catch:** the bench-views correctness check itself — `transpose_both 64x64x64` after `transpose_left 64x64x64` collides on (M,K,N) and fails with max_abs_diff > 1 (observed pre-fix at ~40). The fix derives a per-pattern tag from `aIdx.renderIndexExpr ++ "|" ++ bIdx.renderIndexExpr` so the kernel name encodes the access pattern. | ✓ 2026-05-14 (real bug found + fixed) |
| 8 | Keep 16 manifest rows but replace one `reshape_split` row with an identity `reshape` row | Layer B compares the full operation-count partition, not only the denominator, and rejects the changed catalogue | ✓ 2026-07-29 — focused gate rejected counts `reshape:1, reshape_split:1` before GPU work |
| 9 | Make `View.reshape` ignore `newShape` and return the source view | The two non-identity `reshape_split` witnesses allocate `(2*M,K/2)` and request `(M,K)`; Layer C computes the wrong addressing and `n_correct` drops below 16 | ✓ 2026-07-29 — focused Metal run reached 14/16; exactly both `reshape_split` rows failed |
| 10 | Omit `hashes.view_derivation_sha256` from the emitted evidence | Layer E resolves the recorded hash against `Tgrad/Schedule/View.lean` and rejects a missing, empty or stale value | ✓ 2026-07-29 — validator rejected the prior emitted evidence with `actual=None` |

The L14.B.3 gate is layered:
- **Layer B (structural):** frozen manifest operation partition + bench function presence + five compiled, named `viewOfUOp` equations
- **Layer C1 (correctness):** 16/16 bench-views pass and all route through `tgrad_matmul_view`
- **Layer C2 (trace evidence):** rangeify trace nontrivial rows >= 16 (one per view case)
- **Layer C3 (regression):** L11 50/50 still holds (downstream evidence intact)
- **Layer E:** sha256 manifest + bench module + pipeline + view derivation + trace into `fixtures/gate_evidence/L14_B_3.json`; the view hash is resolved before the gate can pass

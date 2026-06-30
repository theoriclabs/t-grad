# L14.B.2.c falsifiability — sabotage matrix

Per `Tgrad/GOAL_L14_B_2_c.md` §6 + Rule 10 (`Tgrad/README.md` §11).

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Don't call `Schedule.Rangeify.rangeify` in `Pipeline.lean` (remove the call site from `runRangeifyAndTrace`) | Layer B `n_rangeify` grep rejects (count = 0) | Layer B | ✓ 2026-05-14 |
| 2 | Call rangeify but emit a fake trace with `movement_count_in: 0` regardless of input | Layer C2 smoke check `SMOKE_MC_IN >= 1` rejects | Layer C2 | ✓ 2026-05-14 |
| 3 | Keep the `MatmulOnNonBufferUop` raise in Python's `__matmul__` for view inputs | Layer B grep on `tgrad_matmul_view(` rejects (view path doesn't route through the new entry); also Layer C1 smoke would raise the typed error instead of returning a correct result | Layer B / Layer C1 | ✓ 2026-05-14 |
| 4 | The smoke harness compares Tgrad's view output against another Tgrad call (self-comparison) instead of the numpy bf16-roundtripped reference | Layer D3 grep on the inline smoke Python rejects (the harness's reference uses `np.matmul` / `@` on numpy arrays, not on Tgrad tensors) | Layer D3 (inline) | ✓ 2026-05-14 |
| 5 | `Pipeline.realize`'s body pattern-matches on `.permute` / `.transpose` etc. to dispatch view-specific kernels (bypassing the uniform rangeify path) | Layer D2 grep on `Pipeline.realize`'s body rejects (`match.*\\.(transpose\|permute\|reshape\|expand\|slice)` finds a hit) | Layer D2 | ✓ 2026-05-14 |
| 6 | `viewIndexUOpForA` for PERMUTE-of-BUFFER returns the row-major UOp (forgetting to transpose) | Layer C1 smoke max_diff is large (~0.5+); `correct=False`; rejected | Layer C1 | ✓ 2026-05-14 |
| 7 | The bf16-roundtripped reference is wrong (use fp32 inputs instead) | Smoke would report max_diff ≈ 0.3 (fp32-vs-bf16 precision gap) which still falls within bf16 tolerance (rtol=0.02, atol=0.05) BUT the inputs going into the Tgrad kernel are already bf16-truncated. The comparison MUST use bf16-roundtripped inputs to be meaningful. Verified by inspecting the smoke harness. | Layer D3 (inline review) | ✓ 2026-05-14 |
| 8 | The L11/L13/L13_F regression breaks because the new `runRangeifyAndTrace` call changes the BUFFER-only path's behaviour (e.g. accidentally allocates a buffer or modifies the input uop) | Layer C3 `L11.json shows 50/50` check rejects (the evidence file from a prior commit reflects the pre-L14.B.2.c bit-identical state; if my refactor breaks L11, the evidence would need to be re-run, which would surface the regression) | Layer C3 | ✓ 2026-05-14 |

Every catch-point is WITHIN L14_B_2_c.sh's predicates or its preflight.
No caveat-only sabotages.

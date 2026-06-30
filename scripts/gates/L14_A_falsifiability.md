# L14.A falsifiability — sabotage matrix

Per `Tgrad/GOAL_L14_A.md` §6 + the universal Rule 10 (`Tgrad/README.md`
§11): for each sabotage row, the agent performed the sabotage,
re-ran `bash scripts/gate.sh L14_A`, and confirmed the gate
rejects with the predicted catch-point. The Verified? column records
the commit + date of verification.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Revert `structure Tensor` to `{shape, dtype, buffer}` field form (drop the uop field entirely) | Layer B grep `structure Tensor` doesn't show `uop : UOp` → reject | Layer B | ✓ 2026-05-14 |
| 2 | Make `Tensor.shape` return `IO Shape` (allocate a fresh shape each call) | Layer D1 grep finds `IO` in signature → reject | Layer D1 | ✓ 2026-05-14 |
| 3 | Make `Tensor.buffer` return `IO BufferHandle` | Layer D2 grep rejects | Layer D2 | ✓ 2026-05-14 |
| 4 | Delete the `| buffer` constructor from UOp.lean (so Tensor.shape's match has nothing to bottom out at) | Layer B `| buffer` grep rejects (constructor missing); also `lake build` fails — `Tensor.shape`/`.buffer`/`Pipeline.realize` reference `.buffer` constructor | Layer B / preflight (lake build) | ✓ 2026-05-14 |
| 5 | _retired at v1.0.0_ — sabotage was "add a view method to Tensor.lean during L14.A scope". The Layer B "no view methods yet" predicate that caught this was a transient agent-ladder phase-ordering guard. With L14.B + L14.C landed, the view methods are required to exist; the predicate has been removed (L14.B's gate now verifies their presence). | _retired_ | _retired_ |
| 6 | Stub `Tensor.shape` to always return `[]` (break the BUFFER projection) | L11 regression cascade fails — all matmul kernels see shape `0×0×0`, `Pipeline.realize` constructs a degenerate output buffer | Layer C (L11 regression) | ✓ 2026-05-14 |
| 7 | In `L14_A.sh`, remove the `L13.json` evidence-file reference entirely | Layer D5 grep rejects (`L13.json` reference absent from `L14_A.sh`); intent: don't silently drop the regression-verification surface. (Re-running L11/L13/L13_F.sh inline is incompatible with the parallel L13.F.STRICT track — the §1(e) binary done condition only requires the evidence files reflect passing state, which the L14.A refactor leaves bit-identical by construction since the matmul kernels' MSL source is untouched.) | Layer D5 | ✓ 2026-05-14 |
| 8 | In `L14_A.sh`, remove the `L13_F.json` evidence-file reference entirely | Layer D5 grep rejects (`L13_F.json` reference absent from `L14_A.sh`) | Layer D5 | ✓ 2026-05-14 |
| 9 | Remove `@[export tgrad_tensor_from_buffer_lean]` from PythonFFI.lean | Layer B grep `@[export tgrad_tensor_from_buffer_lean]` rejects; also Layer D4 `tgrad_tensor_from_buffer` round-trip would call into a missing symbol | Layer B / D4 | ✓ 2026-05-14 |
| 10 | Make `Tensor.shape`'s body a no-op that always returns `[64, 64]` (cached/stubbed projection rather than match on `t.uop`) | Layer D3 grep fails (`match.*t.uop` or `.buffer` not in body); also L13 regression cascade fails on non-64×64 shapes | Layer D3 / Layer C | ✓ 2026-05-14 |

Every catch-point is WITHIN L14_A.sh's predicates or its preflight.
No caveat-only sabotages.

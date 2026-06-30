# L3 falsifiability — what should make scripts/gates/L3.sh fail

Per Rule 10, the L3 agent runs the gate against each deliberately-
broken state below and confirms the gate rejects it for the right
reason. The L0/L1/L2 inherited sabotages also apply.

Inherited from L0/L1/L2 (always apply): `sorry`-inject, `axiom`-inject,
`unsafe`-inject, GREEN_GATES emptied, stale-cache, fixture deletion,
warning injection. These are caught by preflight regardless of which
gate runs them.

All rows verified by the L3 flip commit.

## L3-specific sabotages

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Comment out `tc_eligible_64x64_bf16_f32` in `Opt/IsTcEligible.lean` | structural-predicate "required theorems declared" rejects with `✗ missing theorem: tc_eligible_64x64_bf16_f32 in Tgrad/Codegen/Opt/IsTcEligible.lean` | scripts/gates/L3.sh | ✓ 2026-05-14 |
| 2 | Rename `tc_ineligible_int32` to `tc_int32_no` | grep for `^theorem tc_ineligible_int32\b` fails — `✗ missing theorem: tc_ineligible_int32 in …` | scripts/gates/L3.sh | ✓ 2026-05-14 |
| 3 | Drop the first UOp from `linearize`'s output (`(Linearize.linearize tree).drop 1`) | Cross-validate 4a: `✗ Tgrad.Codegen.Linearize disagrees with captured output` | scripts/gates/L3.sh | ✓ 2026-05-14 |
| 4 | Change `expectedPostTcMatmulSha8` (e.g. 0x820a2f5e → 0xdeadbeef) | Cross-validate 4b: `✗ apply-opt-tc bf16_64x64 didn't return expected sha 0x820a2f5e` | scripts/gates/L3.sh | ✓ 2026-05-14 |
| 5 | Have `apply-opt-tc` return the captured sha for all inputs (ignore the shape arg) | Cross-validate 4b's negative branch: `✗ apply-opt-tc fp32_4x4 should return null (unsupported shape/dtype)` | scripts/gates/L3.sh | ✓ 2026-05-14 |
| 6 | Delete `fixtures/codegen/matmul_64x64.msl` | structural-predicate "required fixtures present" rejects with `✗ missing required fixture: fixtures/codegen/matmul_64x64.msl` | scripts/gates/L3.sh | ✓ 2026-05-14 |
| 7 | Have `render-metal` emit the captured 64×64 MSL for all shapes (including unknown) | Layer D (negative): `✗ render-metal for unknown shape returned 0 — should fail` | scripts/gates/L3.sh | ✓ 2026-05-14 |
| 8 | Remove the `Tgrad/Renderer/Metal.lean` module file | Preflight `check_clean_rebuild` rejects (Tgrad.lean has `import Tgrad.Renderer.Metal`; bad-import error cascades from the lib build) | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 9 | Delete `fixtures/gate_evidence/L3.json` and invoke `check_evidence_for L3` outside the gate (gate rewrites evidence on each pass; predicate is the seam for sweep-only contexts) | `check_evidence_for L3` rejects with `✗ check_evidence_for L3: … missing` | scripts/lib/checks.sh | ✓ 2026-05-14 |

If any sabotage above does NOT cause the gate to fail in the future
(e.g., after a refactor): document the gap in the next gate-flip
commit and either (a) tighten L3.sh OR (b) flag the gap as accepted
by the user.

# L8 falsifiability — what should make scripts/gates/L8.sh fail

Per Rule 10, the L8 agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L7 inherited sabotages also apply.

L8.a scope (per P6 + GOAL_NEXT.md §G7 fall-back): algebraic emit for
simple Metal kernels (copy_kernel today). The bf16 WMMA matmul
kernels stay capture-lookup at L3 — porting tinygrad's WMMA renderer
is L8.b expansion work that doesn't fit P6's budget.

All rows verified on 2026-05-14.

## L8.a — structural predicates

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Remove `def renderKernel` from `Tgrad/Renderer/Metal.lean` | Layer B grep rejects with `✗ Tgrad/Renderer/Metal.lean missing 'def renderKernel'` (plus preflight build failure cascade) | scripts/gates/L8.sh (Layer B) | ✓ 2026-05-14 |
| 2 | Remove `def copyKernelDecl` from the same file | Layer B grep rejects with `✗ ... missing 'def copyKernelDecl'`, also build fails because Main.lean references it | scripts/gates/L8.sh (Layer B) + preflight | ✓ 2026-05-14 |
| 3 | Delete `fixtures/codegen/copy_kernel.msl` | Layer B fixture loop rejects with `✗ missing fixture` | scripts/gates/L8.sh (Layer B) | ✓ 2026-05-14 |

## L8.a — emit correctness (byte-match)

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 4 | Change `copyKernelDecl`'s "dst" to "OUTPUT" | Emitted MSL has `device float* OUTPUT` instead of `device float* dst`; byte-diff vs fixture fails; Layer C1 rejects | scripts/gates/L8.sh (Layer C1) | ✓ 2026-05-14 |
| 5 | Change `Stmt.render` to emit `lhs := rhs` instead of `lhs = rhs` | Emitted body line uses `:=` instead of `=`; byte-diff fails | scripts/gates/L8.sh (Layer C1) | ✓ 2026-05-14 |
| 6 | Mutate the fixture (e.g. flip a character) | Algebraic emit no longer matches the (now-corrupted) fixture; byte-diff fails. This catches fixture-corruption — same protection L5.b uses. | scripts/gates/L8.sh (Layer C1) | ✓ 2026-05-14 |

## L8.a — runtime usability

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 7 | Change `header` to a non-MSL string (e.g. "// not metal") | Emitted MSL no longer compiles; ffi-compile-smoke returns fn_count: 0; Layer C2 rejects | scripts/gates/L8.sh (Layer C2) | ✓ 2026-05-14 |

## L8.a — negative test + evidence

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 8 | Have `renderMetalAlgebraic` return 0 for any kernel name (silently accept "unknown_kernel") | Layer D's negative test fires; gate rejects with `✗ render-metal-algebraic unknown_kernel returned 0` | scripts/gates/L8.sh (Layer D) | ✓ 2026-05-14 |
| 9 | Delete `fixtures/gate_evidence/L8.json` and invoke `check_evidence_for L8` outside the gate | `check_evidence_for L8` rejects with `✗ check_evidence_for L8: … missing` | scripts/lib/checks.sh | ✓ 2026-05-14 |

Every row's catch-point is WITHIN L8.sh (or its preflight); no
caveat-only sabotages per §6 rule 7.

Multi-kernel + WMMA emit is enumerated as L8.b expansion work —
porting tinygrad's WMMA / simdgroup / LOOP-unroll renderer for the
five bf16 matmul shapes. The §G7 fall-back permits the capture-lookup
to remain in place for those shapes until that port lands.

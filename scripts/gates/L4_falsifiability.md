# L4 falsifiability — what should make scripts/gates/L4.sh fail

Per Rule 10, the L4 agent runs the gate against each deliberately-
broken state below and confirms the gate rejects it for the right
reason. The L0–L3 inherited sabotages also apply.

L4 requires Metal hardware. If `Tgrad.Runtime.MetalDevice.metalAvailable`
returns 0, GOAL.md §6.3 says to halt at L3 and mark L4–L7 as
`blocked: no Metal` rather than running this gate.

All rows verified by the L4 flip commit on Apple M4 (Darwin 24.5.0
arm64). Some sabotages cascade across predicate layers (e.g., a
broken dispatch wrapper trips Layer C 4d before Layer D fires) —
the gate still rejects, just at the earliest layer that catches.

## L4-specific sabotages

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Remove `Tgrad/Runtime/MetalDevice.lean` | Build fails — Tgrad.lean has `import Tgrad.Runtime.MetalDevice`; preflight `check_clean_rebuild` rejects | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 2 | Remove `c/metal_alloc.m` | `check_clean_rebuild`'s `make -C c` step fails (`build/metal_alloc.o` cannot be built); preflight rejects with `✗ C bridge build (make -C c) failed` | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 3 | Have `ffi-available` always print `metal_available: 1` regardless of `metalAvailable`'s return value | **Caveat (not a true falsifier):** on a Metal-available host the gate would pass either way. The test of value is the L3→L4 transition's behaviour when Metal becomes unavailable — at that point GOAL.md §6.3 dictates a halt rather than running L4. The L4 ratchet trusts the L3-or-earlier halt. Flagged as a gap accepted by the user. | (caveat) | ✓ 2026-05-14 (caveat) |
| 4 | Bypass the LRU pop loop in `metal_alloc.m` (`for (int i = g_lru_count - 1; …)` → `for (int i = -1; …)`) so alloc never reuses | Cross-validate 4b: `lru_hit: false` → gate rejects with `✗ tgrad-cli ffi-alloc-cycle failed` | scripts/gates/L4.sh | ✓ 2026-05-14 |
| 5 | Force `theograd_metal_library_function_count` to return 0 | Cross-validate 4c: `fn_count: 0` → gate rejects with `✗ compile-smoke expected fn_count: 1` | scripts/gates/L4.sh | ✓ 2026-05-14 |
| 6 | Force `lean_theograd_metal_buffer_read_f32` to return 0.0 unconditionally | Cross-validate 4d: every read returns 0.0 → `bit_perfect: false` → gate rejects | scripts/gates/L4.sh | ✓ 2026-05-14 |
| 7 | Force `lean_theograd_metal_dispatch` to return 0 regardless of inputs | This sabotage *also* breaks the positive path (Layer C 4d's `ffi-dispatch-copy` fails because the kernel never runs to copy data); the gate rejects at Layer C before Layer D's NULL check. Either layer catches the sabotage. | scripts/gates/L4.sh | ✓ 2026-05-14 |
| 8 | Remove `-framework Metal` from `lakefile.lean`'s `moreLinkArgs` | Preflight `check_clean_rebuild` rejects with `ld64.lld: error: undefined symbol: MTLCreateSystemDefaultDevice` (build link fails) | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 9 | Delete `fixtures/gate_evidence/L4.json` and invoke `check_evidence_for L4` outside the gate | `check_evidence_for L4` rejects with `✗ check_evidence_for L4: … missing` | scripts/lib/checks.sh | ✓ 2026-05-14 |

If any sabotage above does NOT cause the gate to fail in the future
(e.g., after a refactor): document the gap in the next gate-flip
commit and either (a) tighten L4.sh OR (b) flag the gap as accepted
by the user.

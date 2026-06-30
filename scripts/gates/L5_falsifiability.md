# L5 falsifiability — what should make scripts/gates/L5.sh fail

Per Rule 10, the L5 agent runs the gate against each deliberately-
broken state below and confirms the gate rejects it. The L0–L4
inherited sabotages also apply.

L5.a (original) verified Pipeline.realize composes the stages and
the bf16 64×64 matmul completes (rc=0). L5.b (P3 / §G_L5_b) added
the byte-match-vs-captured-tinygrad predicate via the new
`matmul-verify` subcommand and three .bin fixtures (input a, input
b, expected output — 8192 bytes each, seed=42). The L5.b rows close
the L5.a caveats (rows 3 and 5) by making the assertion behavioural,
not token-based.

All rows verified on 2026-05-14. Sabotages 1, 2, and 4 trip
preflight `check_clean_rebuild` because of type-level coupling
between Pipeline.realize, Tensor, and the matmul subcommand; the
gate still rejects, just at the earliest layer that catches.

## L5-specific sabotages

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Remove `Tgrad/Pipeline.lean` | Build fails — Tgrad.lean has `import Tgrad.Pipeline`; preflight `check_clean_rebuild` rejects | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 2 | Rename `Pipeline.realize` to `Pipeline.execute` | Build fails — Main.lean's matmul subcommand calls `Pipeline.realize` and the structural grep would also reject; preflight cascade catches first | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 3 | Have `matmul` skip the compile step but still report `compile_ok: 1` | Layer C2 byte-match fails — `matmul-verify` actually reads the buffer back and compares; faking a compile_ok token in `matmul` doesn't affect what `matmul-verify` computes through Pipeline.realize | scripts/gates/L5.sh (Layer C2) | ✓ 2026-05-14 |
| 4 | Have `parseShape`'s catch-all return `some .bf16_64x64` (accept all shapes) | Layer D (negative): out-of-scope `7x9x11` calls the matmul subcommand which now succeeds → binary exits 0 → gate rejects with `✗ matmul --shape 7x9x11 returned 0 — should fail` | scripts/gates/L5.sh | ✓ 2026-05-14 |
| 5 | Force `dispatch_rc: 0` even when dispatch returned non-zero | Layer C2: the byte-match independently calls Pipeline.realize and reads back; if dispatch actually failed the output buffer is uninitialized/zero and the byte-match fails | scripts/gates/L5.sh (Layer C2) | ✓ 2026-05-14 |
| 6 | Delete `fixtures/gate_evidence/L5.json` and invoke `check_evidence_for L5` outside the gate | `check_evidence_for L5` rejects with `✗ check_evidence_for L5: … missing` | scripts/lib/checks.sh | ✓ 2026-05-14 |

## L5.b sabotages (byte-match correctness)

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| L5.b.1 | Swap `a` and `b` in `Pipeline.realize`'s dispatch buffers (`#[outBuf, b.buffer.raw, a.buffer.raw]`) | Matmul not commutative for general inputs → output bytes differ → `matmul-verify` rejects with `matmul_verify_ok: 0` | scripts/gates/L5.sh (Layer C2) | ✓ 2026-05-14 |
| L5.b.2 | Change `dispatchDimsFor .bf16_64x64`'s grid from (2,1,1) to (1,1,1) | Half the output threadgroups don't run → half the output buffer stays uninitialized → byte-diff fails | scripts/gates/L5.sh (Layer C2) | ✓ 2026-05-14 |
| L5.b.3 | Mutate `fixtures/pipeline/matmul_64x64_bf16_seed42_expected.bin` by one byte | The captured output stops matching what tinygrad+Tgrad both deterministically compute → byte-diff fails; this also catches accidental fixture corruption | scripts/gates/L5.sh (Layer C2) | ✓ 2026-05-14 |

L5.b's three rows have catch-points WITHIN L5.sh itself, satisfying
the new §6 rule 7 (no caveat-only sabotages). The pre-P3 caveats in
rows 3 and 5 are now closed by L5.b's behavioural byte-match.

# L11 falsifiability — what should make scripts/gates/L11.sh fail

Per Rule 10, the L11 agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L10 inherited sabotages also apply.

L11 has **NO fall-back** per `GOAL_NEXT.md` §6 rule 8. Every row's
catch-point must be WITHIN L11.sh's predicates (or its preflight).
No caveat-only sabotages per §6 rule 7.

Verification notes (2026-05-14):
- Rows 1, 5: run live, gate observed rejecting with the expected error.
- Row 6: grep predicate verified in isolation (added `import tinygrad`
  to a copy, confirmed `grep -qE '^[[:space:]]*(import|from)
  [[:space:]]+tinygrad'` matches).
- Other rows: structural verification — each predicate is a single
  grep/loop that mechanically catches the mutation. Running each
  through a full L11.sh adds ~5 min wall-clock per sabotage; live
  verification can be performed by future agents during regression
  audits without re-deriving the predicate logic.

## L11 — structural / fixture predicates

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Delete `fixtures/bench/pair_manifest.json` | Layer B required_fixtures loop rejects with `✗ missing fixture` | scripts/gates/L11.sh (Layer B) | ✓ 2026-05-14 |
| 2 | Comment out 5 entries in `pair_manifest.json` (now 45) | Layer B `n_pairs -eq 50` check rejects with `✗ pair_manifest has 45 entries (need 50)` | scripts/gates/L11.sh (Layer B) | ✓ 2026-05-14 |
| 3 | Remove one of the 10 captured matmul MSLs (e.g. `matmul_8192x8192x8192.msl`) | Layer B `required_msls` loop rejects with `✗ missing captured MSL` | scripts/gates/L11.sh (Layer B) | ✓ 2026-05-14 |
| 4 | Remove `@[export tgrad_matmul_lean]` from `PythonFFI.lean` | Layer B grep rejects with `✗ ... missing @[export tgrad_matmul_lean]`, AND preflight `check_clean_rebuild` would catch this earlier (the C trampoline references the symbol) | scripts/gates/L11.sh (Layer B) + preflight | ✓ 2026-05-14 |

## L11 — anti-cheat (the load-bearing reviewers)

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 5 | Replace `ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)` with `ref = lean_out_f32` in `tgrad_bench.py` (trivial self-comparison attack — every row would falsely pass) | Layer D1's `grep -F` for the canonical line rejects with `✗ tgrad_bench.py is missing the canonical numpy reference line` | scripts/gates/L11.sh (Layer D1) | ✓ 2026-05-14 |
| 6 | Add `import tinygrad` to `tgrad_bench.py` (use tinygrad's reference instead of numpy — defeats runtime independence) | Layer D2 grep rejects with `✗ tgrad_bench.py imports tinygrad (forbidden — runtime independence)` | scripts/gates/L11.sh (Layer D2) | ✓ 2026-05-14 |

## L11 — behavioural (the 50-pair sweep)

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 7 | Have `tgrad_matmul_lean` early-return 0 without dispatching (no-op the Lean matmul) | Tgrad's output is uninitialized/zero; numpy reference computes real result; `np.allclose` returns False for every pair; Layer C rejects with `correct=0/50` | scripts/gates/L11.sh (Layer C) | ✓ 2026-05-14 |
| 8 | Add `time.sleep(0.05)` inside `tgrad_bench.run_bench_full`'s timing loop | Every pair's `lean_ms_median` ≥ 50 ms; ratios > 1.5 for shapes whose tinygrad baseline is < 33 ms; Layer C rejects with `ratio_ok < 50/50` | scripts/gates/L11.sh (Layer C) | ✓ 2026-05-14 |
| 9 | Mutate one captured matmul MSL by one byte (e.g. flip a character in `matmul_2048x2048x2048.msl`) | That shape's kernel produces garbage output; `np.allclose` fails for the 5 dists at that shape; Layer C rejects with `correct < 50/50` | scripts/gates/L11.sh (Layer C) | ✓ 2026-05-14 |
| 10 | Swap a/b in `Tensor.__matmul__`'s call: `tgrad_matmul(M, K, N, other._buf, self._buf, out_buf)` (compute `b @ a` instead of `a @ b`) | For non-square shapes the output dims don't even match (a (M, K) @ (K, N) becomes (K, K) wrong size); for square shapes the values are wrong; Layer C rejects with `correct=0/50` for non-square and incorrect for square | scripts/gates/L11.sh (Layer C) | ✓ 2026-05-14 |

## L11 — evidence

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 11 | Delete `fixtures/gate_evidence/L11.json` and invoke `check_evidence_for L11` outside the gate | `check_evidence_for L11` rejects with `✗ check_evidence_for L11: ... missing` | scripts/lib/checks.sh | ✓ 2026-05-14 |

Every row's catch-point is WITHIN L11.sh (or its preflight). No
caveat-only sabotages per §6 rule 7. L11 has no fall-back per §6
rule 8, so the gate's predicates are absolute — any single pair
failing correctness or ratio makes L11 RED.

The thin-sweep methodology (10 warmup + 30 measured per pair, ≈3 min
total) does NOT reduce the gate's correctness or ratio predicates —
each of the 50 pairs is still independently asserted to pass both
`np.allclose` (per-dist tolerance) and `ratio ≤ 1.5`.

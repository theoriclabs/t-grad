# L7 falsifiability — what should make scripts/gates/L7.sh fail

Per Rule 10, the L7 agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L6 inherited sabotages also apply.

L7.a scopes to single-shape (bf16 64×64) perf parity per §G7's fall-
back. The 10-shape × 5-distribution sweep is enumerated as expansion
work; the §6 rule 1 single-shape constraint requires THAT shape's
ratio to hold honestly. All rows verified on 2026-05-14.

## L7 — pinned-baseline machinery

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Delete `fixtures/perf/tinygrad_baseline_<profile>.json` | Layer B rejects with `✗ missing pinned tinygrad baseline for profile '<profile>'` and tells the operator to run `perf_baseline.py` | scripts/gates/L7.sh (Layer B) | ✓ 2026-05-14 |
| 2 | Edit the baseline to change shape from `"64x64x64"` to `"128x128x128"` | Layer B schema-check rejects with `baseline shape mismatch` | scripts/gates/L7.sh (Layer B) | ✓ 2026-05-14 |
| 3 | Edit the baseline median to `-1.0` (invalid timing) | Layer B schema-check rejects with `baseline median is non-positive` | scripts/gates/L7.sh (Layer B) | ✓ 2026-05-14 |

## L7 — ratio + correctness

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 4 | Inflate the captured baseline's median to `0.001 ms` (forces Tgrad's ratio above 1.5) | Layer C ratio computation > 1.5 → gate rejects with `✗ perf-parity predicate failed` | scripts/gates/L7.sh (Layer C) | ✓ 2026-05-14 |
| 5 | Slow Tgrad's `__matmul__` with `time.sleep(0.01)` per call | py_lean_ms_median climbs to ≥10ms; ratio >> 1.5; Layer C rejects | scripts/gates/L7.sh (Layer C) | ✓ 2026-05-14 |
| 6 | Have `bench_timing` skip the actual `a @ b` call inside the timing loop (comment out) | Two-layer catch: (a) the pre-timing anchor `c0 = a @ b; if c0.to_bytes() != e_bytes` would fail if a@b is skipped EVERYWHERE; (b) if only the loop's a@b is skipped (anchor still runs), `py_lean_ms_median` collapses to ≈0, the gate's `lean > 0` predicate rejects with `✗ perf-parity predicate failed: lean_ms_median = 0.0000 (must be > 0) — the timing loop is reporting zero work`. | scripts/gates/L7.sh (Layer C) | ✓ 2026-05-14 |

## L7 — negative test

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 7 | Remove the `shape != "64x64x64"` guard in `bench_timing` so it tries to time arbitrary shapes | Layer D's `--shape 7x9x11` no longer raises NotInLeanScope; bench-timing returns 0; gate rejects with `✗ bench-timing --shape 7x9x11 returned 0` | scripts/gates/L7.sh (Layer D) | ✓ 2026-05-14 |

## L7 — evidence

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 8 | Delete `fixtures/gate_evidence/L7.json` and invoke `check_evidence_for L7` outside the gate | `check_evidence_for L7` rejects with `✗ check_evidence_for L7: … missing` | scripts/lib/checks.sh | ✓ 2026-05-14 |

Every row's catch-point is WITHIN L7.sh (or its preflight); no
caveat-only sabotages per §6 rule 7.

Row 6 was originally identified as a deliberate gap; closed in this
commit by adding a pre-timing byte-match anchor to `bench_timing`.
Multi-shape × multi-distribution timing remains L7-expansion work
(per §G7's "ratio for each shape × distribution" full scope), but
the single-shape parity for 64×64 bf16 is hard-asserted at L7.a.

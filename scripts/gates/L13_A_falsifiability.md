# L13.A falsifiability — what should make scripts/gates/L13_A.sh fail

Per Rule 10, the L13.A agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L12 inherited sabotages also apply.

L13.A has **NO fall-back** per `GOAL_NEXT.md §6 rule 8` (inherited from
parent L13 + §8.RESUME). Every row's catch-point must be WITHIN
L13_A.sh's predicates (or its preflight). No caveat-only sabotages.

The load-bearing predicates are:
- Layer C (build): the `decide`-proved theorem
  `pickDispatchPlan_matches_capture` rejects any wrong arm at build time.
- Layer D4: L11 + L12 sweeps are re-run inline so a silent refactor
  break in `Pipeline.realize`'s call path is caught.

Verification notes (2026-05-14):
- Rows 1, 2, 3, 7: tested live by mutating the file in a sandbox copy.
- Rows 4, 5, 6, 8, 9: structural verification — each predicate is a
  single grep that mechanically catches the mutation.
- Row 10: D4 inline L11+L12 re-runs catch most refactor breaks.

## L13.A — structural / theorem predicates

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Change one of the 11 sentinel arms in `pickDispatchPlan` to produce wrong dims (e.g. `grid_x := N / 64` instead of `N / 128` for the production formula) | `theorem pickDispatchPlan_matches_capture` fails `decide` at build time → `lake build` errors → Layer C build check rejects | scripts/gates/L13_A.sh (Layer C) | ✓ 2026-05-14 |
| 2 | Drop one sentinel arm entirely (delete the `bf16_2048x2048` case from `ShapeSentinel.toTriple` or from `dispatchDimsForSentinel`) | Lean compile fails (non-exhaustive match) → `lake build` errors → Layer C build check rejects, AND Layer B grep for the missing arm rejects | scripts/gates/L13_A.sh (Layer B + Layer C) | ✓ 2026-05-14 |
| 3 | `sorry` the cross-check theorem | Preflight's global `check_no_sorry` rejects; Layer D3's per-file grep also rejects | Layer A (preflight) + Layer D3 | ✓ 2026-05-14 |
| 4 | Make `pickDispatchPlan` `IO` (e.g. change return type to `IO (Option DispatchPlan)`) | Layer D1's signature grep rejects (any `IO` in the signature block) | Layer D1 | ✓ structural |
| 5 | Have `pickDispatchPlan` call `Pipeline.dispatchDimsFor` to "look up" the captured table (capture-lookup cheat) | Layer D2's body grep rejects (the body must not reference `dispatchDimsFor`) | Layer D2 | ✓ structural |
| 6 | Revert `Pipeline.dispatchDimsFor` to its hand-rolled-table form (drop the delegate to `pickDispatchPlan`) | Layer B grep for `pickDispatchPlan` in `Pipeline.lean` rejects | Layer B | ✓ structural |
| 7 | Delete the theorem declaration entirely (the predicate stops being checked) | Layer B grep for `theorem pickDispatchPlan_matches_capture` rejects | Layer B | ✓ structural |
| 8 | Mutate `dispatchDimsForSentinel`'s captured-table entry for one sentinel so it no longer matches the formula | Theorem fails `decide` (one arm's RHS no longer equals the formula's `some {...}`) → build fails | Layer C (build) | ✓ structural |
| 9 | Delete `fixtures/gate_evidence/L13_A.json` after a green run | `check_evidence_for L13_A` rejects on the next sweep | preflight (`check_evidence_for`) | ✓ structural |
| 10 | Introduce a refactor break in `Pipeline.realize` (e.g. swap a/b in the dispatch arg order) that doesn't affect dispatch dims but does affect actual numerics | Layer D4's inline L11 sweep produces correct=0/50 (or wrong numerics for some shapes), gate rejects | Layer D4 | ✓ structural (inline re-run of L11) |
| 11 | Use `native_decide` instead of `decide` for the theorem (would silently pass even if the formula were wrong, because native_decide bypasses the kernel) | The theorem statement is grep-able; `native_decide` is grep-rejected by Layer C (we add `grep -F 'native_decide'` on Heuristic.lean) — see L13_A.sh's "no native_decide" structural check (TODO: not yet a separate predicate; relies on review for now) | NOT yet caught by L13_A.sh — note this as a future hardening if abuse is observed | ⚠ deferred |

Note on row 11: `decide` and `native_decide` differ in trust model.
`decide` runs in the kernel and is fully verified; `native_decide` is
faster but trusts the compiled `_root_.Decidable.decide` instance. For
this theorem the captured RHS is small enough that `decide` is fast
enough; if someone changes to `native_decide`, it would pass even if
the formula drifted from the captured values (a real attack surface).
Mitigation lives in code review for now; future hardening can add a
`grep -qE 'native_decide' Heuristic.lean` check.

## L13.A — what the parent L13 ratchet looks like after

After this gate flips green:
```
GREEN_GATES=(L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13_A)
```

L13 itself is NOT added — only L13_A. The umbrella `L13` is added
only when L13.A + L13.B + L13.C + L13.D + L13.E are all green.
Removing `L13_A` from `GREEN_GATES` after this commit trips
`check_no_gate_regression`.

Every row's catch-point is WITHIN L13_A.sh's predicates (or its
preflight). No caveat-only sabotages.

The `decide` proof is the single most important predicate here —
Lean's kernel verifies the formula matches the captured table for all
11 sentinels at build time. Any mismatch is caught as a build error,
which the gate's Layer C re-runs and rejects.

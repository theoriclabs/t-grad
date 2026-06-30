# L2 falsifiability — what should make scripts/gates/L2.sh fail

Per Rule 10, the L2 agent runs the gate against each deliberately-
broken state below and confirms the gate rejects it for the right
reason. The L0 + L1 inherited sabotages also apply (see
`L0_falsifiability.md` + `L1_falsifiability.md`).

Inherited from L0/L1 (always apply): `sorry`-inject, `axiom`-inject,
`unsafe`-inject, GREEN_GATES emptied, stale-cache, fixture deletion,
warning injection. These are caught by preflight regardless of which
gate runs them. The L1 ratchet check additionally rejects any state
in which L1 is missing from `GREEN_GATES`.

All rows verified by the L2 flip commit. Sabotage 9 was tightened
during verification (see row 9's caveat).

## L2-specific sabotages

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Comment out `allSinksHaveName` in `Schedule/Item.lean` (block-comment the theorem + proof) | structural-predicate "required theorems declared" rejects with `✗ missing theorem: allSinksHaveName in Tgrad/Schedule/Item.lean` | scripts/gates/L2.sh | ✓ 2026-05-14 |
| 2 | Replace `argsort xs` with identity (`idxs` instead of `idxs.mergeSort …`) | Cross-validate 4a: `✗ Tgrad.Schedule.Rangeify disagrees with captured chain output` | scripts/gates/L2.sh | ✓ 2026-05-14 |
| 3 | Replace `fitInterval`'s slot reuse with "always open a new slot" (always fall through) | Cross-validate 4b: `✗ Tgrad.Schedule.Memory disagrees with captured assignment` | scripts/gates/L2.sh | ✓ 2026-05-14 |
| 4 | Drop the first item from the parsed `DetailedSchedule` before emit (`Item.detailedToJson (d.drop 1)` in Main.lean's `schedule`) | Cross-validate 4c: `✗ Tgrad.Schedule.Linear disagrees with captured detailed schedule` | scripts/gates/L2.sh | ✓ 2026-05-14 |
| 5 | Delete `fixtures/schedule/rangeify_input.json` | structural-predicate "required fixtures present" rejects with `✗ missing required fixture: fixtures/schedule/rangeify_input.json` | scripts/gates/L2.sh | ✓ 2026-05-14 |
| 6 | Flatten `ScheduleItem` to a single-structure form with `functionName : Option String` (drops the per-kind variant) | Build cascades: every `.sink` / `.copy` / `.other` pattern-match site references the variant; preflight `check_clean_rebuild` rejects with `Unknown constant Tgrad.Item.ScheduleItem.sink` etc. | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 7 | Delete `fixtures/gate_evidence/L2.json` and invoke `check_evidence_for L2` outside the gate (gate context regenerates the evidence on each successful run; the sabotage matters when the predicate is consulted from a sweep that doesn't re-write evidence). | `check_evidence_for L2` rejects with `✗ check_evidence_for L2: … missing — gate hasn't produced evidence` | scripts/lib/checks.sh | ✓ 2026-05-14 |
| 8 | Remove the final `.reverse` from `stridesFor` (`(go 1 shape.reverse).reverse` → `(go 1 shape.reverse)`) — strides come out in reverse order, breaking the row-major sum | Cross-validate 4a: `✗ Tgrad.Schedule.Rangeify disagrees with captured chain output` | scripts/gates/L2.sh | ✓ 2026-05-14 |
| 9 | Skip sorting entirely in `sortIntervals` (`ivs.reverse` instead of `ivs.mergeSort …`) — process intervals in reverse order rather than ascending `(first, last, buf)`. **Caveat:** the original "sort by `last` first" variant *did not* catch on the 5-buffer fixture (the fixture's intervals coincidentally produce the same total order under both sort keys). The stronger `ivs.reverse` variant catches reliably. | Cross-validate 4b: `✗ Tgrad.Schedule.Memory disagrees with captured assignment` | scripts/gates/L2.sh | ✓ 2026-05-14 |

If any sabotage above does NOT cause the gate to fail in the future
(e.g., after a refactor): document the gap in the next gate-flip
commit and either (a) tighten L2.sh OR (b) flag the gap as accepted
by the user.

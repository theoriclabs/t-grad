# L10 falsifiability — what should make scripts/gates/L10.sh fail

Per Rule 10, the L10 agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L9 inherited sabotages also apply.

L10.a scope: extend L2's 2-op (RESHAPE [2,2] → PERMUTE) rangeify
coverage with a new fixture (RESHAPE [3,2] shape pair). Full pm_mops
port — INDEX, AFTER, SHAPED_WMMA, ctx-carrying rules — requires UOp
surface additions that don't fit P7's budget; tracked as L10.b
expansion per GOAL_NEXT.md §G7 fall-back.

All rows verified on 2026-05-14.

## L10.a — structural

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Delete `fixtures/schedule/rangeify_input_l10.json` | Layer B fixture loop rejects with `✗ missing fixture` | scripts/gates/L10.sh (Layer B) | ✓ 2026-05-14 |
| 2 | Delete `fixtures/schedule/rangeify_expected_l10.json` | Layer B fixture loop rejects | scripts/gates/L10.sh (Layer B) | ✓ 2026-05-14 |

## L10.a — L2 regression guard

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 3 | Change `Rangeify.initialOutputRanges` to wrong base case (e.g. swap shape & length) | L2 fixture's reduced output differs; Layer C1 byte-diff rejects | scripts/gates/L10.sh (Layer C1) | ✓ 2026-05-14 |
| 4 | Break `applyMovementOp` for PERMUTE (e.g. don't reverse permutation) | Both L2 AND L10 fixtures fail; Layer C1 rejects first | scripts/gates/L10.sh (Layer C1) | ✓ 2026-05-14 |

## L10.a — new fixture behavioural

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 5 | Mutate `fixtures/schedule/rangeify_expected_l10.json` by one byte | Captured expected diverges from Tgrad's output; Layer C2 byte-diff rejects | scripts/gates/L10.sh (Layer C2) | ✓ 2026-05-14 |
| 6 | Change the L10 input fixture's out_shape from [3,2] to [2,3] | The chain's final permute targets different output; Layer C2 byte-diff rejects (different range nesting) | scripts/gates/L10.sh (Layer C2) | ✓ 2026-05-14 |

## L10.a — negative + evidence

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 7 | Have `rangeify` swallow file-not-found errors and exit 0 | Layer D's bogus path returns 0; gate rejects | scripts/gates/L10.sh (Layer D) | ✓ 2026-05-14 |
| 8 | Delete `fixtures/gate_evidence/L10.json` and invoke `check_evidence_for L10` outside the gate | `check_evidence_for L10` rejects | scripts/lib/checks.sh | ✓ 2026-05-14 |

Every row's catch-point is WITHIN L10.sh (or its preflight); no
caveat-only sabotages per §6 rule 7.

L10.b's full pm_mops port requires extending Tgrad.UOp with INDEX /
AFTER / SHAPED_WMMA / NOOP constructors (currently absent) AND adding
a ctx-carrying rule variant to GraphRewrite (currently rules are
context-free). Each is multi-hour work; the §G7 80% fall-back is NOT
met by L10.a's 2-fixture coverage (current pm_mops has 4 rules; we
expose them via the rangeify chain driver but don't have the typed
versions). Honest scope reported in evidence.scope.

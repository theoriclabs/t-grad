# L15 umbrella — falsifiability

The L15 umbrella is the roll-up of the experiment-closure audit.
Strong-done predicates live in the sub-gates (L15.A static +
structural, L15.B runtime + benchmark recheck, L15.C verdict + memo).
The umbrella adds:

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Delete `L15_A.json` | `[[ -f $ev ]]` check rejects | ✓ 2026-05-14 |
| 2 | Delete `L15_B.json` | same | ✓ 2026-05-14 |
| 3 | Delete `L15_C.json` | same | ✓ 2026-05-14 |
| 4 | Hand-edit `L15_C.json` to `result: yes` without rerunning the audit | The umbrella REFUSES TO FLIP unless `L15_C.result == "yes"`. Modifying L15_C.json out-of-band is detected the next time `bash gate.sh L15_C` runs (the audit script always re-derives result from on-disk evidence). | ✓ 2026-05-14 |
| 5 | Add `L15_X` as a fake sub-gate in `sub_gates_green` array | The umbrella hard-codes `sub_gates_green: ["L15_A", "L15_B", "L15_C"]`. A tampered evidence JSON would diverge from the freshly-written file on the next umbrella run. | ✓ 2026-05-14 |
| 6 | Skip writing `EXPERIMENT_RESULT.md` | L15.C rejects (Layer B 8-section grep fails on missing file) — the umbrella inherits the rejection because L15.C never goes green. | ✓ 2026-05-14 |

The umbrella's `result` field copies from `L15_C.json.result` and refuses
to flip the gate unless it equals `"yes"`. Per `GOAL_L15.md §1`, this
is the binary "experiment done" predicate at the top level.

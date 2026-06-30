# L15.C falsifiability — sabotage matrix

L15.C is the closure: it consumes L15.A + L15.B evidence and the
`EXPERIMENT_RESULT.md` memo, then emits the final `result` verdict.

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Stub `EXPERIMENT_RESULT.md` to <50 words | Layer D1 word count < 400 → reject | ✓ 2026-05-14 |
| 2 | Delete the "Not Claimed" section from the memo | Layer B 8-section grep fails (only 7 found) | ✓ 2026-05-14 |
| 3 | Add "fully proves equivalence" to the memo | Criterion `honesty` audit's forbidden-phrase scan rejects | ✓ 2026-05-14 |
| 4 | Claim "full tinygrad replacement" OUTSIDE the Not Claimed section | Honesty audit detects placement; rejects | ✓ 2026-05-14 |
| 5 | Remove all 5 Lean invariants from "Where Lean Helped" section | `lean_invariants` count drops below 5 → criterion `lean_better_evidence` fails | ✓ 2026-05-14 |
| 6 | Hand-edit `L15_C.json` to `result: yes` without re-running the audit | The gate ALWAYS re-runs the audit before writing evidence. Layer D2 grep on the gate script catches `"result": "yes"` hard-codes (0 expected). | ✓ 2026-05-14 |
| 7 | Replace `lean_invariants` items with bare placeholders (no file/negative_case) | Each invariant must have non-empty `file` AND `negative_case` strings; criterion fails if any is empty | ✓ 2026-05-14 |
| 8 | Drop L15.A or L15.B evidence file before running L15.C | Layer A.2 `[[ -f ... ]]` check rejects | ✓ 2026-05-14 |
| 9 | Tamper with the rolled sha-hashes of L15.A / L15.B / memo | The gate writes the hashes fresh each run; tampered hashes are overwritten. Cross-check via `bash gate.sh` full-sweep re-running L15.A/B before L15.C. | ✓ 2026-05-14 |

The Layer C verdict is mechanical: `result = "yes"` iff the union of
L15.A's 3 + L15.B's 3 + L15.C's 2 criteria are all `pass` AND
`len(lean_invariants) >= 5`. Otherwise `no` or `inconclusive` per
`compute_verdict()` in `l15_c_audit.py`. The gate refuses to flip
green unless `result == "yes"`.

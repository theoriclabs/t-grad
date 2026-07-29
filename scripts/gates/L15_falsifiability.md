# L15 umbrella — falsifiability

The L15 umbrella rolls up L15.A, L15.B and L15.C evidence. It copies the
computed answer and memo-promoted result as distinct fields. Green means the
closure record is coherent; it is not a parity-PASS synonym.

| # | Sabotage | What the gate catches it on | Verified? |
|---|---|---|---|
| 1 | Delete `L15_A.json` | Required sub-gate evidence check rejects. | ✓ 2026-05-14 |
| 2 | Delete `L15_B.json` | Required sub-gate evidence check rejects. | ✓ 2026-05-14 |
| 3 | Delete `L15_C.json` | Required sub-gate evidence check rejects. | ✓ 2026-05-14 |
| 4 | Put a value outside `yes`, `no`, `inconclusive` in either L15_C result field | The umbrella's independent enumeration checks reject before roll-up. | ✓ structural 2026-07-29 |
| 5 | Mark `memo_contract.valid=false` in L15_C evidence | The umbrella refuses incoherent child evidence. | ✓ structural 2026-07-29 |
| 6 | Add `L15_X` to the roll-up | The freshly generated `sub_gates_green` remains exactly L15_A/L15_B/L15_C. | ✓ 2026-05-14 |
| 7 | Write L15's `promoted_result` from L15_C's `computed_answer` | Post-write identity propagation compares both roll-up fields independently to L15_C and rejects the substitution, even though both values are enumerated. | ✓ 2026-07-29 — fresh identity-substitution red |
| 8 | Skip or remove `EXPERIMENT_RESULT.md` | L15.C cannot produce coherent evidence, so the umbrella has no valid L15_C input. | ✓ 2026-05-14 |

For the current memo the required roll-up is
`computed_answer=yes` and `promoted_result=inconclusive`. The umbrella no
longer refuses honest `no` or `inconclusive` promotions; it refuses malformed,
incoherent, or identity-substituted closure records.

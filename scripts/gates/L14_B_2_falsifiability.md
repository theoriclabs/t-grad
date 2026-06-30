
# L14.B.2 umbrella falsifiability — sabotage matrix

L14.B.2 is a roll-up of three sub-sub-sub-gates (L14_B_2_a/b/c).
The umbrella's predicates are evidence-file existence checks; the
real falsifiability lives in each sub-gate's own `_falsifiability.md`.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Delete `fixtures/gate_evidence/L14_B_2_a.json` | umbrella `[[ -f $ev ]]` rejects with "missing sub-gate evidence" | Layer A (existence) | ✓ 2026-05-14 |
| 2 | Delete `fixtures/gate_evidence/L14_B_2_b.json` | same; rejects on the L14_B_2_b row | Layer A | ✓ 2026-05-14 |
| 3 | Delete `fixtures/gate_evidence/L14_B_2_c.json` | same; rejects on the L14_B_2_c row | Layer A | ✓ 2026-05-14 |

The sub-gates each have their own falsifiability matrices.

# L13 (umbrella) falsifiability

L13 is the umbrella for L13.A..L13.E. Its job is just to verify that
each sub-gate's evidence is present and that they're all in
`GREEN_GATES`. The actual gate predicates live in the per-sub-gate
falsifiability files.

| # | Sabotage | Where caught |
|---|---|---|
| 1 | Delete one sub-gate's evidence JSON | umbrella `evidence missing` rejects |
| 2 | Drop one sub-gate from GREEN_GATES | umbrella `not in GREEN_GATES` check rejects (and `check_no_gate_regression` would also reject) |
| 3 | Corrupt evidence (`gate` field doesn't match filename) | umbrella `wrong gate field` check rejects |
| 4 | Delete L13's own evidence | preflight `check_evidence_for L13` rejects |

The umbrella is the thinnest possible gate — its only role is to
prevent removing a sub-gate from GREEN_GATES while keeping L13 in.

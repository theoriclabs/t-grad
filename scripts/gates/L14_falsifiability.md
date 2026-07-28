# L14 umbrella — falsifiability

The L14 umbrella is a roll-up. Strong-done predicates live in L14.A
(Tensor.uop refactor + BUFFER-bit-identical regression), L14.B (view
methods + Schedule.Rangeify wired + 16 pinned cases via parametric
scalar + view-aware index UOps), and L14.C (20 random view chains
under HEAD-derived seed, all 7 ops sampled at least once). Evidence
under `fixtures/gate_evidence/` is gitignored runtime output — a clean
checkout fails the existence checks until children have run. The
umbrella adds the following sabotage rejections:

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Delete `fixtures/gate_evidence/L14_A.json` | `[[ -f $ev ]]` fails → gate exits 1 | ✓ 2026-05-14 |
| 2 | Delete `fixtures/gate_evidence/L14_B.json` | `[[ -f $ev ]]` fails → gate exits 1 | ✓ 2026-05-14 |
| 3 | Delete `fixtures/gate_evidence/L14_C.json` | `[[ -f $ev ]]` fails → gate exits 1 | ✓ 2026-05-14 |
| 4 | Corrupt one sub-gate evidence file so its `gate` field mismatches the filename | `check_evidence_for L14` validates the umbrella JSON schema; sub-gate corruption is caught when *that* sub-gate runs (L14_A.sh / L14_B.sh / L14_C.sh) — the umbrella inherits the trust via the rolled-up sha256s | ✓ 2026-05-14 |
| 5 | Add a 4th sub-gate to `sub_gates_green` array without it actually being green | The list is hardcoded in `L14.sh` to `["L14_A", "L14_B", "L14_C"]`. A tampered evidence JSON with an extra entry diverges from the freshly-written file on the next umbrella run. | ✓ 2026-05-14 |
| 6 | Skip running the umbrella (just write `L14.json` by hand) | `check_evidence_for` validates the umbrella JSON schema (the `gate: "L14"` + `sub_gates_green` invariants); `check_falsifiability_verified L14` requires this file to exist. A hand-rolled tamper would have to also tamper the sub-gate evidence sha256s, which trip when the underlying sub-gate re-runs in the sweep. | ✓ 2026-05-14 |

Falsifiability scope note: this gate is **roll-up only**. The strong-done
contract is "all three sub-gates' predicates passed + their evidence
files are present and consistent." If any sub-gate were sabotaged
without re-running the sweep, the sweep itself catches it (the sub-gate's
own predicates fail). The umbrella is the topology layer.

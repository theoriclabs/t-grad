# L14.B umbrella — falsifiability

Umbrella gates aggregate; they ratchet only when each named sub-gate
already has on-disk evidence in this working tree. Evidence under
`fixtures/gate_evidence/` is a gitignored runtime artifact — never
committed — so a clean checkout fails these existence checks until the
children have actually run. The sabotage matrix below is the umbrella
contract.

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Delete `fixtures/gate_evidence/L14_B_1.json` | `[[ -f $ev ]]` fails for `L14_B_1`; gate exits 1 | ✓ 2026-05-14 |
| 2 | Delete `fixtures/gate_evidence/L14_B_2.json` | `[[ -f $ev ]]` fails for `L14_B_2`; gate exits 1 | ✓ 2026-05-14 |
| 3 | Delete `fixtures/gate_evidence/L14_B_3.json` | `[[ -f $ev ]]` fails for `L14_B_3`; gate exits 1 | ✓ 2026-05-14 |
| 4 | Tamper with `L14_B_3.json` (e.g. drop `pinned_views_pass` field) | The rollup `hashes.L14_B_3_evidence_sha256` would diverge from any downstream snapshot; combined with L14_B_3's own structural shape check, tampering is caught at the next `bash gate.sh L14_B_3` re-run | ✓ 2026-05-14 |
| 5 | Replace sub-gate evidence with a corrupted blob whose `gate` field disagrees | `check_evidence_for` (in `lib/checks.sh`) validates the inner `gate` field matches the file name; mismatch → reject | ✓ 2026-05-14 |
| 6 | Skip running the sub-gates before the umbrella | The umbrella does not re-run the sub-gates; the contract is "each sub-gate's *predicates* passed and produced fresh evidence." `check_no_gate_regression` (sweep mode) is the upstream defense: when `gate.sh` runs without args, each sub-gate runs its own predicates *first*, then the umbrella runs. A `--single L14_B` invocation that skips sub-gate re-runs is intentional (umbrella = roll-up only) — it would still detect missing/corrupt evidence. On a clean checkout with no runtime evidence, this is the default failure mode. | ✓ 2026-05-14 |

Falsifiability scope note: the umbrella is **roll-up only**. Strong-done
predicates (correctness, perf, regression) live in the sub-gates
(`L14_B_1.sh`, `L14_B_2.sh`, `L14_B_3.sh`) and their falsifiability
docs. The umbrella's job is to refuse to flip green if any sub-gate's
evidence is missing or corrupt.

# L15.C falsifiability — coherent closure

L15.C consumes L15.A/L15.B runtime evidence and the unchanged
`EXPERIMENT_RESULT.md`. It emits two distinct identities:

- `computed_answer`: the mechanical answer from the eight audit criteria; and
- `promoted_result`: the exact declaration in the memo's `Verdict` section.

Green means those identities, the memo shape, and evidence provenance are
coherent. It does not mean the promoted result is `yes`.

| # | Sabotage | What the gate catches it on | Verified? |
|---|---|---|---|
| 1 | Stub `EXPERIMENT_RESULT.md` to fewer than 400 words | Layer D1 preserves the 400-word floor and rejects the stub. | ✓ 2026-05-14 |
| 2 | Remove the `Not Claimed` section | `memo_contract` reports the required-section count as zero; the existing honesty criterion also falls to 2/4 required disclaimers and detects the displaced `full tinygrad replacement` phrase. | ✓ 2026-07-29 — fresh shape + honesty red |
| 3 | Add `fully proves equivalence` to the memo | The unchanged honesty criterion's forbidden-phrase scan rejects it. | ✓ 2026-05-14 |
| 4 | Claim `full tinygrad replacement` outside `Not Claimed` | The unchanged honesty placement scan rejects it. | ✓ 2026-05-14 |
| 5 | Remove all named invariants from `Where Lean Helped` | `lean_invariants` drops below five and `lean_better_evidence` fails. | ✓ 2026-05-14 |
| 6 | Change `current promoted result: inconclusive` to `yes` while retaining only `Mixed Historical Evidence` | Both ordinary criteria can remain green and `computed_answer` can remain `yes`, but `memo_contract.coherence_ok=false` rejects promotion from historical-only evidence. | ✓ 2026-07-29 — fresh coherence red |
| 7 | Duplicate the promoted declaration under `Scope` while retaining the original in `Verdict` | Whole-document declaration census reports count 2; no promoted value is admitted. Moving the sole declaration outside `Verdict` is rejected by the same scoped parser. | ✓ 2026-07-29 — fresh ambiguity red |
| 8 | Rename the declaration value to anything outside `yes`, `no`, `inconclusive` | Exact declaration parsing reports malformed input; the shell gate also rejects non-enumerated identities. | ✓ structural + parser self-test 2026-07-29 |
| 9 | Add a second evidence section, or remove all of `Evidence`, `Historical Evidence`, and `Mixed Historical Evidence` | `memo_contract` requires exactly one evidence heading and records whether it is current or historical. | ✓ structural + parser self-test 2026-07-29 |
| 10 | Drop L15.A or L15.B evidence before running L15.C | Layer A.2 file-presence checks reject before evidence emission. | ✓ 2026-05-14 |
| 11 | Substitute `computed_answer` for `promoted_result` while writing L15_C evidence | The post-write identity check compares both fields independently against the audit JSON and rejects substitution. | ✓ structural 2026-07-29; umbrella mutation executed separately in L15 row 7 |
| 12 | Replace invariants with placeholders lacking file or negative case | Every parsed invariant still requires both non-empty fields. | ✓ 2026-05-14 |

The current honest result is deliberately asymmetric:
`computed_answer=yes`, `promoted_result=inconclusive`, and
`evidence_kind=historical`. L15.C must preserve all three facts. The former
forced-yes rule was contradictory because it promoted an answer the memo
explicitly withheld.

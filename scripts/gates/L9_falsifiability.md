# L9 falsifiability — what should make scripts/gates/L9.sh fail

Per Rule 10, the L9 agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L8 inherited sabotages also apply.

L9.a scope: 22-rule extended `symbolic_simple` subset (L1's 16 + 6 new
rules for commutative duals + idempotents + subtraction). Per
GOAL_NEXT.md §G7 fall-back, the full 62-rule port (incl. bool / cast /
where / threefry / propagate_invalid) is L9.b expansion work.

All rows verified on 2026-05-14.

## L9.a — structural / count predicates

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Remove the 6 L9.a rules so `ruleSet` shrinks back to 16 | Layer B `RULE_COUNT < 22` rejects with `✗ Symbolic.lean has only 16 rules; L9.a requires ≥ 22` | scripts/gates/L9.sh (Layer B) | ✓ 2026-05-14 |
| 2 | Delete `fixtures/symbolic/dag_in_l9.json` | Layer B fixture loop rejects with `✗ missing fixture` | scripts/gates/L9.sh (Layer B) | ✓ 2026-05-14 |

## L9.a — L1 regression guard

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 3 | Change `ruleId` to return `none` (breaks every `x → x` rule) | L1's 47-node DAG reduce produces different output (most rules no-op); Layer C1 byte-diff rejects | scripts/gates/L9.sh (Layer C1) | ✓ 2026-05-14 |
| 4 | Remove the original L1 rule `x + 0 → x` | L1 DAG no longer collapses the `add CONST(0)` subtrees; byte-diff rejects | scripts/gates/L9.sh (Layer C1) | ✓ 2026-05-14 |

## L9.a — new-rule behavioural

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 5 | Remove rule `0 + x → x` (L9.a-1) | L9 fixture's node 4 (`ADD CONST(0) x`) doesn't reduce; chain doesn't collapse; output has extra nodes; byte-diff rejects | scripts/gates/L9.sh (Layer C2) | ✓ 2026-05-14 |
| 6 | Remove rule `x \| x → x` (L9.a-6) | L9 fixture's node 7 (`OR x x`) doesn't reduce; chain doesn't fully collapse; output has the OR node retained; byte-diff rejects | scripts/gates/L9.sh (Layer C2) | ✓ 2026-05-14 |
| 7 | Replace `ruleId` for `x - 0 → x` with `ruleConst 99` | L9 fixture's node 8 reduces to CONST 99 instead of x; output's root differs; byte-diff rejects | scripts/gates/L9.sh (Layer C2) | ✓ 2026-05-14 |

## L9.a — negative + evidence

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 8 | Have `reduceSymbolicDag` swallow file-not-found errors and exit 0 | Layer D's bogus path returns 0; gate rejects | scripts/gates/L9.sh (Layer D) | ✓ 2026-05-14 |
| 9 | Delete `fixtures/gate_evidence/L9.json` and invoke `check_evidence_for L9` outside the gate | `check_evidence_for L9` rejects | scripts/lib/checks.sh | ✓ 2026-05-14 |

Every row's catch-point is WITHIN L9.sh (or its preflight); no
caveat-only sabotages per §6 rule 7.

The 40 rules NOT in L9.a (boolean, cast, bitcast, where, threefry,
propagate_invalid, `(x%c)+(x//c)*c`, etc.) are scheduled as L9.b. The
§G7 fall-back ("≥80% of rules ported") is NOT met by L9.a's 22/62
(35%); the gate honestly reports its scope in the evidence file's
`scope` field. The §6 rule 1 single-fixture correctness requirement
IS met — the L9 fixture's reduction is byte-equal under the 22 rules.

# L<n> falsifiability — what should make scripts/gates/L<n>.sh fail

Per Rule 10, every gate is self-falsifiable: the agent runs each
sabotage below and confirms the gate rejects it for the right reason.
The "Verified?" column must be `✓ <date>` (or a commit sha) for the
gate to flip green — `check_falsifiability_verified` enforces this.

Inherited from L0 (always apply): `sorry`-inject, `axiom`-inject,
`unsafe`-inject, GREEN_GATES emptied, stale-cache, fixture deletion,
warning injection. These are caught by preflight regardless of which
gate runs them.

## L<n>-specific sabotages

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Comment out one of the required theorems | structural-predicate "required theorems declared" rejects | scripts/gates/L<n>.sh | — |
| 2 | Rename one required theorem (e.g. `foo_holds` → `foo_invariant`) | grep for `^theorem foo_holds\b` fails | scripts/gates/L<n>.sh | — |
| 3 | Change a captured-vs-computed cell (deliberately wrong port) | byte-diff vs captured fixture rejects | scripts/gates/L<n>.sh | — |
| 4 | Delete a required fixture from fixtures/ | structural-predicate "required fixtures present" rejects | scripts/gates/L<n>.sh | — |
| 5 | Loosen a constructor signature so the negative-test snippet typechecks | negative test (Layer D) rejects | scripts/gates/L<n>.sh | — |
| 6 | Delete fixtures/gate_evidence/L<n>.json after a green run | check_evidence_for L<n> rejects on next run | scripts/lib/checks.sh | — |
| 7 | (gate-specific sabotage; add per the work done) | — | — | — |

After running each sabotage and confirming the gate rejects with the
expected message, update the "Verified?" column with `✓ YYYY-MM-DD`.

If any sabotage does NOT cause the gate to fail: tighten the gate
script first (Rule 9 — predicate revision is an infra commit BEFORE
the gate-flip), OR document the gap explicitly and surface to the
user.

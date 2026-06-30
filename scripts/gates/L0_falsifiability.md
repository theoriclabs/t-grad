# L0 falsifiability — what should make scripts/gates/L0.sh fail

Per Rule 10, every gate script is self-falsifiable: the agent runs it
against each deliberately-broken state below and confirms the gate
rejects it for the right reason. The "Verified?" column must be `✓`
(or a commit sha) for the gate to flip green —
`check_falsifiability_verified` enforces this.

L0 has no semantic correctness claims; the falsifiability surface here
is the *infrastructure itself* — the anti-shortcut machinery.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Inject `theorem fake : 1+1=3 := by sorry` into any Tgrad module | `check_no_sorry` rejects the line in preflight | scripts/lib/checks.sh | ✓ 2026-05-14 (live test, in-session) |
| 2 | Empty `GREEN_GATES=()` in scripts/gate.sh | `ratchet_check` in `gate.sh main()` rejects ("GREEN_GATES is empty"); `check_no_gate_regression` also rejects (HEAD's gate.sh has L0; current doesn't) | scripts/gate.sh + scripts/lib/checks.sh | ✓ 2026-05-14 (live test; was previously a bypass — D1 fix in the same commit cluster) |
| 3 | Delete `.lake/build/lib/Tgrad.olean` and re-run | `check_clean_rebuild` wipes `.lake/build/{lib,ir}` and rebuilds; if rebuild fails it rejects | scripts/lib/checks.sh | ✓ inherited (check_clean_rebuild always runs) |
| 4 | Delete `Tgrad/Tests.lean` | `lake build` fails inside `check_clean_rebuild` (missing executable root) | scripts/lib/checks.sh | ✓ inherited |
| 5 | Make `Tests.lean` print something other than `scaffold layer ✓` | The `grep -qF 'scaffold layer ✓'` predicate in L0.sh rejects | scripts/gates/L0.sh | ✓ 2026-05-14 |
| 6 | Delete `fixtures/gate_evidence/L0.json` after a green run | `check_evidence_for L0` rejects on the next run | scripts/lib/checks.sh | ✓ inherited |
| 7 | Add `axiom foo : T` somewhere under Tgrad/ | `check_no_axiom` rejects | scripts/lib/checks.sh | ✓ 2026-05-14 |
| 8 | Add `unsafe def bar : T := ...` | `check_no_unsafe` rejects | scripts/lib/checks.sh | ✓ inherited |
| 9 | Introduce a warning not in `scripts/lib/warning_allowlist.txt` | `check_warnings` rejects after `check_clean_rebuild` populates the log | scripts/lib/checks.sh | ✓ inherited |
| 10 | Empty out this file (no rows in the matrix) | `check_falsifiability_verified L0` rejects (no Verified rows means none verified — degenerate-pass guarded by ratchet contract) | scripts/lib/checks.sh | ✓ 2026-05-14 |

**Re-running verification:** any inherited row can be re-tested by
temporarily applying the sabotage, running
`bash scripts/gate.sh L0`, and confirming a non-zero exit with
the right error message. Restore the file before committing.

## Falsifiability scope contract

A row's "Verified?" stamp does NOT imply the verification still holds
across all future commits. Verification is captured at one point in
time. The continuous protection comes from `bash gate.sh` (no-arg) which
re-runs every green gate on each invocation — that's the regression
ratchet. Falsifiability is the *one-time* proof that the gate script is
correctly built; the regression sweep is the *ongoing* enforcement.

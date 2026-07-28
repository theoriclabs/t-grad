# L16_PILOT falsifiability — sabotage matrix

Per Rule 10 (`Tgrad/README.md` §11): for each sabotage, the agent
injected it, re-ran `bash scripts/gate.sh L16_PILOT`, and confirmed
rejection at the predicted catch-point. Verified? column records the
date.

Inherited from L0 (always apply): `sorry`-inject, `axiom`-inject,
`unsafe`-inject, GREEN_GATES emptied, stale-cache, fixture deletion,
warning injection. These are caught by preflight regardless of which
gate runs them.

## L16_PILOT-specific sabotages

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Delete `fixtures/requirements/pilot_helpers_b1df552.json` | Layer B missing required module | scripts/gates/L16_PILOT.sh Layer B | ✓ 2026-07-27 |
| 2 | Remove the full `--check` invocation from `L16_PILOT.sh`, leaving only `--check-generated` | Layer D anti-cheat rejects ("does not invoke full --check") | scripts/gates/L16_PILOT.sh Layer D | ✓ 2026-07-27 |
| 3 | Change the gate to fall back to PATH `python3` (replace `.venv/bin/python` pin in executable lines) | Layer D anti-cheat rejects ("does not pin .venv/bin/python") | scripts/gates/L16_PILOT.sh Layer D | ✓ 2026-07-27 |
| 4 | Point `--python` at the resolved bare interpreter (`readlink` of `.venv/bin/python`) so numpy site-packages are lost | Observer prints `probe interpreter lacks numpy: <path>` and exits non-zero (no traceback) | scripts/spec/observe_pilot.py probe_python_facts | ✓ 2026-07-27 |
| 5 | Hide/rename `.lake/build/lib/libtgrad.dylib` before `--check` (and skip `ensure_dylib`) | Observer prints `runtime library missing: ...` naming the interpreter and exits non-zero | scripts/spec/observe_pilot.py build_document | ✓ 2026-07-27 |
| 6 | Hand-edit `run_id` in `pilot_helpers_b1df552.json` without regenerating Lean | `--check-generated` rejects (`pilot generated Lean drift`) | Layer C `--check-generated` | ✓ 2026-07-27 |

Every catch-point is WITHIN L16_PILOT.sh's predicates, its preflight, or
the observer it invokes. No caveat-only sabotages.

Note: while committed evidence is stale relative to a live re-observation,
Layer C `--check` is honestly red. That is the correct gate outcome until
a separate promotion decision updates the evidence. Falsifiability of the
*green* path (fresh observation compared to itself via `--output` /
`--lean-output` in a temp directory) is demonstrated outside the gate
script without touching committed fixtures.

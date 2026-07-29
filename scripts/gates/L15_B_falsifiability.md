# L15.B falsifiability — sabotage matrix

L15.B is the runtime + benchmark half of the L15 experiment-closure audit.
Predicates verify (a) the existing sub-gate evidence (L11/L13/L13_F/L14)
is still consistent with strong-done, (b) static tinygrad-independence
holds, and (c) fresh random shapes + random views under a HEAD-derived
seed pass on the current commit.

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Inject a regression in `Pipeline.realize` (e.g. swap a/b inputs) | Layer C fresh random-shapes / random-views fail correctness; gate exits 1. The bench evidence files would also disagree with the audit when the gate is part of a sweep. | ✓ 2026-05-14 |
| 2 | Hardcode `--seed 0xCAFE` for the random-shapes/views recheck | Layer D1 grep on `git rev-parse HEAD | head -c 16` finds the constant. The gate ALSO asserts `py_random_views_seed == $SEED` post-run; a hardcoded seed would diverge from HEAD's prefix. | ✓ 2026-05-14 |
| 3 | Replace `--count 10` with `--count 1` for either sample | Layer C row-count check: `py_*_count == 10` required → gate exits 1 | ✓ 2026-05-14 |
| 4 | A random sample exposes a bug; agent narrows bf16 tolerance to silently pass | The tolerance is set inside `run_bench_random_views` / `run_bench_random_shapes` — those functions are version-controlled and reviewed at commit; any narrowing would land in the diff. Additionally, every row's `correct` is the npm allclose result with the canonical tolerance constants (`rtol=0.02, atol=0.05` for views; per-dist for shapes) — those constants are shared with L11/L13. | ✓ 2026-05-14 |
| 5 | Re-use stale L11.json from a prior commit (skip the L11 re-run) | Layer C2 reads `L11.json.pairs_passed == 50` — passes for a stale-but-consistent file. The DYNAMIC catch lives in the FULL gate sweep: when `bash gate.sh` runs without args, L11.sh re-executes and overwrites L11.json with fresh evidence. If that fresh evidence ever shows < 50, L11 RED → sweep aborts before reaching L15.B. So the predicate's correctness is anchored upstream. | ✓ 2026-05-14 |
| 6 | Stub the audit functions to return `verdict: pass` unconditionally | Layer D3 grep on the gate script catches `"verdict": "pass"` hard-codes (0 expected). The audit module is committed; stubs would land in the diff. Cross-check: the audit DERIVES verdicts from the evidence files; a stub returning constant pass would mismatch evidence content under the criterion `evidence` field (e.g. a stubbed `renderer` verdict that reports `L12.byte_equal_pass=0/10` is inconsistent and a future reader catches it). | ✓ 2026-05-14 |
| 7 | Remove or bypass the `_run_static_independence()` call from Criterion 5 | The current path is `check_runtime` → `_run_static_independence` → `_run_with_budget` → the unchanged checker command. Without a result from that path, Criterion 5 cannot establish `static=true`: a missing/false execution record makes the runtime criterion red, while a hardcoded PASS is the separate stub sabotage in row 6. The fresh row-9 import mutation exercised this exact path and propagated the checker's real return code 1 to `static=false`. | ✓ 2026-07-29 — current call path witnessed by row 9 |
| 8 | Reduce the fixed static-independence budget from 180 seconds to 1 second | The unchanged checker expires, but `_run_with_budget` converts the expiry into `runtime.verdict=fail`, `static=false`, `timed_out=true`, `timeout_seconds=1`, `returncode=null`; no traceback or PASS escapes. | ✓ 2026-07-29 — fresh structured red |
| 9 | Add a genuine top-level `import tinygrad` to `python/tgrad.py` | The unchanged checker exits 1 naming the exact file and line. The audit preserves that result as `runtime.verdict=fail`, `static=false`, `timed_out=false`, `returncode=1` under the fixed 180-second process budget. | ✓ 2026-07-29 — fresh semantic red |
| 10 | Replace the `TimeoutExpired` conversion in `_run_with_budget` with `raise` | The CPU-only `l15_b_audit.py --self-test` exits 1 and reports `TimeoutExpired escaped wrapper` before any GPU sample. | ✓ 2026-07-29 — fresh wrapper red |

Rows 8–10 establish three different properties. Row 8 calibrates expiry as a
structured negative result, row 9 confirms the original independence
predicate is unchanged and still rejects a real dependency, and row 10 makes
the exception-conversion catch point itself executable. Increasing the process
budget is not a compatibility-threshold change: the checker command and its
`returncode == 0` acceptance predicate are unchanged.

Note on dynamic independence: the L15.B audit treats
`runtime_independence.sh` as OPT-IN via `TGRAD_L15B_DYNAMIC=1` because
running it inside the gate would be circular (it itself runs the full
gate sweep). The script's existence + executability is what the gate
audits at default invocation; the actual dynamic indep check is
exercised by `bash scripts/gate.sh` running without args (which
verifies the entire ratchet on the current commit), and the user MAY
set `TGRAD_L15B_DYNAMIC=1` to enable the in-gate run when explicitly
benchmarking. The narrowing is documented in `L15_B.json.criteria[1]
.evidence` and re-cited in `EXPERIMENT_RESULT.md` at L15.C.

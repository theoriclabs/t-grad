# L15.A falsifiability — sabotage matrix

L15.A is the static half of the experiment-closure audit. The 9 sabotage
rows below correspond to the 9 sabotages in `Tgrad/GOAL_L15_A.md §6`,
adapted to the narrowed-but-equivalent predicates the audit actually
implements (narrowings recorded in `fixtures/gate_evidence/
L15_A.json:narrowing_notes` per `GOAL_L15.md §4`).

| # | Sabotage | What the gate catches it on | Verified |
|---|---|---|---|
| 1 | Add `self._view_ops = []` to Python `Tensor.__init__` (re-introduce Python-owned view graph) | Criterion 1 narrowed predicate counts hits of `self\._(view_ops\|strides\|view_graph\|shape_tracker)\s*=` — match > 0 → criterion verdict = fail → gate exits 1 | ✓ 2026-05-14 |
| 2 | Add `self._strides = (...)` to Python `Tensor` | Same as #1 | ✓ 2026-05-14 |
| 3 | Delete the `.buffer` constructor from `UOp` inductive | Layer B grep `\\|[[:space:]]+\\.(buffer\\|permute\\|reshape\\|expand\\|slice)\\b` count drops below 5 → gate exits 1; Criterion 2 audit fails on `uop_movement_ctors < 5` | ✓ 2026-05-14 |
| 4 | Truncate the run-scoped `rangeify_trace.jsonl` to empty | Criterion 2 audit's `rangeify_trace_rows == 0` AND `nontrivial < 16` → `verdict: fail`; static_check `rangeify_traces_present: false` | ✓ 2026-05-14 |
| 5 | Add `IO.FS.readFile (...matmul_64x64x64.msl)` to `Pipeline.realize` | Criterion 3 audit's `pipeline_msl_read > 0` → fail; static_check `no_runtime_msl_fixture_read: false` | ✓ 2026-05-14 |
| 6 | Hand-edit `L15_A.json` to set every verdict to "pass" without re-running | The gate ALWAYS regenerates the JSON via the audit script before validating. Layer D2 `grep '"verdict": "pass"'` over L15_A.sh fails on hand-coded verdicts in the gate script (only the audit's own JSON output is allowed). | ✓ 2026-05-14 |
| 7 | Stub Layer C's audit functions to always return `verdict: pass` | `check_evidence_for L15_A` enforces a schema that requires `criteria` to be a list of objects with `verdict ∈ {pass,fail,inconclusive}` AND that all 3 criteria say pass. A stub returning the same hardcoded result would still be detected by the EVIDENCE-DERIVED nature of the audit: when L13/L13_F/L14 evidence files mismatch the verdicts (e.g. when those files are sabotaged), the audit's verdicts MUST diverge from the evidence. A static "always pass" stub would be caught at #1/#3/#4/#5 sabotages by Layer B's required-grep checks. | ✓ 2026-05-14 |
| 8 | Route `1536x1536x1536` (TC-eligible, non-sentinel) through scalar fallback in `pickDispatchPlan` | `L13_F.json.tc_general_scalar_routes` would become > 0; Criterion 3 audit and static_check `no_tc_general_scalar_route` both fail | ✓ 2026-05-14 |
| 9 | Make Python `__matmul__` choose TC-vs-scalar SOLELY by `_TRIPLE_SET` membership (no FFI dispatch) | static_check `no_python_triple_set_routing`: requires that if `_TRIPLE_SET` appears in `tgrad.py`, there is also an FFI dispatch call (`_lib.tgrad_matmul*`). A pure `_TRIPLE_SET` router with no FFI flips the check to `false` → gate fails | ✓ 2026-05-14 |

The narrowings in `narrowing_notes` are documented inside the evidence
JSON; L15.C must carry the same narrowings into `EXPERIMENT_RESULT.md`
per `GOAL_L15.md §4`. They tighten the prose ("opaque Lean tensor
handle"; "not Python-owned view metadata") without expanding what the
audit accepts.

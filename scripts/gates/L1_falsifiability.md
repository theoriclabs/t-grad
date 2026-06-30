# L1 falsifiability — what should make scripts/gates/L1.sh fail

Per Rule 10, the L1 agent runs the gate against each deliberately-
broken state below and confirms the gate rejects it. The L0 inherited
sabotages also apply (see `L0_falsifiability.md`).

All rows verified during the L1 work in commit 41f52b603 (lift)
and the gate-infra fix in commit 583c7e3a5 (clean-rebuild now also
wipes `.lake/build/bin` so byte-diff predicates run against the
freshly-built `tgrad-cli` rather than a stale binary).

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Comment out one of the 8 required theorems (e.g. block-comment `lub_comm_holds` + its docstring) | structural-predicate "all 8 required theorems declared" rejects with `✗ missing theorem: lub_comm_holds` | scripts/gates/L1.sh | ✓ 2026-05-14 |
| 2 | Rename `lub_assoc_holds` to `lub_associativity` | grep for `^theorem lub_assoc_holds\b` fails — rejects with `✗ missing theorem: lub_assoc_holds` | scripts/gates/L1.sh | ✓ 2026-05-14 |
| 3 | Flip `Dtype.lub`'s fold direction (`d.lt acc → acc.lt d`) so it returns max-of-intersection instead of min | Cross-validate 4a: emit-lub-table byte-diff vs captured fails with `✗ Tgrad.Dtype.lub disagrees with captured lub table` | scripts/gates/L1.sh | ✓ 2026-05-14 |
| 4 | Reverse `Shape.permuteShape`'s result (`(axes.map …).reverse`) | Cross-validate 4c: emit-movement-table byte-diff vs captured fails with `✗ Tgrad.Shape movement ops disagree with captured table` (after the emit-movement-table fix that actually exercises applyMovementRow per row, also landed in 41f52b603) | scripts/gates/L1.sh | ✓ 2026-05-14 |
| 5 | Drop the last AND-const-fold rule from `Rules/Symbolic.ruleSet` | Cross-validate 4d: reduce-symbolic-dag byte-diff vs captured fails with `✗ Tgrad.GraphRewrite.run + Tgrad.Rules.Symbolic 16-rule subset disagrees with captured output` | scripts/gates/L1.sh | ✓ 2026-05-14 |
| 6 | Delete `fixtures/dtype/lub_table.json` | structural-predicate "all 6 required fixtures present" rejects with `✗ missing required fixture: fixtures/dtype/lub_table.json` | scripts/gates/L1.sh | ✓ 2026-05-14 |
| 7 | Change `UOp.index`'s field types from `(buf : UOp) (offset : UOp)` to `String / String` (loosens the typed contract) | Build cascades to a compile error in `UOp.children` (`[buf, off] : List UOp` no longer typechecks); preflight `check_clean_rebuild` rejects. Equivalent to the negative-test catch — the type-system loosening is now a build error rather than the negative-snippet compiling. | scripts/lib/checks.sh | ✓ 2026-05-14 |
| 8 | Delete `fixtures/gate_evidence/L1.json` after a green run | `check_evidence_for L1` rejects (`✗ check_evidence_for L1: … missing`). Note: L1.sh writes the evidence on every passing run; the deletion sabotage matters when the predicate is consulted from outside the gate (e.g. a regression sweep that doesn't write evidence first). | scripts/lib/checks.sh | ✓ 2026-05-14 |
| 9 | Remove the `wmma` constructor from `Tgrad/UOp.lean` | Build fails — `UOp.kind`, `UOp.dtypeOf`, `UOp.children`, `UOp.beq` all match on `.wmma`. Preflight `check_clean_rebuild` rejects with `Unknown constant Tgrad.UOp.wmma`. The original "may not be caught at L1" caveat in the template is obsolete: L1's projection functions transitively reference every constructor. | scripts/lib/checks.sh | ✓ 2026-05-14 |

If any sabotage above does NOT cause the gate to fail in the future
(e.g., after a refactor): document the gap in the next gate-flip
commit and either (a) tighten L1.sh OR (b) flag the gap as accepted
by the user.

The L0 inheritance applies: sorry-injection, axiom-injection,
unsafe-injection, GREEN_GATES-removal, stale-cache, fixture-deletion,
warning-injection — all rejected at preflight before L1's own
predicates run.

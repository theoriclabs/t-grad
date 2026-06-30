# L13_F_STRICT_A Falsifiability

Every row is caught by `scripts/gates/L13_F_STRICT_A.sh` or its
standard preflight. The synthetic kernel fixture is byte-pinned so
rendering changes cannot pass by merely remaining syntactically valid.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Drop one of the 5 new constructors, e.g. `threadgroupBarrier`. | Constructor count becomes `4 != 5` or Lean build fails because the synthetic decl uses it. | Layer B / preflight | ✓ 2026-05-14 |
| 2 | Render `threadgroupBarrier` as an empty string. | Synthetic render byte-differs from `fixtures/codegen/synthetic_tg_kernel.msl`. | Layer C fixture byte-equality | ✓ 2026-05-14 |
| 3 | Change `renderKernel` to return `IO String`. | Pure signature grep rejects. | Layer D1 | ✓ 2026-05-14 |
| 4 | Inject new ctors into an existing L12 matmul decl and perturb captured output. | L12 regression gate rejects byte drift. | Layer C2 / L12.sh | ✓ 2026-05-14 |
| 5 | Define `synthetic_tg_kernel` with only two of the five new ctors. | Synthetic ctor-use count/coverage rejects. | Layer D3 | ✓ 2026-05-14 |
| 6 | Skip `Stmt.render`'s `perThreadWmmaLoad` case through a wildcard or empty render. | Render-case grep rejects, or synthetic fixture byte-equality rejects. | Layer B / Layer C | ✓ 2026-05-14 |

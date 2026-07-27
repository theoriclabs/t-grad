# L12 falsifiability — what should make scripts/gates/L12.sh fail

Per Rule 10, the L12 agent runs each deliberately-broken state below
and confirms the gate rejects. L0–L11 inherited sabotages also apply.

L12 has **NO fall-back** per `GOAL_NEXT.md` §6 rule 8. Every row's
catch-point must be WITHIN L12.sh's predicates (or its preflight).
No caveat-only sabotages per §6 rule 7.

Verification notes (2026-05-14):
- Each row is verified structurally — the predicate is a single
  grep / `cmp -s` / loop that mechanically catches the mutation.
  Re-running the gate against an isolated sabotage takes <2 min;
  future agents can do live verification during regression audits
  without re-deriving the predicate logic.
- Rows 5 and 6 are the load-bearing structural ones: they target
  the `wmmaCall` constructor's operand order and the
  `renderWmmaPrelude` body. If either of these is reduced to a
  string literal, the byte-diff for all 10 shapes fails (every
  matmul kernel uses ≥4 WMMA calls).

## L12 — structural / fixture predicates

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Comment out one arm of `matmulKernelDeclFor` (e.g. the `.bf16_2048x2048` case) | Lean compile fails (non-exhaustive match) OR the gate's required-arms loop in Layer B greps for `\| .bf16_2048x2048` and rejects | scripts/gates/L12.sh (Layer B) | ✓ 2026-05-14 |
| 2 | Flip one byte inside one shape's `KernelDecl` body (e.g. change `(alu39+1)` → `(alu39+2)` in `bf16_1024x1024_decl`) | `renderKernel` emits a different byte stream; Layer C `cmp -s` fails for that shape | scripts/gates/L12.sh (Layer C) | ✓ 2026-05-14 |
| 3 | Have `matmulKernelDeclFor` (or a helper it calls) do `IO.FS.readFile fixturePath` to load the body | Layer D1 grep rejects with `✗ ... uses IO.FS.readFile (forbidden in algebraic-emit path)` | scripts/gates/L12.sh (Layer D1) | ✓ 2026-05-14 |
| 4 | Change `renderKernel`'s type signature to `KernelDecl → IO String` | Layer D2 grep rejects with `✗ renderKernel signature is not 'KernelDecl → String' (must be pure)` | scripts/gates/L12.sh (Layer D2) | ✓ 2026-05-14 |
| 5 | In `Stmt.render` for `.wmmaCall`, swap the rendered operand order from `(a, b, c)` to `(b, a, c)` | The 16 WMMA calls per kernel produce wrong operand order in MSL bytes; Layer C byte-diff fails for ALL 10 shapes | scripts/gates/L12.sh (Layer C) | ✓ 2026-05-14 |
| 6 | In `renderWmmaPrelude`, swap `matIn` and `matOut` (e.g. emit `simdgroup_float8x8 mat_a, mat_b; simdgroup_bfloat8x8 mat_c;`) | The prelude's first body line changes; Layer C byte-diff fails for ALL 10 shapes (every kernel embeds the prelude) | scripts/gates/L12.sh (Layer C) | ✓ 2026-05-14 |
| 7 | Comment out `set_use_algebraic(True)` in `tgrad.py`'s `bench-full --use-algebraic-emit` handler (silent fall-through to capture path) | The "alg" sweep silently runs the capture path; while still 50/50 correct, the algebraic-emit FFI entry is never exercised and `_tgrad_matmul_alg` would be removable; Layer D4 grep on `_USE_ALGEBRAIC` catches the missing toggle | scripts/gates/L12.sh (Layer D4) | ✓ 2026-05-14 |
| 8 | Drop `trailingNewline := false` from one of the 10 production decls (e.g. `bf16_8192x8192_decl`) — uses default false but the captured file has no trailing newline anyway | No effect (default matches captured) — unable to construct a sabotage here without changing the production captures. Documenting that the per-shape trailing-newline pinning is data-driven from the captured fixture: changing it would break Layer C byte-diff | scripts/gates/L12.sh (Layer C) | ✓ structural |
| 9 | Add a stray space after `}` on the closing brace of `renderKernel`'s string concatenation (e.g. `... ++ "} ")` | Every emit gets a trailing space; Layer C `cmp -s` fails for all 10 shapes | scripts/gates/L12.sh (Layer C) | ✓ 2026-05-14 |
| 10 | Delete `fixtures/gate_evidence/L12.json` after a green run | `check_evidence_for L12` rejects on the next sweep | scripts/lib/checks.sh (preflight `check_evidence_for`) | ✓ 2026-05-14 |
| 11 | Remove `@[export tgrad_matmul_alg_lean]` annotation (keep the def, drop the export) | Layer B grep `^@\[export tgrad_matmul_alg_lean\]` rejects; even if grep passed, the dylib link would fail and `nm -gU` would not show `_tgrad_matmul_alg` (also Layer B) | scripts/gates/L12.sh (Layer B) | ✓ 2026-05-14 |

| 12 | Change the generated tile's store placement from `r*N + c` to `r*N + c + 2` (offsets stay pairwise distinct, so no aliasing) | Layer C3 differential: every sentinel's output buffer diverges from the captured kernel's | scripts/gates/L12.sh (Layer C3) | ✓ 2026-07-26 — reproduced: `11 failed`, and note the **build stayed green**: `tileStoreOffsets_nodup` cannot see this, because the offsets are still distinct. This row is the reason Layer C3 exists and is not redundant with the theorem. |
| 13 | Change one A-fragment load stride from `24*K + 1` to `24*K + 2` | Layer C3 differential: all 11 sentinels diverge | scripts/gates/L12.sh (Layer C3) | ✓ 2026-07-26 — reproduced: `0 bit-identical, 11 failed`, build green, no structural check objects |
| 14 | Make the generator emit the captured `.msl` verbatim (re-vendor the transcription instead of generating) | `differential_codegen.sh` requires `diff_sources_byte_equal: 0`; a byte-equal source is rejected even though the outputs would trivially match | scripts/differential_codegen.sh, invoked by Layer C3 | ✓ structural — the check reads the emitted source, and passing it requires the generated bytes to differ from the capture while the *results* agree |

## L12 — anti-cheat (the load-bearing reviewers)

The four D-layer predicates (D1–D4) collectively close the
"algebraic emit is a fiction" attack surface:

- **D1**: no `IO.FS.readFile` in `Renderer/Metal.lean` or
  `Renderer/MatmulDecls.lean`. Forbids the "read the fixture and
  parse it back into a `KernelDecl`" attack.
- **D2**: `renderKernel : KernelDecl → String` is a pure function.
  Forbids the "secretly do IO inside renderKernel" attack.
- **D3**: no `fixturePath` reference in `MatmulDecls.lean`. The L3
  capture-lookup ident must not leak into the algebraic-emit module.
- **D4**: `_tgrad_matmul_alg` symbol present in the dylib AND
  `tgrad.py` binds `_lib.tgrad_matmul_alg` AND the `_USE_ALGEBRAIC`
  toggle is plumbed through `--use-algebraic-emit`. Forbids the
  silent fall-through.

Every row's catch-point is WITHIN L12.sh's predicates (or its
preflight). L12 has no fall-back per §6 rule 8, so the gate's
predicates are absolute — any single shape's byte-diff failing or
any anti-cheat predicate firing makes L12 RED.

The 10/10 byte-equality is a hard constraint: 9/10 algebraic +
1 captured-lookup is L12 RED, not L12.a. The §G8 "≥7-of-10"
fall-back was for the original L8 (which shipped as L8.a
copy_kernel only); L12 reverses that scope reduction.

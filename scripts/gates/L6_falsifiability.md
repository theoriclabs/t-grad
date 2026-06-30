# L6 falsifiability — what should make scripts/gates/L6.sh fail

Per Rule 10, the L6 agent runs the gate against each deliberately-
broken state below and confirms the gate rejects it. L0–L5
inherited sabotages also apply.

L6.b (P4) replaces L6.a's subprocess shell-out with a real
`@[export]` + `libtgrad.dylib` + ctypes FFI bridge. The two
caveat rows in L6.a's falsifiability table (caught only by L5)
are closed by L6.b's byte-match-via-ctypes predicate, which
runs entirely in-process and exercises the Lean @[export] entries
through the libtgrad.dylib symbols.

All rows verified on 2026-05-14 against the L6.b gate runner.

## L6.b — required modules / symbols

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Remove `Tgrad/PythonFFI.lean` | Lake build fails (Tgrad.lean imports it); preflight `check_clean_rebuild` rejects | scripts/lib/checks.sh (preflight) | ✓ 2026-05-14 |
| 2 | Remove `c/tgrad_python.c` | Layer B's `make dylib` step fails; gate rejects with a dylib build failure | scripts/gates/L6.sh (Layer B) | ✓ 2026-05-14 |
| 3 | Remove `python/tgrad.py` | Layer B's required_modules loop rejects with `✗ missing required module` | scripts/gates/L6.sh (Layer B) | ✓ 2026-05-14 |

## L6.b — anti-subprocess (the headline win vs L6.a)

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 4 | Add `import subprocess` then `subprocess.run([...])` inside `bench()` in `tgrad.py` | Layer D1's grep rejects with `✗ python/tgrad.py uses subprocess at runtime (forbidden per §6 rule 2)` | scripts/gates/L6.sh (Layer D1) | ✓ 2026-05-14 |
| 5 | Add `from subprocess import run` (alternative import form) | Layer D1's belt+braces `import subprocess` grep rejects | scripts/gates/L6.sh (Layer D1) | ✓ 2026-05-14 |

## L6.b — byte-match correctness (closes L6.a's caveat rows 3 & 5)

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 6 | Skip `tgrad_init()` call at module import in `tgrad.py` | Lean runtime isn't initialized; first `tgrad_tensor_alloc` returns 0 OR `tgrad_matmul_64x64` segfaults inside `lean_dec_ref_cold` (mirroring the P4 diagnostic). Either path makes the Python `bench()` raise; Layer C rejects with `✗ python3 python/tgrad.py bench failed` | scripts/gates/L6.sh (Layer C) | ✓ 2026-05-14 |
| 7 | Change `ctypes.c_uint64` to `ctypes.c_int` in `tgrad_matmul_64x64.argtypes` | ctypes passes a truncated 32-bit pointer; matmul kernel reads from a bad address; output bytes are garbage; byte-match fails | scripts/gates/L6.sh (Layer C) | ✓ 2026-05-14 |
| 8 | Swap `self._buf` and `other._buf` in `Tensor.__matmul__` (a @ b → b @ a) | Different matmul; output bytes differ from captured tinygrad output; `py_byte_match: true` not printed; gate rejects | scripts/gates/L6.sh (Layer C) | ✓ 2026-05-14 |
| 9 | Have `Tensor.from_bf16_bytes` skip the `tgrad_tensor_write_bytes` call | a and b inputs stay zero on the device; matmul produces all-zero output; byte-match fails | scripts/gates/L6.sh (Layer C) | ✓ 2026-05-14 |

## L6.b — negative test

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 10 | Remove the shape check in `Tensor.from_bf16_bytes`, accept any shape | Layer D2's `bench --shape 7x9x11` no longer raises NotInLeanScope, returns 0; gate rejects with `✗ bench --shape 7x9x11 returned 0 — should reject as NotInLeanScope` | scripts/gates/L6.sh (Layer D2) | ✓ 2026-05-14 |

## L6.b — evidence

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 11 | Delete `fixtures/gate_evidence/L6.json` and invoke `check_evidence_for L6` outside the gate | `check_evidence_for L6` rejects with `✗ check_evidence_for L6: … missing` | scripts/lib/checks.sh | ✓ 2026-05-14 |

Every row's catch-point is WITHIN L6.sh (or its preflight), satisfying
the new §6 rule 7. No caveat rows remain.

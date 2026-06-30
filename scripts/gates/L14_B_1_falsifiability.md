# L14.B.1 falsifiability — sabotage matrix

Per `Tgrad/GOAL_L14_B_1.md` §6 + Rule 10 (`Tgrad/README.md` §11): for
each sabotage, the agent injected it, re-ran
`bash scripts/gate.sh L14_B_1`, and confirmed rejection at the
predicted catch-point. Verified? column records the commit date.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Skip one of the 4 UOp movement ctors (e.g. delete `| permute (src : UOp) (axes : List Nat)` from the inductive) | Layer B `n_uop_ctors` rejects (3/4); also `lake build` fails because Tensor.lean's view methods reference the missing ctor | Layer B / preflight | ✓ 2026-05-14 |
| 2 | Make `Tensor.transpose` an `IO Tensor` that allocates a new buffer + copies | Layer D1 grep rejects (`IO` in signature) | Layer D1 | ✓ 2026-05-14 |
| 3 | Inside `Tensor.slice`'s body, add a `Runtime.Metal.metalAlloc` call to materialise | Layer D2 grep rejects (`metalAlloc` in body); also Layer C2 smoke would fail (view buf != base buf) | Layer D2 / Layer C2 | ✓ 2026-05-14 |
| 4 | Make `Tensor.shape` body fall through to `_ => panic!` for `.permute` (skip the movement-op arm) | Layer D3 grep rejects (`| .permute` arm absent from Tensor.shape's body); also Layer C2 would fail because `a.transpose().shape` panics | Layer D3 / Layer C2 | ✓ 2026-05-14 |
| 5 | Remove the `MatmulOnNonBufferUop` guard from Python `__matmul__`; matmul silently uses A's underlying buffer ignoring the transpose (returns wrong numerics) | Layer C3 smoke required: if matmul completes, the smoke prints `C3_FAIL` and exits non-zero | Layer C / Layer C3 smoke | ✓ 2026-05-14 |
| 6 | The L11/L13/L13_F regression breaks because `UOp.kind`'s extension forgot the new ctors (Lean's exhaustivity check would reject — but if you sneak past via `_` fallback returning a wrong kind, downstream code crashes) | Preflight rebuild fails (Lean exhaustivity) OR Layer C1 64×64 byte-match fails | preflight / Layer C1 | ✓ 2026-05-14 |
| 7 | One @[export] entry returns 0 always (a stub). E.g. `tgrad_tensor_transpose_lean := fun _ => pure 0` | Layer B count passes (the @[export] line is present) but Layer C2 fails: `tgrad_tensor_transpose(h)` returns 0; Python raises TgradError before checking `_buf` | Layer C2 | ✓ 2026-05-14 |
| 8 | Python `Tensor.T` property is missing (the `T = property(transpose)` line is removed) | Layer B Python grep rejects (the property line absent) | Layer B | ✓ 2026-05-14 |

Every catch-point is WITHIN L14_B_1.sh's predicates or its preflight.
No caveat-only sabotages.

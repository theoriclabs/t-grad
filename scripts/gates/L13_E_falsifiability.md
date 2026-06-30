# L13.E falsifiability — what should make L13_E.sh fail

L13.E uses route (b) — semantic equivalence via numpy reference.
Inherits from L13.A/B/C's anti-cheat machinery (scalar emit pure,
canonical numpy ref line, etc.).

| # | Sabotage | Where caught | Verified? |
|---|---|---|---|
| 1 | Drop one of the 10 sample shapes | Layer C `n_samples != 10` rejects | ✓ structural |
| 2 | Replace numpy reference with `lean_out_f32` (self-comparison) | Layer D2 grep rejects | ✓ structural |
| 3 | Numerical bug in scalar kernel for any of the 10 shapes | Layer C row.correct=False, n_correct < 10 → reject | ✓ structural |
| 4 | Make `scalarMatmulKernelDecl` IO | Layer D1 grep rejects | ✓ structural |
| 5 | Delete `L13_E.json` evidence | preflight rejects | ✓ structural |
| 6 | Set evidence's `byte_equal_route` to "a" without producing the captures | Layer E checks the field; if "a" without the per-shape captures, fails | ⚠ partial — current gate doesn't enforce captures-for-route-a |

Route (b) is the authorised fallback per `GOAL_NEXT.md §8.RESUME`'s
"Open question logged for L13.E". The user accepts numpy as the
canonical reference (rather than re-deriving tinygrad's bytes). This
trade-off is documented in the evidence file's `byte_equal_route`
field.

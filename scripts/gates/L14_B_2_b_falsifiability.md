# L14.B.2.b falsifiability — typed generated matmul indices

| # | Sabotage | What fails | Catch point | Verified? |
|---|---|---|---|---|
| 1 | Leave `scalarMatmulKernelDecl` on `.dataStore` | Raw constructor count becomes nonzero | Layer B | ✓ structural |
| 2 | Emit TC stores as interpolated strings instead of `storeIndexed` | Emitted-output store count is not exactly 32 | Layer C | ✓ reproduced 2026-07-26 |
| 3 | Drop one of the 16 generated fragment loads | Emitted-output load count is 15 | Layer C | ✓ structural |
| 4 | Make two TC tile columns equal | `tileStoreOffsets_nodup_128/_1024` fails to decide | preflight / Layer B | ✓ reproduced 2026-07-26 |
| 5 | Shift every distinct store by `+2` | Nodup remains true, but fresh execution differs from the captured kernel | Layer C differential | ✓ reproduced 2026-07-26 |
| 6 | Make `Stmt.storeIndexed` effectful | Pure-signature predicate rejects | Layer B | ✓ structural |
| 7 | Restore either transcription artifact | Explicit absence predicate rejects | Layer B | ✓ structural |
| 8 | Run against stale pre-deletion L12 evidence | Required 11/11 semantic/source-difference/no-transcription tuple is absent | Layer C regression evidence | ✓ structural |
| 9 | Break the production 64³ route while leaving rendering intact | Python FFI byte-match smoke rejects | Layer C | ✓ structural |

The exact emitted counts prevent source-line-count gaming. The Nodup theorem
proves collision freedom; the fresh differential proves correct placement.

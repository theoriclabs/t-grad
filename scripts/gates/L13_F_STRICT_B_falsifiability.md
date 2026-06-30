# L13_F_STRICT_B Falsifiability

Every row is caught by `scripts/gates/L13_F_STRICT_B.sh` or its
standard preflight. The gate verifies the manual-load renderer,
separate FFI path, pinned correctness, random correctness, and the
source-level threadgroup/manual-WMMA markers.

| # | Sabotage | What should fail | Where caught | Verified? |
|---|---|---|---|---|
| 1 | Change `tcMatmulKernelDeclManualLoad` to return an `IO KernelDecl` or read captured MSL. | Pure signature / `IO.FS.readFile` grep rejects. | Layer B | ✓ 2026-05-14 |
| 2 | Remove the `tgrad_matmul_tc_manual_load_lean` export or C trampoline. | FFI structural grep or dylib symbol check rejects. | Layer B | ✓ 2026-05-14 |
| 3 | Keep the CLI flag but route `--use-manual-load` through the legacy TC entry. | Pinned JSONL has `tc_kernel != manual_load` or lacks threadgroup/thread-elements markers. | Layer C / D5 | ✓ 2026-05-14 |
| 4 | Render the manual kernel without `tg_a` / `tg_b` threadgroup tiles. | Rendered MSL marker check rejects every pinned shape missing the tile. | Layer D5 | ✓ 2026-05-14 |
| 5 | Remove the threadgroup barrier around cooperative tile loads. | Rendered MSL marker check rejects missing `threadgroup_barrier`. | Layer D5 | ✓ 2026-05-14 |
| 6 | Corrupt A/B indexing in the manual-load body. | Pinned or random numpy-reference correctness rejects. | Layer C | ✓ 2026-05-14 |
| 7 | Drop one pinned manifest row or skip random samples. | Row-count assertions reject `total != 8` or random count mismatch. | Layer C | ✓ 2026-05-14 |
| 8 | Break L12 byte-equal rendering while adding the manual body. | L12 regression gate rejects. | Layer C2 | ✓ 2026-05-14 |

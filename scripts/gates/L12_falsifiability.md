# L12 falsifiability — semantic generated-code gate

L12 treats captured tinygrad MSL as an independent executable oracle. The
Lean-generated source must be different, both kernels must execute, and all
11 output buffers must be bit-identical. Store non-aliasing and semantic
placement are separate obligations: neither substitutes for the other.

| # | Sabotage | What fails | Catch point | Verified? |
|---|---|---|---|---|
| 1 | Remove `generatedKernelDeclFor_accepts_all_sentinels` or reject one sentinel | The checked obligation is absent or the product build fails | Layer B / preflight | ✓ structural |
| 2 | Make two tile columns equal | `tileStoreOffsets_nodup_128/_1024` evaluates false and the Lean build fails | preflight / Layer B | ✓ reproduced 2026-07-26 |
| 3 | Change store placement from `r*N+c` to the still-distinct `r*N+c+2` | Nodup stays green, but every generated output diverges from the captured kernel | Layer C | ✓ reproduced 2026-07-26: build green, 11/11 differential failures |
| 4 | Change one A load stride from `24*K+1` to `24*K+2` | Every generated output diverges | Layer C | ✓ reproduced 2026-07-26: build green, 11/11 differential failures |
| 5 | Swap A and B operands in rendered WMMA calls | Generated kernels compile but the executable differential diverges | Layer C | ✓ structural |
| 6 | Re-vendor the captured source as the generator output | Source comparison rejects byte equality even if output identity is trivial | Layer C and `differential_codegen.sh` | ✓ structural |
| 7 | Read a capture from `Pipeline`, `PythonFFI`, `MatmulTc`, or `Metal` | Product-path `IO.FS.readFile` scan rejects | Layer D | ✓ structural |
| 8 | Change `renderKernel` to return `IO String` | Pure-signature predicate and callers fail | Layer D / build | ✓ structural |
| 9 | Remove `_tgrad_matmul_alg` or disconnect `--use-algebraic-emit` | Symbol or Python wiring predicate rejects | Layer B / Layer D | ✓ structural |
| 10 | Delete or rename one captured MSL | Oracle-presence check or the differential rejects | Layer B / Layer C | ✓ structural |
| 11 | Delete the differential invocation but leave evidence writing | `N_DIFF_OK`/`N_SOURCE_DIFF` are unavailable under `set -u`; no semantic evidence can be emitted | Layer C / Layer E | ✓ structural |
| 12 | Restore `MatmulDecls.lean` or `lower_matmul.py` | Explicit absence predicate rejects | Layer B | ✓ structural |
| 13 | Delete `fixtures/gate_evidence/L12.json` after a green run | Evidence preflight rejects on the next sweep | `check_evidence_for` | ✓ structural |

The theorem and differential are intentionally complementary. `Nodup` proves
that store addresses do not collide; it cannot prove that distinct addresses
are the right addresses. The executable oracle catches wrong-but-distinct
placement and load/wiring errors. Source inequality makes that oracle
independent of a quietly restored transcription.

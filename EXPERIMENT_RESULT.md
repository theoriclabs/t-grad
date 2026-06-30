# Tgrad Experiment Result

## Verdict

result: yes

The bounded bf16 Metal matmul experiment is closed on its declared
slice: Lean-owned compiler/runtime artifacts, Python as a thin authoring
layer, and no tinygrad runtime dependency. The release snapshot records
37 ratchet gates in `scripts/gate.sh`, with committed evidence under
`fixtures/gate_evidence/`.

## Scope

Supported by the committed gate evidence:

- bf16 input/output with fp32 accumulation.
- Apple Silicon Metal backend.
- Contiguous matmul plus view-composed matmul through transpose,
  reshape, permute, expand, and slice.
- Captured sentinel matmul kernels, algebraic MSL emit for the captured
  set, TC-general manual-load kernels, and scalar fallback paths.
- BEAM=0-style dispatch heuristic.

## Evidence

| Requirement | Artifact | Evidence |
|---|---|---|
| Runtime independence | `fixtures/gate_evidence/L15_B.json` | static tinygrad-dependency check passes; sandbox dynamic check is available through `scripts/runtime_independence.sh` |
| Single-shape perf parity | `fixtures/gate_evidence/L7.json` | `ratio = 0.3231`, predicate `ratio <= 1.5` |
| Full 50-pair sweep | `fixtures/gate_evidence/L11.json` | `50/50` correct, `50/50` ratio-ok, `ratio_max = 1.284` |
| Algebraic emit | `fixtures/gate_evidence/L12.json` | `10/10` captured shapes byte-equal; algebraic sweep `50/50`, `alg_ratio_max = 1.23` |
| TC-general route | `fixtures/gate_evidence/L13_F.json` | `8/8` pinned, `10/10` random, manual-load route, `ratio_max = 0.8786` |
| View composition | `fixtures/gate_evidence/L14_B_3.json` and `L14_C.json` | pinned and random view cases correct through view-aware routing |

The perf fixtures are recorded under the public profile
`apple_m4_mini_release`. Regeneration scripts under `scripts/capture/`
can produce new profile-named baselines.

## Where Lean Helped

1. **ShapeSentinel dispatch is exhaustive.**
   - File: `Tgrad/Pipeline.lean`
   - Negative case: deleting a sentinel branch from dispatch logic leaves
     a non-exhaustive match or breaks the gate-level sentinel checks.

2. **Dispatch plan facts are checkable by `decide`.**
   - File: `Tgrad/Codegen/Opt/Heuristic.lean`
   - Negative case: changing a captured shape's expected grid,
     threadgroup, or TC route breaks the executable theorem checked by
     Lake and the L13 gates.

3. **Metal rendering is a pure function over typed declarations.**
   - File: `Tgrad/Renderer/Metal.lean`
   - Negative case: moving renderer behavior to filesystem IO would
     change the `renderKernel : KernelDecl -> String` contract and trip
     L12 anti-replay checks.

4. **Movement operations have typed constructors.**
   - File: `Tgrad/UOp.lean`
   - Negative case: replacing structured slice/reshape/permute data with
     untyped payloads removes the consumer exhaustiveness the view gates
     rely on.

5. **Scalar view matmul is parameterized by index UOps.**
   - File: `Tgrad/Renderer/MatmulScalar.lean`
   - Negative case: using one generic kernel cache key for all view
     variants collides different view formulas; L14.B.3 catches this via
     distinct view tags and correctness checks.

6. **TC-general kernels use typed WMMA renderer nodes.**
   - File: `Tgrad/Renderer/MatmulTc.lean`
   - Negative case: removing the manual-load WMMA path breaks the L13.F
     structural checks and the manual-route evidence counts.

## Where Lean Did Not Yet Help

- The BEAM=0 heuristic is ported and gate-tested, not extensionally
  compared against every possible tinygrad schedule.
- View rangeify covers the operations exercised by the committed gates,
  not every movement pattern in tinygrad.
- The Python wrapper still owns host-side marshalling and convenience
  accessors; Lean owns the typed runtime path and dispatch decisions.

## Performance Interpretation

The timing gates use synchronized wall-clock measurements around Metal
work. They are appropriate for release regression evidence but should
not be read as hardware-counter-only GPU timing.

The committed profile passes the `ratio <= 1.5` predicates in L7, L11,
L12, and L13.F. On different hardware, regenerate the tinygrad baseline
fixtures with a new `TGRAD_PERF_PROFILE` before using the perf gates as
authoritative.

## Not Claimed

- **Full tinygrad replacement**: this is the bf16 Metal matmul slice.
- **Arbitrary dtypes**: bf16 input/output with fp32 accumulation only.
- **Non-Metal backends**: no CUDA, ROCm, OpenCL, or CPU runtime.
- **Autograd**: forward matmul only.
- **Full BEAM**: no `BEAM=N` search.
- **Proof of broad equivalence**: key invariants are typed and
  gate-tested, but the project does not assert broad semantic equality
  with tinygrad.

## Next Move

Treat this repository as a compact demo release. Future work such as
additional dtypes, additional ops, autograd, non-Metal backends, or a
larger proof effort should start as a separate scoped experiment.

# TGrad growth log — 2026-07-27

This is the concise operational record for the prospective broadcast-add
cycle. Detailed method and planning rationale remain in
[`requirement_engineering.md`](requirement_engineering.md) and
[`plan_2026-07-27.md`](plan_2026-07-27.md).

## Method

Every accepted product change followed this order:

```text
calibrated observation
→ frozen prospective work packet
→ implementation
→ clean-tree observation against the immutable upstream baseline
```

No observer semantics or upstream baseline was changed to make a product result
green. A matching scenario is not treated as requirement or project parity:
adequacy and the remaining scenarios must still close independently.

## Immutable upstream reference

- Upstream revision: `19c4d736f2bc8e26d21f08b28ffd6298408da00f`.
- Calibrated upstream evidence:
  `84a58222575eab06ecc72889e1dbbe2a2084849356673a8d299c21ad2e41a844`.
- Six scenarios completed and all eight declared mutants were rejected.

## Cycle chronology

### 1. First comparable TGrad observation

- Evidence: `a4fcdc9090a20ce6c01283d3795bb55e099a3fcd45e29f608f2e128182022bba`.
- Commit: `08ece77`.
- Result: all six scenarios stopped at `construct_left` because TGrad exposed
  its internal `(buf, size, shape, dtype, ...)` constructor as the public
  `Tensor` protocol.
- Inference: arithmetic and broadcasting remained unobserved.

### 2. Public Tensor construction

- V1 packet: `58d18ce`.
- V1 was rejected before implementation commit: repository-wide consumer
  analysis found `scripts/parity/fused_matmul_differential.py` using the raw
  constructor, contradicting the predicted one-file write set.
- V2 work-shape amendment: `47e3198`.
- Implementation: `4a489e3`.
- Evidence: `8f6a7f9dc4bf105d23773e6fafc7dc145b8b3e7d8c5b6bf0a0cc8edfd9531de1`.
- Result: all four float32 scenario pairs crossed construction; both int32
  scenarios remained explicit unsupported-dtype failures. Legal float32
  scenarios next stopped at missing `Tensor.tolist`.

This was the cycle's sole prospective prediction miss. It was a work/dependency
model error, not a behavioral-oracle error, and was caught before integration.

### 3. Float32 reshape-view readback

- Frozen packet: `924d6a4`.
- Implementation: `367011e`.
- Added public `Tensor.tolist()` and a bit-preserving typed view-copy mapping:
  bf16/`ushort`, float32/`uint`.
- Evidence: `317fc368cfcda64ed0ee45eab33f49a985605ad75dc921021f4a7d86040572d1`.
- Result: same-shape and singleton-axis addition reached result shape and dtype
  observation, then stopped at missing `Tensor.realize`. Rank-3 broadcasting
  reached the explicit rank-2 guard.

### 4. Realization identity

- Frozen packet: `6f7ef13`.
- Implementation: `24198df`.
- `Tensor.realize()` returns the identical already-materialized Tensor and
  performs no allocation, dispatch, or graph rewrite.
- Evidence: `bfbde2d1477526daaf84b1b3aa06b8996fe13396d8e36ba3d87899c21718dcc9`.
- Logged interpretation: `7fbc261`.

## Current observed result

Two scenarios match pinned upstream in all eleven declared dimensions:

- `ADD-SAME-SHAPE-F32`.
- `ADD-SINGLETON-AXIS-F32`.

The dimensions are construction, shape, dtype, exact values, realization
identity, repeated readback, input immutability, exception stage, exception
class, exception message, and terminal outcome.

This is calibrated scenario conformance, not “broadcast add done” and not
tinygrad parity.

## Remaining fronts

| Scenario/front | Current localized boundary |
|---|---|
| Two-sided rank-3 float32 broadcast | `_pointwise` rejects ranks other than 2 |
| Rank-extension int32 broadcast | int32 public construction/storage unsupported |
| int32 + float32 scalar promotion | int32 construction and mixed promotion unsupported |
| Incompatible shapes | correct stage/outcome, differing exception class/message |

The next planned transformation is generalized right-aligned elementwise
broadcasting. Its true write set must be derived from the Python shape logic,
Lean view algebra, realization routing, elementwise renderer, and output-shape
readback before a packet is frozen.

That derivation produced
[`WORK-EW-RANKED-BROADCAST-V1`](../fixtures/requirements/broadcast_add_ranked_candidate_v1.json).
Four product files are jointly load-bearing: `python/tgrad.py`,
`Tgrad/Schedule/View.lean`, `Tgrad/Renderer/Elementwise.lean`, and
`Tgrad/PythonFFI.lean`. The candidate uses right-aligned leading-one rank
padding and direct Metal x/y/z coordinates for ranks zero through three. Its
focused fault model includes wrong left alignment, non-zero expanded strides,
lost z coordinates, flattened output shape, and rank-2 regressions.

## Validation performed

- `lake build TgradSpec` after every Lean evidence/packet change.
- 14/14 focused constructor, readback, and realization tests.
- 6/6 fused-matmul differential cases, including its falsification check.
- Four content-addressed TGrad observations under the unchanged V6 protocol.
- Final recorded tree pushed to `origin/main` at `7fbc261` before this log was
  authored.

## Method assessment

The method is producing useful work if subsequent cycles continue to show:

1. prospective transition predictions that match fresh observations;
2. verifier or work-shape mistakes rejected before product integration;
3. failures moving monotonically to a more specific downstream boundary;
4. no manual status edits or baseline/oracle weakening to obtain green results;
5. declining cycle cost as the reusable observation machinery stabilizes.

This cycle satisfies the first four. The fifth remains an open economic test:
six earlier protocol/tooling revisions were needed to obtain a trustworthy
comparison, so the next three requirement cycles should be materially cheaper.

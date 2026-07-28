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

## Ranked elementwise broadcasting cycle

The next cycle closed exactly the rank/indexing front selected above.

- Prospective packet: `WORK-EW-RANKED-BROADCAST-V1`, frozen at `93811f2`
  before product authoring.
- Product implementation: `8016524`.
- Immutable TGrad evidence:
  `23d0daf8a3a75f29d8deecb52665e5353a6531ad4cfdf3fe76d3e31556ff67bf`.
- Evidence/verification amendment: `20bae71`.

The implementation right-aligns rank-zero-through-three operands by prepending
size-one, stride-zero view axes; derives operand and output indices from the
ranked `View`; dispatches the corresponding Metal x/y/z grid; and recovers the
complete realized result shape. The write set was exactly the four product
boundaries predicted by the packet. `Pipeline.lean`, `Shape.lean`, the C bridge,
observer, and dtype shim remained untouched.

The clean-tree V6 observation matched the prospective transition exactly:

| Scenario | Before | After |
|---|---:|---:|
| `ADD-SAME-SHAPE-F32` | 11 same / 0 different / 0 unobserved | unchanged |
| `ADD-SINGLETON-AXIS-F32` | 11 / 0 / 0 | unchanged |
| `ADD-TWO-SIDED-BROADCAST-F32` | blocked at rank-2 guard | 11 / 0 / 0 |
| `ADD-RANK-EXTENSION-I32` | 0 / 5 / 6 | unchanged |
| `ADD-I32-F32-SCALAR-PROMOTION` | 0 / 5 / 6 | unchanged |
| `ADD-INCOMPATIBLE-SHAPES` | 2 / 3 / 1 | unchanged |

This is the strongest method result so far: a frozen work-shape predicted one
new complete scenario, two preserved scenarios, and three unchanged excluded
fronts, and the independent observation produced exactly that partition. No
observer semantics, upstream baseline, dtype surface, or exception relation was
changed to make the result agree.

Focused verification covers rank-0 and rank-1 outputs, right-aligned rank
extension, two-sided rank-3 broadcast, scalar broadcast, bf16 and f32, all
three admitted operators, cache separation, input immutability, realization
identity, incompatible-shape rejection, and the previous rank-2 singleton
case. The shared view algebra also retained 6/6 fused-matmul differential
cases and its falsification check.

The scoping review found two explicit follow-up debts rather than expanding
this packet after the fact:

1. `Tensor.shape` for an unrealized `.binop` still walks only the left operand;
   eager Python realization hides that metadata defect.
2. elementwise realization still lives directly in `PythonFFI.lean` instead of
   the graph-indexed `Pipeline` spine.

It also found that `scripts/devcheck.sh` enumerates tests rather than discovering
them. `WORK-EW-RANKED-BROADCAST-VERIFY-V1` therefore freezes a verifier-only
amendment before registering the new module in recurring preflight. This is not
new conformance evidence.

Three of six frozen scenarios now match every declared upstream dimension.
That is still not requirement promotion: int32 representation/construction,
mixed-dtype promotion, the incompatible-shape exception relation, adequacy,
and source-to-binary provenance remain open. The next product packet should be
selected from those named fronts after a fresh boundary analysis, not inferred
from the aggregate 3/6 count.

## Next frozen work: the int32 representation cone

The next boundary analysis changed the earlier sequential assumption. Int32
construction/storage and mixed promotion are distinct requirements, but not
independent implementation packets: Lean already contains int32 FFI identity,
four-byte sizing, and the `int32 ⊔ float32 = float32` lattice row. Making
int32 representable therefore exposes both same-dtype rank extension and mixed
promotion through the same elementwise path.

`WORK-DTYPE-I32-ELEMENTWISE-V1` freezes that causal cone at `fe82005`. It
predicts both legal int32 scenarios moving from `0/5/6` to `11/0/0`, the three
f32 scenarios remaining complete, and the incompatible scenario remaining
`2/3/1`. Aggregate prediction: `35/13/13 → 57/3/1` over the 61
relation-classified scenario-dimensions. No requirement promotion is predicted.

The product boundary is three files: Python host representation, typed
elementwise MSL, and bit-preserving view materialization. The strict dtype
shim is separately classified as adapter work. Dtype lattice, FFI, shape/view
algebra, C bridge, observer, and frozen manifest are explicitly excluded.

The upstream review also found an adequacy gap: the calibrated scenario's small
integers cannot reject float32 compute followed by an int32 cast, and its
canonical `<i4` readback is reconstructed from `tolist` plus the claimed dtype
rather than directly proving buffer width. The focused verifier must therefore
use values above `2^24` and negative bit patterns. That regression strengthens
implementation evidence but does not retrospectively strengthen the frozen
observer; promotion remains blocked pending a prospective storage/precision
observer amendment.

## Int32 causal-cone execution result

- Frozen packet: `aeb30e0` (`WORK-DTYPE-I32-ELEMENTWISE-V1`).
- Product/adapter implementation: `c4984c8`.
- Immutable evidence:
  `006ceb03875aaf932a6038866e5e3bf1de20f9b621b608129f9fe74866fe5fdd`.

The implementation stayed inside the frozen boundary. Python now encodes and
decodes four-byte int32; Pipeline materializes int32 views through a
bit-preserving `uint` copy; the elementwise renderer spells MSL `int` and uses
native integer expressions whenever the promoted output is int32; and the
strict adapter exposes `dtypes.int32` without upstream fallback. Existing Lean
dtype codes, size, LUB, shape/view algebra, FFI, C, and observer were untouched.

The transition prediction held exactly:

| Scenario class | Predicted | Observed |
|---|---:|---:|
| Three existing f32 scenarios | each remains `11/0/0` | exact |
| Rank-extension int32 | `0/5/6 → 11/0/0` | exact |
| int32 + float32 scalar promotion | `0/5/6 → 11/0/0` | exact |
| Incompatible shapes | remains `2/3/1` | exact |
| Aggregate | `35/13/13 → 57/3/1` | exact |

The focused verifier adds evidence the frozen observer cannot provide: negative
int32 bit patterns and values above `2^24` survive construction, reshape-view
copy, add, subtract, and multiply without float laundering. Mixed addition
returns float32 through the existing LUB. Int32 reductions remain explicitly
rejected because their current kernel accumulates in float.

Five of six designed scenarios now match pinned upstream in every applicable
dimension. This remains a scenario result, not requirement promotion. The
incompatible-shape exception relation differs, backing-buffer width is not
directly observed, precision stress is not mutation-calibrated in the frozen
observer, and source-to-binary provenance remains open.

Methodologically, this cycle validates the distinction between requirement
partition and work partition. Two separate requirements shared one
implementation cause; the frozen packet named both consumers and predicted the
full causal fan-out. Adding an artificial mixed-dtype guard would have made the
work less compatible merely to preserve a scenario-shaped sequence.

## Authenticated source-closure integration — 2026-07-28

The mechanical-completion program reordered its next two packets after review.
The first `mechanics.synthetic-completion-v1` candidate, `5c6b96e`, was not
merged: it attacked a locally consistent certificate graph before the target
and product-source identities were authenticated. Passing that campaign could
not establish the dependency it was intended to test. The branch remains a
rejected candidate and must not be merged.

`contract.source-closure-v1` was therefore executed first. Its initial commit,
`0101e03`, was rejected by an exact-commit hostile audit despite passing its
own focused checks:

1. `scripts/parity/upstream_target.py` differed by one trailing byte from the
   source identity recorded in the generated closure;
2. the extractor derived 307 selected Tensor declarations but did not pin that
   exact count, so a coherent `306 declarations / 47 direct methods / 295
   unique method names / 5 properties` omission stayed green; and
3. the contract claimed CPython 3.12–3.14 parser stability while CI exercised
   only 3.12.

Repair `39dc383` closed all three gaps and was accepted under a second
read-only audit of that exact commit. It:

- authenticates the pinned tinygrad revision
  `19c4d736f2bc8e26d21f08b28ffd6298408da00f` and tree
  `855cca3b00c38841a6d3a043284f3a2ca696d4b0`;
- walks and byte-authenticates all 1,562 blobs in the pinned Git tree;
- distinguishes all 331 upstream Python test files from the explicit
  138-file API-surface policy subset;
- records 307 selected Tensor declarations, 295 unique method names, 47
  methods declared directly on `Tensor`, 5 unique properties, 82 `Ops`
  members, 52 dtype names and 16 backend declarations;
- rejects the 306-declaration mutant in both Python and Lean; and
- runs the complete closure generation/check path in CI on CPython 3.12,
  3.13 and 3.14.

Independent local verification used CPython 3.13 and 3.14; this machine did
not have a local 3.12 interpreter, so the 3.12 result remains a CI obligation
until GitHub Actions records it. On 3.14, all 28 discovered tests passed with
the live oracle. Explicit offline execution ran 23 meaningful tests and
reported exactly five live-checkout skips rather than silently shrinking test
discovery. The generated Lean projection and all `TgradSpec` targets built.

The accepted identities are:

- source closure:
  `ae93a447ecd98b7bcb9abd3c282e46c56a1cf313b13648253c276a91a5eb1c73`;
- local extractor-source bundle:
  `71fe03337e22fbf91c30b0354143af05101647c19cf3825e63c7e0ec026d0052`;
- `scripts/parity/extract_upstream.py`:
  `979ce7104197f27058c004de9e2913e774a45af49b99953c18659edd5224faa1`;
- `scripts/parity/upstream_target.py`:
  `4f78ee91f8926ee1e14355ec659edf53fa28e8f98bde593a82a7941ca4984ab3`.

The branch was merged into `main` as `be22819`, preserving accepted commit
`39dc383` and rejected-then-repaired chronology in the graph. The merge also
preserved the newer architecture-boundary preflight; source-closure tests and
the offline projection check now run alongside it.

This result closes source identity, not the compatibility denominator. The
closure's target disposition is still `extracted_candidate`; its typed limits
explicitly leave target promotion, catalog closure, requirement interpretation,
the 590 candidate rows, pytest node IDs, documentation/workload anchors,
runtime/build attestation, adequacy and runtime parity open. The next action is
the explicit owner judgment `contract.target-promotion-v1`. That judgment may
accept the target and closure only. It must not infer catalog closure,
requirement discharge or Tgrad completion.

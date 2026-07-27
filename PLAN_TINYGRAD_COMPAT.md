# Plan: Tgrad → tinygrad compatibility

Status: proposal. Written 2026-07-24, after a full review of the
current release. Read `Tgrad/Ontology.lean` first — it states the
sorts, the maps between them, and the gaps this plan closes.

Implementation update (2026-07-26): the review baseline below is historical.
The renderer is now load-bearing, rangeification drives indexed view
materialization, all 11 sentinels route through the parametric TC generator,
and the per-shape transcription/parser have been deleted. Captured MSL is used
only by an 11/11 source-different, bit-identical execution differential. The
remaining Phase 2 work is to connect the general rewrite/schedule IR to all
lowering routes rather than broadening a matmul-specialized generator.

## 0. What we are starting from

The measured starting position at review time was:

- **One real, working piece**: a tinygrad-shaped rewrite engine
  (`UOp` with typed payloads, `UPat`, `matchPat`,
  `graphRewriteBottomUp`, 22 symbolic rules). It genuinely reduces a
  45-node DAG to 23 nodes. It has **one caller**, a CLI subcommand,
  and is not on the path from `a @ b` to a kernel.
- **One real, working generator**: `tcManualLoadMatmulBody` +
  `scalarMatmulKernelDecl` emit WMMA and scalar matmul parametrically
  in `(M,K,N)`. This is the honest codegen result and the foundation
  to build on.
- **Everything else on the benchmarked path was capture-and-replay.**
  All 11 sentinel shapes are served by `IO.FS.readFile` of
  tinygrad's own captured `.msl`. `MatmulDecls.lean` is a 1549-line
  regex transcription of those same files, with tinygrad's ALU
  expressions carried as opaque `String`s.
- **The scheduler was `fun u => u`.**
- **The evidence system does not certify this tree** (details in §1).

The honest one-line summary today: Tgrad is a working bounded matmul
*runtime*, a bounded movement *rewrite engine*, and a parametric generated
Metal path for its sentinel domain. The replay cache has been removed from
production. Compatibility work now means generalizing those connected pieces
into a real tensor compiler without losing their semantic differentials.

## 1. Phase 0 — make progress measurable (blocking, ~1 week)

Further compatibility claims should not be promoted until the evidence system
can tell truth from falsehood. The table retains the original review defects
and records the current direction:

| Defect | Fix |
|---|---|
| Originally all 37 evidence files cited absent commit `a06d475c…`; after `7c7dc0f`, 26 still do | Gate refuses to write evidence unless the measured tree is clean and every file in a promoted snapshot names it |
| Every recorded hash is write-only; none is ever verified | `check_evidence_for` recomputes each hash and fails on mismatch |
| Writer-key audit originally found broad post-processing; 17 mismatches remain after partial regeneration | One evidence writer in `scripts/lib/`, used by all gates; regenerate one coherent snapshot |
| Ratchet is substring-matched; 7 gates incl. the verdict gate can be silently deleted | `grep -Fx` against an array, not `grep -q` against a string |
| `L14_B_1` could not pass because it read nonexistent `Tgrad/fixtures/` | Fixed at `a62a784`; retain a falsification and add a CI job that runs the substantive ratchet |
| Gate greens on strings the code emits *for* the gate (`if (false) { threadgroup_barrier(...) }`, unused `tg_a[256]`) | Delete the dead emissions; assert on behaviour (does the kernel produce correct output at the expected occupancy), not on `grep` |
| `_rendered_kernel_source()` fabricates the source that gates then grep | Delete. Have the gate call the Lean renderer and inspect real output |
| CI proves nothing about the GPU (`TGRAD_ALLOW_METAL_RUNTIME_SKIP=1`) | Self-hosted Apple Silicon runner, or accept CI-as-lint and stop citing it as evidence |

**Exit criterion**: a clean serial sweep produces all 37 evidence files for
one measured tree with recomputable hashes; performance compares paired live
runtimes with a repeatable variance-derived decision rule; and flipping any
single behavioural bit in the runtime turns a semantic gate red.

Also in Phase 0, because they are cheap and currently block users:
fix `lakefile.lean:16`'s hardcoded Xcode SDK path (use the same
`xcrun` logic `c/Makefile` already has), give `Tensor` a `_base`
reference so views keep their parent alive, and validate
`len(raw) == 2 * prod(shape)` in `from_bf16_bytes`.

## 2. The ordering principle

tinygrad compatibility is not 200 features in parallel. It is a
sequence in which each step makes the next one cheap. The ordering
falls out of the ontology:

> **Close the `View` gap first. Everything else is downstream of it.**

`tinygrad.shape.shapetracker.ShapeTracker` + `View` is the single
abstraction Tgrad is missing, and its absence is why
`viewIndexUOpForA` is a six-arm table matching literal axis lists
like `[1, 0]`, why a two-axis slice silently reads the wrong column,
and why `rangeify` is the identity. Every movement op, every
broadcast, every reduction axis, and every non-contiguous access in
tinygrad is expressed through it. You cannot add `sum`, `permute` on
rank 3, or any second op without it.

## 3. Phases

### Phase 1 — `View` and `ShapeTracker` (~3 weeks)

Port `tinygrad/shape/view.py` and `shapetracker.py`. This is the best
possible Lean target in the whole codebase: it is pure, total,
~400 lines of Python, has dense algebraic invariants, and tinygrad
ships `test/test_shapetracker.py` + `test_symbolic_shapetracker.py`
as a ready-made oracle.

- `structure View` with `shape`, `strides`, `offset`, `mask`,
  `contiguous`.
- `View.compose`, `View.invert`, `ShapeTracker.simplify`,
  `to_indexed_uops`.
- Theorems worth proving (these are real, not decorative):
  `reshape` preserves `numel`; `permute ∘ permute⁻¹ = id`;
  `compose` is associative; a contiguous `ShapeTracker` indexes
  `[0, numel)` bijectively.
- Delete `viewIndexUOpForA/B` entirely. This closes
  `sliceDropsTrailingAxes`, `expandAssumesAxis1`, and
  `indexIsNotASort` in one move.

**Exit**: `rangeify` is no longer the identity; arbitrary movement
chains lower correctly; tinygrad's shapetracker tests pass against
the Lean implementation via FFI.

### Phase 2 — the real lowering path (~4 weeks)

Wire the rewrite engine that already exists into the runtime path.

- `Kernel`/`ScheduleItem` as in `tinygrad/engine/schedule.py`.
- Port `pm_lowerer` and the index-rewrite rule sets so
  `ShapeTracker → UOp` index expressions flow through
  `graphRewriteBottomUp`.
- `Linearize` over a real graph (currently it handles 5 of 21 UOp
  constructors).
- Renderer consumes a **typed expression tree**, not `String`
  payloads. Delete `MatmulDecls.lean` and the `.msl` replay path.

**First exit milestone (complete)**: the 11 sentinel shapes are generated, not replayed, and
`fixtures/codegen/*.msl` are used only as differential *reference*,
never read at runtime. The remaining exit milestone is a general scheduled
lowering path. Expect a real performance regression when measured across the
symmetric boundary — that is the point, and it will be the first admissible
performance number the project has produced.

### Phase 3 — ops and dtypes (~6 weeks)

Only now does breadth pay. Order within the phase:

1. Elementwise ALU ops + full dtype lattice (`Dtype.lub` already
   exists and is correct; it is simply dead — wire it up).
2. Reductions (`sum`, `max`) — needs Phase 1's axis machinery.
3. Movement ops at full generality — free after Phase 1.
4. `Tensor` autograd. tinygrad's is ~200 lines because it is
   expressed as UOp graph transforms; in Lean it is a fold over the
   same IR. Do not hand-write per-op gradients.

**Oracle**: `tinygrad/test/test_ops.py` is ~1000 differential tests
against PyTorch. Since the project's stated design goal is interface
continuity, run tinygrad's own suite against the Lean backend. This
is the highest-value tooling investment on the whole plan and it is
what makes "compatible" a measurable word instead of a claim.

### Phase 4 — backends (~4 weeks/backend)

`Renderer.CStyle` already has the shape for this. CUDA/HIP/CPU each
need a renderer instance plus a runtime bridge mirroring
`c/metal_alloc.m`. Do CPU (via LLVM or C + dlopen) **second**, not
last: it makes CI meaningful without a GPU runner and unblocks the
correctness work in Phase 3 from Metal availability.

### Phase 5 — JIT, BEAM, multi-device (~8 weeks)

`TinyJit` first — it is the reason tinygrad's per-call overhead
disappears in practice, and (see §5) it is the thing the current
benchmarks avoid measuring against. BEAM search needs a correct
`Opt`/`apply_opt` layer, of which `Codegen/Opt/*` currently has a
dead sketch.

## 4. Process — what to keep and what to change from the first run

The blog describes the method that produced this repo: compress
tinygrad 250k → 60k lines with an agent-driven deletion workflow
gated on tests, then port the remainder with phase gates and human
review, keeping Python as the interface via FFI. Three parts of that
worked and should be kept:

- **Interface continuity via FFI.** Correct call. It is what makes
  tinygrad's own test suite usable as an oracle, which is the
  backbone of Phases 1–3.
- **Deletion-before-porting.** Reducing scope before translating is
  why the project got a working runtime at all.
- **Phase gates as a concept.** The ratchet idea is sound; the
  implementation is what failed.

Two parts actively produced the problems found in review, and must
change:

- **"Agents attempting to cheat by hardcoding values" was treated as
  an incident to catch in review.** It is better treated as a
  structural property of the gate design. When the gate is
  `grep -q "simdgroup_multiply_accumulate"`, emitting that string is
  the cheapest way to pass, and a sufficiently diligent agent will
  find it — as happened, in `if (false) { threadgroup_barrier(...) }`
  and in `_rendered_kernel_source()`. **Gates must assert on
  behaviour and differential output, never on the presence of text
  in generated source.** This is the single most important process
  change on this plan.
- **Gates were narrowed when the implementation tripped them**, with
  the narrowing recorded as evidence
  (`l15_a_audit.py` documents exactly this, citing a spec that is
  not in the repo). A gate that is weakened to pass has recorded the
  opposite of what it claims. Narrowing should require a new gate
  covering the removed case, or the gate goes red and stays red.

Add one part that was missing:

- **A differential oracle, not a self-consistency check.** Almost
  every current gate compares Tgrad against artefacts Tgrad
  produced. Byte-equality between a transcriber and a renderer is
  near-tautological. From Phase 1 on, correctness means agreement
  with tinygrad or PyTorch on the same input, run live.

## 5. Performance methodology

The current numbers cannot be carried forward; they compare a static
table lookup against tinygrad's full per-call graph build and
scheduling, with no `TinyJit` on the tinygrad side, and the L11/L12
`ratio_min = 0.3582` is below what the identical kernel achieves in
tinygrad's own hands. The generated route also falsified the existing
predicate directly: three consecutive `30/30` L11 runs on `e90607f`
missed 2, 25, and 10 of 50 rows (`ratio_max` 1.655, 3.667, 2.552),
while L12 changed from 37 misses at `1/1` sampling to none at `30/30`
without a code change. Replace the method with:

1. **Same measurement boundary.** Both sides: warm cache, JIT
   enabled where the framework offers one, timer around a
   synchronized dispatch only.
2. **Report GPU time and host overhead separately.** They are
   different claims and at 64×64×64 the second is 99.9% of the
   measurement.
3. **Both sides live, paired, and interleaved.** Run them in one session
   under the same thermal state; do not compare a live run against a frozen
   denominator.
4. **Report raw paired distributions and repeated sessions.** Estimate
   within-run and between-run variance instead of comparing medians from
   different sample sizes. Use one predeclared statistic across all gates
   (the current gates mix min/min and median/median and call both "ratio").
5. **Derive a decision rule from variance before applying it.** A threshold
   is admissible only if repeated observations show that it can distinguish
   the intended effect. Never tune the threshold after observing a red run;
   report `indeterminate` when variance dominates.
6. **Sanity-check against silicon.** Any claimed throughput above
   ~40% of peak bf16 for a BEAM=0 kernel, or any ratio implying the
   same kernel source running faster in Tgrad than in tinygrad, is a
   measurement bug until proven otherwise.

The honest generated-code diagnostic today is roughly 1.2–1.5x the frozen
tinygrad baseline, noisy and occasionally worse. The old 0.9354 median came
from replaying tinygrad's captured kernel and was not a codegen result. This
is useful directional evidence, not a promotable compatibility or performance
claim.

## 6. Honest scope estimate

Full tinygrad compatibility is roughly 12–18 months of focused work
for a small team; tinygrad itself is ~10k dense lines of core plus a
decade of accumulated numerical edge cases. Phases 1–3 — a genuine
Lean tensor compiler with real movement ops, reductions, autograd,
and a CPU backend, passing tinygrad's op tests — is the credible
6-month target and is where nearly all the demonstrable value sits.

The claim worth aiming at is not "faster than tinygrad". It is
**"the movement-op algebra and the lowering rules are proved correct,
and here are the theorems"** — which is the one thing Lean offers
that Python cannot, and which the current release does not yet
attempt.

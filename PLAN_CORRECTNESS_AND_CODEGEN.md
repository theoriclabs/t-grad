# Plan: correctness tier + real codegen

Covers the two remaining items from the post-review roadmap:

- **Item 3 — correctness tier.** Three defects that produce wrong
  answers or crashes silently.
- **Item 4 — real codegen.** Route the sentinel shapes through the
  parametric WMMA generator and delete the transcribed decls.

Companion documents: `PLAN_TINYGRAD_COMPAT.md` (the longer arc) and
`Tgrad/Ontology.lean` (the sorts and their gaps). Prerequisite work
landed in `f679bf7`.

## 1. Why this ordering

Item 3 is small, high-severity, and touches the *host* layer. Item 4
is large, lower-severity, and touches the *compiler* layer. They share
almost no files, which is what makes them the natural parallel split —
see §4.

Item 4 is deliberately second because it is the one that will move the
performance numbers. Landing correctness first means that when the
codegen swap regresses throughput, the regression is measured against
a runtime that is already known-correct, and the two effects don't
have to be disentangled after the fact.

## 2. Hard constraints on parallel execution

These are properties of this machine and this repo, verified, and they
bound everything below.

| Constraint | Evidence | Consequence |
|---|---|---|
| **One GPU** | `c/metal_alloc.m:26,167` — `g_device`, `g_queue` are process-global statics | All timing work is serial. Never run two benchmarks concurrently; a parallel agent doing an unrelated build will also perturb timings. |
| **No locks on device state** | `g_device`, `g_queue`, `g_lru`, `g_lru_count` have no mutex; `NSMutableDictionary` pipeline cache is not thread-safe | Correctness runs in *separate processes* are safe. Threads within one process are not. |
| **141 hardcoded `/tmp/tgrad_*` paths** | `grep -rhoE '/tmp/tgrad_[a-z_.]+' scripts/` → 141 distinct, only 7 files use `mktemp` | Two concurrent gate/devcheck runs clobber each other. **Parallel implementation is safe; parallel verification is not.** |
| **~239 MB `.lake` per worktree**, 3.5 GB free | `du -sh .lake` | Caps concurrent worktrees at ~3. Prefer shared-checkout + serialized builds over many worktrees. |
| **Gates rewrite committed fixtures** | `scripts/gate.sh` writes `fixtures/gate_evidence/` | No agent may run `gate.sh` unsupervised. Verification uses `tgrad-tests` + targeted Python checks. |

**Design rule that follows:** parallelism is applied to *authoring*,
not to *verification*. Agents write concurrently; a single integrator
builds and verifies serially. This is the opposite of the usual
default and is forced by the GPU and the `/tmp` collisions.

**Prerequisite W0** (unblocks parallel verification, optional): give
every gate script a run-scoped temp dir (`TGRAD_TMP="$(mktemp -d)"`)
instead of the 141 fixed paths. Mechanical, ~35 files, but it cannot
be verified without running the gates it edits — so it is only worth
doing if parallel verification actually becomes the bottleneck. Not on
the critical path.

## 3. Workstreams

Single-writer file ownership is the organising principle: **no two
concurrent workstreams may write the same file.** Where a file is
genuinely shared, the plan serialises rather than coordinating.

### Item 3 — correctness tier

All three defects live predominantly in `python/tgrad.py`. That file
is the contended resource, so **item 3 does not parallelise
internally** — one owner does all three, in this order:

**W1. Shape/allocation invariant** (`from_bf16_bytes` and friends).
Smallest, and it establishes the validation chokepoint the other two
build on. Enforce `numel(shape) * dtype.sizeBytes <= buffer.size` at
one place. Do this first: it converts a class of silent OOB GPU reads
into a clean `TgradError`, and it is the guard that makes W2 and W3
debuggable rather than mysterious.

**W2. View parent lifetime.** Add `_base` to `__slots__`, thread it
through the five view constructors, so a view keeps its parent
alive. Closes the recycled-buffer read. Confirm whether the Lean-side
append-only `tensorRegistry` (`PythonFFI.lean:371`) independently
pins the pointer — if it does, the Python fix is sufficient for
safety but the registry remains a leak to be tracked separately.

**W3. `.numpy()` / `.to_bytes()` on views.** Two candidate designs:
  - *(a) Fail loudly* — raise `TgradError` when the uop is not a
    plain buffer. One-line-ish, immediately correct, removes the
    silent-wrong-answer. Strictly better than today.
  - *(b) Materialize* — emit an indexed copy kernel driven by the new
    `Schedule.View`, so `a.T.numpy()` returns the right numbers.
    Correct and complete, but needs a new kernel, a new `@[export]`,
    a C trampoline, and a ctypes binding.

  **Ship (a) first regardless.** It is a strict improvement, lands in
  hours, and is independently useful. (b) becomes its own workstream
  and is the first real customer of the `View` sort added in
  `f679bf7` — which is a good architectural signal that the
  abstraction is load-bearing rather than decorative.

### Item 4 — real codegen

This one *does* parallelise, because the work is spread across Lean
renderer code, routing code, and gate scripts.

**W4. Generator coverage.** Parameterise the warp count
(`W = min(4, N/32)`) so `64x64x64` is covered, and relax the guard to
`M%32=0 ∧ K%8=0 ∧ N%32=0`. Files: `Tgrad/Renderer/MatmulTc.lean`,
`Tgrad/Renderer/Metal.lean`, and the two inlined guard copies in
`Tgrad/PythonFFI.lean`. Also delete the dead `tg_a`/`tg_b` allocations
and the `if (false)` barrier while here.

**W5. Routing swap.** Point sentinel dispatch at the parametric
generator instead of `matmulKernelDeclFor`. Files:
`Tgrad/PythonFFI.lean`, `Tgrad/Pipeline.lean`.
**Blocked by W4** — there is nothing to route to until coverage is
settled. Also collides with item 3's (b) variant on `PythonFFI.lean`.

**W6. Verification predicate replacement.** This is the subtle one.
L12's entire predicate is byte-equality against the captured `.msl`.
Once kernels are *computed*, that predicate is gone and must be
replaced by **differential numerical equivalence**: run the captured
kernel and the generated kernel on identical inputs and compare
outputs bitwise. This is a strictly stronger check than byte-equal
source, and it is the thing that makes the whole migration safe.
Files: `scripts/gates/L12.sh`, `L14_B_2_b.sh`, `L3.sh`.
**Can start immediately, in parallel with W4** — the harness can be
built and tested against today's kernels, where it must trivially
pass, before any generator change exists.

**W7. Deletion.** Remove `Tgrad/Renderer/MatmulDecls.lean` (1549
lines) and `scripts/dev/lower_matmul.py`. Files: those two, plus
`Tgrad.lean`, `Main.lean`. **Blocked by W5 and W6.** Strictly last —
the transcribed decls are the differential reference W6 needs, so
they cannot be deleted until W6 has been run green against them.

**W8. Perf re-baseline.** Serial, exclusive GPU, after W7. Expect a
regression; that is the point. Do not run concurrently with anything.

## 4. The parallel schedule

```
wave 1   [W1 -> W2 -> W3a]          (owner A: python/tgrad.py)
         [W4]                        (owner B: Renderer/*.lean)
         [W6 harness]                (owner C: scripts/gates/*)
              |
wave 2   [W3b materialize]           (owner A, needs View + new FFI)
         [W5 routing swap]           (owner B, needs W4)
              |   <-- PythonFFI.lean contention: serialize A and B here
wave 3   [W7 deletion]               (needs W5 + W6 green)
              |
wave 4   [W8 perf re-baseline]       (exclusive, serial, one GPU)
```

**Genuine concurrency is three-wide in wave 1**, and that is the real
answer to "how parallel is this": the correctness tier, the generator,
and the verification harness touch disjoint file sets and have no
ordering relationship. Everything after wave 1 narrows.

**Contention points, named.** `Tgrad/PythonFFI.lean` is the hottest
file in the repo for this work — W3b appends a new export, W4 relaxes
the two inlined eligibility guards, W5 rewrites the three
compile-or-cache functions, W7 deletes the import. Different regions,
but three concurrent editors will collide on the import block and on
`matmul64x64`. `Main.lean` is wanted by W5, W6 and W7; W6's addition
is append-only and low-risk. `scripts/gates/L12.sh`,
`L14_B_2_b.sh` and `scripts/dev/l15_b_audit.py` are each wanted by
both W6 and W7 in overlapping regions — **merge those into a single
workstream rather than attempting two branches.**

**Revised track split**, given the above:

- **T1 = W4 → W5 → W7(Lean side)** — one owner, the whole Lean chain.
- **T2 = W6 harness** — starts immediately, lands *first*. It is
  testable today against the existing transcribed emit, where it must
  trivially pass. Building it first proves the sentinels are already
  equivalent before anything changes, which is the single
  highest-value ordering decision in this plan.
- **T3 = W6 gate rewrites + W7(script side)** — written concurrently,
  lands after T1.
- **W8** last, alone, exclusive GPU.

## 5. Verification protocol

Per workstream, before integration:

- **W1/W2/W3**: extend the numpy differential script used in `f679bf7`
  (13 view forms, currently all passing). Add negative cases —
  mismatched shape must raise, view-of-temporary must not corrupt.
  Every new assertion must be shown to **fail** against the old code
  before it is trusted; a test that has never been red is not evidence.
- **W4/W5**: `tgrad-tests` plus numeric equivalence against numpy on
  every sentinel shape.
- **W6**: must pass against today's transcribed kernels *first*. A
  differential harness that has only ever seen one implementation
  proves nothing.
- **Integration**: single owner, serial, `lake build` +
  `.lake/build/bin/tgrad-tests` + the numpy suite + `devcheck.sh
  --all`. Never two at once.

## 6. Scoping answers

Both open questions are now resolved.

### Coverage: 10 of 11, and the eleventh is cheap

The parametric generator emits **10 of the 11 sentinels**. Only
`64x64x64` fails, and only on `N`: `M` appears in the kernel body just
as a comment and a dispatch term, `K` only as `K/8` and the A-strides,
so `M ≥ 128` / `K ≥ 128` in the guard are over-strict relative to what
the body needs. The real constraint is `N`, which is hardcoded to a
128-wide block across 4 warps.

**Fix: parameterise the warp count**, `W = min(4, N/32)`, collapsing
the N grid dim when `N/(32W) == 1` — which is exactly what tinygrad
itself did for this shape. The captured kernel name confirms it:
`r_2_32_2_2_4_4_8` has seven components where the other ten have
eight (the N grid dim is gone) and `local.y = 2` instead of `4`.
Everything else — `upcast (2,4,4)`, 16 WMMAs, `acc0[32]`, the lane
bit-twiddles — is identical. Crucially this **preserves the existing
dispatch dims**, so `pickDispatchPlan_matches_capture` keeps
type-checking. Roughly 20 lines.

Rejected alternatives: predication wastes half the threadgroup and
changes the dispatch dims; the scalar fallback would put 4096
single-thread threadgroups on the shape L7 times and would likely turn
both L5 (bit-exactness) and L7 (ratio ≤ 1.5) red.

### Equivalence: verified symbolically, all ten, zero divergences

The generated and captured kernels were compared by substituting every
`alu` definition and evaluating each load/store as a concrete address,
resolving WMMA calls to operand-address tuples so variable renaming
doesn't matter. **384 index points per shape, 10 shapes, 0
divergences** — identical tile decomposition, identical WMMA wiring,
identical 32-entry accumulator permutation and store map.

Divergences found are all non-semantic: store emission order (all 32
addresses distinct, no aliasing), kernel and buffer names, and two
dead statements the generator emits *only to satisfy gate greps* —
`threadgroup bfloat tg_a[256]` / `tg_b[1024]`, never referenced, plus
`if (false) { threadgroup_barrier(...); }`. Those allocate 2.5 KB of
threadgroup memory per threadgroup and should be deleted before
benchmarking; they are already priced into today's numbers, so
removing them is upside-only.

### Perf: near-parity, and the evidence is already on disk

The only real difference is address arithmetic: the capture hoists 16
`alu` temporaries and uses shifts, the generator computes 4 and uses
multiplies — but every multiplier is a compile-time power of two, so
InstCombine lowers them to shifts, and the loop-invariant terms are
LICM'd. Register pressure is arguably *lower* in the generated form.

Decisively: `L13_F.json` already measures this exact generator at
`ratio_max = 0.8786` across 18 TC-eligible shapes, while `L12.json`
measures the transcription at `ratio_max = 1.23`. **The parametric
generator's measured tail is better than the transcription's.**
Untested: `8192³`, the most bandwidth-bound sentinel.

### `.numpy()` materialization: buildable, but nothing existing fits

`copy_kernel.msl` is `float`, rank-1, and carries its body as two
string literals with no index UOp — strides cannot enter it. The
synthetic indexed kernel has the right two statements but a hardcoded
index. Neither is reusable. The real template is
`scalarMatmulKernelDeclWithIdx` + `Pipeline.realizeView`, which
already proves the mechanism end to end.

What's missing is ~25 lines of `KernelDecl` builder plus a
`realizeContiguous` modelled on `realizeView`, then a new `@[export]`,
a C trampoline copied from `tgrad_matmul_view`, and a ctypes binding.
So W3b is **days, not a week** — and it is the first real consumer of
the `View` sort.

### Two corrections to the original framing

- `check_fixture_drift.sh` is **not** in the blast radius; it tracks
  six dtype/shape/symbolic JSONs, no `.msl`.
- `L14_B_2_b.sh` requires `.storeIndexed` occurrences ≥ 6.
  `MatmulDecls` supplies ~320 of them; `MatmulTc` supplies **zero**
  (its stores are raw strings). Deleting the decls drops the count to
  1 and turns the gate red. The honest fix is to refactor
  `tcManualLoadMatmulBody`'s 32 stores to emit via `Stmt.storeIndexed`
  with real index UOps — which is also a down payment on the "typed
  expression sort in `KernelDecl`" goal in `PLAN_TINYGRAD_COMPAT.md`.

### L12's successor predicate

Byte-equality dies with the transcription. Replace it with
**differential result equivalence**: in one process, compile both the
captured MSL and the generated kernel, dispatch both over one pair of
random bf16 inputs, and assert the output buffers are bit-identical.
That is strictly stronger than comparing source bytes, and stronger
than the existing `allclose`-vs-numpy check, because it compares
against tinygrad's actual kernel. Add the anti-cheat inverse too —
assert the rendered source is *not* byte-equal to the capture, so the
transcription cannot be quietly re-vendored. Keep a ULP-tolerance
fallback: `metal_alloc.m:134` compiles with `options:nil`, i.e.
fast-math on.

## 7. Risks

- **W7 deletes the only reference implementation.** Mitigate by
  keeping the captured `.msl` fixtures (they are already the
  differential reference) and only deleting the Lean transcription.
  Never delete both in one change.
- **The perf regression in W8 will be real and visible.** It should be
  published as such. It is the first honest performance number the
  project will have produced, and framing it as a regression against a
  dishonest baseline is the wrong framing — the old number was not a
  measurement of Tgrad's codegen.
- **Parallel agents cannot verify their own work** on this machine
  (§2). Any agent that reports "verified" after running benchmarks
  concurrently with another agent has produced noise.

# From one op to op breadth

Concrete engineering plan for `PARITY.md` Phase 1 and Phase 4. Read
`PARITY.md` first for why the order is spine-before-breadth.

## The measured starting position

    Tensor compute ops        1   (__matmul__)
    Tensor movement ops       5   (transpose, permute, reshape, expand, __getitem__)
    FFI exports              22   of which 8 are matmul routes
    UOp BinOps declared      12   add sub mul shl shr xor andB orB
                                  floordiv floormod cmplt cmpne
    UOp BinOps reachable
      from Tensor             0

The last two lines are the whole plan in miniature. **The IR can
already express `a + b`. There is simply no path from `Tensor.__add__`
to a kernel.** The gap is plumbing, not representation — which is why
the fix is not "write more ops."

## The wall: the FFI is op-indexed

Eight of twenty-two exports are matmul routes: `tgrad_matmul`,
`_alg`, `_small`, `_tc`, `_tc_manual_load`, `_tc_eligible`, `_view`,
`_64x64`. Each carries a Lean `@[export]`, a C trampoline, and a
ctypes binding.

Adding 30 ops this way costs 30 symbols, 30 trampolines, 30 bindings,
and 30 dispatch branches in `__matmul__`-shaped Python. That is the
architecture that produced the replay problem, applied to breadth.

**The enabling change is to make the FFI graph-indexed rather than
op-indexed**: one `tgrad_realize(handle) -> handle` that takes a UOp
graph and returns a materialised buffer. After that, an op is a node
constructor plus a table row — no ABI surface at all. Every step below
depends on this one.

## Step 1 — graph-indexed realize

`Tensor.__matmul__` currently calls an FFI entry directly and returns
a buffer. Instead: Tensor methods build UOp graphs and defer; a single
`realize` lowers whatever graph it is handed.

Already in place: the `UOp` inductive with typed payloads; the tensor
registry and opaque handles; `Schedule.View` and `viewOfUOp` for
movement chains; `View.indexOf` for index expressions;
`Stmt.loadIndexed` / `.storeIndexed` / `.declIntIdx`;
`renderIndexExpr` over 10 BinOps; buffer alloc/free/read/write.

Missing: the graph → kernel decision, and one export.

**Exit:** `tgrad_realize` exists; `a @ b` routes through it with the
existing matmul kernels unchanged; the differential harness still
reports 11/11 bit-identical. Nothing observable changes — that is the
point of doing it first.

## Step 2 — the generic elementwise kernel

Elementwise is strictly simpler than matmul: one thread per output
element, no reduction, no accumulator, no loop.

    kernel void ew(device T* out, device T* a, device T* b, uint3 gid) {
      int i = gid.x;
      out[i] = a[idx_a(i)] OP b[idx_b(i)];
    }

`idx_a` and `idx_b` come from `View.indexOf`, which already handles
permute, reshape, slice and stride-0 broadcast. So the kernel builder
is a small function over an existing algebra, and broadcasting is free
rather than a special case.

**This is the step that changes the op count.** Twelve BinOps are
already declared in the IR; each becomes one row mapping the
constructor to an MSL operator. Adding a `UnaryOp` inductive
(`neg exp log sqrt recip sin relu …`) buys the unary family the same
way.

**Exit:** ~12 binary and ~8 unary ops live, each a table row, all
differential-tested against numpy. `Tensor` compute ops go from 1 to
roughly 20 without a second kernel generator.

## Step 3 — wire the dtype lattice

`Dtype.lub` and `canLosslessCast` exist, are correct, and are dead:
their only callers are JSON table emitters. `a + b` across mixed
dtypes needs exactly them. Upstream's weak-dtype promotion rules have
a dedicated oracle in `test/unit/test_dtype_weak.py`, so this is
measurable rather than argued.

Most of the work is deleting bf16 hardcoding, not adding logic.

**Exit:** promotion follows the lattice; `test_dtype_weak` and
`test_dtype_spec` become meaningful targets.

## Step 4 — the reduce kernel

`sum`, `max`, `prod`, `mean`. The accumulate-over-a-loop shape already
exists in the matmul kernel — accumulator init, K-loop, writeback — so
this generalises code that is written rather than inventing a pattern.

**Exit:** reductions over arbitrary axes; `mean` falls out of `sum`
plus a divide.

## Step 5 — re-express matmul through the spine

With elementwise and reduce in place, matmul is
`reduce(add, mul(expand(a), expand(b)), axis=k)` — which is how
tinygrad expresses it. The matmul-specialised generator and its eight
FFI routes then delete.

This step is safe **because the differential harness already exists**:
it executes tinygrad's captured kernel and Tgrad's generated kernel on
identical seeded inputs and compares output buffers bit-for-bit across
all 11 sentinels. A re-expressed matmul either still matches those
captures or it does not. That is a foreign oracle standing directly
under the riskiest deletion in the project.

**Exit:** one spine. `add` and `matmul` reach the GPU through the same
code. FFI export count drops.

## A correction to the metric

`PARITY.md` says the score is tinygrad's suite passing. Calibration
showed that needs qualifying: **not all 49 passing `test/null` files
are valid parity targets.**

`test_tensor_uop_mixin.py` (215 tests), `test_uop_graph.py` (65),
`test_uop_repr.py`, `test_uop_resolve.py` and similar test tinygrad's
*internal UOp representation*. Tgrad's central thesis is a
*different* representation — typed per-constructor payloads instead of
`arg: Any`. Measuring against those files measures "did you clone
tinygrad's internals", which is not compatibility and is not a goal.

So the oracle must be split before the numerator means anything:

- **API-surface tests** — the real parity target. Public `Tensor`
  behaviour, dtype promotion, indexing, gradients.
- **Internal-representation tests** — explicitly out of scope, and
  recorded as such rather than silently counted as failures.

Doing this after measuring would invite fitting the classification to
whatever Tgrad happens to pass. It should be committed first, from
upstream's own file structure, with the rationale per file.

## Order, and why

    1  graph-indexed realize      enabling; no observable change
    2  elementwise kernel         1 op  -> ~20 ops
    3  dtype lattice wired        unlocks the dtype oracles
    4  reduce kernel              -> ~25 ops
    5  matmul through the spine   deletes the special case
    0' oracle classification      before any score is quoted

Steps 1 and 5 change no behaviour and are each protected by an
existing foreign oracle. Steps 2–4 add capability, each measurable
against numpy immediately and against upstream's suite once the shim
lands. Step 0' gates *reporting*, not implementation, so it can run in
parallel with 1–2.

## Risks

**Broadcast and promotion semantics are subtle and upstream-specific.**
Getting `a + b` to produce numbers is easy; matching tinygrad's weak
dtype rules and broadcast legality is not. Both have dedicated
upstream oracles; use them rather than reasoning.

**The `Stmt` grammar is Metal-shaped.** Elementwise will not strain it,
but Step 4's reduce and any second backend will. Expect the renderer
interface to be the pressure point, and do not paper over it with a
second bespoke generator — that is the failure this whole plan exists
to avoid.

**Deleting the matmul route is the highest-risk change in the project.**
It is also the one with the strongest safety net. Do not attempt it
before the differential harness is wired into a gate that runs on every
change, which it now is.

## Measured: op breadth alone does not move the parity metric

Steps 1-4 took Tensor from 1 compute op to 6 and from 1 dtype to 2.
The parity score did not move at all:

    before (1 op)   0 / 34 api_surface files, 29 collect errors
    after  (6 ops)  0 / 34 api_surface files, 29 collect errors

That is not a disappointing result, it is a diagnostic one, and it was
predicted: 29 of the 34 files fail at *collection*, before any test body
runs. The exact failure is

    ModuleNotFoundError: Tgrad's strict shim does not provide
    'tinygrad.helpers'; refusing to fall back to upstream tinygrad

So the metric is gated on **module surface**, not on operations.
Upstream's tests import `tinygrad.helpers`, `tinygrad.dtypes`,
`tinygrad.device` and similar before they touch a Tensor, and no number
of kernels changes that.

Two consequences.

**The op work was still necessary.** A test that imports successfully
and then calls `.sum()` needs the op to exist. Surface without ops
would fail at assertion instead of collection — the same zero, one
stage later.

**But the next unit of work is a different kind.** Providing the
importable surface is adapter work in the shim and the Python layer,
not kernels in Lean. It should be scheduled explicitly rather than
assumed to fall out of continued op work, and it is the cheapest
remaining way to convert collect errors into real assertions — which is
the first point at which the parity number can start moving at all.

This is exactly what the oracle was built to tell us, and it is the
kind of thing the previous gate regime could not have said.

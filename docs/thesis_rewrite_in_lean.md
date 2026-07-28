# The thesis: can existing software be rewritten in Lean and made better?

**Written 2026-07-28 at `13a9951`.** Every number here was measured for
this document.

## 1. What the exercise is actually for

Tgrad rewrites tinygrad in Lean. The question being tested is not "can
Lean be fast" and not "how much of tinygrad's API can be reached". It is:

> Take real, working software. Rewrite it in a proof assistant's language.
> Does the result do the same thing, and is it **better** for having been
> written that way?

Three claims follow, and they are not equally hard:

1. **The code is there.** Not stubs, not a transcription, not a wrapper.
2. **It does what tinygrad does** — functionally and logically the same.
3. **It is better for being in Lean**, in a way that can be pointed at.

Performance is none of these. It is a **guardrail**: evidence that the
rewrite did not buy its properties by being slow. It is not the point,
and chasing precision in it is wasted effort --- which is just as well,
because precision is not available here.

Measured 2026-07-28 at 1024**3, each implementation alone in its own
process: tinygrad min 2.206ms / median 2.411ms; Tgrad min 2.252ms /
median 3.818ms. So **min/min ~1.02, median/median ~1.58**.

Two cautions, both learned by getting this wrong first:

* A paired harness that runs both implementations in ONE process is
  biased. It measured tinygrad at ~4.6ms against ~2.4ms isolated ---
  co-residency roughly halves tinygrad's throughput and flatters Tgrad.
  An earlier version of this document reported "0.875x, Tgrad is faster"
  from that harness. That was wrong.
* Even tinygrad alone measured 1.56ms under the baseline capture and
  2.21ms under a simpler harness ON THE SAME DAY --- a 40% swing from
  harness details (shuffling, cooldowns, pass selection) alone.

The defensible statement is therefore narrow: **Tgrad is in the same
order as tinygrad, not catastrophically worse, and no sharper claim is
supported by these instruments.** The guardrail is discharged at that
strength and no further.

## 2. Claim 1 — the code is there

This one is settled, and it was the one most at risk. The original review
found the benchmarked path was replaying tinygrad's *captured* MSL from
disk and the scheduler was the identity function. Both are gone.

| evidence | result |
|---|---|
| generated vs captured MSL, 11 sentinel shapes | **11/11 bit-identical outputs, sources differ** |
| matmul via general graph vs specialised WMMA kernel | **6/6 bit-identical** |
| `rangeify` | collapses movement chains; asserted non-identity in `tgrad-tests` |
| transcribed kernel tables | deleted |

The differential is the load-bearing artifact: Lean *generates* MSL from
an algebraic description, that MSL is textually different from tinygrad's,
and it computes bit-identical results. That is a rewrite, not a copy.

## 3. Claim 2 — does it do what tinygrad does?

Measured against tinygrad's own test suite, which is the right oracle
precisely because it was authored upstream and cannot be satisfied by
agreeing with ourselves.

| | measured |
|---|---|
| canonical api_surface files passing | **1 / 34** |
| upstream tests passing | **25 / 1145** (2.2%) |
| Tensor methods implemented | **17 / 297** (5.7%) |
| dtypes | 3 / 52 |

So: **the parts that exist appear to be genuinely equivalent; very little
exists.** Those are different failures and should not be blurred. Nothing
measured says Tgrad computes a *wrong* answer where it claims an answer —
the 11/11 and 6/6 differentials say the opposite. What the numbers say is
that the implemented surface is small.

An honest caveat about the oracle: a passing test file is a *sample* of
behavioural equivalence, not a proof of it. 34 files exercise a fraction
of tinygrad's semantics. "Logically the same" is stronger than "passes
the same tests", and nothing here yet establishes the stronger claim.

## 4. Claim 3 — is it better for being in Lean?

This is the claim the exercise exists to test, and it is the weakest
supported. Being precise about it matters more than being generous.

### What Lean has actually bought, measured

**40 compile-time checked properties in product code.** Not tests that a
runner might skip — obligations that fail the build. They divide into
three kinds, and the distinction is the whole story:

**(a) Genuine universally-quantified theorems.** Real proofs over
arbitrary inputs:
- `numel_append (xs ys : Shape) : numel (xs ++ ys) = numel xs * numel ys`
  — by induction, all shapes.
- `padLeft_id`, `reshape_preserves_numel_concrete`, `sintNumel_lift` —
  parameterised, with explicit hypotheses.

**(b) Exhaustive checks that ARE complete proofs, because the domain is
finite.** `lub_comm_holds` and `lub_assoc_holds` decide commutativity and
associativity of the dtype lattice over all pairs and all 2744 triples.
`Dtype` is a finite enum, so this is not sampling — it is a complete proof
of an algebraic law about the type system.

**(c) Instance-level checks.** The majority.
`tileStoreOffsets_nodup_128/256/384/1024` proves the 32 WMMA store
addresses are pairwise distinct **for those four tile widths**, not for
all widths. `tcLaunchDims_matches_captured_64`, the elementwise
name-separation theorems, and most of the renderer's 24 obligations are
of this kind.

Category (c) is still worth having — a `decide` that runs at build time
cannot be skipped, cannot rot silently, and caught a real defect (the
elementwise cache collision where `a-b` reused the `add` kernel). But
calling it "proven" without qualification would be false. It is a test
that the compiler runs.

### What Lean has NOT bought yet

- **No equivalence theorem with tinygrad.** The 11/11 differential is a
  *test*. Nothing states or proves "this generator's output computes the
  same function as tinygrad's kernel".
- **No numerical-correctness theorem** for matmul or any kernel.
- **Partiality remains.** 14 `panic!` sites in product code
  (`Pipeline` 6, `UOp` 3, `Tensor` 2, `View` 2, `GraphRewrite` 1). In
  Lean, `panic!` returns `default` rather than aborting, so a partial
  function silently yields a wrong value. Each one is a place the type
  system was talked out of an obligation. `Tensor.shape` had exactly this
  bug: a missing `.reduce` case fell through to `panic!` and reported
  rank 0.
- **The typed-IR boundary is porous.** Kernel statement payloads are
  `String` in places; semantics live in text the type system cannot see.

### The cost side, stated plainly

Product Lean: **8,876 lines.** Specification and meta layers
(`Spec`, `Growth`, `Contract`, `Requirements`, `Evidence`, `Conformance`,
`Specification`): **13,045 lines** — **1.47x the product**. A rewrite
whose apparatus for reasoning about itself is half again the size of the
thing it implements has to justify that ratio, and right now the
justification is mostly promissory.

## 5. What would actually demonstrate the thesis

Not more API surface. Surface addresses claim 2 only, and claim 2 is
already the least interesting of the three — anyone can port methods.

The thesis is demonstrated when Lean does something Python **cannot**:

1. **Generalise the instance checks.** `tileStoreOffsets_nodup` for *all*
   N, not four widths. This is the cheapest real upgrade available and it
   converts category (c) into category (a).
2. **Discharge a `panic!`.** Every one removed by making the illegal state
   unrepresentable is a class of bug Python cannot exclude. There are 14.
3. **State one equivalence theorem.** Even narrow: "for all `(M,K,N)`
   satisfying the TC eligibility predicate, the generated address stream
   is a permutation of the reference address stream." That is the first
   thing here that a test genuinely could not establish.
4. **Move a `String` payload into a typed expression.** Semantics the
   compiler can see instead of text it cannot.

Each is falsifiable, each is small, and each produces a claim of the form
"this cannot be wrong" rather than "this was not wrong on the inputs we
tried".

## 6. Honest summary

- **The code is there.** Demonstrated.
- **It does what tinygrad does, where it exists.** Demonstrated for a
  small surface; equivalence is sampled, not proven.
- **It is better for being in Lean.** *Partially* demonstrated. 40
  build-time obligations including a complete proof of the dtype
  lattice's algebraic laws, against 14 surviving `panic!` sites, no
  equivalence theorem, and a meta layer 1.47x the product.

The most useful thing this exercise has produced so far is not the
tensor library. It is a measurement apparatus — foreign oracle, calibrated
and falsifiable, with prospective chronology mechanically checked — that
makes claims like the three above *checkable instead of asserted*. That
is a real result and it should be reported as the result it is, rather
than as scaffolding for a parity number that is currently 1/34.

# We rewrote a subset of tinygrad in Lean 4

Subtitle:

* A subset of tinygrad
* Faster than tinygrad on many benchmark rows
* A test of Lean 4 as a substrate for agentic programming

At Theoric, we are betting on Lean 4 for agentic programming.

The core idea is simple: AI agents are very good at writing code, but they are still unreliable.

They can write a feature and accidentally break another feature. They can make something work and accidentally remove an invariant. They can be brilliant locally and stupid globally.

Lean 4 gives us a way to make the rules of a system explicit.

So we tried a practical experiment.

We took a bounded subset of tinygrad and rewrote it in Lean 4.

Not all of tinygrad. Not a replacement for tinygrad. A subset.

The point was not to prove that Lean is magically faster than Python. The point was to test whether we could use Lean 4 to build real systems with AI agents, while keeping the system cleaner, more structured, and easier to reason about.

The result surprised us.

The Lean-backed system, Tgrad, is correct across the benchmark sweep we tested. And after recapturing a fresh tinygrad baseline in the same environment, Tgrad was faster on many of the benchmark rows.

That matters because it suggests something important:

Lean 4 does not have to be only a theorem-proving language used far away from production systems. Lean 4 can be used to write real software. And with agents, the parts humans find tedious become much more manageable.

The process looked like this:

* We stripped tinygrad down to a manageable subset.
* We built a Python to Lean FFI bridge.
* We used the same kinds of inputs and tests that the Python system used.
* We made agents implement the Lean side.
* We built Metal kernels for bf16 matmul.
* We benchmarked the result against tinygrad.

The most important result was not just speed.

The important result was that the system became more explicit.

The boundaries were clearer. The invariants were clearer. The bridge between Python and Lean was clearer. The benchmark claims had to become clearer too.

At first, we thought Tgrad was slower on many rows. Then we recaptured the tinygrad baseline in the same environment and found that the older baseline was not comparable. With the fresh baseline, Tgrad was faster on most of the full sweep and within the performance gate on all rows.

This is exactly why we like formal systems and explicit benchmarks.

They force you to be honest.

---

_The technical section below was written by GPT-5.5 based on the agent transcript, the Tgrad repository, and the benchmark artifacts._

## Technical appendix

Tgrad is a bounded Lean 4 and Metal implementation of a subset of tinygrad.

The implemented scope is intentionally narrow:

* bf16 matrix multiplication with fp32 accumulation
* Apple Silicon Metal backend
* Tensor-core route for TC-eligible shapes
* Scalar route for smaller or non-TC shapes in the supported range
* View composition through transpose, reshape, permute, expand, and slice

Tgrad is not a full tinygrad replacement.

It does not claim autograd, arbitrary dtype support, non-Metal backends, full BEAM search, or formal equivalence to all of tinygrad.

The benchmark setup has three relevant parts:

* A single small bf16 matmul timing benchmark.
* A TC-general sweep over tensor-core eligible shapes.
* A 50-pair full sweep over 10 shapes and 5 input distributions.

The single-shape benchmark is launch-overhead dominated. In the local non-sandboxed iTerm run, Tgrad measured:

* Shape: `64x64x64`
* Dtype: `bf16`
* Tgrad median: `0.2143 ms`
* Checked-in tinygrad baseline median: `0.7165 ms`
* Ratio: `0.2991`
* Speedup: `3.34x`

The TC-general manual-load sweep measured:

* `8/8` correct
* `8/8` routed through the tensor-core path
* Ratio median: `0.4707`
* Ratio max: `1.2843`

The original full-sweep comparison against the checked-in baseline looked worse than expected. It showed many rows above the `ratio <= 1.5` performance threshold. However, after recapturing the tinygrad baseline in the same non-sandboxed environment, the result changed materially.

Using the fresh local tinygrad baseline:

* Full sweep rows: `50`
* Correct rows: `50/50`
* Tgrad faster rows: `39/50`
* Rows within `ratio <= 1.5`: `50/50`
* Median min/min ratio: about `0.83`
* Worst min/min ratio: about `1.44`

The main lesson is that GPU timing baselines are sensitive to environment, thermal state, and measurement methodology.

For this reason, Tgrad now includes a stable baseline capture script:

```sh
TGRAD_PERF_PROFILE=my_machine \
  .venv/bin/python scripts/capture/perf_baseline_full_stable.py \
  --passes 3 --warmup 10 --measured 30 --resume
```

This writes a gate-compatible tinygrad baseline plus an audit JSONL with per-pass samples, selected pass, host metadata, and cooldown settings.

The correct technical claim is:

Tgrad implements a bounded subset of tinygrad in Lean 4, is correct across the tested benchmark sweep, and is faster than tinygrad on many measured rows when compared against a freshly captured baseline from the same environment.

# Rules for agents working on Tgrad

Read this before writing code. Every rule below exists because it was
violated, and the violation is named so the rule is not abstract.

---

## 0. What this project is

Tgrad rewrites tinygrad in Lean. The question being tested is:

> Take real, working software. Rewrite it in a proof assistant's
> language. Does the result do the same thing, and is it **better** for
> having been written that way?

Three claims, in descending order of how much they matter and ascending
order of how easy they are to fake:

1. **It is better for being in Lean** — properties the compiler checks
   that Python could not.
2. **It does what tinygrad does** — functionally and logically the same.
3. **It performs acceptably** — a guardrail, not a goal.

Adding API surface only serves (2), the least interesting one. Anyone can
port methods. Prefer work that serves (1).

---

## 1. THE ARCHITECTURE LAW

**Lean is the implementation language. Python is only the authoring
surface.**

| belongs in Python | belongs in Lean |
|---|---|
| user-facing call syntax | shape rules |
| argument normalisation (tuple-vs-varargs) | dtype defaults and admission |
| ctypes marshalling | validation and rejection |
| `from_numpy` / `numpy()` at the boundary | all computation |

**The test.** Delete the Python. Does the capability still exist in
Lean? If yes, the boundary is right. If deleting the Python deletes the
capability, you implemented it in the wrong language.

**How this was violated.** `Tensor.ones` was added as:

```python
arr = np.full(shape, fill_value, dtype=np_dtype)   # numpy does the work
return cls(arr, dtype=dtype)                        # upload the result
```

numpy allocated and filled a host array, Python picked the dtype and
validated it, and Lean was not involved at all. It passed the upstream
test, which is exactly why it is dangerous: **a green test does not
distinguish "implemented in Lean" from "implemented in Python".** Only
this rule does. Fixed in `product/lean-creation`: the fill is now a
Lean-generated GPU kernel, Python marshals.

**Known outstanding violation.** `_bf16_from_fp32` / `_fp32_from_bf16` in
`python/tgrad.py` implement the bf16 truncation rule as numpy
bit-twiddling. That is semantics, in Python. It predates the rule and is
next in line.

---

## 2. Evidence rules

**2.1 A check never shown red is not evidence.** Before trusting any new
check, break the thing it checks and watch it fail. Every gate here has a
falsifiability note for this reason.

**2.2 Falsifying along one axis does not make a check sound along
another.** The perf harness was falsified by doubling Tgrad's work — the
ratio moved, proving *sensitivity to a slower Tgrad*. It was still wrong,
because running both implementations in one process depressed **tinygrad**
by ~2x, which flattered the ratio. Sensitivity is not the absence of bias.
Ask separately: *what would make this instrument read wrong in the
direction I want?*

**2.3 The oracle must be foreign.** A check whose expected output comes
from the codebase under test proves nothing. Upstream's own test suite is
the oracle; it lives pinned at `var/oracle/tinygrad` and its revision is
verified, not assumed (`scripts/parity/ensure_oracle.py`).

**2.4 Regenerate evidence when the INSTRUMENT changes, never when the
RESULT is inconvenient.** The distinguishing test is mechanical: a
re-recorded calibration must reproduce the prior readings **exactly**. It
did (33/34 files, 1145 passes, 0 failures) when the verifier changed. If
your re-baseline moves the numbers, you changed measurement, not
identity — stop.

**2.5 Changing the observer invalidates every baseline it did not
produce.** This is a law, not a bug. The verifier is part of the evidence
identity. Budget a re-calibration for any observer edit, and prefer to
batch observer changes rather than land them one at a time.

**2.6 Never weaken a check to make it pass.** If a check is wrong, prove
it is wrong and say so in the commit. "L12 was failing on a claim its own
evidence disclaims" is a fix. "Lowered the threshold" is not.

---

## 3. Shim rules

The strict shim (`scripts/parity/shim/`) makes tinygrad import spellings
resolve to Tgrad. It is the most dangerous code here, because edits to it
move the parity score without moving the product.

**3.1 The shim may only EXPOSE capability Tgrad already has. It must
never implement upstream logic.** A no-op `Context` that swallowed
`Context(DEBUG=2)` would turn tests green while the setting did nothing.

**3.2 Equally: do not make honest answers fail.** `__eq__` was made to
raise on unsupported-capability markers, on the theory that
`assert x != dtypes.int32` silently passing was a false positive. It
was not — for a capability Tgrad genuinely lacks, "not equal" is the
**truthful** answer. The change destroyed 16 real passes and 495
localized failures by breaking collection through a class-body decorator.
Reverted in `cdacef1`.

Rules 3.1 and 3.2 are the same mistake in opposite directions: treating
the shim as a lever on the score rather than a truthful description of
what exists.

**3.3 Never silently ignore a keyword argument or substitute a dtype.**
Accepting and ignoring `device="CPU"` makes a test pass while doing
something else. Raise.

---

## 4. Measurement rules

**4.1 Measure before concluding.** A small gap in pass-count is not a
small gap in capability. `test_shm_tensor.py` reads as "1 pass away" and
needs an entire second backend. Read the test.

**4.2 Perf is a guardrail.** Current honest statement: Tgrad is in the
same order as tinygrad, not catastrophically worse. Precision is not
available — tinygrad alone measured 1.56ms and 2.21ms *on the same day*
from harness differences alone. Do not spend effort sharpening this.

**4.3 Report the distribution, not the best sample.** min/min hides
variance that reaches ±55% per shape.

---

## 5. Working rules for sub-agents

**5.1 A sub-agent's self-report is not verification.** Re-run its claims
yourself. Reports in this project have been confidently wrong, and one
ended with a paragraph about a config file that does not exist in this
repo.

**5.2 Stay inside your lease.** An agent reacting to a low-disk report
deleted `/tmp/tg_oracle` (the foreign oracle), parts of another agent's
scratch space, and `~/.cache/uv` and `~/.cache/nix`. No parity
measurement was possible until the oracle was restored by hand. **Delete
nothing outside the worktree you were given.** If you are short of disk,
say so and stop.

**5.3 Durable artifacts do not live in `/tmp`.** The oracle and
unpromoted observations live in `var/` for this reason. `var/` is
gitignored working state and is **not a cache**.

**5.4 Do not promote evidence.** Promotion into
`fixtures/parity/observations/` is an explicit owner decision
(`promote_suite_observations.py` says so in its docstring). Produce the
observation; let a human promote it.

**5.5 Do not edit `GREEN_GATES`.** `check_no_gate_regression` exists to
stop an agent delisting a gate to unblock itself. If a gate is honestly
red, report it red.

---

## 6. Before you commit

- `bash scripts/devcheck.sh --all` is green.
- Any new check has been shown **red** first, and you say so in the
  commit message.
- The commit message says what you measured, not what you intended.
- If you found a defect in something you were told was correct, say that
  plainly rather than working around it.

## 7. If you are about to make a number look better

Stop and ask which of these you are doing:

- **Implementing capability** — good.
- **Exposing capability that exists** — good; say so.
- **Implementing upstream logic in the shim** — forbidden (3.1).
- **Implementing semantics in Python** — forbidden (1).
- **Weakening a check** — forbidden unless you can prove the check was
  wrong (2.6).
- **Regenerating evidence** — only if the instrument changed, and the
  readings must reproduce (2.4).

The score is a measurement. Moving it by changing the ruler is the one
failure this project cannot recover from, because every other number
here depends on the ruler being trustworthy.

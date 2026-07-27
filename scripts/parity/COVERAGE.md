# Total parity coverage matrices

The generated upstream target is a denominator, not a score.  The coverage
matrix joins that denominator to one immutable Tgrad subject and records every
row without collapsing gaps into a flattering percentage.

## Artifacts

- `upstream_19c4d736f2bc.json` is the checked foreign inventory.
- `requirements_19c4d736f2bc.json` is its deterministic 590-row translation
  into the same requirement IDs and policy fields used by
  `Tgrad.Spec.Parity`. The reviewed literal `590`, ordered-ID digest, row
  ordinals, row identity hashes, and upstream source locators must all agree.
- A coverage matrix is a separate, subject-specific artifact.  It is not
  committed until its subject/profile/classification/adapter/relation/
  environment/oracle identities have been reviewed.

The requirement inventory is regenerated with:

```sh
python3 scripts/parity/coverage_matrix.py requirements
```

and checked without rewriting with:

```sh
python3 scripts/parity/coverage_matrix.py requirements --check
```

## Matrix lifecycle

`init` requires a clean subject Git tree, a clean verifier Git tree, and a
versioned profile definition.  Classification, adapter, relation, environment,
and oracle identities may initially be epistemically `unknown`, but each must
carry a concrete resolution path.  In-profile test rows then remain
`unclassified`; importing the denominator cannot silently make them required,
excluded, or passing.

A profile definition is content-hashed JSON with `kind` set to
`tgrad-parity-profile`, the selected `profile`, and ordered
`environment_ids`. Semantic/public/ecosystem profiles use `host`; Metal uses
`metal`; all-backends must reproduce all 16 generated backend names exactly;
portable must name `cpu` plus at least one declared accelerator. This makes
the execution product explicit instead of hiding it behind
`declared-backend`.

```sh
python3 scripts/parity/coverage_matrix.py init \
  --subject-repo /path/to/clean/tgrad-subject \
  --profile publicApi \
  --profile-definition /path/to/reviewed-public-api-profile.json \
  --output /private/tmp/tgrad-coverage.json
```

The classification and substitution-shim workers own the artifacts that will
replace those unknown identities.  The matrix tool deliberately does not
guess their outcomes or inspect which features Tgrad currently implements.
After importing a reviewed classification, regenerate the obligation product
into a new file (never in place):

```sh
python3 scripts/parity/coverage_matrix.py reconcile \
  --matrix /path/to/classified-matrix.json \
  --output /path/to/reconciled-matrix.json
```

Reconciliation deliberately clears observations and evidence: changing
applicability changes the question, so old answers cannot be retained by
position.

Validation has three monotonically stronger levels:

1. **Structurally total:** exactly one ordered cell for every generated
   requirement; no missing, duplicate, extra, or reordered denominator rows.
2. **Reportable:** every identity is confirmed and no row remains
   `unclassified`.  Exact counts may be published, including failures and
   unobserved gaps.
3. **Fully observed / conformant:** each required row expands to
   `dimension × environment` obligations. Every obligation has exactly one
   selected observation; conformance additionally requires every observation
   to pass.

```sh
python3 scripts/parity/coverage_matrix.py validate \
  --matrix /path/to/matrix.json

python3 scripts/parity/coverage_matrix.py validate \
  --matrix /path/to/matrix.json --require-reportable
```

There is intentionally no scalar score or percentage.  The summary reports
exact disposition/result counts and booleans for structural totality,
reportability, full observation, and universal conformance.

## Observation and evidence law

Cells contain applicability decisions, never authored result fields. Their
states are derived with this precedence: a missing obligation is unobserved;
any fail/error makes the row fail; otherwise any blocked observation makes it
blocked; only an all-pass product makes it pass.

A `pass` observation is accepted only when its evidence:

- names one exact requirement/dimension/environment obligation and relation;
- matches the pinned upstream revision;
- matches the matrix subject tree and verifier tree;
- matches the classification, adapter, relation-registry, environment, and
  oracle-contract hashes;
- uses a validator with a rejected mutant for that exact dimension,
  environment, relation, adapter, oracle, and scenario identity; and
- has an upstream, independent, mathematical, or internal-differential origin,
  never a solely self-referential origin.

Every observation selects one evidence record and every evidence record is
selected exactly once. Orphan evidence, duplicate observations for one
obligation, stale identities, and evidence written against a different
subject are fatal. `not_applicable` and `excluded` cells require a rationale;
exclusion additionally requires an imported pre-score classification.

## Required integration order

1. Freeze the public compatibility classification before any Tgrad suite run.
2. Calibrate the upstream-on-upstream runner and retain diagnosable failures.
3. Land the audited thin substitution shim.
4. Initialize the matrix for an immutable Tgrad candidate and reviewed profile.
5. Import observation packets without rewriting expected results.
6. Validate exact counts and joins, then render the checked Lean snapshot.

Until step 6, `Tgrad.Spec.Parity.targetContract` remains unknown.

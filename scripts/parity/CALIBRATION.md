# Upstream-suite calibration

The upstream test suite is an oracle only after its runner is shown to be
complete, attributable, diagnosable, and repeatable against tinygrad itself.
`suite_upstream_null_19c4d736f2bc.json` predates this contract: it has useful
counts, but no checkout tree, environment lock, raw diagnostics, per-test
outcomes, content identity, or repeated-run comparison. It is retained as
unpromoted history.

## Protocol

1. Capture the exact Python executable, package set, host, and forced selector
   environment into a content-hashed manifest.
2. Run from a clean checkout at the generated upstream commit and tree.
3. Refuse an inventory that differs from the generated 54/43/41 file groups.
4. Run each file separately, retaining stdout, stderr, and JUnit XML.
5. Publish the run directory atomically only after all selected files finish.
6. Repeat the full run under the same identities.
7. Compare stable per-test semantic outcomes. Raw diagnostic hashes remain
   separate so timing text cannot masquerade as semantic drift.

```sh
python3 scripts/parity/calibrate_upstream_suite.py environment \
  --python /path/to/venv/bin/python \
  --output /private/tmp/tgrad-oracle-environment.json

python3 scripts/parity/calibrate_upstream_suite.py run \
  --checkout /path/to/clean/tinygrad \
  --environment /private/tmp/tgrad-oracle-environment.json \
  --group null --output /private/tmp/tgrad-oracle-null-a

python3 scripts/parity/calibrate_upstream_suite.py compare \
  --first /private/tmp/tgrad-oracle-null-a \
  --second /private/tmp/tgrad-oracle-null-b \
  --output /private/tmp/tgrad-oracle-null-comparison.json
```

`--limit` is diagnostic only and records `complete=false`; it can never
promote. A comparison is promotion-ready only when identities match, both
runs are complete, semantic outcomes are equal, and every file has a strict
pass with no skipped tests. Existing upstream failures, collection errors,
empty files, and skips remain explicit blockers until the environment is
repaired or a reviewed oracle-disposition contract marks the affected cases
unavailable. They are never silently counted as Tgrad failures or passes.

This protocol performs no Tgrad substitution. The active adapter worker owns
that separate boundary.

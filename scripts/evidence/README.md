# Gate evidence protocol

Gate execution and release publication are separate operations.

Every ordinary gate run writes to `TGRAD_EVIDENCE_DIR`, which defaults to the
owned `TGRAD_RUN_DIR/gate_evidence` directory. It never rewrites
`fixtures/gate_evidence`. Umbrella gates inherit the same directory, so their
roll-up hashes refer to one run-owned set.

A release candidate is collected serially from a clean commit:

```sh
bash scripts/gate.sh --regenerate-evidence \
  /private/tmp/tgrad-evidence-COMMIT \
  /private/tmp/tgrad-performance/prepared_runtime_certificate.json
```

The performance certificate and every completed paired-runtime artifact set it
names remain external run products until candidate collection copies them into
the snapshot. This avoids the impossible demand that a committed certificate
name the commit that contains itself. The certificate does not get to declare
its own run/session counts or verdict: `scripts/perf/release_certificate.py`
parses each retained completion marker, summary, and raw observation stream;
reconstructs independent run identities, sessions, paired log ratios, and
hierarchical intervals; then evaluates the source-controlled reviewed rule.

This command deliberately continues after red gates. It retains:

- the exact source commit and tree;
- one process outcome and log for every gate in `ALL_GATES`;
- an explicit blocker list for every blocked outcome;
- the final evidence set used by umbrella gates;
- immutable before/after observations and a producer record for every cited
  run/build artifact, plus a content-addressed copy of its bytes;
- a manifest written only after exact-cover, roll-up, and referent checks.

The gate inventory and each gate's hash locators are independent reviewed
contracts in `release_inventory.json` and `hash_contract.json`. Their digests
are pinned in both Python and `Tgrad/Spec/EvidenceSnapshot.lean`; shrinking the
runner arrays, swapping a same-content source path, or satisfying an output
claim with a different produced file does not change the denominator silently.

The command exits nonzero but preserves the directory when any gate is red,
blocked, missing, or has an unresolved claim. Such a candidate is diagnostic
history, not release evidence.

Only a `complete_green` candidate can be promoted:

```sh
python3 scripts/evidence/candidate.py promote \
  --candidate /private/tmp/tgrad-evidence-COMMIT \
  --backup /private/tmp/tgrad-evidence-previous
```

Promotion is explicit. It assembles only the manifest's exact closure, rejects
symlinks and extras, runs the fatal provenance audit before touching canonical
evidence, replaces the canonical directory with same-filesystem renames, and
audits again after the rename. An in-process failure restores the exact backup.
The backup is retained for review after success. Git publication is a separate
reviewed commit: filesystem replacement and Git ref movement are not presented
as one atomic transaction.

The promoted snapshot contains `RUN_MANIFEST.json`, outcomes, logs, producer
records, the performance inputs, and the minimal retained artifact closure.
Validate it with the fatal release check:

```sh
bash scripts/gate.sh --verify-evidence
```

The current committed legacy evidence has no run manifest and therefore fails
this strict check. That is intentional. The most recent honest regeneration
found L11 empirically non-repeatable. The checked decision rule remains in
`calibration_required`, so no performance certificate—and therefore no green
release snapshot—can yet be promoted. This is an explicit dependency, not a
missing file silently replaced by a frozen denominator.

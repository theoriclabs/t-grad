# `var/` — durable working state (not a cache)

Holds two things the parity program cannot work without:

- `var/oracle/tinygrad` — the pinned upstream checkout at
  `19c4d736f2bc8e26d21f08b28ffd6298408da00f`. This is the **foreign
  oracle**: the whole parity metric divides by tinygrad's own suite, and
  its authority comes from not being authored here.
- `var/observations/` — freshly computed suite observations. These are
  evidence that has not been **promoted** yet; promotion into
  `fixtures/parity/observations/` is an explicit owner decision made with
  `scripts/parity/promote_suite_observations.py`.

The contents are gitignored and regenerable, but regeneration is not
free: the oracle is a 165 MB clone and an observation is a full suite run.

**Do not treat this as a cache.** It lives here rather than in `/tmp` or
`~/.cache` because on 2026-07-27 a disk-pressure sweep deleted
`/tmp/tg_oracle` (the oracle), parts of `/tmp/claude-501`, and
`~/.cache/uv` and `~/.cache/nix`. No parity measurement was possible
until the oracle was restored by hand, and restoring it *correctly*
required knowing where the revision pin was recorded. A working tree is
not a cache; those two directories are the first things a sweep reclaims.

Restore or verify the oracle at any time — it is idempotent and checks
the revision rather than assuming an existing directory is the right
tree:

    python3 scripts/parity/ensure_oracle.py
    python3 scripts/parity/ensure_oracle.py --verify-only

Override locations with `TGRAD_PARITY_ORACLE` and
`TGRAD_PARITY_OUTPUT_DIR`.

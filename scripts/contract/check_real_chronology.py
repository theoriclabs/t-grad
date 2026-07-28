#!/usr/bin/env python3
"""Bind the freeze-integrity chronology check to THIS repository's real cycles.

`check_cycle_chronology.py` exercises Git ancestry mechanics on synthetic
temporary repositories, and says so plainly: "The Python exercise proves Git
mechanics only on synthetic temporary repositories; it does not generate a
live Tgrad certificate." That leaves the property those mechanics exist to
establish --- that Tgrad's own cycles were prospective --- checked only by
hand, in prose, in `docs/growth_log_2026-07-27.md`.

This module points the same mechanics at the real history.

For each declared cycle it requires:

  1. the freeze commit is an ancestor of the candidate commit, and
  2. the judging closure is byte-identical at EVERY commit on the
     ancestry path between them --- not merely at the two endpoints.

(2) is the part a human review does not do. Comparing only the endpoints
accepts a history that mutated the observer mid-interval and reverted it
before the candidate landed; the intermediate commits are where that shows.
`--ancestry-path` is likewise what keeps an unrelated merge side from
contributing commits that were never between the freeze and the candidate.

The judging closure is the set of files that determine the verdict: the
observer, the relation it applies, and the immutable upstream baseline. If
those bytes never move while the product changes, the product was measured
against a fixed ruler.

WHAT THIS DOES NOT ESTABLISH. The commit pairs below are transcribed from the
growth log, and a transcription is not an attestation: nothing here proves the
freeze commit was authored before the implementation was known, only that the
recorded graph is consistent with that claim. Cryptographic artifact identity
and prospective protocol attestation remain open, exactly as the assurance
packet states. This closes the gap between "asserted in prose" and "checked
against the object graph" --- no more.

Usage:
    python3 scripts/contract/check_real_chronology.py
    python3 scripts/contract/check_real_chronology.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_cycle_chronology import (  # noqa: E402
    ancestry_commits,
    ensure_freeze_is_ancestor,
)

REPO = Path(__file__).resolve().parents[2]

# Files that determine the verdict. If any of these changes between a freeze
# and its candidate, the ruler moved while the product was being measured.
JUDGING_CLOSURE = (
    "scripts/spec/observe_broadcast_add.py",
    "scripts/spec/broadcast_add_relation.py",
    "fixtures/requirements/observations/"
    "84a58222575eab06ecc72889e1dbbe2a2084849356673a8d299c21ad2e41a844",
)

# Transcribed from docs/growth_log_2026-07-27.md. Each pair is
# (label, freeze commit, candidate/implementation commit).
REAL_CYCLES = (
    ("public Tensor constructor", "58d18ce", "4a489e3"),
    ("float32 view readback", "924d6a4", "367011e"),
    ("realization identity", "6f7ef13", "24198df"),
    ("ranked broadcast", "93811f2", "8016524"),
    ("int32 elementwise", "aeb30e0", "c4984c8"),
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def closure_digest(repo: Path, commit: str, paths=JUDGING_CLOSURE) -> str:
    """Content identity of the judging closure at one commit.

    Uses blob/tree object ids rather than file contents: git already
    content-addresses them, so this is both cheaper and exactly as
    discriminating. A path absent at this commit is recorded as absent
    rather than skipped --- deleting the observer must not read as "the
    observer did not change".
    """
    entries = []
    for path in sorted(paths):
        out = _git(repo, "rev-parse", f"{commit}:{path}")
        oid = out.stdout.strip() if out.returncode == 0 else "<absent>"
        entries.append(f"{len(path)}:{path}={oid}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def _require_commit(repo: Path, rev: str) -> None:
    """Distinguish "commit is gone" from "commit is not an ancestor".

    The registry pins real SHAs, so a history rewrite (squash, filter,
    fresh shallow clone) makes them vanish. Reporting that as "not an
    ancestor" would send a reader looking for a chronology violation that
    does not exist.
    """
    if _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").returncode != 0:
        raise RuntimeError(
            f"commit {rev} is not in this repository "
            "(history rewritten, or a shallow/partial clone)"
        )


def check_cycle(repo: Path, label: str, freeze: str, candidate: str) -> dict:
    _require_commit(repo, freeze)
    _require_commit(repo, candidate)
    ensure_freeze_is_ancestor(repo, freeze, candidate)
    path = ancestry_commits(repo, freeze, candidate)
    expected = closure_digest(repo, freeze)
    drifted = [c for c in path if closure_digest(repo, c) != expected]
    return {
        "label": label,
        "freeze": freeze,
        "candidate": candidate,
        "commits_on_path": len(path),
        "closure": expected[:12],
        "drifted": drifted,
        "ok": not drifted,
    }


def run(repo: Path, cycles=REAL_CYCLES) -> int:
    failures = 0
    for label, freeze, candidate in cycles:
        try:
            r = check_cycle(repo, label, freeze, candidate)
        except RuntimeError as exc:
            print(f"  x {label}: {exc}")
            failures += 1
            continue
        if r["ok"]:
            print(
                f"  ok {label}: freeze {freeze} -> {candidate}, "
                f"{r['commits_on_path']} commits on ancestry path, "
                f"closure {r['closure']} unchanged throughout"
            )
        else:
            print(
                f"  x {label}: judging closure MOVED at "
                f"{', '.join(c[:9] for c in r['drifted'])}"
            )
            failures += 1
    return failures


def self_test(repo: Path) -> int:
    """Show the check can fail. A check never seen red is not evidence.

    Two injected faults, each isolating one of the two guarantees:
    a reversed pair (candidate before freeze) must be rejected by the
    ancestry requirement, and a pair spanning a commit that genuinely
    rewrote the observer must be rejected by the closure requirement.
    """
    print("  self-test: reversed pair must be rejected")
    bad = (("reversed", "4a489e3", "58d18ce"),)
    if run(repo, bad) == 0:
        print("  x SELF-TEST FAILED: reversed pair accepted")
        return 1

    print("  self-test: interval containing an observer rewrite must be rejected")
    # bf49fbc introduced the observer; db984c0 is a later observer change.
    # Any interval spanning them has a moving judging closure.
    spanning = (("observer-rewrite span", "bf49fbc", "58d18ce"),)
    if run(repo, spanning) == 0:
        print("  x SELF-TEST FAILED: moving judging closure accepted")
        return 1
    print("  self-test: both faults correctly rejected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.repo)

    print(f"real cycle chronology ({len(REAL_CYCLES)} declared cycles)")
    failures = run(args.repo)
    print()
    if failures:
        print(f"real_chronology: FAIL ({failures} cycle(s))")
        return 1
    print("real_chronology: all declared cycles prospective and ruler-stable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

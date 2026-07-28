#!/usr/bin/env python3
"""Fail if fixtures/gate_evidence/ is tracked or staged in git.

Gate evidence is a per-sweep runtime artifact: a child writes JSON, a parent
confirms the child ran by checking that file exists and hashing it. When those
JSON files are committed, `[[ -f ]]` is satisfied on every fresh clone without
any child ever running, and umbrella roll-ups become vacuous.

This check replaces scripts/dev/evidence_provenance_audit.py. The old auditor
inspected committed evidence that should never have been committed. The safety
worth keeping is automatic detection of re-tracking: if anything under
fixtures/gate_evidence/ re-enters the index, fail immediately.

Exit 0 when the directory is untracked and no *.json under it is staged.
Exit 1 otherwise.

Usage:  python3 scripts/dev/gate_evidence_not_tracked.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREFIX = "fixtures/gate_evidence"


def sh(*args: str) -> tuple[int, str]:
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def main() -> int:
    # Present in the index (not merely staged-for-deletion).
    _, tracked = sh("git", "ls-files", "--", PREFIX)
    tracked_files = [line for line in tracked.splitlines() if line]
    # Staged additions/modifications only. Diff-filter excludes deletions so
    # retiring previously-committed evidence (git rm) does not self-fail.
    # Includes force-adds of gitignored files (A).
    _, staged = sh(
        "git", "diff", "--cached", "--name-only", "--diff-filter=ACMR",
        "--", PREFIX,
    )
    staged_files = [line for line in staged.splitlines() if line]

    if not tracked_files and not staged_files:
        print(
            "gate_evidence_not_tracked: fixtures/gate_evidence/ is untracked "
            "and no evidence JSON is staged"
        )
        return 0

    print("gate_evidence_not_tracked: FAILED", file=sys.stderr)
    if tracked_files:
        print(
            "  fixtures/gate_evidence/ must not be tracked by git "
            f"({len(tracked_files)} path(s)):",
            file=sys.stderr,
        )
        for path in tracked_files:
            print(f"    {path}", file=sys.stderr)
    if staged_files:
        print(
            "  no fixtures/gate_evidence/*.json may be staged "
            f"({len(staged_files)} path(s)):",
            file=sys.stderr,
        )
        for path in staged_files:
            print(f"    {path}", file=sys.stderr)
    print(
        "\n  Evidence is a per-sweep runtime artifact. Re-tracking it makes\n"
        "  umbrella [[ -f ]] checks vacuous again. Unstage/remove these paths\n"
        "  and keep fixtures/gate_evidence/ in .gitignore.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

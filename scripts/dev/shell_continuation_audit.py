#!/usr/bin/env python3
"""Reject a bash line-continuation immediately followed by a comment.

    cmd --a \
        # why --b matters
        --b

does not do what it looks like. Bash joins line 1 to the `#` line, so
the comment swallows the rest of that joined line and `--b` runs as its
own command. The flags are silently dropped and the shell prints
`--b: command not found`, which scrolls past inside a gate that is
printing plenty of other output.

This is not hypothetical. `scripts/gates/L12.sh` had exactly this shape
around its `bench-full` invocation: the sweep ran with no
`--use-algebraic-emit` (so it benchmarked the wrong emitter --- the
whole purpose of the gate), no `--output`, and default warmup/measured,
which is the single-sample regime that the dropped comment itself
spends eight lines explaining must never be used. The gate reported
`perf_miss: 30` and nobody could tell why.

The failure mode is what makes it worth a permanent check: the comment
explains a rigour the code is not applying, and the louder the comment,
the more convincing the gate looks. Grep-level review does not catch it,
`bash -n` does not catch it (the result is valid syntax), and the gate
does not fail --- it just measures something else.

Exits 1 on any occurrence.

Usage:  python3 scripts/dev/shell_continuation_audit.py [ROOT ...]
"""
from __future__ import annotations

import pathlib
import sys


def offenders(root: pathlib.Path) -> list[tuple[pathlib.Path, int, str]]:
    found = []
    for f in sorted(root.rglob("*.sh")):
        lines = f.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith("\\") and lines[i + 1].lstrip().startswith("#"):
                found.append((f, i + 1, line.strip()))
    return found


def main() -> int:
    roots = [pathlib.Path(a) for a in sys.argv[1:]] or [pathlib.Path("scripts")]
    bad = [o for r in roots for o in offenders(r)]
    if not bad:
        n = sum(1 for r in roots for _ in r.rglob("*.sh"))
        print(f"shell_continuation_audit: {n} scripts, no continuation-then-comment")
        return 0
    print("shell_continuation_audit: FAILED", file=sys.stderr)
    for f, ln, text in bad:
        print(f"  {f}:{ln}: continuation followed by a comment", file=sys.stderr)
        print(f"      {text}", file=sys.stderr)
    print(
        "\n  Bash joins the continued line into the comment. Everything after\n"
        "  it runs as a separate command, so the flags are dropped silently.\n"
        "  Move the comment ABOVE the command.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

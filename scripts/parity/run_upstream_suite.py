#!/usr/bin/env python3
"""Run tinygrad's own test suite and record per-file results.

PARITY.md makes tinygrad's suite the parity metric because it is
authored upstream: an agent cannot satisfy it without implementing the
feature. This runs it and writes the numbers.

Two modes, and the order matters:

  --against upstream   calibration. Run the suite against tinygrad
                       itself. Until this passes, a low Tgrad score is
                       uninterpretable — you cannot tell an unimplemented
                       feature from a broken harness or a missing
                       backend. This is the same discipline as showing a
                       check red before trusting it green.

  --against tgrad      the score. Same suite, Tgrad substituted for
                       tinygrad.

Results are per file, never a single percentage: one number would hide
which capability is missing, and `test/null` alone is 54 files that
fail for entirely different reasons.

Usage:
    python3 scripts/parity/run_upstream_suite.py --against upstream
    python3 scripts/parity/run_upstream_suite.py --against upstream --group null
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "fixtures" / "parity"
VENV_PY = REPO / ".venv" / "bin" / "python"
DEFAULT_CHECKOUT = Path("/tmp/tg_oracle/tinygrad")

SUMMARY = re.compile(
    r"(?:(\d+) failed)?,?\s*(?:(\d+) passed)?,?\s*(?:(\d+) skipped)?"
    r",?\s*(?:(\d+) errors?)?"
)


def run_file(py: Path, checkout: Path, rel: str, timeout: int, env_extra: dict) -> dict:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(checkout)
    env.update(env_extra)
    try:
        p = subprocess.run(
            [str(py), "-m", "pytest", rel, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=checkout, capture_output=True, text=True, timeout=timeout, env=env,
        )
        out = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return {"file": rel, "status": "timeout", "passed": 0, "failed": 0,
                "skipped": 0, "errors": 0, "collected": 0}

    passed = failed = skipped = errors = 0
    for m in re.finditer(r"(\d+) (passed|failed|skipped|error[s]?|xfailed|xpassed)", out):
        n, kind = int(m.group(1)), m.group(2)
        if kind == "passed":
            passed = n
        elif kind == "failed":
            failed = n
        elif kind == "skipped":
            skipped = n
        elif kind.startswith("error"):
            errors = n

    collected = passed + failed + skipped + errors
    if rc == 0 and collected:
        status = "pass"
    elif errors or "ImportError" in out or "ModuleNotFoundError" in out:
        status = "collect_error"
    elif failed:
        status = "fail"
    elif collected == 0:
        status = "empty"
    else:
        status = "fail"
    return {"file": rel, "status": status, "passed": passed, "failed": failed,
            "skipped": skipped, "errors": errors, "collected": collected}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", choices=["upstream", "tgrad"], required=True)
    ap.add_argument("--group", default="null", choices=["null", "unit", "backend"])
    ap.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not VENV_PY.exists():
        print(f"missing venv python at {VENV_PY}", file=sys.stderr)
        return 1
    group_dir = args.checkout / "test" / args.group
    if not group_dir.is_dir():
        print(f"missing {group_dir}", file=sys.stderr)
        return 1

    sha = subprocess.run(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    files = sorted(
        f"test/{args.group}/{p.name}"
        for p in group_dir.glob("*.py")
        if p.name != "__init__.py"
    )
    if args.limit:
        files = files[: args.limit]

    env_extra = {}
    if args.against == "tgrad":
        # The substitution shim goes here. Until it exists this mode is
        # refused rather than silently reporting upstream's numbers as
        # Tgrad's, which would be the exact self-referential score
        # PARITY.md rules out.
        print(
            "  --against tgrad is not implemented yet.\n"
            "  Refusing to run: without a substitution shim this would\n"
            "  measure tinygrad against itself and report it as Tgrad's\n"
            "  score.",
            file=sys.stderr,
        )
        return 2

    results = []
    for i, rel in enumerate(files, 1):
        r = run_file(VENV_PY, args.checkout, rel, args.timeout, env_extra)
        results.append(r)
        print(f"  [{i:3d}/{len(files)}] {r['status']:14s} {rel}  "
              f"({r['passed']}p/{r['failed']}f/{r['skipped']}s/{r['errors']}e)")

    agg = {
        "files": len(results),
        "files_pass": sum(1 for r in results if r["status"] == "pass"),
        "files_fail": sum(1 for r in results if r["status"] == "fail"),
        "files_collect_error": sum(1 for r in results if r["status"] == "collect_error"),
        "files_timeout": sum(1 for r in results if r["status"] == "timeout"),
        "tests_passed": sum(r["passed"] for r in results),
        "tests_failed": sum(r["failed"] for r in results),
        "tests_skipped": sum(r["skipped"] for r in results),
        "tests_errors": sum(r["errors"] for r in results),
    }

    doc = {
        "against": args.against,
        "group": args.group,
        "upstream_ref": sha,
        "ran_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregate": agg,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"suite_{args.against}_{args.group}_{sha[:12]}.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print()
    for k, v in agg.items():
        print(f"  {k:22s} {v}")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

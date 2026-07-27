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
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "fixtures" / "parity"
DEFAULT_VENV_PY = Path(
    os.environ.get("TGRAD_PARITY_PYTHON", REPO / ".venv" / "bin" / "python")
)
DEFAULT_CHECKOUT = Path("/tmp/tg_oracle/tinygrad")
SHIM_ROOT = Path(__file__).resolve().parent / "shim"
SHIM_RUNNER = SHIM_ROOT / "run_pytest.py"

SUMMARY = re.compile(
    r"(?:(\d+) failed)?,?\s*(?:(\d+) passed)?,?\s*(?:(\d+) skipped)?"
    r",?\s*(?:(\d+) errors?)?"
)


def run_file(py: Path, checkout: Path, rel: str, timeout: int,
             env_extra: dict, pytest_runner: Path | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(checkout)
    env.update(env_extra)
    pytest_command = ["-m", "pytest"] if pytest_runner is None else [str(pytest_runner)]
    try:
        p = subprocess.run(
            [str(py), *pytest_command, rel, "-q", "--no-header", "-p", "no:cacheprovider"],
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
    ap.add_argument("--python", type=Path, default=DEFAULT_VENV_PY,
                    help="pytest environment (or set TGRAD_PARITY_PYTHON)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--file", action="append", default=[], metavar="NAME",
                    help="run one file from the selected group; repeatable")
    args = ap.parse_args()

    if not args.python.exists():
        print(f"missing venv python at {args.python}", file=sys.stderr)
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
    if args.file and args.limit:
        print("--file and --limit cannot be used together", file=sys.stderr)
        return 1
    if args.file:
        available = {Path(rel).name: rel for rel in files}
        unknown = [name for name in args.file if Path(name).name not in available]
        if unknown:
            print(
                f"unknown test file(s) for group {args.group}: {', '.join(unknown)}",
                file=sys.stderr,
            )
            return 1
        files = [available[Path(name).name] for name in args.file]
    if args.limit:
        files = files[: args.limit]

    env_extra = {}
    pytest_runner = None
    if args.against == "tgrad":
        # Keep Tgrad's authoring package reachable, but never put the upstream
        # checkout on tinygrad's package path.  The bootstrap imports and
        # verifies the regular shim package before pytest changes sys.path.
        inherited_pythonpath = os.environ.get("PYTHONPATH", str(REPO / "python"))
        env_extra["PYTHONPATH"] = os.pathsep.join(
            [str(SHIM_ROOT), inherited_pythonpath]
        )
        # Child interpreters spawned by upstream tests must not prepend their
        # upstream cwd ahead of sitecustomize and the strict shim.
        env_extra["PYTHONSAFEPATH"] = "1"
        pytest_runner = SHIM_RUNNER
        verify = subprocess.run(
            [str(args.python), str(SHIM_RUNNER), "--verify-only"],
            cwd=args.checkout,
            capture_output=True,
            text=True,
            env={**os.environ, **env_extra},
        )
        if verify.returncode != 0:
            print(
                "Tgrad substitution preflight failed; refusing to record a score:\n"
                + verify.stdout + verify.stderr,
                file=sys.stderr,
            )
            return 2
        print(f"  {verify.stdout.strip()}")

    results = []
    for i, rel in enumerate(files, 1):
        r = run_file(args.python, args.checkout, rel, args.timeout,
                     env_extra, pytest_runner)
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

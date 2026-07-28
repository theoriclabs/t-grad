#!/usr/bin/env python3
"""Provision and verify the pinned upstream oracle checkout.

The parity metric divides by tinygrad's own test suite. That checkout is
the foreign oracle: the one artifact in this repository whose authority
comes from NOT being authored here. Everything downstream --- the
canonical api_surface score, the calibration, the classification --- is
meaningless if it is silently the wrong revision.

It used to live at /tmp/tg_oracle/tinygrad, and on 2026-07-27 an agent
reacting to a low-disk report deleted it along with other /tmp state. No
parity measurement was possible until it was restored by hand, and
restoring it *correctly* depended on someone knowing the pin lived in
scripts/parity/extract_upstream.py. Both of those are the failure this
script exists to remove: the location is now durable, and the identity is
checked rather than assumed.

The default location is inside the repository (`var/oracle/`) and is
gitignored. That is deliberate. `/tmp` and `~/.cache` are the two places
a disk-pressure sweep reclaims first --- the same incident also removed
`~/.cache/uv` and `~/.cache/nix` --- and a working tree is not a cache.

Idempotent: run it any time. It clones when absent, fetches when the pin
is missing, and always verifies HEAD before reporting success.

Usage:
    python3 scripts/parity/ensure_oracle.py
    python3 scripts/parity/ensure_oracle.py --verify-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The same revision extract_upstream.py pins. Kept in one place per file
# deliberately: a second copy that drifts would silently score against a
# different upstream than the manifest describes.
PINNED_REF = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
UPSTREAM_URL = "https://github.com/tinygrad/tinygrad.git"
DEFAULT_ORACLE = REPO / "var" / "oracle" / "tinygrad"


def git(repo: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    base = ["git"] if repo is None else ["git", "-C", str(repo)]
    return subprocess.run(
        [*base, *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def head_of(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    out = git(path, "rev-parse", "HEAD")
    return out.stdout.strip() if out.returncode == 0 else None


def verify(path: Path) -> tuple[bool, str]:
    head = head_of(path)
    if head is None:
        return False, f"no checkout at {path}"
    if head != PINNED_REF:
        return False, f"HEAD is {head[:12]}, expected pin {PINNED_REF[:12]}"
    tests = list((path / "test" / "null").glob("*.py")) if (path / "test" / "null").is_dir() else []
    if not tests:
        return False, "checkout has no test/null/*.py; wrong tree or partial clone"
    return True, f"pinned at {PINNED_REF[:12]}, {len(tests)} test/null files"


def ensure(path: Path) -> int:
    ok, detail = verify(path)
    if ok:
        print(f"oracle ok: {path} ({detail})")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        print(f"cloning upstream oracle into {path} ...")
        res = git(None, "clone", "--quiet", UPSTREAM_URL, str(path))
        if res.returncode != 0:
            print(f"clone failed: {res.stderr.strip()[:300]}", file=sys.stderr)
            return 1

    if git(path, "cat-file", "-e", f"{PINNED_REF}^{{commit}}").returncode != 0:
        print("pinned revision absent; fetching ...")
        if git(path, "fetch", "--quiet", "origin", PINNED_REF).returncode != 0:
            git(path, "fetch", "--quiet", "--all")

    res = git(path, "checkout", "--quiet", "--detach", PINNED_REF)
    if res.returncode != 0:
        print(f"checkout of pin failed: {res.stderr.strip()[:300]}", file=sys.stderr)
        return 1

    ok, detail = verify(path)
    if not ok:
        # Never report success on an unverified oracle: a wrong tree here
        # silently rebases every parity number downstream of it.
        print(f"oracle verification FAILED after provisioning: {detail}", file=sys.stderr)
        return 1
    print(f"oracle ready: {path} ({detail})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        ok, detail = verify(args.path)
        print(("oracle ok: " if ok else "oracle NOT usable: ") + detail)
        return 0 if ok else 1
    return ensure(args.path)


if __name__ == "__main__":
    raise SystemExit(main())

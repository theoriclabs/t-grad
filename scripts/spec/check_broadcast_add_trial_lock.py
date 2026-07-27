#!/usr/bin/env python3
"""Verify the immutable prospective broadcast-add trial definition."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = REPO / "fixtures" / "requirements" / "broadcast_add_trial_lock_v1.json"


def git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=REPO, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(path: Path, require_current: bool) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("unsupported trial-lock schema")
    if document.get("status") != "definition_frozen_observer_unimplemented":
        raise RuntimeError("trial lock has an unexpected status")
    revision = document.get("definition_revision")
    expected_tree = document.get("definition_tree")
    actual_tree = git("rev-parse", f"{revision}^{{tree}}").decode().strip()
    if actual_tree != expected_tree:
        raise RuntimeError("frozen definition tree does not match the lock")
    files = document.get("definition_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("trial lock has no definition files")
    for relative, expected_hash in sorted(files.items()):
        frozen = git("show", f"{revision}:{relative}")
        if digest(frozen) != expected_hash:
            raise RuntimeError(f"frozen file hash mismatch: {relative}")
        if require_current:
            current = REPO / relative
            if not current.is_file() or digest(current.read_bytes()) != expected_hash:
                raise RuntimeError(f"current trial definition drifted: {relative}")
    policy = document.get("promotion_policy", {})
    required_true = (
        "definition_drift_invalidates_trial",
        "observer_must_bind_this_lock_sha256",
        "product_candidate_forbidden_before_baseline_observation",
        "adequacy_remains_open",
        "legacy_requirement_remains_historical",
    )
    if any(policy.get(key) is not True for key in required_true):
        raise RuntimeError("trial lock weakens a required promotion policy")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--allow-current-drift", action="store_true")
    args = parser.parse_args()
    document = verify(args.lock.resolve(), not args.allow_current_drift)
    print(
        "broadcast-add trial definition locked "
        f"at {document['definition_revision']} ({len(document['definition_files'])} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

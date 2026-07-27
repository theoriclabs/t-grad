#!/usr/bin/env python3
"""Verify the immutable prospective broadcast-add trial definition."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = REPO / "fixtures" / "requirements" / "broadcast_add_trial_lock_v1.json"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


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
    baseline = document.get("baseline_revision")
    revision = document.get("definition_revision")
    upstream = document.get("upstream_revision")
    if not all(isinstance(item, str) and FULL_SHA.fullmatch(item)
               for item in (baseline, revision, upstream)):
        raise RuntimeError("trial revisions must be full hexadecimal commit ids")
    git("rev-parse", f"{baseline}^{{commit}}")
    git("rev-parse", f"{revision}^{{commit}}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, revision],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("trial baseline is not an ancestor of its definition")
    expected_tree = document.get("definition_tree")
    actual_tree = git("rev-parse", f"{revision}^{{tree}}").decode().strip()
    if actual_tree != expected_tree:
        raise RuntimeError("frozen definition tree does not match the lock")
    expected_changes = document.get("definition_change_set")
    if not isinstance(expected_changes, list) or not expected_changes:
        raise RuntimeError("trial lock has no definition change set")
    if any(not isinstance(item, str) or not item for item in expected_changes):
        raise RuntimeError("trial definition change set contains an invalid path")
    if len(expected_changes) != len(set(expected_changes)):
        raise RuntimeError("trial definition change set contains duplicates")
    actual_changes = sorted(
        line for line in git("diff", "--name-only", baseline, revision)
        .decode().splitlines() if line
    )
    if actual_changes != sorted(expected_changes):
        raise RuntimeError("frozen definition change set does not match the lock")
    observer_path = document.get("pending_artifacts", {}).get("observer")
    if not isinstance(observer_path, str) or not observer_path:
        raise RuntimeError("trial lock does not name its pending observer")
    observer_at_freeze = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}:{observer_path}"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if observer_at_freeze.returncode == 0:
        raise RuntimeError("observer already existed in the definition revision")
    files = document.get("definition_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("trial lock has no definition files")
    for relative, expected_hash in sorted(files.items()):
        frozen = git("show", f"{revision}:{relative}")
        if not isinstance(expected_hash, str) or not HEX_DIGEST.fullmatch(expected_hash):
            raise RuntimeError(f"invalid frozen file hash: {relative}")
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
    boundary = document.get("execution_boundary", {})
    if boundary.get("backend") != "METAL":
        raise RuntimeError("trial execution backend changed")
    for key in ("hardware_identity_sha256", "python_environment_facts_sha256"):
        value = boundary.get(key)
        if not isinstance(value, str) or not HEX_DIGEST.fullmatch(value):
            raise RuntimeError(f"trial execution boundary has invalid {key}")
    if boundary.get("strict_no_upstream_fallback") is not True:
        raise RuntimeError("trial execution boundary permits upstream fallback")
    if boundary.get("serial_gpu_verification") is not True:
        raise RuntimeError("trial execution boundary permits concurrent GPU verification")
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

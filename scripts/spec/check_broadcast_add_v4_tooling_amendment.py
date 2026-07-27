#!/usr/bin/env python3
"""Check V4's single chronology-tooling correction against frozen Git objects."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "fixtures/requirements/broadcast_add_prospective_v4_tooling_amendment.json"
EXPECTED_BASELINE = "15db155fadbaf68ab26ee2f08747064b0f08480e"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_show(revision: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=REPO, capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace").strip())
    return completed.stdout


def verify(path: Path = DEFAULT_INPUT) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("unsupported V4 tooling-amendment schema")
    if document.get("v3_disposition") != "work_packet_refuted_before_observation":
        raise RuntimeError("V3 disposition must remain explicit")
    if document.get("baseline_revision") != EXPECTED_BASELINE:
        raise RuntimeError("V4 baseline revision changed")
    for path_key, hash_key in (
        ("v3_lock_path", "v3_lock_sha256"),
        ("v3_amendment_path", "v3_amendment_sha256"),
    ):
        bound = REPO / document[path_key]
        if not bound.is_file() or digest(bound.read_bytes()) != document[hash_key]:
            raise RuntimeError(f"V4 does not bind exact {path_key}")
    frozen_generator = git_show(
        document["v3_definition_revision"], document["v3_generator_path"]
    )
    if digest(frozen_generator) != document.get("v3_generator_sha256"):
        raise RuntimeError("V4 does not bind the refuted V3 generator")
    if document.get("tooling_amendment") != {
        "binding": "v2_observer_sha256",
        "from": "current_worktree_file",
        "to": "frozen_git_object_at_v3_definition_revision",
    }:
        raise RuntimeError("V4 changes more than the observer-hash lookup boundary")
    required_true = (
        "v3_trace_footprint_amendment_inherited_unchanged",
        "v3_behavioral_manifest_inherited_unchanged",
        "frozen_before_v4_tooling_implementation",
        "frozen_before_v4_observer_implementation",
        "frozen_before_product_candidate",
    )
    if any(document.get(key) is not True for key in required_true):
        raise RuntimeError("V4 weakens inheritance or chronology")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    document = verify(args.input.resolve())
    print(f"checked {document['trial_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate and verify the V3 verifier-contract lock from frozen Git objects."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "fixtures/requirements/broadcast_add_trial_lock_v3.json"
AMENDMENT = REPO / "fixtures/requirements/broadcast_add_prospective_v3_amendment.json"
V2_LOCK = REPO / "fixtures/requirements/broadcast_add_trial_lock_v2.json"
V4_AMENDMENT = REPO / "fixtures/requirements/broadcast_add_prospective_v4_tooling_amendment.json"
BASELINE = "aac178ff1eebf299075f0adce593328380dfa02a"
DEFINITION = "fbb13c585da75f3e9fa00cf5d87aa72f0ff38ab4"
DEFINITION_FILES = (
    "Tgrad/Growth/BroadcastAddManifestV3.lean",
    "Tgrad/Growth/BroadcastAddManifestV3Generated.lean",
    "Tgrad/Growth/BroadcastAddPacketV3.lean",
    "fixtures/requirements/broadcast_add_prospective_v3_amendment.json",
    "scripts/spec/generate_broadcast_add_amendment_v3.py",
    "scripts/spec/test_broadcast_add_amendment_v3.py",
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def build() -> dict:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    v2_lock = json.loads(V2_LOCK.read_text(encoding="utf-8"))
    if amendment.get("baseline_revision") != BASELINE:
        raise RuntimeError("V3 amendment baseline revision changed")
    if git("rev-parse", f"{DEFINITION}^{{commit}}").decode().strip() != DEFINITION:
        raise RuntimeError("V3 definition commit is absent")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE, DEFINITION], cwd=REPO
    ).returncode != 0:
        raise RuntimeError("V3 baseline is not an ancestor of its definition")
    changes = sorted(filter(None, git(
        "diff", "--name-only", BASELINE, DEFINITION
    ).decode().splitlines()))
    frozen_files = {
        path: digest(git("show", f"{DEFINITION}:{path}"))
        for path in DEFINITION_FILES
    }
    observer_path = amendment["v2_observer_path"]
    if digest(git("show", f"{DEFINITION}:{observer_path}")) != amendment[
        "v2_observer_sha256"
    ]:
        raise RuntimeError("V3 definition was not frozen over the V2 observer")
    diagnostic_path = (
        "fixtures/requirements/observations/" + amendment["refuting_evidence_id"] +
        "/diagnostic.json"
    )
    if digest(git("show", f"{DEFINITION}:{diagnostic_path}")) != amendment[
        "refuting_diagnostic_sha256"
    ]:
        raise RuntimeError("V3 definition does not contain the bound diagnostic")
    return {
        "schema_version": 1,
        "trial_id": amendment["trial_id"],
        "status": "verifier_contract_frozen_v2_observer_present",
        "baseline_revision": BASELINE,
        "definition_revision": DEFINITION,
        "definition_tree": git("rev-parse", f"{DEFINITION}^{{tree}}").decode().strip(),
        "upstream_revision": v2_lock["upstream_revision"],
        "supersedes_v2_lock_sha256": digest(V2_LOCK.read_bytes()),
        "refuting_evidence_id": amendment["refuting_evidence_id"],
        "refuting_diagnostic_sha256": amendment["refuting_diagnostic_sha256"],
        "v2_observer_sha256": amendment["v2_observer_sha256"],
        "definition_change_set": changes,
        "definition_files": frozen_files,
        "execution_boundary": v2_lock["execution_boundary"],
        "pending_artifacts": {
            "observer_v3": "scripts/spec/observe_broadcast_add.py",
            "upstream_observation": "missing",
            "tgrad_observation": "missing",
            "derived_state": "missing",
        },
        "promotion_policy": {
            "definition_drift_invalidates_trial": True,
            "observer_v3_must_bind_this_lock_sha256": True,
            "product_candidate_forbidden_before_baseline_observation": True,
            "diagnostic_cannot_be_used_as_baseline": True,
            "behavioral_manifest_is_inherited_from_v2": True,
            "adequacy_remains_open": True,
        },
    }


def verify(document: dict, require_current: bool = True) -> None:
    expected = build()
    if document != expected:
        raise RuntimeError("V3 trial lock differs from frozen Git facts")
    if require_current:
        v4 = json.loads(V4_AMENDMENT.read_text(encoding="utf-8"))
        authorized = v4.get("v3_generator_path")
        if v4.get("tooling_amendment") != {
            "binding": "v2_observer_sha256",
            "from": "current_worktree_file",
            "to": "frozen_git_object_at_v3_definition_revision",
        }:
            raise RuntimeError("V4 does not authorize V3 generator evolution")
        for relative, expected_hash in document["definition_files"].items():
            current = REPO / relative
            if relative == authorized:
                continue
            if not current.is_file() or digest(current.read_bytes()) != expected_hash:
                raise RuntimeError(f"current V3 definition drifted: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("ascii")
    target = args.output.resolve()
    if args.check:
        if not target.is_file():
            raise RuntimeError("V3 trial lock is missing")
        committed = json.loads(target.read_text(encoding="utf-8"))
        verify(committed, require_current=True)
        if target.read_bytes() != raw:
            raise RuntimeError("V3 trial lock is not canonical")
        print(f"checked {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

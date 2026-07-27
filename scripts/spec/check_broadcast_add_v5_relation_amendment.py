#!/usr/bin/env python3
"""Check V5's verifier-equivalence relation amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "fixtures/requirements/broadcast_add_prospective_v5_relation_amendment.json"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(path: Path = PATH) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("unsupported V5 amendment schema")
    observation = (
        REPO / "fixtures/requirements/observations" /
        document.get("upstream_evidence_id", "") / "observation.json"
    )
    if not observation.is_file() or digest(observation.read_bytes()) != document.get(
        "upstream_observation_sha256"
    ):
        raise RuntimeError("V5 does not bind the exact upstream baseline")
    upstream = json.loads(observation.read_text(encoding="utf-8"))
    if upstream.get("result_kind") != "observation" or upstream.get("against") != "upstream":
        raise RuntimeError("V5 source artifact is not an upstream observation")
    relation = document.get("relation_amendment")
    if relation != {
        "from": ["observer_sha256", "probe_sha256", "schema_version", "git.revision",
                 "git.tree", "git.clean", "git.files", "git.sha256"],
        "to": ["observer_sha256", "probe_sha256", "schema_version", "git.clean", "git.files"],
        "retain_per_run_provenance": ["git.revision", "git.tree", "git.sha256"],
    }:
        raise RuntimeError("V5 changes more than cross-run verifier equivalence")
    required = (
        "upstream_baseline_remains_immutable", "v4_contract_inherited_unchanged",
        "frozen_before_v5_relation_implementation", "frozen_before_tgrad_observation",
        "product_candidate_forbidden_before_tgrad_observation",
    )
    if any(document.get(key) is not True for key in required):
        raise RuntimeError("V5 weakens inheritance or chronology")
    return document


if __name__ == "__main__":
    result = verify()
    print(f"checked {result['trial_id']}")

#!/usr/bin/env python3
"""Check V6's prospective observer-identity split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "fixtures/requirements/broadcast_add_prospective_v6_identity_split.json"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(path: Path = PATH) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("unsupported V6 identity-split schema")
    v5 = REPO / document.get("v5_amendment_path", "")
    if not v5.is_file() or digest(v5.read_bytes()) != document.get("v5_amendment_sha256"):
        raise RuntimeError("V6 does not bind the exact refuted V5 relation")
    if document.get("v5_disposition") != "relation_definition_refuted_before_implementation":
        raise RuntimeError("V5 disposition must remain explicit")
    if document.get("identity_split") != {
        "subject_protocol_equivalence": ["probe_sha256", "schema_version",
                                           "manifest_effective_sha256", "semantic_lock_sha256"],
        "per_run_provenance_only": ["observer_sha256", "git.revision", "git.tree",
                                     "git.files", "git.sha256"],
        "relation_module": "scripts/spec/broadcast_add_relation.py",
        "relation_tests": "scripts/spec/test_broadcast_add_relation.py",
    }:
        raise RuntimeError("V6 identity split changed")
    if len(document.get("relation_fault_model", [])) != 6:
        raise RuntimeError("V6 relation fault model is incomplete")
    required = (
        "upstream_baseline_remains_immutable", "v4_observation_contract_inherited_unchanged",
        "frozen_before_relation_module_implementation", "frozen_before_tgrad_observation",
        "product_candidate_forbidden_before_tgrad_observation",
    )
    if any(document.get(key) is not True for key in required):
        raise RuntimeError("V6 weakens inheritance or chronology")
    return document


if __name__ == "__main__":
    result = verify()
    print(f"checked {result['trial_id']}")

#!/usr/bin/env python3
"""Pure cross-run identity relation for broadcast-add observations."""
from __future__ import annotations

import re

SHA256 = re.compile(r"[0-9a-f]{64}")


def subject_protocol_identity(document: dict) -> dict:
    identity = document.get("identity", {})
    verifier = identity.get("verifier", {})
    manifest = identity.get("manifest", {})
    lock = identity.get("lock", {})
    result = {
        "probe_sha256": verifier.get("probe_sha256"),
        "schema_version": verifier.get("schema_version"),
        "manifest_effective_sha256": manifest.get("effective_sha256"),
        "semantic_lock_sha256": lock.get("semantic_lock_sha256"),
    }
    for name in ("probe_sha256", "manifest_effective_sha256", "semantic_lock_sha256"):
        if not isinstance(result[name], str) or not SHA256.fullmatch(result[name]):
            raise RuntimeError(f"missing or invalid subject-protocol identity: {name}")
    if not isinstance(result["schema_version"], int) or result["schema_version"] < 1:
        raise RuntimeError("missing or invalid subject-protocol identity: schema_version")
    return result


def equivalent(reference: dict, candidate: dict) -> bool:
    return subject_protocol_identity(reference) == subject_protocol_identity(candidate)

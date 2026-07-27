#!/usr/bin/env python3
"""Load the independently reviewed, versioned release-gate contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "scripts" / "evidence" / "release_inventory.json"
HASH_CONTRACT = ROOT / "scripts" / "evidence" / "hash_contract.json"
# Changing the release denominator requires an explicit reviewed code change,
# not merely editing the data file or the gate runner's arrays.
REVIEWED_INVENTORY_SHA256 = "0a44ea4eaeca5169e79a68e9d9aec4c7e6c7597d90498dc680c86282576e3b9a"
REVIEWED_HASH_CONTRACT_SHA256 = "b99286b4a8eed0908b03525a9aed85dfa040381ce7594e3608cf6753d58dd4af"


class ContractError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict[str, Any]:
    actual = file_sha256(INVENTORY)
    if actual != REVIEWED_INVENTORY_SHA256:
        raise ContractError(
            f"release inventory digest {actual} is not reviewed {REVIEWED_INVENTORY_SHA256}"
        )
    try:
        value = json.loads(INVENTORY.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot parse release inventory: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ContractError("unsupported release inventory schema")
    gates = value.get("gates")
    if not isinstance(gates, list) or len(gates) != 37:
        raise ContractError("release inventory must contain exactly 37 gates")
    names = [gate.get("name") for gate in gates if isinstance(gate, dict)]
    if len(names) != 37 or len(set(names)) != 37 or any(not name for name in names):
        raise ContractError("release gate names are malformed or duplicated")
    seen: set[str] = set()
    for gate in gates:
        if set(gate) != {"name", "writer", "depends_on"}:
            raise ContractError(f"{gate.get('name')}: unexpected gate contract keys")
        expected_writer = f"scripts/gates/{gate['name']}.sh"
        if gate["writer"] != expected_writer:
            raise ContractError(f"{gate['name']}: writer path mismatch")
        dependencies = gate["depends_on"]
        if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
            raise ContractError(f"{gate['name']}: malformed dependencies")
        if any(dependency not in seen for dependency in dependencies):
            raise ContractError(f"{gate['name']}: dependency is absent or not earlier")
        seen.add(gate["name"])
    prerequisite = value.get("performance_prerequisite")
    if not isinstance(prerequisite, dict) or set(prerequisite) != {
        "id", "snapshot_path", "required_state", "same_source",
        "variance_model_path", "decision_rule_path"
    }:
        raise ContractError("performance prerequisite is malformed")
    for key in ("snapshot_path", "variance_model_path", "decision_rule_path"):
        path = prerequisite[key]
        if not isinstance(path, str) or not path or path.startswith("/") or \
           ".." in Path(path).parts:
            raise ContractError(f"performance prerequisite {key} is unsafe")
    if prerequisite["id"] != "prepared-runtime-repeatability-v1" or \
       prerequisite["required_state"] != "promoted" or \
       prerequisite["same_source"] is not True or \
       not prerequisite["snapshot_path"].startswith("performance/"):
        raise ContractError("performance prerequisite identity/policy mismatch")
    return value


def gate_names(contract: dict[str, Any]) -> list[str]:
    return [gate["name"] for gate in contract["gates"]]


def gate_definition(contract: dict[str, Any], name: str) -> dict[str, Any]:
    for gate in contract["gates"]:
        if gate["name"] == name:
            return gate
    raise ContractError(f"unknown release gate {name}")


def load_hash_contract(release_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    actual = file_sha256(HASH_CONTRACT)
    if actual != REVIEWED_HASH_CONTRACT_SHA256:
        raise ContractError(
            f"hash contract digest {actual} is not reviewed {REVIEWED_HASH_CONTRACT_SHA256}"
        )
    try:
        value = json.loads(HASH_CONTRACT.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot parse hash contract: {error}") from error
    if not isinstance(value, dict) or set(value) != {"schema_version", "contract_id", "gates"}:
        raise ContractError("hash contract root keys are not exact")
    if value["schema_version"] != 1 or value["contract_id"] != "tgrad-evidence-hashes-v1":
        raise ContractError("unsupported hash contract identity")
    release = release_contract or load_contract()
    expected_gates = set(gate_names(release))
    policies = value["gates"]
    if not isinstance(policies, dict) or set(policies) != expected_gates:
        raise ContractError("hash contract does not exactly cover release gates")
    for gate, claims in policies.items():
        if not isinstance(claims, dict) or not claims:
            raise ContractError(f"{gate}: hash claims must be a non-empty object")
        for key, locator in claims.items():
            if not isinstance(key, str) or not key or not isinstance(locator, str):
                raise ContractError(f"{gate}: malformed hash claim")
            if locator.startswith("artifact:"):
                target = locator[len("artifact:"):]
                if not target or target.startswith("/") or ".." in Path(target).parts:
                    raise ContractError(f"{gate}:{key}: unsafe artifact locator")
                continue
            if not locator.startswith(("source:", "evidence:")):
                raise ContractError(f"{gate}:{key}: unsupported locator policy")
            target = locator.split(":", 1)[1]
            if not target or target.startswith("/") or ".." in Path(target).parts:
                raise ContractError(f"{gate}:{key}: unsafe locator target")
            if locator.startswith("evidence:") and target not in expected_gates:
                raise ContractError(f"{gate}:{key}: unknown child evidence")
    return value


def evidence_document_problems(
    hash_contract: dict[str, Any], gate: str, document: Any, source_commit: str
) -> list[str]:
    if not isinstance(document, dict):
        return ["document root is not an object"]
    problems: list[str] = []
    required = {"gate", "ts_utc", "platform", "commit", "hashes"}
    missing = sorted(required - set(document))
    if missing:
        problems.append(f"missing common fields: {missing}")
    host_keys = {key for key in ("host", "host_profile") if key in document}
    if len(host_keys) != 1:
        problems.append("exactly one of host or host_profile is required")
    if document.get("gate") != gate:
        problems.append("gate field mismatch")
    if document.get("commit") != source_commit:
        problems.append("source commit mismatch")
    for key in ("ts_utc", "platform", *host_keys):
        if not isinstance(document.get(key), str) or not document[key]:
            problems.append(f"{key} must be a non-empty string")
    timestamp = document.get("ts_utc")
    if isinstance(timestamp, str) and not timestamp.endswith("Z"):
        problems.append("ts_utc is not UTC-Z formatted")
    hashes = document.get("hashes")
    expected = hash_contract.get("gates", {}).get(gate)
    if not isinstance(hashes, dict):
        problems.append("hashes is not an object")
    elif not isinstance(expected, dict) or set(hashes) != set(expected):
        problems.append(
            f"hash keys mismatch: missing={sorted(set(expected or {})-set(hashes))}, "
            f"extra={sorted(set(hashes)-set(expected or {}))}"
        )
    return problems

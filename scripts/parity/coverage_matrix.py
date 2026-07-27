#!/usr/bin/env python3
"""Build and validate total, identity-bound Tgrad parity coverage matrices.

The matrix is intentionally not a percentage calculator.  It proves that the
generated denominator is total, records every gap, and permits a conformance
claim only when every required row has independent, calibrated evidence bound
to one immutable subject/profile/verifier/adapter/environment/oracle tuple.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.parity.coverage_model import (
        CoverageModelError,
        PROFILE_IDS,
        attach_content_sha256,
        build_requirement_inventory,
        canonical_sha256,
        content_body,
        file_sha256,
        load_requirement_inventory,
        verify_content_sha256,
    )
except ModuleNotFoundError:  # Direct execution from scripts/parity.
    from coverage_model import (  # type: ignore[no-redef]
        CoverageModelError,
        PROFILE_IDS,
        attach_content_sha256,
        build_requirement_inventory,
        canonical_sha256,
        content_body,
        file_sha256,
        load_requirement_inventory,
        verify_content_sha256,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = ROOT / "fixtures" / "parity" / "upstream_19c4d736f2bc.json"
DEFAULT_REQUIREMENTS = (
    ROOT / "fixtures" / "parity" / "requirements_19c4d736f2bc.json"
)
MATRIX_SCHEMA_VERSION = 2
CONFIRMED = "confirmed"
UNKNOWN = "unknown"
DISPOSITIONS = {"unclassified", "required", "not_applicable", "excluded"}
OBSERVATION_OUTCOMES = {"pass", "fail", "error", "blocked"}
ORIGINS = {
    "upstream_suite",
    "upstream_runtime",
    "independent_framework",
    "mathematical_law",
    "internal_differential",
    "self_referential",
}


class CoverageMatrixError(RuntimeError):
    pass


def _require_string(record: dict[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise CoverageMatrixError(f"{label}.{key}: expected non-empty string")
    return value


def _require_sha256(record: dict[str, Any], key: str, label: str) -> str:
    value = _require_string(record, key, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CoverageMatrixError(f"{label}.{key}: expected lowercase SHA-256")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageMatrixError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CoverageMatrixError(f"{label}: root must be an object")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CoverageMatrixError(
            f"cannot inspect Git repository {repository}: {error}"
        ) from error
    return completed.stdout.strip()


def git_identity(repository: Path) -> dict[str, Any]:
    resolved = repository.resolve()
    commit = _git(resolved, "rev-parse", "HEAD")
    tree = _git(resolved, "rev-parse", "HEAD^{tree}")
    status = _git(resolved, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise CoverageMatrixError(
            f"coverage subject/verifier must be clean; {resolved} has uncommitted state"
        )
    return {
        "state": CONFIRMED,
        "repository": str(resolved),
        "commit": commit,
        "tree": tree,
        "dirty": False,
    }


def unknown_identity(missing: str, resolve_by: str) -> dict[str, Any]:
    return {"state": UNKNOWN, "missing": missing, "resolve_by": resolve_by}


def file_identity(identifier: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise CoverageMatrixError(f"identity input does not exist: {resolved}")
    return {
        "state": CONFIRMED,
        "id": identifier,
        "path": str(resolved),
        "sha256": file_sha256(resolved),
    }


def load_profile_definition(
    identifier: str, path: Path, backend_names: list[str]
) -> dict[str, Any]:
    definition = _load_json(path, "profile definition")
    try:
        verify_content_sha256(definition, "profile definition")
    except CoverageModelError as error:
        raise CoverageMatrixError(str(error)) from error
    if definition.get("schema_version") != 1:
        raise CoverageMatrixError("profile definition: unsupported schema_version")
    if definition.get("kind") != "tgrad-parity-profile":
        raise CoverageMatrixError("profile definition: unexpected kind")
    if definition.get("profile") != identifier:
        raise CoverageMatrixError("profile definition: profile/id mismatch")
    environments = _require_string_list(
        definition.get("environment_ids"),
        "profile definition.environment_ids",
        nonempty=True,
    )
    if identifier in ("semanticCore", "publicApi", "ecosystem"):
        if environments != ["host"]:
            raise CoverageMatrixError(
                f"profile {identifier} must use exactly ['host']"
            )
    elif identifier == "metal":
        if environments != ["metal"]:
            raise CoverageMatrixError("profile metal must use exactly ['metal']")
    elif identifier == "allBackends":
        if environments != backend_names:
            raise CoverageMatrixError(
                "profile allBackends must preserve the exact generated backend order"
            )
    elif identifier == "portable":
        unknown = sorted(set(environments) - set(backend_names))
        if unknown:
            raise CoverageMatrixError(
                f"profile portable names undeclared backends: {unknown}"
            )
        if "cpu" not in environments or len(environments) < 2:
            raise CoverageMatrixError(
                "profile portable requires cpu plus at least one declared accelerator"
            )
    identity = file_identity(identifier, path)
    identity["environment_ids"] = environments
    identity["definition_content_sha256"] = definition["content_sha256"]
    return identity


def _identity_from_optional(
    identifier: str | None,
    path: Path | None,
    missing: str,
    resolve_by: str,
) -> dict[str, Any]:
    if identifier is None and path is None:
        return unknown_identity(missing, resolve_by)
    if identifier is None or path is None:
        raise CoverageMatrixError("identity id and path must be supplied together")
    return file_identity(identifier, path)


def initial_cell(requirement: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in requirement["profiles"]:
        return {
            "requirement_id": requirement["id"],
            "disposition": "not_applicable",
            "rationale": f"requirement does not belong to profile {profile}",
            "classification": "generated-profile-membership",
        }
    if requirement["kind"] == "test":
        return {
            "requirement_id": requirement["id"],
            "disposition": "unclassified",
            "rationale": "awaiting pre-score public-contract classification",
            "classification": "unresolved",
        }
    return {
        "requirement_id": requirement["id"],
        "disposition": "required",
        "rationale": "generated symbol requirement belongs to the selected profile",
        "classification": "generated-profile-membership",
    }


def required_environments(
    requirement: dict[str, Any], profile_environments: list[str]
) -> list[str]:
    selector = requirement.get("environment_selector")
    if not isinstance(selector, dict):
        raise CoverageMatrixError(
            f"requirement {requirement.get('id')}: missing environment_selector"
        )
    kind = selector.get("kind")
    if kind == "profile":
        return profile_environments
    if kind == "specific":
        selected = _require_string_list(
            selector.get("ids"),
            f"requirement {requirement.get('id')}.environment_selector.ids",
            nonempty=True,
        )
        outside_profile = sorted(set(selected) - set(profile_environments))
        if outside_profile:
            raise CoverageMatrixError(
                f"requirement {requirement.get('id')}: required environments "
                f"outside selected profile: {outside_profile}"
            )
        return selected
    raise CoverageMatrixError(
        f"requirement {requirement.get('id')}: unknown environment selector {kind!r}"
    )


def obligation_for(
    requirement: dict[str, Any], dimension: str, environment_id: str
) -> dict[str, Any]:
    body = {
        "requirement_id": requirement["id"],
        "requirement_identity_sha256": requirement["identity_sha256"],
        "dimension": dimension,
        "environment_id": environment_id,
        "equivalence_relation": requirement["equivalence_relation"],
    }
    return {"id": f"obligation:{canonical_sha256(body)}", **body}


def expected_obligations(
    cells: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    profile_environments: list[str],
) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    for cell, requirement in zip(cells, requirements, strict=True):
        if cell.get("requirement_id") != requirement["id"]:
            raise CoverageMatrixError("cannot derive obligations from misordered cells")
        if cell.get("disposition") != "required":
            continue
        environments = required_environments(requirement, profile_environments)
        for dimension in requirement["dimensions"]:
            for environment_id in environments:
                obligations.append(obligation_for(requirement, dimension, environment_id))
    return obligations


def reconcile_obligations(
    matrix: dict[str, Any], requirements: dict[str, Any]
) -> dict[str, Any]:
    result = dict(matrix)
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise CoverageMatrixError("cells: expected list")
    profile = result.get("profile")
    if not isinstance(profile, dict):
        raise CoverageMatrixError("profile: expected identity object")
    environments = _require_string_list(
        profile.get("environment_ids"), "profile.environment_ids", nonempty=True
    )
    result["obligations"] = expected_obligations(
        cells, requirements["requirements"], environments
    )
    # Applicability changes invalidate prior observation joins.  The caller
    # must import them again under the new immutable obligation identities.
    result["observations"] = []
    result["evidence"] = []
    result.pop("content_sha256", None)
    return attach_content_sha256(result)


def initialize_matrix(
    requirements: dict[str, Any],
    subject_repository: Path,
    verifier_repository: Path,
    profile: str,
    profile_definition: Path,
    *,
    classification_id: str | None = None,
    classification_path: Path | None = None,
    adapter_id: str | None = None,
    adapter_path: Path | None = None,
    relation_id: str | None = None,
    relation_path: Path | None = None,
    environment_id: str | None = None,
    environment_path: Path | None = None,
    oracle_id: str | None = None,
    oracle_path: Path | None = None,
) -> dict[str, Any]:
    if profile not in PROFILE_IDS:
        raise CoverageMatrixError(
            f"unknown profile {profile!r}; expected one of {', '.join(PROFILE_IDS)}"
        )
    rows = requirements["requirements"]
    backend_names = [
        row["required_environments"][0]
        for row in rows
        if row["category"] == "backend"
    ]
    profile_identity = load_profile_definition(
        profile, profile_definition, backend_names
    )
    cells = [initial_cell(requirement, profile) for requirement in rows]
    document: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "kind": "tgrad-parity-coverage-matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            **requirements["target"],
            "requirement_inventory_content_sha256": requirements["content_sha256"],
            "requirement_ids_sha256": requirements["requirement_ids_sha256"],
            "requirement_count": requirements["requirement_count"],
        },
        "subject": git_identity(subject_repository),
        "verifier": git_identity(verifier_repository),
        "profile": profile_identity,
        "classification": _identity_from_optional(
            classification_id,
            classification_path,
            "reviewed test applicability classification",
            "complete oracle.classify-public-contract before importing observations",
        ),
        "adapter": _identity_from_optional(
            adapter_id,
            adapter_path,
            "audited Tgrad upstream-suite adapter",
            "complete parity.import-test-contract without numerical fallback behavior",
        ),
        "relation_registry": _identity_from_optional(
            relation_id,
            relation_path,
            "versioned equivalence-relation registry",
            "bind each requirement relation to an immutable reviewed definition",
        ),
        "environment": _identity_from_optional(
            environment_id,
            environment_path,
            "reproducible execution-environment manifest",
            "record Python, dependencies, OS, hardware, backend, and runtime selectors",
        ),
        "oracle_contract": _identity_from_optional(
            oracle_id,
            oracle_path,
            "calibrated foreign-or-independent oracle contract",
            "bind validators and calibration mutants before promoting pass cells",
        ),
        "cells": cells,
        "obligations": expected_obligations(
            cells, rows, profile_identity["environment_ids"]
        ),
        "observations": [],
        "evidence": [],
        "validators": [],
        "calibrations": [],
    }
    return attach_content_sha256(document)


def _validate_identity(identity: Any, label: str, *, git: bool = False) -> bool:
    if not isinstance(identity, dict):
        raise CoverageMatrixError(f"{label}: expected identity object")
    state = identity.get("state")
    if state == UNKNOWN:
        _require_string(identity, "missing", label)
        _require_string(identity, "resolve_by", label)
        return False
    if state != CONFIRMED:
        raise CoverageMatrixError(f"{label}.state: expected confirmed or unknown")
    if git:
        _require_string(identity, "repository", label)
        _require_string(identity, "commit", label)
        _require_string(identity, "tree", label)
        if identity.get("dirty") is not False:
            raise CoverageMatrixError(f"{label}: dirty subject is not attributable")
    else:
        _require_string(identity, "id", label)
        _require_string(identity, "path", label)
        _require_sha256(identity, "sha256", label)
    return True


def _require_string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise CoverageMatrixError(f"{label}: expected string list")
    if nonempty and not value:
        raise CoverageMatrixError(f"{label}: must not be empty")
    if len(value) != len(set(value)):
        raise CoverageMatrixError(f"{label}: duplicate values")
    return value


def _validate_calibrations(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = matrix.get("calibrations")
    if not isinstance(values, list):
        raise CoverageMatrixError("calibrations: expected list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        label = f"calibrations[{index}]"
        if not isinstance(value, dict):
            raise CoverageMatrixError(f"{label}: expected object")
        identifier = _require_string(value, "id", label)
        if identifier in indexed:
            raise CoverageMatrixError(f"duplicate calibration id: {identifier}")
        if _require_string(value, "verifier_tree", label) != matrix["verifier"].get("tree"):
            raise CoverageMatrixError(f"{label}: verifier tree mismatch")
        _require_sha256(value, "validator_definition_sha256", label)
        _require_string(value, "mutant_tree", label)
        _require_string(value, "fault_model", label)
        _require_string(value, "dimension", label)
        _require_string(value, "environment_id", label)
        _require_string(value, "equivalence_relation", label)
        for identity_name, field_name in (
            ("adapter", "adapter_sha256"),
            ("relation_registry", "relation_registry_sha256"),
            ("environment", "environment_sha256"),
            ("oracle_contract", "oracle_contract_sha256"),
        ):
            expected = _confirmed_sha(matrix, identity_name)
            if expected is None or value.get(field_name) != expected:
                raise CoverageMatrixError(
                    f"{label}.{field_name}: calibration identity mismatch"
                )
        _require_sha256(value, "scenario_manifest_sha256", label)
        _require_sha256(value, "artifact_sha256", label)
        if value.get("outcome") not in ("validator_rejected_mutant", "mutant_survived", "indeterminate"):
            raise CoverageMatrixError(f"{label}.outcome: invalid calibration outcome")
        indexed[identifier] = value
    return indexed


def _validate_validators(
    matrix: dict[str, Any], calibrations: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    values = matrix.get("validators")
    if not isinstance(values, list):
        raise CoverageMatrixError("validators: expected list")
    verifier_tree = matrix["verifier"].get("tree")
    indexed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        label = f"validators[{index}]"
        if not isinstance(value, dict):
            raise CoverageMatrixError(f"{label}: expected object")
        identifier = _require_string(value, "id", label)
        if identifier in indexed:
            raise CoverageMatrixError(f"duplicate validator id: {identifier}")
        if _require_string(value, "verifier_tree", label) != verifier_tree:
            raise CoverageMatrixError(f"{label}: verifier tree mismatch")
        definition_sha256 = _require_sha256(value, "definition_sha256", label)
        dimensions = _require_string_list(value.get("dimensions"), f"{label}.dimensions", nonempty=True)
        calibration_ids = _require_string_list(
            value.get("calibration_ids"), f"{label}.calibration_ids", nonempty=True
        )
        if any(identifier not in calibrations for identifier in calibration_ids):
            raise CoverageMatrixError(f"{label}: unknown calibration id")
        if any(
            calibrations[identifier]["outcome"] != "validator_rejected_mutant"
            for identifier in calibration_ids
        ):
            raise CoverageMatrixError(f"{label}: calibration did not reject every mutant")
        if any(
            calibrations[calibration_id]["validator_definition_sha256"]
            != definition_sha256
            for calibration_id in calibration_ids
        ):
            raise CoverageMatrixError(f"{label}: calibration/validator definition mismatch")
        if not dimensions:
            raise CoverageMatrixError(f"{label}: validator has no dimensions")
        indexed[identifier] = value
    return indexed


def _confirmed_sha(matrix: dict[str, Any], identity_name: str) -> str | None:
    identity = matrix[identity_name]
    return identity.get("sha256") if identity.get("state") == CONFIRMED else None


def _validate_evidence(
    matrix: dict[str, Any],
    obligation_by_id: dict[str, dict[str, Any]],
    validators: dict[str, dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values = matrix.get("evidence")
    if not isinstance(values, list):
        raise CoverageMatrixError("evidence: expected list")
    expected_bindings = {
        "upstream_revision": matrix["target"].get("upstream_ref"),
        "subject_tree": matrix["subject"].get("tree"),
        "verifier_tree": matrix["verifier"].get("tree"),
        "adapter_sha256": _confirmed_sha(matrix, "adapter"),
        "classification_sha256": _confirmed_sha(matrix, "classification"),
        "relation_registry_sha256": _confirmed_sha(matrix, "relation_registry"),
        "environment_sha256": _confirmed_sha(matrix, "environment"),
        "oracle_contract_sha256": _confirmed_sha(matrix, "oracle_contract"),
    }
    indexed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        label = f"evidence[{index}]"
        if not isinstance(value, dict):
            raise CoverageMatrixError(f"{label}: expected object")
        identifier = _require_string(value, "id", label)
        if identifier in indexed:
            raise CoverageMatrixError(f"duplicate evidence id: {identifier}")
        origin = _require_string(value, "origin", label)
        if origin not in ORIGINS:
            raise CoverageMatrixError(f"{label}.origin: unknown origin")
        _require_string(value, "kind", label)
        obligation_id = _require_string(value, "obligation_id", label)
        if obligation_id not in obligation_by_id:
            raise CoverageMatrixError(f"{label}: unknown obligation id")
        obligation = obligation_by_id[obligation_id]
        for key in (
            "requirement_id",
            "dimension",
            "environment_id",
            "equivalence_relation",
        ):
            if value.get(key) != obligation[key]:
                raise CoverageMatrixError(f"{label}.{key}: obligation binding mismatch")
        for key, expected in expected_bindings.items():
            actual = value.get(key)
            if expected is None:
                raise CoverageMatrixError(
                    f"{label}: evidence exists while matrix identity {key} is unknown"
                )
            if actual != expected:
                raise CoverageMatrixError(
                    f"{label}.{key}: {actual!r} does not match matrix identity {expected!r}"
                )
        outcome = _require_string(value, "outcome", label)
        if outcome not in OBSERVATION_OUTCOMES:
            raise CoverageMatrixError(f"{label}.outcome: invalid observation outcome")
        _require_sha256(value, "scenario_manifest_sha256", label)
        _require_sha256(value, "artifact_sha256", label)
        artifact_size = value.get("artifact_size_bytes")
        if not isinstance(artifact_size, int) or artifact_size <= 0:
            raise CoverageMatrixError(f"{label}.artifact_size_bytes: expected positive int")
        oracle_tree = _require_string(value, "oracle_tree", label)
        _require_sha256(value, "independence_basis_sha256", label)
        if outcome == "pass" and origin == "self_referential":
            raise CoverageMatrixError(f"{label}: self-referential evidence cannot pass")
        if outcome == "pass" and oracle_tree == matrix["subject"].get("tree"):
            raise CoverageMatrixError(f"{label}: oracle tree equals subject tree")
        validator_id = _require_string(value, "validator_id", label)
        if validator_id not in validators:
            raise CoverageMatrixError(f"{label}: unknown validator {validator_id}")
        if obligation["dimension"] not in validators[validator_id]["dimensions"]:
            raise CoverageMatrixError(
                f"{label}: validator does not declare obligation dimension"
            )
        calibration_ids = _require_string_list(
            value.get("calibration_ids"), f"{label}.calibration_ids", nonempty=True
        )
        if calibration_ids != validators[validator_id]["calibration_ids"]:
            raise CoverageMatrixError(f"{label}: calibration set differs from validator")
        if outcome == "pass":
            matching_calibrations = [
                calibrations[calibration_id]
                for calibration_id in calibration_ids
                if calibrations[calibration_id]["dimension"] == obligation["dimension"]
                and calibrations[calibration_id]["environment_id"]
                == obligation["environment_id"]
                and calibrations[calibration_id]["equivalence_relation"]
                == obligation["equivalence_relation"]
            ]
            if not matching_calibrations:
                raise CoverageMatrixError(
                    f"{label}: no rejected mutant calibrates this exact obligation"
                )
        indexed[identifier] = value
    return indexed


def _validate_observations(
    matrix: dict[str, Any],
    obligation_by_id: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values = matrix.get("observations")
    if not isinstance(values, list):
        raise CoverageMatrixError("observations: expected list")
    indexed: dict[str, dict[str, Any]] = {}
    seen_obligations: set[str] = set()
    for index, value in enumerate(values):
        label = f"observations[{index}]"
        if not isinstance(value, dict):
            raise CoverageMatrixError(f"{label}: expected object")
        identifier = _require_string(value, "id", label)
        if identifier in indexed:
            raise CoverageMatrixError(f"duplicate observation id: {identifier}")
        obligation_id = _require_string(value, "obligation_id", label)
        if obligation_id not in obligation_by_id:
            raise CoverageMatrixError(f"{label}: unknown obligation id")
        if obligation_id in seen_obligations:
            raise CoverageMatrixError(
                f"{label}: more than one observation selects obligation {obligation_id}"
            )
        outcome = _require_string(value, "outcome", label)
        if outcome not in OBSERVATION_OUTCOMES:
            raise CoverageMatrixError(f"{label}.outcome: invalid value")
        evidence_id = _require_string(value, "evidence_id", label)
        if evidence_id not in evidence:
            raise CoverageMatrixError(f"{label}: unknown evidence id {evidence_id}")
        record = evidence[evidence_id]
        if record["obligation_id"] != obligation_id or record["outcome"] != outcome:
            raise CoverageMatrixError(f"{label}: evidence/observation binding mismatch")
        if outcome != "pass":
            _require_string(value, "diagnostic", label)
        indexed[identifier] = value
        seen_obligations.add(obligation_id)
    return indexed


def validate_matrix(
    matrix: dict[str, Any], requirements: dict[str, Any]
) -> dict[str, Any]:
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise CoverageMatrixError("unsupported coverage matrix schema_version")
    if matrix.get("kind") != "tgrad-parity-coverage-matrix":
        raise CoverageMatrixError("unexpected coverage matrix kind")
    try:
        verify_content_sha256(matrix, "coverage matrix")
    except CoverageModelError as error:
        raise CoverageMatrixError(str(error)) from error
    target = matrix.get("target")
    if not isinstance(target, dict):
        raise CoverageMatrixError("target: expected object")
    for key, expected in (
        ("upstream_ref", requirements["target"]["upstream_ref"]),
        ("source_manifest_content_sha256", requirements["target"]["source_manifest_content_sha256"]),
        ("requirement_inventory_content_sha256", requirements["content_sha256"]),
        ("requirement_ids_sha256", requirements["requirement_ids_sha256"]),
        ("requirement_count", requirements["requirement_count"]),
    ):
        if target.get(key) != expected:
            raise CoverageMatrixError(f"target.{key}: matrix/requirement inventory mismatch")

    subject_confirmed = _validate_identity(matrix.get("subject"), "subject", git=True)
    verifier_confirmed = _validate_identity(matrix.get("verifier"), "verifier", git=True)
    profile_confirmed = _validate_identity(matrix.get("profile"), "profile")
    optional_identity_names = (
        "classification",
        "adapter",
        "relation_registry",
        "environment",
        "oracle_contract",
    )
    optional_confirmed = {
        name: _validate_identity(matrix.get(name), name)
        for name in optional_identity_names
    }
    if not subject_confirmed or not verifier_confirmed or not profile_confirmed:
        raise CoverageMatrixError("subject, verifier, and profile identities must be confirmed")
    if matrix["profile"].get("id") not in PROFILE_IDS:
        raise CoverageMatrixError("profile.id is not a declared Tgrad parity profile")

    rows = requirements["requirements"]
    requirement_by_id = {row["id"]: row for row in rows}
    expected_ids = [row["id"] for row in rows]
    cells = matrix.get("cells")
    if not isinstance(cells, list):
        raise CoverageMatrixError("cells: expected list")
    cell_ids = [cell.get("requirement_id") for cell in cells if isinstance(cell, dict)]
    if len(cell_ids) != len(cells):
        raise CoverageMatrixError("cells: malformed row")
    if cell_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(cell_ids))
        extra = sorted(set(cell_ids) - set(expected_ids))
        duplicate_count = len(cell_ids) - len(set(cell_ids))
        raise CoverageMatrixError(
            "cells are not an exact ordered total cover of requirements: "
            f"missing={missing[:5]}, extra={extra[:5]}, duplicates={duplicate_count}"
        )

    profile_id = matrix["profile"]["id"]
    profile_environments = _require_string_list(
        matrix["profile"].get("environment_ids"),
        "profile.environment_ids",
        nonempty=True,
    )
    counts: Counter[str] = Counter()
    for index, cell in enumerate(cells):
        label = f"cells[{index}]"
        requirement = requirement_by_id[cell_ids[index]]
        disposition = cell.get("disposition")
        forbidden_authored_state = sorted(set(cell) & {"result", "evidence_ids", "state"})
        if forbidden_authored_state:
            raise CoverageMatrixError(
                f"{label}: coverage state is derived, not authored; remove "
                f"{forbidden_authored_state}"
            )
        if disposition not in DISPOSITIONS:
            raise CoverageMatrixError(f"{label}.disposition: invalid value")
        _require_string(cell, "rationale", label)
        classification = _require_string(cell, "classification", label)
        in_profile = profile_id in requirement["profiles"]
        if not in_profile and disposition != "not_applicable":
            raise CoverageMatrixError(f"{label}: out-of-profile row must be not_applicable")
        if in_profile and disposition == "not_applicable":
            raise CoverageMatrixError(
                f"{label}: in-profile row requires reviewed exclusion, not not_applicable"
            )
        if disposition == "unclassified" and classification != "unresolved":
            raise CoverageMatrixError(f"{label}: unclassified row must remain unresolved")
        if disposition == "excluded" and classification in (
            "unresolved",
            "generated-profile-membership",
        ):
            raise CoverageMatrixError(
                f"{label}: exclusion requires an imported classification decision"
            )
        counts[f"disposition.{disposition}"] += 1

    obligations = matrix.get("obligations")
    if not isinstance(obligations, list):
        raise CoverageMatrixError("obligations: expected list")
    expected = expected_obligations(cells, rows, profile_environments)
    if obligations != expected:
        raise CoverageMatrixError(
            "obligations are not the exact requirement × dimension × environment product; "
            "reconcile after changing applicability"
        )
    obligation_ids = [value.get("id") for value in obligations if isinstance(value, dict)]
    if len(obligation_ids) != len(obligations) or len(obligation_ids) != len(set(obligation_ids)):
        raise CoverageMatrixError("obligations: malformed or duplicate rows")
    obligation_by_id = dict(zip(obligation_ids, obligations, strict=True))

    calibrations = _validate_calibrations(matrix)
    validators = _validate_validators(matrix, calibrations)
    evidence = _validate_evidence(
        matrix, obligation_by_id, validators, calibrations
    )
    observations = _validate_observations(matrix, obligation_by_id, evidence)
    observation_by_obligation = {
        observation["obligation_id"]: observation
        for observation in observations.values()
    }
    used_evidence_ids = {
        observation["evidence_id"] for observation in observations.values()
    }
    if len(used_evidence_ids) != len(observations):
        raise CoverageMatrixError("one evidence record cannot stand for multiple obligations")
    orphan_evidence = sorted(set(evidence) - used_evidence_ids)
    if orphan_evidence:
        raise CoverageMatrixError(f"orphan evidence records: {orphan_evidence[:5]}")

    cell_results: dict[str, str] = {}
    obligation_counts: Counter[str] = Counter()
    obligations_by_requirement: dict[str, list[dict[str, Any]]] = {}
    for obligation in obligations:
        obligations_by_requirement.setdefault(obligation["requirement_id"], []).append(
            obligation
        )
        observation = observation_by_obligation.get(obligation["id"])
        obligation_counts[
            "unobserved" if observation is None else observation["outcome"]
        ] += 1
    for cell in cells:
        if cell["disposition"] != "required":
            result = "unobserved"
        else:
            requirement_obligations = obligations_by_requirement.get(
                cell["requirement_id"], []
            )
            outcomes = [
                observation_by_obligation[obligation["id"]]["outcome"]
                if obligation["id"] in observation_by_obligation
                else "unobserved"
                for obligation in requirement_obligations
            ]
            if not outcomes or "unobserved" in outcomes:
                result = "unobserved"
            elif "fail" in outcomes or "error" in outcomes:
                result = "fail"
            elif "blocked" in outcomes:
                result = "blocked"
            else:
                result = "pass"
        cell_results[cell["requirement_id"]] = result
        counts[f"result.{result}"] += 1
        counts[f"pair.{cell['disposition']}.{result}"] += 1

    all_optional_confirmed = all(optional_confirmed.values())
    unclassified = counts["disposition.unclassified"]
    required_count = counts["disposition.required"]
    required_unobserved = counts["pair.required.unobserved"]
    required_blocked = counts["pair.required.blocked"]
    required_failed = counts["pair.required.fail"]
    required_passed = counts["pair.required.pass"]
    reportable = all_optional_confirmed and unclassified == 0
    fully_observed = reportable and required_unobserved == 0 and required_blocked == 0
    conformant = fully_observed and required_count > 0 and required_failed == 0
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_content_sha256": matrix["content_sha256"],
        "requirement_count": len(expected_ids),
        "cell_count": len(cells),
        "obligation_count": len(obligations),
        "observation_count": len(observations),
        "obligation_counts": dict(sorted(obligation_counts.items())),
        "counts": dict(sorted(counts.items())),
        "identity_states": {
            name: matrix[name]["state"]
            for name in ("subject", "verifier", "profile", *optional_identity_names)
        },
        "structurally_total": True,
        "reportable": reportable,
        "fully_observed": fully_observed,
        "conformant": conformant,
        "claim_policy": (
            "no scalar percentage; every required row expands to dimension × "
            "environment obligations; row states are derived from atomic outcomes; "
            "conformant requires every required obligation to pass"
        ),
        "cell_results": cell_results,
        "required_counts": {
            "required": required_count,
            "pass": required_passed,
            "fail": required_failed,
            "blocked": required_blocked,
            "unobserved": required_unobserved,
        },
    }


def _command_requirements(args: argparse.Namespace) -> int:
    try:
        generated = build_requirement_inventory(args.manifest)
    except (CoverageModelError, ValueError) as error:
        raise CoverageMatrixError(str(error)) from error
    if args.check:
        existing = load_requirement_inventory(args.output)
        if content_body(existing) != content_body(generated):
            raise CoverageMatrixError(f"stale requirement inventory: {args.output}")
        print(f"coverage requirements: OK — {args.output}")
        return 0
    _atomic_write_json(args.output, generated)
    print(f"wrote {args.output}")
    return 0


def _command_init(args: argparse.Namespace) -> int:
    requirements = load_requirement_inventory(args.requirements)
    matrix = initialize_matrix(
        requirements,
        args.subject_repo,
        args.verifier_repo,
        args.profile,
        args.profile_definition,
        classification_id=args.classification_id,
        classification_path=args.classification,
        adapter_id=args.adapter_id,
        adapter_path=args.adapter,
        relation_id=args.relation_registry_id,
        relation_path=args.relation_registry,
        environment_id=args.environment_id,
        environment_path=args.environment,
        oracle_id=args.oracle_contract_id,
        oracle_path=args.oracle_contract,
    )
    if args.output.exists():
        raise CoverageMatrixError(f"refusing to overwrite {args.output}")
    _atomic_write_json(args.output, matrix)
    print(f"wrote {args.output}")
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    requirements = load_requirement_inventory(args.requirements)
    matrix = _load_json(args.matrix, "coverage matrix")
    summary = validate_matrix(matrix, requirements)
    print(json.dumps(summary, sort_keys=True, indent=2))
    if args.require_reportable and not summary["reportable"]:
        raise CoverageMatrixError("matrix is structurally valid but not reportable")
    if args.require_observed and not summary["fully_observed"]:
        raise CoverageMatrixError("matrix is reportable but not fully observed")
    if args.require_conformant and not summary["conformant"]:
        raise CoverageMatrixError("matrix is not conformant")
    return 0


def _command_reconcile(args: argparse.Namespace) -> int:
    requirements = load_requirement_inventory(args.requirements)
    matrix = _load_json(args.matrix, "coverage matrix")
    reconciled = reconcile_obligations(matrix, requirements)
    if args.output.exists():
        raise CoverageMatrixError(f"refusing to overwrite {args.output}")
    _atomic_write_json(args.output, reconciled)
    print(f"wrote {args.output}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    requirements = commands.add_parser("requirements")
    requirements.add_argument("--manifest", type=Path, default=DEFAULT_TARGET)
    requirements.add_argument("--output", type=Path, default=DEFAULT_REQUIREMENTS)
    requirements.add_argument("--check", action="store_true")
    requirements.set_defaults(handler=_command_requirements)

    initialize = commands.add_parser("init")
    initialize.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    initialize.add_argument("--subject-repo", type=Path, required=True)
    initialize.add_argument("--verifier-repo", type=Path, default=ROOT)
    initialize.add_argument("--profile", required=True)
    initialize.add_argument("--profile-definition", type=Path, required=True)
    for name, destination in (
        ("classification", "classification"),
        ("adapter", "adapter"),
        ("relation-registry", "relation_registry"),
        ("environment", "environment"),
        ("oracle-contract", "oracle_contract"),
    ):
        initialize.add_argument(f"--{name}", dest=destination, type=Path)
        initialize.add_argument(f"--{name}-id", dest=f"{destination}_id")
    initialize.add_argument("--output", type=Path, required=True)
    initialize.set_defaults(handler=_command_init)

    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    reconcile.add_argument("--matrix", type=Path, required=True)
    reconcile.add_argument("--output", type=Path, required=True)
    reconcile.set_defaults(handler=_command_reconcile)

    validate = commands.add_parser("validate")
    validate.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    validate.add_argument("--matrix", type=Path, required=True)
    validate.add_argument("--require-reportable", action="store_true")
    validate.add_argument("--require-observed", action="store_true")
    validate.add_argument("--require-conformant", action="store_true")
    validate.set_defaults(handler=_command_validate)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (CoverageMatrixError, CoverageModelError) as error:
        print(f"coverage_matrix: FAILED — {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

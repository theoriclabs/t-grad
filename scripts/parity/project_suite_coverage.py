#!/usr/bin/env python3
"""Project frozen suite diagnostics onto the 590-requirement Metal profile.

This is intentionally a diagnostic projection, not a parity score.  It keeps
one ordered cell for every reviewed requirement, distinguishes collection
failures from failures after collection, and refuses promotion while the
legacy Tgrad suite fixtures lack an attributable subject tree, execution
environment, and raw diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.parity.coverage_model import (
        CoverageModelError,
        EXPECTED_CATEGORY_COUNTS,
        EXPECTED_REQUIREMENT_COUNT,
        canonical_sha256,
        load_requirement_inventory,
    )
except ModuleNotFoundError:  # Direct execution from scripts/parity.
    from coverage_model import (  # type: ignore[no-redef]
        CoverageModelError,
        EXPECTED_CATEGORY_COUNTS,
        EXPECTED_REQUIREMENT_COUNT,
        canonical_sha256,
        load_requirement_inventory,
    )


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REF = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
REFERENCE_ARTIFACT_COMMIT = "fdc741dee8ebffec424ff8845177c16931347861"
SCHEMA_VERSION = 1
PROFILE = "metal"
GROUPS = ("null", "unit", "backend")
FILE_STATUSES = ("pass", "fail", "collect_error", "timeout", "empty")
COUNTER_FIELDS = ("collected", "passed", "failed", "errors", "skipped")

EXPECTED_ORACLE_CLASS_COUNTS = {
    "api_surface": 34,
    "internal_repr": 80,
    "infrastructure": 24,
    "ambiguous": 3,
}
EXPECTED_API_BY_GROUP = {"null": 4, "unit": 18, "backend": 12}
EXPECTED_DISPOSITIONS = {"required": 471, "excluded": 104, "not_applicable": 15}
EXPECTED_UPSTREAM_STATUS_COUNTS = {
    "pass": 133,
    "fail": 1,
    "collect_error": 3,
    "timeout": 0,
    "empty": 1,
}
EXPECTED_TGRAD_STATUS_COUNTS = {
    "pass": 0,
    "fail": 5,
    "collect_error": 29,
    "timeout": 0,
    "empty": 0,
}
EXPECTED_UPSTREAM_TESTS_PASSED = 3419
EXPECTED_REQUIREMENT_CONTENT_SHA256 = (
    "1843c762a3b16e72b351bfd4f1447b05644e06253b7ab20c0593e05fb28cda9b"
)
EXPECTED_REQUIREMENT_IDS_SHA256 = (
    "00d6421732a3df2f33f6d1520626c5084a8b66a91c81a2f4386fa82d8f5041c0"
)
EXPECTED_PUBLIC_UPSTREAM_STATUS_COUNTS = {
    "pass": 34,
    "fail": 0,
    "collect_error": 0,
    "timeout": 0,
    "empty": 0,
}
EXPECTED_PUBLIC_UPSTREAM_TEST_COUNTS = {
    "passed": 1003,
    "failed": 0,
    "errors": 0,
    "skipped": 146,
}

SOURCE_FIXTURES = (
    "fixtures/parity/requirements_19c4d736f2bc.json",
    "fixtures/parity/oracle_classification.json",
    "fixtures/parity/suite_upstream_null_19c4d736f2bc.json",
    "fixtures/parity/suite_upstream_unit_19c4d736f2bc.json",
    "fixtures/parity/suite_upstream_backend_19c4d736f2bc.json",
    "fixtures/parity/suite_tgrad_null_19c4d736f2bc.json",
    "fixtures/parity/suite_tgrad_unit_19c4d736f2bc.json",
    "fixtures/parity/suite_tgrad_backend_19c4d736f2bc.json",
)
REFERENCE_BOUND_FIXTURES = frozenset(SOURCE_FIXTURES[1:])
ADAPTER_FILES = (
    "scripts/parity/shim/run_pytest.py",
    "scripts/parity/shim/sitecustomize.py",
    "scripts/parity/shim/tinygrad/__init__.py",
    "scripts/parity/shim/tinygrad/tensor.py",
)
LEAN_PROJECTION_PATH = "Tgrad/Spec/ParityCoverage.lean"

UNKNOWN_ATTRIBUTION_BLOCKERS = (
    {
        "field": "tgrad_subject_tree",
        "state": "unknown",
        "missing": "the Tgrad suite fixtures do not record the tested Tgrad commit or tree",
        "resolve_by": "rerun the strict adapter while recording the clean Tgrad subject commit and tree",
    },
    {
        "field": "verifier_tree",
        "state": "unknown",
        "missing": "the suite fixtures do not record the exact verifier commit and tree used at execution",
        "resolve_by": "rerun while recording the clean verifier commit and tree in every observation envelope",
    },
    {
        "field": "execution_environment",
        "state": "unknown",
        "missing": "the Tgrad suite fixtures do not record a normalized execution environment",
        "resolve_by": "rerun with an environment manifest covering host, Python, packages, variables, and Metal device",
    },
    {
        "field": "raw_diagnostics",
        "state": "unknown",
        "missing": "the Tgrad suite fixtures contain counters but no stdout, stderr, or JUnit identities",
        "resolve_by": "rerun with content-addressed stdout, stderr, and JUnit artifacts for every file",
    },
    {
        "field": "equivalence_relation_registry",
        "state": "unknown",
        "missing": "the historical observations are not joined to a reviewed equivalence relation",
        "resolve_by": "bind every obligation to a content-addressed relation registry",
    },
    {
        "field": "validator_calibration",
        "state": "unknown",
        "missing": "the historical observations do not identify rejected validator mutants",
        "resolve_by": "calibrate each validator/relation/environment tuple and bind its falsifier bundle",
    },
)


class ProjectionError(RuntimeError):
    """A source fixture cannot support the requested diagnostic projection."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_document_sha256(document: Mapping[str, Any]) -> str:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return canonical_sha256(body)


def _attach_content_sha256(document: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    result["content_sha256"] = _canonical_document_sha256(result)
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProjectionError(f"{label}: root must be an object")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectionError(f"{label}: expected a non-negative integer")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectionError(f"{label}: expected a non-empty string")
    return value


def _status_stage(status: str) -> str:
    if status in ("collect_error", "empty"):
        return "collection"
    if status in ("pass", "fail"):
        return "post_collection_execution"
    if status == "timeout":
        return "indeterminate"
    raise ProjectionError(f"unknown suite status {status!r}")


def _test_path(requirement: Mapping[str, Any]) -> str:
    sources = requirement.get("upstream_sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ProjectionError(
            f"{requirement.get('id')}: test requirement must have one upstream source"
        )
    source = _require_string(sources[0], f"{requirement.get('id')}.upstream_sources[0]")
    parts = source.split("/")
    if len(parts) != 3 or parts[0] != "test" or parts[1] not in GROUPS:
        raise ProjectionError(f"{requirement.get('id')}: malformed test source {source!r}")
    expected_id = f"test:{parts[1]}:{parts[2]}"
    if requirement.get("id") != expected_id:
        raise ProjectionError(
            f"{requirement.get('id')}: source implies requirement id {expected_id!r}"
        )
    return source


def validate_requirements(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the exact reviewed denominator without regenerating it."""
    if document.get("schema_version") != 2:
        raise ProjectionError("requirement inventory: unsupported schema_version")
    if document.get("kind") != "tgrad-parity-requirement-inventory":
        raise ProjectionError("requirement inventory: unexpected kind")
    if document.get("reviewed_requirement_count") != EXPECTED_REQUIREMENT_COUNT:
        raise ProjectionError("requirement inventory: reviewed count is not exactly 590")
    if document.get("requirement_count") != EXPECTED_REQUIREMENT_COUNT:
        raise ProjectionError("requirement inventory: count is not exactly 590")
    if document.get("category_counts") != EXPECTED_CATEGORY_COUNTS:
        raise ProjectionError("requirement inventory: category counts changed")
    target = document.get("target")
    if not isinstance(target, dict) or target.get("upstream_ref") != UPSTREAM_REF:
        raise ProjectionError("requirement inventory: wrong upstream target")
    rows = document.get("requirements")
    if not isinstance(rows, list) or len(rows) != EXPECTED_REQUIREMENT_COUNT:
        raise ProjectionError("requirement inventory: expected exactly 590 rows")
    identifiers: list[str] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProjectionError(f"requirements[{ordinal}]: expected an object")
        if row.get("ordinal") != ordinal:
            raise ProjectionError(f"requirements[{ordinal}]: ordinal mismatch")
        identifier = _require_string(row.get("id"), f"requirements[{ordinal}].id")
        identifiers.append(identifier)
        identity = row.get("identity_sha256")
        body = {key: value for key, value in row.items() if key != "identity_sha256"}
        if identity != canonical_sha256(body):
            raise ProjectionError(f"requirements[{ordinal}]: identity hash mismatch")
        kind = row.get("kind")
        if kind not in ("symbol", "test"):
            raise ProjectionError(f"requirements[{ordinal}]: unexpected kind {kind!r}")
        if kind == "test":
            _test_path(row)
    if len(set(identifiers)) != EXPECTED_REQUIREMENT_COUNT:
        raise ProjectionError("requirement inventory: identifiers are not unique")
    if document.get("requirement_ids_sha256") != canonical_sha256(identifiers):
        raise ProjectionError("requirement inventory: requirement_ids_sha256 mismatch")
    recorded = document.get("content_sha256")
    if recorded != _canonical_document_sha256(document):
        raise ProjectionError("requirement inventory: content_sha256 mismatch")
    return rows


def validate_oracle(
    document: Mapping[str, Any], test_requirements: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != 1:
        raise ProjectionError("oracle: unsupported schema_version")
    upstream = document.get("upstream")
    if not isinstance(upstream, dict):
        raise ProjectionError("oracle: missing upstream identity")
    if upstream.get("repository") != "tinygrad/tinygrad" or upstream.get("commit") != UPSTREAM_REF:
        raise ProjectionError("oracle: wrong upstream identity")
    if upstream.get("test_directories") != [f"test/{group}" for group in GROUPS]:
        raise ProjectionError("oracle: test directory order changed")
    policy = document.get("classification_policy")
    if not isinstance(policy, dict) or policy.get("unit") != "file":
        raise ProjectionError("oracle: classification unit must be file")
    basis = policy.get("basis")
    if not isinstance(basis, str) or "no Tgrad implementation or score was consulted" not in basis:
        raise ProjectionError("oracle: classification is not implementation-independent")

    files = document.get("files")
    if not isinstance(files, list) or len(files) != 141:
        raise ProjectionError("oracle: expected 138 tests plus three package initializers")
    by_path: dict[str, dict[str, Any]] = {}
    class_counts: Counter[str] = Counter()
    group_class_counts: dict[str, Counter[str]] = {group: Counter() for group in GROUPS}
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ProjectionError(f"oracle.files[{index}]: expected an object")
        group = entry.get("group")
        filename = entry.get("filename")
        classification = entry.get("class")
        if group not in GROUPS or not isinstance(filename, str):
            raise ProjectionError(f"oracle.files[{index}]: malformed path")
        if classification not in EXPECTED_ORACLE_CLASS_COUNTS:
            raise ProjectionError(f"oracle.files[{index}]: unknown class {classification!r}")
        path = f"test/{group}/{filename}"
        if path in by_path:
            raise ProjectionError(f"oracle: duplicate classification for {path}")
        by_path[path] = dict(entry)
        class_counts[classification] += 1
        group_class_counts[group][classification] += 1
    if dict(class_counts) != EXPECTED_ORACLE_CLASS_COUNTS:
        raise ProjectionError(
            f"oracle: class counts changed: {dict(class_counts)!r}"
        )
    api_by_group = {
        group: group_class_counts[group]["api_surface"] for group in GROUPS
    }
    if api_by_group != EXPECTED_API_BY_GROUP:
        raise ProjectionError(f"oracle: API group counts changed: {api_by_group!r}")
    computed_totals = {
        "files": len(files),
        "by_class": EXPECTED_ORACLE_CLASS_COUNTS,
        "by_group": {
            group: {
                "files": sum(group_class_counts[group].values()),
                "by_class": {
                    classification: group_class_counts[group][classification]
                    for classification in EXPECTED_ORACLE_CLASS_COUNTS
                },
            }
            for group in GROUPS
        },
    }
    if document.get("totals") != computed_totals:
        raise ProjectionError("oracle: declared totals do not match classified rows")
    ambiguous = {
        path for path, entry in by_path.items() if entry["class"] == "ambiguous"
    }
    expected_ambiguous = {f"test/{group}/__init__.py" for group in GROUPS}
    if ambiguous != expected_ambiguous:
        raise ProjectionError("oracle: only the three __init__.py files may be ambiguous")

    requirement_paths = {_test_path(row) for row in test_requirements}
    classified_test_paths = set(by_path) - expected_ambiguous
    if requirement_paths != classified_test_paths:
        missing = sorted(classified_test_paths - requirement_paths)
        extra = sorted(requirement_paths - classified_test_paths)
        raise ProjectionError(
            f"oracle/requirement test inventory mismatch: missing={missing}, extra={extra}"
        )
    if requirement_paths & expected_ambiguous:
        raise ProjectionError("ambiguous __init__.py entered the requirement denominator")
    excluded = {
        path
        for path in requirement_paths
        if by_path[path]["class"] in ("internal_repr", "infrastructure")
    }
    if len(excluded) != 104:
        raise ProjectionError(f"oracle: expected 104 excluded test files, found {len(excluded)}")
    return by_path


def _computed_aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    statuses = Counter(result["status"] for result in results)
    return {
        "files": len(results),
        "files_collect_error": statuses["collect_error"],
        "files_fail": statuses["fail"],
        "files_pass": statuses["pass"],
        "files_timeout": statuses["timeout"],
        "tests_errors": sum(int(result["errors"]) for result in results),
        "tests_failed": sum(int(result["failed"]) for result in results),
        "tests_passed": sum(int(result["passed"]) for result in results),
        "tests_skipped": sum(int(result["skipped"]) for result in results),
    }


def validate_suite(
    document: Mapping[str, Any], *, against: str, group: str
) -> dict[str, dict[str, Any]]:
    if document.get("against") != against:
        raise ProjectionError(f"suite {against}/{group}: against mismatch")
    if document.get("group") != group:
        raise ProjectionError(f"suite {against}/{group}: group mismatch")
    if document.get("upstream_ref") != UPSTREAM_REF:
        raise ProjectionError(f"suite {against}/{group}: wrong upstream_ref")
    results = document.get("results")
    if not isinstance(results, list):
        raise ProjectionError(f"suite {against}/{group}: results must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(results):
        label = f"suite {against}/{group}.results[{index}]"
        if not isinstance(raw, dict):
            raise ProjectionError(f"{label}: expected an object")
        path = _require_string(raw.get("file"), f"{label}.file")
        if not path.startswith(f"test/{group}/") or not path.endswith(".py"):
            raise ProjectionError(f"{label}: path is outside test/{group}")
        if path in by_path:
            raise ProjectionError(f"suite {against}/{group}: duplicate result for {path}")
        status = raw.get("status")
        if status not in FILE_STATUSES:
            raise ProjectionError(f"{label}: unknown status {status!r}")
        normalized = {"file": path, "status": status}
        for field in COUNTER_FIELDS:
            normalized[field] = _require_nonnegative_int(raw.get(field), f"{label}.{field}")
        if status == "pass" and (normalized["failed"] or normalized["errors"]):
            raise ProjectionError(f"{label}: pass result contains failures or errors")
        if status == "fail" and normalized["failed"] == 0:
            raise ProjectionError(f"{label}: fail result contains no failed tests")
        if status == "empty" and any(normalized[field] for field in COUNTER_FIELDS):
            raise ProjectionError(f"{label}: empty result contains test counters")
        by_path[path] = normalized
    aggregate = document.get("aggregate")
    if aggregate != _computed_aggregate(list(by_path.values())):
        raise ProjectionError(
            f"suite {against}/{group}: aggregate does not match per-file results"
        )
    return by_path


def _merge_suites(
    documents: Mapping[str, Mapping[str, Any]], *, against: str
) -> dict[str, dict[str, Any]]:
    if set(documents) != set(GROUPS):
        raise ProjectionError(f"{against} suites must contain exactly {list(GROUPS)!r}")
    merged: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        rows = validate_suite(documents[group], against=against, group=group)
        overlap = set(merged) & set(rows)
        if overlap:
            raise ProjectionError(f"{against} suites duplicate files: {sorted(overlap)}")
        merged.update(rows)
    return merged


def _status_counts(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(result["status"] for result in results)
    return {status: counts[status] for status in FILE_STATUSES}


def _test_totals(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    rows = list(results)
    return {
        field: sum(int(row[field]) for row in rows)
        for field in ("passed", "failed", "errors", "skipped")
    }


def _observation(result: Mapping[str, Any], source_fixture: str) -> dict[str, Any]:
    return {
        "state": "observed",
        "source_fixture": source_fixture,
        "status": result["status"],
        "stage": _status_stage(str(result["status"])),
        "counters": {field: int(result[field]) for field in COUNTER_FIELDS},
    }


def _suite_fixture_for(against: str, path: str) -> str:
    group = path.split("/", 2)[1]
    return f"fixtures/parity/suite_{against}_{group}_19c4d736f2bc.json"


def _disposition(
    requirement: Mapping[str, Any], oracle_by_path: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str, str]:
    if requirement.get("kind") == "symbol":
        profiles = requirement.get("profiles")
        if not isinstance(profiles, list):
            raise ProjectionError(f"{requirement.get('id')}: missing generated profiles")
        if PROFILE not in profiles:
            if requirement.get("category") != "backend":
                raise ProjectionError(
                    f"{requirement.get('id')}: non-backend symbol unexpectedly leaves Metal profile"
                )
            return (
                "not_applicable",
                "generated-metal-backend-selection",
                "the Metal profile excludes each non-Metal backend requirement",
            )
        return (
            "required",
            "generated-metal-symbol-selection",
            "the symbol requirement belongs to the Metal compatibility contract",
        )
    path = _test_path(requirement)
    classification = oracle_by_path[path]["class"]
    if classification == "api_surface":
        return (
            "required",
            "frozen-upstream-oracle",
            "the pre-score oracle classifies this file as public API behavior",
        )
    if classification in ("internal_repr", "infrastructure"):
        return (
            "excluded",
            "frozen-upstream-oracle",
            f"the pre-score oracle classifies this file as {classification}",
        )
    raise ProjectionError(f"test requirement {requirement.get('id')}: ambiguous classification")


def _project_cell(
    requirement: Mapping[str, Any],
    oracle_by_path: Mapping[str, Mapping[str, Any]],
    upstream_results: Mapping[str, Mapping[str, Any]],
    tgrad_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    disposition, origin, rationale = _disposition(requirement, oracle_by_path)
    cell: dict[str, Any] = {
        "ordinal": requirement["ordinal"],
        "requirement_id": requirement["id"],
        "requirement_identity_sha256": requirement["identity_sha256"],
        "kind": requirement["kind"],
        "category": requirement["category"],
        "disposition": disposition,
        "classification_origin": origin,
        "rationale": rationale,
    }
    if requirement["kind"] == "symbol":
        cell["oracle_class"] = "symbol"
        cell["upstream_calibration"] = {
            "state": "not_observed",
            "reason": "the calibration fixtures are test-file observations, not symbol probes",
        }
        if disposition == "required":
            cell["tgrad_diagnostic"] = {
                "state": "unknown",
                "reason": "no symbol-level Tgrad observation is present in the suite fixtures",
            }
        else:
            cell["tgrad_diagnostic"] = {
                "state": "not_applicable",
                "reason": "the requirement is outside the Metal backend profile",
            }
        return cell

    path = _test_path(requirement)
    cell["upstream_test_file"] = path
    cell["oracle_class"] = oracle_by_path[path]["class"]
    cell["upstream_calibration"] = _observation(
        upstream_results[path], _suite_fixture_for("upstream", path)
    )
    if disposition == "required":
        cell["tgrad_diagnostic"] = _observation(
            tgrad_results[path], _suite_fixture_for("tgrad", path)
        )
    else:
        cell["tgrad_diagnostic"] = {
            "state": "not_run_by_contract",
            "reason": "the frozen oracle excludes this implementation-specific test file",
        }
    return cell


def _validate_no_scalar_score(value: Any, location: str = "document") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in {"score", "percentage", "percent"}:
                raise ProjectionError(f"{location}: scalar parity key {key!r} is forbidden")
            _validate_no_scalar_score(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_no_scalar_score(nested, f"{location}[{index}]")


def _validate_projection_structure(
    document: Mapping[str, Any], *, require_source_bindings: bool
) -> None:
    """Validate projection structure before or after source binding."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ProjectionError("projection: unsupported schema_version")
    if document.get("kind") != "tgrad-parity-coverage-diagnostic":
        raise ProjectionError("projection: unexpected kind")
    if document.get("profile") != {"id": PROFILE, "backend": "metal"}:
        raise ProjectionError("projection: wrong profile")
    target = document.get("target")
    if not isinstance(target, dict):
        raise ProjectionError("projection: missing target")
    if target.get("upstream_ref") != UPSTREAM_REF:
        raise ProjectionError("projection: wrong upstream target")
    if target.get("repository") != "tinygrad/tinygrad":
        raise ProjectionError("projection: wrong upstream repository")
    if target.get("requirement_count") != EXPECTED_REQUIREMENT_COUNT:
        raise ProjectionError("projection: wrong requirement count")
    if target.get("requirement_inventory_content_sha256") != \
            EXPECTED_REQUIREMENT_CONTENT_SHA256:
        raise ProjectionError("projection: wrong requirement inventory identity")
    if target.get("requirement_ids_sha256") != EXPECTED_REQUIREMENT_IDS_SHA256:
        raise ProjectionError("projection: wrong ordered requirement identity")
    boundary = document.get("claim_boundary")
    if not isinstance(boundary, dict):
        raise ProjectionError("projection: missing claim boundary")
    if boundary.get("diagnostic_only") is not True:
        raise ProjectionError("projection must remain diagnostic-only")
    if boundary.get("promotable") is not False:
        raise ProjectionError("projection cannot be promotable with unknown attribution")
    if boundary.get("reportable_as_conformance") is not False:
        raise ProjectionError("projection cannot report conformance with unknown attribution")
    if boundary.get("subject_identity") != {
        "state": "unknown",
        "forbidden_inference": (
            "do not infer the Tgrad subject from the fixture-producing commit or its parent"
        ),
    }:
        raise ProjectionError("projection: subject identity must remain explicitly unknown")
    blockers = boundary.get("unknown_attribution_blockers")
    if blockers != list(UNKNOWN_ATTRIBUTION_BLOCKERS):
        raise ProjectionError("projection: attribution blockers changed or were dropped")

    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_REQUIREMENT_COUNT:
        raise ProjectionError("projection must contain exactly one cell per 590 requirements")
    dispositions: Counter[str] = Counter()
    identifiers: set[str] = set()
    tgrad_statuses: Counter[str] = Counter()
    required_test_count = 0
    for ordinal, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("ordinal") != ordinal:
            raise ProjectionError(f"projection cell {ordinal}: order mismatch")
        identifier = _require_string(cell.get("requirement_id"), f"cells[{ordinal}].requirement_id")
        if identifier in identifiers:
            raise ProjectionError(f"projection: duplicate cell {identifier}")
        identifiers.add(identifier)
        disposition = cell.get("disposition")
        if disposition not in EXPECTED_DISPOSITIONS:
            raise ProjectionError(f"projection cell {identifier}: unknown disposition")
        dispositions[disposition] += 1
        for observation_name in ("upstream_calibration", "tgrad_diagnostic"):
            observation = cell.get(observation_name)
            if not isinstance(observation, dict):
                raise ProjectionError(f"projection cell {identifier}: missing {observation_name}")
            if observation.get("state") == "observed":
                status = observation.get("status")
                if observation.get("stage") != _status_stage(str(status)):
                    raise ProjectionError(
                        f"projection cell {identifier}: status/stage conflation in {observation_name}"
                    )
        if cell.get("kind") == "test" and disposition == "required":
            required_test_count += 1
            observation = cell["tgrad_diagnostic"]
            if observation.get("state") != "observed":
                raise ProjectionError(f"projection cell {identifier}: required API test is unobserved")
            tgrad_statuses[str(observation["status"])] += 1
    if target.get("requirement_ids_sha256") != canonical_sha256(
        [cell["requirement_id"] for cell in cells]
    ):
        raise ProjectionError("projection: ordered requirement identity mismatch")
    if dict(dispositions) != EXPECTED_DISPOSITIONS:
        raise ProjectionError(f"projection: disposition counts changed: {dict(dispositions)!r}")
    if required_test_count != 34:
        raise ProjectionError(f"projection: expected 34 required API test files, found {required_test_count}")
    if {status: tgrad_statuses[status] for status in FILE_STATUSES} != EXPECTED_TGRAD_STATUS_COUNTS:
        raise ProjectionError("projection: Tgrad collection and execution failures were conflated")

    calibration = document.get("suite_diagnostics")
    if not isinstance(calibration, dict):
        raise ProjectionError("projection: missing suite diagnostics")
    upstream = calibration.get("upstream")
    public_upstream = calibration.get("public_contract_upstream")
    tgrad = calibration.get("tgrad")
    if (
        not isinstance(upstream, dict)
        or not isinstance(public_upstream, dict)
        or not isinstance(tgrad, dict)
    ):
        raise ProjectionError("projection: malformed suite diagnostics")
    if upstream.get("file_status_counts") != EXPECTED_UPSTREAM_STATUS_COUNTS:
        raise ProjectionError("projection: upstream file status distribution changed")
    upstream_tests = upstream.get("test_outcome_counts")
    if not isinstance(upstream_tests, dict) or upstream_tests.get("passed") != 3419:
        raise ProjectionError("projection: upstream test pass count changed")
    if public_upstream.get("file_status_counts") != EXPECTED_PUBLIC_UPSTREAM_STATUS_COUNTS:
        raise ProjectionError("projection: public upstream file status distribution changed")
    if public_upstream.get("test_outcome_counts") != EXPECTED_PUBLIC_UPSTREAM_TEST_COUNTS:
        raise ProjectionError("projection: public upstream test outcomes changed")
    if tgrad.get("file_status_counts") != EXPECTED_TGRAD_STATUS_COUNTS:
        raise ProjectionError("projection: Tgrad file status distribution changed")
    if tgrad.get("failure_stage_counts") != {
        "collection": 29,
        "post_collection_execution": 5,
    }:
        raise ProjectionError("projection: Tgrad failure stages changed")
    source_bindings_value = document.get("source_bindings")
    if source_bindings_value is None:
        if require_source_bindings:
            raise ProjectionError("projection: final artifact requires source bindings")
    else:
        _validate_source_binding_summary(source_bindings_value)
    _validate_no_scalar_score(document)
    if "content_sha256" in document and document.get("content_sha256") != \
            _canonical_document_sha256(document):
        raise ProjectionError("projection: content_sha256 mismatch")


def project_documents(
    requirements_document: Mapping[str, Any],
    oracle_document: Mapping[str, Any],
    upstream_suite_documents: Mapping[str, Mapping[str, Any]],
    tgrad_suite_documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    requirements = validate_requirements(requirements_document)
    test_requirements = [row for row in requirements if row["kind"] == "test"]
    if len(test_requirements) != 138:
        raise ProjectionError("requirement inventory must contain exactly 138 test files")
    oracle_by_path = validate_oracle(oracle_document, test_requirements)
    upstream_results = _merge_suites(upstream_suite_documents, against="upstream")
    tgrad_results = _merge_suites(tgrad_suite_documents, against="tgrad")

    required_test_paths = {
        path for path, row in oracle_by_path.items() if row["class"] == "api_surface"
    }
    requirement_test_paths = {_test_path(row) for row in test_requirements}
    if set(upstream_results) != requirement_test_paths:
        missing = sorted(requirement_test_paths - set(upstream_results))
        extra = sorted(set(upstream_results) - requirement_test_paths)
        raise ProjectionError(
            f"upstream suite coverage mismatch: missing={missing}, extra={extra}"
        )
    if set(tgrad_results) != required_test_paths:
        missing = sorted(required_test_paths - set(tgrad_results))
        extra = sorted(set(tgrad_results) - required_test_paths)
        raise ProjectionError(
            f"Tgrad API coverage mismatch: missing={missing}, extra={extra}"
        )

    upstream_statuses = _status_counts(upstream_results.values())
    if upstream_statuses != EXPECTED_UPSTREAM_STATUS_COUNTS:
        raise ProjectionError(f"upstream status distribution changed: {upstream_statuses!r}")
    upstream_test_totals = _test_totals(upstream_results.values())
    if upstream_test_totals["passed"] != EXPECTED_UPSTREAM_TESTS_PASSED:
        raise ProjectionError(
            f"upstream tests_passed changed: {upstream_test_totals['passed']}"
        )
    tgrad_statuses = _status_counts(tgrad_results.values())
    if tgrad_statuses != EXPECTED_TGRAD_STATUS_COUNTS:
        raise ProjectionError(f"Tgrad status distribution changed: {tgrad_statuses!r}")
    public_upstream_results = {
        path: upstream_results[path] for path in sorted(required_test_paths)
    }
    public_upstream_statuses = _status_counts(public_upstream_results.values())
    public_upstream_test_totals = _test_totals(public_upstream_results.values())
    if public_upstream_statuses != EXPECTED_PUBLIC_UPSTREAM_STATUS_COUNTS:
        raise ProjectionError(
            f"public upstream status distribution changed: {public_upstream_statuses!r}"
        )
    if public_upstream_test_totals != EXPECTED_PUBLIC_UPSTREAM_TEST_COUNTS:
        raise ProjectionError(
            f"public upstream test outcomes changed: {public_upstream_test_totals!r}"
        )

    cells = [
        _project_cell(row, oracle_by_path, upstream_results, tgrad_results)
        for row in requirements
    ]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "tgrad-parity-coverage-diagnostic",
        "target": {
            "repository": "tinygrad/tinygrad",
            "upstream_ref": UPSTREAM_REF,
            "requirement_count": EXPECTED_REQUIREMENT_COUNT,
            "requirement_inventory_content_sha256": requirements_document["content_sha256"],
            "requirement_ids_sha256": requirements_document["requirement_ids_sha256"],
        },
        "profile": {"id": PROFILE, "backend": "metal"},
        "projection_policy": {
            "unit": "one ordered cell per reviewed requirement",
            "symbol_requirements": "required except fifteen non-Metal backend requirements",
            "api_surface_tests": "required",
            "internal_repr_tests": "excluded",
            "infrastructure_tests": "excluded",
            "ambiguous_package_initializers": "outside the extracted 590-requirement denominator",
            "aggregation": "status distributions only; no scalar parity score or percentage",
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "promotable": False,
            "reportable_as_conformance": False,
            "subject_identity": {
                "state": "unknown",
                "forbidden_inference": (
                    "do not infer the Tgrad subject from the fixture-producing commit or its parent"
                ),
            },
            "unknown_attribution_blockers": list(UNKNOWN_ATTRIBUTION_BLOCKERS),
        },
        "suite_diagnostics": {
            "upstream": {
                "files_observed": len(upstream_results),
                "file_status_counts": upstream_statuses,
                "test_outcome_counts": upstream_test_totals,
            },
            "public_contract_upstream": {
                "files_observed": len(public_upstream_results),
                "file_status_counts": public_upstream_statuses,
                "test_outcome_counts": public_upstream_test_totals,
            },
            "tgrad": {
                "files_observed": len(tgrad_results),
                "file_status_counts": tgrad_statuses,
                "failure_stage_counts": {
                    "collection": tgrad_statuses["collect_error"],
                    "post_collection_execution": tgrad_statuses["fail"],
                },
                "test_outcome_counts": _test_totals(tgrad_results.values()),
            },
        },
        "cells": cells,
    }
    _validate_projection_structure(document, require_source_bindings=False)
    return document


def load_projection_inputs(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    requirements_path = root / SOURCE_FIXTURES[0]
    try:
        requirements = load_requirement_inventory(requirements_path)
    except CoverageModelError as error:
        raise ProjectionError(str(error)) from error
    # Run our stricter projection-specific checks as well as the shared loader.
    validate_requirements(requirements)
    return {
        "requirements": requirements,
        "oracle": _load_json(root / SOURCE_FIXTURES[1], "oracle fixture"),
        "upstream": {
            group: _load_json(
                root / f"fixtures/parity/suite_upstream_{group}_19c4d736f2bc.json",
                f"upstream {group} suite",
            )
            for group in GROUPS
        },
        "tgrad": {
            group: _load_json(
                root / f"fixtures/parity/suite_tgrad_{group}_19c4d736f2bc.json",
                f"Tgrad {group} suite",
            )
            for group in GROUPS
        },
    }


def _git_blob(root: Path, relative_path: str) -> tuple[bytes, str] | None:
    specification = f"{REFERENCE_ARTIFACT_COMMIT}:{relative_path}"
    try:
        content = subprocess.run(
            ["git", "-C", str(root), "show", specification],
            check=True,
            capture_output=True,
        ).stdout
        object_id = subprocess.run(
            ["git", "-C", str(root), "rev-parse", specification],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return content, object_id


def _bind_file(
    root: Path, relative_path: str, *, expected_at_reference: bool
) -> dict[str, Any]:
    path = root / relative_path
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ProjectionError(f"cannot bind {relative_path}: {error}") from error
    binding: dict[str, Any] = {
        "path": relative_path,
        "sha256": _sha256_bytes(content),
        "bytes": len(content),
    }
    reference = _git_blob(root, relative_path)
    if expected_at_reference:
        if reference is None:
            raise ProjectionError(
                f"{relative_path}: absent from reference commit {REFERENCE_ARTIFACT_COMMIT}"
            )
        reference_content, object_id = reference
        if reference_content != content:
            raise ProjectionError(
                f"{relative_path}: bytes differ from reference commit {REFERENCE_ARTIFACT_COMMIT}"
            )
        binding["reference_artifact"] = {
            "state": "byte_identical",
            "commit": REFERENCE_ARTIFACT_COMMIT,
            "git_blob_oid": object_id,
            "sha256": _sha256_bytes(reference_content),
        }
    else:
        if reference is not None:
            raise ProjectionError(
                f"{relative_path}: unexpectedly exists at {REFERENCE_ARTIFACT_COMMIT}; review binding policy"
            )
        binding["reference_artifact"] = {
            "state": "not_present",
            "commit": REFERENCE_ARTIFACT_COMMIT,
            "reason": "the reviewed requirement inventory was generated after the diagnostic suite artifacts",
        }
    return binding


def source_bindings(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    fixtures = [
        _bind_file(
            root,
            relative_path,
            expected_at_reference=relative_path in REFERENCE_BOUND_FIXTURES,
        )
        for relative_path in SOURCE_FIXTURES
    ]
    adapters = [
        _bind_file(root, relative_path, expected_at_reference=True)
        for relative_path in ADAPTER_FILES
    ]
    compact_fixtures = [
        {"path": binding["path"], "sha256": binding["sha256"]}
        for binding in fixtures
    ]
    compact_adapters = [
        {"path": binding["path"], "sha256": binding["sha256"]}
        for binding in adapters
    ]
    tgrad_results = [
        binding for binding in compact_fixtures
        if binding["path"].startswith("fixtures/parity/suite_tgrad_")
    ]
    return {
        "reference_artifact_commit": REFERENCE_ARTIFACT_COMMIT,
        "reference_commit_role": "artifact container only; never inferred as the tested Tgrad subject",
        "source_fixtures": fixtures,
        "source_fixture_bundle_sha256": canonical_sha256(compact_fixtures),
        "strict_adapter_files": adapters,
        "strict_adapter_bundle_sha256": canonical_sha256(compact_adapters),
        "tgrad_result_bundle_sha256": canonical_sha256(tgrad_results),
    }


def _validate_source_binding_summary(value: Any) -> None:
    if not isinstance(value, dict):
        raise ProjectionError("projection: source_bindings must be an object")
    if value.get("reference_artifact_commit") != REFERENCE_ARTIFACT_COMMIT:
        raise ProjectionError("projection: source binding commit changed")
    if value.get("reference_commit_role") != (
        "artifact container only; never inferred as the tested Tgrad subject"
    ):
        raise ProjectionError("projection: source binding commit role changed")
    fixtures = value.get("source_fixtures")
    adapters = value.get("strict_adapter_files")
    if not isinstance(fixtures, list) or not isinstance(adapters, list):
        raise ProjectionError("projection: malformed source binding lists")
    if [row.get("path") for row in fixtures if isinstance(row, dict)] != list(SOURCE_FIXTURES):
        raise ProjectionError("projection: source fixture binding order changed")
    if [row.get("path") for row in adapters if isinstance(row, dict)] != list(ADAPTER_FILES):
        raise ProjectionError("projection: adapter binding order changed")
    for label, rows in (("source fixture", fixtures), ("adapter", adapters)):
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ProjectionError(f"projection: malformed {label} binding {index}")
            digest = row.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ProjectionError(f"projection: malformed {label} sha256 {index}")
            if not isinstance(row.get("bytes"), int) or row["bytes"] <= 0:
                raise ProjectionError(f"projection: malformed {label} size {index}")
            reference = row.get("reference_artifact")
            if not isinstance(reference, dict):
                raise ProjectionError(f"projection: missing {label} reference binding {index}")
            if reference.get("commit") != REFERENCE_ARTIFACT_COMMIT:
                raise ProjectionError(f"projection: wrong {label} reference commit {index}")
            expected_at_reference = (
                label == "adapter" or row["path"] in REFERENCE_BOUND_FIXTURES
            )
            if expected_at_reference:
                object_id = reference.get("git_blob_oid")
                if (
                    reference.get("state") != "byte_identical"
                    or reference.get("sha256") != digest
                    or not isinstance(object_id, str)
                    or len(object_id) != 40
                ):
                    raise ProjectionError(
                        f"projection: malformed {label} byte-identical binding {index}"
                    )
            elif reference.get("state") != "not_present":
                raise ProjectionError(
                    f"projection: malformed {label} absent-reference binding {index}"
                )
    compact_fixtures = [
        {"path": row["path"], "sha256": row["sha256"]} for row in fixtures
    ]
    compact_adapters = [
        {"path": row["path"], "sha256": row["sha256"]} for row in adapters
    ]
    compact_tgrad = [
        row for row in compact_fixtures
        if row["path"].startswith("fixtures/parity/suite_tgrad_")
    ]
    expected = {
        "source_fixture_bundle_sha256": canonical_sha256(compact_fixtures),
        "strict_adapter_bundle_sha256": canonical_sha256(compact_adapters),
        "tgrad_result_bundle_sha256": canonical_sha256(compact_tgrad),
    }
    for field, digest in expected.items():
        if value.get(field) != digest:
            raise ProjectionError(f"projection: {field} mismatch")


def build_document(root: Path = ROOT) -> dict[str, Any]:
    inputs = load_projection_inputs(root)
    document = project_documents(
        inputs["requirements"],
        inputs["oracle"],
        inputs["upstream"],
        inputs["tgrad"],
    )
    document["source_bindings"] = source_bindings(root)
    document = _attach_content_sha256(document)
    _validate_projection_structure(document, require_source_bindings=True)
    if document["content_sha256"] != _canonical_document_sha256(document):
        raise ProjectionError("projection: content hash attachment failed")
    return document


def validate_projection(document: Mapping[str, Any], root: Path = ROOT) -> None:
    """Strictly validate a final artifact by rederiving every semantic join."""
    _validate_projection_structure(document, require_source_bindings=True)
    expected = build_document(root)
    if dict(document) != expected:
        raise ProjectionError(
            "projection: artifact differs from the projection rederived from bound inputs"
        )


def render_document(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")


def validate_lean_projection_pin(
    document: Mapping[str, Any], root: Path = ROOT
) -> None:
    """Require the checked Lean snapshot to pin this exact JSON document."""
    digest = _require_string(document.get("content_sha256"), "projection.content_sha256")
    try:
        source = (root / LEAN_PROJECTION_PATH).read_text(encoding="utf-8")
    except OSError as error:
        raise ProjectionError(f"cannot read Lean projection snapshot: {error}") from error
    pattern = rf'projectionContentHash\s*:=\s*\n?\s*"{re.escape(digest)}"'
    if re.search(pattern, source) is None:
        raise ProjectionError(
            "Lean projection snapshot does not pin the generated JSON content_sha256"
        )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Tgrad repository root")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures/parity/coverage_diagnostic_19c4d736f2bc.json",
        help="diagnostic projection path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless --output already has exactly the deterministic projected bytes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = build_document(args.root)
        validate_lean_projection_pin(document, args.root)
        rendered = render_document(document)
    except ProjectionError as error:
        print(f"coverage projection failed: {error}", file=sys.stderr)
        return 2
    if args.check:
        try:
            current = args.output.read_bytes()
        except FileNotFoundError:
            current = b""
        if current != rendered:
            print(f"out of date: {args.output}", file=sys.stderr)
            return 1
        print(f"up to date: {args.output}")
        return 0
    _atomic_write(args.output, rendered)
    print(f"wrote diagnostic-only projection: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

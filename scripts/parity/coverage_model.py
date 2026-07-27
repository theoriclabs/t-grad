#!/usr/bin/env python3
"""Pure model for Tgrad's generated tinygrad parity requirements.

This module translates the checked foreign inventory into the same 590
requirement identifiers and policy fields used by ``Tgrad.Spec.Parity``.  It
contains no observations about Tgrad and therefore cannot turn implementation
facts into a smaller denominator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.parity.render_lean_target import load_checked
except ModuleNotFoundError:  # Direct execution/import from scripts/parity.
    from render_lean_target import load_checked


SCHEMA_VERSION = 2
EXPECTED_REQUIREMENT_COUNT = 590
EXPECTED_CATEGORY_COUNTS = {
    "tensor-method": 297,
    "tensor-property": 5,
    "dtype": 52,
    "ops": 82,
    "backend": 16,
    "test-null": 54,
    "test-unit": 43,
    "test-backend": 41,
}
PROFILE_IDS = (
    "semanticCore",
    "publicApi",
    "metal",
    "portable",
    "ecosystem",
    "allBackends",
)
API_PROFILES = ("publicApi", "metal", "portable", "ecosystem", "allBackends")
SEMANTIC_PROFILES = PROFILE_IDS


class CoverageModelError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_body(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in ("generated_at_utc", "content_sha256")
    }


def attach_content_sha256(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result["content_sha256"] = canonical_sha256(content_body(result))
    return result


def verify_content_sha256(document: dict[str, Any], label: str) -> None:
    recorded = document.get("content_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise CoverageModelError(f"{label}: missing content_sha256")
    actual = canonical_sha256(content_body(document))
    if recorded != actual:
        raise CoverageModelError(
            f"{label}: content_sha256 mismatch: recorded={recorded}, actual={actual}"
        )


def _symbol_requirement(
    category: str,
    symbol: str,
    domains: Iterable[str],
    dimensions: Iterable[str],
    profiles: Iterable[str],
    relation: str,
    *,
    environments: Iterable[str] = ("host",),
    upstream_sources: Iterable[str],
) -> dict[str, Any]:
    return {
        "id": f"{category}:{symbol}",
        "kind": "symbol",
        "category": category,
        "domains": list(domains),
        "dimensions": list(dimensions),
        "profiles": list(profiles),
        "upstream_symbols": [symbol],
        "upstream_tests": [],
        "upstream_sources": list(upstream_sources),
        "required_environments": list(environments),
        "environment_selector": (
            {"kind": "specific", "ids": list(environments)}
            if category == "backend"
            else {"kind": "profile"}
        ),
        "equivalence_relation": relation,
    }


def _test_requirement(
    group: str, path: str, byte_count: int, backend_names: Iterable[str]
) -> dict[str, Any]:
    profiles = (
        ("metal", "portable", "allBackends")
        if group == "backend"
        else SEMANTIC_PROFILES
    )
    return {
        "id": f"test:{group}:{path}",
        "kind": "test",
        "category": f"test-{group}",
        "domains": ["workloads"],
        "dimensions": ["api", "semantic", "runtime"],
        "profiles": list(profiles),
        "upstream_symbols": [],
        "upstream_tests": [path],
        "upstream_sources": [f"test/{group}/{path}"],
        "upstream_byte_count": byte_count,
        # A profile may select a subset later, but the allBackends denominator
        # is never represented by the ambiguous string "declared-backend".
        "required_environments": (
            list(backend_names) if group == "backend" else ["host"]
        ),
        "environment_selector": {"kind": "profile"},
        "equivalence_relation": "tinygrad-test-file-v1",
    }


def requirements_from_checked_manifest(data: dict[str, Any]) -> list[dict[str, Any]]:
    checked = data.get("_checked")
    if not isinstance(checked, dict):
        raise CoverageModelError("manifest was not validated by load_checked")
    rows: list[dict[str, Any]] = []
    tensor_sources: dict[str, list[str]] = {}
    for source, names in data["tensor_api"]["by_source"].items():
        for name in names:
            tensor_sources.setdefault(name, []).append(source)

    def sources_for_tensor_name(name: str) -> list[str]:
        sources = tensor_sources.get(name, [])
        if not sources:
            raise CoverageModelError(f"Tensor.{name}: no upstream source locator")
        return sources

    rows.extend(
        _symbol_requirement(
            "tensor-method",
            f"Tensor.{name}",
            ("tensorSurface",),
            ("api", "semantic"),
            API_PROFILES,
            "tinygrad-tensor-api-v1",
            upstream_sources=sources_for_tensor_name(name),
        )
        for name in checked["tensor_methods"]
    )
    rows.extend(
        _symbol_requirement(
            "tensor-property",
            f"Tensor.{name}",
            ("tensorSurface",),
            ("api", "semantic"),
            API_PROFILES,
            "tinygrad-tensor-api-v1",
            upstream_sources=sources_for_tensor_name(name),
        )
        for name in checked["tensor_properties"]
    )
    rows.extend(
        _symbol_requirement(
            "dtype",
            f"dtypes.{name}",
            ("dtypeSystem", "scalarSemantics"),
            ("api", "semantic", "numerical"),
            SEMANTIC_PROFILES,
            "tinygrad-dtype-v1",
            upstream_sources=(data["dtypes"]["source"],),
        )
        for name in checked["dtype_names"]
    )
    rows.extend(
        _symbol_requirement(
            "ops",
            f"Ops.{name}",
            ("uopIr", "rewriteSystem", "lowering"),
            ("semantic", "compiler"),
            SEMANTIC_PROFILES,
            "tinygrad-ops-v1",
            upstream_sources=(data["ops"]["source"],),
        )
        for name in checked["ops_members"]
    )
    for name in checked["backend_names"]:
        profiles = ("metal", "allBackends") if name == "metal" else ("allBackends",)
        rows.append(
            _symbol_requirement(
                "backend",
                f"tinygrad.runtime.ops_{name}",
                ("renderer", "runtime"),
                ("backend", "runtime"),
                profiles,
                "tinygrad-backend-v1",
                environments=(name,),
                upstream_sources=(f"{data['backends']['source']}ops_{name}.py",),
            )
        )
    for group in ("null", "unit", "backend"):
        files = data["tests"]["groups"][group]["files"]
        rows.extend(
            _test_requirement(
                group,
                file["name"],
                file["bytes"],
                checked["backend_names"],
            )
            for file in files
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise CoverageModelError("generated requirement identifiers are not unique")
    expected = data["counts"]
    expected_count = (
        expected["tensor_methods"]
        + expected["tensor_properties"]
        + expected["dtype_names"]
        + expected["ops_members"]
        + expected["backends"]
        + expected["test_files"]
    )
    if len(rows) != expected_count:
        raise CoverageModelError(
            f"generated {len(rows)} requirements, expected {expected_count}"
        )
    if expected_count != EXPECTED_REQUIREMENT_COUNT:
        raise CoverageModelError(
            "upstream denominator changed: "
            f"manifest has {expected_count}, reviewed contract pins "
            f"{EXPECTED_REQUIREMENT_COUNT}; promote a new target explicitly"
        )
    for ordinal, row in enumerate(rows):
        row["ordinal"] = ordinal
        row["identity_sha256"] = canonical_sha256(row)
    return rows


def build_requirement_inventory(manifest_path: Path) -> dict[str, Any]:
    data = load_checked(manifest_path)
    requirements = requirements_from_checked_manifest(data)
    category_counts: dict[str, int] = {}
    for requirement in requirements:
        category = requirement["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "tgrad-parity-requirement-inventory",
        "target": {
            "upstream_repo": data["upstream_repo"],
            "upstream_ref": data["upstream_ref"],
            "upstream_committed_at": data["upstream_committed_at"],
            # Paths are machine-local identities.  The basename plus checked
            # content hash is reproducible across worktrees and hosts.
            "source_manifest": manifest_path.name,
            "source_manifest_content_sha256": data["content_sha256"],
        },
        "mapping_policy": {
            "id": "tgrad-parity-requirement-mapping-v1",
            "source": "Tgrad.Spec.Parity generated-target mapping",
            "profile_ids": list(PROFILE_IDS),
        },
        "reviewed_requirement_count": EXPECTED_REQUIREMENT_COUNT,
        "requirement_count": len(requirements),
        "category_counts": category_counts,
        "requirement_ids_sha256": canonical_sha256([row["id"] for row in requirements]),
        "requirements": requirements,
    }
    return attach_content_sha256(document)


def load_requirement_inventory(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageModelError(f"cannot read requirement inventory {path}: {error}") from error
    if not isinstance(document, dict):
        raise CoverageModelError("requirement inventory root must be an object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CoverageModelError("unsupported requirement inventory schema_version")
    if document.get("kind") != "tgrad-parity-requirement-inventory":
        raise CoverageModelError("unexpected requirement inventory kind")
    verify_content_sha256(document, "requirement inventory")
    requirements = document.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise CoverageModelError("requirements must be a non-empty list")
    ids = [row.get("id") for row in requirements if isinstance(row, dict)]
    if len(ids) != len(requirements) or not all(isinstance(value, str) and value for value in ids):
        raise CoverageModelError("malformed requirement row")
    if len(ids) != len(set(ids)):
        raise CoverageModelError("duplicate requirement identifiers")
    if document.get("requirement_count") != len(ids):
        raise CoverageModelError("requirement_count mismatch")
    if document.get("reviewed_requirement_count") != EXPECTED_REQUIREMENT_COUNT:
        raise CoverageModelError("reviewed_requirement_count mismatch")
    if len(ids) != EXPECTED_REQUIREMENT_COUNT:
        raise CoverageModelError(
            f"requirement denominator must remain exactly {EXPECTED_REQUIREMENT_COUNT}"
        )
    if document.get("category_counts") != EXPECTED_CATEGORY_COUNTS:
        raise CoverageModelError("category_counts differ from the reviewed target")
    for ordinal, row in enumerate(requirements):
        if row.get("ordinal") != ordinal:
            raise CoverageModelError(f"requirements[{ordinal}].ordinal mismatch")
        identity = row.get("identity_sha256")
        body = {key: value for key, value in row.items() if key != "identity_sha256"}
        if identity != canonical_sha256(body):
            raise CoverageModelError(f"requirements[{ordinal}].identity_sha256 mismatch")
    if document.get("requirement_ids_sha256") != canonical_sha256(ids):
        raise CoverageModelError("requirement_ids_sha256 mismatch")
    return document

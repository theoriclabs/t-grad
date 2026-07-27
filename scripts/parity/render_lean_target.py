#!/usr/bin/env python3
"""Render the checked Lean upstream target from an extracted parity manifest.

The extractor owns discovery from tinygrad.  This consumer owns only a total,
deterministic translation of that foreign inventory into Lean data.  It
recomputes every count and digest before rendering, rejects duplicates and
implicit exclusions, and supports ``--check`` so generated drift is fatal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "fixtures/parity/upstream_19c4d736f2bc.json"
DEFAULT_OUTPUT = ROOT / "Tgrad/Spec/ParityTarget.lean"
SECTIONS = ("tensor_api", "dtypes", "ops", "backends", "tests")


class ManifestError(RuntimeError):
    pass


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require_string(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key}: expected non-empty string")
    return value


def require_unique_strings(values: object, label: str) -> list[str]:
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ManifestError(f"{label}: expected a non-empty string list")
    if len(values) != len(set(values)):
        raise ManifestError(f"{label}: duplicate entries")
    if values != sorted(values):
        raise ManifestError(f"{label}: entries are not sorted")
    return values


def load_checked(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be an object")
    if data.get("schema_version") != 1:
        raise ManifestError("unsupported or missing schema_version")
    for key in ("upstream_repo", "upstream_ref", "upstream_committed_at",
                "extractor", "extractor_sha256", "content_sha256"):
        require_string(data, key)
    if not isinstance(data.get("exclusions"), list):
        raise ManifestError("exclusions: expected an explicit list")
    if data["exclusions"]:
        raise ManifestError("the pinned bootstrap target requires an empty exclusions ledger")

    body = {
        key: value for key, value in data.items()
        if key not in ("extracted_at_utc", "content_sha256")
    }
    if canonical_sha(body) != data["content_sha256"]:
        raise ManifestError("content_sha256 does not match canonical manifest content")

    section_hashes = data.get("section_sha256")
    if not isinstance(section_hashes, dict):
        raise ManifestError("section_sha256: expected an object")
    for section in SECTIONS:
        if section not in data or canonical_sha(data[section]) != section_hashes.get(section):
            raise ManifestError(f"{section}: section hash mismatch")

    tensor = data.get("tensor_api")
    dtypes = data.get("dtypes")
    ops = data.get("ops")
    backends = data.get("backends")
    tests = data.get("tests")
    if not all(isinstance(section, dict) for section in
               (tensor, dtypes, ops, backends, tests)):
        raise ManifestError("one or more inventory sections are not objects")

    methods = require_unique_strings(tensor.get("methods"), "tensor_api.methods")
    properties = require_unique_strings(tensor.get("properties"), "tensor_api.properties")
    dtype_names = require_unique_strings(dtypes.get("names"), "dtypes.names")
    ops_members = require_unique_strings(ops.get("members"), "ops.members")
    backend_names = require_unique_strings(backends.get("names"), "backends.names")

    groups = tests.get("groups")
    if not isinstance(groups, dict) or set(groups) != {"null", "unit", "backend"}:
        raise ManifestError("tests.groups must contain exactly null, unit, and backend")
    test_files: dict[str, list[str]] = {}
    for group in ("null", "unit", "backend"):
        record = groups[group]
        if not isinstance(record, dict) or not isinstance(record.get("files"), list):
            raise ManifestError(f"tests.groups.{group}: malformed")
        names = require_unique_strings(
            [entry.get("name") for entry in record["files"]
             if isinstance(entry, dict)],
            f"tests.groups.{group}.files",
        )
        if len(names) != len(record["files"]):
            raise ManifestError(f"tests.groups.{group}: malformed file record")
        if record.get("count") != len(names):
            raise ManifestError(f"tests.groups.{group}: count mismatch")
        test_files[group] = [f"test/{group}/{name}" for name in names]

    expected_counts = {
        "tensor_methods": len(methods),
        "tensor_properties": len(properties),
        "dtype_names": len(dtype_names),
        "ops_members": len(ops_members),
        "backends": len(backend_names),
        "test_files": sum(len(files) for files in test_files.values()),
        "test_files_no_backend": len(test_files["null"]),
    }
    if data.get("counts") != expected_counts:
        raise ManifestError("counts do not agree with the extracted inventories")

    data["_checked"] = {
        "tensor_methods": methods,
        "tensor_properties": properties,
        "dtype_names": dtype_names,
        "ops_members": ops_members,
        "backend_names": backend_names,
        "test_files": test_files,
    }
    return data


def lean_string(value: str) -> str:
    # The generated inventories are ASCII today; JSON escaping is also valid
    # for the quote, slash, and control characters admitted by Lean strings.
    return json.dumps(value, ensure_ascii=True)


def lean_list(name: str, values: list[str]) -> str:
    body = "\n".join(f"    {lean_string(value)}," for value in values)
    return f"def {name} : List String :=\n  [\n{body}\n  ]\n"


def render(data: dict, manifest_path: Path, manifest_label: str | None = None) -> str:
    checked = data["_checked"]
    section_hashes = data["section_sha256"]
    if manifest_label is None:
        try:
            manifest_label = str(manifest_path.resolve().relative_to(ROOT))
        except ValueError:
            manifest_label = str(manifest_path.resolve())
    chunks = [
        "/-! This file is generated by scripts/parity/render_lean_target.py.\n"
        "Do not edit it by hand; regenerate it from the pinned foreign manifest. -/\n\n",
        "namespace Tgrad.Spec.ParityTarget\n\n",
        f"def manifestPath : String := {lean_string(manifest_label)}\n",
        f"def repository : String := {lean_string(data['upstream_repo'])}\n",
        f"def revision : String := {lean_string(data['upstream_ref'])}\n",
        f"def committedAt : String := {lean_string(data['upstream_committed_at'])}\n",
        f"def extractorSha256 : String := {lean_string(data['extractor_sha256'])}\n",
        f"def manifestContentSha256 : String := {lean_string(data['content_sha256'])}\n",
        f"def sourceManifestSha256 : String := {lean_string(data['content_sha256'])}\n",
        f"def apiManifestSha256 : String := {lean_string(section_hashes['tensor_api'])}\n",
        f"def testManifestSha256 : String := {lean_string(section_hashes['tests'])}\n\n",
        lean_list("tensorMethods", checked["tensor_methods"]), "\n",
        lean_list("tensorProperties", checked["tensor_properties"]), "\n",
        lean_list("dtypeNames", checked["dtype_names"]), "\n",
        lean_list("opsMembers", checked["ops_members"]), "\n",
        lean_list("backendNames", checked["backend_names"]), "\n",
        lean_list("nullTestFiles", checked["test_files"]["null"]), "\n",
        lean_list("unitTestFiles", checked["test_files"]["unit"]), "\n",
        lean_list("backendTestFiles", checked["test_files"]["backend"]), "\n",
        "def exclusions : List String := []\n\n",
        "def allTestFiles : List String :=\n"
        "  nullTestFiles ++ unitTestFiles ++ backendTestFiles\n\n",
        "def requirementCount : Nat :=\n"
        "  tensorMethods.length + tensorProperties.length + dtypeNames.length +\n"
        "  opsMembers.length + backendNames.length + allTestFiles.length\n\n",
        "end Tgrad.Spec.ParityTarget\n",
    ]
    return "".join(chunks)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-label")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        rendered = render(load_checked(args.manifest), args.manifest, args.manifest_label)
    except (ManifestError, ValueError) as error:
        print(f"render_lean_target: FAILED — {error}")
        return 1
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            print(f"render_lean_target: STALE — {args.output}")
            return 1
        print(f"render_lean_target: OK — {args.output}")
        return 0
    atomic_write(args.output, rendered)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

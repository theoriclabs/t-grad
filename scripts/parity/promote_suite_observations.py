#!/usr/bin/env python3
"""Promote validated suite observations and their raw artifacts into the repo.

Observation writes to an external content-addressed directory by default.  This
tool is the explicit owner decision that copies an upstream/Tgrad pair into the
committed evidence tree without changing their bytes or identities.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.parity import run_upstream_suite as observer


DEFAULT_DESTINATION = REPO / "fixtures" / "parity" / "observations"


def artifact_refs(document: dict) -> list[dict]:
    refs: list[dict] = []
    for cell in document.get("observation", {}).get("cells", []):
        report = cell.get("report_artifact")
        if report:
            refs.append(report)
        refs.extend(cell.get("diagnostics", {}).get("raw_artifacts", {}).values())
    unique = {ref["path"]: ref for ref in refs}
    if len(unique) != len({(ref["path"], ref["sha256"]) for ref in refs}):
        raise RuntimeError("one artifact path names multiple hashes")
    return [unique[path] for path in sorted(unique)]


def validate_document(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema_version") != observer.SCHEMA_VERSION:
        raise RuntimeError(f"unsupported observation schema: {path}")
    if document.get("against") not in {"upstream", "tgrad"}:
        raise RuntimeError(f"invalid observation subject: {path}")
    if document.get("identity", {}).get("verifier", {}).get(
        "runner_sha256"
    ) != observer.file_hash(Path(observer.__file__).resolve()):
        raise RuntimeError(f"observation was produced by a different verifier: {path}")
    cells = document.get("observation", {}).get("cells")
    if not isinstance(cells, list):
        raise RuntimeError(f"observation cells missing: {path}")
    expected = document.get("identity", {}).get("contract", {}).get(
        "executed_file_count"
    )
    if expected != len(cells):
        raise RuntimeError(f"cell count disagrees with contract: {path}")
    names = [cell.get("file") for cell in cells]
    if not all(isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise RuntimeError(f"cell files are missing or duplicated: {path}")
    if document["observation"].get("aggregate") != observer.aggregate(cells):
        raise RuntimeError(f"aggregate disagrees with cells: {path}")
    scenario = document.get("identity", {}).get("scenario", {})
    expected_scenario = observer.digest(observer.canonical({
        key: value for key, value in scenario.items() if key != "sha256"
    }))
    if scenario.get("sha256") != expected_scenario:
        raise RuntimeError(f"scenario hash is inconsistent: {path}")
    if document.get("scenario_id") != expected_scenario:
        raise RuntimeError(f"scenario_id is inconsistent: {path}")
    if document.get("result_id") != observer.computed_result_id(document):
        raise RuntimeError(f"result_id is inconsistent: {path}")
    if document.get("run_artifact_id") != observer.computed_run_artifact_id(document):
        raise RuntimeError(f"run_artifact_id is inconsistent: {path}")
    return document, raw


def validate_pair(upstream: dict, upstream_raw: bytes, tgrad: dict) -> None:
    if upstream.get("against") != "upstream" or tgrad.get("against") != "tgrad":
        raise RuntimeError("promotion requires one upstream and one Tgrad observation")
    upstream_identity = upstream["identity"]
    tgrad_identity = tgrad["identity"]
    for key in ("upstream", "contract", "verifier", "scenario"):
        if upstream_identity.get(key) != tgrad_identity.get(key):
            raise RuntimeError(f"observation pair differs at identity.{key}")
    for key in ("facts",):
        if upstream_identity["environment"].get(key) != tgrad_identity["environment"].get(key):
            raise RuntimeError(f"observation pair differs at environment.{key}")
    if (
        upstream_identity["environment"]["backend"].get("hardware") !=
        tgrad_identity["environment"]["backend"].get("hardware")
    ):
        raise RuntimeError("observation pair used different hardware")
    reference = tgrad_identity.get("upstream_baseline", {})
    if reference.get("artifact_sha256") != observer.digest(upstream_raw):
        raise RuntimeError("Tgrad evidence does not bind the supplied upstream artifact")
    if reference.get("result_id") != upstream.get("result_id"):
        raise RuntimeError("Tgrad evidence binds a different upstream result")


def copy_artifacts(source_document: Path, document: dict,
                   destination: Path, check: bool) -> None:
    for ref in artifact_refs(document):
        raw = observer.read_bound_artifact(source_document, ref)
        relative = Path(ref["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"artifact path escapes promotion root: {relative}")
        target = (destination / relative).resolve()
        root = destination.resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"artifact path escapes promotion root: {relative}")
        if check:
            if not target.is_file() or target.read_bytes() != raw:
                raise RuntimeError(f"promoted artifact missing or different: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != raw:
                raise RuntimeError(f"refusing to overwrite different artifact: {target}")
            if not target.exists():
                target.write_bytes(raw)


def promote(source: Path, raw: bytes, destination: Path, check: bool) -> Path:
    target = destination / source.name
    if check:
        if not target.is_file() or target.read_bytes() != raw:
            raise RuntimeError(f"promoted observation missing or different: {target}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite different observation: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--tgrad", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    upstream, upstream_raw = validate_document(args.upstream.resolve())
    tgrad, tgrad_raw = validate_document(args.tgrad.resolve())
    observer.validate_tgrad_observation(
        args.tgrad.resolve(), tgrad, args.upstream.resolve(),
        observer.DEFAULT_CHECKOUT,
    )
    validate_pair(upstream, upstream_raw, tgrad)
    root = args.destination.resolve()
    bundle_name = (
        f"pair_{upstream['result_id'][:12]}_{tgrad['result_id'][:12]}"
    )
    destination = root / bundle_name
    if args.check:
        if not destination.is_dir():
            raise RuntimeError(f"promoted bundle is missing: {destination}")
        copy_artifacts(args.upstream.resolve(), upstream, destination, True)
        copy_artifacts(args.tgrad.resolve(), tgrad, destination, True)
        promoted = [
            promote(args.upstream.resolve(), upstream_raw, destination, True),
            promote(args.tgrad.resolve(), tgrad_raw, destination, True),
        ]
    else:
        root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(
                f"bundle already exists; use --check instead: {destination}"
            )
        stage = Path(tempfile.mkdtemp(prefix=f".{bundle_name}.", dir=root))
        try:
            copy_artifacts(args.upstream.resolve(), upstream, stage, False)
            copy_artifacts(args.tgrad.resolve(), tgrad, stage, False)
            promoted = [
                promote(args.upstream.resolve(), upstream_raw, stage, False),
                promote(args.tgrad.resolve(), tgrad_raw, stage, False),
            ]
            os.replace(stage, destination)
            promoted = [destination / path.name for path in promoted]
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    print("checked" if args.check else "promoted")
    for path in promoted:
        print(path.relative_to(REPO) if path.is_relative_to(REPO) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

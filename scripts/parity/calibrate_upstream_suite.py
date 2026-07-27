#!/usr/bin/env python3
"""Create diagnosable, repeatable upstream-on-upstream oracle calibrations.

This is intentionally separate from ``run_upstream_suite.py`` so the active
Tgrad-substitution worker can evolve that adapter without sharing a write set.
The runner publishes a directory only after every selected file has a raw
stdout, stderr, and (when pytest produced one) JUnit artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from scripts.parity.coverage_model import (
        CoverageModelError,
        attach_content_sha256,
        canonical_sha256,
        file_sha256,
    )
    from scripts.parity.render_lean_target import load_checked
except ModuleNotFoundError:
    from coverage_model import (  # type: ignore[no-redef]
        CoverageModelError,
        attach_content_sha256,
        canonical_sha256,
        file_sha256,
    )
    from render_lean_target import load_checked  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = ROOT / "fixtures" / "parity" / "upstream_19c4d736f2bc.json"
ENVIRONMENT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1
PINNED_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
    "LC_ALL": "C",
    "LANG": "C",
}
FILE_STATUSES = {
    "pass",
    "pass_with_skips",
    "fail",
    "collect_error",
    "timeout",
    "empty",
    "runner_error",
}


class CalibrationError(RuntimeError):
    pass


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
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


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CalibrationError(f"command failed: {command!r}: {error}") from error
    return completed.stdout.strip()


def _git(checkout: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(checkout), *arguments])


def checkout_identity(checkout: Path, expected_revision: str) -> dict[str, Any]:
    checkout = checkout.resolve()
    commit = _git(checkout, "rev-parse", "HEAD")
    tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    if commit != expected_revision:
        raise CalibrationError(
            f"upstream checkout is {commit}, expected exact pin {expected_revision}"
        )
    status = _git(checkout, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise CalibrationError("upstream checkout must be clean, including untracked files")
    return {
        "repository": str(checkout),
        "commit": commit,
        "tree": tree,
        "dirty": False,
    }


def python_facts(python: Path) -> dict[str, Any]:
    resolved = python.resolve()
    if not resolved.is_file():
        raise CalibrationError(f"Python executable does not exist: {resolved}")
    probe = _run(
        [
            str(resolved),
            "-c",
            (
                "import json,platform,sys; "
                "print(json.dumps({'version':platform.python_version(),"
                "'implementation':platform.python_implementation(),"
                "'cache_tag':sys.implementation.cache_tag}))"
            ),
        ]
    )
    facts = json.loads(probe)
    packages = _run(
        [str(resolved), "-m", "pip", "--disable-pip-version-check", "freeze", "--all"]
    ).splitlines()
    packages = sorted(line.strip() for line in packages if line.strip())
    return {
        "executable": str(resolved),
        "executable_sha256": file_sha256(resolved),
        **facts,
        "packages": packages,
        "packages_sha256": canonical_sha256(packages),
    }


def build_environment(python: Path) -> dict[str, Any]:
    document = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "kind": "tgrad-upstream-calibration-environment",
        "python": python_facts(python),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "selectors": PINNED_ENVIRONMENT,
    }
    return attach_content_sha256(document)


def load_environment(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read environment manifest {path}: {error}") from error
    if not isinstance(document, dict):
        raise CalibrationError("environment manifest root must be an object")
    if document.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise CalibrationError("unsupported environment manifest schema")
    if document.get("kind") != "tgrad-upstream-calibration-environment":
        raise CalibrationError("unexpected environment manifest kind")
    recorded_hash = document.get("content_sha256")
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    if recorded_hash != canonical_sha256(body):
        raise CalibrationError("environment manifest content hash mismatch")
    actual = build_environment(Path(document["python"]["executable"]))
    if actual != document:
        raise CalibrationError(
            "current Python, packages, host, or selectors differ from environment manifest"
        )
    return document


def expected_files(target: dict[str, Any], checkout: Path, group: str) -> list[str]:
    checked = target["_checked"]
    names = checked["test_files"][group]
    expected = [f"test/{group}/{name}" for name in names]
    actual = sorted(
        f"test/{group}/{path.name}"
        for path in (checkout / "test" / group).glob("*.py")
        if path.name != "__init__.py"
    )
    if actual != expected:
        raise CalibrationError(
            "checkout test inventory differs from the generated target: "
            f"missing={sorted(set(expected) - set(actual))[:5]}, "
            f"extra={sorted(set(actual) - set(expected))[:5]}"
        )
    return expected


def _xml_counts(path: Path) -> tuple[dict[str, int], list[dict[str, str]]]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {"collected": 0, "failed": 0, "errors": 0, "skipped": 0, "passed": 0}
    cases: list[dict[str, str]] = []
    for suite in suites:
        counts["collected"] += int(suite.attrib.get("tests", "0"))
        counts["failed"] += int(suite.attrib.get("failures", "0"))
        counts["errors"] += int(suite.attrib.get("errors", "0"))
        counts["skipped"] += int(suite.attrib.get("skipped", "0"))
        for case in suite.iter("testcase"):
            status = "pass"
            if case.find("failure") is not None:
                status = "fail"
            elif case.find("error") is not None:
                status = "error"
            elif case.find("skipped") is not None:
                status = "skipped"
            cases.append(
                {
                    "node": f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}",
                    "status": status,
                }
            )
    counts["passed"] = (
        counts["collected"] - counts["failed"] - counts["errors"] - counts["skipped"]
    )
    return counts, cases


def _status(returncode: int, counts: dict[str, int], timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if counts["collected"] == 0:
        return "empty" if returncode in (0, 5) else "collect_error"
    if counts["errors"]:
        return "collect_error"
    if counts["failed"] or returncode:
        return "fail"
    if counts["skipped"]:
        return "pass_with_skips"
    return "pass"


def run_file(
    python: Path,
    checkout: Path,
    relative: str,
    timeout_seconds: int,
    artifact_root: Path,
    index: int,
) -> dict[str, Any]:
    stem = f"{index:03d}-{Path(relative).stem}"
    stdout_path = artifact_root / "stdout" / f"{stem}.txt"
    stderr_path = artifact_root / "stderr" / f"{stem}.txt"
    junit_path = artifact_root / "junit" / f"{stem}.xml"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-m",
        "pytest",
        relative,
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit_path}",
    ]
    environment = {key: value for key, value in os.environ.items() if key not in PINNED_ENVIRONMENT}
    environment.update(PINNED_ENVIRONMENT)
    environment["PYTHONPATH"] = str(checkout)
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=checkout,
            env=environment,
            capture_output=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        if isinstance(stdout, str):
            stdout = stdout.encode()
        if isinstance(stderr, str):
            stderr = stderr.encode()
    duration_ms = round((time.monotonic() - started) * 1000, 3)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if junit_path.is_file():
        try:
            counts, cases = _xml_counts(junit_path)
            junit_error = ""
        except (ET.ParseError, ValueError) as error:
            counts = {"collected": 0, "failed": 0, "errors": 0, "skipped": 0, "passed": 0}
            cases = []
            junit_error = str(error)
    else:
        counts = {"collected": 0, "failed": 0, "errors": 0, "skipped": 0, "passed": 0}
        cases = []
        junit_error = "pytest did not produce JUnit XML"
    status = _status(returncode, counts, timed_out)
    if junit_error and status not in ("timeout", "collect_error", "empty"):
        status = "runner_error"
    semantic = {
        "file": relative,
        "status": status,
        "returncode": returncode,
        "counts": counts,
        "cases": cases,
    }
    return {
        **semantic,
        "semantic_sha256": canonical_sha256(semantic),
        "duration_ms": duration_ms,
        "command": command,
        "junit_error": junit_error,
        "stdout": {"path": str(stdout_path.relative_to(artifact_root)), "sha256": file_sha256(stdout_path), "bytes": stdout_path.stat().st_size},
        "stderr": {"path": str(stderr_path.relative_to(artifact_root)), "sha256": file_sha256(stderr_path), "bytes": stderr_path.stat().st_size},
        "junit": (
            {"path": str(junit_path.relative_to(artifact_root)), "sha256": file_sha256(junit_path), "bytes": junit_path.stat().st_size}
            if junit_path.is_file()
            else None
        ),
    }


def result_semantics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": result["file"],
        "status": result["status"],
        "returncode": result["returncode"],
        "counts": result["counts"],
        "cases": result["cases"],
    }


def build_run(
    target_path: Path,
    checkout: Path,
    environment_path: Path,
    group: str,
    output: Path,
    timeout_seconds: int,
    limit: int,
) -> dict[str, Any]:
    try:
        target = load_checked(target_path)
    except (CoverageModelError, ValueError) as error:
        raise CalibrationError(str(error)) from error
    environment = load_environment(environment_path)
    before = checkout_identity(checkout, target["upstream_ref"])
    inventory = expected_files(target, checkout, group)
    selected = inventory[:limit] if limit else inventory
    if not selected:
        raise CalibrationError("selection is empty")
    if output.exists():
        raise CalibrationError(f"refusing to overwrite completed output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        results = [
            run_file(
                Path(environment["python"]["executable"]),
                checkout.resolve(),
                relative,
                timeout_seconds,
                staging,
                index,
            )
            for index, relative in enumerate(selected, 1)
        ]
        after = checkout_identity(checkout, target["upstream_ref"])
        if before != after:
            raise CalibrationError("upstream checkout identity changed during calibration")
        status_counts = {status: 0 for status in sorted(FILE_STATUSES)}
        for result in results:
            status_counts[result["status"]] += 1
        semantics = [result_semantics(result) for result in results]
        diagnostics = [
            {
                "file": result["file"],
                "stdout_sha256": result["stdout"]["sha256"],
                "stderr_sha256": result["stderr"]["sha256"],
                "junit_sha256": result["junit"]["sha256"] if result["junit"] else None,
            }
            for result in results
        ]
        complete = selected == inventory
        document = attach_content_sha256(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "kind": "tgrad-upstream-suite-calibration-run",
                "target": {
                    "manifest": target_path.name,
                    "manifest_content_sha256": target["content_sha256"],
                    "upstream_ref": target["upstream_ref"],
                    "checkout_tree": before["tree"],
                },
                "environment": {
                    "manifest": environment_path.name,
                    "content_sha256": environment["content_sha256"],
                },
                "selection": {
                    "group": group,
                    "inventory_count": len(inventory),
                    "selected_count": len(selected),
                    "complete": complete,
                    "inventory_sha256": canonical_sha256(inventory),
                    "selected_sha256": canonical_sha256(selected),
                },
                "scenario": {
                    "timeout_seconds": timeout_seconds,
                    "pytest_arguments": ["-q", "--no-header", "-p", "no:cacheprovider", "--junitxml=<per-file>"],
                    "selectors": PINNED_ENVIRONMENT,
                },
                "status_counts": status_counts,
                "outcome_sha256": canonical_sha256(semantics),
                "diagnostic_manifest_sha256": canonical_sha256(diagnostics),
                "results": results,
                "promotion_ready": False,
                "promotion_blockers": [
                    "requires a second complete run with the same outcome_sha256",
                    "non-pass and skipped cases require an explicit reviewed oracle disposition",
                ],
            }
        )
        _atomic_write(staging / "run.json", document)
        os.replace(staging, output)
        return document
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_run(path: Path) -> dict[str, Any]:
    artifact_root = path if path.is_dir() else path.parent
    run_path = path / "run.json" if path.is_dir() else path
    try:
        document = json.loads(run_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read calibration run {run_path}: {error}") from error
    if document.get("schema_version") != RUN_SCHEMA_VERSION or document.get("kind") != "tgrad-upstream-suite-calibration-run":
        raise CalibrationError("unexpected calibration run schema/kind")
    recorded = document.get("content_sha256")
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    if recorded != canonical_sha256(body):
        raise CalibrationError("calibration run content hash mismatch")
    results = document.get("results")
    if not isinstance(results, list):
        raise CalibrationError("calibration results must be a list")
    if len(results) != document["selection"]["selected_count"]:
        raise CalibrationError("result count does not match selected files")
    files = [result.get("file") for result in results if isinstance(result, dict)]
    if len(files) != len(results) or len(files) != len(set(files)):
        raise CalibrationError("result files are malformed or duplicated")
    recomputed_statuses = {status: 0 for status in sorted(FILE_STATUSES)}
    for index, result in enumerate(results):
        status = result.get("status")
        if status not in FILE_STATUSES:
            raise CalibrationError(f"results[{index}]: unknown status")
        recomputed_statuses[status] += 1
        semantic = result_semantics(result)
        if result.get("semantic_sha256") != canonical_sha256(semantic):
            raise CalibrationError(f"results[{index}]: semantic hash mismatch")
        for artifact_name in ("stdout", "stderr", "junit"):
            artifact = result.get(artifact_name)
            if artifact is None and artifact_name == "junit":
                continue
            if not isinstance(artifact, dict):
                raise CalibrationError(f"results[{index}].{artifact_name}: expected object")
            artifact_path = (artifact_root / artifact.get("path", "")).resolve()
            try:
                artifact_path.relative_to(artifact_root.resolve())
            except ValueError as error:
                raise CalibrationError(
                    f"results[{index}].{artifact_name}: artifact escapes run directory"
                ) from error
            if not artifact_path.is_file():
                raise CalibrationError(f"results[{index}].{artifact_name}: missing artifact")
            if artifact.get("sha256") != file_sha256(artifact_path):
                raise CalibrationError(f"results[{index}].{artifact_name}: hash mismatch")
            if artifact.get("bytes") != artifact_path.stat().st_size:
                raise CalibrationError(f"results[{index}].{artifact_name}: size mismatch")
            if artifact_name == "junit":
                try:
                    xml_counts, xml_cases = _xml_counts(artifact_path)
                except (ET.ParseError, ValueError) as error:
                    raise CalibrationError(
                        f"results[{index}].junit: cannot parse: {error}"
                    ) from error
                if xml_counts != result.get("counts") or xml_cases != result.get("cases"):
                    raise CalibrationError(
                        f"results[{index}].junit: parsed outcomes differ from manifest"
                    )
    if document["status_counts"] != recomputed_statuses:
        raise CalibrationError("status counts do not match result rows")
    if sum(document["status_counts"].values()) != document["selection"]["selected_count"]:
        raise CalibrationError("status counts do not account for selected files")
    semantics = [result_semantics(result) for result in results]
    if document["outcome_sha256"] != canonical_sha256(semantics):
        raise CalibrationError("calibration outcome hash mismatch")
    diagnostics = [
        {
            "file": result["file"],
            "stdout_sha256": result["stdout"]["sha256"],
            "stderr_sha256": result["stderr"]["sha256"],
            "junit_sha256": result["junit"]["sha256"] if result["junit"] else None,
        }
        for result in results
    ]
    if document.get("diagnostic_manifest_sha256") != canonical_sha256(diagnostics):
        raise CalibrationError("diagnostic manifest hash mismatch")
    return document


def compare_runs(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = load_run(first_path)
    second = load_run(second_path)
    identity_fields = ("target", "environment", "selection", "scenario")
    identity_equal = all(first[field] == second[field] for field in identity_fields)
    complete = first["selection"]["complete"] and second["selection"]["complete"]
    outcomes_equal = first["outcome_sha256"] == second["outcome_sha256"]
    first_by_file = {result["file"]: result for result in first["results"]}
    second_by_file = {result["file"]: result for result in second["results"]}
    transitions = []
    for file in sorted(set(first_by_file) | set(second_by_file)):
        left = first_by_file.get(file)
        right = second_by_file.get(file)
        if left is None or right is None or left["semantic_sha256"] != right["semantic_sha256"]:
            transitions.append(
                {
                    "file": file,
                    "first": left["status"] if left else "missing",
                    "second": right["status"] if right else "missing",
                }
            )
    nonpass_or_skipped = any(
        result["status"] != "pass" for result in first["results"] + second["results"]
    )
    blockers = []
    if not identity_equal:
        blockers.append("run identities differ")
    if not complete:
        blockers.append("one or both runs are partial")
    if not outcomes_equal:
        blockers.append("semantic outcomes are not repeatable")
    if nonpass_or_skipped:
        blockers.append("non-pass or skipped cases need reviewed oracle dispositions")
    return attach_content_sha256(
        {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "kind": "tgrad-upstream-suite-calibration-comparison",
            "first_run_content_sha256": first["content_sha256"],
            "second_run_content_sha256": second["content_sha256"],
            "identity_equal": identity_equal,
            "complete": complete,
            "outcomes_equal": outcomes_equal,
            "transitions": transitions,
            "promotion_ready": not blockers,
            "promotion_blockers": blockers,
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    environment = commands.add_parser("environment")
    environment.add_argument("--python", type=Path, required=True)
    environment.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    run.add_argument("--checkout", type=Path, required=True)
    run.add_argument("--environment", type=Path, required=True)
    run.add_argument("--group", choices=("null", "unit", "backend"), required=True)
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "environment":
            if args.output.exists():
                raise CalibrationError(f"refusing to overwrite {args.output}")
            _atomic_write(args.output, build_environment(args.python))
            print(f"wrote {args.output}")
        elif args.command == "run":
            build_run(
                args.target,
                args.checkout,
                args.environment,
                args.group,
                args.output,
                args.timeout,
                args.limit,
            )
            print(f"wrote {args.output}")
        else:
            if args.output.exists():
                raise CalibrationError(f"refusing to overwrite {args.output}")
            comparison = compare_runs(args.first, args.second)
            _atomic_write(args.output, comparison)
            print(f"wrote {args.output}")
        return 0
    except (CalibrationError, CoverageModelError, KeyError, TypeError, ValueError) as error:
        print(f"calibrate_upstream_suite: FAILED — {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a frozen slice of tinygrad's suite and emit revision-bound evidence.

The same observer supports two subjects:

  --against upstream   calibrates the selected contract against the pinned
                       tinygrad checkout.
  --against tgrad      runs that contract through Tgrad's strict substitution.

Use ``--oracle-class api_surface --group all --expect-files 34`` for the public
compatibility contract.  Classification is read from the reviewed upstream
manifest; it is never inferred from Tgrad's result.  Evidence is content
addressed and never silently overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "fixtures" / "parity"
OBSERVATION_DIR = Path(
    os.environ.get("TGRAD_PARITY_OUTPUT_DIR", "/tmp/tgrad-parity-observations")
)
DEFAULT_VENV_PY = Path(
    os.environ.get("TGRAD_PARITY_PYTHON", REPO / ".venv" / "bin" / "python")
)
DEFAULT_CHECKOUT = Path("/tmp/tg_oracle/tinygrad")
DEFAULT_CLASSIFICATION = OUT_DIR / "oracle_classification.json"
UPSTREAM_MANIFEST = OUT_DIR / "upstream_19c4d736f2bc.json"
SHIM_ROOT = Path(__file__).resolve().parent / "shim"
SHIM_RUNNER = SHIM_ROOT / "run_pytest.py"
REPORTER_ROOT = Path(__file__).resolve().parent
REPORTER_MODULE = "tgrad_pytest_reporter"
PRODUCT_PYTHON = REPO / "python"
PRODUCT_MODULE = PRODUCT_PYTHON / "tgrad.py"
RUNTIME_LIBRARY = REPO / ".lake" / "build" / "lib" / "libtgrad.dylib"
GROUPS = ("null", "unit", "backend")
ORACLE_CLASSES = ("api_surface", "internal_repr", "infrastructure", "ambiguous")
OUTCOMES = (
    "passed", "all_skipped", "nonconforming", "blocked_product_surface",
    "blocked_environment", "collection_error", "timeout", "empty",
    "verifier_error",
)
SCHEMA_VERSION = 2


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def git_blob(revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{revision}:{path}"],
        capture_output=True, check=True,
    ).stdout


def git_directory_hash(revision: str, root: str) -> str:
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", revision,
         "--", root],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    prefix = root.rstrip("/") + "/"
    entries = [
        (path.removeprefix(prefix), digest(git_blob(revision, path)))
        for path in sorted(path for path in listing if path)
    ]
    if not entries:
        raise RuntimeError(f"no tracked files beneath {root!r} at {revision}")
    return digest(canonical(entries))


def checkout_value(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def controlled_environment(against: str, checkout: Path,
                           isolated_root: Path) -> dict[str, str]:
    env = {
        "HOME": str(isolated_root / "home"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TMPDIR": str(isolated_root / "tmp"),
    }
    (isolated_root / "home").mkdir()
    (isolated_root / "tmp").mkdir()
    if against == "tgrad":
        env.update({
            "PYTHONPATH": os.pathsep.join(
                [str(SHIM_ROOT), str(PRODUCT_PYTHON), str(REPORTER_ROOT)]
            ),
            "TGRAD_ROOT": str(REPO),
            "TGRAD_LIB": str(RUNTIME_LIBRARY),
        })
    else:
        env["PYTHONPATH"] = os.pathsep.join([str(checkout), str(REPORTER_ROOT)])
    return env


def interpreter_facts(py: Path) -> dict:
    probe = r'''
import importlib.metadata
import hashlib
import json
import platform
import sys

selected = {"pytest", "numpy", "torch", "hypothesis", "einops"}
all_distributions = sorted(
    (dist.metadata.get("Name", "").lower(), dist.version)
    for dist in importlib.metadata.distributions()
    if dist.metadata.get("Name")
)
selected_records = {}
for name in selected:
    try:
        dist = importlib.metadata.distribution(name)
        record = dist.read_text("RECORD") or ""
        selected_records[name] = hashlib.sha256(record.encode()).hexdigest()
    except importlib.metadata.PackageNotFoundError:
        pass
print(json.dumps({
    "python": platform.python_version(),
    "implementation": platform.python_implementation(),
    "cache_tag": sys.implementation.cache_tag,
    "platform": platform.system(),
    "platform_release": platform.release(),
    "machine": platform.machine(),
    "runtime_executable": sys.executable,
    "selected_dependencies": {
        name: version for name, version in all_distributions if name in selected
    },
    "selected_dependency_record_sha256": selected_records,
    "distribution_manifest": all_distributions,
}, sort_keys=True))
'''.strip()
    completed = subprocess.run(
        [str(py), "-c", probe],
        env={"LANG": "C", "LC_ALL": "C", "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True, check=True, timeout=30,
    )
    facts = json.loads(completed.stdout)
    distributions = facts.pop("distribution_manifest")
    facts["distribution_manifest_sha256"] = digest(canonical(distributions))
    facts["launcher"] = str(py.absolute())
    facts["launcher_target"] = str(py.resolve())
    facts["launcher_target_sha256"] = file_hash(py.resolve())
    return facts


def backend_facts(py: Path, checkout: Path,
                  environment: dict[str, str]) -> dict:
    probe = (
        "import json; from tinygrad import Device; "
        "print(json.dumps({'default_device': str(Device.DEFAULT)}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(py), "-c", probe], cwd=checkout, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cannot identify selected backend: " +
            (completed.stdout + completed.stderr)[-2000:]
        )
    selected = json.loads(completed.stdout)
    hardware = tool_output(["system_profiler", "SPHardwareDataType", "-json"])
    hardware_doc = json.loads(hardware)
    entries = hardware_doc.get("SPHardwareDataType", [])
    first = entries[0] if entries else {}
    safe_hardware = {
        "machine_model": first.get("machine_model"),
        "machine_name": first.get("machine_name"),
        "chip_type": first.get("chip_type"),
    }
    return {
        **selected,
        "hardware": {
            **safe_hardware,
            "identity_sha256": digest(canonical(safe_hardware)),
        },
    }


def tool_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, check=True,
    )
    return (completed.stdout + completed.stderr).strip()


def rebuild_runtime(revision: str) -> dict:
    commands = [
        ["lake", "-H", "build", "Tgrad:shared"],
        ["make", "-B", "-C", "c", "dylib"],
    ]
    logs = []
    for command in commands:
        completed = subprocess.run(
            command, cwd=REPO, capture_output=True, text=True,
        )
        combined = completed.stdout + completed.stderr
        logs.append({
            "command": command,
            "exit_code": completed.returncode,
            "log_sha256": digest(combined.encode()),
        })
        if completed.returncode != 0:
            raise RuntimeError(
                "runtime provenance build failed: " + " ".join(command) +
                "\n" + combined[-4000:]
            )
    lean_library = REPO / ".lake" / "build" / "lib" / "libtgrad_Tgrad.dylib"
    inputs = {
        "lean_library_sha256": file_hash(lean_library),
        "metal_object_sha256": file_hash(REPO / "c" / "build" / "metal_alloc.o"),
        "metal_lean_object_sha256": file_hash(REPO / "c" / "build" / "metal_alloc_lean.o"),
        "python_bridge_object_sha256": file_hash(REPO / "c" / "build" / "tgrad_python.o"),
    }
    return {
        "status": "built_by_observer",
        "source_revision": revision,
        "source_tree": git_value("rev-parse", f"{revision}^{{tree}}"),
        "commands": commands,
        "command_logs": logs,
        "toolchain": {
            "lake": tool_output(["lake", "--version"]),
            "lean": tool_output(["lean", "--version"]),
            "clang": tool_output(["clang", "--version"]),
            "macos_sdk": tool_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"]),
        },
        "link_inputs": inputs,
        "artifact_sha256": file_hash(RUNTIME_LIBRARY),
    }


def load_contract(path: Path, checkout: Path, checkout_commit: str, group: str,
                  oracle_class: str | None) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    document = json.loads(raw)
    recorded_commit = document.get("upstream", {}).get("commit")
    if recorded_commit != checkout_commit:
        raise RuntimeError(
            f"classification targets {recorded_commit}, checkout is {checkout_commit}"
        )
    selected_groups = GROUPS if group == "all" else (group,)
    selected_paths = [
        f"test/{item['group']}/{item['filename']}"
        for item in document.get("files", [])
        if item.get("group") in selected_groups and
           (oracle_class is None or item.get("class") == oracle_class) and
           item.get("filename") != "__init__.py"
    ]
    if len(selected_paths) != len(set(selected_paths)):
        raise RuntimeError("classification selected duplicate test files")
    if oracle_class is not None:
        if group == "all":
            expected = document.get("totals", {}).get("by_class", {}).get(oracle_class)
        else:
            expected = (
                document.get("totals", {}).get("by_group", {}).get(group, {})
                .get("by_class", {}).get(oracle_class)
            )
        if expected != len(selected_paths):
            raise RuntimeError(
                f"classification totals claim {expected} {group}/{oracle_class} files; "
                f"selected {len(selected_paths)}"
            )
    selected = []
    for rel in sorted(selected_paths):
        source = checkout / rel
        if not source.is_file():
            raise RuntimeError(f"selected file missing from checkout: {rel}")
        selected.append({
            "path": rel,
            "group": Path(rel).parts[1],
            "source_sha256": file_hash(source),
        })
    group_counts = {
        selected_group: sum(item["group"] == selected_group for item in selected)
        for selected_group in GROUPS
        if any(item["group"] == selected_group for item in selected)
    }
    upstream_manifest = {
        "path": UPSTREAM_MANIFEST.relative_to(REPO).as_posix(),
        "sha256": file_hash(UPSTREAM_MANIFEST),
    } if UPSTREAM_MANIFEST.is_file() else None
    return selected, {
        "path": path.relative_to(REPO).as_posix(),
        "sha256": digest(raw),
        "schema_version": document.get("schema_version"),
        "class": oracle_class,
        "policy_unit": document.get("classification_policy", {}).get("unit"),
        "selection_sha256": digest(canonical(selected)),
        "file_count": len(selected),
        "group_counts": group_counts,
        "upstream_manifest": upstream_manifest,
    }


def normalize_output(output: str, checkout: Path) -> tuple[str, str]:
    normalized = re.sub(r"\x1b\[[0-9;]*m", "", output)
    replacements = (
        (str(checkout.resolve()), "<upstream-checkout>"),
        (str(REPO.resolve()), "<tgrad-repo>"),
        (str(Path.home().resolve()), "<home>"),
    )
    for source, replacement in replacements:
        normalized = normalized.replace(source, replacement)
    normalized = re.sub(
        r"(?:/private)?/tmp/tgrad_parity_observer_[^/\s]+",
        "<observer-root>", normalized,
    )
    normalized = re.sub(r"\b\d+(?:\.\d+)?s\b", "<duration>", normalized)
    normalized = re.sub(r"\bin \d+(?:\.\d+)? seconds?\b", "in <duration>", normalized)
    normalized = normalized.strip()
    lines = normalized.splitlines()
    if len(lines) > 160:
        excerpt = "\n".join(lines[:80] + ["<diagnostic-truncated>"] + lines[-80:])
    else:
        excerpt = normalized
    return digest(normalized.encode()), excerpt


def diagnostics(stdout: str, stderr: str, checkout: Path) -> dict:
    stdout_normalized_hash, stdout_excerpt = normalize_output(stdout, checkout)
    stderr_normalized_hash, stderr_excerpt = normalize_output(stderr, checkout)
    combined = "\n".join(part for part in (stdout_excerpt, stderr_excerpt) if part)
    return {
        "stdout_bytes": len(stdout.encode()),
        "stdout_raw_sha256": digest(stdout.encode()),
        "stdout_normalized_sha256": stdout_normalized_hash,
        "stderr_bytes": len(stderr.encode()),
        "stderr_raw_sha256": digest(stderr.encode()),
        "stderr_normalized_sha256": stderr_normalized_hash,
        "normalized_excerpt": combined,
    }


def read_report(path: Path) -> tuple[list[dict], str]:
    if not path.is_file():
        return [], ""
    raw = path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return events, digest(raw.encode())


def report_counts(events: list[dict]) -> tuple[dict, int, list[str]]:
    collection_count = 0
    collection_errors = 0
    phases_with_error: list[str] = []
    counts = {
        "passed": 0, "failed": 0, "skipped": 0,
        "xfailed": 0, "xpassed": 0, "errors": 0,
    }
    for event in events:
        if event.get("event") == "collection_finish":
            collection_count = int(event.get("count", 0))
        elif event.get("event") == "collection_error":
            collection_errors += 1
        elif event.get("event") == "test_report":
            phase = event.get("phase")
            outcome = event.get("outcome")
            was_xfail = "wasxfail" in event
            if phase == "call":
                if was_xfail and outcome == "skipped":
                    counts["xfailed"] += 1
                elif was_xfail and outcome == "passed":
                    counts["xpassed"] += 1
                elif outcome in {"passed", "failed", "skipped"}:
                    counts[outcome] += 1
            elif outcome == "failed":
                counts["errors"] += 1
                phases_with_error.append(str(phase))
            elif phase == "setup" and outcome == "skipped":
                counts["skipped"] += 1
    counts["collected"] = collection_count
    return counts, collection_errors, sorted(set(phases_with_error))


def classify_result(returncode: int, output: str, events: list[dict],
                    counts: dict, collection_errors: int,
                    error_phases: list[str]) -> tuple[str, str, list[str]]:
    missing = re.findall(
        r"(?:ModuleNotFoundError|ImportError): No module named ['\"]([^'\"]+)|"
        r"No module named ['\"]?([^'\"\s]+)",
        output,
    )
    missing = [left or right for left, right in missing]
    unsupported_surface = (
        "strict Tgrad substitution did not own the tinygrad package" in output or
        "Tgrad's strict shim does not provide" in output or
        "TgradCapabilityError" in output or
        "Tgrad does not provide tinygrad capability" in output
    )
    if not events:
        if any(name.split(".", 1)[0] not in {"tinygrad", "tgrad"} for name in missing):
            return "harness", "blocked_environment", ["external_dependency_missing"]
        return "harness", "verifier_error", ["missing_machine_report"]
    if collection_errors:
        if any(name.split(".", 1)[0] not in {"tinygrad", "tgrad"} for name in missing):
            return "collection", "blocked_environment", ["external_dependency_missing"]
        if unsupported_surface:
            reason = (
                "strict_fallback_rejected"
                if "strict Tgrad substitution did not own the tinygrad package" in output
                else "unsupported_surface"
            )
            return "collection", "blocked_product_surface", [reason]
        return "collection", "collection_error", ["import_time_exception"]
    if counts["errors"]:
        phase = "setup_teardown" if error_phases else "execution"
        if unsupported_surface:
            return phase, "blocked_product_surface", ["unsupported_surface"]
        return phase, "nonconforming", ["fixture_or_teardown_error", *error_phases]
    if counts["failed"]:
        if unsupported_surface:
            return "execution", "blocked_product_surface", ["unsupported_surface"]
        return "execution", "nonconforming", ["assertion_failure"]
    observed = counts["passed"] + counts["skipped"] + counts["xfailed"] + counts["xpassed"]
    if returncode == 0 and counts["passed"] + counts["xpassed"] > 0:
        return "execution", "passed", []
    if returncode == 0 and observed > 0:
        return "execution", "all_skipped", ["all_tests_skipped_or_xfailed"]
    if returncode in {0, 5} and counts["collected"] == 0:
        return "no_tests", "empty", ["no_tests_collected"]
    return "harness", "verifier_error", ["pytest_or_process_error"]


def run_file(py: Path, checkout: Path, rel: str, timeout: int,
             environment: dict[str, str], report_root: Path,
             pytest_runner: Path | None = None) -> dict:
    pytest_command = ["-m", "pytest"] if pytest_runner is None else [str(pytest_runner)]
    report_path = report_root / (digest(rel.encode()) + ".jsonl")
    file_environment = {**environment, "TGRAD_PYTEST_REPORT": str(report_path)}
    command = [
        str(py), *pytest_command, rel, "-q", "--no-header",
        "-p", "no:cacheprovider", "-p", REPORTER_MODULE,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=checkout, capture_output=True, text=True, timeout=timeout,
            env=file_environment,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "file": rel, "process": "timed_out", "returncode": None,
            "phase": "execution", "outcome": "timeout",
            "reason_codes": ["timeout"],
            "counts": {"passed": 0, "failed": 0, "skipped": 0,
                       "xfailed": 0, "xpassed": 0, "errors": 0,
                       "collected": 0},
            "report_sha256": file_hash(report_path) if report_path.is_file() else "",
            "diagnostics": diagnostics(stdout, stderr, checkout),
        }

    output = completed.stdout + completed.stderr
    events, report_hash = read_report(report_path)
    counts, collection_errors, error_phases = report_counts(events)
    phase, outcome, reason_codes = classify_result(
        completed.returncode, output, events, counts, collection_errors,
        error_phases,
    )
    process = "signaled" if completed.returncode < 0 else "exited"
    if completed.returncode < 0:
        reason_codes = ["signal", *reason_codes]
    return {
        "file": rel, "process": process, "returncode": completed.returncode,
        "phase": phase, "outcome": outcome, "reason_codes": reason_codes,
        "counts": counts,
        "report_sha256": report_hash,
        "diagnostics": diagnostics(completed.stdout, completed.stderr, checkout),
    }


def aggregate(results: list[dict]) -> dict:
    outcomes = sorted({result["outcome"] for result in results})
    return {
        "files": len(results),
        "files_by_outcome": {
            outcome: sum(result["outcome"] == outcome for result in results)
            for outcome in outcomes
        },
        "tests_passed": sum(result["counts"]["passed"] for result in results),
        "tests_failed": sum(result["counts"]["failed"] for result in results),
        "tests_skipped": sum(result["counts"]["skipped"] for result in results),
        "tests_xfailed": sum(result["counts"]["xfailed"] for result in results),
        "tests_xpassed": sum(result["counts"]["xpassed"] for result in results),
        "tests_errors": sum(result["counts"]["errors"] for result in results),
    }


def subject_identity(against: str, checkout: Path, checkout_commit: str,
                     runtime_provenance: dict | None = None) -> dict:
    if against == "upstream":
        return {
            "kind": "upstream",
            "revision": checkout_commit,
            "tree": checkout_value(checkout, "rev-parse", f"{checkout_commit}^{{tree}}"),
            "dirty": bool(checkout_value(checkout, "status", "--porcelain")),
        }
    dirty = git_value("status", "--porcelain")
    if dirty:
        raise RuntimeError(
            "refusing to observe Tgrad from a dirty tree; commit the observer and product first"
        )
    if runtime_provenance is None:
        raise RuntimeError("Tgrad observation requires observer-built runtime provenance")
    if not RUNTIME_LIBRARY.is_file():
        raise RuntimeError(f"missing observer-built runtime artifact: {RUNTIME_LIBRARY}")
    revision = git_value("rev-parse", "HEAD")
    source_components = {
        root: git_directory_hash(revision, root) for root in ("Tgrad", "python", "c")
    }
    return {
        "kind": "tgrad",
        "revision": revision,
        "tree": git_value("rev-parse", f"{revision}^{{tree}}"),
        "dirty": False,
        "product_sources": {
            "components": source_components,
            "combined_sha256": digest(canonical(source_components)),
            "product_module_sha256": file_hash(PRODUCT_MODULE),
        },
        "adapter": {
            "source_revision": revision,
            "content_sha256": git_directory_hash(revision, "scripts/parity/shim"),
        },
        "runtime": {
            "artifact_sha256": file_hash(RUNTIME_LIBRARY),
            "source_provenance": runtime_provenance,
        },
    }


def revalidate_inputs(identity: dict, checkout: Path,
                      selected: list[dict]) -> None:
    errors = []
    upstream = identity["upstream"]
    if checkout_value(checkout, "status", "--porcelain"):
        errors.append("upstream checkout became dirty")
    if checkout_value(checkout, "rev-parse", "HEAD") != upstream["revision"]:
        errors.append("upstream revision changed")
    if checkout_value(checkout, "rev-parse", "HEAD^{tree}") != upstream["tree"]:
        errors.append("upstream tree changed")
    for item in selected:
        if file_hash(checkout / item["path"]) != item["source_sha256"]:
            errors.append(f"upstream test changed: {item['path']}")
    contract = identity["contract"]
    if file_hash(REPO / contract["path"]) != contract["sha256"]:
        errors.append("oracle classification changed")
    manifest = contract.get("upstream_manifest")
    if manifest and file_hash(REPO / manifest["path"]) != manifest["sha256"]:
        errors.append("upstream inventory manifest changed")
    verifier = identity["verifier"]
    if file_hash(Path(__file__).resolve()) != verifier["runner_sha256"]:
        errors.append("suite observer changed")
    if file_hash(SHIM_RUNNER) != verifier["shim_runner_sha256"]:
        errors.append("strict shim runner changed")
    if file_hash(REPORTER_ROOT / f"{REPORTER_MODULE}.py") != verifier["reporter_sha256"]:
        errors.append("pytest reporter changed")
    subject = identity["subject"]
    if subject["kind"] == "tgrad":
        if git_value("status", "--porcelain"):
            errors.append("Tgrad tree became dirty")
        if git_value("rev-parse", "HEAD") != subject["revision"]:
            errors.append("Tgrad revision changed")
        if git_value("rev-parse", "HEAD^{tree}") != subject["tree"]:
            errors.append("Tgrad tree changed")
        if file_hash(PRODUCT_MODULE) != subject["product_sources"]["product_module_sha256"]:
            errors.append("Tgrad Python module changed")
        for root, expected in subject["product_sources"]["components"].items():
            if git_directory_hash(subject["revision"], root) != expected:
                errors.append(f"recorded product source hash drifted: {root}")
        if git_directory_hash(
            subject["revision"], "scripts/parity/shim"
        ) != subject["adapter"]["content_sha256"]:
            errors.append("strict adapter changed")
        if file_hash(RUNTIME_LIBRARY) != subject["runtime"]["artifact_sha256"]:
            errors.append("runtime artifact changed")
    if errors:
        raise RuntimeError("inputs changed during observation:\n  " + "\n  ".join(errors))


def validate_upstream_baseline(path: Path, identity: dict) -> dict:
    raw = path.read_bytes()
    baseline = json.loads(raw)
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("upstream baseline schema does not match observer")
    if baseline.get("against") != "upstream":
        raise RuntimeError("--upstream-baseline is not an upstream observation")
    baseline_identity = baseline.get("identity", {})
    for key in ("upstream", "contract"):
        if baseline_identity.get(key) != identity.get(key):
            raise RuntimeError(f"upstream baseline {key} does not match Tgrad scenario")
    baseline_environment = baseline_identity.get("environment", {})
    current_environment = identity.get("environment", {})
    if baseline_environment.get("facts") != current_environment.get("facts"):
        raise RuntimeError("upstream baseline Python/dependency environment differs")
    if baseline_environment.get("backend") != current_environment.get("backend"):
        raise RuntimeError("upstream baseline backend/hardware environment differs")
    common_policy = (
        "LANG", "LC_ALL", "PYTHONHASHSEED", "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH", "isolated_home", "isolated_tmp", "PATH",
        "path_sha256", "inherited_backend_overrides",
    )
    if any(
        baseline_environment.get("policy", {}).get(key) !=
        current_environment.get("policy", {}).get(key)
        for key in common_policy
    ):
        raise RuntimeError("upstream baseline controlled-environment policy differs")
    baseline_scenario = baseline_identity.get("scenario", {})
    current_scenario = identity.get("scenario", {})
    for key in ("group", "oracle_class", "files", "timeout_seconds", "pytest_flags"):
        if baseline_scenario.get(key) != current_scenario.get(key):
            raise RuntimeError(f"upstream baseline scenario field differs: {key}")
    if baseline_identity.get("verifier") != identity.get("verifier"):
        raise RuntimeError("upstream baseline used a different verifier")
    outcomes = baseline.get("observation", {}).get("aggregate", {}).get(
        "files_by_outcome", {}
    )
    unexpected = sorted(set(outcomes) - {"passed", "all_skipped"})
    if unexpected:
        raise RuntimeError(
            "upstream baseline is red for: " + ", ".join(unexpected)
        )
    return {
        "artifact_sha256": digest(raw),
        "result_id": baseline.get("result_id"),
        "run_artifact_id": baseline.get("run_artifact_id"),
    }


def write_evidence(document: dict, output: Path | None, scope: str) -> Path:
    stable_cells = [
        {key: value for key, value in cell.items() if key != "diagnostics"}
        for cell in document["observation"]["cells"]
    ]
    semantic_subject = json.loads(json.dumps(document["identity"]["subject"]))
    provenance = semantic_subject.get("runtime", {}).get("source_provenance")
    if isinstance(provenance, dict):
        provenance.pop("command_logs", None)
    document["scenario_id"] = document["identity"]["scenario"]["sha256"]
    document["result_id"] = digest(canonical({
        "scenario_id": document["scenario_id"],
        "subject": semantic_subject,
        "cells": stable_cells,
    }))
    document["run_artifact_id"] = digest(canonical(document))
    rendered = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    if output is None:
        subject = document["identity"]["subject"]["revision"][:12]
        upstream = document["identity"]["upstream"]["revision"][:12]
        output = OBSERVATION_DIR / (
            f"suite_observation_{document['against']}_{scope}_{upstream}_{subject}_"
            f"{document['result_id'][:12]}_{document['run_artifact_id'][:12]}.json"
        )
    if output.exists():
        if output.read_bytes() != rendered:
            raise RuntimeError(f"refusing to overwrite different evidence: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(rendered)
    except FileExistsError:
        if output.read_bytes() != rendered:
            raise RuntimeError(f"evidence path raced with different bytes: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--against", choices=["upstream", "tgrad"], required=True)
    parser.add_argument("--group", default="null", choices=["all", *GROUPS])
    parser.add_argument("--oracle-class", choices=ORACLE_CLASSES)
    parser.add_argument(
        "--canonical-api-surface", action="store_true",
        help="run exactly the reviewed 34-file api_surface contract",
    )
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_VENV_PY,
                        help="pytest environment (or set TGRAD_PARITY_PYTHON)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--expect-files", type=int, default=0)
    parser.add_argument(
        "--require-outcome", action="append", default=[], choices=OUTCOMES,
        help="after writing diagnostic evidence, return 3 if any file has another outcome",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--file", action="append", default=[], metavar="NAME")
    parser.add_argument("--list", action="store_true", help="print the frozen selection without running it")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upstream-baseline", type=Path)
    args = parser.parse_args()

    if args.canonical_api_surface:
        if args.file or args.limit:
            parser.error("canonical api_surface mode forbids --file and --limit")
        if args.group not in {"null", "all"}:
            parser.error("canonical api_surface mode spans all groups")
        if args.oracle_class not in {None, "api_surface"}:
            parser.error("canonical api_surface mode requires class api_surface")
        if args.expect_files not in {0, 34}:
            parser.error("canonical api_surface mode requires 34 files")
        args.group = "all"
        args.oracle_class = "api_surface"
        args.expect_files = 34
        if args.against == "upstream":
            args.require_outcome = sorted(
                set(args.require_outcome) | {"passed", "all_skipped"}
            )
        elif args.upstream_baseline is None:
            parser.error("canonical Tgrad mode requires --upstream-baseline")

    if not args.python.is_file():
        parser.error(f"missing venv python at {args.python}")
    if not args.checkout.is_dir():
        parser.error(f"missing checkout at {args.checkout}")
    checkout_commit = checkout_value(args.checkout, "rev-parse", "HEAD")
    if checkout_value(args.checkout, "status", "--porcelain"):
        parser.error("refusing to use a dirty upstream checkout")
    try:
        selected, contract = load_contract(
            args.classification.resolve(), args.checkout, checkout_commit,
            args.group, args.oracle_class,
        )
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    available = {Path(item["path"]).name: item for item in selected}
    if args.file:
        unknown = [name for name in args.file if Path(name).name not in available]
        if unknown:
            parser.error("files outside selected contract: " + ", ".join(unknown))
        selected = [available[Path(name).name] for name in args.file]
    if args.limit:
        if args.file:
            parser.error("--file and --limit cannot be used together")
        selected = selected[:args.limit]
    if args.expect_files and len(selected) != args.expect_files:
        parser.error(f"expected {args.expect_files} files, selected {len(selected)}")
    contract["executed_selection_sha256"] = digest(canonical(selected))
    contract["executed_file_count"] = len(selected)
    contract["selection_kind"] = (
        "canonical_api_surface" if args.canonical_api_surface else "explicit_subset"
    )
    if args.list:
        print("\n".join(item["path"] for item in selected))
        return 0

    runtime_provenance = None
    if args.against == "tgrad":
        if git_value("status", "--porcelain"):
            parser.error(
                "refusing to build or observe Tgrad from a dirty tree; "
                "commit the observer and product first"
            )
        try:
            runtime_provenance = rebuild_runtime(git_value("rev-parse", "HEAD"))
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
    try:
        subject = subject_identity(
            args.against, args.checkout, checkout_commit, runtime_provenance
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    with tempfile.TemporaryDirectory(
        prefix="tgrad_parity_observer_", dir="/tmp"
    ) as temp:
        environment = controlled_environment(
            args.against, args.checkout, Path(temp)
        )
        pytest_runner = None
        if args.against == "tgrad":
            pytest_runner = SHIM_RUNNER
            verify = subprocess.run(
                [str(args.python), str(SHIM_RUNNER), "--verify-only"],
                cwd=args.checkout, capture_output=True, text=True, env=environment,
            )
            if verify.returncode != 0:
                print(
                    "Tgrad substitution preflight failed; refusing to record evidence:\n"
                    + verify.stdout + verify.stderr,
                    file=sys.stderr,
                )
                return 2
            print(f"  {verify.stdout.strip()}")

        results = []
        for index, item in enumerate(selected, 1):
            result = run_file(
                args.python.absolute(), args.checkout, item["path"], args.timeout,
                environment, Path(temp) / "reports", pytest_runner,
            )
            result["source_sha256"] = item["source_sha256"]
            results.append(result)
            counts = result["counts"]
            print(
                f"  [{index:3d}/{len(selected)}] {result['outcome']:24s} "
                f"{item['path']}  ({counts['passed']}p/{counts['failed']}f/"
                f"{counts['skipped']}s/{counts['errors']}e)"
            )

        result_summary = aggregate(results)
        verifier_hash = file_hash(Path(__file__).resolve())
        identity = {
            "upstream": {
                "revision": checkout_commit,
                "tree": checkout_value(args.checkout, "rev-parse", f"{checkout_commit}^{{tree}}"),
            },
            "subject": subject,
            "verifier": {
                "runner_sha256": verifier_hash,
                "runner_revision": f"sha256:{verifier_hash}",
                "shim_runner_sha256": file_hash(SHIM_RUNNER),
                "reporter_sha256": file_hash(
                    REPORTER_ROOT / f"{REPORTER_MODULE}.py"
                ),
            },
            "contract": contract,
            "environment": {
                "facts": interpreter_facts(args.python.absolute()),
                "backend": backend_facts(
                    args.python.absolute(), args.checkout, environment
                ),
                "policy": {
                    "isolated_home": True,
                    "isolated_tmp": True,
                    "PATH": environment["PATH"],
                    "path_sha256": digest(environment["PATH"].encode()),
                    "python_path": [
                        component
                        .replace(str(REPO), "<tgrad-repo>")
                        .replace(str(args.checkout), "<upstream-checkout>")
                        for component in environment["PYTHONPATH"].split(os.pathsep)
                    ],
                    "tgrad_root": (
                        "<tgrad-repo>" if "TGRAD_ROOT" in environment else None
                    ),
                    "tgrad_lib": (
                        environment.get("TGRAD_LIB", "").replace(
                            str(REPO), "<tgrad-repo>"
                        ) or None
                    ),
                    "inherited_backend_overrides": [],
                    **{
                        key: environment[key]
                        for key in (
                            "LANG", "LC_ALL", "PYTHONHASHSEED",
                            "PYTHONNOUSERSITE", "PYTHONSAFEPATH",
                        )
                    },
                },
            },
            "scenario": {
                "group": args.group,
                "oracle_class": args.oracle_class,
                "files": selected,
                "timeout_seconds": args.timeout,
                "required_outcomes": sorted(set(args.require_outcome)),
                "pytest_flags": [
                    "-q", "--no-header", "-p", "no:cacheprovider",
                    "-p", REPORTER_MODULE,
                ],
            },
        }
        identity["environment"]["sha256"] = digest(canonical(identity["environment"]))
        identity["scenario"]["sha256"] = digest(canonical(identity["scenario"]))
        if args.against == "tgrad" and args.upstream_baseline is not None:
            try:
                identity["upstream_baseline"] = validate_upstream_baseline(
                    args.upstream_baseline.resolve(), identity
                )
            except (OSError, RuntimeError, json.JSONDecodeError) as exc:
                parser.error(str(exc))
        try:
            revalidate_inputs(identity, args.checkout, selected)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
        document = {
            "schema_version": SCHEMA_VERSION,
            "against": args.against,
            "identity": identity,
            "observation": {
                "aggregate": result_summary,
                "cells": results,
            },
        }
        scope = (
            "api_surface" if args.canonical_api_surface
            else "subset_" + contract["executed_selection_sha256"][:12]
        )
        output = write_evidence(document, args.output, scope)

    print()
    print(json.dumps(result_summary, indent=2, sort_keys=True))
    try:
        display_output = output.relative_to(REPO)
    except ValueError:
        display_output = output
    print(f"\nwrote {display_output}")
    allowed = set(args.require_outcome)
    unexpected = sorted(
        outcome for outcome in result_summary["files_by_outcome"]
        if allowed and outcome not in allowed
    )
    if unexpected:
        print(
            "unexpected outcome(s): " + ", ".join(unexpected),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

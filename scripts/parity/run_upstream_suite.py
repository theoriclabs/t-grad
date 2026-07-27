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
import ast
import fcntl
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import shutil
import io
import tarfile
from contextlib import contextmanager
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
    "passed", "unobserved_environment", "nonconforming",
    "blocked_product_surface", "blocked_environment", "mixed",
    "unobserved_upstream", "collection_mismatch", "collection_error", "timeout", "empty",
    "verifier_error",
)
SCHEMA_VERSION = 8


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


def snapshot_directory_hash(revision: str, git_root: str,
                            snapshot_root: Path) -> str:
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", revision,
         "--", git_root],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    prefix = git_root.rstrip("/") + "/"
    entries = []
    for path in sorted(path for path in listing if path):
        relative = path.removeprefix(prefix)
        source = snapshot_root / relative
        if not source.is_file():
            raise RuntimeError(f"execution snapshot omitted tracked file: {path}")
        entries.append((relative, file_hash(source)))
    return digest(canonical(entries))


def checkout_value(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def checkout_snapshot_hash(checkout: Path, revision: str,
                           snapshot: Path) -> str:
    paths = subprocess.run(
        ["git", "-C", str(checkout), "ls-tree", "-r", "--name-only", revision],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    entries = []
    for relative in sorted(path for path in paths if path):
        source = snapshot / relative
        if not source.is_file():
            raise RuntimeError(f"upstream execution snapshot omitted: {relative}")
        entries.append((relative, file_hash(source)))
    return digest(canonical(entries))


def controlled_environment(against: str, checkout: Path,
                           isolated_root: Path, reporter_root: Path,
                           shim_root: Path | None = None,
                           product_python: Path | None = None,
                           runtime_library: Path | None = None) -> dict[str, str]:
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
        if shim_root is None or product_python is None or runtime_library is None:
            raise RuntimeError("Tgrad execution snapshot is incomplete")
        env.update({
            "PYTHONPATH": os.pathsep.join(
                [str(shim_root), str(product_python), str(reporter_root), str(checkout)]
            ),
            "TGRAD_ROOT": str(isolated_root / "snapshot"),
            "TGRAD_LIB": str(runtime_library),
        })
    else:
        env["DEV"] = "METAL"
        env["PYTHONPATH"] = os.pathsep.join([str(checkout), str(reporter_root)])
    return env


@contextmanager
def observer_lock():
    lock_path = Path("/tmp/tgrad-parity-observer.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def copy_execution_snapshot(root: Path, against: str, checkout: Path,
                            checkout_commit: str) -> dict[str, Path]:
    snapshot = root / "snapshot"
    reporter = snapshot / "reporter"
    reporter.mkdir(parents=True)
    shutil.copyfile(
        REPORTER_ROOT / f"{REPORTER_MODULE}.py",
        reporter / f"{REPORTER_MODULE}.py",
    )
    upstream_checkout = snapshot / "upstream"
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", checkout_commit],
        capture_output=True, check=True,
    ).stdout
    upstream_checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(upstream_checkout, filter="data")
    result = {
        "reporter_root": reporter,
        "upstream_checkout": upstream_checkout,
    }
    if against == "tgrad":
        shim = snapshot / "shim"
        product = snapshot / "python"
        runtime = snapshot / "runtime"
        shutil.copytree(SHIM_ROOT, shim, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(PRODUCT_PYTHON, product, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        runtime.mkdir()
        runtime_library = runtime / RUNTIME_LIBRARY.name
        shutil.copyfile(RUNTIME_LIBRARY, runtime_library)
        shutil.copyfile(
            REPO / ".lake" / "build" / "lib" / "libtgrad_Tgrad.dylib",
            runtime / "libtgrad_Tgrad.dylib",
        )
        result.update({
            "shim_root": shim,
            "shim_runner": shim / "run_pytest.py",
            "product_python": product,
            "runtime_library": runtime_library,
        })
    return result


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
selected_files = {}
for name in selected:
    try:
        dist = importlib.metadata.distribution(name)
        record = dist.read_text("RECORD") or ""
        selected_records[name] = hashlib.sha256(record.encode()).hexdigest()
        files = []
        for entry in sorted(dist.files or [], key=str):
            # RECORD entries without a hash are generated/mutable artifacts
            # such as pyc files.  They are not installation inputs.
            if getattr(entry, "hash", None) is None:
                continue
            path = dist.locate_file(entry)
            if path.is_file():
                try:
                    files.append((str(entry), hashlib.sha256(path.read_bytes()).hexdigest()))
                except OSError:
                    files.append((str(entry), "<unreadable>"))
        selected_files[name] = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
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
    "selected_dependency_files_sha256": selected_files,
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


def upstream_backend_readiness(py: Path, checkout: Path,
                               environment: dict[str, str]) -> dict:
    probe = (
        "import json; from tinygrad import Tensor; "
        "value=Tensor([1.0]).realize().numpy().tolist(); "
        "print(json.dumps({'value': value}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(py), "-c", probe], cwd=checkout, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    output = completed.stdout + completed.stderr
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "diagnostic_sha256": normalize_output(output, checkout)[0],
        "observed": (
            json.loads(completed.stdout).get("value")
            if completed.returncode == 0 else None
        ),
    }


def prerequisite_facts(py: Path, environment: dict[str, str],
                       selected: list[dict]) -> dict:
    external_modules = sorted({
        module for item in selected for module in item.get("external_modules", [])
    })
    probe = r'''
import importlib.util
import json
import os
from multiprocessing import shared_memory
try:
    block = shared_memory.SharedMemory(create=True, size=1)
except BaseException as exc:
    result = {
        "available": False, "exception_type": type(exc).__name__,
        "errno": getattr(exc, "errno", None),
    }
else:
    block.close()
    block.unlink()
    result = {"available": True, "exception_type": None, "errno": None}
modules = json.loads(os.environ["TGRAD_EXTERNAL_MODULES"])
external = {}
for module in modules:
    try:
        spec = importlib.util.find_spec(module)
    except BaseException as exc:
        external[module] = {"available": False, "exception_type": type(exc).__name__}
    else:
        external[module] = {"available": spec is not None, "exception_type": None}
print(json.dumps({"posix_shared_memory": result, "external_modules": external}, sort_keys=True))
'''.strip()
    probe_environment = {
        **environment,
        "TGRAD_EXTERNAL_MODULES": json.dumps(external_modules, sort_keys=True),
    }
    completed = subprocess.run(
        [str(py), "-c", probe], env=probe_environment,
        capture_output=True, text=True, timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "environment prerequisite probe failed: " +
            (completed.stdout + completed.stderr)[-2000:]
        )
    return json.loads(completed.stdout)


def verifier_identity() -> dict:
    runner_hash = file_hash(Path(__file__).resolve())
    return {
        "runner_sha256": runner_hash,
        "runner_revision": f"sha256:{runner_hash}",
        "shim_runner_sha256": file_hash(SHIM_RUNNER),
        "reporter_sha256": file_hash(REPORTER_ROOT / f"{REPORTER_MODULE}.py"),
    }


def tool_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, check=True,
    )
    return (completed.stdout + completed.stderr).strip()


def executable_identity(name: str) -> dict:
    path = Path(tool_output(["which", name])).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def stable_repo_identity(expected_revision: str, expected_tree: str) -> None:
    if git_value("status", "--porcelain"):
        raise RuntimeError("Tgrad tree became dirty during runtime build")
    if git_value("rev-parse", "HEAD") != expected_revision:
        raise RuntimeError("Tgrad revision changed during runtime build")
    if git_value("rev-parse", "HEAD^{tree}") != expected_tree:
        raise RuntimeError("Tgrad tree changed during runtime build")


def rebuild_runtime(revision: str, source_tree: str) -> dict:
    commands = [
        ["lake", "-H", "build", "Tgrad:shared"],
        ["make", "-B", "-C", "c", "dylib"],
    ]
    build_environment = {
        key: value for key, value in {
            "HOME": os.environ.get("HOME"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C",
            "LC_ALL": "C",
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "ELAN_HOME": os.environ.get("ELAN_HOME"),
            "SDKROOT": os.environ.get("SDKROOT"),
            "TGRAD_MACOS_SDK": os.environ.get("TGRAD_MACOS_SDK"),
            "MACOSX_DEPLOYMENT_TARGET": os.environ.get("MACOSX_DEPLOYMENT_TARGET"),
        }.items() if value is not None
    }
    logs = []
    for command in commands:
        stable_repo_identity(revision, source_tree)
        completed = subprocess.run(
            command, cwd=REPO, capture_output=True, text=True,
            env=build_environment,
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
    stable_repo_identity(revision, source_tree)
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
        "source_tree": source_tree,
        "build_environment": {
            "values": build_environment,
            "sha256": digest(canonical(build_environment)),
        },
        "commands": commands,
        "command_logs": logs,
        "toolchain": {
            "lake": tool_output(["lake", "--version"]),
            "lean": tool_output(["lean", "--version"]),
            "clang": tool_output(["clang", "--version"]),
            "macos_sdk": tool_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"]),
            "lake_executable": executable_identity("lake"),
            "lean_executable": executable_identity("lean"),
            "clang_executable": executable_identity("clang"),
            "xcrun_executable": executable_identity("xcrun"),
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
        syntax = ast.parse(source.read_text(encoding="utf-8"), filename=rel)
        imported_roots = set()
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        checkout_local_roots = {
            entry.stem if entry.is_file() else entry.name
            for entry in checkout.iterdir()
            if entry.is_dir() or entry.suffix == ".py"
        }
        external_modules = sorted(
            imported_roots - set(sys.stdlib_module_names) -
            {"tinygrad", "test", "tests"} - checkout_local_roots
        )
        selected.append({
            "path": rel,
            "group": Path(rel).parts[1],
            "source_sha256": file_hash(source),
            "external_modules": external_modules,
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
    snapshot_replacements = (
        (r"(?:/private)?/tmp/tgrad_parity_observer_[^/\s]+/snapshot/upstream", "<upstream-checkout>"),
        (r"(?:/private)?/tmp/tgrad_parity_observer_[^/\s]+/snapshot/shim", "<tgrad-shim>"),
        (r"(?:/private)?/tmp/tgrad_parity_observer_[^/\s]+/snapshot/python", "<tgrad-product>"),
        (r"(?:/private)?/tmp/tgrad_parity_observer_[^/\s]+/snapshot/runtime", "<tgrad-runtime>"),
    )
    for pattern, replacement in snapshot_replacements:
        normalized = re.sub(pattern, replacement, normalized)
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
    stdout_artifact = persist_artifact("stdout", stdout.encode())
    stderr_artifact = persist_artifact("stderr", stderr.encode())
    return {
        "stdout_bytes": len(stdout.encode()),
        "stdout_raw_sha256": digest(stdout.encode()),
        "stdout_normalized_sha256": stdout_normalized_hash,
        "stderr_bytes": len(stderr.encode()),
        "stderr_raw_sha256": digest(stderr.encode()),
        "stderr_normalized_sha256": stderr_normalized_hash,
        "normalized_excerpt": combined,
        "raw_artifacts": {
            "stdout": stdout_artifact,
            "stderr": stderr_artifact,
        },
    }


def persist_artifact(kind: str, raw: bytes) -> dict:
    content_hash = digest(raw)
    destination = OBSERVATION_DIR / "artifacts" / kind / content_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            with destination.open("xb") as handle:
                handle.write(raw)
        except FileExistsError:
            pass
    if destination.read_bytes() != raw:
        raise RuntimeError(f"content-addressed artifact collision: {destination}")
    return {
        "sha256": content_hash,
        "bytes": len(raw),
        "path": destination.relative_to(OBSERVATION_DIR).as_posix(),
    }


def read_report(path: Path) -> tuple[list[dict], str, dict | None]:
    if not path.is_file():
        return [], "", None
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8")
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return events, digest(raw_bytes), persist_artifact("pytest-report", raw_bytes)


def analyze_report(events: list[dict], returncode: int) -> tuple[dict, dict]:
    collection_events = [e for e in events if e.get("event") == "collection_finish"]
    session_events = [e for e in events if e.get("event") == "session_finish"]
    protocol_errors: list[str] = []
    if len(collection_events) != 1:
        protocol_errors.append("collection_finish_count")
    if len(session_events) != 1:
        protocol_errors.append("session_finish_count")
    collection_event = collection_events[0] if len(collection_events) == 1 else {}
    nodeids = collection_event.get("nodeids", [])
    if not isinstance(nodeids, list) or not all(isinstance(nodeid, str) for nodeid in nodeids):
        protocol_errors.append("invalid_nodeid_manifest")
        nodeids = []
    if len(nodeids) != len(set(nodeids)):
        protocol_errors.append("duplicate_collected_nodeid")
    collection_count = int(collection_event.get("count", 0)) if collection_event else 0
    if collection_count != len(nodeids):
        protocol_errors.append("collection_count_mismatch")
    collection_cases = collection_event.get("cases", [])
    if not isinstance(collection_cases, list) or not all(
        isinstance(case, dict) and isinstance(case.get("nodeid"), str)
        for case in collection_cases
    ):
        protocol_errors.append("invalid_collection_case_manifest")
        collection_cases = []
    if sorted(case["nodeid"] for case in collection_cases) != sorted(nodeids):
        protocol_errors.append("collection_case_manifest_mismatch")
    if len(session_events) == 1 and int(session_events[0].get("exitstatus", -999)) != returncode:
        protocol_errors.append("session_exitstatus_mismatch")

    collection_errors = 0
    phases_with_error: list[str] = []
    failure_types: list[str] = []
    failed_reports = 0
    report_keys: set[tuple[str, str, str]] = set()
    terminal_nodeids: set[str] = set()
    terminal_cases: list[dict] = []
    case_results: list[dict] = []
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
            nodeid = str(event.get("nodeid", ""))
            phase = str(event.get("phase", ""))
            outcome = event.get("outcome")
            subtest_identity = json.dumps(
                event.get("subtest"), sort_keys=True, separators=(",", ":")
            )
            key = (nodeid, phase, subtest_identity)
            if key in report_keys:
                protocol_errors.append("duplicate_test_phase")
            report_keys.add(key)
            if nodeid not in set(nodeids):
                protocol_errors.append("report_for_uncollected_nodeid")
            if phase == "call" or (phase == "setup" and outcome in {"failed", "skipped"}):
                terminal_nodeids.add(nodeid)
                case_key = {
                    "nodeid": nodeid,
                    "subtest": event.get("subtest"),
                }
                terminal_cases.append(case_key)
                normalized_outcome = outcome
                if "wasxfail" in event:
                    normalized_outcome = (
                        "xfailed" if outcome == "skipped" else "xpassed"
                    )
                elif phase == "setup" and outcome == "failed":
                    normalized_outcome = "error"
                case_results.append({
                    "case": case_key,
                    "outcome": normalized_outcome,
                    "terminal_phase": phase,
                    "failure_type": event.get("failure_type"),
                })
            if event.get("failure_type"):
                failure_types.append(str(event["failure_type"]))
            if outcome == "failed":
                failed_reports += 1
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
    if not collection_errors:
        missing_terminal = set(nodeids) - terminal_nodeids
        if missing_terminal:
            protocol_errors.append("collected_test_without_terminal_outcome")
    return counts, {
        "collection_errors": collection_errors,
        "error_phases": sorted(set(phases_with_error)),
        "failure_types": sorted(set(failure_types)),
        "failed_reports": failed_reports,
        "nodeids": sorted(nodeids),
        "collection_cases": sorted(
            collection_cases, key=lambda case: canonical(case)
        ),
        "nodeid_manifest_sha256": digest(canonical(sorted(nodeids))),
        "collection_case_manifest_sha256": digest(canonical(sorted(
            collection_cases, key=lambda case: canonical(case)
        ))),
        "case_manifest_sha256": digest(canonical(sorted(
            terminal_cases, key=lambda case: canonical(case)
        ))),
        "case_count": len(terminal_cases),
        "case_results": sorted(case_results, key=lambda result: canonical(result["case"])),
        "case_outcome_manifest_sha256": digest(canonical(sorted(
            case_results, key=lambda result: canonical(result["case"])
        ))),
        "protocol_errors": sorted(set(protocol_errors)),
    }


def classify_result(returncode: int, output: str, events: list[dict],
                    counts: dict, report: dict, rel: str,
                    prerequisites: dict) -> tuple[str, str, list[str]]:
    collection_errors = report["collection_errors"]
    error_phases = report["error_phases"]
    if report["protocol_errors"]:
        return "harness", "verifier_error", [
            "invalid_machine_report", *report["protocol_errors"]
        ]
    missing = re.findall(
        r"(?:ModuleNotFoundError|ImportError): No module named ['\"]([^'\"]+)|"
        r"No module named ['\"]?([^'\"\s]+)",
        output,
    )
    missing = [left or right for left, right in missing]
    declared_unavailable = {
        name for name, fact in prerequisites.get("external_modules", {}).items()
        if not fact.get("available", False)
    }
    missing_due_to_environment = any(
        name.split(".", 1)[0] in declared_unavailable for name in missing
    )
    unsupported_surface = (
        "strict Tgrad substitution did not own the tinygrad package" in output or
        "Tgrad's strict shim does not provide" in output or
        "TgradCapabilityError" in output or
        "Tgrad does not provide tinygrad capability" in output
    )
    shared_memory_blocked = (
        rel == "test/unit/test_shm_tensor.py" and
        not prerequisites.get("posix_shared_memory", {}).get("available", False) and
        "PermissionError" in report["failure_types"]
    )
    environment_failures = ["posix_shared_memory_unavailable"] if shared_memory_blocked else []
    non_environment_failure_types = sorted(set(report["failure_types"]) - (
        {"PermissionError"} if shared_memory_blocked else set()
    ))
    semantic_failures = (
        bool(non_environment_failure_types) or
        report["failed_reports"] > len(environment_failures)
    )
    if environment_failures and unsupported_surface:
        return "mixed", "mixed", [
            "environment_and_product_surface", *environment_failures
        ]
    if environment_failures and semantic_failures:
        return "mixed", "mixed", [
            "environment_and_semantic_failure", *environment_failures,
            *non_environment_failure_types,
        ]
    if environment_failures:
        return "environment", "blocked_environment", [
            "runtime_environment_failure", *environment_failures
        ]
    if not events:
        if missing_due_to_environment:
            return "harness", "blocked_environment", ["external_dependency_missing"]
        if missing:
            return "harness", "nonconforming", ["undeclared_runtime_dependency_missing"]
        return "harness", "verifier_error", ["missing_machine_report"]
    if collection_errors:
        if missing_due_to_environment:
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
        return "environment", "unobserved_environment", [
            "all_tests_skipped_or_xfailed"
        ]
    if returncode in {0, 5} and counts["collected"] == 0:
        return "no_tests", "empty", ["no_tests_collected"]
    return "harness", "verifier_error", ["pytest_or_process_error"]


def run_file(py: Path, checkout: Path, rel: str, timeout: int,
             environment: dict[str, str], report_root: Path,
             prerequisites: dict,
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
            "report_artifact": (
                persist_artifact("pytest-report", report_path.read_bytes())
                if report_path.is_file() else None
            ),
            "nodeid_manifest_sha256": "",
            "nodeid_count": 0,
            "collection_case_manifest_sha256": "",
            "collection_cases": [],
            "case_manifest_sha256": "",
            "case_count": 0,
            "case_results": [],
            "case_outcome_manifest_sha256": "",
            "diagnostics": diagnostics(stdout, stderr, checkout),
        }

    output = completed.stdout + completed.stderr
    events, report_hash, report_artifact = read_report(report_path)
    counts, report = analyze_report(events, completed.returncode)
    phase, outcome, reason_codes = classify_result(
        completed.returncode, output, events, counts, report, rel, prerequisites,
    )
    process = "signaled" if completed.returncode < 0 else "exited"
    if completed.returncode < 0:
        reason_codes = ["signal", *reason_codes]
    return {
        "file": rel, "process": process, "returncode": completed.returncode,
        "phase": phase, "outcome": outcome, "reason_codes": reason_codes,
        "counts": counts,
        "report_sha256": report_hash,
        "report_artifact": report_artifact,
        "nodeid_manifest_sha256": report["nodeid_manifest_sha256"],
        "nodeid_count": len(report["nodeids"]),
        "collection_case_manifest_sha256": report[
            "collection_case_manifest_sha256"
        ],
        "collection_cases": report["collection_cases"],
        "case_manifest_sha256": report["case_manifest_sha256"],
        "case_count": report["case_count"],
        "case_results": report["case_results"],
        "case_outcome_manifest_sha256": report[
            "case_outcome_manifest_sha256"
        ],
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
                     runtime_provenance: dict | None = None,
                     runtime_library: Path = RUNTIME_LIBRARY) -> dict:
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
    if not runtime_library.is_file():
        raise RuntimeError(f"missing observer-built runtime artifact: {runtime_library}")
    revision = git_value("rev-parse", "HEAD")
    tree = git_value("rev-parse", f"{revision}^{{tree}}")
    if runtime_provenance.get("source_revision") != revision:
        raise RuntimeError("runtime source revision does not match observed subject")
    if runtime_provenance.get("source_tree") != tree:
        raise RuntimeError("runtime source tree does not match observed subject")
    source_components = {
        root: git_directory_hash(revision, root) for root in ("Tgrad", "python", "c")
    }
    return {
        "kind": "tgrad",
        "revision": revision,
        "tree": tree,
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
            "artifact_sha256": file_hash(runtime_library),
            "lean_artifact_sha256": file_hash(
                runtime_library.parent / "libtgrad_Tgrad.dylib"
            ),
            "source_provenance": runtime_provenance,
        },
    }


def revalidate_inputs(identity: dict, checkout: Path,
                      selected: list[dict], execution_snapshot: dict[str, Path],
                      py: Path, environment: dict[str, str]) -> None:
    errors = []
    upstream = identity["upstream"]
    if checkout_value(checkout, "status", "--porcelain"):
        errors.append("upstream checkout became dirty")
    if checkout_value(checkout, "rev-parse", "HEAD") != upstream["revision"]:
        errors.append("upstream revision changed")
    if checkout_value(checkout, "rev-parse", "HEAD^{tree}") != upstream["tree"]:
        errors.append("upstream tree changed")
    expected_snapshot_hash = checkout_snapshot_hash(
        checkout, upstream["revision"], execution_snapshot["upstream_checkout"]
    )
    if expected_snapshot_hash != identity["upstream"].get("snapshot_content_sha256"):
        errors.append("upstream execution snapshot content changed")
    for item in selected:
        if file_hash(checkout / item["path"]) != item["source_sha256"]:
            errors.append(f"upstream test changed: {item['path']}")
        if file_hash(
            execution_snapshot["upstream_checkout"] / item["path"]
        ) != item["source_sha256"]:
            errors.append(f"upstream execution snapshot differs: {item['path']}")
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
    if file_hash(
        execution_snapshot["reporter_root"] / f"{REPORTER_MODULE}.py"
    ) != verifier["reporter_sha256"]:
        errors.append("execution reporter differs from recorded verifier")
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
        if file_hash(execution_snapshot["runtime_library"]) != subject["runtime"]["artifact_sha256"]:
            errors.append("execution runtime artifact changed")
        if file_hash(
            execution_snapshot["runtime_library"].parent / "libtgrad_Tgrad.dylib"
        ) != subject["runtime"]["lean_artifact_sha256"]:
            errors.append("execution Lean runtime artifact changed")
        if file_hash(execution_snapshot["product_python"] / "tgrad.py") != \
                subject["product_sources"]["product_module_sha256"]:
            errors.append("execution product module differs from recorded source")
        if snapshot_directory_hash(
            subject["revision"], "python", execution_snapshot["product_python"]
        ) != subject["product_sources"]["components"]["python"]:
            errors.append("execution Python tree differs from recorded source")
        if snapshot_directory_hash(
            subject["revision"], "scripts/parity/shim", execution_snapshot["shim_root"]
        ) != subject["adapter"]["content_sha256"]:
            errors.append("execution adapter tree differs from recorded source")
        if file_hash(execution_snapshot["shim_runner"]) != verifier["shim_runner_sha256"]:
            errors.append("execution shim differs from recorded verifier")
    if errors:
        raise RuntimeError("inputs changed during observation:\n  " + "\n  ".join(errors))
    if interpreter_facts(py) != identity["environment"]["facts"]:
        raise RuntimeError("Python interpreter or installed dependencies changed during observation")
    if prerequisite_facts(py, environment, selected) != identity["environment"]["prerequisites"]:
        raise RuntimeError("environment prerequisites changed during observation")


def stable_result_payload(document: dict) -> dict:
    stable_cells = []
    for cell in document["observation"]["cells"]:
        stable = {key: value for key, value in cell.items() if key != "diagnostics"}
        diagnostics_doc = cell.get("diagnostics", {})
        stable["diagnostic_identity"] = {
            key: diagnostics_doc.get(key) for key in (
                "stdout_normalized_sha256", "stderr_normalized_sha256"
            )
        }
        stable_cells.append(stable)
    stable_identity = json.loads(json.dumps(document["identity"]))
    provenance = stable_identity.get("subject", {}).get("runtime", {}).get(
        "source_provenance"
    )
    if isinstance(provenance, dict):
        provenance.pop("command_logs", None)
    return {"identity": stable_identity, "cells": stable_cells}


def computed_result_id(document: dict) -> str:
    return digest(canonical(stable_result_payload(document)))


def computed_run_artifact_id(document: dict) -> str:
    copy = json.loads(json.dumps(document))
    copy.pop("run_artifact_id", None)
    return digest(canonical(copy))


def read_bound_artifact(evidence_path: Path, ref: dict) -> bytes:
    if not isinstance(ref, dict):
        raise RuntimeError("artifact reference is missing")
    relative = Path(str(ref.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"artifact path escapes evidence root: {relative}")
    root = evidence_path.parent.resolve()
    artifact = (root / relative).resolve()
    if artifact != root and root not in artifact.parents:
        raise RuntimeError(f"artifact path escapes evidence root: {relative}")
    if not artifact.is_file():
        raise RuntimeError(f"bound artifact is missing: {artifact}")
    raw = artifact.read_bytes()
    if ref.get("sha256") != digest(raw) or ref.get("bytes") != len(raw):
        raise RuntimeError(f"bound artifact identity differs: {artifact}")
    return raw


def diagnostics_from_artifacts(cell: dict, evidence_path: Path,
                               checkout: Path) -> tuple[dict, str, str]:
    refs = cell.get("diagnostics", {}).get("raw_artifacts", {})
    stdout_raw = read_bound_artifact(evidence_path, refs.get("stdout", {}))
    stderr_raw = read_bound_artifact(evidence_path, refs.get("stderr", {}))
    stdout = stdout_raw.decode("utf-8")
    stderr = stderr_raw.decode("utf-8")
    stdout_normalized_hash, stdout_excerpt = normalize_output(stdout, checkout)
    stderr_normalized_hash, stderr_excerpt = normalize_output(stderr, checkout)
    combined = "\n".join(part for part in (stdout_excerpt, stderr_excerpt) if part)
    recomputed = {
        "stdout_bytes": len(stdout_raw),
        "stdout_raw_sha256": digest(stdout_raw),
        "stdout_normalized_sha256": stdout_normalized_hash,
        "stderr_bytes": len(stderr_raw),
        "stderr_raw_sha256": digest(stderr_raw),
        "stderr_normalized_sha256": stderr_normalized_hash,
        "normalized_excerpt": combined,
        "raw_artifacts": refs,
    }
    return recomputed, stdout, stderr


def replay_upstream_cell(cell: dict, evidence_path: Path, checkout: Path,
                         prerequisites: dict) -> dict:
    report_ref = cell.get("report_artifact")
    report_raw = read_bound_artifact(evidence_path, report_ref)
    events = [
        json.loads(line) for line in report_raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    returncode = cell.get("returncode")
    if not isinstance(returncode, int):
        raise RuntimeError(f"upstream baseline has no replayable return code: {cell.get('file')}")
    counts, report = analyze_report(events, returncode)
    diagnostics_doc, stdout, stderr = diagnostics_from_artifacts(
        cell, evidence_path, checkout
    )
    phase, outcome, reason_codes = classify_result(
        returncode, stdout + stderr, events, counts, report,
        cell.get("file", ""), prerequisites,
    )
    process = "signaled" if returncode < 0 else "exited"
    if returncode < 0:
        reason_codes = ["signal", *reason_codes]
    return {
        "file": cell.get("file"),
        "process": process,
        "returncode": returncode,
        "phase": phase,
        "outcome": outcome,
        "reason_codes": reason_codes,
        "counts": counts,
        "report_sha256": digest(report_raw),
        "report_artifact": report_ref,
        "nodeid_manifest_sha256": report["nodeid_manifest_sha256"],
        "nodeid_count": len(report["nodeids"]),
        "collection_case_manifest_sha256": report[
            "collection_case_manifest_sha256"
        ],
        "collection_cases": report["collection_cases"],
        "case_manifest_sha256": report["case_manifest_sha256"],
        "case_count": report["case_count"],
        "case_results": report["case_results"],
        "case_outcome_manifest_sha256": report[
            "case_outcome_manifest_sha256"
        ],
        "source_sha256": cell.get("source_sha256"),
        "diagnostics": diagnostics_doc,
    }


def validate_upstream_baseline(path: Path, identity: dict,
                               checkout: Path) -> tuple[dict, dict[str, dict]]:
    raw = path.read_bytes()
    baseline = json.loads(raw)
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("upstream baseline schema does not match observer")
    if baseline.get("against") != "upstream":
        raise RuntimeError("--upstream-baseline is not an upstream observation")
    baseline_identity = baseline.get("identity", {})
    if not baseline_identity.get("upstream", {}).get("snapshot_content_sha256"):
        raise RuntimeError("upstream baseline lacks immutable snapshot identity")
    subject = baseline_identity.get("subject", {})
    if subject.get("kind") != "upstream" or subject.get("dirty") is not False:
        raise RuntimeError("upstream baseline subject identity is invalid")
    if (
        subject.get("revision") != baseline_identity.get("upstream", {}).get("revision") or
        subject.get("tree") != baseline_identity.get("upstream", {}).get("tree")
    ):
        raise RuntimeError("upstream baseline subject does not match pinned upstream")
    for key in ("upstream", "contract"):
        if baseline_identity.get(key) != identity.get(key):
            raise RuntimeError(f"upstream baseline {key} does not match Tgrad scenario")
    baseline_environment = baseline_identity.get("environment", {})
    current_environment = identity.get("environment", {})
    for label, environment_doc in (
        ("upstream baseline", baseline_environment),
        ("Tgrad observation", current_environment),
    ):
        if environment_doc.get("sha256") != digest(canonical({
            key: value for key, value in environment_doc.items()
            if key != "sha256"
        })):
            raise RuntimeError(f"{label} environment hash is inconsistent")
    if not baseline_environment.get("oracle_backend_readiness", {}).get(
        "available", False
    ):
        raise RuntimeError("upstream baseline backend readiness was not established")
    if baseline_environment.get("facts") != current_environment.get("facts"):
        raise RuntimeError("upstream baseline Python/dependency environment differs")
    if baseline_environment.get("prerequisites") != current_environment.get("prerequisites"):
        raise RuntimeError("upstream baseline prerequisite environment differs")
    baseline_backend = baseline_environment.get("backend", {})
    current_backend = current_environment.get("backend", {})
    if baseline_backend.get("hardware") != current_backend.get("hardware"):
        raise RuntimeError("upstream baseline hardware environment differs")
    backend_profile = identity.get("scenario", {}).get(
        "backend_profile", {}
    )
    if baseline_backend.get("default_device") != backend_profile.get("upstream"):
        raise RuntimeError("upstream baseline did not use the declared oracle backend")
    if current_backend.get("default_device") != backend_profile.get("tgrad"):
        raise RuntimeError("Tgrad did not use the declared subject backend")
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
    for key in (
        "group", "oracle_class", "files", "timeout_seconds", "pytest_flags",
        "backend_profile",
    ):
        if baseline_scenario.get(key) != current_scenario.get(key):
            raise RuntimeError(f"upstream baseline scenario field differs: {key}")
    if baseline_identity.get("verifier") != identity.get("verifier"):
        raise RuntimeError("upstream baseline used a different verifier")
    scenario = baseline_identity.get("scenario", {})
    if scenario.get("sha256") != digest(canonical({
        key: value for key, value in scenario.items() if key != "sha256"
    })):
        raise RuntimeError("upstream baseline scenario hash is inconsistent")
    if baseline.get("scenario_id") != scenario.get("sha256"):
        raise RuntimeError("upstream baseline scenario_id is inconsistent")
    cells = baseline.get("observation", {}).get("cells")
    if not isinstance(cells, list) or len(cells) != 34:
        raise RuntimeError("upstream baseline must contain exactly 34 cells")
    by_file = {}
    expected_files = {item["path"]: item for item in scenario.get("files", [])}
    for cell in cells:
        name = cell.get("file")
        if not isinstance(name, str) or name in by_file:
            raise RuntimeError("upstream baseline has missing or duplicate file cells")
        if name not in expected_files:
            raise RuntimeError(f"upstream baseline has an unexpected cell: {name}")
        if cell.get("source_sha256") != expected_files[name].get("source_sha256"):
            raise RuntimeError(f"upstream baseline source hash differs: {name}")
        if not cell.get("nodeid_manifest_sha256"):
            raise RuntimeError(f"upstream baseline lacks collection identity: {name}")
        if not cell.get("collection_case_manifest_sha256"):
            raise RuntimeError(f"upstream baseline lacks collection case identity: {name}")
        if not cell.get("case_manifest_sha256"):
            raise RuntimeError(f"upstream baseline lacks executed-case identity: {name}")
        if not cell.get("case_outcome_manifest_sha256"):
            raise RuntimeError(f"upstream baseline lacks case outcome identity: {name}")
        replayed = replay_upstream_cell(
            cell, path, checkout,
            baseline_identity.get("environment", {}).get("prerequisites", {}),
        )
        if replayed != cell:
            differing = sorted(
                key for key in set(replayed) | set(cell)
                if replayed.get(key) != cell.get(key)
            )
            raise RuntimeError(
                f"upstream baseline cell cannot be replayed: {name}: " +
                ", ".join(differing)
            )
        by_file[name] = cell
    if set(by_file) != set(expected_files):
        raise RuntimeError("upstream baseline cell set differs from scenario")
    recorded_aggregate = baseline.get("observation", {}).get("aggregate")
    if recorded_aggregate != aggregate(cells):
        raise RuntimeError("upstream baseline aggregate is inconsistent with cells")
    if baseline.get("result_id") != computed_result_id(baseline):
        raise RuntimeError("upstream baseline result_id is inconsistent")
    if baseline.get("run_artifact_id") != computed_run_artifact_id(baseline):
        raise RuntimeError("upstream baseline run_artifact_id is inconsistent")
    if not baseline.get("result_id") or not baseline.get("run_artifact_id"):
        raise RuntimeError("upstream baseline lacks content identities")
    outcomes = recorded_aggregate.get("files_by_outcome", {})
    unexpected = sorted(
        set(outcomes) - {"passed", "unobserved_environment", "blocked_environment"}
    )
    if unexpected:
        raise RuntimeError(
            "upstream baseline is red for: " + ", ".join(unexpected)
        )
    reference = {
        "artifact_sha256": digest(raw),
        "result_id": baseline.get("result_id"),
        "run_artifact_id": baseline.get("run_artifact_id"),
        "oracle_eligible_cases": {
            name: [
                result["case"] for result in cell["case_results"]
                if result["outcome"] == "passed"
            ]
            for name, cell in sorted(by_file.items())
        },
        "oracle_eligible_case_count": sum(
            result["outcome"] == "passed"
            for cell in by_file.values() for result in cell["case_results"]
        ),
        "upstream_unobserved_case_count": sum(
            result["outcome"] != "passed"
            for cell in by_file.values() for result in cell["case_results"]
        ),
    }
    return reference, by_file


def apply_upstream_oracle(result: dict, expected: dict) -> None:
    eligible = {
        canonical(case_result["case"]): case_result
        for case_result in expected["case_results"]
        if case_result["outcome"] == "passed"
    }
    if not eligible:
        prior = result["outcome"]
        result["outcome"] = "unobserved_upstream"
        result["phase"] = "oracle"
        result["reason_codes"] = sorted(set([
            *result["reason_codes"],
            "upstream_has_no_passed_cases",
            f"subject_outcome:{prior}",
        ]))
        result["upstream_oracle"] = {
            "eligible_case_count": 0,
            "matched_case_count": 0,
            "passed_case_count": 0,
            "nonpassing_case_count": 0,
            "missing_case_count": 0,
            "descriptor_mismatch_count": 0,
        }
        return
    current_cases = {
        canonical(case_result["case"]): case_result
        for case_result in result["case_results"]
    }
    expected_collection = {
        descriptor["nodeid"]: descriptor
        for descriptor in expected["collection_cases"]
    }
    current_collection = {
        descriptor["nodeid"]: descriptor
        for descriptor in result["collection_cases"]
    }
    eligible_nodeids = {
        case_result["case"]["nodeid"] for case_result in eligible.values()
    }
    missing = sorted(key.decode() for key in eligible if key not in current_cases)
    descriptor_mismatch = sorted(
        nodeid for nodeid in eligible_nodeids
        if current_collection.get(nodeid) != expected_collection.get(nodeid)
    )
    result["upstream_oracle"] = {
        "eligible_case_count": len(eligible),
        "eligible_case_manifest_sha256": digest(canonical(sorted(
            [case_result["case"] for case_result in eligible.values()],
            key=lambda case: canonical(case),
        ))),
        "matched_case_count": sum(key in current_cases for key in eligible),
        "passed_case_count": sum(
            key in current_cases and current_cases[key]["outcome"] == "passed"
            for key in eligible
        ),
        "nonpassing_case_count": sum(
            key in current_cases and current_cases[key]["outcome"] != "passed"
            for key in eligible
        ),
        "missing_case_count": len(missing),
        "descriptor_mismatch_count": len(descriptor_mismatch),
    }
    if missing or descriptor_mismatch:
        prior = result["outcome"]
        if prior == "passed":
            result["outcome"] = "collection_mismatch"
        result["phase"] = "collection"
        result["reason_codes"] = sorted(set([
            *result["reason_codes"],
            "upstream_eligible_case_scope_differs",
            *(["missing_eligible_cases"] if missing else []),
            *(["eligible_case_descriptor_differs"] if descriptor_mismatch else []),
            f"prior_outcome:{prior}",
        ]))
        return
    nonpassing = [
        current_cases[key] for key in eligible
        if current_cases[key]["outcome"] != "passed"
    ]
    if nonpassing:
        prior = result["outcome"]
        if prior == "passed":
            result["outcome"] = "nonconforming"
            result["phase"] = "execution"
        result["reason_codes"] = sorted(set([
            *result["reason_codes"],
            "upstream_passed_case_not_passed_by_subject",
        ]))
    else:
        result["outcome"] = "passed"
        result["phase"] = "execution"
        result["reason_codes"] = []


def oracle_case_summary(results: list[dict], baseline: dict) -> dict:
    cells = [result.get("upstream_oracle", {}) for result in results]
    return {
        "upstream_eligible_case_count": baseline["oracle_eligible_case_count"],
        "upstream_unobserved_case_count": baseline[
            "upstream_unobserved_case_count"
        ],
        "subject_matched_case_count": sum(
            cell.get("matched_case_count", 0) for cell in cells
        ),
        "subject_passed_case_count": sum(
            cell.get("passed_case_count", 0) for cell in cells
        ),
        "subject_nonpassing_case_count": sum(
            cell.get("nonpassing_case_count", 0) for cell in cells
        ),
        "subject_missing_case_count": sum(
            cell.get("missing_case_count", 0) for cell in cells
        ),
        "subject_descriptor_mismatch_count": sum(
            cell.get("descriptor_mismatch_count", 0) for cell in cells
        ),
    }


def validate_tgrad_observation(path: Path, document: dict,
                               upstream_path: Path, checkout: Path) -> None:
    identity = document.get("identity", {})
    subject = identity.get("subject", {})
    if subject.get("kind") != "tgrad" or subject.get("dirty") is not False:
        raise RuntimeError("Tgrad observation subject identity is invalid")
    baseline_ref, baseline_cells = validate_upstream_baseline(
        upstream_path, identity, checkout
    )
    if identity.get("upstream_baseline") != baseline_ref:
        raise RuntimeError("Tgrad observation baseline reference is inconsistent")
    cells = document.get("observation", {}).get("cells", [])
    by_file = {}
    for stored in cells:
        name = stored.get("file")
        if name in by_file or name not in baseline_cells:
            raise RuntimeError("Tgrad observation has missing or duplicate file cells")
        replayed = replay_upstream_cell(
            stored, path, checkout,
            identity.get("environment", {}).get("prerequisites", {}),
        )
        apply_upstream_oracle(replayed, baseline_cells[name])
        if replayed != stored:
            differing = sorted(
                key for key in set(replayed) | set(stored)
                if replayed.get(key) != stored.get(key)
            )
            raise RuntimeError(
                f"Tgrad observation cell cannot be replayed: {name}: " +
                ", ".join(differing)
            )
        by_file[name] = stored
    if set(by_file) != set(baseline_cells):
        raise RuntimeError("Tgrad observation file set differs from upstream baseline")
    if document.get("observation", {}).get("aggregate") != aggregate(cells):
        raise RuntimeError("Tgrad observation aggregate is inconsistent with cells")
    if document.get("observation", {}).get("oracle_cases") != oracle_case_summary(
        cells, baseline_ref
    ):
        raise RuntimeError("Tgrad observation oracle-case summary is inconsistent")


def write_evidence(document: dict, output: Path | None, scope: str) -> Path:
    document["scenario_id"] = document["identity"]["scenario"]["sha256"]
    document["result_id"] = computed_result_id(document)
    document["run_artifact_id"] = computed_run_artifact_id(document)
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
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="per-file timeout; canonical calibration uses 600 seconds",
    )
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
                set(args.require_outcome) |
                {"passed", "unobserved_environment", "blocked_environment"}
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

    with observer_lock(), tempfile.TemporaryDirectory(
        prefix="tgrad_parity_observer_", dir="/tmp"
    ) as temp:
        temp_root = Path(temp)
        runtime_provenance = None
        if args.against == "tgrad":
            if git_value("status", "--porcelain"):
                parser.error(
                    "refusing to build or observe Tgrad from a dirty tree; "
                    "commit the observer and product first"
                )
            try:
                build_revision = git_value("rev-parse", "HEAD")
                build_tree = git_value("rev-parse", f"{build_revision}^{{tree}}")
                runtime_provenance = rebuild_runtime(build_revision, build_tree)
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                parser.error(str(exc))
        execution_snapshot = copy_execution_snapshot(
            temp_root, args.against, args.checkout, checkout_commit
        )
        execution_checkout = execution_snapshot["upstream_checkout"]
        upstream_snapshot_hash = checkout_snapshot_hash(
            args.checkout, checkout_commit, execution_checkout
        )
        try:
            subject = subject_identity(
                args.against, args.checkout, checkout_commit, runtime_provenance,
                execution_snapshot.get("runtime_library", RUNTIME_LIBRARY),
            )
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
        environment = controlled_environment(
            args.against, execution_checkout, temp_root,
            execution_snapshot["reporter_root"],
            execution_snapshot.get("shim_root"),
            execution_snapshot.get("product_python"),
            execution_snapshot.get("runtime_library"),
        )
        prerequisites = prerequisite_facts(
            args.python.absolute(), environment, selected
        )
        oracle_backend = (
            upstream_backend_readiness(
                args.python.absolute(), execution_checkout, environment
            ) if args.against == "upstream" else None
        )
        if (
            args.canonical_api_surface and args.against == "upstream" and
            not oracle_backend.get("available", False)
        ):
            parser.error(
                "declared upstream Metal backend failed its readiness probe; "
                "refusing to calibrate the contract in this environment"
            )
        verifier = verifier_identity()
        environment_identity = {
            "facts": interpreter_facts(args.python.absolute()),
            "backend": backend_facts(
                args.python.absolute(), execution_checkout, environment
            ),
            "prerequisites": prerequisites,
            "oracle_backend_readiness": oracle_backend,
            "policy": {
                "isolated_home": True,
                "isolated_tmp": True,
                "PATH": environment["PATH"],
                "path_sha256": digest(environment["PATH"].encode()),
                "python_path": [
                    component
                    .replace(str(REPO), "<tgrad-repo>")
                    .replace(str(args.checkout), "<upstream-checkout>")
                    .replace(str(temp_root), "<execution-root>")
                    for component in environment["PYTHONPATH"].split(os.pathsep)
                ],
                "tgrad_root": (
                    "<execution-root>/snapshot"
                    if "TGRAD_ROOT" in environment else None
                ),
                "tgrad_lib": (
                    environment.get("TGRAD_LIB", "")
                    .replace(str(temp_root), "<execution-root>") or None
                ),
                "inherited_backend_overrides": [],
                "controlled_backend_selector": environment.get("DEV"),
                **{
                    key: environment[key]
                    for key in (
                        "LANG", "LC_ALL", "PYTHONHASHSEED",
                        "PYTHONNOUSERSITE", "PYTHONSAFEPATH",
                    )
                },
            },
        }
        environment_identity["sha256"] = digest(canonical(environment_identity))
        pytest_runner = None
        if args.against == "tgrad":
            pytest_runner = execution_snapshot["shim_runner"]
            verify = subprocess.run(
                [str(args.python), str(pytest_runner), "--verify-only"],
                cwd=execution_checkout, capture_output=True, text=True, env=environment,
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
                args.python.absolute(), execution_checkout, item["path"], args.timeout,
                environment, Path(temp) / "reports", prerequisites, pytest_runner,
            )
            result["source_sha256"] = item["source_sha256"]
            results.append(result)
            counts = result["counts"]
            print(
                f"  [{index:3d}/{len(selected)}] {result['outcome']:24s} "
                f"{item['path']}  ({counts['passed']}p/{counts['failed']}f/"
                f"{counts['skipped']}s/{counts['errors']}e)"
            )

        identity = {
            "upstream": {
                "revision": checkout_commit,
                "tree": checkout_value(args.checkout, "rev-parse", f"{checkout_commit}^{{tree}}"),
                "snapshot_content_sha256": upstream_snapshot_hash,
            },
            "subject": subject,
            "verifier": verifier,
            "contract": contract,
            "acceptance": {
                "allowed_file_outcomes": sorted(set(args.require_outcome)),
            },
            "environment": environment_identity,
            "scenario": {
                "group": args.group,
                "oracle_class": args.oracle_class,
                "files": selected,
                "timeout_seconds": args.timeout,
                "pytest_flags": [
                    "-q", "--no-header", "-p", "no:cacheprovider",
                    "-p", REPORTER_MODULE,
                ],
                "backend_profile": {
                    "upstream": "METAL",
                    "tgrad": "METAL",
                    "relation": "same_backend_public_observations",
                },
            },
        }
        identity["scenario"]["sha256"] = digest(canonical(identity["scenario"]))
        if args.against == "tgrad" and args.upstream_baseline is not None:
            try:
                baseline_ref, baseline_cells = validate_upstream_baseline(
                    args.upstream_baseline.resolve(), identity, args.checkout
                )
                identity["upstream_baseline"] = baseline_ref
                for result in results:
                    expected = baseline_cells[result["file"]]
                    apply_upstream_oracle(result, expected)
            except (OSError, RuntimeError, json.JSONDecodeError) as exc:
                parser.error(str(exc))
        try:
            revalidate_inputs(
                identity, args.checkout, selected, execution_snapshot,
                args.python.absolute(), environment,
            )
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
        document = {
            "schema_version": SCHEMA_VERSION,
            "against": args.against,
            "identity": identity,
            "observation": {
                "aggregate": aggregate(results),
                "cells": results,
            },
        }
        if args.against == "tgrad" and "upstream_baseline" in identity:
            document["observation"]["oracle_cases"] = oracle_case_summary(
                results, identity["upstream_baseline"]
            )
        result_summary = document["observation"]["aggregate"]
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
    if args.canonical_api_surface and args.against == "tgrad":
        invalid = {
            "verifier_error", "timeout",
        }
        observed = set(result_summary["files_by_outcome"])
        if observed & invalid:
            print(
                "canonical evidence is invalid: " +
                ", ".join(sorted(observed & invalid)),
                file=sys.stderr,
            )
            return 3
        nonparity = observed - {"passed", "unobserved_upstream"}
        if nonparity:
            print(
                "canonical evidence is valid but parity is not established: " +
                ", ".join(sorted(nonparity)),
                file=sys.stderr,
            )
            return 4
        baseline_scope = identity.get("upstream_baseline", {})
        if (
            baseline_scope.get("oracle_eligible_case_count", 0) == 0 or
            baseline_scope.get("upstream_unobserved_case_count", 0) > 0
        ):
            print(
                "canonical comparison is incomplete: upstream left "
                f"{baseline_scope.get('upstream_unobserved_case_count', 0)} "
                "cases unobserved",
                file=sys.stderr,
            )
            return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

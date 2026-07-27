#!/usr/bin/env python3
"""Stage, finalize, and promote one attributable 37-gate evidence snapshot."""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import (  # noqa: E402
    HASH_CONTRACT,
    INVENTORY,
    REVIEWED_HASH_CONTRACT_SHA256,
    REVIEWED_INVENTORY_SHA256,
    evidence_document_problems,
    gate_definition as contract_gate_definition,
    gate_names as contract_gate_names,
    load_contract,
    load_hash_contract,
)
sys.path.insert(0, str(ROOT / "scripts" / "perf"))
from release_certificate import validate_release_certificate  # noqa: E402
SCHEMA_VERSION = 1
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STATUSES = {"pass", "red", "blocked"}
OUTCOME_KEYS = {
    "schema_version", "kind", "run_id", "gate", "gate_definition_id",
    "source", "status", "returncode", "problems", "blocked_by", "log",
    "observed_evidence", "produced_artifacts", "content_sha256",
}


class CandidateError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = canonical_sha256(
        {key: child for key, child in result.items() if key != "content_sha256"}
    )
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


@contextmanager
def promotion_lock(destination: Path):
    """Serialize publication attempts for the one canonical snapshot."""
    identity = hashlib.sha256(str(destination.resolve()).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"tgrad-evidence-promotion-{identity}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def command(*arguments: str, cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(
            arguments, cwd=cwd, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CandidateError(f"command failed: {arguments!r}: {error}") from error
    return result.stdout.strip()


def source_identity() -> dict[str, Any]:
    commit = command("git", "rev-parse", "HEAD")
    tree = command("git", "rev-parse", "HEAD^{tree}")
    status = command("git", "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise CandidateError("evidence source tree must be clean, including untracked files")
    repository = command("git", "config", "--get", "remote.origin.url")
    return {
        "repository": repository,
        "commit": commit,
        "tree": tree,
        "dirty": False,
        "clean_observation_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def executable_identity(requested: str) -> dict[str, Any]:
    resolved = shutil.which(requested)
    if resolved is None:
        return {"requested": requested, "resolved": None, "sha256": None, "version": None}
    path = Path(resolved).resolve()
    version = subprocess.run(
        [str(path), "--version"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "requested": requested,
        "resolved": str(path),
        "sha256": file_sha256(path) if path.is_file() else None,
        "version": (version.stdout + version.stderr).strip(),
        "version_returncode": version.returncode,
    }


def environment_identity() -> dict[str, Any]:
    selected = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("TGRAD_") or key in {
            "BEAM", "DEBUG", "DEV", "GRAPH_ONE_KERNEL", "JIT", "PROFILE"
        }
    }
    python_requested = os.environ.get("TGRAD_PY", sys.executable)
    return attach_hash({
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_runtime": {
            "executable": str(Path(sys.executable).resolve()),
            "sha256": file_sha256(Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "gate_python": executable_identity(python_requested),
        "lake": executable_identity("lake"),
        "clang": executable_identity("clang"),
        "selected_environment": selected,
    })


def validate_release_environment() -> None:
    """Keep reviewed source locators equal to the files the gates consume."""
    overrides = [
        name for name in (
            "TGRAD_PERF_BASELINE",
            "TGRAD_PERF_BASELINE_FULL",
            "TGRAD_PERF_BASELINE_TC",
        )
        if os.environ.get(name)
    ]
    if overrides:
        raise CandidateError(
            "release evidence forbids runtime baseline overrides: " + ", ".join(overrides)
        )
    profile = os.environ.get(
        "TGRAD_PERF_PROFILE", os.environ.get("TGRAD_HOST", "apple_m4_mini_release")
    )
    if profile != "apple_m4_mini_release":
        raise CandidateError(
            "reviewed release hash locators require profile apple_m4_mini_release; "
            f"got {profile!r}"
        )


def build_gate_definitions(gates: list[str]) -> list[dict[str, Any]]:
    contract = load_contract()
    if gates != contract_gate_names(contract):
        raise CandidateError("requested gates disagree with reviewed release inventory")
    definitions: list[dict[str, Any]] = []
    for gate in gates:
        contract_gate = contract_gate_definition(contract, gate)
        relative = contract_gate["writer"]
        writer = ROOT / relative
        if not writer.is_file():
            raise CandidateError(f"gate writer is missing: {relative}")
        fields = {
            "gate": gate,
            "writer_path": relative,
            "writer_sha256": file_sha256(writer),
            "depends_on": contract_gate["depends_on"],
        }
        definitions.append({"id": canonical_sha256(fields), **fields})
    return definitions


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CandidateError(f"{label}: root must be an object")
    return value


def validate_hashed(value: dict[str, Any], label: str) -> None:
    recorded = value.get("content_sha256")
    actual = canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    )
    if recorded != actual:
        raise CandidateError(f"{label}: content hash mismatch")


def candidate_root(value: str) -> Path:
    if re.fullmatch(r"/[A-Za-z0-9._/-]+", value) is None or ".." in Path(value).parts:
        raise CandidateError("candidate path contains unsafe characters or parent traversal")
    path = Path(value).expanduser().resolve()
    if not path.is_absolute():
        raise CandidateError("candidate path must be absolute")
    try:
        path.relative_to(ROOT)
    except ValueError:
        return path
    raise CandidateError("candidate must live outside the repository")


def retain_performance_input(
    root: Path, certificate_path: Path | None, source: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any] | None:
    if certificate_path is None:
        return None
    certificate_path = certificate_path.expanduser()
    if not certificate_path.is_absolute() or not certificate_path.is_file() or \
       certificate_path.is_symlink():
        raise CandidateError("performance certificate must be an absolute regular file")
    resolved_certificate = certificate_path.resolve()
    if resolved_certificate != certificate_path:
        raise CandidateError("performance certificate path must not traverse symlinks")
    certificate_path = resolved_certificate
    try:
        certificate_path.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise CandidateError("performance certificate must be a run artifact outside the repository")
    document, artifacts, attestation, problems = validate_release_certificate(
        certificate_path, source, specification["id"],
        ROOT / specification["variance_model_path"],
        ROOT / specification["decision_rule_path"],
    )
    if document is None or attestation is None or problems:
        raise CandidateError(
            "performance certificate is not derivable:\n" + "\n".join(problems)
        )
    retained_artifacts: list[dict[str, Any]] = []
    performance_root = root / "performance"
    seen_artifacts: set[Path] = set()
    certificate_relative = Path(specification["snapshot_path"]).relative_to("performance")
    for artifact in artifacts:
        relative = Path(artifact["path"])
        if relative in seen_artifacts or relative == certificate_relative:
            raise CandidateError("performance evaluation artifact path is duplicated or reserved")
        seen_artifacts.add(relative)
        source_path = (certificate_path.parent / relative).resolve()
        destination = performance_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        retained_artifacts.append({
            "path": f"performance/{relative}",
            "sha256": artifact["sha256"],
            "bytes": destination.stat().st_size,
            "kind": artifact["kind"],
        })
    snapshot_path = Path(specification["snapshot_path"])
    destination = root / snapshot_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(certificate_path, destination)
    return {
        "certificate": {
            "path": str(snapshot_path),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
        },
        "evaluation_artifacts": retained_artifacts,
        "derived_attestation": attestation,
        "declared_source": document.get("source"),
        "source_matches_candidate": document.get("source") == source,
    }


def init_candidate(
    root: Path, gates: list[str], run_root: Path, green_gates: list[str] | None = None,
    performance_certificate: Path | None = None,
) -> None:
    if root.exists():
        raise CandidateError(f"refusing to reuse candidate path {root}")
    if not gates or len(gates) != len(set(gates)):
        raise CandidateError("gate inventory must be non-empty and unique")
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir() or run_root.is_symlink() or \
       run_root in {Path("/"), Path("/tmp"), Path("/private/tmp")}:
        raise CandidateError("run root must be a dedicated regular directory")
    try:
        run_root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise CandidateError("run root must live outside the repository")
    if not (run_root / ".tgrad-run-owner").is_file() and \
       not (run_root / ".tgrad-run-shared").is_dir():
        raise CandidateError("run root has no Tgrad ownership marker")
    validate_release_environment()
    identity = source_identity()
    contract = load_contract()
    hash_contract = load_hash_contract(contract)
    if gates != contract_gate_names(contract):
        raise CandidateError("gate runner arrays disagree with reviewed release inventory")
    if green_gates is None:
        green_gates = gates
    if green_gates != gates:
        raise CandidateError("GREEN_GATES must exactly equal the reviewed release inventory")
    root.mkdir(parents=True)
    (root / "evidence").mkdir()
    (root / "logs").mkdir()
    (root / "outcomes").mkdir()
    (root / "invocations").mkdir()
    (root / "artifact_records").mkdir()
    performance_input = retain_performance_input(
        root, performance_certificate, identity, contract["performance_prerequisite"]
    )
    document = attach_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "tgrad-gate-evidence-candidate",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": identity,
            "environment": environment_identity(),
            "gate_ids": gates,
            "green_gate_ids": green_gates,
            "gate_ids_sha256": canonical_sha256(gates),
            "gate_definitions": build_gate_definitions(gates),
            "release_contract": {
                "id": contract["contract_id"],
                "path": str(INVENTORY.relative_to(ROOT)),
                "sha256": REVIEWED_INVENTORY_SHA256,
                "performance_prerequisite": contract["performance_prerequisite"],
                "hash_contract": {
                    "id": hash_contract["contract_id"],
                    "path": str(HASH_CONTRACT.relative_to(ROOT)),
                    "sha256": REVIEWED_HASH_CONTRACT_SHA256,
                },
            },
            "performance_input": performance_input,
            "run_root": str(run_root),
            "state": "collecting",
        }
    )
    atomic_json(root / "candidate.initial.json", document)
    atomic_json(root / "candidate.json", document)


def load_candidate(root: Path) -> dict[str, Any]:
    document = load_json(root / "candidate.json", "candidate")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CandidateError("unsupported candidate schema")
    if document.get("kind") != "tgrad-gate-evidence-candidate":
        raise CandidateError("unexpected candidate kind")
    validate_hashed(document, "candidate")
    if source_identity() != document.get("source"):
        raise CandidateError("source commit/tree changed during evidence collection")
    gates = document.get("gate_ids")
    if not isinstance(gates, list) or not gates or len(gates) != len(set(gates)):
        raise CandidateError("candidate gate inventory is malformed")
    if document.get("gate_ids_sha256") != canonical_sha256(gates):
        raise CandidateError("candidate gate inventory hash mismatch")
    if document.get("green_gate_ids") != gates:
        raise CandidateError("candidate green-gate inventory mismatch")
    if document.get("gate_definitions") != build_gate_definitions(gates):
        raise CandidateError("candidate writer definitions changed")
    contract = load_contract()
    hash_contract = load_hash_contract(contract)
    expected_contract = {
        "id": contract["contract_id"],
        "path": str(INVENTORY.relative_to(ROOT)),
        "sha256": REVIEWED_INVENTORY_SHA256,
        "performance_prerequisite": contract["performance_prerequisite"],
        "hash_contract": {
            "id": hash_contract["contract_id"],
            "path": str(HASH_CONTRACT.relative_to(ROOT)),
            "sha256": REVIEWED_HASH_CONTRACT_SHA256,
        },
    }
    if document.get("release_contract") != expected_contract:
        raise CandidateError("candidate release contract changed")
    environment = document.get("environment")
    if not isinstance(environment, dict):
        raise CandidateError("candidate environment is missing")
    validate_hashed(environment, "candidate environment")
    return document


def artifact_patterns_for(producer: str) -> list[str]:
    policies = load_hash_contract(load_contract())["gates"]
    gates = policies if producer == "PREFLIGHT" else {producer: policies.get(producer, {})}
    return sorted({
        locator[len("artifact:"):]
        for claims in gates.values()
        for locator in claims.values()
        if locator.startswith("artifact:")
    })


def observable_artifacts(
    run_root: Path, patterns: list[str]
) -> dict[str, dict[str, Any]]:
    roots = [
        ("run", run_root),
        ("lake-build", ROOT / ".lake" / "build"),
        ("c-build", ROOT / "c" / "build"),
    ]
    result: dict[str, dict[str, Any]] = {}
    for namespace, artifact_root in roots:
        if not artifact_root.is_dir():
            continue
        for path in sorted(artifact_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            logical = f"{namespace}/{path.relative_to(artifact_root)}"
            if not any(fnmatch.fnmatchcase(logical, pattern) for pattern in patterns):
                continue
            result[logical] = {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return result


def observation_document(
    candidate: dict[str, Any], producer: str, phase: str,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return attach_hash({
        "schema_version": SCHEMA_VERSION,
        "kind": "tgrad-artifact-observation",
        "run_id": candidate["content_sha256"],
        "producer": producer,
        "phase": phase,
        "artifacts": {
            logical: {
                key: value[key] for key in ("sha256", "bytes", "mtime_ns")
            }
            for logical, value in artifacts.items()
        },
    })


def load_observation(
    path: Path, candidate: dict[str, Any], producer: str, phase: str
) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise CandidateError(f"{producer} {phase} observation is not a regular file")
    document = load_json(path, f"{producer} {phase} observation")
    validate_hashed(document, f"{producer} {phase} observation")
    if set(document) != {
        "schema_version", "kind", "run_id", "producer", "phase", "artifacts",
        "content_sha256",
    } or document.get("schema_version") != SCHEMA_VERSION or \
       document.get("kind") != "tgrad-artifact-observation" or \
       document.get("run_id") != candidate["content_sha256"] or \
       document.get("producer") != producer or document.get("phase") != phase or \
       not isinstance(document.get("artifacts"), dict):
        raise CandidateError(f"{producer} {phase} observation identity/schema mismatch")
    for logical, artifact in document["artifacts"].items():
        if not isinstance(logical, str) or logical.startswith("/") or \
           ".." in Path(logical).parts or not isinstance(artifact, dict) or \
           set(artifact) != {"sha256", "bytes", "mtime_ns"} or \
           not isinstance(artifact.get("sha256"), str) or \
           SHA256.fullmatch(artifact["sha256"]) is None or \
           not isinstance(artifact.get("bytes"), int) or artifact["bytes"] < 0 or \
           not isinstance(artifact.get("mtime_ns"), int) or artifact["mtime_ns"] < 0:
            raise CandidateError(f"{producer} {phase} observation artifact is malformed")
    return document["artifacts"]


def begin_invocation(root: Path, producer: str) -> None:
    candidate = load_candidate(root)
    if producer != "PREFLIGHT" and producer not in candidate["gate_ids"]:
        raise CandidateError(f"unknown artifact producer {producer}")
    path = root / "invocations" / f"{producer}.before.json"
    if path.exists():
        raise CandidateError(f"invocation already began for {producer}")
    patterns = artifact_patterns_for(producer)
    atomic_json(path, observation_document(
        candidate, producer, "before",
        observable_artifacts(Path(candidate["run_root"]), patterns),
    ))


def capture_produced_artifacts(root: Path, producer: str) -> dict[str, Any]:
    candidate = load_candidate(root)
    before_path = root / "invocations" / f"{producer}.before.json"
    after_path = root / "invocations" / f"{producer}.after.json"
    record_path = root / "artifact_records" / f"{producer}.json"
    if after_path.exists() or record_path.exists():
        raise CandidateError(f"artifact capture already completed for {producer}")
    before = load_observation(before_path, candidate, producer, "before")
    current = observable_artifacts(
        Path(candidate["run_root"]), artifact_patterns_for(producer)
    )
    atomic_json(after_path, observation_document(candidate, producer, "after", current))
    store = root / "artifacts" / "sha256"
    records: list[dict[str, Any]] = []
    for logical, value in current.items():
        old = before.get(logical)
        if old is not None and all(
            old.get(key) == value[key] for key in ("sha256", "bytes", "mtime_ns")
        ):
            continue
        digest = value["sha256"]
        destination = store / digest[:2] / digest
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}")
            shutil.copyfile(Path(value["path"]), temporary)
            if file_sha256(temporary) != digest:
                temporary.unlink(missing_ok=True)
                raise CandidateError(f"artifact changed while preserving {logical}")
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        records.append(
            {
                "producer": producer,
                "logical_path": logical,
                "sha256": digest,
                "bytes": value["bytes"],
                "retained_path": str(destination.relative_to(root)),
            }
        )
    record = attach_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "tgrad-produced-artifacts",
            "run_id": candidate["content_sha256"],
            "producer": producer,
            "before_observation_sha256": file_sha256(before_path),
            "after_observation_sha256": file_sha256(after_path),
            "artifacts": records,
        }
    )
    atomic_json(record_path, record)
    return {"count": len(records), "record_sha256": file_sha256(record_path)}


def record_outcome(
    root: Path, gate: str, status: str, returncode: int, log_path: Path,
    blocked_by: list[str] | None = None,
) -> None:
    candidate = load_candidate(root)
    if candidate.get("state") != "collecting":
        raise CandidateError("candidate is no longer collecting")
    if gate not in candidate["gate_ids"]:
        raise CandidateError(f"gate {gate} is outside candidate inventory")
    gate_definition = next(
        (value for value in candidate["gate_definitions"] if value.get("gate") == gate),
        None,
    )
    if gate_definition is None:
        raise CandidateError(f"gate {gate} has no immutable writer definition")
    destination = root / "outcomes" / f"{gate}.json"
    if destination.exists():
        raise CandidateError(f"outcome already recorded for {gate}")
    if status not in STATUSES:
        raise CandidateError(f"invalid gate status {status}")
    blocked_by = [
        line.strip()
        for value in (blocked_by or [])
        for line in value.splitlines()
        if line.strip()
    ]
    if status == "blocked" and not blocked_by:
        raise CandidateError("blocked outcome must name at least one blocker")
    if status != "blocked" and blocked_by:
        raise CandidateError("only blocked outcomes may name blockers")
    if not log_path.is_file():
        raise CandidateError(f"gate log does not exist: {log_path}")
    try:
        log_relative = log_path.resolve().relative_to(root)
    except ValueError as error:
        raise CandidateError("gate log must be inside the candidate") from error
    evidence_path = root / "evidence" / f"{gate}.json"
    evidence: dict[str, Any] | None = None
    problems: list[str] = []
    if evidence_path.is_file():
        try:
            evidence = load_json(evidence_path, f"{gate} evidence")
        except CandidateError as error:
            problems.append(str(error))
        if evidence is not None:
            hash_contract = load_hash_contract(load_contract())
            problems.extend(
                evidence_document_problems(
                    hash_contract, gate, evidence, candidate["source"]["commit"]
                )
            )
    elif status == "pass":
        problems.append("passing gate produced no evidence file")
    if returncode == 0 and status != "pass":
        problems.append("zero return code recorded with non-pass status")
    if returncode != 0 and status == "pass":
        problems.append("nonzero return code recorded as pass")
    if problems:
        status = "red"
    produced = (
        {"count": 0, "record_sha256": None}
        if status == "blocked"
        else capture_produced_artifacts(root, gate)
    )
    outcome = attach_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "tgrad-gate-outcome",
            "run_id": candidate["content_sha256"],
            "gate": gate,
            "gate_definition_id": gate_definition["id"],
            "source": candidate["source"],
            "status": status,
            "returncode": returncode,
            "problems": problems,
            "blocked_by": blocked_by,
            "log": {
                "path": str(log_relative),
                "sha256": file_sha256(log_path),
                "bytes": log_path.stat().st_size,
            },
            # Umbrella gates can legitimately rerun and replace a child's
            # timestamped JSON later.  This binds what this process observed;
            # the manifest independently binds the coherent final set.
            "observed_evidence": (
                {
                    "path": f"evidence/{gate}.json",
                    "sha256": file_sha256(evidence_path),
                    "bytes": evidence_path.stat().st_size,
                }
                if evidence_path.is_file()
                else None
            ),
            "produced_artifacts": produced,
        }
    )
    atomic_json(destination, outcome)


def blocking_dependencies(root: Path, gate: str) -> list[str]:
    candidate = load_candidate(root)
    definition = next(
        (value for value in candidate["gate_definitions"] if value["gate"] == gate),
        None,
    )
    if definition is None:
        raise CandidateError(f"unknown candidate gate {gate}")
    blockers: list[str] = []
    for dependency in definition["depends_on"]:
        path = root / "outcomes" / f"{dependency}.json"
        if not path.is_file():
            blockers.append(f"{dependency}:missing")
            continue
        outcome = load_json(path, f"{dependency} outcome")
        if outcome.get("status") != "pass":
            blockers.append(f"{dependency}:{outcome.get('status', 'malformed')}")
    return blockers


def performance_prerequisite(
    root: Path, candidate: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    specification = candidate["release_contract"]["performance_prerequisite"]
    retained = candidate.get("performance_input")
    if not isinstance(retained, dict) or set(retained) != {
        "certificate", "evaluation_artifacts", "derived_attestation",
        "declared_source", "source_matches_candidate"
    }:
        return None, ["performance prerequisite was not supplied as a retained run artifact"]
    certificate_entry = retained.get("certificate")
    if not isinstance(certificate_entry, dict) or set(certificate_entry) != {
        "path", "sha256", "bytes"
    } or certificate_entry.get("path") != specification["snapshot_path"]:
        return None, ["retained performance certificate identity is malformed"]
    path = root / specification["snapshot_path"]
    if not path.is_file() or path.is_symlink() or \
       file_sha256(path) != certificate_entry.get("sha256") or \
       path.stat().st_size != certificate_entry.get("bytes"):
        return None, ["retained performance certificate bytes changed"]
    document, artifacts, attestation, problems = validate_release_certificate(
        path, candidate["source"], specification["id"],
        ROOT / specification["variance_model_path"],
        ROOT / specification["decision_rule_path"],
    )
    expected_artifacts = [
        {
            "path": f"performance/{artifact['path']}",
            "sha256": artifact["sha256"],
            "bytes": artifact["bytes"],
            "kind": artifact["kind"],
        }
        for artifact in artifacts
    ]
    if retained.get("evaluation_artifacts") != expected_artifacts:
        problems.append("retained performance evaluation-artifact cover is not exact")
    if retained.get("derived_attestation") != attestation:
        problems.append("retained performance attestation was not derived from current bytes")
    if retained.get("declared_source") != candidate["source"] or \
       retained.get("source_matches_candidate") is not True:
        problems.append("retained performance source differs from candidate source")
    return document, problems


def iter_artifact_files(
    root: Path, run_root: Path, run_id: str, allowed_producers: set[str]
) -> Iterable[tuple[str, Path]]:
    tracked = command("git", "ls-files").splitlines()
    for relative in tracked:
        path = ROOT / relative
        if path.is_file():
            yield f"repo:{relative}", path
    for evidence in (root / "evidence").glob("*.json"):
        if evidence.is_file() and not evidence.is_symlink():
            yield f"evidence:{evidence.stem}", evidence
    for record_path in (root / "artifact_records").glob("*.json"):
        record = load_json(record_path, "artifact producer record")
        validate_hashed(record, "artifact producer record")
        if set(record) != {
            "schema_version", "kind", "run_id", "producer",
            "before_observation_sha256", "after_observation_sha256",
            "artifacts", "content_sha256",
        }:
            raise CandidateError(f"artifact record has unexpected keys: {record_path.name}")
        producer = record.get("producer")
        if record.get("schema_version") != SCHEMA_VERSION or \
           record.get("kind") != "tgrad-produced-artifacts" or \
           record.get("run_id") != run_id or producer != record_path.stem or \
           producer not in allowed_producers:
            raise CandidateError(f"artifact record identity mismatch: {record_path.name}")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list):
            raise CandidateError(f"artifact record list is malformed: {record_path.name}")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {
                "producer", "logical_path", "sha256", "bytes", "retained_path"
            }:
                raise CandidateError(f"artifact entry is malformed: {record_path.name}")
            if artifact.get("producer") != producer:
                raise CandidateError(f"artifact producer mismatch: {record_path.name}")
            if not isinstance(artifact.get("logical_path"), str) or \
               artifact["logical_path"].startswith("/") or \
               ".." in Path(artifact["logical_path"]).parts:
                raise CandidateError(f"artifact logical path is unsafe: {record_path.name}")
            if not isinstance(artifact.get("sha256"), str) or \
               SHA256.fullmatch(artifact["sha256"]) is None:
                raise CandidateError(f"artifact digest is malformed: {record_path.name}")
            retained = root / artifact["retained_path"]
            expected_retained = Path("artifacts/sha256") / artifact["sha256"][:2] / artifact["sha256"]
            if Path(artifact["retained_path"]) != expected_retained or \
               not retained.is_file() or retained.is_symlink() or \
               file_sha256(retained) != artifact["sha256"] or \
               retained.stat().st_size != artifact.get("bytes"):
                raise CandidateError(f"artifact retention mismatch: {record_path.name}")
            yield f"artifact:{artifact['producer']}:{artifact['logical_path']}", retained


def artifact_index(
    root: Path, run_root: Path, run_id: str, allowed_producers: set[str]
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for identity, path in iter_artifact_files(
        root, run_root, run_id, allowed_producers
    ):
        resolved = path.resolve()
        digest = file_sha256(path)
        index.setdefault(digest, []).append(
            {"identity": identity, "path": str(resolved), "bytes": path.stat().st_size}
        )
    return index


def evidence_hash_referents(
    evidence_documents: dict[str, dict[str, Any]],
    index: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    referents: list[dict[str, Any]] = []
    unresolved: list[str] = []
    contract = load_contract()
    policies = load_hash_contract(contract)["gates"]
    for gate, document in evidence_documents.items():
        hashes = document.get("hashes")
        if not isinstance(hashes, dict):
            unresolved.append(f"{gate}:hashes is not an object")
            continue
        expected = policies.get(gate)
        if not isinstance(expected, dict) or set(hashes) != set(expected):
            unresolved.append(
                f"{gate}:hash keys mismatch missing={sorted(set(expected or {})-set(hashes))} "
                f"extra={sorted(set(hashes)-set(expected or {}))}"
            )
            continue
        for key, digest in hashes.items():
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                unresolved.append(f"{gate}:{key} is not a SHA-256")
                continue
            matches = index.get(digest, [])
            policy = expected[key]
            if policy.startswith("evidence:"):
                child = policy[len("evidence:"):].casefold()
                matches = [
                    value for value in matches
                    if value["identity"].startswith("evidence:") and
                    value["identity"][len("evidence:"):].casefold() == child
                ]
            elif policy.startswith("source:"):
                expected_identity = f"repo:{policy[len('source:') :]}"
                matches = [value for value in matches if value["identity"] == expected_identity]
            elif policy.startswith("artifact:"):
                allowed_artifacts = (f"artifact:{gate}:", "artifact:PREFLIGHT:")
                logical_pattern = policy[len("artifact:"):]
                matches = [
                    value for value in matches
                    if value["identity"].startswith(allowed_artifacts) and
                    fnmatch.fnmatchcase(value["identity"].split(":", 2)[2], logical_pattern)
                ]
            else:
                matches = []
            if not matches:
                unresolved.append(f"{gate}:{key}")
                continue
            referents.append(
                {"gate": gate, "key": key, "sha256": digest, "referents": matches}
            )
    return referents, unresolved


def retain_non_source_referents(
    root: Path, referents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Make every non-Git hash durable inside the candidate release bundle."""
    retained_root = root / "release_artifacts" / "sha256"
    result: list[dict[str, Any]] = []
    for item in referents:
        copied = {
            "gate": item["gate"],
            "key": item["key"],
            "sha256": item["sha256"],
            "referents": [
                {"identity": value["identity"], "bytes": value["bytes"]}
                for value in item["referents"]
            ],
        }
        source_referents = [
            value for value in item["referents"] if value["identity"].startswith("repo:")
        ]
        if source_referents:
            copied["durable_referent"] = source_referents[0]["identity"]
            result.append(copied)
            continue
        evidence_referents = [
            value for value in item["referents"] if value["identity"].startswith("evidence:")
        ]
        if evidence_referents:
            copied["durable_referent"] = evidence_referents[0]["identity"]
            result.append(copied)
            continue
        digest = item["sha256"]
        source_path = next(
            (Path(value["path"]) for value in item["referents"] if Path(value["path"]).is_file()),
            None,
        )
        if source_path is None:
            raise CandidateError(f"referent vanished before retention: {item['gate']}:{item['key']}")
        destination = retained_root / digest[:2] / digest
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}")
            shutil.copyfile(source_path, temporary)
            if file_sha256(temporary) != digest:
                temporary.unlink(missing_ok=True)
                raise CandidateError(f"retained artifact hash mismatch for {source_path}")
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        copied["durable_referent"] = f"artifact:sha256/{digest[:2]}/{digest}"
        result.append(copied)
    return result


def rollup_problems(
    root: Path, evidence_documents: dict[str, dict[str, Any]]
) -> list[str]:
    problems: list[str] = []
    by_folded_name: dict[str, str] = {}
    for gate in evidence_documents:
        folded = gate.casefold()
        if folded in by_folded_name:
            problems.append(f"ambiguous evidence gate names: {gate}, {by_folded_name[folded]}")
        by_folded_name[folded] = gate
    for gate, document in evidence_documents.items():
        for key, digest in document.get("hashes", {}).items():
            if not key.lower().endswith("_evidence_sha256"):
                continue
            child = key[: -len("_evidence_sha256")]
            child_gate = by_folded_name.get(child.casefold())
            if child_gate is None:
                problems.append(f"{gate}:{key} missing child {child}.json")
                continue
            child_path = root / "evidence" / f"{child_gate}.json"
            if file_sha256(child_path) != digest:
                problems.append(f"{gate}:{key} disagrees with {child_gate}.json")
    return problems


def finalize_candidate(root: Path) -> dict[str, Any]:
    candidate = load_candidate(root)
    if (root / "manifest.json").exists():
        raise CandidateError("candidate is already finalized")
    gates = candidate["gate_ids"]
    outcome_paths = {path.stem: path for path in (root / "outcomes").glob("*.json")}
    missing_outcomes = sorted(set(gates) - set(outcome_paths))
    extra_outcomes = sorted(set(outcome_paths) - set(gates))
    if missing_outcomes or extra_outcomes:
        raise CandidateError(
            f"outcome cover is not total: missing={missing_outcomes}, extra={extra_outcomes}"
        )
    outcomes = {gate: load_json(outcome_paths[gate], f"{gate} outcome") for gate in gates}
    for gate, outcome in outcomes.items():
        if outcome_paths[gate].is_symlink():
            raise CandidateError(f"{gate} outcome must not be a symlink")
        validate_hashed(outcome, f"{gate} outcome")
        if set(outcome) != OUTCOME_KEYS:
            raise CandidateError(f"{gate} outcome schema keys are not exact")
        definition = next(
            value for value in candidate["gate_definitions"] if value["gate"] == gate
        )
        if (outcome.get("gate") != gate or
            outcome.get("gate_definition_id") != definition["id"] or
            outcome.get("source") != candidate["source"] or
            outcome.get("run_id") != candidate["content_sha256"]):
            raise CandidateError(f"{gate} outcome identity mismatch")
        status = outcome.get("status")
        blocked_by = outcome.get("blocked_by")
        if status not in STATUSES or not isinstance(outcome.get("problems"), list) or \
           not isinstance(blocked_by, list) or \
           any(not isinstance(value, str) or not value for value in blocked_by):
            raise CandidateError(f"{gate} outcome state is malformed")
        if status == "pass" and (
            outcome.get("returncode") != 0 or outcome["problems"] or blocked_by
        ):
            raise CandidateError(f"{gate} passing outcome is internally inconsistent")
        if status == "blocked" and (
            outcome.get("returncode") == 0 or not blocked_by or
            outcome.get("observed_evidence") is not None
        ):
            raise CandidateError(f"{gate} blocked outcome is internally inconsistent")
        if status == "red" and outcome.get("returncode") == 0 and not outcome["problems"]:
            raise CandidateError(f"{gate} red outcome has neither process failure nor problems")
    pass_gates = [gate for gate in gates if outcomes[gate]["status"] == "pass"]
    red_gates = [gate for gate in gates if outcomes[gate]["status"] == "red"]
    blocked_gates = [gate for gate in gates if outcomes[gate]["status"] == "blocked"]
    evidence_paths = {path.stem: path for path in (root / "evidence").glob("*.json")}
    evidence_documents = {
        gate: load_json(path, f"{gate} evidence") for gate, path in evidence_paths.items()
    }
    evidence_problems: list[str] = []
    if any(path.is_symlink() for path in evidence_paths.values()):
        evidence_problems.append("evidence documents must not be symlinks")
    if set(evidence_paths) != set(pass_gates):
        evidence_problems.append(
            "evidence files must exist exactly for passing gates: "
            f"missing={sorted(set(pass_gates) - set(evidence_paths))}, "
            f"unexpected={sorted(set(evidence_paths) - set(pass_gates))}"
        )
    for gate, document in evidence_documents.items():
        for problem in evidence_document_problems(
            load_hash_contract(load_contract()), gate, document,
            candidate["source"]["commit"],
        ):
            evidence_problems.append(f"{gate}: {problem}")
    for gate in pass_gates:
        observed = outcomes[gate].get("observed_evidence")
        final_path = evidence_paths.get(gate)
        if not isinstance(observed, dict) or final_path is None:
            evidence_problems.append(f"{gate}: passing outcome has no observed evidence")
        elif (observed.get("sha256") != file_sha256(final_path) or
              observed.get("bytes") != final_path.stat().st_size):
            evidence_problems.append(f"{gate}: evidence changed after the passing process")
    allowed_producers = {"PREFLIGHT", *gates}
    index = artifact_index(
        root, Path(candidate["run_root"]), candidate["content_sha256"],
        allowed_producers,
    )
    referents, unresolved = evidence_hash_referents(evidence_documents, index)
    evidence_problems.extend(f"unresolved hash {value}" for value in unresolved)
    evidence_problems.extend(rollup_problems(root, evidence_documents))
    artifact_record_paths = {
        path.stem: path for path in (root / "artifact_records").glob("*.json")
    }
    expected_record_producers = {"PREFLIGHT", *pass_gates, *red_gates}
    if set(artifact_record_paths) != expected_record_producers:
        evidence_problems.append(
            "artifact producer records do not exactly cover executed producers: "
            f"missing={sorted(expected_record_producers-set(artifact_record_paths))}, "
            f"extra={sorted(set(artifact_record_paths)-expected_record_producers)}"
        )
    for producer, path in artifact_record_paths.items():
        if path.is_symlink():
            evidence_problems.append(f"{producer}: artifact record must not be a symlink")
        try:
            record = load_json(path, f"{producer} artifact record")
            before_path = root / "invocations" / f"{producer}.before.json"
            after_path = root / "invocations" / f"{producer}.after.json"
            before = load_observation(before_path, candidate, producer, "before")
            after = load_observation(after_path, candidate, producer, "after")
            expected_changed = {
                logical: artifact for logical, artifact in after.items()
                if before.get(logical) != artifact
            }
            recorded = record.get("artifacts")
            if record.get("before_observation_sha256") != file_sha256(before_path) or \
               record.get("after_observation_sha256") != file_sha256(after_path):
                evidence_problems.append(f"{producer}: artifact record does not bind observations")
            if not isinstance(recorded, list) or {
                value.get("logical_path"): {
                    key: value.get(key) for key in ("sha256", "bytes")
                }
                for value in recorded if isinstance(value, dict)
            } != {
                logical: {key: artifact[key] for key in ("sha256", "bytes")}
                for logical, artifact in expected_changed.items()
            }:
                evidence_problems.append(f"{producer}: produced artifacts disagree with before/after observations")
        except CandidateError as error:
            evidence_problems.append(str(error))
    for gate in pass_gates + red_gates:
        produced = outcomes[gate].get("produced_artifacts")
        path = artifact_record_paths.get(gate)
        if not isinstance(produced, dict) or set(produced) != {"count", "record_sha256"}:
            evidence_problems.append(f"{gate}: produced-artifact outcome is malformed")
        elif path is None or produced.get("record_sha256") != file_sha256(path):
            evidence_problems.append(f"{gate}: outcome does not bind its artifact record")
    for gate in blocked_gates:
        if outcomes[gate].get("produced_artifacts") != {
            "count": 0, "record_sha256": None
        }:
            evidence_problems.append(f"{gate}: blocked outcome claims produced artifacts")
    prerequisite_document, prerequisite_problems = performance_prerequisite(root, candidate)
    evidence_problems.extend(prerequisite_problems)
    durable_referents = retain_non_source_referents(root, referents)
    state = (
        "complete_green"
        if not red_gates and not blocked_gates and not evidence_problems
        else "complete_red"
    )
    manifest = attach_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "tgrad-gate-evidence-candidate-manifest",
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": candidate["content_sha256"],
            "candidate_initial_file_sha256": file_sha256(root / "candidate.initial.json"),
            "source": candidate["source"],
            "gate_ids": gates,
            "gate_ids_sha256": candidate["gate_ids_sha256"],
            "state": state,
            "pass_gates": pass_gates,
            "red_gates": red_gates,
            "blocked_gates": blocked_gates,
            "outcomes": {
                gate: {
                    "sha256": file_sha256(outcome_paths[gate]),
                    "status": outcomes[gate]["status"],
                }
                for gate in gates
            },
            "artifact_records": {
                path.stem: {"sha256": file_sha256(path)}
                for path in sorted(artifact_record_paths.values())
            },
            "invocations": {
                producer: {
                    phase: {
                        "sha256": file_sha256(
                            root / "invocations" / f"{producer}.{phase}.json"
                        )
                    }
                    for phase in ("before", "after")
                }
                for producer in sorted(artifact_record_paths)
            },
            "evidence": {
                gate: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                for gate, path in sorted(evidence_paths.items())
            },
            "hash_referents": durable_referents,
            "performance_prerequisite": (
                {
                    "path": candidate["release_contract"]["performance_prerequisite"]["snapshot_path"],
                    "sha256": file_sha256(
                        root / candidate["release_contract"]["performance_prerequisite"]["snapshot_path"]
                    ),
                    "content_sha256": prerequisite_document["content_sha256"],
                    "evaluation_artifacts": candidate["performance_input"]["evaluation_artifacts"],
                    "derived_attestation": candidate["performance_input"]["derived_attestation"],
                    "variance_model": {
                        "path": candidate["release_contract"]["performance_prerequisite"]["variance_model_path"],
                        "sha256": prerequisite_document["variance_model_sha256"],
                    },
                    "decision_rule": {
                        "path": candidate["release_contract"]["performance_prerequisite"]["decision_rule_path"],
                        "sha256": prerequisite_document["decision_rule_sha256"],
                    },
                }
                if prerequisite_document is not None and not prerequisite_problems
                else None
            ),
            "problems": evidence_problems,
            "promotion_ready": state == "complete_green",
        }
    )
    atomic_json(root / "manifest.json", manifest)
    finalized_candidate = dict(candidate)
    finalized_candidate["state"] = state
    finalized_candidate["manifest_content_sha256"] = manifest["content_sha256"]
    finalized_candidate.pop("content_sha256", None)
    atomic_json(root / "candidate.json", attach_hash(finalized_candidate))
    return manifest


def copy_regular(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or source.is_symlink():
        raise CandidateError(f"snapshot source is not a regular file: {source}")
    if file_sha256(source) != expected_sha256:
        raise CandidateError(f"snapshot source hash changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_sha256(destination) != expected_sha256:
        raise CandidateError(f"snapshot copy hash mismatch: {destination}")


def validate_promotable_candidate(
    root: Path, candidate: dict[str, Any], manifest: dict[str, Any]
) -> None:
    contract = load_contract()
    gates = contract_gate_names(contract)
    if candidate.get("state") != "complete_green" or \
       candidate.get("manifest_content_sha256") != manifest.get("content_sha256") or \
       candidate.get("source") != manifest.get("source"):
        raise CandidateError("candidate and manifest final states are not bound")
    if manifest.get("state") != "complete_green" or \
       manifest.get("promotion_ready") is not True or manifest.get("problems") != [] or \
       manifest.get("gate_ids") != gates or manifest.get("pass_gates") != gates or \
       manifest.get("red_gates") != [] or manifest.get("blocked_gates") != []:
        raise CandidateError("manifest is not an exact complete-green release run")
    if set(manifest.get("evidence", {})) != set(gates) or \
       set(manifest.get("outcomes", {})) != set(gates):
        raise CandidateError("manifest gate closure is not exact")
    if set(manifest.get("artifact_records", {})) != {"PREFLIGHT", *gates}:
        raise CandidateError("manifest producer-record closure is not exact")
    if set(manifest.get("invocations", {})) != {"PREFLIGHT", *gates}:
        raise CandidateError("manifest invocation-observation closure is not exact")
    performance = manifest.get("performance_prerequisite")
    specification = contract["performance_prerequisite"]
    if not isinstance(performance, dict) or \
       performance.get("path") != specification["snapshot_path"]:
        raise CandidateError("manifest performance prerequisite path is not reviewed")
    evaluation_artifacts = performance.get("evaluation_artifacts")
    if not isinstance(evaluation_artifacts, list) or not evaluation_artifacts or any(
        not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes", "kind"} or
        not isinstance(value.get("path"), str) or
        not value["path"].startswith("performance/") or
        Path(value["path"]).is_absolute() or ".." in Path(value["path"]).parts
        for value in (evaluation_artifacts if isinstance(evaluation_artifacts, list) else [])
    ):
        raise CandidateError("manifest performance evaluation-artifact paths are unsafe")
    attestation = performance.get("derived_attestation")
    if not isinstance(attestation, dict) or attestation.get("accepted") is not True:
        raise CandidateError("manifest performance decision was not derived as accepted")
    if file_sha256(root / "candidate.initial.json") != \
       manifest.get("candidate_initial_file_sha256"):
        raise CandidateError("immutable candidate bytes changed")


def assemble_snapshot(root: Path, snapshot: Path, manifest: dict[str, Any]) -> None:
    """Materialize only the manifest's exact closure, never ambient files."""
    if snapshot.exists():
        raise CandidateError(f"snapshot staging path already exists: {snapshot}")
    snapshot.mkdir(parents=True)
    gates = manifest["gate_ids"]
    for gate in gates:
        evidence = manifest["evidence"][gate]
        copy_regular(
            root / "evidence" / f"{gate}.json",
            snapshot / f"{gate}.json",
            evidence["sha256"],
        )
        outcome = manifest["outcomes"][gate]
        copy_regular(
            root / "outcomes" / f"{gate}.json",
            snapshot / "outcomes" / f"{gate}.json",
            outcome["sha256"],
        )
        outcome_document = load_json(
            root / "outcomes" / f"{gate}.json", f"{gate} outcome"
        )
        if outcome_document.get("log", {}).get("path") != f"logs/{gate}.log":
            raise CandidateError(f"{gate} outcome log path is not canonical")
        copy_regular(
            root / outcome_document["log"]["path"],
            snapshot / "logs" / f"{gate}.log",
            outcome_document["log"]["sha256"],
        )
    artifact_records = manifest.get("artifact_records")
    if not isinstance(artifact_records, dict):
        raise CandidateError("manifest artifact-record map is malformed")
    for producer, record in artifact_records.items():
        if not isinstance(record, dict) or set(record) != {"sha256"}:
            raise CandidateError(f"malformed artifact record manifest entry: {producer}")
        copy_regular(
            root / "artifact_records" / f"{producer}.json",
            snapshot / "artifact_records" / f"{producer}.json",
            record["sha256"],
        )
        invocation = manifest["invocations"].get(producer)
        if not isinstance(invocation, dict) or set(invocation) != {"before", "after"}:
            raise CandidateError(f"malformed invocation manifest entry: {producer}")
        for phase in ("before", "after"):
            entry = invocation[phase]
            if not isinstance(entry, dict) or set(entry) != {"sha256"}:
                raise CandidateError(f"malformed {phase} observation entry: {producer}")
            copy_regular(
                root / "invocations" / f"{producer}.{phase}.json",
                snapshot / "invocations" / f"{producer}.{phase}.json",
                entry["sha256"],
            )
    performance = manifest.get("performance_prerequisite")
    if not isinstance(performance, dict):
        raise CandidateError("promotable manifest has no retained performance prerequisite")
    copy_regular(
        root / performance["path"],
        snapshot / performance["path"],
        performance["sha256"],
    )
    for artifact in performance["evaluation_artifacts"]:
        copy_regular(
            root / artifact["path"],
            snapshot / artifact["path"],
            artifact["sha256"],
        )
    artifact_paths = {
        item["durable_referent"][len("artifact:"):]
        for item in manifest["hash_referents"]
        if item["durable_referent"].startswith("artifact:")
    }
    for relative in sorted(artifact_paths):
        if not relative.startswith("sha256/") or ".." in Path(relative).parts:
            raise CandidateError(f"unsafe retained artifact locator: {relative}")
        digest = Path(relative).name
        if SHA256.fullmatch(digest) is None:
            raise CandidateError(f"malformed retained artifact locator: {relative}")
        copy_regular(
            root / "release_artifacts" / relative,
            snapshot / "artifacts" / relative,
            digest,
        )
    atomic_json(snapshot / "RUN_MANIFEST.json", manifest)
    copy_regular(
        root / "candidate.initial.json",
        snapshot / "RUN_CANDIDATE.json",
        manifest["candidate_initial_file_sha256"],
    )


def strict_audit(snapshot: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dev" / "evidence_provenance_audit.py"),
            "--strict",
            "--quiet",
            "--evidence-dir",
            str(snapshot),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CandidateError(
            "fatal provenance audit rejected snapshot:\n" + result.stdout + result.stderr
        )


def promote_candidate(root: Path, destination: Path, backup: Path) -> None:
    candidate = load_json(root / "candidate.json", "candidate")
    manifest = load_json(root / "manifest.json", "candidate manifest")
    validate_hashed(candidate, "candidate")
    validate_hashed(manifest, "candidate manifest")
    validate_promotable_candidate(root, candidate, manifest)
    if source_identity() != manifest.get("source"):
        raise CandidateError("source tree changed after candidate finalization")
    expected_destination = ROOT / "fixtures" / "gate_evidence"
    if destination.is_symlink() or destination.absolute() != expected_destination.absolute():
        raise CandidateError("promotion destination must be fixtures/gate_evidence")
    destination = destination.resolve()
    if not destination.is_dir():
        raise CandidateError("canonical evidence directory must exist before promotion")
    try:
        backup.resolve().relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise CandidateError("promotion backup must live outside the repository")
    if not backup.parent.is_dir() or \
       backup.parent.stat().st_dev != destination.parent.stat().st_dev:
        raise CandidateError("promotion backup must be on the canonical directory's filesystem")
    if backup.exists():
        raise CandidateError(f"backup path already exists: {backup}")
    temporary = Path(tempfile.mkdtemp(prefix=".gate-evidence-promotion.", dir=destination.parent))
    promoted = False
    try:
        staged = temporary / "gate_evidence"
        assemble_snapshot(root, staged, manifest)
        # Reject before the canonical path is touched. The second audit binds
        # the bytes after the same-filesystem rename as well.
        strict_audit(staged)
        with promotion_lock(destination):
            os.replace(destination, backup)
            os.replace(staged, destination)
            promoted = True
            strict_audit(destination)
    except BaseException:
        if promoted and destination.exists() and backup.exists():
            failed = temporary / "rejected"
            os.replace(destination, failed)
            os.replace(backup, destination)
        elif not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--candidate", required=True)
    init.add_argument("--gates", nargs="+", required=True)
    init.add_argument("--green-gates", nargs="+", required=True)
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--performance-certificate", type=Path)
    record = commands.add_parser("record")
    record.add_argument("--candidate", required=True)
    record.add_argument("--gate", required=True)
    record.add_argument("--status", choices=sorted(STATUSES), required=True)
    record.add_argument("--returncode", type=int, required=True)
    record.add_argument("--log", type=Path, required=True)
    record.add_argument("--blocked-by", action="append", default=[])
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--candidate", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--destination", type=Path, default=ROOT / "fixtures" / "gate_evidence")
    promote.add_argument("--backup", type=Path, required=True)
    blockers = commands.add_parser("blockers")
    blockers.add_argument("--candidate", required=True)
    blockers.add_argument("--gate", required=True)
    begin = commands.add_parser("begin")
    begin.add_argument("--candidate", required=True)
    begin.add_argument("--producer", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--candidate", required=True)
    capture.add_argument("--producer", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        root = candidate_root(args.candidate)
        if args.command == "init":
            init_candidate(
                root, args.gates, args.run_root, args.green_gates,
                args.performance_certificate,
            )
        elif args.command == "record":
            record_outcome(
                root, args.gate, args.status, args.returncode, args.log,
                args.blocked_by,
            )
        elif args.command == "finalize":
            manifest = finalize_candidate(root)
            print(json.dumps(manifest, sort_keys=True, indent=2))
            return 0 if manifest["promotion_ready"] else 1
        elif args.command == "blockers":
            values = blocking_dependencies(root, args.gate)
            print("\n".join(values))
            return 0
        elif args.command == "begin":
            begin_invocation(root, args.producer)
            return 0
        elif args.command == "capture":
            print(json.dumps(capture_produced_artifacts(root, args.producer), sort_keys=True))
            return 0
        else:
            promote_candidate(root, args.destination, args.backup.resolve())
        return 0
    except (CandidateError, KeyError, TypeError, ValueError) as error:
        print(f"evidence_candidate: FAILED — {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

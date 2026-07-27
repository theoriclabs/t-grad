#!/usr/bin/env python3
"""Strictly verify that a promoted gate-evidence snapshot certifies this tree.

Legacy snapshots without RUN_MANIFEST.json are reported as non-attributable.
A promoted snapshot is accepted only when its source commit/tree, exact gate
cover, process outcomes, evidence bytes, roll-ups, and every cited SHA-256 can
all be re-derived.  There are no "transient hash" exemptions: non-source
artifacts must be retained in the snapshot's content-addressed closure.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "evidence"))
from contract import (  # noqa: E402
    HASH_CONTRACT,
    INVENTORY,
    REVIEWED_HASH_CONTRACT_SHA256,
    REVIEWED_INVENTORY_SHA256,
    evidence_document_problems,
    gate_names as contract_gate_names,
    load_contract,
    load_hash_contract,
)
sys.path.insert(0, str(REPO / "scripts" / "perf"))
from release_certificate import validate_release_certificate  # noqa: E402
DEFAULT_EVIDENCE = REPO / "fixtures" / "gate_evidence"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_KEYS = {
    "schema_version", "kind", "finalized_at_utc", "run_id",
    "candidate_initial_file_sha256", "source", "gate_ids",
    "gate_ids_sha256", "state", "pass_gates", "red_gates",
    "blocked_gates", "outcomes", "artifact_records", "invocations", "evidence",
    "hash_referents", "performance_prerequisite", "problems",
    "promotion_ready", "content_sha256",
}
OUTCOME_KEYS = {
    "schema_version", "kind", "run_id", "gate", "gate_definition_id",
    "source", "status", "returncode", "problems", "blocked_by", "log",
    "observed_evidence", "produced_artifacts", "content_sha256",
}
CANDIDATE_KEYS = {
    "schema_version", "kind", "created_at_utc", "source", "environment",
    "gate_ids", "green_gate_ids", "gate_ids_sha256", "gate_definitions",
    "release_contract", "performance_input", "run_root", "state", "content_sha256",
}


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


def run(*arguments: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        arguments,
        cwd=REPO,
        capture_output=True,
        text=text,
        check=False,
    )


def load_json(path: Path, failures: list[str], label: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"{label}: cannot parse {path}: {error}")
        return None
    if not isinstance(value, dict):
        failures.append(f"{label}: root is not an object")
        return None
    return value


def validate_content_hash(document: dict[str, Any], failures: list[str], label: str) -> None:
    actual = canonical_sha256(
        {key: value for key, value in document.items() if key != "content_sha256"}
    )
    if document.get("content_sha256") != actual:
        failures.append(f"{label}: content_sha256 mismatch")


def exact_snapshot_closure(
    evidence_dir: Path, gates: list[str], manifest: dict[str, Any],
    failures: list[str],
) -> None:
    """Reject symlinks, path aliases, omissions, and unmanifested extras."""
    expected_files = {
        Path("RUN_MANIFEST.json"),
        Path("RUN_CANDIDATE.json"),
        *(Path(f"{gate}.json") for gate in gates),
        *(Path("outcomes") / f"{gate}.json" for gate in gates),
        *(Path("logs") / f"{gate}.log" for gate in gates),
    }
    records = manifest.get("artifact_records")
    if isinstance(records, dict):
        expected_files.update(
            Path("artifact_records") / f"{producer}.json" for producer in records
        )
    invocations = manifest.get("invocations")
    if isinstance(invocations, dict):
        expected_files.update(
            Path("invocations") / f"{producer}.{phase}.json"
            for producer in invocations for phase in ("before", "after")
        )
    referents = manifest.get("hash_referents")
    if isinstance(referents, list):
        for item in referents:
            if not isinstance(item, dict):
                continue
            durable = item.get("durable_referent")
            if isinstance(durable, str) and durable.startswith("artifact:sha256/"):
                expected_files.add(Path("artifacts") / durable[len("artifact:"):])
    performance = manifest.get("performance_prerequisite")
    if isinstance(performance, dict):
        path = performance.get("path")
        if isinstance(path, str):
            expected_files.add(Path(path))
        artifacts = performance.get("evaluation_artifacts")
        if isinstance(artifacts, list):
            expected_files.update(
                Path(value["path"])
                for value in artifacts
                if isinstance(value, dict) and isinstance(value.get("path"), str)
            )
    expected_dirs = {Path(".")}
    for relative in expected_files:
        expected_dirs.update(relative.parents)

    if evidence_dir.is_symlink() or not evidence_dir.is_dir():
        failures.append("snapshot root is not a regular directory")
        return
    observed_files: set[Path] = set()
    observed_dirs: set[Path] = {Path(".")}
    for path in evidence_dir.rglob("*"):
        relative = path.relative_to(evidence_dir)
        if path.is_symlink():
            failures.append(f"snapshot contains symlink: {relative}")
            continue
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_dirs.add(relative)
        else:
            failures.append(f"snapshot contains non-regular entry: {relative}")
    missing = sorted(str(path) for path in expected_files - observed_files)
    extra = sorted(str(path) for path in observed_files - expected_files)
    extra_dirs = sorted(str(path) for path in observed_dirs - expected_dirs)
    if missing or extra or extra_dirs:
        failures.append(
            "snapshot closure is not exact: "
            f"missing={missing}, extra={extra}, extra_dirs={extra_dirs}"
        )


def expected_gates(failures: list[str]) -> list[str]:
    try:
        contract = load_contract()
        hash_contract = load_hash_contract(contract)
    except Exception as error:
        failures.append(f"cannot load reviewed release inventory: {error}")
        return []
    reviewed = contract_gate_names(contract)
    text = (REPO / "scripts" / "gate.sh").read_text()
    arrays: dict[str, list[str]] = {}
    for name in ("ALL_GATES", "GREEN_GATES"):
        match = re.search(rf"^{name}=\(([^)]*)\)", text, re.MULTILINE)
        if match is None:
            failures.append(f"cannot derive {name} from scripts/gate.sh")
            arrays[name] = []
        else:
            arrays[name] = shlex.split(match.group(1))
        if arrays[name] != reviewed:
            failures.append(f"{name} disagrees with reviewed release inventory")
    return reviewed


def git_blob_sha256(commit: str, relative: str) -> str | None:
    if relative.startswith("/") or ".." in Path(relative).parts:
        return None
    result = run("git", "show", f"{commit}:{relative}", text=False)
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def git_blob_bytes(commit: str, relative: str) -> bytes | None:
    if relative.startswith("/") or ".." in Path(relative).parts:
        return None
    result = run("git", "show", f"{commit}:{relative}", text=False)
    return result.stdout if result.returncode == 0 else None


def verify_performance_prerequisite(
    evidence_dir: Path, manifest: dict[str, Any], source: dict[str, Any],
    failures: list[str]
) -> None:
    contract = load_contract()
    specification = contract["performance_prerequisite"]
    recorded = manifest.get("performance_prerequisite")
    if not isinstance(recorded, dict) or set(recorded) != {
        "path", "sha256", "content_sha256", "evaluation_artifacts",
        "derived_attestation", "variance_model", "decision_rule"
    }:
        failures.append("run manifest has no exact performance prerequisite attestation")
        return
    if recorded.get("path") != specification["snapshot_path"]:
        failures.append("performance prerequisite path mismatch")
        return
    certificate_path = evidence_dir / specification["snapshot_path"]
    if not certificate_path.is_file() or certificate_path.is_symlink() or \
       file_sha256(certificate_path) != recorded.get("sha256"):
        failures.append("performance prerequisite does not resolve in the snapshot")
        return
    document, artifacts, attestation, problems = validate_release_certificate(
        certificate_path, source, specification["id"],
        REPO / specification["variance_model_path"],
        REPO / specification["decision_rule_path"],
    )
    failures.extend(f"performance: {problem}" for problem in problems)
    if document is None:
        return
    if document.get("content_sha256") != recorded.get("content_sha256"):
        failures.append("performance prerequisite content identity mismatch")
    expected_retained = [
        {
            "path": f"performance/{artifact['path']}",
            "sha256": artifact["sha256"],
            "bytes": artifact["bytes"],
            "kind": artifact["kind"],
        }
        for artifact in artifacts
    ]
    if recorded.get("evaluation_artifacts") != expected_retained:
        failures.append("performance evaluation-artifact manifest is not exact or ordered")
    if recorded.get("derived_attestation") != attestation:
        failures.append("performance decision is not reproducible from retained observations")
    for manifest_key, digest_key, path_key in (
        ("variance_model", "variance_model_sha256", "variance_model_path"),
        ("decision_rule", "decision_rule_sha256", "decision_rule_path"),
    ):
        entry = recorded.get(manifest_key)
        expected_path = specification[path_key]
        actual = git_blob_sha256(source.get("commit", ""), expected_path)
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or \
           entry.get("path") != expected_path or entry.get("sha256") != actual or \
           document.get(digest_key) != actual:
            failures.append(f"performance {manifest_key} does not resolve at the source commit")


def source_problems(
    source: Any, evidence_dir: Path, failures: list[str]
) -> tuple[str, str] | None:
    if not isinstance(source, dict):
        failures.append("manifest source is not an object")
        return None
    if set(source) != {
        "repository", "commit", "tree", "dirty", "clean_observation_sha256"
    }:
        failures.append("manifest source schema keys are not exact")
    commit, tree = source.get("commit"), source.get("tree")
    if not isinstance(commit, str) or not isinstance(tree, str):
        failures.append("manifest source lacks commit/tree")
        return None
    if not isinstance(source.get("repository"), str) or not source["repository"]:
        failures.append("manifest source lacks repository identity")
    else:
        remote = run("git", "config", "--get", "remote.origin.url")
        if remote.returncode != 0 or remote.stdout.strip() != source["repository"]:
            failures.append("manifest repository identity disagrees with this checkout")
    if source.get("dirty") is not False:
        failures.append("manifest source was not recorded clean")
    if source.get("clean_observation_sha256") != hashlib.sha256(b"").hexdigest():
        failures.append("manifest clean-tree observation is not the empty status digest")
    kind = run("git", "cat-file", "-t", commit)
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        failures.append(f"source commit is absent: {commit}")
        return None
    actual_tree = run("git", "rev-parse", f"{commit}^{{tree}}")
    if actual_tree.returncode != 0 or actual_tree.stdout.strip() != tree:
        failures.append("source tree disagrees with source commit")

    # A promoted evidence-only commit is allowed to differ beneath the
    # snapshot directory.  Any other tracked or untracked drift invalidates
    # the claim that this evidence describes the current checkout.
    diff = run(
        "git",
        "diff",
        "--quiet",
        commit,
        "--",
        ".",
        ":(exclude)fixtures/gate_evidence",
    )
    if diff.returncode != 0:
        failures.append("current tracked source differs from the recorded source commit")
    status = run("git", "status", "--porcelain", "--untracked-files=all")
    allowed = str(evidence_dir.relative_to(REPO)) + "/" if evidence_dir.is_relative_to(REPO) else None
    for line in status.stdout.splitlines():
        relative = line[3:] if len(line) >= 4 else line
        if allowed is None or not relative.startswith(allowed):
            failures.append(f"working tree drift outside evidence snapshot: {line}")
    return commit, tree


def verify_writer_contract(
    gate: str, document: dict[str, Any], commit: str, failures: list[str]
) -> None:
    result = run("git", "show", f"{commit}:scripts/gates/{gate}.sh")
    if result.returncode != 0:
        failures.append(f"{gate}: writer script absent at source commit")
        return
    writes_host = re.search(r"^\s*['\"]host['\"]\s*:", result.stdout, re.MULTILINE) is not None
    writes_profile = re.search(
        r"^\s*['\"]host_profile['\"]\s*:", result.stdout, re.MULTILINE
    ) is not None
    if "host" in document and not writes_host:
        failures.append(f"{gate}: evidence has host but source writer does not emit it")
    if "host_profile" in document and not writes_profile:
        failures.append(f"{gate}: evidence has host_profile but source writer does not emit it")
    if "host" not in document and "host_profile" not in document:
        failures.append(f"{gate}: evidence has no host identity")


def audit(evidence_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    gates = expected_gates(failures)
    try:
        contract = load_contract()
        hash_contract = load_hash_contract(contract)
    except Exception as error:
        failures.append(f"cannot load reviewed evidence contracts: {error}")
        contract = {"gates": [], "performance_prerequisite": {}}
        hash_contract = {"gates": {}}
    manifest_path = evidence_dir / "RUN_MANIFEST.json"
    manifest = load_json(manifest_path, failures, "run manifest")
    if manifest is None:
        return {"ok": False, "failures": failures, "gate_count": len(gates)}
    validate_content_hash(manifest, failures, "run manifest")
    if set(manifest) != MANIFEST_KEYS:
        failures.append("run manifest schema keys are not exact")
    if manifest.get("schema_version") != 1:
        failures.append("run manifest schema version mismatch")
    if manifest.get("kind") != "tgrad-gate-evidence-candidate-manifest":
        failures.append("run manifest has unexpected kind")
    if manifest.get("state") != "complete_green" or manifest.get("promotion_ready") is not True:
        failures.append("run manifest is not complete_green and promotion-ready")
    if manifest.get("gate_ids") != gates:
        failures.append("run manifest gate order/cover disagrees with ALL_GATES")
    if manifest.get("gate_ids_sha256") != canonical_sha256(gates):
        failures.append("run manifest gate inventory hash mismatch")
    if manifest.get("pass_gates") != gates:
        failures.append("not every expected gate is recorded as passing")
    if manifest.get("red_gates") != [] or manifest.get("blocked_gates") != []:
        failures.append("run manifest contains red or blocked gates")
    if manifest.get("problems") != []:
        failures.append("run manifest contains unresolved problems")
    exact_snapshot_closure(evidence_dir, gates, manifest, failures)

    candidate_path = evidence_dir / "RUN_CANDIDATE.json"
    candidate = load_json(candidate_path, failures, "initial candidate")
    if candidate is not None:
        validate_content_hash(candidate, failures, "initial candidate")
        if set(candidate) != CANDIDATE_KEYS:
            failures.append("initial candidate schema keys are not exact")
        if candidate.get("schema_version") != 1:
            failures.append("initial candidate schema version mismatch")
        if candidate.get("kind") != "tgrad-gate-evidence-candidate":
            failures.append("initial candidate has unexpected kind")
        if candidate.get("state") != "collecting":
            failures.append("initial candidate is not the immutable collecting record")
        if candidate.get("content_sha256") != manifest.get("run_id"):
            failures.append("run manifest run_id disagrees with initial candidate")
        if file_sha256(candidate_path) != manifest.get("candidate_initial_file_sha256"):
            failures.append("initial candidate bytes disagree with run manifest")
        if candidate.get("source") != manifest.get("source"):
            failures.append("initial candidate source disagrees with run manifest")
        if candidate.get("gate_ids") != gates:
            failures.append("initial candidate gate inventory disagrees with ALL_GATES")
        if candidate.get("green_gate_ids") != gates:
            failures.append("initial candidate green-gate inventory mismatch")
        expected_contract = {
            "id": contract["contract_id"],
            "path": str(INVENTORY.relative_to(REPO)),
            "sha256": REVIEWED_INVENTORY_SHA256,
            "performance_prerequisite": contract["performance_prerequisite"],
            "hash_contract": {
                "id": hash_contract["contract_id"],
                "path": str(HASH_CONTRACT.relative_to(REPO)),
                "sha256": REVIEWED_HASH_CONTRACT_SHA256,
            },
        }
        if candidate.get("release_contract") != expected_contract:
            failures.append("initial candidate release contract mismatch")
        performance_input = candidate.get("performance_input")
        recorded_performance = manifest.get("performance_prerequisite")
        if not isinstance(performance_input, dict) or set(performance_input) != {
            "certificate", "evaluation_artifacts", "derived_attestation",
            "declared_source", "source_matches_candidate"
        }:
            failures.append("initial candidate has no exact retained performance input")
        elif not isinstance(recorded_performance, dict):
            failures.append("run manifest lost the retained performance input")
        else:
            certificate = performance_input.get("certificate")
            if not isinstance(certificate, dict) or \
               certificate.get("path") != recorded_performance.get("path") or \
               certificate.get("sha256") != recorded_performance.get("sha256"):
                failures.append("performance certificate changed after candidate initialization")
            if performance_input.get("evaluation_artifacts") != \
               recorded_performance.get("evaluation_artifacts"):
                failures.append("performance evaluation artifacts changed after candidate initialization")
            if performance_input.get("derived_attestation") != \
               recorded_performance.get("derived_attestation"):
                failures.append("performance derived decision changed after candidate initialization")
            if performance_input.get("declared_source") != manifest.get("source") or \
               performance_input.get("source_matches_candidate") is not True:
                failures.append("performance input does not name the candidate source")
        definitions = candidate.get("gate_definitions")
        if not isinstance(definitions, list) or len(definitions) != len(gates):
            failures.append("initial candidate gate definitions do not cover ALL_GATES")
            definitions = []
        by_gate = {
            value.get("gate"): value for value in definitions if isinstance(value, dict)
        }
        if set(by_gate) != set(gates):
            failures.append("initial candidate gate-definition names are not exact")
        for gate in gates:
            definition = by_gate.get(gate)
            if not isinstance(definition, dict):
                continue
            expected_path = f"scripts/gates/{gate}.sh"
            contract_gate = next(value for value in contract["gates"] if value["name"] == gate)
            writer_digest = git_blob_sha256(manifest.get("source", {}).get("commit", ""), expected_path)
            fields = {
                "gate": gate,
                "writer_path": expected_path,
                "writer_sha256": writer_digest,
                "depends_on": contract_gate["depends_on"],
            }
            if definition != {"id": canonical_sha256(fields), **fields}:
                failures.append(f"{gate}: immutable writer definition mismatch")
        environment = candidate.get("environment")
        if not isinstance(environment, dict):
            failures.append("initial candidate has no environment identity")
        else:
            validate_content_hash(environment, failures, "candidate environment")
            selected = environment.get("selected_environment")
            if not isinstance(selected, dict):
                failures.append("candidate environment has no selected-variable map")
            else:
                forbidden = {
                    name for name in (
                        "TGRAD_PERF_BASELINE", "TGRAD_PERF_BASELINE_FULL",
                        "TGRAD_PERF_BASELINE_TC",
                    ) if selected.get(name)
                }
                profile = selected.get(
                    "TGRAD_PERF_PROFILE",
                    selected.get("TGRAD_HOST", "apple_m4_mini_release"),
                )
                if forbidden or profile != "apple_m4_mini_release":
                    failures.append(
                        "candidate environment does not consume the reviewed performance inputs"
                    )

    source = source_problems(manifest.get("source"), evidence_dir, failures)
    commit = source[0] if source is not None else ""
    if isinstance(manifest.get("source"), dict):
        verify_performance_prerequisite(
            evidence_dir, manifest, manifest["source"], failures
        )

    json_files = {
        path.stem: path
        for path in evidence_dir.glob("*.json")
        if path.name not in {"RUN_MANIFEST.json", "RUN_CANDIDATE.json"}
    }
    if set(json_files) != set(gates):
        failures.append(
            "evidence cover is not exact: "
            f"missing={sorted(set(gates)-set(json_files))}, "
            f"extra={sorted(set(json_files)-set(gates))}"
        )
    documents: dict[str, dict[str, Any]] = {}
    for gate, path in json_files.items():
        document = load_json(path, failures, f"{gate} evidence")
        if document is None:
            continue
        documents[gate] = document
        if document.get("gate") != gate:
            failures.append(f"{gate}: gate field disagrees with filename")
        if document.get("commit") != commit:
            failures.append(f"{gate}: commit disagrees with run source")
        if not isinstance(document.get("hashes"), dict):
            failures.append(f"{gate}: hashes is not an object")
        for problem in evidence_document_problems(
            hash_contract, gate, document, commit
        ):
            failures.append(f"{gate}: {problem}")
        if commit:
            verify_writer_contract(gate, document, commit, failures)

    manifest_evidence = manifest.get("evidence")
    if not isinstance(manifest_evidence, dict) or set(manifest_evidence) != set(gates):
        failures.append("manifest evidence map does not exactly cover ALL_GATES")
        manifest_evidence = {}
    for gate, path in json_files.items():
        entry = manifest_evidence.get(gate)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"} or \
           entry.get("sha256") != file_sha256(path) or \
           entry.get("bytes") != path.stat().st_size:
            failures.append(f"{gate}: evidence bytes disagree with run manifest")

    outcomes = manifest.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != set(gates):
        failures.append("manifest outcomes do not exactly cover ALL_GATES")
        outcomes = {}
    outcome_documents: dict[str, dict[str, Any]] = {}
    for gate in gates:
        outcome_path = evidence_dir / "outcomes" / f"{gate}.json"
        outcome = load_json(outcome_path, failures, f"{gate} outcome")
        entry = outcomes.get(gate)
        if outcome is None:
            continue
        outcome_documents[gate] = outcome
        validate_content_hash(outcome, failures, f"{gate} outcome")
        if set(outcome) != OUTCOME_KEYS:
            failures.append(f"{gate}: outcome schema keys are not exact")
        gate_definition_id = None
        if candidate is not None and isinstance(candidate.get("gate_definitions"), list):
            gate_definition_id = next(
                (value.get("id") for value in candidate["gate_definitions"]
                 if isinstance(value, dict) and value.get("gate") == gate),
                None,
            )
        if (outcome.get("gate") != gate or
            outcome.get("gate_definition_id") != gate_definition_id or
            outcome.get("source") != manifest.get("source") or
            outcome.get("run_id") != manifest.get("run_id")):
            failures.append(f"{gate}: outcome identity mismatch")
        if outcome.get("status") != "pass" or outcome.get("returncode") != 0:
            failures.append(f"{gate}: promoted outcome was not a zero-return pass")
        if outcome.get("problems") != [] or outcome.get("blocked_by") != []:
            failures.append(f"{gate}: passing outcome contains problems or blockers")
        if not isinstance(entry, dict) or entry.get("status") != outcome.get("status"):
            failures.append(f"{gate}: manifest outcome status mismatch")
        if not isinstance(entry, dict) or entry.get("sha256") != file_sha256(outcome_path):
            failures.append(f"{gate}: outcome bytes disagree with run manifest")
        log_path = evidence_dir / "logs" / f"{gate}.log"
        log = outcome.get("log")
        if not log_path.is_file() or not isinstance(log, dict):
            failures.append(f"{gate}: retained process log is missing")
        else:
            if log.get("path") != f"logs/{gate}.log":
                failures.append(f"{gate}: process log locator is not snapshot-relative")
            if set(log) != {"path", "sha256", "bytes"} or \
               log.get("sha256") != file_sha256(log_path) or \
               log.get("bytes") != log_path.stat().st_size:
                failures.append(f"{gate}: retained process log hash mismatch")
        observed = outcome.get("observed_evidence")
        final_evidence = json_files.get(gate)
        if not isinstance(observed, dict) or final_evidence is None:
            failures.append(f"{gate}: passing outcome lacks observed evidence")
        elif (set(observed) != {"path", "sha256", "bytes"} or
              observed.get("path") != f"evidence/{gate}.json" or
              observed.get("sha256") != file_sha256(final_evidence) or
              observed.get("bytes") != final_evidence.stat().st_size):
            failures.append(f"{gate}: final evidence was not produced by its passing process")

    artifact_record_manifest = manifest.get("artifact_records")
    invocation_manifest = manifest.get("invocations")
    expected_producers = {"PREFLIGHT", *gates}
    if not isinstance(artifact_record_manifest, dict) or \
       set(artifact_record_manifest) != expected_producers:
        failures.append("artifact records do not exactly cover PREFLIGHT and ALL_GATES")
        artifact_record_manifest = {}
    if not isinstance(invocation_manifest, dict) or \
       set(invocation_manifest) != expected_producers:
        failures.append("invocation observations do not exactly cover PREFLIGHT and ALL_GATES")
        invocation_manifest = {}
    produced_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    for producer in expected_producers:
        record_path = evidence_dir / "artifact_records" / f"{producer}.json"
        record = load_json(record_path, failures, f"{producer} artifact record")
        if record is None:
            continue
        validate_content_hash(record, failures, f"{producer} artifact record")
        required_record_keys = {
            "schema_version", "kind", "run_id", "producer",
            "before_observation_sha256", "after_observation_sha256",
            "artifacts", "content_sha256",
        }
        if set(record) != required_record_keys or record.get("schema_version") != 1 or \
           record.get("kind") != "tgrad-produced-artifacts" or \
           record.get("run_id") != manifest.get("run_id") or \
           record.get("producer") != producer:
            failures.append(f"{producer}: artifact record identity/schema mismatch")
        manifest_entry = artifact_record_manifest.get(producer)
        if not isinstance(manifest_entry, dict) or set(manifest_entry) != {"sha256"} or \
           manifest_entry.get("sha256") != file_sha256(record_path):
            failures.append(f"{producer}: artifact record bytes disagree with manifest")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list):
            failures.append(f"{producer}: artifact record list is malformed")
            continue
        by_logical: dict[str, dict[str, Any]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {
                "producer", "logical_path", "sha256", "bytes", "retained_path"
            }:
                failures.append(f"{producer}: malformed produced-artifact entry")
                continue
            logical = artifact.get("logical_path")
            digest = artifact.get("sha256")
            if artifact.get("producer") != producer or not isinstance(logical, str) or \
               logical.startswith("/") or ".." in Path(logical).parts or \
               not isinstance(digest, str) or SHA256.fullmatch(digest) is None or \
               artifact.get("retained_path") != f"artifacts/sha256/{digest[:2]}/{digest}":
                failures.append(f"{producer}: produced-artifact identity is malformed")
                continue
            if logical in by_logical:
                failures.append(f"{producer}: duplicate produced logical path {logical}")
            by_logical[logical] = artifact
        observation_entries = invocation_manifest.get(producer)
        observations: dict[str, dict[str, Any]] = {}
        if not isinstance(observation_entries, dict) or \
           set(observation_entries) != {"before", "after"}:
            failures.append(f"{producer}: invocation manifest entry is malformed")
        else:
            for phase in ("before", "after"):
                observation_path = evidence_dir / "invocations" / f"{producer}.{phase}.json"
                observation = load_json(
                    observation_path, failures, f"{producer} {phase} observation"
                )
                manifest_observation = observation_entries.get(phase)
                if observation is None:
                    continue
                validate_content_hash(
                    observation, failures, f"{producer} {phase} observation"
                )
                if set(observation) != {
                    "schema_version", "kind", "run_id", "producer", "phase",
                    "artifacts", "content_sha256",
                } or observation.get("schema_version") != 1 or \
                   observation.get("kind") != "tgrad-artifact-observation" or \
                   observation.get("run_id") != manifest.get("run_id") or \
                   observation.get("producer") != producer or \
                   observation.get("phase") != phase or \
                   not isinstance(observation.get("artifacts"), dict):
                    failures.append(f"{producer}: {phase} observation identity/schema mismatch")
                    continue
                if not isinstance(manifest_observation, dict) or \
                   set(manifest_observation) != {"sha256"} or \
                   manifest_observation.get("sha256") != file_sha256(observation_path) or \
                   record.get(f"{phase}_observation_sha256") != file_sha256(observation_path):
                    failures.append(f"{producer}: {phase} observation bytes are not bound")
                malformed = False
                for logical, artifact in observation["artifacts"].items():
                    if not isinstance(logical, str) or logical.startswith("/") or \
                       ".." in Path(logical).parts or not isinstance(artifact, dict) or \
                       set(artifact) != {"sha256", "bytes", "mtime_ns"} or \
                       not isinstance(artifact.get("sha256"), str) or \
                       SHA256.fullmatch(artifact["sha256"]) is None or \
                       not isinstance(artifact.get("bytes"), int) or artifact["bytes"] < 0 or \
                       not isinstance(artifact.get("mtime_ns"), int) or artifact["mtime_ns"] < 0:
                        malformed = True
                if malformed:
                    failures.append(f"{producer}: {phase} observation artifact is malformed")
                observations[phase] = observation["artifacts"]
        if set(observations) == {"before", "after"}:
            expected_changed = {
                logical: artifact for logical, artifact in observations["after"].items()
                if observations["before"].get(logical) != artifact
            }
            observed_changed = {
                logical: {key: artifact.get(key) for key in ("sha256", "bytes")}
                for logical, artifact in by_logical.items()
            }
            expected_changed_reduced = {
                logical: {key: artifact.get(key) for key in ("sha256", "bytes")}
                for logical, artifact in expected_changed.items()
            }
            if observed_changed != expected_changed_reduced:
                failures.append(
                    f"{producer}: produced artifacts disagree with retained observations"
                )
        produced_artifacts[producer] = by_logical
        if producer != "PREFLIGHT":
            outcome = outcome_documents.get(producer)
            produced = outcome.get("produced_artifacts") if outcome is not None else None
            if not isinstance(produced, dict) or set(produced) != {"count", "record_sha256"} or \
               produced.get("record_sha256") != file_sha256(record_path) or \
               produced.get("count") != len(artifacts):
                failures.append(f"{producer}: outcome does not bind its artifact record")

    expected_referents: dict[tuple[str, str], str] = {}
    hash_policies = hash_contract["gates"]
    for gate, document in documents.items():
        hashes = document.get("hashes")
        if not isinstance(hashes, dict):
            continue
        if set(hashes) != set(hash_policies.get(gate, {})):
            failures.append(f"{gate}: hash keys disagree with reviewed hash contract")
        for key, digest in hashes.items():
            if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
                failures.append(f"{gate}:{key} is not a SHA-256")
                continue
            expected_referents[(gate, key)] = digest

    seen_referents: dict[tuple[str, str], str] = {}
    referents = manifest.get("hash_referents")
    if not isinstance(referents, list):
        failures.append("manifest hash_referents is not a list")
        referents = []
    for item in referents:
        if not isinstance(item, dict):
            failures.append("manifest contains a malformed hash referent")
            continue
        if set(item) != {"gate", "key", "sha256", "referents", "durable_referent"}:
            failures.append("manifest hash referent schema keys are not exact")
        identity = (item.get("gate"), item.get("key"))
        digest, durable = item.get("sha256"), item.get("durable_referent")
        if not all(isinstance(value, str) for value in (*identity, digest, durable)):
            failures.append("manifest contains an incomplete hash referent")
            continue
        key = (identity[0], identity[1])
        if key in seen_referents:
            failures.append(f"duplicate hash referent for {key[0]}:{key[1]}")
        seen_referents[key] = digest
        policy = hash_policies.get(key[0], {}).get(key[1])
        provenance = item.get("referents")
        if not isinstance(provenance, list) or not provenance or any(
            not isinstance(value, dict) or set(value) != {"identity", "bytes"} or
            not isinstance(value.get("identity"), str) or
            not isinstance(value.get("bytes"), int) or value["bytes"] < 0
            for value in (provenance if isinstance(provenance, list) else [])
        ):
            failures.append(f"{key[0]}:{key[1]} provenance entries are malformed")
        provenance_ids = {
            value.get("identity") for value in provenance if isinstance(value, dict)
        } if isinstance(provenance, list) else set()
        if isinstance(policy, str) and policy.startswith("source:"):
            expected_provenance = f"repo:{policy[len('source:'):]}"
            if provenance_ids != {expected_provenance} or durable != expected_provenance:
                failures.append(f"{key[0]}:{key[1]} source locator violates hash contract")
        elif isinstance(policy, str) and policy.startswith("evidence:"):
            expected_provenance = policy
            if provenance_ids != {expected_provenance} or durable != expected_provenance:
                failures.append(f"{key[0]}:{key[1]} evidence locator violates hash contract")
        elif isinstance(policy, str) and policy.startswith("artifact:"):
            allowed = (f"artifact:{key[0]}:", "artifact:PREFLIGHT:")
            logical_pattern = policy[len("artifact:"):]
            if not provenance_ids or any(
                not isinstance(value, str) or not value.startswith(allowed) or
                len(value.split(":", 2)) != 3 or
                not fnmatch.fnmatchcase(value.split(":", 2)[2], logical_pattern)
                for value in provenance_ids
            ) or not durable.startswith("artifact:sha256/"):
                failures.append(f"{key[0]}:{key[1]} artifact producer violates hash contract")
            for provenance_id in provenance_ids:
                if not isinstance(provenance_id, str) or not provenance_id.startswith("artifact:"):
                    continue
                parts = provenance_id.split(":", 2)
                if len(parts) != 3:
                    failures.append(f"{key[0]}:{key[1]} has malformed artifact provenance")
                    continue
                _, producer, logical = parts
                produced = produced_artifacts.get(producer, {}).get(logical)
                if produced is None or produced.get("sha256") != digest:
                    failures.append(
                        f"{key[0]}:{key[1]} is not present in {producer}'s retained producer record"
                    )
        else:
            failures.append(f"{key[0]}:{key[1]} has no reviewed hash policy")
        if durable.startswith("repo:"):
            raw = git_blob_bytes(commit, durable[len("repo:"):]) if commit else None
            actual = hashlib.sha256(raw).hexdigest() if raw is not None else None
            actual_bytes = len(raw) if raw is not None else None
        elif durable.startswith("evidence:"):
            child = durable[len("evidence:"):]
            child_path = json_files.get(child)
            actual = file_sha256(child_path) if child_path is not None else None
            actual_bytes = child_path.stat().st_size if child_path is not None else None
        elif durable.startswith("artifact:sha256/"):
            relative = durable[len("artifact:"):]
            artifact = evidence_dir / "artifacts" / relative
            actual = file_sha256(artifact) if artifact.is_file() else None
            actual_bytes = artifact.stat().st_size if artifact.is_file() else None
        else:
            actual = None
            actual_bytes = None
        if actual != digest:
            failures.append(f"{key[0]}:{key[1]} durable referent does not hash to {digest}")
        if isinstance(provenance, list) and actual_bytes is not None and any(
            value.get("bytes") != actual_bytes for value in provenance
            if isinstance(value, dict)
        ):
            failures.append(f"{key[0]}:{key[1]} provenance size disagrees with durable bytes")
    if seen_referents != expected_referents:
        failures.append("manifest hash referents do not exactly cover evidence hash claims")

    folded = {gate.casefold(): gate for gate in documents}
    for gate, document in documents.items():
        for key, digest in (document.get("hashes") or {}).items():
            if not key.lower().endswith("_evidence_sha256"):
                continue
            child = folded.get(key[: -len("_evidence_sha256")].casefold())
            if child is None:
                failures.append(f"{gate}:{key} names a missing child evidence file")
            elif file_sha256(json_files[child]) != digest:
                failures.append(f"{gate}:{key} disagrees with {child}.json")

    return {
        "ok": not failures,
        "source": manifest.get("source"),
        "gate_count": len(gates),
        "evidence_count": len(documents),
        "hash_claim_count": len(expected_referents),
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    result.add_argument("--strict", action="store_true", help="retained for explicit release-check spelling")
    result.add_argument("--quiet", action="store_true")
    result.add_argument("--json", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = audit(args.evidence_dir.resolve())
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2))
    elif not args.quiet or not result["ok"]:
        verdict = "OK" if result["ok"] else "FAIL"
        print(f"evidence_provenance: {verdict}")
        print(
            f"  gates={result.get('gate_count', 0)} "
            f"evidence={result.get('evidence_count', 0)} "
            f"hash_claims={result.get('hash_claim_count', 0)}"
        )
        for failure in result["failures"]:
            print(f"  x {failure}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

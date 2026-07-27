#!/usr/bin/env python3
"""Offline falsifiers for candidate collection and strict provenance audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load_module("evidence_candidate", ROOT / "scripts/evidence/candidate.py")
auditor = load_module(
    "evidence_provenance_audit", ROOT / "scripts/dev/evidence_provenance_audit.py"
)
REAL_PERFORMANCE_PREREQUISITE = candidate.performance_prerequisite


SOURCE = {
    "repository": "test://tgrad",
    "commit": "1" * 40,
    "tree": "2" * 40,
    "dirty": False,
    "clean_observation_sha256": hashlib.sha256(b"").hexdigest(),
}

TEST_CONTRACT = {
    "schema_version": 1,
    "contract_id": "test-release",
    "performance_prerequisite": {
        "id": "prepared-runtime-repeatability-v1",
        "snapshot_path": "performance/prepared_runtime_certificate.json",
        "required_state": "promoted",
        "same_source": True,
        "variance_model_path": "scripts/perf/VARIANCE_MODEL.md",
        "decision_rule_path": "scripts/perf/repeatability_decision.json",
    },
    "gates": [
        {"name": "A", "writer": "scripts/gates/A.sh", "depends_on": []},
        {"name": "B", "writer": "scripts/gates/B.sh", "depends_on": ["A"]},
    ],
}
TEST_HASH_CONTRACT = {
    "schema_version": 1,
    "contract_id": "test-hashes",
    "gates": {
        "A": {"runtime_sha256": "artifact:run/result.bin"},
        "B": {"A_evidence_sha256": "evidence:A"},
    },
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_gate_definitions(gates: list[str]) -> list[dict]:
    result = []
    for gate in gates:
        fields = {
            "gate": gate,
            "writer_path": f"scripts/gates/{gate}.sh",
            "writer_sha256": digest((gate + "-writer").encode()),
            "depends_on": [] if gate == "A" else ["A"],
        }
        result.append({"id": candidate.canonical_sha256(fields), **fields})
    return result


class EvidenceSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="tgrad_evidence_test.")
        self.base = Path(self.temporary.name)
        self.run_root = self.base / "run"
        self.run_root.mkdir()
        (self.run_root / ".tgrad-run-owner").write_text("test-owner\n")
        self.performance_root = self.base / "performance-input"
        self.performance_root.mkdir()
        raw = self.performance_root / "run.raw.jsonl"
        raw.write_text("{}\n")
        self.performance_certificate = self.performance_root / "certificate.json"
        self.performance_certificate.write_text(json.dumps({
            "source": SOURCE,
            "raw_artifacts": [{"path": raw.name, "sha256": digest(raw.read_bytes())}],
        }))
        self.candidate_root = self.base / "candidate"
        self.source_patch = mock.patch.object(candidate, "source_identity", return_value=SOURCE)
        self.command_patch = mock.patch.object(candidate, "command", return_value="")
        self.gates_patch = mock.patch.object(
            candidate,
            "build_gate_definitions",
            side_effect=fake_gate_definitions,
        )
        self.contract_patch = mock.patch.object(
            candidate, "load_contract", return_value=TEST_CONTRACT
        )
        self.contract_names_patch = mock.patch.object(
            candidate, "contract_gate_names", side_effect=lambda contract: ["A", "B"]
        )
        self.hash_contract_patch = mock.patch.object(
            candidate, "load_hash_contract", return_value=TEST_HASH_CONTRACT
        )
        self.prerequisite_patch = mock.patch.object(
            candidate,
            "performance_prerequisite",
            return_value=(
                {
                    "content_sha256": "a" * 64,
                    "variance_model_sha256": "b" * 64,
                    "decision_rule_sha256": "c" * 64,
                },
                [],
            ),
        )
        def fake_retain(root, certificate_path, source, specification):
            performance = root / "performance"
            performance.mkdir(parents=True)
            certificate = performance / "prepared_runtime_certificate.json"
            shutil.copyfile(certificate_path, certificate)
            raw = performance / "run.raw.jsonl"
            shutil.copyfile(self.performance_root / "run.raw.jsonl", raw)
            return {
                "certificate": {
                    "path": "performance/prepared_runtime_certificate.json",
                    "sha256": candidate.file_sha256(certificate),
                    "bytes": certificate.stat().st_size,
                },
                "evaluation_artifacts": [{
                    "path": "performance/run.raw.jsonl",
                    "sha256": candidate.file_sha256(raw),
                    "bytes": raw.stat().st_size,
                    "kind": "raw",
                }],
                "derived_attestation": {"accepted": True},
                "declared_source": source,
                "source_matches_candidate": True,
            }
        self.retention_patch = mock.patch.object(
            candidate, "retain_performance_input", side_effect=fake_retain
        )
        self.source_patch.start()
        self.command_patch.start()
        self.gates_patch.start()
        self.contract_patch.start()
        self.contract_names_patch.start()
        self.hash_contract_patch.start()
        self.prerequisite_patch.start()
        self.retention_patch.start()

    def tearDown(self) -> None:
        self.command_patch.stop()
        self.source_patch.stop()
        self.gates_patch.stop()
        self.contract_patch.stop()
        self.contract_names_patch.stop()
        self.hash_contract_patch.stop()
        self.prerequisite_patch.stop()
        self.retention_patch.stop()
        self.temporary.cleanup()

    def initialize(self) -> None:
        candidate.init_candidate(
            self.candidate_root, ["A", "B"], self.run_root,
            performance_certificate=self.performance_certificate,
        )

    def collect_green(self) -> dict:
        self.initialize()
        candidate.begin_invocation(self.candidate_root, "PREFLIGHT")
        candidate.capture_produced_artifacts(self.candidate_root, "PREFLIGHT")
        candidate.begin_invocation(self.candidate_root, "A")
        payload = b"runtime-result"
        runtime_artifact = self.run_root / "result.bin"
        runtime_artifact.write_bytes(payload)
        a = {
            "gate": "A",
            "ts_utc": "2026-07-27T00:00:00Z",
            "commit": SOURCE["commit"],
            "host": "test",
            "platform": "test",
            "hashes": {"runtime_sha256": digest(payload)},
        }
        candidate.atomic_json(self.candidate_root / "evidence/A.json", a)
        log_a = self.candidate_root / "logs/A.log"
        log_a.write_text("A passed\n")
        candidate.record_outcome(self.candidate_root, "A", "pass", 0, log_a)

        candidate.begin_invocation(self.candidate_root, "B")
        a_digest = candidate.file_sha256(self.candidate_root / "evidence/A.json")
        b = {
            "gate": "B",
            "ts_utc": "2026-07-27T00:00:01Z",
            "commit": SOURCE["commit"],
            "host": "test",
            "platform": "test",
            "hashes": {"A_evidence_sha256": a_digest},
        }
        candidate.atomic_json(self.candidate_root / "evidence/B.json", b)
        log_b = self.candidate_root / "logs/B.log"
        log_b.write_text("B passed\n")
        candidate.record_outcome(self.candidate_root, "B", "pass", 0, log_b)
        return candidate.finalize_candidate(self.candidate_root)

    def promoted_directory(self, manifest: dict) -> Path:
        snapshot = self.base / "promoted"
        shutil.copytree(self.candidate_root / "evidence", snapshot)
        shutil.copytree(self.candidate_root / "outcomes", snapshot / "outcomes")
        shutil.copytree(self.candidate_root / "logs", snapshot / "logs")
        shutil.copytree(
            self.candidate_root / "artifact_records", snapshot / "artifact_records"
        )
        shutil.copytree(self.candidate_root / "invocations", snapshot / "invocations")
        shutil.copytree(self.candidate_root / "release_artifacts", snapshot / "artifacts")
        shutil.copytree(self.candidate_root / "performance", snapshot / "performance")
        candidate.atomic_json(snapshot / "RUN_MANIFEST.json", manifest)
        shutil.copyfile(
            self.candidate_root / "candidate.initial.json",
            snapshot / "RUN_CANDIDATE.json",
        )
        return snapshot

    def audit(self, snapshot: Path) -> dict:
        with (
            mock.patch.object(auditor, "expected_gates", return_value=["A", "B"]),
            mock.patch.object(
                auditor,
                "source_problems",
                return_value=(SOURCE["commit"], SOURCE["tree"]),
            ),
            mock.patch.object(auditor, "verify_writer_contract", return_value=None),
            mock.patch.object(auditor, "verify_performance_prerequisite", return_value=None),
            mock.patch.object(auditor, "load_contract", return_value=TEST_CONTRACT),
            mock.patch.object(
                auditor, "load_hash_contract", return_value=TEST_HASH_CONTRACT
            ),
            mock.patch.object(
                auditor,
                "git_blob_sha256",
                side_effect=lambda _commit, path: digest(
                    (Path(path).stem + "-writer").encode()
                ),
            ),
        ):
            return auditor.audit(snapshot)

    def test_complete_green_snapshot_is_self_contained(self) -> None:
        manifest = self.collect_green()
        self.assertEqual(manifest["state"], "complete_green")
        snapshot = self.promoted_directory(manifest)
        result = self.audit(snapshot)
        self.assertTrue(result["ok"], result["failures"])

    def test_tampered_retained_artifact_is_rejected(self) -> None:
        manifest = self.collect_green()
        snapshot = self.promoted_directory(manifest)
        artifact = next((snapshot / "artifacts").rglob("runtime-result-does-not-exist"), None)
        self.assertIsNone(artifact)
        retained = [path for path in (snapshot / "artifacts").rglob("*") if path.is_file()]
        self.assertTrue(retained)
        retained[0].write_bytes(b"tampered")
        result = self.audit(snapshot)
        self.assertFalse(result["ok"])
        self.assertTrue(any("durable referent" in failure for failure in result["failures"]))

    def test_evidence_replacement_after_pass_is_rejected(self) -> None:
        manifest = self.collect_green()
        snapshot = self.promoted_directory(manifest)
        evidence = json.loads((snapshot / "A.json").read_text())
        evidence["host"] = "replacement"
        candidate.atomic_json(snapshot / "A.json", evidence)
        result = self.audit(snapshot)
        self.assertFalse(result["ok"])
        self.assertTrue(any("evidence bytes" in failure or "passing process" in failure
                            for failure in result["failures"]))

    def test_unmanifested_file_and_symlink_are_rejected(self) -> None:
        manifest = self.collect_green()
        snapshot = self.promoted_directory(manifest)
        (snapshot / "extra.txt").write_text("not in closure\n")
        (snapshot / "alias.json").symlink_to(snapshot / "A.json")
        result = self.audit(snapshot)
        self.assertFalse(result["ok"])
        self.assertTrue(any("snapshot closure" in failure or "symlink" in failure
                            for failure in result["failures"]))

    def test_gate_inventory_cannot_be_shrunk(self) -> None:
        with self.assertRaises(candidate.CandidateError):
            candidate.init_candidate(self.candidate_root, ["A"], self.run_root, ["A"])

    def test_artifact_from_another_gate_does_not_resolve(self) -> None:
        payload = b"other producer"
        value = digest(payload)
        documents = {"A": {"hashes": {"runtime_sha256": value}}}
        index = {
            value: [{
                "identity": "artifact:B:run/result.bin",
                "path": str(self.run_root / "result.bin"),
                "bytes": len(payload),
            }]
        }
        _, unresolved = candidate.evidence_hash_referents(documents, index)
        self.assertEqual(unresolved, ["A:runtime_sha256"])

    def test_artifact_capture_requires_a_begin_observation(self) -> None:
        self.initialize()
        with self.assertRaises(candidate.CandidateError):
            candidate.capture_produced_artifacts(self.candidate_root, "A")

    def test_tampered_invocation_observation_is_rejected(self) -> None:
        manifest = self.collect_green()
        snapshot = self.promoted_directory(manifest)
        observation = snapshot / "invocations/A.before.json"
        document = json.loads(observation.read_text())
        document["producer"] = "B"
        candidate.atomic_json(observation, document)
        result = self.audit(snapshot)
        self.assertFalse(result["ok"])
        self.assertTrue(any("observation" in failure for failure in result["failures"]))

    def test_artifact_from_wrong_logical_path_does_not_resolve(self) -> None:
        payload = b"wrong path"
        value = digest(payload)
        documents = {"A": {"hashes": {"runtime_sha256": value}}}
        index = {
            value: [{
                "identity": "artifact:A:run/not-result.bin",
                "path": str(self.run_root / "not-result.bin"),
                "bytes": len(payload),
            }]
        }
        _, unresolved = candidate.evidence_hash_referents(documents, index)
        self.assertEqual(unresolved, ["A:runtime_sha256"])

    def test_source_hash_cannot_resolve_to_a_different_path(self) -> None:
        value = digest(b"same bytes")
        documents = {"A": {"hashes": {"runtime_sha256": value}}}
        index = {
            value: [{
                "identity": "repo:wrong/path",
                "path": str(self.run_root / "wrong"),
                "bytes": 10,
            }]
        }
        source_contract = {
            "schema_version": 1,
            "contract_id": "test-hashes",
            "gates": {"A": {"runtime_sha256": "source:right/path"}},
        }
        with mock.patch.object(candidate, "load_hash_contract", return_value=source_contract):
            _, unresolved = candidate.evidence_hash_referents(documents, index)
        self.assertEqual(unresolved, ["A:runtime_sha256"])

    def test_unresolved_hash_makes_candidate_red(self) -> None:
        self.initialize()
        evidence = {
            "gate": "A",
            "ts_utc": "2026-07-27T00:00:00Z",
            "commit": SOURCE["commit"],
            "host": "test",
            "platform": "test",
            "hashes": {"missing_sha256": "f" * 64},
        }
        for gate in ("A", "B"):
            candidate.begin_invocation(self.candidate_root, gate)
            value = dict(evidence)
            value["gate"] = gate
            candidate.atomic_json(self.candidate_root / f"evidence/{gate}.json", value)
            log = self.candidate_root / f"logs/{gate}.log"
            log.write_text(f"{gate} passed\n")
            candidate.record_outcome(self.candidate_root, gate, "pass", 0, log)
        manifest = candidate.finalize_candidate(self.candidate_root)
        self.assertEqual(manifest["state"], "complete_red")
        self.assertTrue(any("unresolved hash" in problem for problem in manifest["problems"]))

    def test_partial_outcome_cover_cannot_finalize(self) -> None:
        self.initialize()
        candidate.begin_invocation(self.candidate_root, "A")
        log = self.candidate_root / "logs/A.log"
        log.write_text("A failed\n")
        candidate.record_outcome(self.candidate_root, "A", "red", 1, log)
        with self.assertRaises(candidate.CandidateError):
            candidate.finalize_candidate(self.candidate_root)

    def test_dependency_failure_is_recorded_as_blocked(self) -> None:
        self.initialize()
        candidate.begin_invocation(self.candidate_root, "PREFLIGHT")
        candidate.capture_produced_artifacts(self.candidate_root, "PREFLIGHT")
        candidate.begin_invocation(self.candidate_root, "A")
        log_a = self.candidate_root / "logs/A.log"
        log_a.write_text("A failed\n")
        candidate.record_outcome(self.candidate_root, "A", "red", 1, log_a)
        log_b = self.candidate_root / "logs/B.log"
        log_b.write_text("B blocked by A\n")
        candidate.record_outcome(
            self.candidate_root, "B", "blocked", 125, log_b, ["A:red"]
        )
        manifest = candidate.finalize_candidate(self.candidate_root)
        self.assertEqual(manifest["state"], "complete_red")
        self.assertEqual(manifest["blocked_gates"], ["B"])

    def test_missing_performance_prerequisite_prevents_green(self) -> None:
        with mock.patch.object(
            candidate,
            "performance_prerequisite",
            return_value=(None, ["performance prerequisite missing"]),
        ):
            manifest = self.collect_green()
        self.assertEqual(manifest["state"], "complete_red")
        self.assertFalse(manifest["promotion_ready"])

    def test_uncalibrated_decision_rule_rejects_certificate(self) -> None:
        self.initialize()
        candidate_document = candidate.load_candidate(self.candidate_root)
        _, problems = REAL_PERFORMANCE_PREREQUISITE(
            self.candidate_root, candidate_document
        )
        self.assertTrue(any("calibrated and reviewed" in problem for problem in problems))

    def test_unsafe_candidate_path_is_rejected(self) -> None:
        with self.assertRaises(candidate.CandidateError):
            candidate.candidate_root("/private/tmp/../tgrad-evidence")

    def test_red_candidate_cannot_promote(self) -> None:
        self.initialize()
        for gate in ("A", "B"):
            candidate.begin_invocation(self.candidate_root, gate)
            log = self.candidate_root / f"logs/{gate}.log"
            log.write_text(f"{gate} red\n")
            candidate.record_outcome(self.candidate_root, gate, "red", 1, log)
        candidate.finalize_candidate(self.candidate_root)
        with self.assertRaises(candidate.CandidateError):
            candidate.promote_candidate(
                self.candidate_root,
                ROOT / "fixtures/gate_evidence",
                self.base / "backup",
            )

    def test_post_swap_audit_failure_restores_canonical_snapshot(self) -> None:
        self.collect_green()
        fake_repo = self.base / "fake-repo"
        destination = fake_repo / "fixtures/gate_evidence"
        destination.mkdir(parents=True)
        (destination / "old.txt").write_text("old snapshot\n")
        backup = self.base / "promotion-backup"
        with (
            mock.patch.object(candidate, "ROOT", fake_repo),
            mock.patch.object(
                candidate,
                "strict_audit",
                side_effect=[None, candidate.CandidateError("post-swap rejection")],
            ),
        ):
            with self.assertRaises(candidate.CandidateError):
                candidate.promote_candidate(
                    self.candidate_root, destination, backup
                )
        self.assertEqual((destination / "old.txt").read_text(), "old snapshot\n")
        self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for the parity observer's evidence boundary."""
from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

from scripts.parity import run_upstream_suite as observer


def report_events() -> list[dict]:
    descriptor = {
        "nodeid": "x.py::test_x", "parameters": {},
        "fixtures": [], "marks": [],
    }
    return [
        {"event": "collection_finish", "count": 1,
         "nodeids": ["x.py::test_x"], "cases": [descriptor]},
        {"event": "test_report", "nodeid": "x.py::test_x", "phase": "setup", "outcome": "passed"},
        {"event": "test_report", "nodeid": "x.py::test_x", "phase": "call", "outcome": "passed",
         "subtest": {"msg": None, "kwargs": {"i": "0"}}},
        {"event": "test_report", "nodeid": "x.py::test_x", "phase": "call", "outcome": "passed",
         "subtest": {"msg": None, "kwargs": {"i": "1"}}},
        {"event": "test_report", "nodeid": "x.py::test_x", "phase": "call", "outcome": "passed"},
        {"event": "test_report", "nodeid": "x.py::test_x", "phase": "teardown", "outcome": "passed"},
        {"event": "session_finish", "exitstatus": 0},
    ]


def store_artifact(root: Path, kind: str, raw: bytes) -> dict:
    content_hash = observer.digest(raw)
    relative = Path("artifacts") / kind / content_hash
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"sha256": content_hash, "bytes": len(raw), "path": relative.as_posix()}


def valid_baseline(root: Path) -> dict:
    files = [
        {"path": f"test/unit/test_{index}.py", "group": "unit",
         "source_sha256": f"source-{index}"}
        for index in range(34)
    ]
    scenario_without_hash = {
        "group": "all", "oracle_class": "api_surface", "files": files,
        "timeout_seconds": 600,
        "pytest_flags": ["-q"],
        "backend_profile": {
            "upstream": "METAL", "tgrad": "METAL",
            "relation": "same_backend_public_observations",
        },
    }
    scenario = {
        **scenario_without_hash,
        "sha256": observer.digest(observer.canonical(scenario_without_hash)),
    }
    identity = {
        "upstream": {
            "revision": "upstream", "tree": "upstream-tree",
            "snapshot_content_sha256": "snapshot",
        },
        "subject": {
            "kind": "upstream", "revision": "upstream",
            "tree": "upstream-tree", "dirty": False,
        },
        "verifier": {"runner_sha256": "runner", "reporter_sha256": "reporter"},
        "acceptance": {"allowed_file_outcomes": ["passed"]},
        "contract": {"sha256": "contract", "file_count": 34},
        "environment": {
            "facts": {"python": "test"},
            "backend": {"default_device": "METAL", "hardware": {"chip": "test"}},
            "prerequisites": {"posix_shared_memory": {"available": True}},
            "oracle_backend_readiness": {"available": True},
            "policy": {
                "LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
                "isolated_home": True, "isolated_tmp": True,
                "PATH": "/bin", "path_sha256": "path",
                "inherited_backend_overrides": [],
            },
        },
        "scenario": scenario,
    }
    identity["environment"]["sha256"] = observer.digest(observer.canonical(
        identity["environment"]
    ))
    empty_ref = store_artifact(root, "stdout", b"")
    stderr_ref = store_artifact(root, "stderr", b"")
    cells = []
    for item in files:
        nodeid = item["path"] + "::test_case"
        events = [
            {"event": "collection_finish", "count": 1, "nodeids": [nodeid],
             "cases": [{"nodeid": nodeid, "parameters": {},
                        "fixtures": [], "marks": []}]},
            {"event": "test_report", "nodeid": nodeid, "phase": "setup", "outcome": "passed"},
            {"event": "test_report", "nodeid": nodeid, "phase": "call", "outcome": "passed"},
            {"event": "test_report", "nodeid": nodeid, "phase": "teardown", "outcome": "passed"},
            {"event": "session_finish", "exitstatus": 0},
        ]
        report_raw = "".join(
            json.dumps(event, sort_keys=True) + "\n" for event in events
        ).encode()
        report_ref = store_artifact(root, "pytest-report", report_raw)
        counts, report = observer.analyze_report(events, 0)
        cells.append({
            "file": item["path"], "process": "exited", "returncode": 0,
            "phase": "execution", "outcome": "passed", "reason_codes": [],
            "counts": counts,
            "report_sha256": observer.digest(report_raw),
            "report_artifact": report_ref,
            "nodeid_manifest_sha256": report["nodeid_manifest_sha256"],
            "nodeid_count": 1,
            "collection_case_manifest_sha256": report[
                "collection_case_manifest_sha256"
            ],
            "collection_cases": report["collection_cases"],
            "case_manifest_sha256": report["case_manifest_sha256"],
            "case_count": 1,
            "case_results": report["case_results"],
            "case_outcome_manifest_sha256": report[
                "case_outcome_manifest_sha256"
            ],
            "source_sha256": item["source_sha256"],
            "diagnostics": {
                "stdout_bytes": 0,
                "stdout_raw_sha256": observer.digest(b""),
                "stdout_normalized_sha256": observer.digest(b""),
                "stderr_bytes": 0,
                "stderr_raw_sha256": observer.digest(b""),
                "stderr_normalized_sha256": observer.digest(b""),
                "normalized_excerpt": "",
                "raw_artifacts": {"stdout": empty_ref, "stderr": stderr_ref},
            },
        })
    document = {
        "schema_version": observer.SCHEMA_VERSION,
        "against": "upstream", "identity": identity,
        "observation": {"aggregate": observer.aggregate(cells), "cells": cells},
        "scenario_id": scenario["sha256"],
    }
    document["result_id"] = observer.computed_result_id(document)
    document["run_artifact_id"] = observer.computed_run_artifact_id(document)
    return document


def tgrad_identity_for(baseline: dict) -> dict:
    identity = copy.deepcopy(baseline["identity"])
    identity["subject"] = {
        "kind": "tgrad", "revision": "tgrad", "tree": "tgrad-tree",
        "dirty": False,
    }
    identity["environment"]["backend"]["default_device"] = "METAL"
    identity["environment"].pop("sha256", None)
    identity["environment"]["sha256"] = observer.digest(observer.canonical(
        identity["environment"]
    ))
    return identity


def valid_tgrad(root: Path, upstream_path: Path, baseline: dict) -> dict:
    identity = tgrad_identity_for(baseline)
    reference, baseline_cells = observer.validate_upstream_baseline(
        upstream_path, identity, observer.REPO
    )
    identity["upstream_baseline"] = reference
    cells = copy.deepcopy(baseline["observation"]["cells"])
    for cell in cells:
        observer.apply_upstream_oracle(cell, baseline_cells[cell["file"]])
    document = {
        "schema_version": observer.SCHEMA_VERSION,
        "against": "tgrad", "identity": identity,
        "observation": {
            "aggregate": observer.aggregate(cells), "cells": cells,
            "oracle_cases": observer.oracle_case_summary(cells, reference),
        },
        "scenario_id": identity["scenario"]["sha256"],
    }
    document["result_id"] = observer.computed_result_id(document)
    document["run_artifact_id"] = observer.computed_run_artifact_id(document)
    return document


class ProtocolTests(unittest.TestCase):
    def test_snapshot_diagnostics_are_independent_of_observer_root(self) -> None:
        first_root = "/tmp/tgrad_parity_observer_first/snapshot"
        second_root = "/private/tmp/tgrad_parity_observer_second/snapshot"
        suffixes = (
            ("upstream", "test/unit/test_tensor.py"),
            ("shim", "tinygrad/__init__.py"),
            ("python", "tgrad.py"),
            ("runtime", "libtgrad.dylib"),
        )
        first = "\n".join(
            f"{first_root}/{tree}/{path}: diagnostic" for tree, path in suffixes
        )
        second = "\n".join(
            f"{second_root}/{tree}/{path}: diagnostic" for tree, path in suffixes
        )

        self.assertEqual(
            observer.normalize_output(first, Path("/unrelated/upstream")),
            observer.normalize_output(second, Path("/another/upstream")),
        )

    def test_subtests_are_distinct_protocol_events(self) -> None:
        counts, report = observer.analyze_report(report_events(), 0)
        self.assertEqual([], report["protocol_errors"])
        self.assertEqual(3, counts["passed"])
        self.assertEqual(1, counts["collected"])

    def test_partial_report_is_verifier_error(self) -> None:
        events = report_events()[:-1]
        counts, report = observer.analyze_report(events, 0)
        self.assertIn("session_finish_count", report["protocol_errors"])
        self.assertEqual(
            "verifier_error",
            observer.classify_result(
                0, "", events, counts, report, "x.py", {}
            )[1],
        )

    def test_environment_failure_is_not_product_nonconformance(self) -> None:
        events = report_events()
        events[2] = {
            **events[2], "outcome": "failed", "failure_type": "PermissionError"
        }
        # pytest's terminal event carries the same non-zero status.
        events[-1]["exitstatus"] = 1
        counts, report = observer.analyze_report(events, 1)
        phase, outcome, reasons = observer.classify_result(
            1, "PermissionError", events, counts, report,
            "test/unit/test_shm_tensor.py",
            {"posix_shared_memory": {"available": False}},
        )
        self.assertEqual("environment", phase)
        self.assertEqual("blocked_environment", outcome)
        self.assertIn("posix_shared_memory_unavailable", reasons)


class BaselineTests(unittest.TestCase):
    def write(self, document: dict, root: Path) -> Path:
        path = root / "baseline.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_valid_baseline_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = valid_baseline(root)
            reference, cells = observer.validate_upstream_baseline(
                self.write(baseline, root), tgrad_identity_for(baseline),
                observer.REPO,
            )
        self.assertEqual(34, len(cells))
        self.assertEqual(baseline["result_id"], reference["result_id"])

    def test_empty_baseline_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = valid_baseline(root)
            baseline["observation"] = {
                "aggregate": observer.aggregate([]), "cells": []
            }
            baseline["result_id"] = observer.computed_result_id(baseline)
            baseline["run_artifact_id"] = observer.computed_run_artifact_id(baseline)
            with self.assertRaisesRegex(RuntimeError, "exactly 34 cells"):
                observer.validate_upstream_baseline(
                    self.write(baseline, root), tgrad_identity_for(baseline),
                    observer.REPO,
                )

    def test_missing_report_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = valid_baseline(root)
            report = root / baseline["observation"]["cells"][0]["report_artifact"]["path"]
            report.unlink()
            with self.assertRaisesRegex(RuntimeError, "bound artifact is missing"):
                observer.validate_upstream_baseline(
                    self.write(baseline, root), tgrad_identity_for(baseline),
                    observer.REPO,
                )

    def test_authored_green_cell_is_rejected_by_event_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = valid_baseline(root)
            baseline["observation"]["cells"][0]["counts"]["passed"] = 99
            baseline["observation"]["aggregate"] = observer.aggregate(
                baseline["observation"]["cells"]
            )
            baseline["result_id"] = observer.computed_result_id(baseline)
            baseline["run_artifact_id"] = observer.computed_run_artifact_id(baseline)
            with self.assertRaisesRegex(RuntimeError, "cannot be replayed"):
                observer.validate_upstream_baseline(
                    self.write(baseline, root), tgrad_identity_for(baseline),
                    observer.REPO,
                )

    def test_result_id_binds_verifier_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            baseline = valid_baseline(Path(raw))
            original = observer.computed_result_id(baseline)
            baseline["identity"]["verifier"]["runner_sha256"] = "different"
            self.assertNotEqual(original, observer.computed_result_id(baseline))
            baseline = valid_baseline(Path(raw))
            baseline["identity"]["environment"]["facts"]["python"] = "different"
            self.assertNotEqual(original, observer.computed_result_id(baseline))
            baseline = valid_baseline(Path(raw))
            baseline["observation"]["cells"][0]["diagnostics"][
                "stdout_normalized_sha256"
            ] = "different"
            self.assertNotEqual(original, observer.computed_result_id(baseline))

    def test_tgrad_cells_are_replayed_before_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = valid_baseline(root)
            upstream_path = self.write(baseline, root)
            tgrad = valid_tgrad(root, upstream_path, baseline)
            tgrad_path = root / "tgrad.json"
            tgrad_path.write_text(json.dumps(tgrad), encoding="utf-8")
            observer.validate_tgrad_observation(
                tgrad_path, tgrad, upstream_path, observer.REPO
            )
            tgrad["observation"]["cells"][0]["counts"]["passed"] = 99
            tgrad["observation"]["aggregate"] = observer.aggregate(
                tgrad["observation"]["cells"]
            )
            tgrad["result_id"] = observer.computed_result_id(tgrad)
            tgrad["run_artifact_id"] = observer.computed_run_artifact_id(tgrad)
            tgrad_path.write_text(json.dumps(tgrad), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot be replayed"):
                observer.validate_tgrad_observation(
                    tgrad_path, tgrad, upstream_path, observer.REPO
                )


if __name__ == "__main__":
    unittest.main()

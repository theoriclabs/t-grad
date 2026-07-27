#!/usr/bin/env python3
"""Offline falsifiers for upstream calibration provenance and repeatability."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity import calibrate_upstream_suite as calibration  # noqa: E402
from scripts.parity.coverage_model import attach_content_sha256, canonical_sha256, file_sha256  # noqa: E402


def write_run(
    root: Path,
    *,
    status: str = "pass",
    complete: bool = True,
    raw: bytes = b"1 passed in 0.01s\n",
) -> dict:
    for directory in ("stdout", "stderr", "junit"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    stdout = root / "stdout" / "001-test_one.txt"
    stderr = root / "stderr" / "001-test_one.txt"
    junit = root / "junit" / "001-test_one.xml"
    stdout.write_bytes(raw)
    stderr.write_bytes(b"")
    returncode = 0 if status in ("pass", "pass_with_skips") else 1
    case_status = "pass" if status == "pass" else ("skipped" if status == "pass_with_skips" else "fail")
    counts = {
        "collected": 1,
        "failed": 1 if case_status == "fail" else 0,
        "errors": 0,
        "skipped": 1 if case_status == "skipped" else 0,
        "passed": 1 if case_status == "pass" else 0,
    }
    cases = [{"node": "test_one.Test::test_ok", "status": case_status}]
    junit.write_text(
        f'<testsuite tests="1" failures="{counts["failed"]}" errors="0" skipped="{counts["skipped"]}">'
        '<testcase classname="test_one.Test" name="test_ok">'
        + ("" if case_status == "pass" else ('<skipped/>' if case_status == "skipped" else '<failure message="expected"/>'))
        + "</testcase></testsuite>"
    )
    semantic = {
        "file": "test/null/test_one.py",
        "status": status,
        "returncode": returncode,
        "counts": counts,
        "cases": cases,
    }
    result = {
        **semantic,
        "semantic_sha256": canonical_sha256(semantic),
        "duration_ms": 10.0,
        "command": ["python", "-m", "pytest"],
        "junit_error": "",
        "stdout": {"path": "stdout/001-test_one.txt", "sha256": file_sha256(stdout), "bytes": stdout.stat().st_size},
        "stderr": {"path": "stderr/001-test_one.txt", "sha256": file_sha256(stderr), "bytes": stderr.stat().st_size},
        "junit": {"path": "junit/001-test_one.xml", "sha256": file_sha256(junit), "bytes": junit.stat().st_size},
    }
    statuses = {name: 0 for name in sorted(calibration.FILE_STATUSES)}
    statuses[status] = 1
    diagnostics = [{
        "file": result["file"],
        "stdout_sha256": result["stdout"]["sha256"],
        "stderr_sha256": result["stderr"]["sha256"],
        "junit_sha256": result["junit"]["sha256"],
    }]
    document = attach_content_sha256(
        {
            "schema_version": calibration.RUN_SCHEMA_VERSION,
            "kind": "tgrad-upstream-suite-calibration-run",
            "target": {"manifest": "target.json", "manifest_content_sha256": "a" * 64, "upstream_ref": "1" * 40, "checkout_tree": "2" * 40},
            "environment": {"manifest": "environment.json", "content_sha256": "b" * 64},
            "selection": {"group": "null", "inventory_count": 1, "selected_count": 1, "complete": complete, "inventory_sha256": "c" * 64, "selected_sha256": "c" * 64},
            "scenario": {"timeout_seconds": 120, "pytest_arguments": ["pinned"], "selectors": calibration.PINNED_ENVIRONMENT},
            "status_counts": statuses,
            "outcome_sha256": canonical_sha256([semantic]),
            "diagnostic_manifest_sha256": canonical_sha256(diagnostics),
            "results": [result],
            "promotion_ready": False,
            "promotion_blockers": ["comparison required"],
        }
    )
    (root / "run.json").write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return document


class CalibrationTests(unittest.TestCase):
    def test_raw_timing_noise_does_not_change_semantic_repeatability(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-calibration-test-") as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            write_run(first, raw=b"1 passed in 0.01s\n")
            write_run(second, raw=b"1 passed in 0.99s\n")
            comparison = calibration.compare_runs(first, second)
            self.assertTrue(comparison["identity_equal"])
            self.assertTrue(comparison["outcomes_equal"])
            self.assertTrue(comparison["promotion_ready"])
            self.assertEqual(comparison["transitions"], [])

    def test_partial_run_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-calibration-test-") as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            write_run(first, complete=False)
            write_run(second, complete=False)
            comparison = calibration.compare_runs(first, second)
            self.assertFalse(comparison["complete"])
            self.assertFalse(comparison["promotion_ready"])
            self.assertIn("one or both runs are partial", comparison["promotion_blockers"])

    def test_nonpass_and_transition_remain_diagnosable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-calibration-test-") as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            write_run(first)
            write_run(second, status="fail")
            comparison = calibration.compare_runs(first, second)
            self.assertFalse(comparison["outcomes_equal"])
            self.assertFalse(comparison["promotion_ready"])
            self.assertEqual(comparison["transitions"][0]["first"], "pass")
            self.assertEqual(comparison["transitions"][0]["second"], "fail")

    def test_tampered_raw_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-calibration-test-") as directory:
            run = Path(directory) / "run"
            write_run(run)
            (run / "stdout" / "001-test_one.txt").write_text("tampered\n")
            with self.assertRaisesRegex(calibration.CalibrationError, "hash mismatch"):
                calibration.load_run(run)

    def test_tampered_semantic_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-calibration-test-") as directory:
            run = Path(directory) / "run"
            document = write_run(run)
            document["results"][0]["counts"]["passed"] = 99
            document.pop("content_sha256")
            document.update(attach_content_sha256(document))
            (run / "run.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(calibration.CalibrationError, "semantic hash mismatch"):
                calibration.load_run(run)

    def test_dirty_checkout_is_rejected_before_execution(self) -> None:
        responses = iter(["1" * 40, "2" * 40, " M tinygrad/tensor.py"])
        with mock.patch.object(calibration, "_git", side_effect=lambda *_: next(responses)):
            with self.assertRaisesRegex(calibration.CalibrationError, "must be clean"):
                calibration.checkout_identity(Path("/upstream"), "1" * 40)

    def test_environment_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-calibration-test-") as directory:
            path = Path(directory) / "environment.json"
            recorded = attach_content_sha256(
                {
                    "schema_version": calibration.ENVIRONMENT_SCHEMA_VERSION,
                    "kind": "tgrad-upstream-calibration-environment",
                    "python": {"executable": "/python"},
                    "host": {},
                    "selectors": calibration.PINNED_ENVIRONMENT,
                }
            )
            path.write_text(json.dumps(recorded))
            current = copy.deepcopy(recorded)
            current["host"] = {"release": "different"}
            current.pop("content_sha256")
            current.update(attach_content_sha256(current))
            with mock.patch.object(calibration, "build_environment", return_value=current):
                with self.assertRaisesRegex(calibration.CalibrationError, "differ"):
                    calibration.load_environment(path)


if __name__ == "__main__":
    unittest.main()

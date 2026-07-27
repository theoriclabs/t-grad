#!/usr/bin/env python3
"""CPU-only focused tests for scripts/perf/paired_runtime.py."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from unittest import mock
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "perf" / "paired_runtime.py"
SPEC = importlib.util.spec_from_file_location("tgrad_paired_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
paired = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paired
SPEC.loader.exec_module(paired)


class FakeSession:
    def __init__(
        self, adapter: "FakeAdapter", output: bytes, base_duration_ns: int,
        fail_measure_at: int | None,
    ):
        self.adapter = adapter
        self.output = output
        self.base_duration_ns = base_duration_ns
        self.fail_measure_at = fail_measure_at

    def boundary_specs(self) -> Mapping[str, Any]:
        return {
            "repeated": paired.BoundarySpec(
                id="repeated",
                category="fake_runtime",
                description="deterministic CPU-only fake boundary",
                includes=("scripted duration",),
                excludes=("wall clock", "GPU"),
            ),
            "unavailable_compile": paired.BoundarySpec(
                id="unavailable_compile",
                category="compile_capture",
                description="fake has no compile boundary",
                includes=(),
                excludes=(),
                available=False,
                unavailable_reason="not represented by the fake adapter",
            ),
        }

    def correctness_output(self) -> bytes:
        self.adapter.correctness_calls += 1
        return self.output

    def prepare(self) -> Sequence[Any]:
        self.adapter.prepare_calls += 1
        return (
            paired.PhaseObservation(
                boundary_id="repeated",
                label="fake preparation",
                measurement=paired.Measurement(
                    self.base_duration_ns // 2,
                    {"fake_preparation": True},
                ),
            ),
        )

    def measure(self, boundary_id: str) -> Any:
        if boundary_id != "repeated":
            raise AssertionError(f"unexpected fake boundary: {boundary_id}")
        self.adapter.measure_calls += 1
        if self.fail_measure_at == self.adapter.measure_calls:
            raise RuntimeError(f"scripted failure from {self.adapter.name}")
        return paired.Measurement(
            self.base_duration_ns + 10 * self.adapter.measure_calls,
            {"fake_call_index": self.adapter.measure_calls},
        )

    def close(self) -> None:
        self.adapter.close_calls += 1


class FakeAdapter:
    def __init__(
        self, name: str, output: bytes, base_duration_ns: int,
        fail_measure_at: int | None = None,
    ):
        self.name = name
        self.output = output
        self.base_duration_ns = base_duration_ns
        self.fail_measure_at = fail_measure_at
        self.correctness_calls = 0
        self.prepare_calls = 0
        self.measure_calls = 0
        self.close_calls = 0

    def provenance(self) -> Any:
        return paired.AdapterProvenance(
            name=self.name,
            implementation=f"CPU fake {self.name}",
            source=paired.GitState(
                path=f"/fake/{self.name}",
                commit=f"{self.name}-commit",
                tree=f"{self.name}-tree",
                dirty=False,
                status_digest="0" * 64,
            ),
            device="CPU_FAKE",
            environment={"clock": "scripted"},
            revision_validation="test_fixture",
        )

    def create_session(
        self, workload: Any, a_payload: bytes, b_payload: bytes
    ) -> FakeSession:
        self.assert_payloads(workload, a_payload, b_payload)
        return FakeSession(
            self, self.output, self.base_duration_ns, self.fail_measure_at
        )

    @staticmethod
    def assert_payloads(workload: Any, a_payload: bytes, b_payload: bytes) -> None:
        assert len(a_payload) == workload.m * workload.k * 2
        assert len(b_payload) == workload.k * workload.n * 2


def comparison() -> Any:
    return paired.Comparison(
        id="fake_tgrad_vs_tinygrad",
        numerator_adapter="tgrad",
        numerator_boundary="repeated",
        denominator_adapter="tinygrad",
        denominator_boundary="repeated",
        interpretation="CPU-only harness calibration",
    )


def forbidden_summary_keys(value: Any) -> set[str]:
    forbidden = {"verdict", "pass", "passed", "parity"}
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                found.add(key)
            found |= forbidden_summary_keys(child)
    elif isinstance(value, list):
        for child in value:
            found |= forbidden_summary_keys(child)
    return found


class PairedRuntimeTests(unittest.TestCase):
    def make_config(self, directory: Path) -> Any:
        return paired.HarnessConfig(
            raw_output=directory / "raw.jsonl",
            summary_output=directory / "summary.json",
            sessions=2,
            samples_per_session=6,
            warmup_pairs=2,
            order_seed=111,
            analysis_seed=222,
            bootstrap_resamples=100,
            confidence_level=0.90,
            run_instance_id="fake-calibration-run",
            captured_at_utc="2026-07-27T00:00:00+00:00",
        )

    @staticmethod
    def make_adapters(
        tinygrad_output: bytes = b"\x00" * 8,
        fail_measure_at: int | None = None,
    ) -> dict[str, FakeAdapter]:
        return {
            "tgrad": FakeAdapter("tgrad", b"\x00" * 8, 1_000),
            "tinygrad": FakeAdapter(
                "tinygrad", tinygrad_output, 2_000, fail_measure_at=fail_measure_at
            ),
        }

    def test_balanced_orders_exact_count_statistics_and_no_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.make_config(directory)
            summary = paired.run_harness(
                config, paired.Workload(m=2, k=2, n=2),
                self.make_adapters(), [comparison()],
            )
            raw = [json.loads(line) for line in config.raw_output.read_text().splitlines()]
            measured = [row for row in raw if row["phase"] == "measured"]
            orders = {row["order"] for row in measured}
            self.assertEqual(orders, {"AB", "BA"})
            for session_index in range(config.sessions):
                first_sides = [
                    row["order"] for row in measured
                    if row["session_index"] == session_index and row["order_position"] == 0
                ]
                self.assertEqual(Counter(first_sides), Counter({"AB": 3, "BA": 3}))
            expected_raw_count = config.sessions * (
                len(self.make_adapters())
                + (config.warmup_pairs + config.samples_per_session) * 2
            )
            self.assertEqual(len(raw), expected_raw_count)
            self.assertEqual(summary["raw_observation_count"], expected_raw_count)
            self.assertEqual(
                summary["complete_pair_count"],
                config.sessions * config.samples_per_session,
            )
            analysis = summary["analysis"][comparison().id]
            self.assertEqual(analysis["count"], config.sessions * config.samples_per_session)
            self.assertTrue(math.isfinite(analysis["mean_log_ratio"]))
            self.assertTrue(math.isfinite(analysis["geometric_mean_ratio"]))
            self.assertGreater(
                analysis["absolute"]["numerator"]["duration_ns_quantiles"]["p50"], 0
            )
            self.assertGreater(
                analysis["absolute"]["denominator"]["throughput_tflops_quantiles"]["p50"], 0
            )
            self.assertFalse(summary["comparisons"][0]["kernel_speed_claim_eligible"])
            self.assertEqual(forbidden_summary_keys(summary), set())

    def test_deterministic_fake_reruns_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            config_a = self.make_config(Path(first))
            config_b = self.make_config(Path(second))
            summary_a = paired.run_harness(
                config_a, paired.Workload(m=2, k=3, n=2),
                self.make_adapters(), [comparison()],
            )
            summary_b = paired.run_harness(
                config_b, paired.Workload(m=2, k=3, n=2),
                self.make_adapters(), [comparison()],
            )
            self.assertEqual(config_a.raw_output.read_bytes(), config_b.raw_output.read_bytes())
            self.assertEqual(summary_a, summary_b)
            self.assertEqual(
                summary_a["analysis"][comparison().id]["bootstrap"],
                summary_b["analysis"][comparison().id]["bootstrap"],
            )

    def test_distinct_run_instances_cannot_alias(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            config_a = self.make_config(Path(first))
            config_b = paired.HarnessConfig(
                **{
                    **config_a.__dict__,
                    "raw_output": Path(second) / "raw.jsonl",
                    "summary_output": Path(second) / "summary.json",
                    "run_instance_id": "second-fake-run",
                }
            )
            summary_a = paired.run_harness(
                config_a, paired.Workload(m=2, k=2, n=2),
                self.make_adapters(), [comparison()],
            )
            summary_b = paired.run_harness(
                config_b, paired.Workload(m=2, k=2, n=2),
                self.make_adapters(), [comparison()],
            )
            self.assertNotEqual(summary_a["run_id"], summary_b["run_id"])
            self.assertEqual(summary_a["run_instance"]["id"], "fake-calibration-run")
            self.assertEqual(summary_b["run_instance"]["id"], "second-fake-run")

    def test_hierarchical_bootstrap_weights_sessions_equally(self) -> None:
        class ScriptedRandom:
            def __init__(self, _seed: int):
                self.values = iter([0, 0, 1, *([10] * 9)])

            def choice(self, _values: Any) -> Any:
                return next(self.values)

        with mock.patch.object(paired.random, "Random", ScriptedRandom):
            low, high = paired._hierarchical_bootstrap(
                {0: [0.0], 1: [10.0] * 9}, 1, 0.90, 123
            )
        self.assertEqual((low, high), (5.0, 5.0))

    def test_correctness_mismatch_aborts_before_timed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.make_config(Path(temporary))
            adapters = self.make_adapters(tinygrad_output=b"\x01" * 8)
            with self.assertRaises(paired.CorrectnessError):
                paired.run_harness(
                    config, paired.Workload(m=2, k=2, n=2), adapters, [comparison()]
                )
            self.assertFalse(config.raw_output.exists())
            self.assertFalse(config.summary_output.exists())
            self.assertEqual(sum(a.prepare_calls for a in adapters.values()), 0)
            self.assertEqual(sum(a.measure_calls for a in adapters.values()), 0)

    def test_timed_failure_is_retained_without_performance_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.make_config(Path(temporary))
            adapters = self.make_adapters(fail_measure_at=1)
            with self.assertRaises(paired.MeasurementRunError) as caught:
                paired.run_harness(
                    config, paired.Workload(m=2, k=2, n=2), adapters, [comparison()]
                )
            self.assertTrue(config.raw_output.exists())
            self.assertTrue(config.summary_output.exists())
            summary = json.loads(config.summary_output.read_text())
            raw = [json.loads(line) for line in config.raw_output.read_text().splitlines()]
            self.assertEqual(summary["completion"], "measurement_error")
            self.assertEqual(len(summary["errors"]), 1)
            self.assertEqual(sum(row["record_type"] == "observation_error" for row in raw), 1)
            self.assertEqual(caught.exception.summary, summary)
            self.assertEqual(forbidden_summary_keys(summary), set())

    def test_official_pinned_source_contract(self) -> None:
        self.assertEqual(
            paired.DEFAULT_TINYGRAD_SOURCE,
            Path("/tmp/tgrad-upstream-19c4d736"),
        )
        self.assertEqual(
            paired.EXPECTED_TINYGRAD_COMMIT,
            "19c4d736f2bc8e26d21f08b28ffd6298408da00f",
        )
        self.assertEqual(
            paired.EXPECTED_TINYGRAD_TREE,
            "855cca3b00c38841a6d3a043284f3a2ca696d4b0",
        )
        self.assertEqual(
            paired.KNOWN_UPSTREAM_PYTHON,
            Path("/tmp/tgrad-upstream-py312/bin/python"),
        )
        # Requesting the interpreter already in use is a no-op, not a child process.
        paired._maybe_reexec_python(Path(sys.executable), [])
        parsed = paired._parser().parse_args([
            "--python-executable", str(paired.KNOWN_UPSTREAM_PYTHON),
            "--raw-output", "/tmp/fake-raw.jsonl",
            "--summary-output", "/tmp/fake-summary.json",
        ])
        self.assertEqual(parsed.python_executable, paired.KNOWN_UPSTREAM_PYTHON)
        state, validation, diagnostics = paired.validate_tinygrad_checkout(
            paired.DEFAULT_TINYGRAD_SOURCE
        )
        self.assertEqual(state.commit, paired.EXPECTED_TINYGRAD_COMMIT)
        self.assertEqual(state.tree, paired.EXPECTED_TINYGRAD_TREE)
        self.assertEqual(validation, "pinned_clean")
        self.assertEqual(diagnostics, ())

    def test_unknown_or_dirty_upstream_requires_diagnostic_override(self) -> None:
        unknown = paired.GitState(
            path="/fake/upstream",
            commit="unknown",
            tree="unknown",
            dirty=True,
            status_digest="f" * 64,
        )
        with mock.patch.object(paired, "inspect_git_checkout", return_value=unknown):
            with self.assertRaises(paired.RevisionError):
                paired.validate_tinygrad_checkout(Path("/fake/upstream"))
            state, validation, diagnostics = paired.validate_tinygrad_checkout(
                Path("/fake/upstream"), allow_unknown_revision=True
            )
        self.assertEqual(state, unknown)
        self.assertEqual(validation, "diagnostic_override")
        self.assertGreaterEqual(len(diagnostics), 3)

    def test_dirty_tgrad_requires_diagnostic_override(self) -> None:
        dirty = paired.GitState(
            path="/fake/tgrad", commit="current", tree="tree", dirty=True,
            status_digest="e" * 64,
        )
        with mock.patch.object(paired, "inspect_git_checkout", return_value=dirty):
            with self.assertRaises(paired.RevisionError):
                paired.validate_tgrad_checkout(Path("/fake/tgrad"))
            state, validation, diagnostics = paired.validate_tgrad_checkout(
                Path("/fake/tgrad"), allow_dirty=True
            )
        self.assertEqual(state, dirty)
        self.assertEqual(validation, "diagnostic_override")
        self.assertTrue(diagnostics)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""CPU-only adversarial tests for the V2 broadcast-add observer.

No test imports tinygrad/Tgrad, loads a dylib, or invokes the child runner.
Synthetic child protocols test the parent relation; compilation and AST/source
checks test that all eight immutable execution branches exist in the one probe.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.spec import observe_broadcast_add as observer


def definition() -> tuple[dict, dict, dict]:
    return observer.validate_v2_definition()


def ownership(mode: str) -> dict:
    package_owner = "<upstream>" if mode == "upstream" else "<shim>"
    tensor_owner = "<upstream>" if mode == "upstream" else "<product>"
    return {
        "package": {"kind": "file", "owner": package_owner,
                    "path": f"{package_owner}/tinygrad/__init__.py"},
        "package_paths": [f"{package_owner}/tinygrad"],
        "tensor_module": {"kind": "file", "owner": tensor_owner,
                          "path": f"{tensor_owner}/tgrad.py"},
        "loaded_tinygrad_modules": [
            {"module": "tinygrad", "kind": "file", "owner": package_owner,
             "path": f"{package_owner}/tinygrad/__init__.py"},
        ],
        "contaminated_modules": [], "fallback_detected": False,
        "strict_finder_active": mode == "tgrad",
        "product_module": (
            {"kind": "file", "owner": "<product>",
             "path": "<product>/tgrad.py"}
            if mode == "tgrad" else None
        ),
        "tensor_class_identity": True if mode == "tgrad" else None,
    }


def value_payload(label: str, shape: list[int], dtype: str) -> dict:
    payload = {
        "shape": shape, "dtype": dtype,
        "storage_dtype": "<f4" if dtype == "float32" else "<i4",
        "storage_hex": label.encode().hex(), "values": [label],
    }
    payload["sha256"] = observer.digest(observer.canonical(payload))
    return payload


def legal_dimensions(scenario: dict) -> dict:
    dtype = (
        "float32" if scenario["id"] == "ADD-I32-F32-SCALAR-PROMOTION"
        else scenario["left"]["dtype"]
    )
    shape = list(scenario["left"]["shape"])
    value = value_payload("baseline-" + scenario["id"], shape, dtype)
    return {
        "construction": observer.observed({"left": True, "right": True}),
        "shape": observer.observed(shape), "dtype": observer.observed(dtype),
        "value": observer.observed(value),
        "realize_identity": observer.observed({
            "first_is_result": True, "second_is_result": True,
            "same_realized_object": True,
        }),
        "repeated_readback": observer.observed({
            "equal": True, "first_sha256": value["sha256"],
            "second_sha256": value["sha256"],
        }),
        "inputs_unchanged": observer.observed({
            "equal": True, "before_sha256": "inputs",
            "after_sha256": "inputs",
        }),
        "exception_stage": observer.observed(None),
        "exception_class": observer.observed(None),
        "exception_message": observer.observed(None),
        "terminal_outcome": observer.observed("returned"),
    }


def baseline_scenarios(manifest: dict) -> list[dict]:
    results = []
    for scenario in manifest["scenarios"]:
        if scenario["id"] == "ADD-INCOMPATIBLE-SHAPES":
            trace = [
                {"stage": "construct_left", "status": "completed"},
                {"stage": "construct_right", "status": "completed"},
                {"stage": "invoke_add", "status": "raised",
                 "exception_class": "ValueError",
                 "exception_message": "incompatible shapes"},
                {"stage": "observe_shape_if_constructed", "status": "skipped",
                 "reason": "prior_stage_raised"},
                {"stage": "record_terminal_stage_class_message",
                 "status": "completed"},
            ]
            dimensions = {
                "construction": observer.observed({"left": True, "right": True}),
                "shape": observer.unobserved("result_shape_not_available"),
                "exception_stage": observer.observed("invoke_add"),
                "exception_class": observer.observed("ValueError"),
                "exception_message": observer.observed("incompatible shapes"),
                "terminal_outcome": observer.observed("raised"),
            }
            terminal = {"outcome": "raised", "exception": {
                "stage": "invoke_add", "class": "ValueError",
                "message": "incompatible shapes",
            }}
        else:
            trace = [{"stage": stage, "status": "completed"}
                     for stage in manifest["profile"]["legal_trace"]]
            dimensions = legal_dimensions(scenario)
            terminal = {"outcome": "returned", "exception": None}
        results.append({
            "id": scenario["id"], "requirements": scenario["requirements"],
            "trace": trace, "dimensions": dimensions, "terminal": terminal,
        })
    return results


def protocol(manifest: dict, mode: str = "upstream",
             mutation: dict | None = None) -> dict:
    return {
        "schema_version": observer.PROBE_SCHEMA_VERSION,
        "protocol_token": observer.protocol_token(manifest, mutation),
        "subject_mode": mode, "status": "complete", "backend": "METAL",
        "scenario_manifest_sha256": manifest["effective_manifest_sha256"],
        "mutation": mutation, "ownership": ownership(mode),
        "scenarios": baseline_scenarios(manifest),
    }


def set_dimension(result: dict, name: str, value: dict) -> None:
    result["dimensions"][name] = value


def child_protocol(manifest: dict, declaration: dict, config: dict) -> dict:
    child = protocol(manifest, "upstream", config)
    result = next(item for item in child["scenarios"]
                  if item["id"] == declaration["target_scenarios"][0])
    mutation_id = declaration["id"]
    if mutation_id == "MUT-ADD-SUBTRACT":
        changed = value_payload("subtraction", result["dimensions"]["shape"]["value"],
                                result["dimensions"]["dtype"]["value"])
        set_dimension(result, "value", observer.observed(changed))
        set_dimension(result, "repeated_readback", observer.observed({
            "equal": True, "first_sha256": changed["sha256"],
            "second_sha256": changed["sha256"],
        }))
    elif mutation_id in {
        "MUT-ADD-WRONG-RIGHT-ALIGNMENT",
        "MUT-ADD-NO-SINGLETON-EXPANSION",
    }:
        failure = "wrong alignment" if "ALIGNMENT" in mutation_id else "singleton rejected"
        for name in (
            "shape", "dtype", "value", "realize_identity",
            "repeated_readback", "inputs_unchanged",
        ):
            set_dimension(result, name, observer.unobserved("prior_stage_raised"))
        set_dimension(result, "exception_stage", observer.observed("invoke_add"))
        set_dimension(result, "exception_class", observer.observed("ValueError"))
        set_dimension(result, "exception_message", observer.observed(failure))
        set_dimension(result, "terminal_outcome", observer.observed("raised"))
        for event in result["trace"]:
            if event["stage"] == "invoke_add":
                event.clear()
                event.update({"stage": "invoke_add", "status": "raised",
                              "exception_class": "ValueError",
                              "exception_message": failure})
            elif event["stage"] in observer.MUTATION_TRACE_FOOTPRINTS[mutation_id]:
                event.clear()
                event.update({"stage": event.get("stage", ""),
                              "status": "skipped", "reason": "prior_stage_raised"})
        # event.clear above loses the stage; restore from the frozen order.
        for event, stage in zip(result["trace"], manifest["profile"]["legal_trace"]):
            event["stage"] = stage
        result["terminal"] = {"outcome": "raised", "exception": {
            "stage": "invoke_add", "class": "ValueError", "message": failure,
        }}
    elif mutation_id == "MUT-ADD-WRONG-PROMOTION":
        changed = value_payload("wrong-promotion", result["dimensions"]["shape"]["value"],
                                "int32")
        set_dimension(result, "dtype", observer.observed("int32"))
        set_dimension(result, "value", observer.observed(changed))
        set_dimension(result, "repeated_readback", observer.observed({
            "equal": True, "first_sha256": changed["sha256"],
            "second_sha256": changed["sha256"],
        }))
    elif mutation_id == "MUT-ADD-ACCEPT-INCOMPATIBLE":
        set_dimension(result, "shape", observer.observed([2, 3]))
        set_dimension(result, "exception_stage", observer.observed(None))
        set_dimension(result, "exception_class", observer.observed(None))
        set_dimension(result, "exception_message", observer.observed(None))
        set_dimension(result, "terminal_outcome", observer.observed("returned"))
        result["trace"][2] = {"stage": "invoke_add", "status": "completed"}
        result["trace"][3] = {"stage": "observe_shape_if_constructed",
                              "status": "completed"}
        result["terminal"] = {"outcome": "returned", "exception": None}
    elif mutation_id == "MUT-REALIZE-RETURNS-COPY":
        set_dimension(result, "realize_identity", observer.observed({
            "first_is_result": False, "second_is_result": False,
            "same_realized_object": False,
        }))
    elif mutation_id == "MUT-REALIZE-SECOND-READBACK":
        before = result["dimensions"]["repeated_readback"]["value"]
        set_dimension(result, "repeated_readback", observer.observed({
            "equal": False, "first_sha256": before["first_sha256"],
            "second_sha256": "changed-second-readback",
        }))
    elif mutation_id == "MUT-ADD-MUTATES-LEFT":
        set_dimension(result, "inputs_unchanged", observer.observed({
            "equal": False, "before_sha256": "inputs",
            "after_sha256": "mutated-left",
        }))
    else:
        raise AssertionError(mutation_id)
    return child


def probe_payloads(probe: dict) -> tuple[dict, dict[str, bytes]]:
    protocol_raw = observer.canonical(probe)
    streams = {
        "probe-protocol": protocol_raw,
        "probe-stdout-raw": protocol_raw + b"\n",
        "probe-stderr-raw": b"",
        "probe-stdout-normalized": protocol_raw + b"\n",
        "probe-stderr-normalized": b"",
    }
    payloads = {}
    refs = {}
    for kind, raw in streams.items():
        ref = observer.artifact_ref(kind, raw)
        payloads[ref["path"]] = raw
        refs[kind] = ref
    return {"protocol": probe, "artifacts": refs}, payloads


class DefinitionTests(unittest.TestCase):
    def test_v2_definition_and_impossible_dtype_amendment(self) -> None:
        lock, amendment, manifest = definition()
        self.assertEqual(observer.EXPECTED_TRIAL, lock["trial_id"])
        self.assertEqual("definition_refuted_before_observation",
                         amendment["v1_disposition"])
        mutations = {item["id"]: item for item in manifest["mutations"]}
        for mutation_id in observer.EXPECTED_MUTATION_IDS[1:3]:
            self.assertNotIn("dtype", mutations[mutation_id]["must_not_change"])
            self.assertIn("dtype", mutations[mutation_id]["may_be_unobserved"])

    def test_v1_and_tampered_v2_locks_are_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-V2 or tampered"):
            observer.validate_v2_definition(observer.V1_LOCK)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "lock.json"
            doc = json.loads(observer.V2_LOCK.read_text())
            doc["trial_id"] = "tampered"
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "tampered"):
                observer.validate_v2_definition(path)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, _, cls.manifest = definition()

    def test_common_probe_and_all_branch_sources_compile(self) -> None:
        compile(observer.PROBE_SOURCE, "<broadcast-add-probe>", "exec")
        sources = []
        for mutation_id, function in observer.MUTATION_BRANCH_FUNCTIONS.items():
            source = observer.probe_function_source(function)
            compile(source, f"<{mutation_id}>", "exec")
            sources.append(observer.digest(source))
        self.assertEqual(8, len(set(sources)))

    def test_both_subject_ownership_modes_validate(self) -> None:
        for mode in ("upstream", "tgrad"):
            observer.validate_probe_protocol(
                protocol(self.manifest, mode), self.manifest, mode, None
            )

    def test_fallback_and_protocol_contamination_are_rejected(self) -> None:
        contaminated = protocol(self.manifest, "tgrad")
        contaminated["ownership"]["fallback_detected"] = True
        contaminated["ownership"]["contaminated_modules"] = ["tinygrad.tensor"]
        with self.assertRaisesRegex(RuntimeError, "fallback contamination"):
            observer.validate_probe_protocol(
                contaminated, self.manifest, "tgrad", None
            )
        extra = protocol(self.manifest)
        extra["authored_green"] = True
        with self.assertRaisesRegex(RuntimeError, "top-level shape"):
            observer.validate_probe_protocol(extra, self.manifest, "upstream", None)

    def test_ordered_trace_and_mutation_binding_are_fail_closed(self) -> None:
        altered = protocol(self.manifest)
        altered["scenarios"][0]["trace"].reverse()
        with self.assertRaisesRegex(RuntimeError, "ordered trace"):
            observer.validate_probe_protocol(altered, self.manifest, "upstream", None)
        declaration = self.manifest["mutations"][0]
        config, _ = observer.mutation_config(declaration, "verifier")
        child = child_protocol(self.manifest, declaration, config)
        child["mutation"]["configuration_sha256"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "token"):
            observer.validate_probe_protocol(child, self.manifest, "upstream", config)


class ChildCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, _, cls.manifest = definition()
        cls.baseline = protocol(cls.manifest)

    def calibration(self, declaration: dict) -> tuple[dict, dict, bytes]:
        config, source = observer.mutation_config(declaration, "verifier")
        child = child_protocol(self.manifest, declaration, config)
        observer.validate_probe_protocol(
            child, self.manifest, "upstream", config
        )
        source_ref = observer.artifact_ref("mutant-source", source)
        record, _ = probe_payloads(child)
        result = observer.evaluate_mutant_child(
            declaration, self.baseline, child, config,
            {"branch_source": source_ref, "probe": record},
        )
        return result, child, source

    def test_all_eight_isolated_children_are_rejected(self) -> None:
        identities = []
        for declaration in self.manifest["mutations"]:
            result, child, source = self.calibration(declaration)
            self.assertEqual("validator_rejected_mutant", result["outcome"])
            self.assertIsNot(child, self.baseline)
            identities.append(result["configuration"]["configuration_sha256"])
            self.assertEqual(
                result["configuration"]["branch_source_sha256"],
                observer.digest(source),
            )
        self.assertEqual(8, len(set(identities)))

    def test_v2_result_prevention_records_unobserved_dtype(self) -> None:
        for declaration in self.manifest["mutations"][1:3]:
            result, child, _ = self.calibration(declaration)
            target = next(item for item in child["scenarios"]
                          if item["id"] in declaration["target_scenarios"])
            self.assertEqual("unobserved", target["dimensions"]["dtype"]["state"])
            evaluation = next(item for item in result["evaluations"]
                              if item["targeted"])
            self.assertEqual("unobserved",
                             evaluation["may_be_unobserved"]["dtype"])

    def test_parent_side_fabrication_or_collateral_change_fails(self) -> None:
        declaration = self.manifest["mutations"][0]
        config, source = observer.mutation_config(declaration, "verifier")
        child = child_protocol(self.manifest, declaration, config)
        child["scenarios"][0]["dimensions"]["dtype"] = observer.observed("int32")
        source_ref = observer.artifact_ref("mutant-source", source)
        record, _ = probe_payloads(child)
        result = observer.evaluate_mutant_child(
            declaration, self.baseline, child, config,
            {"branch_source": source_ref, "probe": record},
        )
        self.assertNotEqual("validator_rejected_mutant", result["outcome"])

    def test_non_target_scenario_change_fails(self) -> None:
        declaration = self.manifest["mutations"][0]
        config, source = observer.mutation_config(declaration, "verifier")
        child = child_protocol(self.manifest, declaration, config)
        child["scenarios"][1]["terminal"]["outcome"] = "raised"
        source_ref = observer.artifact_ref("mutant-source", source)
        record, _ = probe_payloads(child)
        result = observer.evaluate_mutant_child(
            declaration, self.baseline, child, config,
            {"branch_source": source_ref, "probe": record},
        )
        self.assertNotEqual("validator_rejected_mutant", result["outcome"])


class ArtifactAndPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, _, cls.manifest = definition()

    def test_literal_and_normalized_streams_replay(self) -> None:
        probe = protocol(self.manifest)
        record, payloads = probe_payloads(probe)
        with tempfile.TemporaryDirectory() as raw:
            evidence = Path(raw) / "evidence.json"
            for relative, content in payloads.items():
                observer.write_once(evidence.parent / relative, content)
            observer.replay_probe_record(
                evidence, record, self.manifest, "upstream", None
            )
            stderr_ref = record["artifacts"]["probe-stderr-raw"]
            (evidence.parent / stderr_ref["path"]).write_bytes(b"warning")
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                observer.replay_probe_record(
                    evidence, record, self.manifest, "upstream", None
                )

    def test_full_eight_child_artifact_closure_replays(self) -> None:
        baseline = protocol(self.manifest)
        baseline_record, payloads = probe_payloads(baseline)
        calibrations = []
        verifier_hash = "synthetic-verifier"
        for declaration in self.manifest["mutations"]:
            config, source = observer.mutation_config(
                declaration, verifier_hash
            )
            child = child_protocol(self.manifest, declaration, config)
            child_record, child_payloads = probe_payloads(child)
            payloads.update(child_payloads)
            source_ref = observer.artifact_ref("mutant-source", source)
            payloads[source_ref["path"]] = source
            calibrations.append(observer.evaluate_mutant_child(
                declaration, baseline, child, config,
                {"branch_source": source_ref, "probe": child_record},
            ))
        document = {
            "against": "upstream",
            "identity": {"verifier": {"observer_sha256": verifier_hash}},
            "observation": baseline_record,
            "calibrations": calibrations,
            "artifacts": observer.refs_for_payloads(payloads),
        }
        with tempfile.TemporaryDirectory() as raw:
            evidence = Path(raw) / "evidence.json"
            for relative, content in payloads.items():
                observer.write_once(evidence.parent / relative, content)
            observer.replay_observation(evidence, document, self.manifest)
            document["artifacts"] = document["artifacts"][:-1]
            with self.assertRaisesRegex(RuntimeError, "artifact list differs"):
                observer.replay_observation(evidence, document, self.manifest)

    def test_baseline_first_and_canonicalization(self) -> None:
        observer.enforce_baseline_first("upstream", None)
        observer.enforce_baseline_first("tgrad", Path("baseline.json"))
        with self.assertRaisesRegex(RuntimeError, "requires"):
            observer.enforce_baseline_first("tgrad", None)
        first = {"b": 2, "a": [1]}
        second = {"a": [1], "b": 2}
        self.assertEqual(observer.canonical(first), observer.canonical(second))

    def test_normalization_handles_macos_path_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            left, right = Path(one), Path(two)
            self.assertEqual(
                observer.normalize_process_stream(
                    f"{left}/x".encode(), {left: "<root>"}
                ),
                observer.normalize_process_stream(
                    f"{right}/x".encode(), {right: "<root>"}
                ),
            )


if __name__ == "__main__":
    unittest.main()

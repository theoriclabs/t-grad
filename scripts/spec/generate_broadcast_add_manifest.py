#!/usr/bin/env python3
"""Validate and project the prospective broadcast-add manifest into Lean."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "fixtures" / "requirements" / "broadcast_add_prospective_v1.json"
DEFAULT_OUTPUT = REPO / "Tgrad" / "Growth" / "BroadcastAddManifestGenerated.lean"

DIMENSIONS = {
    "value": ".value", "shape": ".shape", "dtype": ".dtype",
    "exception": ".exception", "exception_stage": ".exceptionStage",
    "exception_class": ".exceptionClass",
    "exception_message": ".exceptionMessage",
    "realize_identity": ".objectIdentity",
    "repeated_readback": ".readbackStability",
    "inputs_unchanged": ".inputImmutability",
    "terminal_outcome": ".terminalOutcome",
}
STATES = {
    "unobserved": ".unobserved", "failed": ".failed",
    "passed_calibrated": ".passedCalibrated", "blocked": ".blocked",
    "verifier_error": ".verifierError",
}
DTYPES = {"float32": ".float32", "int32": ".int32"}
KINDS = {"tensor": ".tensor", "scalar": ".scalar"}
TRACE_EVENTS = {
    "construct_left": ".constructLeft",
    "construct_right": ".constructRight",
    "snapshot_inputs": ".snapshotInputs",
    "invoke_add": ".invokeAdd",
    "capture_result_identity": ".captureResultIdentity",
    "observe_shape": ".observeShape",
    "observe_dtype": ".observeDtype",
    "realize_1": ".realize1",
    "readback_1": ".readback1",
    "realize_2": ".realize2",
    "readback_2": ".readback2",
    "snapshot_inputs_after": ".snapshotInputsAfter",
    "observe_shape_if_constructed": ".observeShapeIfConstructed",
    "record_terminal_stage_class_message": ".recordTerminalStageClassMessage",
}
OBSERVATIONS = {
    "construction": ".construction", "shape": ".shape",
    "shape_access": ".shapeAccess", "dtype": ".dtype",
    "exact_values": ".exactValues", "realize_identity": ".realizeIdentity",
    "repeated_readback": ".repeatedReadback",
    "inputs_unchanged": ".inputsUnchanged",
    "exception_stage": ".exceptionStage",
    "exception_class": ".exceptionClass",
    "exception_message": ".exceptionMessage",
}
FORBIDDEN = {
    "conformant": ".conformant", "promoted": ".promoted",
    "full_tinygrad_parity": ".fullTinygradParity",
}


def q(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def lean_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def required_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{where} must be a non-empty string")
    return value


def unique_strings(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{where} must be a non-empty list")
    values = [required_string(item, where) for item in value]
    if len(values) != len(set(values)):
        raise RuntimeError(f"{where} contains duplicates")
    return values


def parse(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise RuntimeError("unsupported broadcast-add manifest schema")
    if data.get("frozen_before_product_change") is not True:
        raise RuntimeError("manifest is not declared prospectively frozen")
    requirements = unique_strings(data.get("requirements"), "requirements")
    profile = data.get("profile")
    if not isinstance(profile, dict):
        raise RuntimeError("profile must be an object")
    if profile.get("backend") != "METAL" or profile.get("deterministic") is not True:
        raise RuntimeError("profile must pin deterministic METAL execution")
    if profile.get("operator") != "Tensor.__add__":
        raise RuntimeError("operator must be exactly Tensor.__add__")
    legal_trace = unique_strings(profile.get("legal_trace"), "legal trace")
    incompatible_trace = unique_strings(
        profile.get("incompatible_trace"), "incompatible trace"
    )
    if set(legal_trace + incompatible_trace) - set(TRACE_EVENTS):
        raise RuntimeError("trace contains an unknown event")
    unique_strings(profile.get("excluded"), "profile exclusions")
    scenarios = data.get("scenarios")
    mutations = data.get("mutations")
    if not isinstance(scenarios, list) or len(scenarios) != 6:
        raise RuntimeError("exactly six scenarios are required")
    if not isinstance(mutations, list) or len(mutations) != 8:
        raise RuntimeError("exactly eight atomic mutations are required")
    scenario_ids = unique_strings([item.get("id") for item in scenarios], "scenario ids")
    mutation_ids = unique_strings([item.get("id") for item in mutations], "mutation ids")
    known_requirements = set(requirements)
    for scenario in scenarios:
        refs = unique_strings(scenario.get("requirements"), f"{scenario['id']} requirements")
        if not set(refs) <= known_requirements:
            raise RuntimeError(f"{scenario['id']} references an unknown requirement")
        observations = unique_strings(
            scenario.get("observe"), f"{scenario['id']} observations"
        )
        if set(observations) - set(OBSERVATIONS):
            raise RuntimeError(f"{scenario['id']} has unknown observation tokens")
        for side in ("left", "right"):
            operand = scenario.get(side)
            if not isinstance(operand, dict):
                raise RuntimeError(f"{scenario['id']}.{side} must be an object")
            if operand.get("kind") not in KINDS:
                raise RuntimeError(f"{scenario['id']}.{side} has unknown kind")
            required_string(
                operand.get("construction"), f"{scenario['id']}.{side}.construction"
            )
            if operand.get("dtype") not in DTYPES:
                raise RuntimeError(f"{scenario['id']}.{side} has unknown dtype")
            shape = operand.get("shape")
            values = operand.get("values")
            if (not isinstance(shape, list) or
                any(not isinstance(dim, int) or dim < 0 for dim in shape)):
                raise RuntimeError(f"{scenario['id']}.{side} has invalid shape")
            if not isinstance(values, list):
                raise RuntimeError(f"{scenario['id']}.{side}.values must be a list")
            expected = 1
            for dim in shape:
                expected *= dim
            if len(values) != expected:
                raise RuntimeError(
                    f"{scenario['id']}.{side} value count does not match shape"
                )
    for mutation in mutations:
        targets = unique_strings(
            mutation.get("target_scenarios"), f"{mutation['id']} targets"
        )
        if not set(targets) <= set(scenario_ids):
            raise RuntimeError(f"{mutation['id']} targets an unknown scenario")
        required_string(mutation.get("fault"), f"{mutation['id']}.fault")
        required_string(
            mutation.get("implementation"), f"{mutation['id']}.implementation"
        )
        dimension_sets = []
        for key in ("must_fail", "must_not_change", "may_be_unobserved"):
            values = mutation.get(key)
            if not isinstance(values, list):
                raise RuntimeError(f"{mutation['id']}.{key} must be a list")
            unknown = set(values) - set(DIMENSIONS)
            if unknown:
                raise RuntimeError(f"{mutation['id']}.{key} has unknown dimensions: {sorted(unknown)}")
            dimension_sets.append(set(values))
        if not mutation["must_fail"]:
            raise RuntimeError(f"{mutation['id']} must fail at least one dimension")
        if any(dimension_sets[i] & dimension_sets[j]
               for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError(f"{mutation['id']} dimension policies overlap")
    delta = data.get("expected_observation_delta", {})
    if delta.get("before") not in STATES:
        raise RuntimeError("unknown expected before-state")
    allowed = unique_strings(delta.get("allowed_after"), "allowed after-states")
    if set(allowed) - set(STATES):
        raise RuntimeError("unknown allowed after-state")
    forbidden = unique_strings(delta.get("forbidden_inference"), "forbidden claims")
    if set(forbidden) - set(FORBIDDEN):
        raise RuntimeError("unknown forbidden inference")
    data["_requirements"] = requirements
    data["_scenario_ids"] = scenario_ids
    data["_mutation_ids"] = mutation_ids
    data["_allowed"] = allowed
    data["_forbidden"] = forbidden
    return data, hashlib.sha256(raw).hexdigest()


def generated(path: Path, data: dict, content_hash: str) -> str:
    scenarios = []
    for scenario in data["scenarios"]:
        def operand(value: dict) -> str:
            return (
                "{ kind := " + KINDS[value["kind"]] +
                ", construction := " + q(value["construction"]) +
                ", shape := " + lean_list([str(dim) for dim in value["shape"]]) +
                ", dtype := " + DTYPES[value["dtype"]] +
                f", valueCount := {len(value['values'])} }}"
            )
        scenarios.append(
            "{ id := " + q(scenario["id"]) +
            ", requirementIds := " +
            lean_list([f'⟨{q(value)}⟩' for value in scenario["requirements"]]) +
            ", left := " + operand(scenario["left"]) +
            ", right := " + operand(scenario["right"]) +
            ", observations := " +
            lean_list([OBSERVATIONS[value] for value in scenario["observe"]]) + " }"
        )
    mutations = []
    for mutation in data["mutations"]:
        mutations.append(
            "{ id := " + q(mutation["id"]) +
            ", targetScenarios := " +
            lean_list([q(value) for value in mutation["target_scenarios"]]) +
            ", fault := " + q(mutation["fault"]) +
            ", implementation := " + q(mutation["implementation"]) +
            ", mustFail := " + lean_list([DIMENSIONS[v] for v in mutation["must_fail"]]) +
            ", mustNotChange := " +
            lean_list([DIMENSIONS[v] for v in mutation["must_not_change"]]) +
            ", mayBeUnobserved := " +
            lean_list([DIMENSIONS[v] for v in mutation["may_be_unobserved"]]) + " }"
        )
    relative = path.relative_to(REPO).as_posix()
    return f'''import Tgrad.Growth.BroadcastAddManifest

/- GENERATED by scripts/spec/generate_broadcast_add_manifest.py; do not edit. -/

namespace Tgrad.Growth.BroadcastAddManifestGenerated

open Tgrad.Requirements
open Tgrad.Growth
open Tgrad.Growth.BroadcastAddManifest

def manifest : FrozenManifest :=
  {{ schemaVersion := {data["schema_version"]}
    packetId := {q(data["packet_id"])}
    baselineRevision := {q(data["baseline_revision"])}
    upstreamRevision := {q(data["upstream_revision"])}
    backend := {q(data["profile"]["backend"])}
    deterministic := true
    operator := {q(data["profile"]["operator"])}
    legalTrace := {lean_list([TRACE_EVENTS[value] for value in data["profile"]["legal_trace"]])}
    incompatibleTrace := {lean_list([TRACE_EVENTS[value] for value in data["profile"]["incompatible_trace"]])}
    exclusions := {lean_list([q(value) for value in data["profile"]["excluded"]])}
    path := {q(relative)}
    contentHash := {q(content_hash)}
    requirementIds := {lean_list([f'⟨{q(value)}⟩' for value in data['_requirements']])}
    scenarios := {lean_list(scenarios)}
    mutations := {lean_list(mutations)}
    expectedBefore := {STATES[data["expected_observation_delta"]["before"]]}
    allowedAfter := {lean_list([STATES[value] for value in data['_allowed']])}
    forbiddenInferences := {lean_list([FORBIDDEN[value] for value in data['_forbidden']])} }}

theorem manifest_is_well_formed : manifest.wellFormed := by native_decide
theorem manifest_has_six_scenarios_and_eight_atomic_mutations :
    manifest.scenarios.length = 6 ∧ manifest.mutations.length = 8 := by native_decide

end Tgrad.Growth.BroadcastAddManifestGenerated
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.input.resolve()
    data, content_hash = parse(source)
    output = generated(source, data, content_hash)
    target = args.output.resolve()
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != output:
            raise RuntimeError(f"generated broadcast-add manifest is stale: {target}")
        print(f"checked {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(output)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

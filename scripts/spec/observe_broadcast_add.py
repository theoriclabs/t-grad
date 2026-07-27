#!/usr/bin/env python3
"""Prospective V4 broadcast-add observer.

This observer is deliberately an evidence instrument, not a conformance gate.
It executes one fixed probe against either the pinned upstream tinygrad tree or
Tgrad's strict substitution, records exact observations, and (for upstream)
calibrates every declared observation dimension with observer-owned mutants.

The protocol is prospective and fail-closed:

* the V3 semantic lock and V4 chronology amendment are checked before subject code can run;
* the definition commit and every frozen file are revalidated with git;
* V1 is rejected rather than interpreted;
* a Tgrad run requires a replayable upstream evidence artifact first;
* protocol, module ownership, artifacts, and identities are content-bound;
* no timestamp, random run id, temporary path, conformance state, or promotion
  judgment is emitted.

The script never edits product code.  Output is explicit and content-addressed;
an existing different artifact is never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
V2_LOCK = REPO / "fixtures/requirements/broadcast_add_trial_lock_v2.json"
V3_LOCK = REPO / "fixtures/requirements/broadcast_add_trial_lock_v3.json"
V1_LOCK = REPO / "fixtures/requirements/broadcast_add_trial_lock_v1.json"
V1_MANIFEST = REPO / "fixtures/requirements/broadcast_add_prospective_v1.json"
V2_AMENDMENT = (
    REPO / "fixtures/requirements/broadcast_add_prospective_v2_amendment.json"
)
V3_AMENDMENT = (
    REPO / "fixtures/requirements/broadcast_add_prospective_v3_amendment.json"
)
V4_AMENDMENT = (
    REPO / "fixtures/requirements/broadcast_add_prospective_v4_tooling_amendment.json"
)
SHIM_ROOT = REPO / "scripts/parity/shim"
PRODUCT_PYTHON = REPO / "python"
RUNTIME_DIR = REPO / ".lake/build/lib"
RUNTIME_NAMES = ("libtgrad.dylib", "libtgrad_Tgrad.dylib")
DEFAULT_PYTHON = REPO / ".venv/bin/python"

SCHEMA_VERSION = 1
PROBE_SCHEMA_VERSION = 1
EXPECTED_LOCK_SHA256 = (
    "c52fbd1523fb2a57b613181ac68dcdd183af477ed2737848ba6dce41aeb898ef"
)
EXPECTED_TRIAL = "TRIAL-BROADCAST-ADD-PROSPECTIVE-V2"
EXPECTED_DEFINITION_REVISION = "9762bb722c9f76283cf62cb16e8df2f902dc92ba"
EXPECTED_EFFECTIVE_MANIFEST_SHA256 = (
    "0a8cc4f19fdd1177e93e44f93115c6149ea124132380de45eb9c43f64ac57795"
)
EXPECTED_V3_LOCK_SHA256 = (
    "975820bfd46b1eaa3a5826f7c8357e5560d380cd5a6da1c88b1afc278ccb2b2a"
)
EXPECTED_V3_DEFINITION_REVISION = "fbb13c585da75f3e9fa00cf5d87aa72f0ff38ab4"
EXPECTED_V3_CONTRACT_SHA256 = (
    "92ae7cca5e6be6751b242439e600766d75a71c6aab1a3f9dcd838791f634539d"
)
EXPECTED_V4_AMENDMENT_SHA256 = (
    "b5cbcf16482c418875aecc9b1f7eb59b4de1800f49b3ba8c331954072f1cb6b5"
)
EXPECTED_V4_DEFINITION_REVISION = "35ce40fdbf474b3f449452f738e3bf517da6ce7d"
EXPECTED_V4_TRIAL = "TRIAL-BROADCAST-ADD-PROSPECTIVE-V4"
EXPECTED_V4_EFFECTIVE_SHA256 = (
    "02229c32937df15704714282ee28754b8fed3938b9b4f33d1aba254b206b4c1e"
)
EXPECTED_SCENARIO_IDS = (
    "ADD-SAME-SHAPE-F32",
    "ADD-SINGLETON-AXIS-F32",
    "ADD-RANK-EXTENSION-I32",
    "ADD-TWO-SIDED-BROADCAST-F32",
    "ADD-I32-F32-SCALAR-PROMOTION",
    "ADD-INCOMPATIBLE-SHAPES",
)
EXPECTED_MUTATION_IDS = (
    "MUT-ADD-SUBTRACT",
    "MUT-ADD-WRONG-RIGHT-ALIGNMENT",
    "MUT-ADD-NO-SINGLETON-EXPANSION",
    "MUT-ADD-WRONG-PROMOTION",
    "MUT-ADD-ACCEPT-INCOMPATIBLE",
    "MUT-REALIZE-RETURNS-COPY",
    "MUT-REALIZE-SECOND-READBACK",
    "MUT-ADD-MUTATES-LEFT",
)

LEGAL_DIMENSIONS = (
    "construction", "shape", "dtype", "value", "realize_identity",
    "repeated_readback", "inputs_unchanged", "exception_stage",
    "exception_class", "exception_message", "terminal_outcome",
)
INCOMPATIBLE_DIMENSIONS = (
    "construction", "shape", "exception_stage", "exception_class",
    "exception_message", "terminal_outcome",
)


def canonical(value: object) -> bytes:
    """The sole canonical JSON representation used for identities."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def git_bytes(*args: str, repo: Path = REPO) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def git_text(*args: str, repo: Path = REPO) -> str:
    return git_bytes(*args, repo=repo).decode().strip()


def directory_content_hash(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append((path.relative_to(root).as_posix(), file_hash(path)))
    if not entries:
        raise RuntimeError(f"cannot identify empty source tree: {root}")
    return digest(canonical(entries))


def git_directory_hash(revision: str, root: str) -> str:
    paths = git_text(
        "ls-tree", "-r", "--name-only", revision, "--", root
    ).splitlines()
    prefix = root.rstrip("/") + "/"
    entries = [
        (path.removeprefix(prefix), digest(git_bytes("show", f"{revision}:{path}")))
        for path in sorted(path for path in paths if path)
    ]
    if not entries:
        raise RuntimeError(f"no tracked files beneath {root!r} at {revision}")
    return digest(canonical(entries))


def observed(value: object) -> dict:
    return {"state": "observed", "value": value}


def unobserved(reason: str) -> dict:
    return {"state": "unobserved", "reason": reason}


def is_observed(value: object) -> bool:
    return isinstance(value, dict) and value.get("state") == "observed"


def validate_v2_definition(
    lock_path: Path = V2_LOCK,
    *,
    require_current: bool = True,
) -> tuple[dict, dict, dict]:
    """Validate and compose the effective V2 definition.

    Lock hashing occurs before JSON parsing or git work.  A V1 lock therefore
    cannot be accepted through a compatible-looking schema.
    """
    lock_path = lock_path.resolve()
    lock_raw = lock_path.read_bytes()
    actual_lock_hash = digest(lock_raw)
    if actual_lock_hash != EXPECTED_LOCK_SHA256:
        raise RuntimeError(
            "refusing non-V2 or tampered trial lock: "
            f"expected {EXPECTED_LOCK_SHA256}, got {actual_lock_hash}"
        )
    lock = json.loads(lock_raw)
    if lock.get("schema_version") != 1:
        raise RuntimeError("unsupported V2 trial-lock schema")
    if lock.get("trial_id") != EXPECTED_TRIAL:
        raise RuntimeError("V1 or unknown trial is not an executable definition")
    if lock.get("definition_revision") != EXPECTED_DEFINITION_REVISION:
        raise RuntimeError("V2 definition revision changed")
    if lock.get("status") != "definition_frozen_observer_unimplemented":
        raise RuntimeError("V2 lock has an unexpected lifecycle status")
    tree = git_text("rev-parse", f"{EXPECTED_DEFINITION_REVISION}^{{tree}}")
    if tree != lock.get("definition_tree"):
        raise RuntimeError("V2 definition tree does not match the lock")
    definition_files = lock.get("definition_files")
    if not isinstance(definition_files, dict) or not definition_files:
        raise RuntimeError("V2 lock has no frozen definition files")
    for relative, expected in sorted(definition_files.items()):
        frozen = git_bytes("show", f"{EXPECTED_DEFINITION_REVISION}:{relative}")
        if digest(frozen) != expected:
            raise RuntimeError(f"frozen V2 file hash mismatch: {relative}")
        if require_current:
            current = REPO / relative
            if not current.is_file() or file_hash(current) != expected:
                raise RuntimeError(f"current V2 definition drifted: {relative}")
    required_policy = (
        "definition_drift_invalidates_trial",
        "observer_must_bind_this_lock_sha256",
        "product_candidate_forbidden_before_baseline_observation",
        "adequacy_remains_open",
        "legacy_requirement_remains_historical",
    )
    policy = lock.get("promotion_policy", {})
    if any(policy.get(item) is not True for item in required_policy):
        raise RuntimeError("V2 lock weakens a required policy")

    amendment_raw = V2_AMENDMENT.read_bytes()
    amendment = json.loads(amendment_raw)
    if file_hash(V2_AMENDMENT) != definition_files.get(
        "fixtures/requirements/broadcast_add_prospective_v2_amendment.json"
    ):
        raise RuntimeError("V2 amendment is not the lock-bound amendment")
    if amendment.get("trial_id") != EXPECTED_TRIAL:
        raise RuntimeError("V2 amendment trial id changed")
    if amendment.get("v1_disposition") != "definition_refuted_before_observation":
        raise RuntimeError("V1 must remain explicitly definition-refuted")
    if amendment.get("frozen_before_observer_implementation") is not True:
        raise RuntimeError("V2 is not prospective with respect to this observer")
    if file_hash(V1_MANIFEST) != amendment.get("base_manifest_sha256"):
        raise RuntimeError("V2 does not bind the exact immutable V1 manifest")
    if file_hash(V1_LOCK) != amendment.get("supersedes_trial_lock_sha256"):
        raise RuntimeError("V2 does not bind the exact refuted V1 lock")

    base = json.loads(V1_MANIFEST.read_bytes())
    effective = copy.deepcopy(base)
    effective["trial_id"] = EXPECTED_TRIAL
    effective["packet_id"] = amendment["packet_id"]
    effective["baseline_revision"] = amendment["baseline_revision"]
    mutations = {item["id"]: item for item in effective["mutations"]}
    amendments = amendment.get("amendments")
    expected_amendments = [
        {
            "mutation_id": mutation_id, "dimension": "dtype",
            "from": "must_not_change", "to": "may_be_unobserved",
        }
        for mutation_id in EXPECTED_MUTATION_IDS[1:3]
    ]
    if amendments != expected_amendments:
        raise RuntimeError("V2 amendment is not exactly the two dtype moves")
    for item in amendments:
        mutation = mutations[item["mutation_id"]]
        if "dtype" not in mutation["must_not_change"]:
            raise RuntimeError("V2 amendment source dimension is absent")
        mutation["must_not_change"].remove("dtype")
        mutation["may_be_unobserved"].append("dtype")
    effective_hash = digest(canonical({
        "base_manifest_sha256": amendment["base_manifest_sha256"],
        "amendment_sha256": digest(amendment_raw),
    }))
    if effective_hash != EXPECTED_EFFECTIVE_MANIFEST_SHA256:
        raise RuntimeError("effective V2 manifest identity changed")
    if tuple(item["id"] for item in effective["scenarios"]) != EXPECTED_SCENARIO_IDS:
        raise RuntimeError("V2 scenario set or order changed")
    if tuple(item["id"] for item in effective["mutations"]) != EXPECTED_MUTATION_IDS:
        raise RuntimeError("V2 mutation set or order changed")
    for mutation_id in EXPECTED_MUTATION_IDS[1:3]:
        mutation = mutations[mutation_id]
        if "dtype" in mutation["must_not_change"]:
            raise RuntimeError("V1's impossible dtype preservation leaked into V2")
        if "dtype" not in mutation["may_be_unobserved"]:
            raise RuntimeError("V2 does not permit the prevented-result dtype gap")
    effective["effective_manifest_sha256"] = effective_hash
    return lock, amendment, effective


def validate_v4_definition(
    lock_path: Path = V4_AMENDMENT,
    *,
    require_current: bool = True,
) -> tuple[dict, dict, dict]:
    """Compose V2 behavior, V3 trace semantics, and V4 chronology tooling."""
    lock_path = lock_path.resolve()
    v4_raw = lock_path.read_bytes()
    if digest(v4_raw) != EXPECTED_V4_AMENDMENT_SHA256:
        raise RuntimeError("refusing non-V4 or tampered active definition")
    v4 = json.loads(v4_raw)
    if v4.get("trial_id") != EXPECTED_V4_TRIAL:
        raise RuntimeError("V4 trial id changed")
    if digest(git_bytes(
        "show", f"{EXPECTED_V4_DEFINITION_REVISION}:"
        "fixtures/requirements/broadcast_add_prospective_v4_tooling_amendment.json"
    )) != EXPECTED_V4_AMENDMENT_SHA256:
        raise RuntimeError("V4 amendment was not frozen before implementation")
    required_v4 = (
        "v3_trace_footprint_amendment_inherited_unchanged",
        "v3_behavioral_manifest_inherited_unchanged",
        "frozen_before_v4_tooling_implementation",
        "frozen_before_v4_observer_implementation",
        "frozen_before_product_candidate",
    )
    if any(v4.get(key) is not True for key in required_v4):
        raise RuntimeError("V4 weakens chronology or inheritance")
    if v4.get("tooling_amendment") != {
        "binding": "v2_observer_sha256",
        "from": "current_worktree_file",
        "to": "frozen_git_object_at_v3_definition_revision",
    }:
        raise RuntimeError("V4 tooling correction changed")

    v3_raw = V3_LOCK.read_bytes()
    if digest(v3_raw) != EXPECTED_V3_LOCK_SHA256:
        raise RuntimeError("V3 semantic lock changed")
    v3_lock = json.loads(v3_raw)
    if v3_lock.get("definition_revision") != EXPECTED_V3_DEFINITION_REVISION:
        raise RuntimeError("V3 definition revision changed")
    definition_files = v3_lock.get("definition_files", {})
    authorized_generator = v4["v3_generator_path"]
    for relative, expected in sorted(definition_files.items()):
        if digest(git_bytes(
            "show", f"{EXPECTED_V3_DEFINITION_REVISION}:{relative}"
        )) != expected:
            raise RuntimeError(f"frozen V3 definition mismatch: {relative}")
        if require_current and relative != authorized_generator:
            current = REPO / relative
            if not current.is_file() or file_hash(current) != expected:
                raise RuntimeError(f"unauthorized V3 definition drift: {relative}")
    if v4.get("v3_generator_sha256") != definition_files.get(authorized_generator):
        raise RuntimeError("V4 does not authorize the exact refuted generator")

    _, _, manifest = validate_v2_definition(require_current=require_current)
    v3_amendment_raw = V3_AMENDMENT.read_bytes()
    v3_amendment = json.loads(v3_amendment_raw)
    if digest(v3_amendment_raw) != v4.get("v3_amendment_sha256"):
        raise RuntimeError("V4 does not inherit the exact V3 trace amendment")
    v3_contract_hash = digest(canonical({
        "base_v2_amendment_sha256": v3_amendment["base_v2_amendment_sha256"],
        "base_v2_lock_sha256": v3_amendment["base_v2_lock_sha256"],
        "v2_observer_sha256": v3_amendment["v2_observer_sha256"],
        "v3_amendment_sha256": digest(v3_amendment_raw),
    }))
    if v3_contract_hash != EXPECTED_V3_CONTRACT_SHA256:
        raise RuntimeError("V3 trace-contract identity changed")
    effective_hash = digest(canonical({
        "v3_contract_sha256": v3_contract_hash,
        "v4_amendment_sha256": digest(v4_raw),
    }))
    if effective_hash != EXPECTED_V4_EFFECTIVE_SHA256:
        raise RuntimeError("V4 effective definition identity changed")
    manifest = copy.deepcopy(manifest)
    manifest["trial_id"] = EXPECTED_V4_TRIAL
    manifest["packet_id"] = v4["packet_id"]
    manifest["baseline_revision"] = v4["baseline_revision"]
    manifest["effective_manifest_sha256"] = effective_hash
    active = {
        "trial_id": EXPECTED_V4_TRIAL,
        "sha256": EXPECTED_V4_AMENDMENT_SHA256,
        "definition_revision": EXPECTED_V4_DEFINITION_REVISION,
        "definition_tree": git_text(
            "rev-parse", f"{EXPECTED_V4_DEFINITION_REVISION}^{{tree}}"
        ),
        "upstream_revision": v3_lock["upstream_revision"],
        "execution_boundary": copy.deepcopy(v3_lock["execution_boundary"]),
        "semantic_lock_sha256": EXPECTED_V3_LOCK_SHA256,
    }
    return active, {"v3": v3_amendment, "v4": v4}, manifest


# This one source string is used for both subjects.  Subject-specific behavior
# is limited to activation and ownership assertions selected by SUBJECT_MODE.
PROBE_SOURCE = r'''
import hashlib
import importlib
import itertools
import json
import os
import struct
import sys
from pathlib import Path

SCHEMA = 1

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def observed(value):
    return {"state": "observed", "value": value}

def unobserved(reason):
    return {"state": "unobserved", "reason": reason}

def dtype_name(value):
    raw = str(value).lower().replace("dtypes.", "")
    aliases = {
        "float": "float32", "float32": "float32", "f32": "float32",
        "int": "int32", "int32": "int32", "i32": "int32",
    }
    if raw in aliases:
        return aliases[raw]
    name = str(getattr(value, "name", "")).lower()
    if name in aliases:
        return aliases[name]
    return raw

def exact_readback(tensor):
    import numpy as np
    logical_dtype = dtype_name(tensor.dtype)
    # The pinned upstream revision's Buffer.numpy constructs an ndarray but
    # accidentally omits its return. The requirement names logical readback,
    # not that convenience method. Use the public tolist boundary and rebuild
    # canonical storage from the separately observed dtype and shape.
    shape = tuple(int(value) for value in tensor.shape)
    array = np.asarray(tensor.tolist()).reshape(shape)
    if logical_dtype == "float32":
        storage = np.asarray(array, dtype="<f4")
        tokens = [float(value).hex() for value in storage.reshape(-1)]
    elif logical_dtype == "int32":
        storage = np.asarray(array, dtype="<i4")
        tokens = [str(int(value)) for value in storage.reshape(-1)]
    else:
        storage = np.ascontiguousarray(array)
        tokens = [repr(value.item() if hasattr(value, "item") else value)
                  for value in storage.reshape(-1)]
    payload = {
        "shape": list(shape),
        "dtype": logical_dtype,
        "storage_dtype": storage.dtype.str,
        "storage_hex": storage.tobytes(order="C").hex(),
        "values": tokens,
    }
    payload["sha256"] = digest(canonical(payload))
    return payload

def operand_value(value):
    if isinstance(value, str):
        return float(value)
    return value

def construct(spec, Tensor, dtypes):
    dtype = getattr(dtypes, spec["dtype"])
    values = [operand_value(value) for value in spec["values"]]
    if spec["shape"] == []:
        return Tensor(values[0], dtype=dtype)
    return Tensor(values, dtype=dtype).reshape(tuple(spec["shape"]))

def mut_add_subtract(state, scenario, Tensor, dtypes):
    return state["left"] - state["right"]

def mut_wrong_right_alignment(state, scenario, Tensor, dtypes):
    # Deliberately left-align the rank-one right operand as (N, 1), which is
    # incompatible with the scenario's (M, N) left operand.
    wrong = state["right"].reshape((scenario["right"]["shape"][0], 1))
    return state["left"] + wrong

def mut_no_singleton_expansion(state, scenario, Tensor, dtypes):
    raise ValueError("observer mutant rejects singleton-axis expansion")

def mut_wrong_promotion(state, scenario, Tensor, dtypes):
    return (state["left"] + state["right"]).cast(dtypes.int32)

def mut_accept_incompatible(state, scenario, Tensor, dtypes):
    # Return a real constructed Tensor rather than fabricating observations.
    return construct(scenario["left"], Tensor, dtypes)

def mut_realize_copy(result, Tensor, dtypes):
    import numpy as np
    array = np.asarray(result.numpy())
    copied = Tensor(array.reshape(-1).tolist(), dtype=result.dtype).reshape(
        tuple(int(value) for value in array.shape)
    )
    return copied.realize()

def mut_second_readback(payload):
    changed = dict(payload)
    raw = bytearray.fromhex(changed["storage_hex"])
    if not raw:
        raise RuntimeError("second-readback mutant requires nonempty output")
    raw[0] ^= 1
    changed["storage_hex"] = bytes(raw).hex()
    changed["values"] = list(changed["values"])
    changed["values"][0] = "observer-mutant-second-readback"
    changed.pop("sha256", None)
    changed["sha256"] = digest(canonical(changed))
    return changed

def mut_mutate_left(state, scenario, Tensor, dtypes):
    one_spec = dict(scenario["left"])
    one_spec["values"] = [1 for _ in one_spec["values"]]
    delta = construct(one_spec, Tensor, dtypes)
    state["left"].assign(state["left"] + delta).realize()

MUTATION_FUNCTIONS = {
    "MUT-ADD-SUBTRACT": mut_add_subtract,
    "MUT-ADD-WRONG-RIGHT-ALIGNMENT": mut_wrong_right_alignment,
    "MUT-ADD-NO-SINGLETON-EXPANSION": mut_no_singleton_expansion,
    "MUT-ADD-WRONG-PROMOTION": mut_wrong_promotion,
    "MUT-ADD-ACCEPT-INCOMPATIBLE": mut_accept_incompatible,
    "MUT-REALIZE-RETURNS-COPY": mut_realize_copy,
    "MUT-REALIZE-SECOND-READBACK": mut_second_readback,
    "MUT-ADD-MUTATES-LEFT": mut_mutate_left,
}

def applies(mutation, scenario):
    return mutation is not None and scenario["id"] in mutation["target_scenarios"]

def invoke_add_adapter(state, scenario, Tensor, dtypes, mutation):
    if applies(mutation, scenario) and mutation["id"] in {
        "MUT-ADD-SUBTRACT", "MUT-ADD-WRONG-RIGHT-ALIGNMENT",
        "MUT-ADD-NO-SINGLETON-EXPANSION", "MUT-ADD-WRONG-PROMOTION",
        "MUT-ADD-ACCEPT-INCOMPATIBLE",
    }:
        return MUTATION_FUNCTIONS[mutation["id"]](state, scenario, Tensor, dtypes)
    return state["left"] + state["right"]

def realize_adapter(result, scenario, Tensor, dtypes, mutation):
    if applies(mutation, scenario) and mutation["id"] == "MUT-REALIZE-RETURNS-COPY":
        return mut_realize_copy(result, Tensor, dtypes)
    return result.realize()

def readback_adapter(result, scenario, mutation, ordinal):
    payload = exact_readback(result)
    if (applies(mutation, scenario) and
            mutation["id"] == "MUT-REALIZE-SECOND-READBACK" and ordinal == 2):
        return mut_second_readback(payload)
    return payload

def before_final_input_snapshot(state, scenario, Tensor, dtypes, mutation):
    if applies(mutation, scenario) and mutation["id"] == "MUT-ADD-MUTATES-LEFT":
        mut_mutate_left(state, scenario, Tensor, dtypes)

def exception_record(exc):
    return {"class": type(exc).__name__, "message": str(exc)}

def legal_scenario(scenario, trace_names, Tensor, dtypes, mutation):
    state = {}
    trace = []
    failure = None
    for stage in trace_names:
        if failure is not None:
            trace.append({"stage": stage, "status": "skipped",
                          "reason": "prior_stage_raised"})
            continue
        try:
            if stage == "construct_left":
                state["left"] = construct(scenario["left"], Tensor, dtypes)
            elif stage == "construct_right":
                state["right"] = construct(scenario["right"], Tensor, dtypes)
            elif stage == "snapshot_inputs":
                state["inputs_before"] = {
                    "left": exact_readback(state["left"]),
                    "right": exact_readback(state["right"]),
                }
            elif stage == "invoke_add":
                state["result"] = invoke_add_adapter(
                    state, scenario, Tensor, dtypes, mutation
                )
            elif stage == "capture_result_identity":
                state["result_identity_captured"] = True
            elif stage == "observe_shape":
                state["shape"] = [int(value) for value in state["result"].shape]
            elif stage == "observe_dtype":
                state["dtype"] = dtype_name(state["result"].dtype)
            elif stage == "realize_1":
                state["realized_1"] = realize_adapter(
                    state["result"], scenario, Tensor, dtypes, mutation
                )
            elif stage == "readback_1":
                state["readback_1"] = readback_adapter(
                    state["result"], scenario, mutation, 1
                )
            elif stage == "realize_2":
                state["realized_2"] = realize_adapter(
                    state["result"], scenario, Tensor, dtypes, mutation
                )
            elif stage == "readback_2":
                state["readback_2"] = readback_adapter(
                    state["result"], scenario, mutation, 2
                )
            elif stage == "snapshot_inputs_after":
                before_final_input_snapshot(
                    state, scenario, Tensor, dtypes, mutation
                )
                state["inputs_after"] = {
                    "left": exact_readback(state["left"]),
                    "right": exact_readback(state["right"]),
                }
            else:
                raise RuntimeError("unknown legal trace stage: " + stage)
            trace.append({"stage": stage, "status": "completed"})
        except BaseException as exc:
            failure = {"stage": stage, **exception_record(exc)}
            trace.append({"stage": stage, "status": "raised",
                          "exception_class": failure["class"],
                          "exception_message": failure["message"]})

    dimensions = {
        "construction": observed({
            "left": "left" in state, "right": "right" in state,
        }),
        "shape": observed(state["shape"]) if "shape" in state else
                 unobserved("result_shape_not_reached"),
        "dtype": observed(state["dtype"]) if "dtype" in state else
                 unobserved("result_dtype_not_reached"),
        "value": observed(state["readback_1"]) if "readback_1" in state else
                 unobserved("first_readback_not_reached"),
        "realize_identity": observed({
            "first_is_result": state["realized_1"] is state["result"],
            "second_is_result": state["realized_2"] is state["result"],
            "same_realized_object": state["realized_1"] is state["realized_2"],
        }) if "realized_1" in state and "realized_2" in state else
                 unobserved("both_realizations_not_reached"),
        "repeated_readback": observed({
            "equal": state["readback_1"] == state["readback_2"],
            "first_sha256": state["readback_1"]["sha256"],
            "second_sha256": state["readback_2"]["sha256"],
        }) if "readback_1" in state and "readback_2" in state else
                 unobserved("both_readbacks_not_reached"),
        "inputs_unchanged": observed({
            "equal": state["inputs_before"] == state["inputs_after"],
            "before_sha256": digest(canonical(state["inputs_before"])),
            "after_sha256": digest(canonical(state["inputs_after"])),
        }) if "inputs_before" in state and "inputs_after" in state else
                 unobserved("both_input_snapshots_not_reached"),
        "exception_stage": observed(failure["stage"] if failure else None),
        "exception_class": observed(failure["class"] if failure else None),
        "exception_message": observed(failure["message"] if failure else None),
        "terminal_outcome": observed("raised" if failure else "returned"),
    }
    return {"id": scenario["id"], "requirements": scenario["requirements"],
            "trace": trace, "dimensions": dimensions,
            "terminal": {"outcome": "raised" if failure else "returned",
                         "exception": failure}}

def incompatible_scenario(scenario, trace_names, Tensor, dtypes, mutation):
    state = {}
    trace = []
    failure = None
    for stage in trace_names:
        if stage == "record_terminal_stage_class_message":
            trace.append({"stage": stage, "status": "completed"})
            continue
        if failure is not None:
            trace.append({"stage": stage, "status": "skipped",
                          "reason": "prior_stage_raised"})
            continue
        try:
            if stage == "construct_left":
                state["left"] = construct(scenario["left"], Tensor, dtypes)
            elif stage == "construct_right":
                state["right"] = construct(scenario["right"], Tensor, dtypes)
            elif stage == "invoke_add":
                state["result"] = invoke_add_adapter(
                    state, scenario, Tensor, dtypes, mutation
                )
            elif stage == "observe_shape_if_constructed":
                state["shape"] = [int(value) for value in state["result"].shape]
            else:
                raise RuntimeError("unknown incompatible trace stage: " + stage)
            trace.append({"stage": stage, "status": "completed"})
        except BaseException as exc:
            failure = {"stage": stage, **exception_record(exc)}
            trace.append({"stage": stage, "status": "raised",
                          "exception_class": failure["class"],
                          "exception_message": failure["message"]})
    dimensions = {
        "construction": observed({
            "left": "left" in state, "right": "right" in state,
        }),
        "shape": observed(state["shape"]) if "shape" in state else
                 unobserved("result_shape_not_available"),
        "exception_stage": observed(failure["stage"] if failure else None),
        "exception_class": observed(failure["class"] if failure else None),
        "exception_message": observed(failure["message"] if failure else None),
        "terminal_outcome": observed("raised" if failure else "returned"),
    }
    return {"id": scenario["id"], "requirements": scenario["requirements"],
            "trace": trace, "dimensions": dimensions,
            "terminal": {"outcome": "raised" if failure else "returned",
                         "exception": failure}}

def relative_origin(module, root, token):
    raw = getattr(module, "__file__", None)
    if raw is None:
        return {"kind": "namespace", "owner": token, "path": None}
    path = Path(raw).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return {"kind": "file", "owner": "outside", "path": path.name}
    return {"kind": "file", "owner": token,
            "path": token + "/" + relative.as_posix()}

def ownership(mode, tinygrad, Tensor, expected_upstream, expected_shim,
              expected_product):
    package_root = expected_upstream if mode == "upstream" else expected_shim
    package_token = "<upstream>" if mode == "upstream" else "<shim>"
    tensor_module = importlib.import_module(Tensor.__module__)
    loaded = []
    contaminated = []
    for name in sorted(name for name in sys.modules
                       if name == "tinygrad" or name.startswith("tinygrad.")):
        module = sys.modules[name]
        origin = relative_origin(module, package_root, package_token)
        loaded.append({"module": name, **origin})
        if origin["owner"] != package_token:
            contaminated.append(name)
    package_paths = []
    for raw in getattr(tinygrad, "__path__", ()):
        path = Path(raw).resolve()
        try:
            relative = path.relative_to(package_root)
            package_paths.append(package_token + "/" + relative.as_posix())
        except ValueError:
            package_paths.append("<outside>/" + path.name)
    result = {
        "package": relative_origin(tinygrad, package_root, package_token),
        "package_paths": package_paths,
        "tensor_module": relative_origin(
            tensor_module,
            expected_product if mode == "tgrad" else expected_upstream,
            "<product>" if mode == "tgrad" else "<upstream>",
        ),
        "loaded_tinygrad_modules": loaded,
        "contaminated_modules": contaminated,
        "fallback_detected": bool(contaminated),
        "strict_finder_active": any(
            type(finder).__name__ == "_StrictTinygradFinder" or
            getattr(finder, "__name__", "") == "_StrictTinygradFinder"
            for finder in sys.meta_path
        ),
    }
    if mode == "tgrad":
        tgrad = importlib.import_module("tgrad")
        result["product_module"] = relative_origin(
            tgrad, expected_product, "<product>"
        )
        result["tensor_class_identity"] = Tensor is tgrad.Tensor
    else:
        result["product_module"] = None
        result["tensor_class_identity"] = None
    return result

def main():
    mode = os.environ["SUBJECT_MODE"]
    manifest = json.loads(Path(os.environ["MANIFEST_PATH"]).read_text())
    mutation_raw = os.environ.get("MUTATION_CONFIG_JSON", "null")
    mutation = json.loads(mutation_raw)
    if mutation is not None:
        if mutation.get("id") not in MUTATION_FUNCTIONS:
            raise RuntimeError("unknown immutable mutation mode")
        if not isinstance(mutation.get("target_scenarios"), list):
            raise RuntimeError("mutation mode lacks target scenarios")
    expected_upstream = Path(os.environ["EXPECTED_UPSTREAM"]).resolve()
    expected_shim = Path(os.environ.get("EXPECTED_SHIM", expected_upstream)).resolve()
    expected_product = Path(os.environ.get("EXPECTED_PRODUCT", expected_upstream)).resolve()
    if mode == "tgrad":
        run_pytest = importlib.import_module("run_pytest")
        tinygrad, _ = run_pytest.activate()
    elif mode == "upstream":
        tinygrad = importlib.import_module("tinygrad")
    else:
        raise RuntimeError("unknown subject mode")
    Tensor, dtypes = tinygrad.Tensor, tinygrad.dtypes
    scenarios = []
    for scenario in manifest["scenarios"]:
        if scenario["id"] == "ADD-INCOMPATIBLE-SHAPES":
            result = incompatible_scenario(
                scenario, manifest["profile"]["incompatible_trace"], Tensor,
                dtypes, mutation
            )
        else:
            result = legal_scenario(
                scenario, manifest["profile"]["legal_trace"], Tensor, dtypes,
                mutation
            )
        scenarios.append(result)
    owner = ownership(
        mode, tinygrad, Tensor, expected_upstream, expected_shim, expected_product
    )
    document = {
        "schema_version": SCHEMA,
        "protocol_token": os.environ["PROTOCOL_TOKEN"],
        "subject_mode": mode,
        "status": "complete",
        "backend": str(tinygrad.Device.DEFAULT),
        "scenario_manifest_sha256": os.environ["SCENARIO_MANIFEST_SHA256"],
        "mutation": mutation,
        "ownership": owner,
        "scenarios": scenarios,
    }
    sys.stdout.buffer.write(canonical(document) + b"\n")

try:
    main()
except BaseException as exc:
    failure = {
        "schema_version": SCHEMA,
        "protocol_token": os.environ.get("PROTOCOL_TOKEN", "missing"),
        "subject_mode": os.environ.get("SUBJECT_MODE", "missing"),
        "status": "probe_error",
        "error": {"class": type(exc).__name__, "message": str(exc)},
    }
    sys.stdout.buffer.write(canonical(failure) + b"\n")
    raise SystemExit(86)
'''.strip()


def protocol_token(manifest: dict, mutation: dict | None = None) -> str:
    return digest(canonical({
        "probe_sha256": digest(PROBE_SOURCE.encode()),
        "effective_manifest_sha256": manifest["effective_manifest_sha256"],
        "lock_sha256": EXPECTED_V4_AMENDMENT_SHA256,
        "mutation_configuration_sha256": (
            mutation.get("configuration_sha256") if mutation else None
        ),
    }))


def expected_dimensions(scenario_id: str) -> tuple[str, ...]:
    return (
        INCOMPATIBLE_DIMENSIONS
        if scenario_id == "ADD-INCOMPATIBLE-SHAPES"
        else LEGAL_DIMENSIONS
    )


def validate_ownership(ownership_doc: dict, mode: str) -> None:
    expected_owner = "<upstream>" if mode == "upstream" else "<shim>"
    if ownership_doc.get("package", {}).get("owner") != expected_owner:
        raise RuntimeError("subject package is owned by an unexpected provider")
    if ownership_doc.get("fallback_detected") is not False:
        raise RuntimeError("upstream fallback contamination detected")
    if ownership_doc.get("contaminated_modules") != []:
        raise RuntimeError("one or more tinygrad modules escaped the subject boundary")
    loaded = ownership_doc.get("loaded_tinygrad_modules")
    if not isinstance(loaded, list) or not loaded:
        raise RuntimeError("module ownership inventory is absent")
    if any(item.get("owner") != expected_owner for item in loaded):
        raise RuntimeError("loaded tinygrad module has the wrong owner")
    package_paths = ownership_doc.get("package_paths")
    if not isinstance(package_paths, list) or not package_paths:
        raise RuntimeError("subject package path inventory is absent")
    if any(not path.startswith(expected_owner + "/") for path in package_paths):
        raise RuntimeError("subject package path allows an external provider")
    if mode == "tgrad":
        if ownership_doc.get("strict_finder_active") is not True:
            raise RuntimeError("strict Tgrad finder is not active")
        if ownership_doc.get("tensor_class_identity") is not True:
            raise RuntimeError("tinygrad.Tensor is not the Tgrad Tensor")
        if ownership_doc.get("product_module", {}).get("owner") != "<product>":
            raise RuntimeError("Tgrad product module has the wrong owner")
        if ownership_doc.get("tensor_module", {}).get("owner") != "<product>":
            raise RuntimeError("Tgrad Tensor implementation escaped the product tree")
    else:
        if ownership_doc.get("tensor_module", {}).get("owner") != "<upstream>":
            raise RuntimeError("upstream Tensor implementation has the wrong owner")


def validate_probe_protocol(document: dict, manifest: dict, mode: str,
                            mutation: dict | None = None) -> None:
    required_top = {
        "schema_version", "protocol_token", "subject_mode", "status",
        "backend", "scenario_manifest_sha256", "mutation", "ownership",
        "scenarios",
    }
    if set(document) != required_top:
        raise RuntimeError("probe protocol top-level shape changed")
    if document.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise RuntimeError("unsupported probe protocol")
    if document.get("protocol_token") != protocol_token(manifest, mutation):
        raise RuntimeError("probe token does not bind the V2 verifier")
    if document.get("mutation") != mutation:
        raise RuntimeError("probe mutation configuration differs from the request")
    if document.get("subject_mode") != mode or document.get("status") != "complete":
        raise RuntimeError("probe did not complete for the requested subject")
    if document.get("backend") != "METAL":
        raise RuntimeError("probe did not execute at the frozen METAL boundary")
    if document.get("scenario_manifest_sha256") != manifest[
        "effective_manifest_sha256"
    ]:
        raise RuntimeError("probe used a different scenario manifest")
    validate_ownership(document.get("ownership", {}), mode)
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list):
        raise RuntimeError("probe scenarios are not a list")
    if tuple(item.get("id") for item in scenarios) != EXPECTED_SCENARIO_IDS:
        raise RuntimeError("probe scenario set or order changed")
    manifest_scenarios = {item["id"]: item for item in manifest["scenarios"]}
    for result in scenarios:
        if set(result) != {"id", "requirements", "trace", "dimensions", "terminal"}:
            raise RuntimeError(f"scenario protocol shape changed: {result.get('id')}")
        scenario = manifest_scenarios[result["id"]]
        if result["requirements"] != scenario["requirements"]:
            raise RuntimeError(f"scenario requirements changed: {result['id']}")
        expected_trace = (
            manifest["profile"]["incompatible_trace"]
            if result["id"] == "ADD-INCOMPATIBLE-SHAPES"
            else manifest["profile"]["legal_trace"]
        )
        trace = result.get("trace")
        if not isinstance(trace, list) or [item.get("stage") for item in trace] != expected_trace:
            raise RuntimeError(f"ordered trace changed: {result['id']}")
        if any(item.get("status") not in {"completed", "raised", "skipped"}
               for item in trace):
            raise RuntimeError(f"invalid trace event status: {result['id']}")
        raised = [item for item in trace if item["status"] == "raised"]
        if len(raised) > 1:
            raise RuntimeError(f"multiple trace stages raised: {result['id']}")
        final_record = "record_terminal_stage_class_message"
        raised_seen = False
        for event in trace:
            if event["status"] == "raised":
                raised_seen = True
            elif raised_seen and event["stage"] != final_record and \
                    event["status"] != "skipped":
                raise RuntimeError(f"trace continued after exception: {result['id']}")
            elif not raised_seen and event["status"] == "skipped":
                raise RuntimeError(f"trace skipped before exception: {result['id']}")
        final_events = [event for event in trace if event["stage"] == final_record]
        if final_events and final_events[0]["status"] != "completed":
            raise RuntimeError(f"terminal recording stage did not complete: {result['id']}")
        dimensions = result.get("dimensions")
        if not isinstance(dimensions, dict) or tuple(sorted(dimensions)) != tuple(
            sorted(expected_dimensions(result["id"]))
        ):
            raise RuntimeError(f"dimension inventory changed: {result['id']}")
        for name, value in dimensions.items():
            if not isinstance(value, dict) or value.get("state") not in {
                "observed", "unobserved"
            }:
                raise RuntimeError(f"malformed dimension {result['id']}:{name}")
            if value["state"] == "observed" and set(value) != {"state", "value"}:
                raise RuntimeError(f"observed dimension carries non-protocol fields: {name}")
            if value["state"] == "unobserved" and set(value) != {"state", "reason"}:
                raise RuntimeError(f"unobserved dimension lacks one exact reason: {name}")
        terminal = result.get("terminal")
        if not isinstance(terminal, dict) or set(terminal) != {"outcome", "exception"}:
            raise RuntimeError(f"malformed terminal event: {result['id']}")
        if terminal["outcome"] not in {"returned", "raised"}:
            raise RuntimeError(f"unknown terminal outcome: {result['id']}")
        terminal_dimension = dimensions["terminal_outcome"]
        if not is_observed(terminal_dimension) or terminal_dimension["value"] != terminal["outcome"]:
            raise RuntimeError(f"terminal event/dimension disagreement: {result['id']}")
        if terminal["outcome"] == "raised" and terminal["exception"] is None:
            raise RuntimeError(f"raised terminal lacks exception: {result['id']}")
        if terminal["outcome"] == "returned" and terminal["exception"] is not None:
            raise RuntimeError(f"returned terminal carries exception: {result['id']}")
        exception_dimensions = {
            "stage": dimensions["exception_stage"],
            "class": dimensions["exception_class"],
            "message": dimensions["exception_message"],
        }
        if any(not is_observed(value) for value in exception_dimensions.values()):
            raise RuntimeError(f"terminal exception facts are unobserved: {result['id']}")
        dimension_exception = {
            name: value["value"] for name, value in exception_dimensions.items()
        }
        if terminal["outcome"] == "returned":
            if raised or any(value is not None for value in dimension_exception.values()):
                raise RuntimeError(f"returned terminal disagrees with exception facts: {result['id']}")
        else:
            exception = terminal["exception"]
            if set(exception) != {"stage", "class", "message"} or \
                    exception != dimension_exception:
                raise RuntimeError(f"terminal exception/dimension disagreement: {result['id']}")
            if len(raised) != 1 or raised[0]["stage"] != exception["stage"] or \
                    raised[0].get("exception_class") != exception["class"] or \
                    raised[0].get("exception_message") != exception["message"]:
                raise RuntimeError(f"trace exception/terminal disagreement: {result['id']}")


MUTATION_BRANCH_FUNCTIONS = {
    "MUT-ADD-SUBTRACT": "mut_add_subtract",
    "MUT-ADD-WRONG-RIGHT-ALIGNMENT": "mut_wrong_right_alignment",
    "MUT-ADD-NO-SINGLETON-EXPANSION": "mut_no_singleton_expansion",
    "MUT-ADD-WRONG-PROMOTION": "mut_wrong_promotion",
    "MUT-ADD-ACCEPT-INCOMPATIBLE": "mut_accept_incompatible",
    "MUT-REALIZE-RETURNS-COPY": "mut_realize_copy",
    "MUT-REALIZE-SECOND-READBACK": "mut_second_readback",
    "MUT-ADD-MUTATES-LEFT": "mut_mutate_left",
}

MUTATION_DIMENSION_FOOTPRINTS: dict[str, frozenset[str]] = {
    "MUT-ADD-SUBTRACT": frozenset({"value", "repeated_readback"}),
    "MUT-ADD-WRONG-RIGHT-ALIGNMENT": frozenset({
        "shape", "dtype", "value", "realize_identity", "repeated_readback",
        "inputs_unchanged", "exception_stage", "exception_class",
        "exception_message", "terminal_outcome",
    }),
    "MUT-ADD-NO-SINGLETON-EXPANSION": frozenset({
        "shape", "dtype", "value", "realize_identity", "repeated_readback",
        "inputs_unchanged", "exception_stage", "exception_class",
        "exception_message", "terminal_outcome",
    }),
    "MUT-ADD-WRONG-PROMOTION": frozenset({
        "dtype", "value", "repeated_readback",
    }),
    "MUT-ADD-ACCEPT-INCOMPATIBLE": frozenset({
        "shape", "exception_stage", "exception_class",
        "exception_message", "terminal_outcome",
    }),
    "MUT-REALIZE-RETURNS-COPY": frozenset({"realize_identity"}),
    "MUT-REALIZE-SECOND-READBACK": frozenset({"repeated_readback"}),
    "MUT-ADD-MUTATES-LEFT": frozenset({"inputs_unchanged"}),
}

V2_MUTATION_TRACE_FOOTPRINTS: dict[str, frozenset[str]] = {
    "MUT-ADD-SUBTRACT": frozenset(),
    "MUT-ADD-WRONG-RIGHT-ALIGNMENT": frozenset({
        "invoke_add", "capture_result_identity", "observe_shape", "observe_dtype",
        "realize_1", "readback_1", "realize_2", "readback_2",
        "snapshot_inputs_after",
    }),
    "MUT-ADD-NO-SINGLETON-EXPANSION": frozenset({
        "invoke_add", "capture_result_identity", "observe_shape", "observe_dtype",
        "realize_1", "readback_1", "realize_2", "readback_2",
        "snapshot_inputs_after",
    }),
    "MUT-ADD-WRONG-PROMOTION": frozenset(),
    "MUT-ADD-ACCEPT-INCOMPATIBLE": frozenset({
        "invoke_add", "observe_shape_if_constructed",
    }),
    "MUT-REALIZE-RETURNS-COPY": frozenset(),
    "MUT-REALIZE-SECOND-READBACK": frozenset(),
    "MUT-ADD-MUTATES-LEFT": frozenset(),
}


def mutation_trace_footprints() -> dict[str, frozenset[str]]:
    result = dict(V2_MUTATION_TRACE_FOOTPRINTS)
    amendment = json.loads(V3_AMENDMENT.read_text(encoding="utf-8"))
    for item in amendment["trace_footprint_amendments"]:
        mutation_id = item["mutation_id"]
        if result.get(mutation_id) != frozenset(item["from"]):
            raise RuntimeError(f"V3 trace amendment source drifted: {mutation_id}")
        result[mutation_id] = frozenset(item["to"])
    return result


MUTATION_TRACE_FOOTPRINTS = mutation_trace_footprints()


def changed_dimensions(before: dict, after: dict) -> frozenset[str]:
    if set(before) != set(after):
        raise RuntimeError("mutant changed the dimension inventory")
    return frozenset(name for name in before if before[name] != after[name])


def dimension_relation(reference: dict, candidate: dict) -> str:
    if not is_observed(reference) or not is_observed(candidate):
        return "unobserved"
    return "same" if reference == candidate else "different"


def compare_scenarios(reference: list[dict], candidate: list[dict]) -> list[dict]:
    if [item["id"] for item in reference] != [item["id"] for item in candidate]:
        raise RuntimeError("cannot compare different scenario manifests")
    comparisons = []
    for expected, actual in zip(reference, candidate):
        if set(expected["dimensions"]) != set(actual["dimensions"]):
            raise RuntimeError(f"dimension inventory differs: {expected['id']}")
        dimensions = {
            name: dimension_relation(expected["dimensions"][name],
                                     actual["dimensions"][name])
            for name in sorted(expected["dimensions"])
        }
        comparisons.append({
            "scenario_id": expected["id"],
            "dimensions": dimensions,
            "counts": {
                state: sum(value == state for value in dimensions.values())
                for state in ("same", "different", "unobserved")
            },
        })
    return comparisons


def artifact_ref(kind: str, raw: bytes) -> dict:
    content_hash = digest(raw)
    return {
        "kind": kind, "sha256": content_hash, "bytes": len(raw),
        "path": f"artifacts/{kind}/{content_hash}",
    }


def probe_function_source(function_name: str) -> bytes:
    tree = ast.parse(PROBE_SOURCE)
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
        node.name == function_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"probe mutation function is not unique: {function_name}")
    source = ast.get_source_segment(PROBE_SOURCE, matches[0])
    if not source:
        raise RuntimeError(f"cannot recover probe mutation source: {function_name}")
    return (source.rstrip() + "\n").encode("utf-8")


def mutation_config(declaration: dict, verifier_hash: str) -> tuple[dict, bytes]:
    mutation_id = declaration["id"]
    function_name = MUTATION_BRANCH_FUNCTIONS[mutation_id]
    source = probe_function_source(function_name)
    config = {
        "id": mutation_id,
        "target_scenarios": declaration["target_scenarios"],
        "branch_function": function_name,
        "branch_source_sha256": digest(source),
        "common_probe_sha256": digest(PROBE_SOURCE.encode()),
        "verifier_sha256": verifier_hash,
        "declaration_sha256": digest(canonical(declaration)),
        "dimension_footprint": sorted(MUTATION_DIMENSION_FOOTPRINTS[mutation_id]),
        "trace_footprint": sorted(MUTATION_TRACE_FOOTPRINTS[mutation_id]),
    }
    config["executable_tree_sha256"] = digest(canonical({
        "common_probe_sha256": config["common_probe_sha256"],
        "branch_source_sha256": config["branch_source_sha256"],
        "declaration_sha256": config["declaration_sha256"],
    }))
    config["configuration_sha256"] = digest(canonical(config))
    return config, source


def changed_trace_stages(before: list[dict], after: list[dict]) -> frozenset[str]:
    if [item.get("stage") for item in before] != [item.get("stage") for item in after]:
        raise RuntimeError("mutant changed the ordered trace stage inventory")
    return frozenset(
        left["stage"] for left, right in zip(before, after) if left != right
    )


def evaluate_mutant_child(declaration: dict, baseline_protocol: dict,
                          child_protocol: dict, config: dict,
                          child_artifacts: dict) -> dict:
    mutation_id = declaration["id"]
    if child_protocol.get("mutation") != config:
        raise RuntimeError(f"child protocol/config disagreement: {mutation_id}")
    baseline_by_id = {item["id"]: item for item in baseline_protocol["scenarios"]}
    child_by_id = {item["id"]: item for item in child_protocol["scenarios"]}
    if tuple(baseline_by_id) != tuple(child_by_id):
        raise RuntimeError(f"child scenario order changed: {mutation_id}")
    evaluations = []
    target_ids = set(declaration["target_scenarios"])
    for scenario_id in baseline_by_id:
        baseline = baseline_by_id[scenario_id]
        child = child_by_id[scenario_id]
        if scenario_id not in target_ids:
            unchanged = baseline == child
            evaluations.append({
                "scenario_id": scenario_id,
                "targeted": False,
                "unchanged": unchanged,
                "scenario_sha256": digest(canonical(child)),
            })
            continue
        dimension_changes = changed_dimensions(
            baseline["dimensions"], child["dimensions"]
        )
        trace_changes = changed_trace_stages(baseline["trace"], child["trace"])
        must_fail = {
            name: dimension_relation(
                baseline["dimensions"][name], child["dimensions"][name]
            )
            for name in declaration["must_fail"]
        }
        must_not_change = {
            name: dimension_relation(
                baseline["dimensions"][name], child["dimensions"][name]
            )
            for name in declaration["must_not_change"]
        }
        may_be_unobserved = {
            name: dimension_relation(
                baseline["dimensions"][name], child["dimensions"][name]
            )
            for name in declaration["may_be_unobserved"]
        }
        evaluations.append({
            "scenario_id": scenario_id,
            "targeted": True,
            "changed_dimensions": sorted(dimension_changes),
            "changed_trace_stages": sorted(trace_changes),
            "must_fail": must_fail,
            "must_not_change": must_not_change,
            "may_be_unobserved": may_be_unobserved,
            "required_fault_detected": all(
                value == "different" for value in must_fail.values()
            ),
            "required_preservation_held": all(
                value == "same" for value in must_not_change.values()
            ),
            "atomic_dimension_footprint_exact": (
                dimension_changes == MUTATION_DIMENSION_FOOTPRINTS[mutation_id]
            ),
            "atomic_trace_footprint_exact": (
                trace_changes == MUTATION_TRACE_FOOTPRINTS[mutation_id]
            ),
            "scenario_sha256": digest(canonical(child)),
        })
    targeted = [item for item in evaluations if item["targeted"]]
    untouched = [item for item in evaluations if not item["targeted"]]
    accepted = (
        len(targeted) == len(target_ids) and
        all(item["required_fault_detected"] for item in targeted) and
        all(item["required_preservation_held"] for item in targeted) and
        all(item["atomic_dimension_footprint_exact"] for item in targeted) and
        all(item["atomic_trace_footprint_exact"] for item in targeted) and
        all(item["unchanged"] for item in untouched)
    )
    return {
        "mutation_id": mutation_id,
        "configuration": config,
        "branch_source": child_artifacts["branch_source"],
        "child_probe": child_artifacts["probe"],
        "expected": {
            "must_fail": declaration["must_fail"],
            "must_not_change": declaration["must_not_change"],
            "may_be_unobserved": declaration["may_be_unobserved"],
            "dimension_footprint": sorted(MUTATION_DIMENSION_FOOTPRINTS[mutation_id]),
            "trace_footprint": sorted(MUTATION_TRACE_FOOTPRINTS[mutation_id]),
        },
        "evaluations": evaluations,
        "outcome": (
            "validator_rejected_mutant" if accepted
            else "mutant_survived_or_calibration_changed"
        ),
    }


def normalize_process_stream(raw: bytes, replacements: dict[Path, str]) -> bytes:
    text = raw.decode("utf-8", errors="replace")
    aliases = {
        spelling: token
        for path, token in replacements.items()
        for spelling in {str(path), str(path.absolute()), str(path.resolve())}
    }
    for spelling, token in sorted(
        aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = text.replace(spelling, token)
    return text.encode("utf-8")


def interpreter_facts(py: Path) -> dict:
    """Match the environment fact projection frozen by the V2 lock."""
    source = r'''
import importlib.metadata
import hashlib
import json
import platform
import sys
selected = {"pytest", "numpy", "torch", "hypothesis", "einops"}
all_distributions = sorted(
    (dist.metadata.get("Name", "").lower(), dist.version)
    for dist in importlib.metadata.distributions() if dist.metadata.get("Name")
)
records, files_by_name = {}, {}
for name in selected:
    try:
        dist = importlib.metadata.distribution(name)
        records[name] = hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest()
        files = []
        for entry in sorted(dist.files or [], key=str):
            if getattr(entry, "hash", None) is None: continue
            path = dist.locate_file(entry)
            if path.is_file():
                try: files.append((str(entry), hashlib.sha256(path.read_bytes()).hexdigest()))
                except OSError: files.append((str(entry), "<unreadable>"))
        files_by_name[name] = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    except importlib.metadata.PackageNotFoundError: pass
print(json.dumps({
    "python": platform.python_version(),
    "implementation": platform.python_implementation(),
    "cache_tag": sys.implementation.cache_tag,
    "platform": platform.system(), "platform_release": platform.release(),
    "machine": platform.machine(), "runtime_executable": sys.executable,
    "selected_dependencies": {name: version for name, version in all_distributions if name in selected},
    "selected_dependency_record_sha256": records,
    "selected_dependency_files_sha256": files_by_name,
    "distribution_manifest": all_distributions,
}, sort_keys=True))
'''.strip()
    completed = subprocess.run(
        [str(py), "-c", source],
        env={"LANG": "C", "LC_ALL": "C", "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True, timeout=30, check=True,
    )
    facts = json.loads(completed.stdout)
    distributions = facts.pop("distribution_manifest")
    facts["distribution_manifest_sha256"] = digest(canonical(distributions))
    facts["launcher"] = str(py.absolute())
    facts["launcher_target"] = str(py.resolve())
    facts["launcher_target_sha256"] = file_hash(py.resolve())
    return facts


def hardware_facts() -> dict:
    completed = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-json"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    document = json.loads(completed.stdout)
    entries = document.get("SPHardwareDataType", [])
    first = entries[0] if entries else {}
    safe = {
        "machine_model": first.get("machine_model"),
        "machine_name": first.get("machine_name"),
        "chip_type": first.get("chip_type"),
    }
    return {**safe, "identity_sha256": digest(canonical(safe))}


def environment_identity(py: Path, lock: dict) -> dict:
    facts = interpreter_facts(py)
    facts_hash = digest(canonical(facts))
    hardware = hardware_facts()
    boundary = lock["execution_boundary"]
    if facts_hash != boundary["python_environment_facts_sha256"]:
        raise RuntimeError("Python environment differs from the V2 lock")
    if hardware["identity_sha256"] != boundary["hardware_identity_sha256"]:
        raise RuntimeError("hardware differs from the V2 lock")
    result = {
        "facts": facts,
        "facts_sha256": facts_hash,
        "hardware": hardware,
        "backend": boundary["backend"],
        "policy": {
            "LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
            "isolated_home": True, "isolated_tmp": True,
            "strict_no_upstream_fallback": True,
        },
    }
    result["sha256"] = digest(canonical(result))
    return result


def extract_upstream(checkout: Path, revision: str, destination: Path) -> dict:
    resolved = git_text("rev-parse", f"{revision}^{{commit}}", repo=checkout)
    if resolved != revision:
        raise RuntimeError("upstream checkout does not contain the exact pin")
    tree = git_text("rev-parse", f"{revision}^{{tree}}", repo=checkout)
    archive = git_bytes("archive", "--format=tar", revision, repo=checkout)
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(destination, filter="data")
    return {
        "revision": revision, "tree": tree,
        "snapshot_content_sha256": directory_content_hash(destination),
    }


def product_source_identity() -> dict:
    status = git_text(
        "status", "--porcelain", "--", "Tgrad", "python", "c",
        "scripts/parity/shim",
    )
    if status:
        raise RuntimeError("product or strict adapter is dirty; refusing observation")
    revision = git_text("rev-parse", "HEAD")
    components = {
        root: git_directory_hash(revision, root)
        for root in ("Tgrad", "python", "c", "scripts/parity/shim")
    }
    return {
        "revision": revision,
        "tree": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "component_sha256": components,
        "combined_sha256": digest(canonical(components)),
    }


def copy_tgrad_snapshot(root: Path) -> tuple[dict[str, Path], dict]:
    source = product_source_identity()
    shim = root / "shim"
    product = root / "python"
    runtime = root / "runtime"
    shutil.copytree(SHIM_ROOT, shim, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(PRODUCT_PYTHON, product,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runtime.mkdir()
    runtime_artifacts = {}
    for name in RUNTIME_NAMES:
        original = RUNTIME_DIR / name
        if not original.is_file():
            raise RuntimeError(f"missing Tgrad runtime artifact: {original}")
        target = runtime / name
        shutil.copyfile(original, target)
        runtime_artifacts[name] = file_hash(target)
    execution = {
        "shim_content_sha256": directory_content_hash(shim),
        "product_python_content_sha256": directory_content_hash(product),
        "runtime_artifacts": runtime_artifacts,
        "runtime_artifact_sha256": digest(canonical(runtime_artifacts)),
    }
    return {
        "shim": shim, "product": product, "runtime": runtime,
        "runtime_library": runtime / RUNTIME_NAMES[0],
    }, {"source": source, "execution": execution}


def controlled_environment(mode: str, root: Path, upstream: Path,
                           manifest_path: Path, manifest_sha256: str, token: str,
                           tgrad: dict[str, Path] | None,
                           mutation: dict | None) -> dict[str, str]:
    home = root / "home"
    temporary = root / "tmp"
    home.mkdir()
    temporary.mkdir()
    env = {
        "HOME": str(home), "TMPDIR": str(temporary),
        "LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
        "DEV": "METAL", "SUBJECT_MODE": mode,
        "MANIFEST_PATH": str(manifest_path), "PROTOCOL_TOKEN": token,
        "SCENARIO_MANIFEST_SHA256": manifest_sha256,
        "EXPECTED_UPSTREAM": str(upstream),
        "MUTATION_CONFIG_JSON": canonical(mutation).decode("ascii"),
    }
    if mode == "upstream":
        env["PYTHONPATH"] = str(upstream)
    else:
        if tgrad is None:
            raise RuntimeError("Tgrad execution snapshot is absent")
        env.update({
            "PYTHONPATH": os.pathsep.join([
                str(tgrad["shim"]), str(tgrad["product"]), str(upstream),
            ]),
            "TGRAD_ROOT": str(root), "TGRAD_LIB": str(tgrad["runtime_library"]),
            "EXPECTED_SHIM": str(tgrad["shim"] / "tinygrad"),
            "EXPECTED_PRODUCT": str(tgrad["product"]),
        })
    return env


def run_probe(py: Path, mode: str, root: Path, upstream: Path,
              manifest: dict, tgrad: dict[str, Path] | None,
              mutation: dict | None = None) -> tuple[dict, dict, dict[str, bytes]]:
    manifest_path = root / "effective_manifest.json"
    manifest_path.write_bytes(canonical(manifest))
    env = controlled_environment(
        mode, root, upstream, manifest_path, manifest["effective_manifest_sha256"],
        protocol_token(manifest, mutation), tgrad, mutation
    )
    completed = subprocess.run(
        [str(py), "-c", PROBE_SOURCE], cwd=upstream, env=env,
        capture_output=True, timeout=180,
    )
    replacements = {root: "<execution-root>", upstream: "<upstream>"}
    if tgrad:
        replacements.update({
            tgrad["shim"]: "<shim>", tgrad["product"]: "<product>",
            tgrad["runtime"]: "<runtime>",
        })
    raw_stdout = completed.stdout
    raw_stderr = completed.stderr
    stdout = normalize_process_stream(raw_stdout, replacements)
    stderr = normalize_process_stream(raw_stderr, replacements)
    if completed.returncode != 0:
        raise RuntimeError(
            f"probe failed closed with rc={completed.returncode}, "
            f"stdout_sha256={digest(stdout)}, stderr_sha256={digest(stderr)}"
        )
    if raw_stderr:
        raise RuntimeError(
            "successful probe emitted stderr; refusing ambiguous evidence: "
            f"raw_sha256={digest(raw_stderr)}"
        )
    if stdout != raw_stdout or stderr != raw_stderr:
        raise RuntimeError(
            "successful probe leaked an execution-root path; refusing "
            "nondeterministic literal streams"
        )
    # Successful probe output is itself canonical and path-normalized by the
    # fixed child protocol. Parse the literal stream; outer normalization is a
    # separately retained diagnostic projection, never a replacement for raw
    # process evidence.
    lines = raw_stdout.splitlines(keepends=True)
    if len(lines) != 1 or not lines[0].endswith(b"\n"):
        raise RuntimeError("probe protocol must be exactly one newline-terminated record")
    try:
        document = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("probe protocol is not JSON") from exc
    if canonical(document) + b"\n" != raw_stdout:
        raise RuntimeError("probe protocol is not canonical JSON")
    validate_probe_protocol(document, manifest, mode, mutation)
    protocol_raw = canonical(document)
    payloads = {
        artifact_ref("probe-stdout-raw", raw_stdout)["path"]: raw_stdout,
        artifact_ref("probe-stderr-raw", raw_stderr)["path"]: raw_stderr,
        artifact_ref("probe-stdout-normalized", stdout)["path"]: stdout,
        artifact_ref("probe-stderr-normalized", stderr)["path"]: stderr,
        artifact_ref("probe-protocol", protocol_raw)["path"]: protocol_raw,
    }
    refs = {ref["kind"]: ref for ref in refs_for_payloads(payloads)}
    return document, {"protocol": document, "artifacts": refs}, payloads


def refs_for_payloads(payloads: dict[str, bytes]) -> list[dict]:
    refs = []
    for path, raw in sorted(payloads.items()):
        parts = Path(path).parts
        if len(parts) != 3 or parts[0] != "artifacts":
            raise RuntimeError(f"invalid artifact path: {path}")
        ref = artifact_ref(parts[1], raw)
        if ref["path"] != path:
            raise RuntimeError(f"artifact path/content disagreement: {path}")
        refs.append(ref)
    return refs


def read_bound_artifact(evidence_path: Path, ref: dict) -> bytes:
    relative = Path(ref.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("artifact reference escapes the evidence directory")
    root = evidence_path.resolve().parent
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("artifact reference escapes the evidence directory") from exc
    if not target.is_file():
        raise RuntimeError(f"bound artifact is missing: {relative}")
    raw = target.read_bytes()
    if digest(raw) != ref.get("sha256") or len(raw) != ref.get("bytes"):
        raise RuntimeError(f"bound artifact identity mismatch: {relative}")
    return raw


def computed_evidence_id(document: dict) -> str:
    payload = copy.deepcopy(document)
    payload.pop("evidence_id", None)
    return digest(canonical(payload))


def replay_probe_record(evidence_path: Path, record: dict, manifest: dict,
                        mode: str, mutation: dict | None) -> None:
    if set(record) != {"protocol", "artifacts"}:
        raise RuntimeError("probe record shape changed")
    refs = record["artifacts"]
    expected_kinds = {
        "probe-protocol", "probe-stdout-raw", "probe-stderr-raw",
        "probe-stdout-normalized", "probe-stderr-normalized",
    }
    if set(refs) != expected_kinds:
        raise RuntimeError("probe record lacks literal or normalized streams")
    raw = {kind: read_bound_artifact(evidence_path, ref)
           for kind, ref in refs.items()}
    protocol = json.loads(raw["probe-protocol"])
    if protocol != record["protocol"]:
        raise RuntimeError("recorded observation differs from raw probe protocol")
    validate_probe_protocol(protocol, manifest, mode, mutation)
    if raw["probe-stdout-raw"] != canonical(protocol) + b"\n":
        raise RuntimeError("literal probe stdout does not replay the protocol")
    if raw["probe-stderr-raw"] != b"":
        raise RuntimeError("successful replay carries nonempty literal stderr")
    if raw["probe-stdout-normalized"] != raw["probe-stdout-raw"]:
        raise RuntimeError("successful probe stdout required path normalization")
    if raw["probe-stderr-normalized"] != b"":
        raise RuntimeError("normalized successful stderr is nonempty")


def replay_observation(evidence_path: Path, document: dict, manifest: dict) -> None:
    refs = document.get("artifacts")
    if not isinstance(refs, list) or not refs:
        raise RuntimeError("evidence has no raw artifact closure")
    global_refs = {canonical(ref) for ref in refs}
    if len(global_refs) != len(refs):
        raise RuntimeError("global artifact closure contains duplicate references")
    for ref in refs:
        read_bound_artifact(evidence_path, ref)
    referenced = {
        canonical(ref)
        for ref in document["observation"]["artifacts"].values()
    }
    replay_probe_record(
        evidence_path, document["observation"], manifest, document["against"], None
    )
    calibrations = document.get("calibrations")
    if document["against"] == "upstream":
        if not isinstance(calibrations, list) or len(calibrations) != 8:
            raise RuntimeError("upstream evidence lacks eight child probes")
        declarations = {item["id"]: item for item in manifest["mutations"]}
        baseline_protocol = document["observation"]["protocol"]
        for calibration in calibrations:
            mutation_id = calibration.get("mutation_id")
            declaration = declarations.get(mutation_id)
            if declaration is None:
                raise RuntimeError("unknown mutant in evidence")
            replay_probe_record(
                evidence_path, calibration["child_probe"], manifest, "upstream",
                calibration["configuration"],
            )
            referenced.add(canonical(calibration["branch_source"]))
            referenced.update(
                canonical(ref)
                for ref in calibration["child_probe"]["artifacts"].values()
            )
            source = read_bound_artifact(evidence_path, calibration["branch_source"])
            expected_config, expected_source = mutation_config(
                declaration,
                document["identity"]["verifier"]["observer_sha256"],
            )
            if source != expected_source or calibration["configuration"] != expected_config:
                raise RuntimeError("mutant source/configuration identity changed")
            recomputed = evaluate_mutant_child(
                declaration, baseline_protocol,
                calibration["child_probe"]["protocol"], expected_config,
                {"branch_source": calibration["branch_source"],
                 "probe": calibration["child_probe"]},
            )
            if recomputed != calibration:
                raise RuntimeError("mutant child calibration cannot be replayed")
    elif calibrations is not None:
        raise RuntimeError("Tgrad observation must reference upstream calibration")
    if referenced != global_refs:
        raise RuntimeError("global artifact list differs from the child artifact closure")


def enforce_baseline_first(against: str, baseline: Path | None) -> None:
    if against == "upstream" and baseline is not None:
        raise RuntimeError("upstream baseline creation forbids --upstream-baseline")
    if against == "tgrad" and baseline is None:
        raise RuntimeError("Tgrad observation requires an upstream baseline first")


def validate_upstream_baseline(
    path: Path, manifest: dict, lock: dict, environment: dict,
    verifier: dict,
) -> dict:
    raw = path.read_bytes()
    document = json.loads(raw)
    expected_bytes = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    if raw != expected_bytes:
        raise RuntimeError("upstream baseline JSON is not deterministic canonical output")
    if set(document) != {
        "schema_version", "result_kind", "against", "identity", "observation",
        "calibrations", "upstream_comparison", "inference_policy",
        "artifacts", "evidence_id",
    }:
        raise RuntimeError("upstream baseline document shape changed")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported upstream baseline schema")
    if document.get("result_kind") != "observation":
        raise RuntimeError("diagnostic artifact is not an upstream baseline")
    if document.get("against") != "upstream":
        raise RuntimeError("baseline is not an upstream observation")
    if document.get("evidence_id") != computed_evidence_id(document):
        raise RuntimeError("upstream baseline evidence id is invalid")
    if document.get("upstream_comparison") is not None:
        raise RuntimeError("upstream baseline cannot compare against itself")
    if document.get("inference_policy") != {
        "kind": "observation_only",
        "never_infer": ["conformant", "promoted", "full_tinygrad_parity"],
        "adequacy": "open",
    }:
        raise RuntimeError("upstream baseline attempted an unauthorized inference")
    identity = document.get("identity", {})
    if identity.get("lock", {}).get("sha256") != EXPECTED_V4_AMENDMENT_SHA256:
        raise RuntimeError("upstream baseline is not bound to V4")
    if identity.get("manifest", {}).get("effective_sha256") != manifest[
        "effective_manifest_sha256"
    ]:
        raise RuntimeError("upstream baseline used another manifest")
    if identity.get("source", {}).get("revision") != lock["upstream_revision"]:
        raise RuntimeError("upstream baseline used another upstream revision")
    if identity.get("verifier") != verifier:
        raise RuntimeError("upstream baseline used another verifier")
    if identity.get("environment") != environment:
        raise RuntimeError("upstream baseline environment/hardware differs")
    replay_observation(path, document, manifest)
    calibrations = document.get("calibrations")
    if not isinstance(calibrations, list) or tuple(
        item.get("mutation_id") for item in calibrations
    ) != EXPECTED_MUTATION_IDS:
        raise RuntimeError("upstream baseline lacks all eight calibrations")
    if any(item.get("outcome") != "validator_rejected_mutant"
           for item in calibrations):
        raise RuntimeError("upstream baseline has an uncalibrated dimension")
    return {
        "path_sha256": digest(raw),
        "evidence_id": document["evidence_id"],
        "source_revision": identity["source"]["revision"],
        "scenario_manifest_sha256": identity["manifest"]["effective_sha256"],
        "protocol_sha256": digest(canonical(document["observation"]["protocol"])),
        "document": document,
    }


def verifier_git_identity() -> dict:
    status = git_text("status", "--porcelain")
    if status:
        raise RuntimeError("verifier repository is not clean")
    revision = git_text("rev-parse", "HEAD")
    tree = git_text("rev-parse", "HEAD^{tree}")
    files = {}
    for path in (
        "scripts/spec/observe_broadcast_add.py",
        "scripts/spec/test_broadcast_add_observer.py",
    ):
        git_bytes("ls-files", "--error-unmatch", path)
        current = REPO / path
        current_hash = file_hash(current)
        if digest(git_bytes("show", f"{revision}:{path}")) != current_hash:
            raise RuntimeError(f"verifier source differs from HEAD: {path}")
        files[path] = current_hash
    identity = {
        "revision": revision, "tree": tree, "clean": True,
        "files": files,
    }
    identity["sha256"] = digest(canonical(identity))
    return identity


def build_identity(lock: dict, amendment: dict, manifest: dict, mode: str,
                   source: dict, runtime: dict, environment: dict) -> dict:
    verifier_hash = file_hash(Path(__file__))
    verifier_git = verifier_git_identity()
    if mode == "tgrad":
        adapter = {
            "kind": "strict_tgrad_substitution",
            "content_sha256": runtime["execution"]["shim_content_sha256"],
            "runner_sha256": file_hash(SHIM_ROOT / "run_pytest.py"),
            "no_fallback_required": True,
        }
    else:
        adapter = {
            "kind": "direct_pinned_upstream_import",
            "content_sha256": digest(canonical({
                "probe_sha256": digest(PROBE_SOURCE.encode()),
                "upstream_revision": lock["upstream_revision"],
            })),
            "runner_sha256": None,
            "no_fallback_required": True,
        }
    return {
        "lock": {
            "trial_id": lock["trial_id"], "sha256": EXPECTED_V4_AMENDMENT_SHA256,
            "definition_revision": lock["definition_revision"],
            "definition_tree": lock["definition_tree"],
            "semantic_lock_sha256": lock["semantic_lock_sha256"],
        },
        "manifest": {
            "packet_id": amendment["v4"]["packet_id"],
            "base_v2_amendment_sha256": amendment["v3"]["base_v2_amendment_sha256"],
            "v3_amendment_sha256": file_hash(V3_AMENDMENT),
            "v4_amendment_sha256": file_hash(V4_AMENDMENT),
            "effective_sha256": manifest["effective_manifest_sha256"],
        },
        "source": source,
        "runtime": runtime,
        "adapter": adapter,
        "verifier": {
            "observer_sha256": verifier_hash,
            "probe_sha256": digest(PROBE_SOURCE.encode()),
            "schema_version": SCHEMA_VERSION,
            "git": verifier_git,
        },
        "environment": environment,
    }


def write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite different artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def persist(output: Path, document: dict, payloads: dict[str, bytes]) -> None:
    output = output.resolve()
    for relative, raw in sorted(payloads.items()):
        write_once(output.parent / relative, raw)
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    write_once(output, raw.encode("ascii"))


def observe(args: argparse.Namespace) -> tuple[dict, dict[str, bytes]]:
    lock, amendment, manifest = validate_v4_definition(args.lock)
    enforce_baseline_first(args.against, args.upstream_baseline)
    # Preserve the launcher spelling. Environment facts bind the venv launcher
    # and its resolved target separately; resolving here collapses those two
    # identities and makes a matching frozen environment appear different.
    py = args.python.expanduser().absolute()
    if not py.is_file():
        raise RuntimeError(f"observer Python does not exist: {py}")
    checkout = args.upstream_checkout.resolve()
    if not (checkout / ".git").exists():
        raise RuntimeError("--upstream-checkout is not a git checkout")
    environment = environment_identity(py, lock)
    verifier_hash = file_hash(Path(__file__))

    with tempfile.TemporaryDirectory(prefix="tgrad_broadcast_add_observer_") as raw_root:
        root = Path(raw_root)
        upstream = root / "upstream"
        upstream_source = extract_upstream(
            checkout, lock["upstream_revision"], upstream
        )
        payloads: dict[str, bytes] = {}
        if args.against == "upstream":
            tgrad_paths = None
            source = upstream_source
            runtime = {
                "kind": "python_upstream_snapshot",
                "python_launcher_sha256": environment["facts"]["launcher_target_sha256"],
                "snapshot_content_sha256": upstream_source[
                    "snapshot_content_sha256"
                ],
            }
        else:
            tgrad_paths, tgrad_identity = copy_tgrad_snapshot(root)
            source = tgrad_identity["source"]
            runtime = {"kind": "tgrad_native_runtime", **tgrad_identity}
        identity = build_identity(
            lock, amendment, manifest, args.against, source, runtime, environment
        )
        baseline = None
        if args.against == "tgrad":
            baseline = validate_upstream_baseline(
                args.upstream_baseline.resolve(), manifest, lock, environment,
                identity["verifier"],
            )
        run_root = root / "runs" / "baseline"
        run_root.mkdir(parents=True)
        protocol, probe_record, probe_payloads = run_probe(
            py, args.against, run_root, upstream, manifest, tgrad_paths, None
        )
        payloads.update(probe_payloads)
        if args.against == "upstream":
            calibrations = []
            for declaration in manifest["mutations"]:
                config, branch_source = mutation_config(
                    declaration, verifier_hash
                )
                branch_ref = artifact_ref("mutant-source", branch_source)
                payloads[branch_ref["path"]] = branch_source
                child_root = root / "runs" / declaration["id"]
                child_root.mkdir(parents=True)
                child_protocol, child_record, child_payloads = run_probe(
                    py, "upstream", child_root, upstream, manifest, None, config
                )
                payloads.update(child_payloads)
                calibrations.append(evaluate_mutant_child(
                    declaration, protocol, child_protocol, config,
                    {"branch_source": branch_ref, "probe": child_record},
                ))
            comparison = None
        else:
            calibrations = None
            comparison = compare_scenarios(
                baseline["document"]["observation"]["protocol"]["scenarios"],
                protocol["scenarios"],
            )
            identity["upstream_baseline"] = {
                key: value for key, value in baseline.items() if key != "document"
            }
        document = {
            "schema_version": SCHEMA_VERSION,
            "result_kind": "observation",
            "against": args.against,
            "identity": identity,
            "observation": probe_record,
            "calibrations": calibrations,
            "upstream_comparison": comparison,
            "inference_policy": {
                "kind": "observation_only",
                "never_infer": ["conformant", "promoted", "full_tinygrad_parity"],
                "adequacy": "open",
            },
            "artifacts": refs_for_payloads(payloads),
        }
        if args.against == "upstream":
            rejected = [
                {
                    "mutation_id": item["mutation_id"],
                    "outcome": item["outcome"],
                    "expected": item["expected"],
                    "evaluations": item["evaluations"],
                    "baseline_targets": [
                        {
                            "scenario_id": scenario["id"],
                            "trace": scenario["trace"],
                            "terminal": scenario["terminal"],
                        }
                        for scenario in protocol["scenarios"]
                        if scenario["id"] in next(
                            declaration["target_scenarios"]
                            for declaration in manifest["mutations"]
                            if declaration["id"] == item["mutation_id"]
                        )
                    ],
                }
                for item in calibrations
                if item["outcome"] != "validator_rejected_mutant"
            ]
            if rejected:
                if not args.allow_diagnostic:
                    raise RuntimeError(
                        "mutant calibration failed; refusing evidence: " +
                        canonical(rejected).decode("ascii")
                    )
                document["result_kind"] = "diagnostic_blocker"
                document["calibration_rejections"] = rejected
                document["inference_policy"]["baseline_eligible"] = False
        if args.against == "tgrad":
            # The observer is read-only.  Rechecking the product source after
            # execution catches accidental or concurrent mutation of its inputs.
            if product_source_identity() != source:
                raise RuntimeError("product source changed during observation")
        document["evidence_id"] = computed_evidence_id(document)
        return document, payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--against", choices=("upstream", "tgrad"), required=True)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--upstream-baseline", type=Path)
    parser.add_argument("--lock", type=Path, default=V4_AMENDMENT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-diagnostic", action="store_true",
        help="persist a fail-closed upstream calibration blocker; never baseline-eligible",
    )
    args = parser.parse_args()
    try:
        document, payloads = observe(args)
        persist(args.output, document, payloads)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"broadcast-add observer refused execution: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "against": document["against"],
        "evidence_id": document["evidence_id"],
        "inference": "none",
        "result_kind": document["result_kind"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

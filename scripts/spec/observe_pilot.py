#!/usr/bin/env python3
"""Observe the pilot Python-substitution requirement and emit bound evidence.

The observer uses a fake `tgrad` module so it does not load Metal.  It places a
same-named fake upstream package beside the probe and rejects any helper module
whose provider escapes the strict shim.  A copied shim is then deliberately
mutated to append the upstream package path; the validator is calibrated only
when it rejects that fallback.

Both JSON and Lean outputs are deterministic for identical inputs.  There is
no wall-clock timestamp in the evidence identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SHIM = REPO / "scripts" / "parity" / "shim"
DEFAULT_JSON = REPO / "fixtures" / "requirements" / "pilot_helpers_c465f89.json"
DEFAULT_LEAN = REPO / "Tgrad" / "Evidence" / "PilotGenerated.lean"
TARGET_REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
PRODUCT_REVISION = "c465f89999adf0c5fa91d771d3485428d26a2c61"
REQUIRED_NAMES = ("Context", "getenv", "DEV")
SCHEMA_VERSION = 1


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_hash(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append((path.relative_to(root).as_posix(), digest(path.read_bytes())))
    return digest(canonical(entries))


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


PROBE = r"""
import importlib
import json
import os
from pathlib import Path

expected = Path(os.environ["EXPECTED_SHIM_PACKAGE"]).resolve()
required = tuple(os.environ["REQUIRED_NAMES"].split(","))
try:
    helpers = importlib.import_module("tinygrad.helpers")
except ModuleNotFoundError:
    print(json.dumps({"status": "module_unavailable", "missing": list(required)}, sort_keys=True))
    raise SystemExit(0)
except BaseException as exc:
    print(json.dumps({"status": "probe_error", "exception": type(exc).__name__}, sort_keys=True))
    raise SystemExit(0)

origin = Path(helpers.__file__).resolve()
try:
    origin.relative_to(expected)
except ValueError:
    print(json.dumps({"status": "contaminated_provider"}, sort_keys=True))
    raise SystemExit(0)

missing = [name for name in required if not hasattr(helpers, name)]
if missing:
    print(json.dumps({"status": "missing_names", "missing": missing}, sort_keys=True))
else:
    print(json.dumps({"status": "pass", "provider": origin.relative_to(expected).as_posix()}, sort_keys=True))
""".strip()


def run_probe(shim_root: Path, fake_tgrad: Path, fake_upstream: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shim_root), str(fake_tgrad), str(fake_upstream)]
    )
    env["PYTHONSAFEPATH"] = "1"
    env["EXPECTED_SHIM_PACKAGE"] = str((shim_root / "tinygrad").resolve())
    env["REQUIRED_NAMES"] = ",".join(REQUIRED_NAMES)
    completed = subprocess.run(
        [sys.executable, "-S", "-c", PROBE],
        cwd=fake_tgrad.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return {"status": "probe_process_error", "returncode": completed.returncode}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return {"status": "probe_protocol_error", "line_count": len(lines)}
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError:
        return {"status": "probe_protocol_error", "line_count": len(lines)}
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        return {"status": "probe_protocol_error", "line_count": len(lines)}
    return result


def set_up_world(root: Path) -> tuple[Path, Path, Path]:
    fake_tgrad = root / "tgrad_source"
    fake_upstream = root / "upstream"
    upstream_package = fake_upstream / "tinygrad"
    fake_tgrad.mkdir()
    upstream_package.mkdir(parents=True)
    (fake_tgrad / "tgrad.py").write_text("class Tensor:\n    pass\n", encoding="utf-8")
    (upstream_package / "__init__.py").write_text(
        "class Tensor:\n    pass\n", encoding="utf-8"
    )
    (upstream_package / "helpers.py").write_text(
        "class Context:\n    pass\n"
        "def getenv(*args, **kwargs):\n    return None\n"
        "DEV = object()\n"
        "FALLBACK_ONLY = True\n",
        encoding="utf-8",
    )
    return fake_tgrad, fake_upstream, upstream_package


def calibrate(shim_root: Path, root: Path, fake_tgrad: Path,
              fake_upstream: Path, upstream_package: Path) -> dict:
    mutant_root = root / "mutant_shim"
    shutil.copytree(
        shim_root,
        mutant_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    mutant_init = mutant_root / "tinygrad" / "__init__.py"
    with mutant_init.open("a", encoding="utf-8") as handle:
        handle.write(f"\n__path__.append({str(upstream_package)!r})\n")
    result = run_probe(mutant_root, fake_tgrad, fake_upstream)
    fault_model = "strict shim appends a same-named upstream package path"
    return {
        "fault_model": fault_model,
        "mutant_id": digest(canonical({
            "fault_model": fault_model,
            "base_adapter_hash": content_hash(shim_root),
        })),
        "result": result,
        "outcome": (
            "validator_rejected_mutant"
            if result.get("status") == "contaminated_provider"
            else "mutant_survived"
        ),
    }


def lean_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_lean(document: dict) -> str:
    obs = document["observation"]
    cal = document["calibration"]
    calibration_outcome = (
        ".validatorRejectedMutant"
        if cal["outcome"] == "validator_rejected_mutant"
        else ".mutantSurvived"
    )
    observation_outcome = ".passed" if obs["outcome"] == "passed" else ".failed"
    blockage = "" if obs["outcome"] == "passed" else """
def helpersBlockage : Blockage :=
  { id := "BLOCKAGE-HELPERS-PREREQUISITE"
    blocks := [broadcastAdd.id, viewReadbackLifetime.id]
    targetRevision := targetRevision
    subjectTree := subjectTree
    boundary := boundary
    sourceObservationId := helperObservation.id
    reason := "The strict substitution cannot resolve tinygrad.helpers, so dependent upstream scenarios cannot reach tensor behavior."
    artifactHash := artifactHash }

def blockages : List Blockage := [helpersBlockage]
"""
    if obs["outcome"] == "passed":
        blockage = "def blockages : List Blockage := []\n"
    return f'''import Tgrad.Evidence.Observations
import Tgrad.Requirements.Pilot
import Tgrad.Specification.Pilot

/- GENERATED by scripts/spec/observe_pilot.py; do not hand edit. -/

namespace Tgrad.Evidence.PilotGenerated

open Tgrad.Requirements
open Tgrad.Requirements.Pilot
open Tgrad.Specification.Pilot
open Tgrad.Evidence

def targetRevision : String := {lean_string(document["target_revision"])}
def artifactHash : String := {lean_string(obs["artifact_hash"])}

def subjectTree : TreeRef :=
  {{ revision := {lean_string(document["product_revision"])}
    contentHash := {lean_string(document["product_tree_hash"])}
    dirty := false }}

def boundary : BoundaryIdentity :=
  {{ verifierTree :=
      {{ revision := "observe-pilot-v1"
        contentHash := {lean_string(document["verifier_hash"])}
        dirty := false }}
    adapterHash := {lean_string(document["adapter_hash"])}
    environmentId := {lean_string(document["environment"]["id"])}
    environmentHash := {lean_string(document["environment"]["hash"])}
    scenarioManifestHash := {lean_string(document["scenario_hash"])} }}

def helperValidator : ValidatorRef :=
  {{ id := "VALIDATOR-HELPERS-IMPORT-V1"
    version := "1"
    dimensions := importHelpers.relation.dimensions
    calibrations :=
      [{{ faultModel := {lean_string(cal["fault_model"])}
         mutantTree := {lean_string(cal["mutant_id"])}
         artifactHash := {lean_string(digest(canonical(cal["result"])))}
         outcome := {calibration_outcome} }}] }}

def helperObservation : Observation :=
  {{ id := "OBS-HELPERS-IMPORT-C465F89"
    requirement := importHelpers.id
    specification := helpersBoundary.id
    targetRevision := targetRevision
    subjectTree := subjectTree
    boundary := boundary
    validatorId := helperValidator.id
    dimensions := importHelpers.relation.dimensions
    outcome := {observation_outcome}
    blocker := ""
    artifactHash := artifactHash
    runId := {lean_string(document["run_id"])} }}

def validators : List ValidatorRef := [helperValidator]
def observations : List Observation := [helperObservation]

{blockage}
end Tgrad.Evidence.PilotGenerated
'''


def build_document(shim_root: Path, product_revision: str) -> dict:
    product_full = git_value("rev-parse", product_revision)
    product_tree = git_value("rev-parse", f"{product_full}^{{tree}}")
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    environment_doc = {
        "id": "python-import-pilot-v1",
        "hash": digest(canonical(environment)),
        "facts": environment,
    }
    with tempfile.TemporaryDirectory(prefix="tgrad_req_pilot_") as tmp:
        root = Path(tmp)
        fake_tgrad, fake_upstream, upstream_package = set_up_world(root)
        observation_result = run_probe(shim_root, fake_tgrad, fake_upstream)
        calibration = calibrate(
            shim_root, root, fake_tgrad, fake_upstream, upstream_package
        )
    verifier_hash = digest(Path(__file__).read_bytes())
    adapter_hash = content_hash(shim_root)
    scenario_hash = digest(canonical({
        "probe": PROBE,
        "required_names": REQUIRED_NAMES,
        "target_revision": TARGET_REVISION,
    }))
    artifact_hash = digest(canonical(observation_result))
    outcome = "passed" if observation_result.get("status") == "pass" else "failed"
    run_id = digest(canonical({
        "target": TARGET_REVISION,
        "product_tree": product_tree,
        "verifier": verifier_hash,
        "adapter": adapter_hash,
        "environment": environment_doc["hash"],
        "scenario": scenario_hash,
        "artifact": artifact_hash,
    }))
    return {
        "schema_version": SCHEMA_VERSION,
        "requirement": "REQ-PY-IMPORT-HELPERS",
        "target_revision": TARGET_REVISION,
        "product_revision": product_full,
        "product_tree_hash": product_tree,
        "verifier_hash": verifier_hash,
        "adapter_hash": adapter_hash,
        "environment": environment_doc,
        "scenario_hash": scenario_hash,
        "calibration": calibration,
        "observation": {
            "outcome": outcome,
            "result": observation_result,
            "artifact_hash": artifact_hash,
        },
        "run_id": run_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shim-root", type=Path, default=DEFAULT_SHIM)
    parser.add_argument("--product-revision", default=PRODUCT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--lean-output", type=Path, default=DEFAULT_LEAN)
    parser.add_argument("--check", action="store_true",
                        help="compare generated bytes with existing outputs")
    args = parser.parse_args()

    shim_root = args.shim_root.resolve()
    if not (shim_root / "tinygrad" / "__init__.py").is_file():
        parser.error(f"not a strict shim root: {shim_root}")
    document = build_document(shim_root, args.product_revision)
    json_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    lean_bytes = render_lean(document).encode()

    if args.check:
        mismatches = []
        for path, expected in ((args.output, json_bytes), (args.lean_output, lean_bytes)):
            if not path.is_file() or path.read_bytes() != expected:
                mismatches.append(str(path))
        if mismatches:
            print("pilot evidence drift: " + ", ".join(mismatches), file=sys.stderr)
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.lean_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json_bytes)
        args.lean_output.write_bytes(lean_bytes)

    print(json.dumps({
        "calibration": document["calibration"]["outcome"],
        "observation": document["observation"]["outcome"],
        "run_id": document["run_id"],
    }, sort_keys=True))
    return 0 if document["calibration"]["outcome"] == "validator_rejected_mutant" else 2


if __name__ == "__main__":
    raise SystemExit(main())

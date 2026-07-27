#!/usr/bin/env python3
"""Observe the pilot Python-substitution requirement and emit bound evidence.

The observer imports the real Tgrad Python module whose source bytes match the
declared product revision and records the exact built Lean/C runtime artifact.
It places a same-named fake upstream package
beside the probe and rejects any helper module whose provider escapes the strict
shim.  Copied shims are deliberately mutated to append the upstream package
path and to remove a required public name; each observation dimension is
calibrated only by its corresponding caught fault.

Both JSON and Lean outputs are deterministic for identical inputs.  There is
no wall-clock timestamp in the evidence identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SHIM = REPO / "scripts" / "parity" / "shim"
PRODUCT_PYTHON = REPO / "python"
PRODUCT_MODULE = PRODUCT_PYTHON / "tgrad.py"
RUNTIME_LIBRARY = REPO / ".lake" / "build" / "lib" / "libtgrad.dylib"
DEFAULT_JSON = REPO / "fixtures" / "requirements" / "pilot_helpers_b1df552.json"
DEFAULT_LEAN = REPO / "Tgrad" / "Evidence" / "PilotGenerated.lean"
TARGET_REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
PRODUCT_REVISION = "b1df55266a0382ff471f8cf87b04497239495dba"
REQUIRED_NAMES = ("Context", "getenv", "DEV")
SCHEMA_VERSION = 3

FAKE_UPSTREAM_INIT = """\
class Tensor:
    pass
"""

FAKE_UPSTREAM_HELPERS = """\
class Context:
    pass
def getenv(*args, **kwargs):
    return None
DEV = object()
FALLBACK_ONLY = True
"""

DIMENSION_LEAN = {
    "module_resolution": ".importResolution",
    "public_names": ".publicSurface",
}


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


def git_blob(revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{revision}:{path}"],
        capture_output=True,
        check=True,
    )
    return result.stdout


def git_directory_content_hash(revision: str, root: str) -> str:
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", revision, "--", root],
        capture_output=True,
        text=True,
        check=True,
    )
    prefix = root.rstrip("/") + "/"
    entries = []
    for path in sorted(line for line in listing.stdout.splitlines() if line):
        relative = path.removeprefix(prefix)
        if "__pycache__" in Path(relative).parts or relative.endswith(".pyc"):
            continue
        entries.append((relative, digest(git_blob(revision, path))))
    if not entries:
        raise RuntimeError(f"no tracked files beneath {root!r} at {revision}")
    return digest(canonical(entries))


PROBE = r"""
import importlib
import json
import os
from pathlib import Path

expected = Path(os.environ["EXPECTED_SHIM_PACKAGE"]).resolve()
expected_product = Path(os.environ["EXPECTED_PRODUCT_MODULE"]).resolve()
required = tuple(os.environ["REQUIRED_NAMES"].split(","))
try:
    tgrad = importlib.import_module("tgrad")
    helpers = importlib.import_module("tinygrad.helpers")
except ModuleNotFoundError:
    print(json.dumps({
        "status": "module_unavailable",
        "missing": list(required),
        "dimension_results": {
            "module_resolution": "failed",
            "public_names": "blocked",
        },
    }, sort_keys=True))
    raise SystemExit(0)
except BaseException as exc:
    print(json.dumps({"status": "probe_error", "exception": type(exc).__name__}, sort_keys=True))
    raise SystemExit(0)

product_origin = Path(tgrad.__file__).resolve()
if product_origin != expected_product:
    print(json.dumps({
        "status": "contaminated_product_provider",
        "dimension_results": {
            "module_resolution": "failed",
            "public_names": "blocked",
        },
    }, sort_keys=True))
    raise SystemExit(0)

origin = Path(helpers.__file__).resolve()
try:
    origin.relative_to(expected)
except ValueError:
    print(json.dumps({
        "status": "contaminated_provider",
        "dimension_results": {
            "module_resolution": "failed",
            "public_names": "blocked",
        },
    }, sort_keys=True))
    raise SystemExit(0)

missing = [name for name in required if not hasattr(helpers, name)]
if missing:
    print(json.dumps({
        "status": "missing_names",
        "missing": missing,
        "dimension_results": {
            "module_resolution": "passed",
            "public_names": "failed",
        },
    }, sort_keys=True))
else:
    print(json.dumps({
        "status": "pass",
        "provider": origin.relative_to(expected).as_posix(),
        "product_provider": product_origin.name,
        "dimension_results": {
            "module_resolution": "passed",
            "public_names": "passed",
        },
    }, sort_keys=True))
""".strip()


def controlled_probe_environment(
    shim_root: Path, fake_upstream: Path, dependency_root: Path
) -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(shim_root), str(PRODUCT_PYTHON), str(fake_upstream),
             str(dependency_root)]
        ),
        "TGRAD_ROOT": str(REPO),
        "TGRAD_LIB": str(RUNTIME_LIBRARY),
        "EXPECTED_SHIM_PACKAGE": str((shim_root / "tinygrad").resolve()),
        "EXPECTED_PRODUCT_MODULE": str(PRODUCT_MODULE.resolve()),
        "REQUIRED_NAMES": ",".join(REQUIRED_NAMES),
        "TGRAD_CALIBRATION_UPSTREAM": str((fake_upstream / "tinygrad").resolve()),
    }


def run_probe(
    probe_python: Path, shim_root: Path, fake_upstream: Path,
    dependency_root: Path
) -> dict:
    completed = subprocess.run(
        [str(probe_python), "-c", PROBE],
        cwd=fake_upstream.parent,
        env=controlled_probe_environment(shim_root, fake_upstream, dependency_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    diagnostic = completed.stderr.strip().splitlines()
    diagnostic = diagnostic[-1] if diagnostic else ""
    for path, token in (
        (shim_root, "<shim>"),
        (PRODUCT_PYTHON, "<product-python>"),
        (RUNTIME_LIBRARY, "<runtime-library>"),
        (fake_upstream, "<upstream>"),
        (fake_upstream.parent, "<world>"),
        (REPO, "<repo>"),
    ):
        diagnostic = diagnostic.replace(str(path.resolve()), token)
    if completed.returncode == 86:
        reason = (
            "package_path_contaminated"
            if "strict Tgrad substitution did not own the tinygrad package" in diagnostic
            else "activation_error"
        )
        return {
            "status": "strict_activation_rejected",
            "reason": reason,
            "diagnostic": diagnostic,
        }
    if completed.returncode != 0:
        return {
            "status": "probe_process_error",
            "returncode": completed.returncode,
            "diagnostic": diagnostic,
        }
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


def set_up_world(root: Path) -> Path:
    fake_upstream = root / "upstream"
    upstream_package = fake_upstream / "tinygrad"
    upstream_package.mkdir(parents=True)
    (upstream_package / "__init__.py").write_text(
        FAKE_UPSTREAM_INIT, encoding="utf-8"
    )
    (upstream_package / "helpers.py").write_text(
        FAKE_UPSTREAM_HELPERS, encoding="utf-8"
    )
    return fake_upstream


def calibration_outcome(result: dict, caught: bool) -> str:
    indeterminate = result.get("status") in {
        "probe_process_error", "probe_protocol_error", "probe_error",
        "contaminated_product_provider",
    } or (
        result.get("status") == "strict_activation_rejected" and
        result.get("reason") != "package_path_contaminated"
    )
    if caught:
        return "validator_rejected_mutant"
    return "calibration_indeterminate" if indeterminate else "mutant_survived"


def calibrate(probe_python: Path, shim_root: Path, root: Path,
              fake_upstream: Path, dependency_root: Path) -> list[dict]:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")

    fallback_root = root / "mutant_fallback"
    shutil.copytree(shim_root, fallback_root, ignore=ignore)
    with (fallback_root / "tinygrad" / "__init__.py").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\nimport os as _tgrad_calibration_os\n"
            "__path__.append(_tgrad_calibration_os.environ["
            "'TGRAD_CALIBRATION_UPSTREAM'])\n"
        )
    fallback_result = run_probe(
        probe_python, fallback_root, fake_upstream, dependency_root
    )
    fallback_caught = (
        fallback_result.get("status") == "contaminated_provider" or
        (fallback_result.get("status") == "strict_activation_rejected" and
         fallback_result.get("reason") == "package_path_contaminated")
    )

    missing_name_root = root / "mutant_missing_name"
    shutil.copytree(shim_root, missing_name_root, ignore=ignore)
    helpers_path = missing_name_root / "tinygrad" / "helpers.py"
    helpers_source = helpers_path.read_text(encoding="utf-8")
    needle = "DEV = _DeviceInfo()"
    if needle not in helpers_source:
        raise RuntimeError("cannot inject missing-name mutant: DEV definition changed")
    helpers_path.write_text(
        helpers_source.replace(needle, "DELETED_DEV = _DeviceInfo()", 1),
        encoding="utf-8",
    )
    missing_name_result = run_probe(
        probe_python, missing_name_root, fake_upstream, dependency_root
    )
    missing_name_caught = (
        missing_name_result.get("status") == "missing_names" and
        "DEV" in missing_name_result.get("missing", [])
    )

    return [
        {
            "fault_model": "strict shim appends a same-named upstream package path",
            "dimensions": ["module_resolution"],
            "mutant_id": content_hash(fallback_root),
            "result": fallback_result,
            "outcome": calibration_outcome(fallback_result, fallback_caught),
        },
        {
            "fault_model": "required helper name DEV is removed from the shim",
            "dimensions": ["public_names"],
            "mutant_id": content_hash(missing_name_root),
            "result": missing_name_result,
            "outcome": calibration_outcome(missing_name_result, missing_name_caught),
        },
    ]


def lean_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def dimensions_lean(dimensions: list[str]) -> str:
    return "[" + ", ".join(DIMENSION_LEAN[item] for item in dimensions) + "]"


def calibration_lean(calibration: dict) -> str:
    outcome = {
        "validator_rejected_mutant": ".validatorRejectedMutant",
        "mutant_survived": ".mutantSurvived",
        "calibration_indeterminate": ".indeterminate",
    }[calibration["outcome"]]
    return f'''{{ faultModel := {lean_string(calibration["fault_model"])}
         dimensions := {dimensions_lean(calibration["dimensions"])}
         mutantTree := {lean_string(calibration["mutant_id"])}
         artifactHash := {lean_string(digest(canonical(calibration["result"])))}
         outcome := {outcome} }}'''


def render_lean(document: dict) -> str:
    obs = document["observation"]
    calibration_nodes = ",\n       ".join(
        calibration_lean(calibration)
        for calibration in document["calibrations"]
    )
    observation_outcome = {
        "passed": ".passed",
        "failed": ".failed",
        "verifier_error": ".verifierError",
    }[obs["outcome"]]
    observation_id = "OBS-HELPERS-IMPORT-" + document["product_revision"][:7].upper()
    blockage = "" if obs["outcome"] != "failed" else """
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
    if obs["outcome"] != "failed":
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
      {{ revision := {lean_string(document["verifier_revision"])}
        contentHash := {lean_string(document["verifier_hash"])}
        dirty := false }}
    adapterHash := {lean_string(document["adapter_hash"])}
    runtimeArtifactHash := {lean_string(document["runtime_artifact_hash"])}
    environmentId := {lean_string(document["environment"]["id"])}
    environmentHash := {lean_string(document["environment"]["hash"])}
    scenarioManifestHash := {lean_string(document["scenario_hash"])} }}

def helperValidator : ValidatorRef :=
  {{ id := "VALIDATOR-HELPERS-IMPORT-V1"
    version := "1"
    dimensions := {dimensions_lean(document["validator_dimensions"])}
    calibrations :=
      [{calibration_nodes}] }}

def helperObservation : Observation :=
  {{ id := {lean_string(observation_id)}
    requirement := importHelpers.id
    specification := helpersBoundary.id
    targetRevision := targetRevision
    subjectTree := subjectTree
    boundary := boundary
    validatorId := helperValidator.id
    dimensions := {dimensions_lean(obs["dimensions"])}
    outcome := {observation_outcome}
    blocker := ""
    artifactHash := artifactHash
    runId := {lean_string(document["run_id"])} }}

def validators : List ValidatorRef := [helperValidator]
def observations : List Observation := [helperObservation]

{blockage}
end Tgrad.Evidence.PilotGenerated
'''


def probe_python_facts(probe_python: Path) -> dict:
    code = r'''
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy

executable = Path(sys.executable).resolve()
numpy_init = Path(numpy.__file__).resolve()
print(json.dumps({
    "python": platform.python_version(),
    "implementation": platform.python_implementation(),
    "executable": str(executable),
    "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    "cache_tag": sys.implementation.cache_tag,
    "platform": platform.system(),
    "platform_release": platform.release(),
    "machine": platform.machine(),
    "numpy_version": numpy.__version__,
    "numpy_init": str(numpy_init),
    "numpy_init_sha256": hashlib.sha256(numpy_init.read_bytes()).hexdigest(),
    "dependency_root": str(numpy_init.parents[1]),
}, sort_keys=True))
'''.strip()
    result = subprocess.run(
        [str(probe_python), "-c", code],
        env={"LANG": "C", "LC_ALL": "C"},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    facts = json.loads(result.stdout)
    if not isinstance(facts, dict):
        raise RuntimeError("probe interpreter returned invalid environment facts")
    return facts


def build_document(shim_root: Path, product_revision: str,
                   probe_python: Path) -> dict:
    product_full = git_value("rev-parse", product_revision)
    product_tree = git_value("rev-parse", f"{product_full}^{{tree}}")
    probe_python = probe_python.resolve()
    if not probe_python.is_file():
        raise RuntimeError(f"probe interpreter does not exist: {probe_python}")
    if not RUNTIME_LIBRARY.is_file():
        raise RuntimeError(f"runtime library does not exist: {RUNTIME_LIBRARY}")

    adapter_hash = content_hash(shim_root)
    expected_adapter_hash = git_directory_content_hash(
        product_full, "scripts/parity/shim"
    )
    if adapter_hash != expected_adapter_hash:
        raise RuntimeError(
            "shim bytes do not belong to the declared product revision: "
            f"actual={adapter_hash}, expected={expected_adapter_hash}"
        )
    product_module_hash = digest(PRODUCT_MODULE.read_bytes())
    expected_product_module_hash = digest(
        git_blob(product_full, "python/tgrad.py")
    )
    if product_module_hash != expected_product_module_hash:
        raise RuntimeError(
            "python/tgrad.py does not belong to the declared product revision: "
            f"actual={product_module_hash}, expected={expected_product_module_hash}"
        )

    runtime_artifacts = {
        "product_module_sha256": product_module_hash,
        "runtime_library_sha256": digest(RUNTIME_LIBRARY.read_bytes()),
    }
    runtime_artifact_hash = digest(canonical(runtime_artifacts))
    environment = probe_python_facts(probe_python)
    environment_doc = {
        "id": "python-import-pilot-v1",
        "hash": digest(canonical(environment)),
        "facts": environment,
    }
    dependency_root = Path(environment["dependency_root"])
    with tempfile.TemporaryDirectory(prefix="tgrad_req_pilot_") as tmp:
        root = Path(tmp)
        fake_upstream = set_up_world(root)
        observation_result = run_probe(
            probe_python, shim_root, fake_upstream, dependency_root
        )
        calibrations = calibrate(
            probe_python, shim_root, root, fake_upstream, dependency_root
        )
    verifier_hash = digest(Path(__file__).read_bytes())
    verifier_revision = "sha256:" + verifier_hash
    scenario_hash = digest(canonical({
        "probe": PROBE,
        "required_names": REQUIRED_NAMES,
        "target_revision": TARGET_REVISION,
        "fake_upstream_init": digest(FAKE_UPSTREAM_INIT.encode()),
        "fake_upstream_helpers": digest(FAKE_UPSTREAM_HELPERS.encode()),
        "product_python": "python/tgrad.py",
        "python_safe_path": True,
        "python_no_user_site": True,
        "locale": "C",
        "cwd_policy": "controlled-world-root",
    }))
    artifact_hash = digest(canonical(observation_result))
    dimension_results = observation_result.get("dimension_results", {})
    observed_dimensions = [
        dimension for dimension in DIMENSION_LEAN
        if dimension_results.get(dimension) == "passed"
    ]
    validator_dimensions = [
        dimension for dimension in DIMENSION_LEAN
        if any(
            calibration["outcome"] == "validator_rejected_mutant" and
            dimension in calibration["dimensions"]
            for calibration in calibrations
        )
    ]
    observation_status = observation_result.get("status")
    if observation_status == "pass":
        outcome = "passed"
    elif observation_status in {
        "module_unavailable", "missing_names", "contaminated_provider",
        "contaminated_product_provider",
    } or (
        observation_status == "strict_activation_rejected" and
        observation_result.get("reason") == "package_path_contaminated"
    ):
        outcome = "failed"
    else:
        outcome = "verifier_error"
    run_id = digest(canonical({
        "target": TARGET_REVISION,
        "product_tree": product_tree,
        "verifier": verifier_hash,
        "calibrations": digest(canonical(calibrations)),
        "adapter": adapter_hash,
        "runtime_artifact": runtime_artifact_hash,
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
        "verifier_revision": verifier_revision,
        "adapter_hash": adapter_hash,
        "adapter_source_revision": product_full,
        "runtime_artifacts": runtime_artifacts,
        "runtime_artifact_hash": runtime_artifact_hash,
        "environment": environment_doc,
        "scenario_hash": scenario_hash,
        "validator_dimensions": validator_dimensions,
        "calibrations": calibrations,
        "observation": {
            "outcome": outcome,
            "dimensions": observed_dimensions,
            "result": observation_result,
            "artifact_hash": artifact_hash,
        },
        "run_id": run_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shim-root", type=Path, default=DEFAULT_SHIM)
    parser.add_argument("--product-revision", default=PRODUCT_REVISION)
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="interpreter used to import the real Tgrad product")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--lean-output", type=Path, default=DEFAULT_LEAN)
    parser.add_argument("--check", action="store_true",
                        help="compare generated bytes with existing outputs")
    parser.add_argument(
        "--check-generated", action="store_true",
        help="check only that committed JSON renders to committed Lean evidence",
    )
    parser.add_argument(
        "--expected-outcome", choices=["passed", "failed", "verifier_error"],
        default="passed",
    )
    parser.add_argument(
        "--write-on-mismatch", action="store_true",
        help="write diagnostic evidence even when calibration/outcome expectations fail",
    )
    args = parser.parse_args()

    if args.check and args.check_generated:
        parser.error("--check and --check-generated are mutually exclusive")

    shim_root = args.shim_root.resolve()
    if not (shim_root / "tinygrad" / "__init__.py").is_file():
        parser.error(f"not a strict shim root: {shim_root}")
    if args.check_generated:
        if not args.output.is_file():
            print(f"missing pilot evidence: {args.output}", file=sys.stderr)
            return 1
        document = json.loads(args.output.read_text(encoding="utf-8"))
        lean_bytes = render_lean(document).encode()
        if not args.lean_output.is_file() or args.lean_output.read_bytes() != lean_bytes:
            print(f"pilot generated Lean drift: {args.lean_output}", file=sys.stderr)
            return 1
        print(json.dumps({"generated_evidence": "matches_json"}, sort_keys=True))
        return 0

    document = build_document(shim_root, args.product_revision, args.python)
    json_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    lean_bytes = render_lean(document).encode()
    calibrations_ok = all(
        item["outcome"] == "validator_rejected_mutant"
        for item in document["calibrations"]
    )
    outcome_ok = document["observation"]["outcome"] == args.expected_outcome

    summary = json.dumps({
        "calibrations": [item["outcome"] for item in document["calibrations"]],
        "observation": document["observation"]["outcome"],
        "run_id": document["run_id"],
    }, sort_keys=True)

    if not args.check and not args.write_on_mismatch and not (
        calibrations_ok and outcome_ok
    ):
        print(summary)
        print(
            "refusing to overwrite promoted pilot evidence with an unexpected "
            "result; use --write-on-mismatch for a diagnostic artifact",
            file=sys.stderr,
        )
        return 2 if not calibrations_ok else 3

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

    print(summary)
    if not calibrations_ok:
        return 2
    if not outcome_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

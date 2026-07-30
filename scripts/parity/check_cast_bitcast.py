#!/usr/bin/env python3
"""Foreign-grounded Wave 10 Tensor cast/bitcast observer.

The observer has one frozen implementation and two execution modes.  The
``preimplementation`` mode establishes a CPU-only RED on the clean base.  The
default ``full`` mode additionally executes Metal-backed exact-byte, storage,
allocation, dispatch, rejection, and lifetime checks.  Expected conversion
bits are derived here from the frozen evaluator contract, never from Tgrad or
tinygrad conversion helpers.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity.ensure_oracle import EXPECTED, verify


TARGETS = (
    "test/backend/test_dtype.py::TestBFloat16DType::test_bf16",
    "test/backend/test_dtype.py::TestOpsBFloat16::test_cast",
    "test/backend/test_dtype.py::TestBitCast::test_bitcast_float_to_int32",
)
BASE_HEAD = "9d8b57a7a498944723076b1490ad9c8ed831dc00"
BASE_TREE = "a9c9ee63da1a77db00bedd1468dcc96e30cd217a"
TEST_SOURCE_SHA256 = "c040cb6b1c49e6752ced59315d52ea5c0aa9d4075dabd3c9f574e447b2f7de37"
BASE_IDENTITIES = {
    "scripts/parity/run_upstream_suite.py": "def81e4172717c97a441fae590b517e2754f7e5e0cb253a848a1ec9232028c42",
    "scripts/parity/tgrad_pytest_reporter.py": "2c2a1499a2fec7bb2d4f38a9d1cf1358be66af1fffe3add898197c34f8567e0a",
    "fixtures/parity/oracle_classification.json": "8281ef9c195b730ccd48d8af6600592c44d7abbe647a6103c4b9f54f1e2f2ba3",
    "fixtures/parity/upstream_19c4d736f2bc.json": "50438780a69f63d2a73aa6f276dc00f31f11daf024768eeaf7cf2f3c6633ac2b",
}
STRICT_RUNNER_SHA256 = "747f83cc5a94abc888707659eb50046926f23b6cc6be0fc018a55da30b8d560e"
MISSING_AUTHORITY = "missing separate Lean CastPlan/BitcastPlan authority"
STRICT_IDENTITY_MARKER = "strict Tgrad substitution active: "


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def validate_strict_identity(
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    candidate_repo: Path,
) -> tuple[list[str], dict[str, object]]:
    """Validate the real strict runner's ``--verify-only`` observation.

    Expected paths are derived here from the resolved candidate repository,
    so a caller cannot make a mutually wrong observation and expectation pass.
    Keeping the decision pure makes each failure axis directly falsifiable.
    """
    failures: list[str] = []
    candidate_root = candidate_repo.resolve()
    expected_tinygrad = (candidate_root / "scripts" / "parity" / "shim" /
                         "tinygrad" / "__init__.py").resolve()
    expected_tgrad = (candidate_root / "python" / "tgrad.py").resolve()
    marker_lines = [line for line in stdout.splitlines()
                    if line.startswith(STRICT_IDENTITY_MARKER)]
    actual_tinygrad: Path | None = None
    actual_tgrad: Path | None = None
    if returncode != 0:
        failures.append("strict substitution identity subprocess returned nonzero")
    if len(marker_lines) != 1:
        failures.append("strict substitution identity marker is not exact and unique")
    else:
        match = re.fullmatch(
            re.escape(STRICT_IDENTITY_MARKER) + r"tinygrad=(.+), tgrad=(.+)",
            marker_lines[0],
        )
        if match is None:
            failures.append("strict substitution identity marker is malformed")
        else:
            actual_tinygrad = Path(match.group(1)).resolve()
            actual_tgrad = Path(match.group(2)).resolve()
            if actual_tinygrad != expected_tinygrad:
                failures.append("strict substitution tinygrad path mismatch")
            if actual_tgrad != expected_tgrad:
                failures.append("strict substitution tgrad path mismatch")
    diagnostics: dict[str, object] = {
        "returncode": returncode,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "expected_tinygrad_init": str(expected_tinygrad),
        "expected_tgrad_module": str(expected_tgrad),
        "actual_tinygrad_init": str(actual_tinygrad) if actual_tinygrad else None,
        "actual_tgrad_module": str(actual_tgrad) if actual_tgrad else None,
    }
    return failures, diagnostics


def strict_runner_path(candidate_repo: Path) -> Path:
    return (candidate_repo.resolve() / "scripts" / "parity" / "shim" /
            "run_pytest.py")


def validate_strict_runner(candidate_repo: Path) -> tuple[list[str], dict[str, object]]:
    runner = strict_runner_path(candidate_repo)
    actual = sha256(runner) if runner.is_file() else None
    failures = [] if actual == STRICT_RUNNER_SHA256 else [
        "candidate strict runner SHA-256 mismatch"
    ]
    return failures, {
        "path": str(runner),
        "expected_sha256": STRICT_RUNNER_SHA256,
        "actual_sha256": actual,
    }


def isolated_env(temp: Path, **extra: str) -> dict[str, str]:
    home, scratch = temp / "home", temp / "tmp"
    home.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    env = {
        "HOME": str(home), "TMPDIR": str(scratch), "LANG": "C", "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "CACHELEVEL": "0",
    }
    env.update(extra)
    return env


def snapshot_oracle(checkout: Path, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", EXPECTED.revision],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as stream:
        stream.extractall(destination, filter="data")


def source_hygiene_failures(repo: Path) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    checked: set[Path] = {Path(__file__).resolve()}
    for label, arguments in (
        ("unstaged", ("diff", "--check")),
        ("staged", ("diff", "--cached", "--check")),
    ):
        check = subprocess.run(["git", "-C", str(repo), *arguments], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check.returncode != 0:
            failures.append(
                f"source hygiene {label} diff failed: "
                f"{(check.stdout + check.stderr).strip()}")
    for arguments in (("diff", "--name-only", "-z"),
                      ("diff", "--cached", "--name-only", "-z"),
                      ("ls-files", "--others", "--exclude-standard", "-z")):
        names = subprocess.check_output(["git", "-C", str(repo), *arguments])
        for raw_name in names.split(b"\0"):
            if raw_name:
                checked.add((repo / os.fsdecode(raw_name)).resolve())
    root = repo.resolve()
    for path in sorted(checked):
        if path != Path(__file__).resolve() and not path.is_relative_to(root):
            failures.append(f"source hygiene path escaped candidate repository: {path}")
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for line_number, line in enumerate(data.splitlines(), start=1):
            if line.endswith((b" ", b"\t")):
                display = path.relative_to(root) if path.is_relative_to(root) else path
                failures.append(f"source hygiene trailing whitespace {display}:{line_number}")
    return failures, [str(path.relative_to(root) if path.is_relative_to(root) else path)
                      for path in sorted(checked)]


def function_node(source: str, class_name: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return next((item for item in node.body
                         if isinstance(item, ast.FunctionDef) and item.name == name), None)
    return None


def module_function_node(source: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    return next((node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name), None)


def architecture_failures(repo: Path) -> list[str]:
    failures: list[str] = []
    cast_path = repo / "Tgrad" / "Cast.lean"
    cast_source = cast_path.read_text() if cast_path.is_file() else ""
    authority = (
        "private structure CastPlan", "private structure BitcastPlan",
        "inductive TransformReason", "structure TransformResult",
        "buildCastPlan", "buildBitcastPlan", "realizeCast", "realizeBitcast",
        "cast_admitted_pairs_exact", "bitcast_admitted_pairs_exact",
        "cast_bitcast_plans_distinct", "bitcast_bytes_preserved",
        "bitcast_shape_preserved", "identity_preserves_storage",
        "portable_cast_sources", "exactBf16PackExpr",
        "cast_root_is_exact_buffer", "cast_root_dtype_matches",
        "cast_dims_positive", "cast_byte_products_fit",
        "cast_buffer_length_covers_input", "cast_launch_covers_exactly",
        "cast_registers_after_dispatch", "thread_position_in_grid",
        "structure CastArtifact", "artifactForPlan",
        "cast_artifact_exact_identity", "cast_cache_collision_checked",
        "cast_dispatch_uses_artifact_name",
        "maxCastElementCount", "max_cast_element_count_contract",
        "exactCastIndexDecl", "exactCastGuardStmt",
        "exact_cast_index_declaration_contract",
        "exact_cast_guard_statement_contract",
        "exact_bf16_pack_expression_contract",
        "exact_bf16_expand_expression_contract",
    )
    missing = [token for token in authority if token not in cast_source]
    if missing:
        failures.append(MISSING_AUTHORITY)
        failures.extend(f"Lean cast authority missing {token}" for token in missing)
    if "import Tgrad.Cast" not in (repo / "Tgrad.lean").read_text():
        failures.append("Tgrad root does not import the cast authority")
    if "import Tgrad.Cast" not in (repo / "Tgrad" / "PythonFFI.lean").read_text():
        failures.append("PythonFFI does not pin the cast authority")

    uop_source = (repo / "Tgrad" / "UOp.lean").read_text()
    if not re.search(r"^\s*\|\s+bitcast\b", uop_source, flags=re.M):
        failures.append("UOp lacks a distinct compiler-visible bitcast constructor")
    tensor_source = (repo / "Tgrad" / "Tensor.lean").read_text()
    for token in ("| .bitcast", "bitcastStorageRoot"):
        if token not in tensor_source:
            failures.append(f"Tensor queries do not recognize bitcast IR via {token}")
    if ".bitcast" not in cast_source:
        failures.append("BitcastPlan does not produce distinct bitcast IR")
    render_start = cast_source.find("private def renderArtifact")
    render_end = cast_source.find("private def artifactForPlan", render_start)
    render_body = cast_source[render_start:render_end] \
        if render_start >= 0 and render_end > render_start else ""
    if not re.search(r"exactCastGuardStmt\s+launch\.guardExclusive", render_body):
        failures.append("renderArtifact does not consume its exact guard authority")
    dtype_source = (repo / "Tgrad" / "Dtype.lean").read_text()
    for token in ("inductive Dtype.BitcastStorageRelation",
                  "Dtype.bitcastStoragePair_contract",
                  "Dtype.bitcastStoragePair_itemsize"):
        if token not in dtype_source:
            failures.append(f"shared exact bitcast relation missing {token}")
    if "source.bitcastStoragePair target" not in tensor_source \
            or "root.rootDtype.bitcastStoragePair root.targetDtype" not in cast_source:
        failures.append("Tensor classification and BitcastPlan do not share exact relation")
    if "threadgroup_position_in_grid" in cast_source:
        failures.append("cast renderer uses threadgroup position as an element index")
    if not re.search(r"thread_position_in_grid.*\buint\b", cast_source, flags=re.S):
        failures.append("cast renderer lacks a thread-position element index")
    if "String.hash" in cast_source or ".hash" in cast_source:
        failures.append("cast artifact/cache identity relies on a digest alone")

    ffi_source = (repo / "Tgrad" / "PythonFFI.lean").read_text()
    for token in ("tgrad_tensor_cast_lean", "tgrad_tensor_bitcast_lean",
                  "tgrad_tensor_transform_query_lean",
                  "tgrad_tensor_transform_release_lean",
                  "tgrad_tensor_registry_count_lean",
                  "tgrad_tensor_is_materialized_storage_lean"):
        if token not in ffi_source:
            failures.append(f"Lean FFI missing {token}")
    for token in ("def remove?", "TransformOwnership.owned",
                  "TransformOwnership.borrowed"):
        if token not in ffi_source:
            failures.append(f"Lean transform registry lifecycle missing {token}")

    c_source = (repo / "c" / "tgrad_python.c").read_text()
    for token in ("tgrad_tensor_cast", "tgrad_tensor_bitcast",
                  "tgrad_tensor_transform_query", "tgrad_tensor_transform_release",
                  "tgrad_tensor_registry_count", "tgrad_tensor_is_materialized_storage",
                  "tgrad_metal_counter_reset",
                  "tgrad_metal_counter", "tgrad_metal_fault_set",
                  "tgrad_metal_fault_clear", "tgrad_metal_watch_buffer",
                  "tgrad_metal_watch_free_count"):
        if token not in c_source:
            failures.append(f"C ABI missing {token}")
    for name in ("tgrad_tensor_cast", "tgrad_tensor_bitcast"):
        match = re.search(rf"uint8_t\s+{name}\s*\([^)]*uint64_t\s*\*\s*out_handle[^)]*\)\s*\{{(.*?)\n\}}",
                          c_source, flags=re.S)
        if match is None:
            failures.append(f"C trampoline {name} lacks status-plus-out-handle ABI")
        else:
            body = match.group(1)
            for forbidden in ("memcpy", "malloc", "float", "bfloat", "int32_t["):
                if forbidden in body:
                    failures.append(f"C trampoline {name} owns semantics via {forbidden}")

    metal_source = (repo / "c" / "metal_alloc.m").read_text()
    for token in ("TgradMetalLibrary", "exact_source", "generation",
                  "purge_pipeline_cache_for_generation",
                  "record->exact_source", "record->generation"):
        if token not in metal_source:
            failures.append(f"Metal artifact/cache binding missing {token}")
    if 'stringWithFormat:@"%p:%@"' in metal_source:
        failures.append("Metal pipeline cache still keys a recyclable library pointer")
    exact_key = re.search(
        r'stringWithFormat:@"%llu:%@:%@"\s*,\s*'
        r'\(unsigned long long\)record->generation\s*,\s*exactSource\s*,\s*fn',
        metal_source, flags=re.S)
    if exact_key is None:
        failures.append("Metal pipeline key is not bound to generation+exact source+function")
    release = re.search(
        r"void\s+theograd_metal_library_release\s*\([^)]*\)\s*\{(.*?)\n\}",
        metal_source, flags=re.S)
    if release is None or "purge_pipeline_cache_for_generation" not in release.group(1):
        failures.append("Metal library release does not evict its pipeline generation")

    py_path = repo / "python" / "tgrad.py"
    py_source = py_path.read_text()
    for token in ("class TgradTransformError", "def _adopt_transform_result",
                  "tgrad_tensor_transform_release", "ctypes.byref(out_handle)",
                  "tgrad_tensor_is_materialized_storage"):
        if token not in py_source:
            failures.append(f"Python transactional transform boundary missing {token}")
    for method, delegate in (("cast", "tgrad_tensor_cast"),
                             ("bitcast", "tgrad_tensor_bitcast")):
        node = function_node(py_source, "Tensor", method)
        if node is None:
            failures.append(f"public Tensor.{method} is missing")
            continue
        text = ast.get_source_segment(py_source, node) or ""
        if delegate not in text:
            failures.append(f"Tensor.{method} does not delegate to Lean")
        if "_from_result_handle" in text:
            failures.append(f"Tensor.{method} reuses generic owning result adoption")
        forbidden_nodes = (ast.For, ast.AsyncFor, ast.While, ast.ListComp,
                           ast.SetComp, ast.DictComp, ast.GeneratorExp)
        if any(isinstance(item, forbidden_nodes) for item in ast.walk(node)):
            failures.append(f"Tensor.{method} contains Python iteration")
        for token in ("numpy", "to_bytes", "frombuffer", "astype", "struct.",
                      "_bytes_from_numpy", "_numpy_from_bytes"):
            if token in text:
                failures.append(f"Tensor.{method} owns semantics via {token}")
        if "self._dtype ==" in text or "dtype == self" in text:
            failures.append(f"Tensor.{method} bypasses Lean identity admission")
        for item in ast.walk(node):
            if isinstance(item, (ast.Set, ast.List, ast.Tuple)) and len(item.elts) >= 2:
                rendered = ast.get_source_segment(py_source, item) or ""
                if any(name in rendered for name in ("bf16", "f32", "i32", "float32", "int32")):
                    failures.append(f"Tensor.{method} contains a Python pair-admission table")

    transform = function_node(py_source, "Tensor", "_transform")
    if transform is None:
        failures.append("reachable Python Tensor._transform is missing")
    else:
        transform_text = ast.get_source_segment(py_source, transform) or ""
        entry_lines = [node.lineno for node in ast.walk(transform)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "entry"]
        self_returns = [node.lineno for node in ast.walk(transform)
                        if isinstance(node, ast.Return)
                        and isinstance(node.value, ast.Name) and node.value.id == "self"]
        if len(entry_lines) != 1 or any(line <= entry_lines[0] for line in self_returns):
            failures.append("Tensor._transform can return identity before its ABI entry")
        for token in ("numpy", "to_bytes", "frombuffer", "astype", "struct.",
                      "_bytes_from_numpy", "_numpy_from_bytes", "self._dtype =="):
            if token in transform_text:
                failures.append(f"Tensor._transform owns semantics via {token}")
        for item in ast.walk(transform):
            if isinstance(item, (ast.Set, ast.List, ast.Tuple)) and len(item.elts) >= 2:
                rendered = ast.get_source_segment(py_source, item) or ""
                if any(name in rendered for name in
                       ("bf16", "f32", "i32", "float32", "int32")):
                    failures.append("Tensor._transform contains a Python rejection table")

    tree = ast.parse(py_source)
    for item in ast.walk(tree):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in item.names]
            module = item.module or "" if isinstance(item, ast.ImportFrom) else ""
            if module == "tinygrad" or any(name == "tinygrad" or name.startswith("tinygrad.") for name in names):
                failures.append("product Python imports tinygrad runtime")

    observer_path = repo / "scripts" / "parity" / "check_cast_bitcast.py"
    observer_source = observer_path.read_text() if observer_path.is_file() else ""
    validator_node = module_function_node(observer_source, "validate_strict_identity") \
        if observer_source else None
    main_node = module_function_node(observer_source, "main") if observer_source else None
    if validator_node is None or main_node is None:
        failures.append("observer lacks reusable strict identity validator wiring")
    else:
        validator_text = ast.get_source_segment(observer_source, validator_node) or ""
        main_text = ast.get_source_segment(observer_source, main_node) or ""
        validator_constants = {item.value for item in ast.walk(validator_node)
                               if isinstance(item, ast.Constant)
                               and isinstance(item.value, str)}
        for component in ("scripts", "parity", "shim", "tinygrad", "__init__.py",
                          "python", "tgrad.py"):
            if component not in validator_constants:
                failures.append(
                    f"strict identity validator does not derive path component {component}")
        validator_args = {arg.arg for arg in validator_node.args.args +
                          validator_node.args.kwonlyargs}
        if "candidate_repo" not in validator_args \
                or "expected_tinygrad_init" in validator_args \
                or "expected_tgrad_module" in validator_args:
            failures.append("strict identity validator accepts caller-supplied path authority")
        for token in ("STRICT_IDENTITY_MARKER", "actual_tinygrad", "actual_tgrad",
                      "candidate_repo.resolve()"):
            if token not in validator_text:
                failures.append(f"strict identity validator does not bind {token}")

        assignments: dict[str, ast.Assign] = {}
        for item in ast.walk(main_node):
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = item

        runner_assignment = assignments.get("runner")
        strict_identity_assignment = assignments.get("strict_identity")
        strict_assignment = assignments.get("strict")
        if runner_assignment is None or not isinstance(runner_assignment.value, ast.Call) \
                or not isinstance(runner_assignment.value.func, ast.Name) \
                or runner_assignment.value.func.id != "strict_runner_path" \
                or not runner_assignment.value.args \
                or not isinstance(runner_assignment.value.args[0], ast.Name) \
                or runner_assignment.value.args[0].id != "candidate_repo":
            failures.append("full observer does not derive invoked runner from candidate repo")

        runner_validations = [item for item in ast.walk(main_node)
                              if isinstance(item, ast.Assign)
                              and isinstance(item.value, ast.Call)
                              and isinstance(item.value.func, ast.Name)
                              and item.value.func.id == "validate_strict_runner"]
        if len(runner_validations) != 1 \
                or not runner_validations[0].value.args \
                or not isinstance(runner_validations[0].value.args[0], ast.Name) \
                or runner_validations[0].value.args[0].id != "candidate_repo":
            failures.append("full observer does not validate the invoked candidate runner")

        def run_call(assignment: ast.Assign | None) -> ast.Call | None:
            if assignment is None or not isinstance(assignment.value, ast.Call):
                return None
            call = assignment.value
            return call if isinstance(call.func, ast.Name) and call.func.id == "run" else None

        identity_run = run_call(strict_identity_assignment)
        normal_run = run_call(strict_assignment)
        if identity_run is None or normal_run is None:
            failures.append("full observer lacks separate identity and normal strict subprocesses")
        else:
            def str_call(node: ast.AST, name: str) -> bool:
                return isinstance(node, ast.Call) \
                    and isinstance(node.func, ast.Name) and node.func.id == "str" \
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Name) \
                    and node.args[0].id == name and not node.keywords

            command = identity_run.args[0] if identity_run.args else None
            exact_verify_command = isinstance(command, ast.List) \
                and len(command.elts) == 3 \
                and str_call(command.elts[0], "python") \
                and str_call(command.elts[1], "runner") \
                and isinstance(command.elts[2], ast.Starred) \
                and isinstance(command.elts[2].value, ast.List) \
                and len(command.elts[2].value.elts) == 1 \
                and isinstance(command.elts[2].value.elts[0], ast.Constant) \
                and command.elts[2].value.elts[0].value == "--verify-only"
            if not exact_verify_command:
                failures.append("strict identity subprocess command is not exact verify-only")
            if "--verify-only" in {item.value for item in ast.walk(normal_run)
                                   if isinstance(item, ast.Constant)
                                   and isinstance(item.value, str)}:
                failures.append("normal strict pytest was replaced by verify-only")
            if strict_identity_assignment.lineno >= strict_assignment.lineno:
                failures.append("strict identity subprocess does not precede normal strict pytest")
            for label, call in (("identity", identity_run), ("normal", normal_run)):
                env_arg = next((keyword.value for keyword in call.keywords
                                if keyword.arg == "env"), None)
                if not isinstance(env_arg, ast.Name) or env_arg.id != "strict_env":
                    failures.append(f"{label} strict subprocess does not share strict_env")
                if not any(isinstance(item, ast.Name) and item.id == "runner"
                           for item in ast.walk(call.args[0])):
                    failures.append(f"{label} strict subprocess bypasses bound runner")

        trust_guards = [item for item in ast.walk(main_node)
                        if isinstance(item, ast.If)
                        and isinstance(item.test, ast.UnaryOp)
                        and isinstance(item.test.op, ast.Not)
                        and isinstance(item.test.operand, ast.Name)
                        and item.test.operand.id == "runner_failures"]
        if len(trust_guards) != 1:
            failures.append("full observer does not fail closed on untrusted runner identity")
        else:
            guarded = trust_guards[0]
            guarded_nodes = {id(item) for statement in guarded.body for item in ast.walk(statement)}
            if strict_identity_assignment is None or id(strict_identity_assignment) not in guarded_nodes \
                    or strict_assignment is None or id(strict_assignment) not in guarded_nodes:
                failures.append("untrusted runner can execute outside its SHA-256 guard")

        execution_flags: dict[bool, list[ast.Assign]] = {False: [], True: []}
        for item in ast.walk(main_node):
            if not isinstance(item, ast.Assign) or not isinstance(item.value, ast.Constant) \
                    or not isinstance(item.value.value, bool):
                continue
            if any(isinstance(target, ast.Subscript)
                   and isinstance(target.value, ast.Name) and target.value.id == "diagnostics"
                   and isinstance(target.slice, ast.Constant)
                   and target.slice.value == "candidate_runner_executed"
                   for target in item.targets):
                execution_flags[item.value.value].append(item)
        if len(execution_flags[False]) != 1 or len(execution_flags[True]) != 1:
            failures.append("candidate runner execution state is not recorded fail-closed")
        elif trust_guards:
            guarded_nodes = {id(item) for statement in trust_guards[0].body
                             for item in ast.walk(statement)}
            if id(execution_flags[True][0]) not in guarded_nodes \
                    or execution_flags[False][0].lineno >= trust_guards[0].lineno:
                failures.append("candidate runner execution state does not follow trust guard")

        validator_assignments = [item for item in ast.walk(main_node)
                                 if isinstance(item, ast.Assign)
                                 and isinstance(item.value, ast.Call)
                                 and isinstance(item.value.func, ast.Name)
                                 and item.value.func.id == "validate_strict_identity"]
        if len(validator_assignments) != 1:
            failures.append("full observer does not consume exactly one identity decision")
        else:
            decision = validator_assignments[0]
            call = decision.value
            expected_refs = (("strict_identity", "returncode"),
                             ("strict_identity", "stdout"),
                             ("strict_identity", "stderr"))
            for index, (base, attribute) in enumerate(expected_refs):
                argument = call.args[index] if index < len(call.args) else None
                if not isinstance(argument, ast.Attribute) \
                        or not isinstance(argument.value, ast.Name) \
                        or argument.value.id != base or argument.attr != attribute:
                    failures.append(
                        f"strict identity decision bypasses {base}.{attribute}")
            candidate_keyword = next((keyword.value for keyword in call.keywords
                                      if keyword.arg == "candidate_repo"), None)
            if not isinstance(candidate_keyword, ast.Name) \
                    or candidate_keyword.id != "candidate_repo":
                failures.append("strict identity decision lacks resolved candidate repo")
            targets = [name.id for target in decision.targets
                       for name in ast.walk(target) if isinstance(name, ast.Name)]
            if sorted(targets) != ["identity_diagnostics", "identity_failures"]:
                failures.append("strict identity decision result is not structurally captured")

            def direct_extend(statement: ast.stmt, name: str) -> bool:
                return isinstance(statement, ast.Expr) \
                    and isinstance(statement.value, ast.Call) \
                    and isinstance(statement.value.func, ast.Attribute) \
                    and isinstance(statement.value.func.value, ast.Name) \
                    and statement.value.func.value.id == "failures" \
                    and statement.value.func.attr == "extend" \
                    and len(statement.value.args) == 1 \
                    and isinstance(statement.value.args[0], ast.Name) \
                    and statement.value.args[0].id == name \
                    and not statement.value.keywords

            if len(trust_guards) == 1:
                direct_decisions = [index for index, statement in enumerate(trust_guards[0].body)
                                    if statement is decision]
                direct_extensions = [index for index, statement in enumerate(trust_guards[0].body)
                                     if direct_extend(statement, "identity_failures")]
                if len(direct_decisions) != 1 or len(direct_extensions) != 1 \
                        or direct_extensions[0] <= direct_decisions[0]:
                    failures.append(
                        "identity failures are not directly propagated after validation")

        extended_names = {
            call.args[0].id for call in ast.walk(main_node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "failures"
            and call.func.attr == "extend" and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
        }
        for required in ("identity_failures", "runner_failures"):
            if required not in extended_names:
                failures.append(f"full observer does not extend failures with {required}")
        if 'diagnostics["strict_runner_identity"] = runner_diagnostics' not in main_text:
            failures.append("full observer does not record candidate runner identity")
        marker_name_references = [item for item in ast.walk(main_node)
                                  if isinstance(item, ast.Name)
                                  and item.id == "STRICT_IDENTITY_MARKER"]
        marker_literals = [item for item in ast.walk(main_node)
                           if isinstance(item, ast.Constant)
                           and isinstance(item.value, str)
                           and "strict Tgrad substitution active" in item.value]
        if marker_name_references or marker_literals:
            failures.append("normal strict pytest stdout incorrectly owns identity proof")
    return failures


def f32_to_bf16(bits: int) -> int:
    bits &= 0xFFFFFFFF
    exponent, mantissa = bits & 0x7F800000, bits & 0x007FFFFF
    if exponent == 0x7F800000 and mantissa != 0:
        return ((bits | 0x00400000) >> 16) & 0xFFFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def f32_corpus() -> list[int]:
    fixed = {
        0x00000000, 0x80000000, 0x00000001, 0x007FFFFF,
        0x00800000, 0x00800001, 0x3F800000, 0xBF800000,
        0x7F7FFFFE, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
        0xFF800000, 0x7F800001, 0x7FA00000, 0x7FC00001, 0x7FFFFFFF,
        0xFF800001, 0xFFA00000, 0xFFC00001, 0xFFFFFFFF,
    }
    uppers = (0x0000, 0x0001, 0x007F, 0x0080, 0x3F80, 0x3F81,
              0x7F7F, 0x7F80, 0x7FC0, 0x8000, 0x8001, 0x807F,
              0x8080, 0xBF80, 0xFF7F, 0xFF80, 0xFFC0)
    for upper in uppers:
        for low in (0x7FFF, 0x8000, 0x8001):
            fixed.add(((upper << 16) | low) & 0xFFFFFFFF)
    return sorted(fixed)


CANDIDATE_PROBE = r'''
import ctypes, gc, json, os, struct
from pathlib import Path
import numpy as np
import tgrad
from tinygrad import dtypes

case = json.loads(Path(os.environ["TGRAD_CAST_CASES"]).read_text())

def direct(raw, shape, dtype):
  buf = tgrad._lib.tgrad_tensor_alloc(len(raw))
  if not buf: raise RuntimeError("direct allocation failed")
  arr = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
  if tgrad._lib.tgrad_tensor_write_bytes(buf, arr, len(raw)) != 0:
    raise RuntimeError("direct upload failed")
  return tgrad.Tensor._from_buffer(buf, len(raw), tuple(shape), dtype)

def counters():
  return {name:int(tgrad._lib.tgrad_metal_counter(i)) for i,name in enumerate(
    ("alloc","free","compile","dispatch","read","write","owned"))}

def registry_count():
  return int(tgrad._lib.tgrad_tensor_registry_count())

def exception_record(exc):
  return {"type":type(exc).__name__, "reason":getattr(exc, "reason", None),
          "message":str(exc)}

def op_without_host_helpers(source, method, target):
  old_numpy, old_to_bytes = tgrad.Tensor.numpy, tgrad.Tensor.to_bytes
  old_init, old_init_numpy = tgrad.Tensor.__init__, tgrad.Tensor._init_from_numpy
  helper_names = ("_bytes_from_numpy", "_numpy_from_bytes",
                  "_bf16_from_fp32", "_fp32_from_bf16")
  old_helpers = {name:getattr(tgrad, name) for name in helper_names}
  lib_names = ("tgrad_tensor_read_bytes", "tgrad_tensor_write_bytes",
               "tgrad_tensor_alloc")
  old_lib = {name:getattr(tgrad._lib, name) for name in lib_names}
  entry_name = "tgrad_tensor_cast" if method == "cast" else "tgrad_tensor_bitcast"
  old_entry = getattr(tgrad._lib, entry_name)
  entry_calls = 0
  def counted_entry(*args):
    nonlocal entry_calls
    entry_calls += 1
    return old_entry(*args)
  def trap(*_a, **_k): raise RuntimeError("host semantic helper called during transform")
  tgrad.Tensor.numpy = trap
  tgrad.Tensor.to_bytes = trap
  tgrad.Tensor.__init__ = trap
  tgrad.Tensor._init_from_numpy = trap
  for name in helper_names: setattr(tgrad, name, trap)
  for name in lib_names: setattr(tgrad._lib, name, trap)
  setattr(tgrad._lib, entry_name, counted_entry)
  tgrad._lib.tgrad_metal_counter_reset()
  try: result = getattr(source, method)(target)
  finally:
    tgrad.Tensor.numpy, tgrad.Tensor.to_bytes = old_numpy, old_to_bytes
    tgrad.Tensor.__init__, tgrad.Tensor._init_from_numpy = old_init, old_init_numpy
    for name,value in old_helpers.items(): setattr(tgrad, name, value)
    for name,value in old_lib.items(): setattr(tgrad._lib, name, value)
    setattr(tgrad._lib, entry_name, old_entry)
  observed = counters()
  observed["abi_entry"] = entry_calls
  return result, observed

def metadata(t):
  return {"handle":int(t._handle), "raw":int(t._buf), "shape":list(t._shape),
          "dtype":t._dtype, "bytes":int(t._size),
          "lean_rank":int(tgrad._lib.tgrad_tensor_rank(t._handle)),
          "lean_dtype":int(tgrad._lib.tgrad_tensor_dtype(t._handle)),
          "lean_bytes":int(tgrad._lib.tgrad_tensor_size_bytes(t._handle)),
          "uop_kind":int(tgrad._lib.tgrad_tensor_uop_kind(t._handle))}

report = {"casts":[], "bitcasts":[], "identities":[], "rejections":[],
          "low_level_status":[], "failures":[], "adoption":[], "protocol":[],
          "borrowed_adoption":[], "release":{}, "borrowed_release":{},
          "lifetime":{}, "index_bound":{}}

f32_raw = b"".join(struct.pack("<I", x) for x in case["f32_bits"])
f32 = direct(f32_raw, [len(case["f32_bits"])], "f32")
bf16, count = op_without_host_helpers(f32, "cast", "bfloat16")
report["casts"].append({"direction":"f32_bf16", "source":metadata(f32),
  "result":metadata(bf16), "source_after":f32.to_bytes().hex(),
  "result_bytes":bf16.to_bytes().hex(), "counters":count})

bf16_raw = bytes.fromhex(case["all_bf16_hex"])
all_bf16 = direct(bf16_raw, [65536], "bf16")
expanded, count = op_without_host_helpers(all_bf16, "cast", dtypes.float32)
report["casts"].append({"direction":"bf16_f32", "source":metadata(all_bf16),
  "result":metadata(expanded), "source_after":all_bf16.to_bytes().hex(),
  "result_bytes":expanded.to_bytes().hex(), "counters":count})

for shape in ([], [1], [17], [255], [256], [257], [2,3], [9,17], [2,3,5]):
  n = 1
  for dim in shape: n *= dim
  raw = b"".join(struct.pack("<I", (0x3c00 + i) << 16) for i in range(n))
  source = direct(raw, shape, "f32")
  result, count = op_without_host_helpers(source, "cast", "bf16")
  report["casts"].append({"direction":"shape_f32_bf16", "shape":shape,
    "source":metadata(source), "result":metadata(result), "counters":count,
    "source_after":source.to_bytes().hex(), "result_bytes":result.to_bytes().hex()})

for dtype, target in (("f32", "float32"), ("bf16", dtypes.bfloat16), ("i32", "int32")):
  size = {"f32":4,"bf16":2,"i32":4}[dtype]
  source = direct(bytes(range(1, size*3+1)), [3], dtype)
  for method in ("cast", "bitcast"):
    result, count = op_without_host_helpers(source, method, target)
    report["identities"].append({"method":method, "dtype":dtype,
      "same_object":result is source, "source":metadata(source),
      "result":metadata(result), "counters":count})

patterns = case["bitcast_patterns"]
raw = b"".join(struct.pack("<I", value) for value in patterns)
for source_dtype, target, direction in (("f32", dtypes.int32, "f32_i32"),
                                         ("i32", "float32", "i32_f32")):
  source = direct(raw, [len(patterns)], source_dtype)
  result, count = op_without_host_helpers(source, "bitcast", target)
  report["bitcasts"].append({"direction":direction, "source":metadata(source),
    "result":metadata(result), "same_bytes":result.to_bytes().hex() == raw.hex(),
    "numpy_bytes":result.numpy().tobytes().hex(),
    "source_after":source.to_bytes().hex(), "result_bytes":result.to_bytes().hex(),
    "base_retained":result._base is source, "counters":count})

lifetime_raw = bytes.fromhex(case["lifetime_raw_hex"])
source = direct(lifetime_raw, [2,2,2], "f32")
shared = source.bitcast(dtypes.int32)
source_handle, source_raw = source._handle, source._buf
tgrad._lib.tgrad_metal_watch_buffer(source_raw)
del source
gc.collect()
early_frees = int(tgrad._lib.tgrad_metal_watch_free_count())
churn = [direct(bytes(32), [8], "f32") for _ in range(70)]
del churn
gc.collect()
after_churn_frees = int(tgrad._lib.tgrad_metal_watch_free_count())
life_bytes = shared.to_bytes().hex()
life_numpy = shared.numpy().tobytes().hex()
life_metadata = metadata(shared)
base_alive = shared._base is not None
del shared
gc.collect()
terminal_frees = int(tgrad._lib.tgrad_metal_watch_free_count())
churn = [direct(bytes(32), [8], "f32") for _ in range(70)]
del churn
gc.collect()
final_frees = int(tgrad._lib.tgrad_metal_watch_free_count())
report["lifetime"] = {"bytes":life_bytes, "numpy_bytes":life_numpy,
  "metadata":life_metadata, "base_alive":base_alive,
  "source_handle":source_handle,
  "source_raw":source_raw, "early_frees":early_frees,
  "after_churn_frees":after_churn_frees, "terminal_frees":terminal_frees,
  "final_frees":final_frees}

def reject(label, source, method, target):
  entry_name = "tgrad_tensor_cast" if method == "cast" else "tgrad_tensor_bitcast"
  original_entry = getattr(tgrad._lib, entry_name)
  entry_calls = 0
  def counted_entry(*args):
    nonlocal entry_calls
    entry_calls += 1
    return original_entry(*args)
  tgrad._lib.tgrad_metal_counter_reset()
  before_registry, before_owned = registry_count(), counters()["owned"]
  setattr(tgrad._lib, entry_name, counted_entry)
  try:
    getattr(source, method)(target)
    outcome = {"type":"accepted", "reason":None, "message":""}
  except Exception as exc:
    outcome = exception_record(exc)
  finally:
    setattr(tgrad._lib, entry_name, original_entry)
  gc.collect()
  report["rejections"].append({"label":label, "outcome":outcome,
    "counters":counters(), "registry_delta":registry_count()-before_registry,
    "owned_delta":counters()["owned"]-before_owned,
    "abi_entry":entry_calls})

f32_small = direct(struct.pack("<4I", *patterns[:4]), [2,2], "f32")
bf16_small = direct(struct.pack("<4H", 0,1,0x3f80,0x7fc0), [2,2], "bf16")
i32_small = direct(struct.pack("<4i", 0,1,-1,7), [2,2], "i32")
reject("cast_f32_i32", f32_small, "cast", dtypes.int32)
reject("cast_bf16_i32", bf16_small, "cast", "int32")
reject("cast_i32_bf16", i32_small, "cast", dtypes.bfloat16)
reject("cast_unknown", f32_small, "cast", "wave10_unknown")
reject("cast_view", f32_small.transpose(), "cast", "bfloat16")
reject("bitcast_f32_bf16", f32_small, "bitcast", dtypes.bfloat16)
reject("bitcast_unknown", f32_small, "bitcast", "wave10_unknown")
reject("bitcast_view", f32_small.transpose(), "bitcast", dtypes.int32)

# Allocation-free nonidentity launch-domain rejection.  The backing buffer is
# deliberately tiny: plan construction must reject the uint guard/count before
# allocation, compilation, dispatch, or buffer-length execution can matter.
huge_shape = (ctypes.c_size_t * 1)(4294967296)
huge_handle = int(tgrad._lib.tgrad_tensor_from_buffer(
  f32_small._buf, huge_shape, 1, 1))
huge_out = ctypes.c_uint64(99)
tgrad._lib.tgrad_metal_counter_reset()
huge_status = int(tgrad._lib.tgrad_tensor_cast(
  huge_handle, 0, ctypes.byref(huge_out)))
report["index_bound"] = {"status":huge_status, "handle":int(huge_out.value),
  "counters":counters()}

for symbol,label in ((tgrad._lib.tgrad_tensor_cast,"cast_invalid_handle"),
                     (tgrad._lib.tgrad_tensor_bitcast,"bitcast_invalid_handle")):
  out_handle = ctypes.c_uint64(99)
  status = int(symbol(0, 1, ctypes.byref(out_handle)))
  report["low_level_status"].append({"label":label, "status":status,
    "handle":int(out_handle.value)})

def fault(label, kind, count):
  source = direct(struct.pack("<I", 0x3f800000) * count, [count], "f32")
  before_registry, before_owned = registry_count(), counters()["owned"]
  tgrad._lib.tgrad_metal_counter_reset()
  tgrad._lib.tgrad_metal_fault_set(kind)
  try:
    source.cast(dtypes.bfloat16)
    outcome = {"type":"accepted", "reason":None, "message":""}
  except Exception as exc:
    outcome = exception_record(exc)
  finally:
    tgrad._lib.tgrad_metal_fault_clear()
  gc.collect()
  report["failures"].append({"label":label, "outcome":outcome,
    "counters":counters(), "registry_delta":registry_count()-before_registry,
    "owned_delta":counters()["owned"]-before_owned})

fault("allocation", 1, 31)
fault("compile", 2, 37)
fault("dispatch", 3, 41)

def adoption_fault(label, symbol_name, replacement):
  original = getattr(tgrad._lib, symbol_name)
  before_registry, before_owned = registry_count(), counters()["owned"]
  tgrad._lib.tgrad_metal_counter_reset()
  setattr(tgrad._lib, symbol_name, replacement(original))
  try:
    f32_small.cast(dtypes.bfloat16)
    outcome = {"type":"accepted", "reason":None, "message":""}
  except Exception as exc:
    outcome = exception_record(exc)
  finally:
    setattr(tgrad._lib, symbol_name, original)
  gc.collect()
  report["adoption"].append({"label":label, "outcome":outcome,
    "counters":counters(), "registry_delta":registry_count()-before_registry,
    "owned_delta":counters()["owned"]-before_owned})

adoption_fault("wrong_dtype", "tgrad_tensor_dtype", lambda _old: (lambda _h: 3))
adoption_fault("wrong_bytes", "tgrad_tensor_size_bytes",
               lambda old: (lambda h: int(old(h)) + 2))
adoption_fault("wrong_shape", "tgrad_tensor_shape_dim",
               lambda old: (lambda h,i: int(old(h,i)) + (1 if int(i) == 0 else 0)))

def borrowed_adoption_fault():
  source_raw = struct.pack("<4I", 0, 0x3f800000, 0x7f800000, 0x7fc00001)
  source = direct(source_raw, [4], "f32")
  original_dtype = tgrad._lib.tgrad_tensor_dtype
  before_registry, before_owned = registry_count(), counters()["owned"]
  tgrad._lib.tgrad_metal_counter_reset()
  setattr(tgrad._lib, "tgrad_tensor_dtype", lambda _h: 1)
  try:
    source.bitcast(dtypes.int32)
    outcome = {"type":"accepted", "reason":None, "message":""}
  except Exception as exc:
    outcome = exception_record(exc)
  finally:
    setattr(tgrad._lib, "tgrad_tensor_dtype", original_dtype)
  gc.collect()
  report["borrowed_adoption"].append({"label":"wrong_dtype_borrowed",
    "outcome":outcome, "counters":counters(),
    "registry_delta":registry_count()-before_registry,
    "owned_delta":counters()["owned"]-before_owned,
    "source_after":source.to_bytes().hex(), "expected_source":source_raw.hex()})

borrowed_adoption_fault()

def protocol_fault(label, replacement):
  original = tgrad._lib.tgrad_tensor_cast
  before_registry, before_owned = registry_count(), counters()["owned"]
  tgrad._lib.tgrad_metal_counter_reset()
  setattr(tgrad._lib, "tgrad_tensor_cast", replacement(original))
  try:
    f32_small.cast(dtypes.bfloat16)
    outcome = {"type":"accepted", "reason":None, "message":""}
  except Exception as exc:
    outcome = exception_record(exc)
  finally:
    setattr(tgrad._lib, "tgrad_tensor_cast", original)
  gc.collect()
  report["protocol"].append({"label":label, "outcome":outcome,
    "counters":counters(), "registry_delta":registry_count()-before_registry,
    "owned_delta":counters()["owned"]-before_owned})

def nonzero_status_wrapper(original):
  def wrapped(handle, target, out_handle):
    status = int(original(handle, target, out_handle))
    return 8 if status == 0 else status
  return wrapped

def zero_handle_wrapper(_original):
  def wrapped(_handle, _target, out_handle):
    ctypes.cast(out_handle, ctypes.POINTER(ctypes.c_uint64))[0] = 0
    return 0
  return wrapped

protocol_fault("nonzero_status_with_handle", nonzero_status_wrapper)
protocol_fault("zero_status_with_zero_handle", zero_handle_wrapper)

borrow_source_raw = struct.pack("<4I", 0, 0x3f800000, 0x7f800000, 0x7fc00001)
borrow_source = direct(borrow_source_raw, [4], "f32")
borrow_before_registry = registry_count()
borrow_before_owned = counters()["owned"]
tgrad._lib.tgrad_metal_counter_reset()
borrow_alias = borrow_source.bitcast(dtypes.int32)
borrow_after_create_registry = registry_count()
borrow_after_create_owned = counters()["owned"]
borrow_create_counts = counters()
borrow_handle = borrow_alias._handle
borrow_alias._fin.detach()
tgrad._lib.tgrad_metal_counter_reset()
borrow_before_first_registry = registry_count()
borrow_before_first_owned = counters()["owned"]
borrow_first = int(tgrad._lib.tgrad_tensor_transform_release(borrow_handle))
borrow_after_first_registry = registry_count()
borrow_after_first_owned = counters()["owned"]
borrow_after_first_counts = counters()
borrow_source_after_first = borrow_source.to_bytes().hex()
borrow_source_read_counts = counters()
tgrad._lib.tgrad_metal_counter_reset()
borrow_before_second_registry = registry_count()
borrow_before_second_owned = counters()["owned"]
borrow_second = int(tgrad._lib.tgrad_tensor_transform_release(borrow_handle))
borrow_after_second_registry = registry_count()
borrow_after_second_owned = counters()["owned"]
borrow_after_second_counts = counters()
del borrow_alias
gc.collect()
report["borrowed_release"] = {
  "creation_registry_delta":borrow_after_create_registry-borrow_before_registry,
  "creation_owned_delta":borrow_after_create_owned-borrow_before_owned,
  "creation_counters":borrow_create_counts,
  "first_status":borrow_first,
  "first_registry_delta":borrow_after_first_registry-borrow_before_first_registry,
  "first_owned_delta":borrow_after_first_owned-borrow_before_first_owned,
  "first_counters":borrow_after_first_counts,
  "source_after_first":borrow_source_after_first,
  "source_read_counters":borrow_source_read_counts,
  "expected_source":borrow_source_raw.hex(),
  "second_status":borrow_second,
  "second_registry_delta":borrow_after_second_registry-borrow_before_second_registry,
  "second_owned_delta":borrow_after_second_owned-borrow_before_second_owned,
  "second_counters":borrow_after_second_counts}

ordinary_before = counters()["owned"]
ordinary_status = int(tgrad._lib.tgrad_tensor_transform_release(f32_small._handle))
ordinary_after = counters()["owned"]
release_source = direct(struct.pack("<I", 0x3f800000) * 13, [13], "f32")
release_result = release_source.cast(dtypes.bfloat16)
release_handle = release_result._handle
release_result._fin.detach()
tgrad._lib.tgrad_metal_counter_reset()
first_release = int(tgrad._lib.tgrad_tensor_transform_release(release_handle))
second_release = int(tgrad._lib.tgrad_tensor_transform_release(release_handle))
del release_result
gc.collect()
report["release"] = {"ordinary_status":ordinary_status,
  "ordinary_owned_delta":ordinary_after-ordinary_before,
  "first_status":first_release, "second_status":second_release,
  "counters":counters()}

print(json.dumps(report, sort_keys=True, separators=(",", ":")))
'''


def validate_candidate(case: dict, candidate: dict) -> list[str]:
    failures: list[str] = []
    cast_rows = candidate.get("casts")
    if not isinstance(cast_rows, list) or len(cast_rows) != 11:
        return ["candidate cast matrix is incomplete"]
    first, second = cast_rows[0], cast_rows[1]
    expected_bf16 = b"".join(
        int(f32_to_bf16(bits)).to_bytes(2, "little") for bits in case["f32_bits"])
    expected_f32 = b"".join(
        (payload << 16).to_bytes(4, "little") for payload in range(65536))
    if first.get("result_bytes") != expected_bf16.hex():
        failures.append("f32-to-bf16 exact-byte corpus differs")
    actual_bf16 = bytes.fromhex(first.get("result_bytes", ""))
    corpus_index = {value: index for index, value in enumerate(case["f32_bits"])}
    for vector in case["nan_vectors"]:
        index = corpus_index[vector["f32"]]
        actual = int.from_bytes(actual_bf16[index * 2:index * 2 + 2], "little") \
            if len(actual_bf16) >= index * 2 + 2 else None
        if actual != vector["bf16"]:
            failures.append(
                f"exact NaN cast row {vector['f32']:#010x} -> {actual!r}, "
                f"expected {vector['bf16']:#06x}")
    if second.get("result_bytes") != expected_f32.hex():
        failures.append("exhaustive bf16-to-f32 expansion differs")
    if first.get("source_after") != case["f32_raw_hex"]:
        failures.append("f32 cast mutated its source")
    if second.get("source_after") != case["all_bf16_hex"]:
        failures.append("bf16 cast mutated its source")
    for row in cast_rows:
        source, result, counts = row.get("source", {}), row.get("result", {}), row.get("counters", {})
        if result.get("shape") != source.get("shape"):
            failures.append(f"cast shape changed for {row.get('direction')}")
        if counts.get("alloc") != 1 or counts.get("dispatch") != 1 \
                or counts.get("read") != 0 or counts.get("write") != 0:
            failures.append(f"cast runtime counters differ for {row.get('direction')}: {counts}")
        if counts.get("abi_entry") != 1:
            failures.append(f"cast did not cross its ABI exactly once: {row.get('direction')}")
        if result.get("handle") == source.get("handle") or result.get("raw") == source.get("raw"):
            failures.append(f"nonidentity cast did not create distinct storage for {row.get('direction')}")
        if result.get("lean_rank") != len(result.get("shape", [])) \
                or result.get("lean_bytes") != result.get("bytes"):
            failures.append(f"cast Lean metadata differs for {row.get('direction')}")
        if result.get("uop_kind") != 0:
            failures.append(f"numeric cast result is not owned materialized storage for {row.get('direction')}")
    for row in cast_rows[2:]:
        expected = b"".join(
            int(f32_to_bf16((0x3C00 + i) << 16)).to_bytes(2, "little")
            for i in range(max(1, len(bytes.fromhex(row["source_after"])) // 4)))
        if row.get("result_bytes") != expected.hex():
            failures.append(f"shape cast skipped or corrupted elements for {row.get('shape')}")

    identities = candidate.get("identities")
    if not isinstance(identities, list) or len(identities) != 6:
        failures.append("identity cast/bitcast matrix is incomplete")
    else:
        for row in identities:
            if not row.get("same_object") or row.get("source") != row.get("result"):
                failures.append(f"identity {row.get('method')} did not preserve object/metadata")
            counts = row.get("counters", {})
            if any(counts.get(key) != 0 for key in ("alloc", "dispatch", "read", "write")):
                failures.append(f"identity {row.get('method')} performed runtime work")
            if counts.get("abi_entry") != 1:
                failures.append(f"identity {row.get('method')} bypassed its ABI entry")

    bitcasts = candidate.get("bitcasts")
    if not isinstance(bitcasts, list) or len(bitcasts) != 2:
        failures.append("bitcast direction matrix is incomplete")
    else:
        for row in bitcasts:
            source, result, counts = row.get("source", {}), row.get("result", {}), row.get("counters", {})
            if not row.get("same_bytes") or row.get("source_after") != row.get("result_bytes"):
                failures.append(f"bitcast bytes differ for {row.get('direction')}")
            if row.get("numpy_bytes") != row.get("result_bytes"):
                failures.append(f"bitcast NumPy readback differs for {row.get('direction')}")
            if result.get("handle") == source.get("handle") or result.get("raw") != source.get("raw"):
                failures.append(f"bitcast handle/storage relation differs for {row.get('direction')}")
            if result.get("shape") != source.get("shape") or result.get("bytes") != source.get("bytes"):
                failures.append(f"bitcast shape/byte relation differs for {row.get('direction')}")
            if not row.get("base_retained"):
                failures.append(f"bitcast source lifetime not retained for {row.get('direction')}")
            if source.get("uop_kind") != 0 or result.get("uop_kind") in (None, 0):
                failures.append(f"bitcast provenance/materialized classification differs for {row.get('direction')}")
            if any(counts.get(key) != 0 for key in ("alloc", "dispatch", "read", "write")):
                failures.append(f"bitcast performed runtime work for {row.get('direction')}")
            if counts.get("abi_entry") != 1:
                failures.append(f"bitcast did not cross its ABI exactly once: {row.get('direction')}")
    lifetime = candidate.get("lifetime", {})
    if lifetime.get("bytes") != case["lifetime_raw_hex"] \
            or lifetime.get("numpy_bytes") != case["lifetime_raw_hex"] \
            or not lifetime.get("base_alive"):
        failures.append("bitcast lifetime/churn result differs")
    if [lifetime.get(name) for name in ("early_frees", "after_churn_frees",
                                        "terminal_frees", "final_frees")] != [0, 0, 1, 1]:
        failures.append(f"bitcast raw storage release count is not exactly once: {lifetime}")
    rejections = candidate.get("rejections")
    if not isinstance(rejections, list) or len(rejections) != 8:
        failures.append("cast/bitcast rejection matrix is incomplete")
    else:
        expected_reasons = {
            "cast_f32_i32":"unsupportedPair", "cast_bf16_i32":"unsupportedPair",
            "cast_i32_bf16":"unsupportedPair", "cast_unknown":"invalidDtype",
            "cast_view":"nonBuffer", "bitcast_f32_bf16":"unequalItemSize",
            "bitcast_unknown":"invalidDtype", "bitcast_view":"nonBuffer",
        }
        for row in rejections:
            outcome = row.get("outcome", {})
            if outcome.get("type") == "accepted":
                failures.append(f"unsupported transform was admitted: {row.get('label')}")
            if outcome.get("reason") != expected_reasons.get(row.get("label")):
                failures.append(f"structured rejection reason differs: {row}")
            counts = row.get("counters", {})
            if any(counts.get(key) != 0 for key in ("alloc", "compile", "dispatch", "read", "write")):
                failures.append(f"rejected transform performed runtime work: {row.get('label')}")
            if row.get("registry_delta") != 0 or row.get("owned_delta") != 0:
                failures.append(f"rejected transform changed lifecycle state: {row.get('label')}")
            if row.get("abi_entry") != 1:
                failures.append(f"rejected transform bypassed its ABI entry: {row.get('label')}")

    index_bound = candidate.get("index_bound", {})
    if index_bound.get("status") != 6 or index_bound.get("handle") != 0:
        failures.append(f"nonidentity UInt32 index bound did not fail closed: {index_bound}")
    index_counts = index_bound.get("counters", {})
    if any(index_counts.get(key) != 0 for key in
           ("alloc", "compile", "dispatch", "read", "write")):
        failures.append(f"index-bound rejection performed runtime work: {index_bound}")

    low_level = candidate.get("low_level_status")
    if not isinstance(low_level, list) or len(low_level) != 2:
        failures.append("low-level structured status matrix is incomplete")
    else:
        for row in low_level:
            if row.get("status") != 1 or row.get("handle") != 0:
                failures.append(f"invalid handle status collapsed: {row}")

    injected = candidate.get("failures")
    expected_injected = {"allocation":"allocationFailed", "compile":"compileFailed",
                         "dispatch":"dispatchFailed"}
    if not isinstance(injected, list) or len(injected) != 3:
        failures.append("runtime failure-injection matrix is incomplete")
    else:
        for row in injected:
            if row.get("outcome", {}).get("reason") != expected_injected.get(row.get("label")):
                failures.append(f"runtime failure reason differs: {row}")
            if row.get("registry_delta") != 0 or row.get("owned_delta") != 0:
                failures.append(f"runtime failure leaked partial transform state: {row}")

    adoption = candidate.get("adoption")
    if not isinstance(adoption, list) or len(adoption) != 3:
        failures.append("transactional metadata-adoption matrix is incomplete")
    else:
        for row in adoption:
            if row.get("outcome", {}).get("reason") != "transportMetadataMismatch":
                failures.append(f"metadata adoption did not fail closed: {row}")
            if row.get("registry_delta") != 0 or row.get("owned_delta") != 0:
                failures.append(f"metadata adoption leaked registered/storage state: {row}")
            counts = row.get("counters", {})
            if counts.get("alloc") != 1 or counts.get("dispatch") != 1 or counts.get("free") != 1:
                failures.append(f"metadata adoption did not release exactly one cast result: {row}")

    borrowed_adoption = candidate.get("borrowed_adoption")
    if not isinstance(borrowed_adoption, list) or len(borrowed_adoption) != 1:
        failures.append("borrowed metadata-adoption matrix is incomplete")
    else:
        row = borrowed_adoption[0]
        if row.get("outcome", {}).get("reason") != "transportMetadataMismatch":
            failures.append(f"borrowed metadata adoption did not fail closed: {row}")
        if row.get("registry_delta") != 0 or row.get("owned_delta") != 0:
            failures.append(f"borrowed metadata adoption leaked lifecycle state: {row}")
        counts = row.get("counters", {})
        if any(counts.get(key) != 0 for key in
               ("alloc", "free", "compile", "dispatch", "read", "write")):
            failures.append(f"borrowed metadata adoption touched Metal storage: {row}")
        if row.get("source_after") != row.get("expected_source"):
            failures.append("borrowed metadata cleanup invalidated source storage")

    protocol = candidate.get("protocol")
    if not isinstance(protocol, list) or len(protocol) != 2:
        failures.append("malformed transaction protocol matrix is incomplete")
    else:
        for row in protocol:
            if row.get("outcome", {}).get("reason") != "transportMetadataMismatch":
                failures.append(f"malformed transaction did not fail closed: {row}")
            if row.get("registry_delta") != 0 or row.get("owned_delta") != 0:
                failures.append(f"malformed transaction leaked lifecycle state: {row}")
        nonzero = next((row for row in protocol
                        if row.get("label") == "nonzero_status_with_handle"), {})
        if nonzero.get("counters", {}).get("free") != 1:
            failures.append("nonzero-status transform handle was not released exactly once")

    release = candidate.get("release", {})
    if release.get("ordinary_status") != 1 or release.get("ordinary_owned_delta") != 0:
        failures.append("transform release escaped containment onto an ordinary source")
    if release.get("first_status") != 0 or release.get("second_status") != 1 \
            or release.get("counters", {}).get("free") != 1:
        failures.append(f"transform release is not idempotent/exactly-once: {release}")

    borrowed = candidate.get("borrowed_release", {})
    if borrowed.get("creation_registry_delta") != 1 \
            or borrowed.get("creation_owned_delta") != 0:
        failures.append(f"borrowed alias creation lifecycle differs: {borrowed}")
    if any(borrowed.get("creation_counters", {}).get(key) != 0
           for key in ("alloc", "free", "compile", "dispatch", "read", "write")):
        failures.append(f"borrowed alias creation touched Metal storage: {borrowed}")
    if borrowed.get("first_status") != 0 \
            or borrowed.get("first_registry_delta") != -1 \
            or borrowed.get("first_owned_delta") != 0 \
            or any(borrowed.get("first_counters", {}).get(key) != 0
                   for key in ("alloc", "free", "compile", "dispatch", "read", "write")):
        failures.append(f"first borrowed alias release did not unregister exactly once: {borrowed}")
    if borrowed.get("source_after_first") != borrowed.get("expected_source"):
        failures.append("borrowed alias release invalidated source storage")
    source_read = borrowed.get("source_read_counters", {})
    if source_read.get("read") != 1 or any(source_read.get(key) != 0 for key in
            ("alloc", "free", "compile", "dispatch", "write")):
        failures.append(f"borrowed source read was not isolated: {borrowed}")
    if borrowed.get("second_status") != 1 \
            or borrowed.get("second_registry_delta") != 0 \
            or borrowed.get("second_owned_delta") != 0 \
            or any(borrowed.get("second_counters", {}).get(key) != 0
                   for key in ("alloc", "free", "compile", "dispatch", "read", "write")):
        failures.append(f"second borrowed alias release was not inert: {borrowed}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mode", choices=("preimplementation", "full"), default="full")
    parser.add_argument("--candidate-head")
    parser.add_argument("--candidate-tree")
    parser.add_argument("--require-candidate-clean", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    candidate_repo = (args.candidate_repo or repo).resolve()
    checkout = args.checkout.resolve()
    python = args.python if args.python.is_absolute() else Path.cwd() / args.python
    failures = architecture_failures(candidate_repo)
    hygiene, hygiene_paths = source_hygiene_failures(candidate_repo)
    failures.extend(hygiene)
    diagnostics: dict[str, object] = {
        "mode": args.mode,
        "source_hygiene_checked_paths": hygiene_paths,
        "cursor_preflight": "agent --list-models exited 139: SecItemCopyMatching failed -50",
    }
    runner = strict_runner_path(candidate_repo)
    runner_failures, runner_diagnostics = validate_strict_runner(candidate_repo)
    failures.extend(runner_failures)
    diagnostics["strict_runner_identity"] = runner_diagnostics
    candidate_identity = {
        "top_level": git_text(candidate_repo, "rev-parse", "--show-toplevel"),
        "head": git_text(candidate_repo, "rev-parse", "HEAD"),
        "tree": git_text(candidate_repo, "rev-parse", "HEAD^{tree}"),
        "status": git_text(candidate_repo, "status", "--porcelain", "--untracked-files=all"),
    }
    if not os.path.samefile(candidate_identity["top_level"], candidate_repo):
        failures.append("candidate repository discovery escaped subject directory")
    if args.candidate_head and candidate_identity["head"] != args.candidate_head:
        failures.append("candidate HEAD identity mismatch")
    if args.candidate_tree and candidate_identity["tree"] != args.candidate_tree:
        failures.append("candidate tree identity mismatch")
    if args.require_candidate_clean and candidate_identity["status"]:
        failures.append("candidate subject is not clean")

    for relative, expected in BASE_IDENTITIES.items():
        if sha256(repo / relative) != expected:
            failures.append(f"canonical observer/classifier identity changed: {relative}")
    test_source = checkout / "test" / "backend" / "test_dtype.py"
    if sha256(test_source) != TEST_SOURCE_SHA256:
        failures.append("pinned dtype test source identity changed")
    test_text = test_source.read_text()
    for required in (".cast(dtypes.bfloat16)", ".cast(dtypes.float32)",
                     '.cast("bfloat16")', ".bitcast(dtypes.int32)"):
        if required not in test_text:
            failures.append(f"contracted foreign test source missing {required}")

    oracle_ok, oracle_detail = verify(checkout)
    diagnostics["oracle_verification"] = oracle_detail
    if not oracle_ok:
        failures.append("pinned oracle source closure failed")

    foreign_tail = ""
    candidate_payload: dict = {}
    case = {
        "f32_bits": f32_corpus(),
        "all_bf16_hex": b"".join(i.to_bytes(2, "little") for i in range(65536)).hex(),
        "bitcast_patterns": [0x00000000, 0x80000000, 0x3F800000, 0xBF800000,
                             0x7F800000, 0xFF800000, 0x7F800001, 0x7FC01234,
                             0xFF800001, 0xFFC05678],
    }
    case["nan_vectors"] = [
        {"f32": value, "bf16": f32_to_bf16(value)}
        for value in (0x7FA00000, 0x7F800001, 0x7FC00001,
                      0xFFA00000, 0xFF800001, 0xFFC00001)
    ]
    case["f32_raw_hex"] = b"".join(
        int(value).to_bytes(4, "little") for value in case["f32_bits"]).hex()
    case["bitcast_raw_hex"] = b"".join(
        int(value).to_bytes(4, "little") for value in case["bitcast_patterns"]).hex()
    case["lifetime_raw_hex"] = b"".join(
        int(value).to_bytes(4, "little") for value in case["bitcast_patterns"][:8]).hex()

    with tempfile.TemporaryDirectory(prefix="tgrad-wave10-cast-bitcast-") as name:
        temp = Path(name)
        snapshot = temp / "oracle"
        snapshot.mkdir()
        snapshot_oracle(checkout, snapshot)
        pytest_args = ["-q", "-p", "no:cacheprovider", "--hypothesis-seed=0", *TARGETS]
        foreign = run([str(python), "-m", "pytest", *pytest_args], cwd=snapshot,
                      env=isolated_env(temp, PYTHONPATH=str(snapshot), DEV="CPU"))
        foreign_tail = (foreign.stdout + foreign.stderr)[-3000:]
        diagnostics["foreign_returncode"] = foreign.returncode
        diagnostics["foreign_tail"] = foreign_tail
        if foreign.returncode != 0 or not re.search(r"\b3 passed\b", foreign.stdout):
            failures.append("exact pinned foreign leaf selection is not 3 passed")

        if args.mode == "full":
            cases_path = temp / "cases.json"
            cases_path.write_text(json.dumps(case, sort_keys=True, separators=(",", ":")))
            candidate_path = os.pathsep.join((
                str(candidate_repo / "scripts" / "parity" / "shim"),
                str(candidate_repo / "python"),
            ))
            candidate = run([str(python), "-c", CANDIDATE_PROBE], cwd=candidate_repo,
                            env=isolated_env(temp, PYTHONPATH=candidate_path,
                                TGRAD_LIB=str(args.lib.resolve()),
                                TGRAD_CAST_CASES=str(cases_path),
                                DYLD_LIBRARY_PATH=str(candidate_repo / ".lake" / "build" / "lib")))
            diagnostics["candidate_probe_returncode"] = candidate.returncode
            if candidate.returncode != 0:
                failures.append("candidate cast/bitcast probe failed")
                diagnostics["candidate_probe_stdout"] = candidate.stdout[-5000:]
                diagnostics["candidate_probe_stderr"] = candidate.stderr[-5000:]
            else:
                try:
                    candidate_payload = json.loads(candidate.stdout)
                except json.JSONDecodeError as exc:
                    failures.append(f"candidate probe JSON invalid: {exc}")

            strict_env = isolated_env(
                temp,
                PYTHONPATH=str(candidate_repo / "python"),
                TGRAD_LIB=str(args.lib.resolve()),
                DYLD_LIBRARY_PATH=str(candidate_repo / ".lake" / "build" / "lib"),
            )
            diagnostics["candidate_runner_executed"] = False
            if not runner_failures:
                strict_identity = run(
                    [str(python), str(runner), *["--verify-only"]],
                    cwd=snapshot,
                    env=strict_env,
                )
                identity_failures, identity_diagnostics = validate_strict_identity(
                    strict_identity.returncode,
                    strict_identity.stdout,
                    strict_identity.stderr,
                    candidate_repo=candidate_repo,
                )
                diagnostics["strict_identity"] = identity_diagnostics
                failures.extend(identity_failures)
                strict = run([str(python), str(runner), *pytest_args], cwd=snapshot,
                             env=strict_env)
                diagnostics["strict_returncode"] = strict.returncode
                diagnostics["strict_tail"] = (strict.stdout + strict.stderr)[-3000:]
                diagnostics["candidate_runner_executed"] = True
                if strict.returncode != 0 or not re.search(r"\b3 passed\b", strict.stdout):
                    failures.append("strict substitution exact leaf selection is not 3 passed")
        else:
            diagnostics["candidate_probe_executed"] = False
            failures.append("preimplementation mode deliberately does not execute candidate semantics")

        oracle_after, detail_after = verify(checkout)
        diagnostics["oracle_reverification"] = detail_after
        if not oracle_after:
            failures.append("live oracle changed during observation")

    if candidate_payload:
        failures.extend(validate_candidate(case, candidate_payload))

    tracked_paths = [
        Path(__file__), candidate_repo / "Tgrad.lean", candidate_repo / "Tgrad" / "Cast.lean",
        candidate_repo / "Tgrad" / "Dtype.lean", candidate_repo / "Tgrad" / "Tensor.lean",
        candidate_repo / "Tgrad" / "Runtime" / "MetalProgram.lean",
        candidate_repo / "Tgrad" / "PythonFFI.lean", candidate_repo / "c" / "tgrad_python.c",
        candidate_repo / "c" / "metal_alloc.m", candidate_repo / "c" / "metal_alloc_lean.c",
        candidate_repo / "python" / "tgrad.py",
        runner,
        candidate_repo / "scripts" / "parity" / "shim" / "tinygrad" / "tensor.py",
        args.lib.resolve(), candidate_repo / ".lake" / "build" / "lib" / "libtgrad_Tgrad.dylib",
        test_source,
    ]
    hashes = {}
    for path in tracked_paths:
        if path.exists():
            key = str(path.relative_to(candidate_repo) if path.is_relative_to(candidate_repo) else path)
            hashes[key] = sha256(path)
    report = {
        "schema": "tgrad.wave10.cast_bitcast.v1",
        "oracle": {"revision": EXPECTED.revision, "tree": EXPECTED.tree},
        "candidate_identity": candidate_identity,
        "target_count": len(TARGETS),
        "target_manifest_sha256": hashlib.sha256("\n".join(TARGETS).encode()).hexdigest(),
        "f32_corpus_count": len(case["f32_bits"]),
        "f32_corpus_sha256": hashlib.sha256(bytes.fromhex(case["f32_raw_hex"])).hexdigest(),
        "exact_nan_vectors": case["nan_vectors"],
        "bf16_exhaustive_count": 65536,
        "bitcast_pattern_count": len(case["bitcast_patterns"]),
        "python_launcher": str(python),
        "python_resolved": str(python.resolve()),
        "hashes": hashes,
        "diagnostics": diagnostics,
        "failures": failures,
        "result": "pass" if not failures else "fail",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

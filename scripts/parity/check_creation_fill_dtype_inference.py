#!/usr/bin/env python3
"""Foreign-paired, allocation-free check for Tensor.full dtype inference.

Pinned tinygrad supplies the public semantic dtype for each primitive fill and
explicit override. Tgrad's candidate side asks the same Lean resolver used by
the allocation path, without allocating or dispatching a tensor.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PINNED_REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
PINNED_TREE = "855cca3b00c38841a6d3a043284f3a2ca696d4b0"

SEMANTIC_CODES = {
    "bfloat16": 0,
    "float32": 1,
    "int32": 3,
    "bool": 4,
    "int64": 11,
    "float16": 2,
}
COMPUTE_SUPPORTED = frozenset({"bfloat16", "float32", "int32"})

FOREIGN_PROBE = r'''
import json
from tinygrad import Tensor, dtypes

canonical = ("bfloat16", "float32", "int32", "bool", "int64", "float16")
def name(dtype):
  matches = [entry for entry in canonical if dtype is getattr(dtypes, entry)]
  if len(matches) != 1: raise RuntimeError(f"non-canonical dtype {dtype!r}")
  return matches[0]

def inferred(value, dtype=None):
  return name(Tensor.full((2, 3), value, dtype=dtype).dtype)

rows = {
  "primitive_int": inferred(4),
  "primitive_float": inferred(4.0),
  "primitive_bool": inferred(True),
  "explicit_bf16_over_int": inferred(4, dtypes.bfloat16),
  "explicit_i32_over_float": inferred(4.0, dtypes.int32),
  "explicit_f32_over_bool": inferred(True, dtypes.float32),
}
old_int, old_float = dtypes.default_int, dtypes.default_float
try:
  dtypes.default_int = dtypes.int64
  rows["runtime_int64_default"] = inferred(4)
  dtypes.default_int = old_int
  dtypes.default_float = dtypes.bfloat16
  rows["runtime_bf16_default"] = inferred(4.0)
  dtypes.default_float = dtypes.float16
  rows["runtime_f16_default"] = inferred(4.0)
finally:
  dtypes.default_int, dtypes.default_float = old_int, old_float
print(json.dumps(rows, sort_keys=True))
'''

CANDIDATE_PROBE = r'''
import json
import tgrad
from tinygrad import dtypes

def resolved(value, dtype=None):
  return tgrad._creation_dtype_resolve(
    tgrad._fill_scalar_tag(value), tgrad._creation_dtype_code(dtype))

rows = {
  "primitive_int": resolved(4),
  "primitive_float": resolved(4.0),
  "primitive_bool": resolved(True),
  "explicit_bf16_over_int": resolved(4, dtypes.bfloat16),
  "explicit_i32_over_float": resolved(4.0, dtypes.int32),
  "explicit_f32_over_bool": resolved(True, dtypes.float32),
}
old_int, old_float = dtypes.default_int, dtypes.default_float
try:
  dtypes.default_int = dtypes.int64
  rows["runtime_int64_default"] = resolved(4)
  dtypes.default_int = old_int
  dtypes.default_float = dtypes.bfloat16
  rows["runtime_bf16_default"] = resolved(4.0)
  dtypes.default_float = dtypes.float16
  rows["runtime_f16_default"] = resolved(4.0)
finally:
  dtypes.default_int, dtypes.default_float = old_int, old_float

rows["invalid_tag_default"] = tgrad._creation_dtype_resolve(254, 255)
rows["invalid_tag_explicit_f32"] = tgrad._creation_dtype_resolve(254, 1)
rows["invalid_dtype"] = tgrad._creation_dtype_resolve(1, 253)
print(json.dumps(rows, sort_keys=True))
'''


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True).strip()


def run_json(python: Path, source: str, pythonpath: str,
             extra_env: dict[str, str]) -> tuple[object | None, str | None]:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    env.update(extra_env)
    cp = subprocess.run(
        [str(python), "-c", source], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if cp.returncode != 0:
        return None, f"probe rc={cp.returncode}: {cp.stderr.strip() or cp.stdout.strip()}"
    try:
        return json.loads(cp.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"probe emitted invalid JSON: {exc}"


def function_source(tree: ast.Module, source: str, name: str) -> str | None:
    node = next(
        (item for item in tree.body
         if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
         and item.name == name), None)
    return ast.get_source_segment(source, node) if node is not None else None


def architecture_failures(repo: Path) -> list[str]:
    failures: list[str] = []
    python_source = (repo / "python" / "tgrad.py").read_text()
    tree = ast.parse(python_source)
    helper = function_source(tree, python_source, "_fill_scalar_tag")
    if helper is None:
        failures.append("missing Python fill-scalar marshalling helper")
    else:
        for forbidden in (
            "_DTYPE_", "_SUPPORTED_DTYPES", "_dtype_default",
            "_dtype_query", "default_int", "default_float",
        ):
            if forbidden in helper:
                failures.append(
                    f"Python fill-tag helper owns dtype semantics: {forbidden}")
        helper_tree = ast.parse(helper)
        for node in ast.walk(helper_tree):
            if isinstance(node, ast.Return) and node.value is not None:
                answer = ast.unparse(node.value)
                if not answer.startswith("_FILL_SCALAR_"):
                    failures.append(
                        f"Python fill-tag helper returns non-tag answer: {answer}")

    tensor = next(
        (node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == "Tensor"), None)
    full = next(
        (node for node in (tensor.body if tensor else [])
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "full"), None)
    full_source = ast.get_source_segment(python_source, full) if full else ""
    if "fill_tag = _fill_scalar_tag(fill_value)" not in full_source:
        failures.append("Tensor.full does not preserve the scalar fill tag")
    if "shape_arr, len(shape), float(fill_value), fill_tag, code" not in full_source:
        failures.append("Tensor.full does not pass the fill tag to the allocation path")

    lean_source = (repo / "Tgrad" / "PythonFFI.lean").read_text()
    if "def resolveCreationDtype" not in lean_source:
        failures.append("missing shared Lean creation dtype resolver")
    if lean_source.count("← resolveCreationDtype fillTag dtypeCode") != 2:
        failures.append(
            "allocation-free query and tensorFull do not share one Lean resolver")
    if "def creationDtype?" in lean_source:
        failures.append("legacy parallel creation dtype resolver still exists")

    c_source = (repo / "c" / "tgrad_python.c").read_text()
    if "uint8_t fill_tag, uint8_t dtype_code" not in c_source:
        failures.append("C full boundary does not preserve the fill tag")
    c_compact = re.sub(r"\s+", "", c_source)
    if "tgrad_creation_dtype_resolve_lean(fill_tag,dtype_code)" not in c_compact:
        failures.append("C allocation-free query is not wired to Lean")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    repo = args.repo.resolve()
    candidate = (args.candidate_repo or repo).resolve()
    checkout = args.checkout.resolve()
    identity = {
        "revision": git(checkout, "rev-parse", "HEAD"),
        "tree": git(checkout, "rev-parse", "HEAD^{tree}"),
    }
    expected_identity = {"revision": PINNED_REVISION, "tree": PINNED_TREE}
    failures = architecture_failures(candidate)
    if identity != expected_identity:
        failures.append("oracle identity mismatch")

    foreign, foreign_error = run_json(
        args.python, FOREIGN_PROBE, str(checkout), {"DEV": "CPU"})
    candidate_path = os.pathsep.join([
        str(candidate / "scripts" / "parity" / "shim"),
        str(candidate / "python"),
    ])
    candidate_rows, candidate_error = run_json(
        args.python, CANDIDATE_PROBE, candidate_path,
        {"TGRAD_LIB": str(args.lib.resolve())},
    )
    if foreign_error:
        failures.append(f"foreign {foreign_error}")
    if candidate_error:
        failures.append(f"candidate {candidate_error}")

    expected_candidate = None
    if isinstance(foreign, dict):
        expected_candidate = {
            label: (SEMANTIC_CODES[name] if name in COMPUTE_SUPPORTED else 255)
            for label, name in foreign.items()
        }
        expected_candidate.update({
            "invalid_tag_default": 255,
            "invalid_tag_explicit_f32": 1,
            "invalid_dtype": 255,
        })
        if candidate_rows != expected_candidate:
            failures.append("candidate resolver differs from foreign inference/admission")

    result = {
        "status": "green" if not failures else "red",
        "oracle": identity,
        "foreign": foreign,
        "candidate": candidate_rows,
        "expected_candidate": expected_candidate,
        "architecture_failures": architecture_failures(candidate),
        "probe_errors": {
            "foreign": foreign_error, "candidate": candidate_error,
        },
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

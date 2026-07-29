#!/usr/bin/env python3
"""Foreign-paired, allocation-free check for Lean-owned arange resolution."""
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
    "bfloat16": 0, "float32": 1, "float16": 2, "int32": 3,
    "int8": 6, "int64": 11,
}
COMPUTE_ADMITTED_CODES = frozenset({0, 1, 3})

FOREIGN_PROBE = r'''
import json
from tinygrad import Tensor, dtypes

def dtype_name(dtype):
  for name in ("bfloat16", "float32", "float16", "int32", "int8", "int64"):
    if dtype is getattr(dtypes, name): return name
  raise RuntimeError(f"non-canonical dtype {dtype!r}")

def row(start, stop=None, step=1, dtype=None, values=True):
  try:
    out = Tensor.arange(start, stop, step, dtype=dtype)
    return {
      "shape": list(out.shape), "dtype": dtype_name(out.dtype),
      "values": out.numpy().tolist() if values else None,
    }
  except BaseException as exc:
    return {"error": type(exc).__name__, "message": str(exc)}

rows = {
  "stop_omitted_default_i32": row(10),
  "explicit_f32": row(2, 9, 2, dtypes.float32),
  "ascending_nonunit": row(5, 10, 3),
  "descending_nonunit": row(10, 5, -3),
  "long_nonunit": row(1, 78, 2),
  "empty_direction": row(5, 10, -1),
  "zero_step": row(0, 10, 0),
  "overflow_precedes_zero_i32": row(2**40, 0, 0, dtypes.int32, values=False),
  "overflow_precedes_zero_i8": row(2**40, 0, 0, dtypes.int8, values=False),
  "signed_boundary": row(-(2**31), 2**31, 2**31-1),
  "large_boundary_length": row(0, 2**31, 1, values=False),
  "beyond_int64_unrepresentable": row(2**70, 2**70 + 1, 1, values=False),
  "pinned_int8_admission_order": row(125, 130, 3, dtypes.int8, values=False),
  "pinned_empty_one_sided_int8": row(128, 0, 1, dtypes.int8, values=False),
  "explicit_unsupported_f16": row(0, 4, 1, dtypes.float16, values=False),
}
old = dtypes.default_int
try:
  dtypes.default_int = dtypes.int64
  rows["inferred_unsupported_i64"] = row(0, 4, 1, values=False)
finally:
  dtypes.default_int = old
print(json.dumps(rows, sort_keys=True))
'''

CANDIDATE_PROBE = r'''
import json
import tgrad
from tinygrad import dtypes

DTYPE_BY_CODE = {3: dtypes.int32, 6: dtypes.int8}

def row(start, stop=None, step=1, dtype_code=255):
  decision = tgrad._range_decision(start, stop, step, dtype_code)
  if decision != 0:
    dtype = None if dtype_code == 255 else DTYPE_BY_CODE[dtype_code]
    try:
      tgrad.Tensor.arange(start, stop, step, dtype=dtype)
    except BaseException as exc:
      return {"error": type(exc).__name__, "reason": decision}
    return {"error": None, "reason": decision}
  return tgrad._range_resolution(start, stop, step, dtype_code)

rows = {
  "stop_omitted_default_i32": row(10),
  "explicit_f32": row(2, 9, 2, 1),
  "ascending_nonunit": row(5, 10, 3),
  "descending_nonunit": row(10, 5, -3),
  "long_nonunit": row(1, 78, 2),
  "empty_direction": row(5, 10, -1),
  "zero_step": row(0, 10, 0),
  "overflow_precedes_zero_i32": row(2**40, 0, 0, 3),
  "overflow_precedes_zero_i8": row(2**40, 0, 0, 6),
  "signed_boundary": row(-(2**31), 2**31, 2**31-1),
  "large_boundary_length": row(0, 2**31, 1),
  "beyond_int64_unrepresentable": row(2**70, 2**70 + 1, 1),
  "pinned_int8_admission_order": row(125, 130, 3, 6),
  "pinned_empty_one_sided_int8": row(128, 0, 1, 6),
  "explicit_unsupported_f16": row(0, 4, 1, 2),
}
old = tgrad._dtype_default(0)
try:
  if not tgrad._dtype_set_default(0, 11):
    raise RuntimeError("Lean rejected int64 default used by the probe")
  rows["inferred_unsupported_i64"] = row(0, 4, 1)
finally:
  if not tgrad._dtype_set_default(0, old):
    raise RuntimeError("failed to restore Lean integer default")
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


def class_method_source(source: str, name: str) -> str | None:
    tree = ast.parse(source)
    tensor = next(
        (node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == "Tensor"), None)
    method = next(
        (node for node in (tensor.body if tensor else [])
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == name), None)
    return ast.get_source_segment(source, method) if method else None


def architecture_failures(repo: Path) -> list[str]:
    failures: list[str] = []
    python_source = (repo / "python" / "tgrad.py").read_text()
    arange = class_method_source(python_source, "arange")
    if arange is None:
        failures.append("product Tensor.arange is missing")
    else:
        for forbidden in (
            "np.", "numpy", "list(range", "Tensor.full(", "from_numpy",
            "output_len", "ceil", "//", "%",
        ):
            if forbidden in arange:
                failures.append(f"Python arange owns progression semantics: {forbidden}")
        for required in (
            "_range_scalar(start)", "_range_scalar(stop)",
            "_range_scalar(step)", "_creation_dtype_code(dtype)",
            "_lib.tgrad_tensor_arange",
        ):
            if required not in arange:
                failures.append(f"Python arange missing marshalling path: {required}")

    lean_source = (repo / "Tgrad" / "PythonFFI.lean").read_text()
    if "def resolveRange" not in lean_source:
        failures.append("missing shared Lean range resolver")
    if lean_source.count("← resolveRange start stop step hasStop dtypeCode") != 2:
        failures.append("query and allocation do not share one Lean range resolver")
    if "def allocateRangeTensor" not in lean_source:
        failures.append("missing Lean range allocator")
    if "allocateRangeTensor spec" not in lean_source:
        failures.append("allocating path does not consume the resolved Lean range")
    if "rangeByteCountFitsBoundary spec" not in lean_source:
        failures.append("allocating path does not prove byte-count narrowing safe")

    creation_source = (repo / "Tgrad" / "Renderer" / "Creation.lean").read_text()
    if "def rangeKernelDecl" not in creation_source:
        failures.append("missing Lean-generated arithmetic-progression kernel")
    if "spec.start" not in lean_source or "spec.step" not in lean_source:
        failures.append("Lean allocation is not visibly bound to resolved progression values")

    c_source = re.sub(
        r"\s+", "", (repo / "c" / "tgrad_python.c").read_text())
    for symbol in ("tgrad_range_query", "tgrad_tensor_arange"):
        if symbol not in c_source:
            failures.append(f"C boundary missing {symbol}")
    range_binding = re.search(
        r"_RANGE_BOUNDARY_ARGTYPES\s*=\s*\[(.*?)\]", python_source, re.S)
    if range_binding is None or "ctypes.c_char_p" not in range_binding.group(1):
        failures.append("Python range boundary does not carry decimal integer payloads")
    if range_binding is not None and "ctypes.c_int64" in range_binding.group(1):
        failures.append("Python range boundary narrows integers to int64")
    if "lean_cstr_to_int" not in c_source:
        failures.append("C range boundary does not construct arbitrary Lean Int values")
    for signature in re.findall(r"uint64_ttgrad_(?:range_query|tensor_arange)\((.*?)\)\{", c_source):
        if "int64_t" in signature:
            failures.append("C range boundary narrows integers to int64")
    return failures


def expected_candidate(foreign: object) -> dict[str, object] | None:
    if not isinstance(foreign, dict):
        return None
    expected: dict[str, object] = {}
    required_reason = {
        "zero_step": 1,
        "beyond_int64_unrepresentable": 4,
        "overflow_precedes_zero_i32": 4,
        "overflow_precedes_zero_i8": 4,
    }
    for label, raw in foreign.items():
        if not isinstance(raw, dict):
            return None
        if "error" in raw:
            expected[label] = {
                "error": raw["error"], "reason": required_reason.get(label),
            }
            continue
        code = SEMANTIC_CODES[raw["dtype"]]
        expected[label] = {
            "shape": raw["shape"],
            "dtype_code": code,
            "compute_admitted": code in COMPUTE_ADMITTED_CODES,
            "values": raw["values"],
        }
    return expected


def normalize_candidate(observed: object, foreign: object) -> object:
    if not isinstance(observed, dict) or not isinstance(foreign, dict):
        return observed
    normalized: dict[str, object] = {}
    for label, row in observed.items():
        foreign_row = foreign.get(label)
        if row is None or not isinstance(row, dict) or not isinstance(foreign_row, dict):
            normalized[label] = row
            continue
        if "error" in row:
            normalized[label] = row
            continue
        length = row.get("length")
        values = None
        if foreign_row.get("values") is not None and isinstance(length, int):
            if row.get("empty"):
                values = []
            else:
                values = [row["start"] + i * row["step"] for i in range(length)]
        normalized[label] = {
            "shape": [length],
            "dtype_code": row.get("dtype_code"),
            "compute_admitted": row.get("compute_admitted"),
            "values": values,
        }
    return normalized


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
    arch = architecture_failures(candidate)
    failures = list(arch)
    if identity != {"revision": PINNED_REVISION, "tree": PINNED_TREE}:
        failures.append("oracle identity mismatch")

    foreign, foreign_error = run_json(
        args.python, FOREIGN_PROBE, str(checkout),
        {"DEV": "CPU", "CACHELEVEL": "0"})
    candidate_path = os.pathsep.join([
        str(candidate / "scripts" / "parity" / "shim"),
        str(candidate / "python"),
    ])
    observed, candidate_error = run_json(
        args.python, CANDIDATE_PROBE, candidate_path,
        {"TGRAD_LIB": str(args.lib.resolve())},
    )
    if foreign_error:
        failures.append(f"foreign {foreign_error}")
    if candidate_error:
        failures.append(f"candidate {candidate_error}")

    expected = expected_candidate(foreign)
    normalized = normalize_candidate(observed, foreign)
    if expected is not None and normalized != expected:
        failures.append("candidate range resolution differs from pinned foreign")

    result = {
        "status": "green" if not failures else "red",
        "oracle": identity,
        "foreign": foreign,
        "candidate": observed,
        "normalized_candidate": normalized,
        "expected_candidate": expected,
        "architecture_failures": arch,
        "probe_errors": {"foreign": foreign_error, "candidate": candidate_error},
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

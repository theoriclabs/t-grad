#!/usr/bin/env python3
"""Foreign-paired, allocation-free check for Lean-owned full-like resolution."""
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
SEMANTIC_CODES = {"bfloat16": 0, "float32": 1, "int32": 3}

FOREIGN_PROBE = r'''
import json
from tinygrad import Tensor, dtypes

def dtype_name(dtype):
  for name in ("bfloat16", "float32", "int32"):
    if dtype is getattr(dtypes, name): return name
  raise RuntimeError(f"non-canonical dtype {dtype!r}")

def row(source, fill, dtype=None):
  out = Tensor.full_like(source, fill, dtype=dtype)
  return {"shape": list(out.shape), "dtype": dtype_name(out.dtype)}

f32 = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
i32 = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
print(json.dumps({
  "inherit_f32": row(f32, 4),
  "inherit_i32": row(i32, 4),
  "explicit_i32_over_f32": row(f32, 4, dtypes.int32),
  "explicit_bf16_over_i32": row(i32, 4, dtypes.bfloat16),
}, sort_keys=True))
'''

CANDIDATE_PROBE = r'''
import ctypes, json
import tgrad

def register(shape, dtype_code, raw):
  dims = (ctypes.c_size_t * len(shape))(*shape)
  handle = int(tgrad._lib.tgrad_tensor_from_buffer(
    raw, dims, len(shape), dtype_code))
  if handle == 0: raise RuntimeError(f"registration rejected dtype={dtype_code}")
  return handle

def row(handle, fill, dtype_code=255):
  resolved = tgrad._full_like_resolution(
    handle, tgrad._fill_scalar_tag(fill), dtype_code)
  if resolved is None: return None
  shape, code = resolved
  return {"shape": list(shape), "dtype_code": code}

f32 = register((2, 3), 1, 0)
i32 = register((2, 3), 3, 0)
print(json.dumps({
  "inherit_f32": row(f32, 4),
  "inherit_i32": row(i32, 4),
  "explicit_i32_over_f32": row(f32, 4, 3),
  "explicit_bf16_over_i32": row(i32, 4, 0),
  "invalid_explicit": row(f32, 4, 253),
  "missing_source": row(0xffffffffffffffff, 4),
}, sort_keys=True))
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
    full_like = class_method_source(python_source, "full_like")
    zeros_like = class_method_source(python_source, "zeros_like")
    ones_like = class_method_source(python_source, "ones_like")
    if full_like is None:
        failures.append("product Tensor.full_like is missing")
    else:
        for forbidden in (
            "self.shape", "self._shape", "self.dtype", "self._dtype",
            "Tensor.full(", "np.", "numpy",
        ):
            if forbidden in full_like:
                failures.append(f"Python full_like owns inherited semantics: {forbidden}")
        for required in (
            "self._handle", "_fill_scalar_tag(fill_value)",
            "_creation_dtype_code(dtype)", "_lib.tgrad_tensor_full_like",
        ):
            if required not in full_like:
                failures.append(f"Python full_like missing marshalling path: {required}")
    if zeros_like is None or "self.full_like(0.0" not in zeros_like:
        failures.append("zeros_like does not delegate syntax to real full_like")
    if ones_like is None or "self.full_like(1.0" not in ones_like:
        failures.append("ones_like does not delegate syntax to real full_like")

    lean_source = (repo / "Tgrad" / "PythonFFI.lean").read_text()
    if "def resolveFullLike" not in lean_source:
        failures.append("missing shared Lean full-like resolver")
    if lean_source.count("← resolveFullLike sourceHandle fillTag dtypeCode") != 2:
        failures.append("query and allocation do not share one Lean full-like resolver")
    if "def allocateFilledTensor" not in lean_source:
        failures.append("full and full-like lack a shared Lean fill allocator")
    if lean_source.count("allocateFilledTensor") != 3:
        failures.append("shared Lean fill allocator is not used by both creation paths")

    c_source = re.sub(r"\s+", "", (repo / "c" / "tgrad_python.c").read_text())
    for symbol in ("tgrad_full_like_query", "tgrad_tensor_full_like"):
        if symbol not in c_source:
            failures.append(f"C boundary missing {symbol}")
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
    arch = architecture_failures(candidate)
    failures = list(arch)
    if identity != expected_identity:
        failures.append("oracle identity mismatch")

    foreign, foreign_error = run_json(
        args.python, FOREIGN_PROBE, str(checkout), {"DEV": "CPU"})
    candidate_path = os.pathsep.join([
        str(candidate / "scripts" / "parity" / "shim"),
        str(candidate / "python"),
    ])
    observed, candidate_error = run_json(
        args.python, CANDIDATE_PROBE, candidate_path,
        {"TGRAD_LIB": str(args.lib.resolve())},
    )
    if foreign_error: failures.append(f"foreign {foreign_error}")
    if candidate_error: failures.append(f"candidate {candidate_error}")

    expected = None
    if isinstance(foreign, dict):
        expected = {
            label: {
                "shape": row["shape"],
                "dtype_code": SEMANTIC_CODES[row["dtype"]],
            }
            for label, row in foreign.items()
        }
        expected.update({"invalid_explicit": None, "missing_source": None})
        if observed != expected:
            failures.append("candidate full-like resolution differs from pinned foreign")

    result = {
        "status": "green" if not failures else "red",
        "oracle": identity,
        "foreign": foreign,
        "candidate": observed,
        "expected_candidate": expected,
        "architecture_failures": arch,
        "probe_errors": {"foreign": foreign_error, "candidate": candidate_error},
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

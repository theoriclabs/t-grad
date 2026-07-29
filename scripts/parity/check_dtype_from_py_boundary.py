#!/usr/bin/env python3
"""Foreign parity plus architecture check for `dtypes.from_py`."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from check_dtype_public_boundary import PINNED_REVISION, PINNED_TREE, git


PROBE = r'''
import json
from tinygrad.dtype import dtypes, Invalid

cases = [
  ("bool", True), ("invalid", Invalid), ("int", 2), ("float", 3.0),
  ("empty_list", []), ("empty_tuple", ()), ("bool_list", [True]),
  ("bool_int", [True, 2]), ("bool_float", [True, 3.0]),
  ("int_float", [2, 3.0]), ("bool_int_float", [True, 2, 3.0]),
  ("nested_empty", [[]]), ("nested_empty_int", [[], 2]),
]
rows = []
for label, value in cases:
  try: rows.append({"label": label, "kind": "value", "result": repr(dtypes.from_py(value))})
  except Exception as exc: rows.append({"label": label, "kind": "error", "type": type(exc).__name__})
for label, value in (("none", None), ("dict", {}), ("set", set())):
  try: rows.append({"label": label, "kind": "value", "result": repr(dtypes.from_py(value))})
  except Exception as exc: rows.append({"label": label, "kind": "error", "type": type(exc).__name__})
print(json.dumps(rows, sort_keys=True))
'''


def run_probe(python: Path, pythonpath: str,
              extra_env: dict[str, str] | None = None) -> list[dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    if extra_env: env.update(extra_env)
    cp = subprocess.run([str(python), "-c", PROBE], env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError(f"probe rc={cp.returncode}: {cp.stderr.strip()}")
    return json.loads(cp.stdout)


def from_py_source(path: Path) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "from_py":
            return ast.get_source_segment(source, node) or ""
    raise RuntimeError("strict shim has no from_py definition")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    repo, checkout = args.repo.resolve(), args.checkout.resolve()
    identity = {"revision": git(checkout, "rev-parse", "HEAD"),
                "tree": git(checkout, "rev-parse", "HEAD^{tree}")}
    expected_identity = {"revision": PINNED_REVISION, "tree": PINNED_TREE}

    foreign = run_probe(args.python, str(checkout))
    candidate = run_probe(
        args.python,
        os.pathsep.join([str(repo / "scripts" / "parity" / "shim"),
                         str(repo / "python")]),
        {"TGRAD_LIB": str(args.lib.resolve())},
    )
    body = from_py_source(repo / "scripts" / "parity" / "shim" / "tinygrad" / "dtype.py")
    architecture_failures = []
    if "_dtype_infer_python" not in body:
        architecture_failures.append("from_py does not marshal normalized tags to Lean")
    for forbidden in ("strong_dtype", "default_int", "default_float", "max("):
        if forbidden in body:
            architecture_failures.append(f"Python from_py owns semantic operation {forbidden!r}")

    failures = []
    if identity != expected_identity:
        failures.append("oracle identity mismatch")
    if candidate != foreign:
        failures.append("candidate from_py readings differ from pinned foreign readings")
    failures.extend(architecture_failures)
    print(json.dumps({
        "status": "green" if not failures else "red",
        "oracle": identity,
        "foreign": foreign,
        "candidate": candidate,
        "architecture_failures": architecture_failures,
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

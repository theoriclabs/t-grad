#!/usr/bin/env python3
"""Foreign-paired check for Lean-owned Tensor.full shape rejection.

The public half compares the three pinned upstream negative-dimension cases.
The protocol half asks Tgrad's signed Python -> C -> Lean query to distinguish
accepted, negative, zero/non-materializable, and rank-limited shapes without
allocating or dispatching Metal work.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys


PINNED_REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
PINNED_TREE = "855cca3b00c38841a6d3a043284f3a2ca696d4b0"

NEGATIVE_CASES = ((-3,), (2, -3), (2, -3, 0))
PROTOCOL_CASES = {
    "accepted": ((2, 3), 0),
    "negative_rank1": ((-3,), 1),
    "negative_rank2": ((2, -3), 1),
    "negative_precedes_zero_and_rank": ((2, -3, 0), 1),
    "zero_nonmaterializable": ((2, 0), 2),
    "rank_limited": ((1, 2, 3, 4), 3),
}

PUBLIC_PROBE = r'''
import json
from tinygrad import Tensor

cases = json.loads(__import__("os").environ["TGRAD_CREATION_CASES"])
rows = []
for shape in cases:
  try:
    Tensor.full(tuple(shape), 2)
  except Exception as exc:
    rows.append({"shape": shape, "kind": "error", "type": type(exc).__name__})
  else:
    rows.append({"shape": shape, "kind": "value"})
print(json.dumps(rows, sort_keys=True))
'''

CANDIDATE_PROTOCOL_PROBE = r'''
import json, os
import tgrad

cases = json.loads(os.environ["TGRAD_CREATION_PROTOCOL_CASES"])
print(json.dumps({name: tgrad._creation_shape_admission(tuple(shape))
                  for name, shape in cases.items()}, sort_keys=True))
'''


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True).strip()


def run_json(python: Path, source: str, pythonpath: str,
             extra_env: dict[str, str]) -> object:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    env.update(extra_env)
    cp = subprocess.run(
        [str(python), "-c", source], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"probe rc={cp.returncode}: {cp.stderr.strip() or cp.stdout.strip()}")
    return json.loads(cp.stdout)


def boundary_is_marshalling_only(module_path: Path) -> tuple[bool, list[str]]:
    """Reject Python-side shape predicates and a bypass of the Lean query."""
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    helper = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "_creation_shape_admission"),
        None,
    )
    if helper is None:
        return False, ["missing _creation_shape_admission helper"]
    tensor_class = next(
        (node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == "Tensor"), None)
    full = next(
        (node for node in (tensor_class.body if tensor_class else [])
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "full"), None)
    if full is None:
        return False, ["missing Tensor.full"]

    forbidden: list[str] = []
    for node in ast.walk(helper):
        if isinstance(node, (ast.Compare, ast.BoolOp, ast.If, ast.IfExp)):
            forbidden.append(f"helper:{type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in {"any", "all", "min", "max"}:
            forbidden.append(f"helper:call:{node.func.id}")

    calls: dict[str, list[int]] = {}
    for node in ast.walk(full):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            calls.setdefault(name, []).append(node.lineno)
            if name in {"any", "all", "min", "max"}:
                forbidden.append(f"Tensor.full:call:{name}")
        if isinstance(node, ast.Compare):
            names = {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}
            if "shape" in names:
                forbidden.append("Tensor.full:shape-comparison")
    admission_lines = calls.get("_creation_shape_admission", [])
    full_lines = calls.get("_lib.tgrad_tensor_full", [])
    if len(admission_lines) != 1:
        forbidden.append("Tensor.full:Lean-admission-call-count")
    if len(full_lines) != 1:
        forbidden.append("Tensor.full:FFI-full-call-count")
    if admission_lines and full_lines and admission_lines[0] >= full_lines[0]:
        forbidden.append("Tensor.full:admission-after-allocation-path")
    return not forbidden, sorted(set(forbidden))


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
    if identity != expected_identity:
        print(json.dumps({
            "status": "red", "reason": "oracle identity mismatch",
            "oracle": identity, "expected_oracle": expected_identity,
        }, indent=2, sort_keys=True))
        return 1

    public_env = {"TGRAD_CREATION_CASES": json.dumps(NEGATIVE_CASES)}
    foreign = run_json(
        args.python, PUBLIC_PROBE, str(checkout),
        {**public_env, "DEV": "CPU"},
    )
    candidate_path = os.pathsep.join([
        str(candidate / "scripts" / "parity" / "shim"),
        str(candidate / "python"),
    ])
    candidate_public = run_json(
        args.python, PUBLIC_PROBE, candidate_path,
        {**public_env, "TGRAD_LIB": str(args.lib.resolve())},
    )
    protocol = run_json(
        args.python, CANDIDATE_PROTOCOL_PROBE, str(candidate / "python"),
        {
            "TGRAD_LIB": str(args.lib.resolve()),
            "TGRAD_CREATION_PROTOCOL_CASES": json.dumps({
                name: shape for name, (shape, _) in PROTOCOL_CASES.items()}),
        },
    )

    expected_public = [
        {"shape": list(shape), "kind": "error", "type": "ValueError"}
        for shape in NEGATIVE_CASES
    ]
    expected_protocol = {
        name: code for name, (_, code) in PROTOCOL_CASES.items()
    }
    marshalling_only, forbidden_nodes = boundary_is_marshalling_only(
        candidate / "python" / "tgrad.py")
    checks = {
        "foreign_matches_pinned_contract": foreign == expected_public,
        "candidate_matches_foreign": candidate_public == foreign,
        "signed_reason_protocol_exact": protocol == expected_protocol,
        "python_admission_helper_marshalling_only": marshalling_only,
    }
    result = {
        "status": "green" if all(checks.values()) else "red",
        "oracle": identity,
        "checks": checks,
        "foreign_public": foreign,
        "candidate_public": candidate_public,
        "protocol": protocol,
        "expected_protocol": expected_protocol,
        "forbidden_python_admission_nodes": forbidden_nodes,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired foreign/candidate check for the public scalar DType object boundary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PINNED_REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
PINNED_TREE = "855cca3b00c38841a6d3a043284f3a2ca696d4b0"

REQUIREMENT_SCHEMA = "tgrad.foreign-dtype-core.v2"

PROBE = r'''
import json, os
from tinygrad.dtype import dtypes, least_upper_dtype

names = json.loads(os.environ["TGRAD_DTYPE_PROBE_NAMES"])
witnesses = json.loads(os.environ["TGRAD_DTYPE_NARY_WITNESSES"])
values = {name: getattr(dtypes, name) for name in names}

def outcome(fn):
  try: return {"kind": "value", "value": bool(fn())}
  except Exception as exc: return {"kind": "error", "type": type(exc).__name__}

objects = {}
for name, value in values.items():
  objects[name] = {
    "str": str(value),
    "repr": repr(value),
    "eq_self": value == value,
    "eq_str_value": value == str(value),
    "eq_public_name": value == name,
    "hash_uses_identity": hash(value) == object.__hash__(value),
  }

pairs = []
for left_name, left in values.items():
  for right_name, right in values.items():
    pairs.append({
      "left": left_name, "right": right_name,
      "eq": left == right,
      "lt": outcome(lambda: left < right),
      "gt": outcome(lambda: left > right),
      "le": outcome(lambda: left <= right),
      "ge": outcome(lambda: left >= right),
    })

mutation_value = values["float32"]
mutations = {}
for field, replacement in (
    ("priority", 123456), ("bitsize", 123456), ("name", "mutated"),
    ("fmt", "Z"), ("itemsize", 123456)):
  before = getattr(mutation_value, field)
  try:
    setattr(mutation_value, field, replacement)
    result = {"kind": "value"}
  except Exception as exc:
    result = {"kind": "error", "type": type(exc).__name__}
  result.update({"before": before, "after": getattr(mutation_value, field),
                 "unchanged": getattr(mutation_value, field) == before})
  mutations[field] = result

def canonical_name(value):
  matches = [name for name, singleton in values.items() if value is singleton]
  if len(matches) != 1:
    raise RuntimeError(f"public LUB result is not one canonical singleton: {value!r}")
  return matches[0]

nary_lub = {}
for row in witnesses:
  inputs = row["inputs"]
  result = least_upper_dtype(*(values[name] for name in inputs))
  nary_lub[",".join(inputs)] = canonical_name(result)

print(json.dumps({"objects": objects, "pairs": pairs, "mutations": mutations,
                  "nary_lub": nary_lub}, sort_keys=True))
'''


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True).strip()


def run_probe(python: Path, pythonpath: str, names: list[str], witnesses: list[dict],
              extra_env: dict[str, str] | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    env["TGRAD_DTYPE_PROBE_NAMES"] = json.dumps(names)
    env["TGRAD_DTYPE_NARY_WITNESSES"] = json.dumps(witnesses)
    if extra_env: env.update(extra_env)
    cp = subprocess.run(
        [str(python), "-c", PROBE], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"probe rc={cp.returncode}: {cp.stderr.strip()}")
    return json.loads(cp.stdout)


def differences(left, right, path: str = "") -> list[dict]:
    if type(left) is not type(right):
        return [{"path": path, "foreign": left, "candidate": right}]
    if isinstance(left, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            out.extend(differences(left.get(key), right.get(key), f"{path}/{key}"))
        return out
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path + "/length", "foreign": len(left),
                     "candidate": len(right)}]
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out.extend(differences(a, b, f"{path}/{index}"))
        return out
    return [] if left == right else [
        {"path": path, "foreign": left, "candidate": right}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--requirement", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument(
        "--candidate-repo", type=Path,
        help="candidate repo providing the strict shim and Python boundary",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    repo, checkout = args.repo.resolve(), args.checkout.resolve()
    candidate_repo = (args.candidate_repo or repo).resolve()

    identity = {
        "revision": git(checkout, "rev-parse", "HEAD"),
        "tree": git(checkout, "rev-parse", "HEAD^{tree}"),
    }
    expected_identity = {"revision": PINNED_REVISION, "tree": PINNED_TREE}
    if identity != expected_identity:
        print(json.dumps({"status": "red", "identity": identity,
                          "expected_identity": expected_identity}, indent=2))
        return 1

    requirement = json.loads(args.requirement.read_text())
    if requirement.get("schema") != REQUIREMENT_SCHEMA:
        print(json.dumps({
            "status": "red", "reason": "unsupported requirement schema",
            "schema": requirement.get("schema"), "expected": REQUIREMENT_SCHEMA,
        }, indent=2, sort_keys=True))
        return 1
    witnesses = requirement.get("nary_lub_witnesses")
    if not isinstance(witnesses, list) or len(witnesses) != 24:
        print(json.dumps({
            "status": "red", "reason": "missing exact N-ary witness set",
            "witness_count": len(witnesses) if isinstance(witnesses, list) else None,
        }, indent=2, sort_keys=True))
        return 1
    names = [row["public_name"] for row in requirement["descriptors"]]
    foreign = run_probe(args.python, str(checkout), names, witnesses)
    candidate = run_probe(
        args.python,
        os.pathsep.join([str(candidate_repo / "scripts" / "parity" / "shim"),
                         str(candidate_repo / "python")]),
        names,
        witnesses,
        {"TGRAD_LIB": str(args.lib.resolve())},
    )

    diff = differences(foreign, candidate)
    exact = not diff
    result = {
        "status": "green" if exact else "red",
        "oracle": identity,
        "exact": exact,
        "difference_count": len(diff),
        "differences": diff[:100],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())

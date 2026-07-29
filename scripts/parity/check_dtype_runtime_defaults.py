#!/usr/bin/env python3
"""CPU-only executable contract for Lean-owned runtime dtype defaults."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SEQUENCE = r'''
import json
import tgrad as _tgrad
from tinygrad.dtype import dtypes, strong_dtype, least_upper_float

integer_names = ["int8", "int16", "int32", "int64"]
floating_names = [
  "fp8e4m3", "fp8e5m2", "fp8e4m3fnuz", "fp8e5m2fnuz",
  "float16", "bfloat16", "float32", "float64",
]
initial = [dtypes.default_int.code, dtypes.default_float.code]
rows = []

for name in integer_names:
  dtype = getattr(dtypes, name)
  dtypes.default_int = dtype
  rows.append({
    "role": "integer", "name": name,
    "getter": dtypes.default_int is dtype,
    "from_py": dtypes.from_py(1) is dtype,
    "strong": strong_dtype(dtypes.weakint) is dtype,
  })

for name in floating_names:
  dtype = getattr(dtypes, name)
  dtypes.default_float = dtype
  creation = _tgrad._dtype_creation_default()
  expected_creation = dtype.code if _tgrad._dtype_query(dtype.code, 15) == 1 else 255
  rows.append({
    "role": "floating", "name": name,
    "getter": dtypes.default_float is dtype,
    "from_py": dtypes.from_py(1.0) is dtype,
    "empty": dtypes.from_py([]) is dtype,
    "strong": strong_dtype(dtypes.weakfloat) is dtype,
    "least_upper_float": least_upper_float(dtypes.int32) is dtype,
    "creation": creation,
    "expected_creation": expected_creation,
  })

rejections = []
for role, dtype in [
  (0, dtypes.float32), (1, dtypes.int32),
  (0, dtypes.weakint), (0, dtypes.weakfloat), (0, dtypes.void),
  (1, dtypes.weakint), (1, dtypes.weakfloat), (1, dtypes.void),
]:
  before = _tgrad._dtype_default(role)
  accepted = _tgrad._dtype_set_default(role, dtype.code)
  after = _tgrad._dtype_default(role)
  rejections.append({"role": role, "code": dtype.code,
                     "accepted": accepted, "unchanged": before == after})
for role in (0, 1, 2):
  before = _tgrad._dtype_default(role) if role < 2 else None
  accepted = _tgrad._dtype_set_default(role, 253)
  after = _tgrad._dtype_default(role) if role < 2 else None
  rejections.append({"role": role, "code": 253,
                     "accepted": accepted, "unchanged": before == after})

dtypes.default_int = getattr(dtypes, "int32")
dtypes.default_float = getattr(dtypes, "float32")
restored = [dtypes.default_int.code, dtypes.default_float.code]
print(json.dumps({"initial": initial, "rows": rows,
                  "rejections": rejections, "restored": restored}, sort_keys=True))
'''

FRESH = r'''
import json
from tinygrad.dtype import dtypes
print(json.dumps([dtypes.default_int.code, dtypes.default_float.code]))
'''


def run(python: Path, env: dict[str, str], source: str):
    cp = subprocess.run([str(python), "-c", source], env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError(f"probe rc={cp.returncode}: {cp.stderr.strip()}")
    return json.loads(cp.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    repo = args.repo.resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(repo / "scripts" / "parity" / "shim"), str(repo / "python")])
    env["TGRAD_LIB"] = str(args.lib.resolve())
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"

    sequence = run(args.python, env, SEQUENCE)
    fresh = run(args.python, env, FRESH)
    failures: list[str] = []
    if sequence["initial"] != [3, 1]:
        failures.append(f"process did not start int32/float32: {sequence['initial']}")
    for row in sequence["rows"]:
        bool_fields = [key for key, value in row.items() if isinstance(value, bool)]
        if not all(row[key] for key in bool_fields):
            failures.append(f"dependent query disagreed after {row['role']} {row['name']}")
        if row.get("creation") != row.get("expected_creation"):
            failures.append(f"creation admission substituted default {row['name']}")
    for row in sequence["rejections"]:
        if row["accepted"] or not row["unchanged"]:
            failures.append(
                f"invalid default mutation accepted role={row['role']} code={row['code']}")
    if sequence["restored"] != [3, 1]:
        failures.append(f"defaults were not restored: {sequence['restored']}")
    if fresh != [3, 1]:
        failures.append(f"fresh process inherited mutable defaults: {fresh}")

    print(json.dumps({
        "status": "green" if not failures else "red",
        "sequence": sequence, "fresh_process": fresh, "failures": failures,
    }, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

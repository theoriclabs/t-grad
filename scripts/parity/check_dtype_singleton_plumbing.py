#!/usr/bin/env python3
"""CPU-only probe for strict Tensor/device DType identity presentation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROBE = r'''
import json
import tgrad as _tgrad
from tinygrad import Device, Tensor, dtypes

tensor_rows = []
for storage, singleton in (("bf16", dtypes.bfloat16),
                           ("f32", dtypes.float32),
                           ("i32", dtypes.int32)):
  tensor = object.__new__(Tensor)
  tensor._dtype = storage
  tensor_rows.append({"storage": storage, "exact": tensor.dtype is singleton})

unknown = object.__new__(Tensor)
unknown._dtype = "unknown"
try:
  unknown.dtype
  unknown_failed_closed = False
except _tgrad.TgradTypeError:
  unknown_failed_closed = True

supported = Device[Device.DEFAULT].renderer.supported_dtypes()
code_target = dtypes.float64
code_before = code_target.code
try:
  code_target.code = dtypes.float32.code
  code_assignment = {"kind": "value"}
except Exception as exc:
  code_assignment = {"kind": "error", "type": type(exc).__name__}
code_assignment["stable"] = code_target.code == code_before
print(json.dumps({
  "tensor_rows": tensor_rows,
  "unknown_failed_closed": unknown_failed_closed,
  "supported_codes": [dtype.code for dtype in supported],
  "supported_are_singletons": all(
    dtype is {dtypes.bfloat16.code: dtypes.bfloat16,
              dtypes.float32.code: dtypes.float32,
              dtypes.int32.code: dtypes.int32}.get(dtype.code)
    for dtype in supported),
  "identity_membership": all(dtype in supported for dtype in
                             (dtypes.bfloat16, dtypes.float32, dtypes.int32)),
  "metadata_only_absent": all(dtype not in supported for dtype in
                              (dtypes.float16, *dtypes.fp8s)),
  "lean_admitted": all(_tgrad._dtype_query(dtype.code, 15) == 1
                       for dtype in supported),
  "code_assignment": code_assignment,
}, sort_keys=True))
'''


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
    cp = subprocess.run([str(args.python), "-c", PROBE], env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    failures: list[str] = []
    if cp.returncode != 0:
        result = {"probe_error": cp.stderr.strip()}
        failures.append(f"probe exited {cp.returncode}")
    else:
        result = json.loads(cp.stdout)
        if result["supported_codes"] != [0, 1, 3]:
            failures.append(f"supported singleton codes differ: {result['supported_codes']}")
        for key in ("unknown_failed_closed", "supported_are_singletons",
                    "identity_membership", "metadata_only_absent", "lean_admitted"):
            if not result[key]: failures.append(f"{key} is false")
        if result["code_assignment"].get("type") != "FrozenInstanceError":
            failures.append("DType code assignment did not raise FrozenInstanceError")
        if not result["code_assignment"]["stable"]:
            failures.append("DType code assignment changed Lean identity")
        for row in result["tensor_rows"]:
            if not row["exact"]:
                failures.append(f"Tensor dtype {row['storage']} is not its exact singleton")
    print(json.dumps({"status": "green" if not failures else "red",
                      "probe": result, "failures": failures},
                     indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

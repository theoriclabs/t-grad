#!/usr/bin/env python3
"""Wave 8: foreign-paired, allocation-free arange authority observer.

The foreign side executes the three pinned upstream leaves.  The candidate
side never allocates a Tgrad range: it asks Lean for the shared decision,
storage plan, and (checker-only) stored bits.  Public construction is also
audited to ensure its Python work is constant-size in the resolved length.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


PINNED_REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
PINNED_TREE = "855cca3b00c38841a6d3a043284f3a2ca696d4b0"
TARGETS = (
    "test/backend/test_ops.py::TestOps::test_arange",
    "test/backend/test_edgecases.py::TestEdgeCases::test_arange_float_step",
    "test/unit/test_dtype_spec.py::TestTypeSpec::test_arange",
)
DTYPE_CODES = {
    "bfloat16": 0, "float32": 1, "float16": 2, "int32": 3,
    "int8": 6, "int64": 11,
}
DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "int32": 4, "int8": 1, "int64": 8}
GLOBAL_SUPPORTED = {0, 1, 3}
RANGE_SUPPORTED = {0, 1, 3, 6, 11}
EXPECTED_CONTEXT_SHA256 = "59c0bed9cc7f2e6f19301702ef039f1d87fdbf19944970f302ed560c45d04412"


FOREIGN_PLUGIN = r'''
import atexit, json, os, struct, sys
import hypothesis, numpy, pytest
from tinygrad import Tensor, Device, dtypes

TRACE = []
LIFECYCLE = {
  "collection_patch_applied": False, "collection_restore_called": False,
  "collection_supported": [], "execution_supported": None,
}
OUT = os.environ["TGRAD_WAVE8_FOREIGN_TRACE"]
TARGET_FRAGMENTS = (
  "test/backend/test_ops.py::TestOps::test_arange",
  "test/backend/test_edgecases.py::TestEdgeCases::test_arange_float_step",
  "test/unit/test_dtype_spec.py::TestTypeSpec::test_arange",
)

def dtype_name(dtype):
  for name in ("int8", "int32", "int64", "bfloat16", "float32", "float16"):
    if dtype is getattr(dtypes, name): return name
  raise RuntimeError(f"non-canonical dtype {dtype!r}")

def scalar(value):
  if isinstance(value, bool): return {"kind":"bool", "value":bool(value)}
  if isinstance(value, int): return {"kind":"int", "value":str(value)}
  if isinstance(value, float):
    return {"kind":"float", "bits":struct.unpack("<Q", struct.pack("<d", value))[0]}
  raise RuntimeError(f"unexpected arange scalar {value!r}")

def output_values(array):
  if array.dtype.kind in "iu":
    return [{"kind":"int", "value":str(int(value))} for value in array.reshape(-1)]
  return [{"kind":"f32", "bits":struct.unpack("<I", struct.pack("<f", float(value)))[0]}
          for value in array.reshape(-1)]

ORIGINAL_ARANGE = Tensor.arange

def observed_arange(cls, *args, **kwargs):
  nodeid = os.environ.get("PYTEST_CURRENT_TEST", "")
  active = any(fragment in nodeid for fragment in TARGET_FRAGMENTS)
  if not active: return ORIGINAL_ARANGE(*args, **kwargs)
  if LIFECYCLE["execution_supported"] is None:
    supported = RENDERER.supported_dtypes()
    LIFECYCLE["execution_supported"] = [
      name for name in ("int8", "int64") if getattr(dtypes, name) in supported
    ]
  if not args: raise RuntimeError("pinned arange call unexpectedly lacks start")
  start = args[0]
  stop_present = len(args) >= 2 or "stop" in kwargs
  stop = args[1] if len(args) >= 2 else kwargs.get("stop")
  step = args[2] if len(args) >= 3 else kwargs.get("step", 1)
  dtype = args[3] if len(args) >= 4 else kwargs.get("dtype")
  record = {
    "nodeid": nodeid.split(" (")[0],
    "call": {
      "start": scalar(start), "stop_present": stop_present,
      "stop": scalar(stop) if stop_present else None,
      "step": scalar(step), "dtype_arg": None if dtype is None else dtype_name(dtype),
    },
    "defaults": {
      "int": dtype_name(dtypes.default_int),
      "float": dtype_name(dtypes.default_float),
    },
  }
  try:
    result = ORIGINAL_ARANGE(*args, **kwargs)
    array = result.numpy()
    record["result"] = {
      "shape": list(result.shape), "dtype": dtype_name(result.dtype),
      "values": output_values(array),
    }
  except BaseException as exc:
    record["result"] = {"error": type(exc).__name__, "message": str(exc)}
    TRACE.append(record)
    raise
  TRACE.append(record)
  return result

Tensor.arange = classmethod(observed_arange)

# Collection must see Tgrad's truthful global compute surface.  Restore the
# real CPU renderer immediately after collection so range-local int8/int64
# executions remain foreign, not restricted by this observer.
RENDERER = Device[Device.DEFAULT].renderer
RENDERER_CLASS = RENDERER.__class__
ORIGINAL_SUPPORTED = RENDERER_CLASS.supported_dtypes
RENDERER_CLASS.supported_dtypes = lambda self: (dtypes.int32, dtypes.bfloat16, dtypes.float32)
LIFECYCLE["collection_patch_applied"] = True
LIFECYCLE["collection_supported"] = ["int32", "bfloat16", "float32"]

def pytest_collection_finish(session):
  RENDERER_CLASS.supported_dtypes = ORIGINAL_SUPPORTED
  LIFECYCLE["collection_restore_called"] = True

@atexit.register
def write_trace():
  with open(OUT, "w", encoding="utf-8") as handle:
    json.dump({
      "rows": TRACE, "lifecycle": LIFECYCLE,
      "versions": {
        "python": sys.version, "pytest": pytest.__version__,
        "hypothesis": hypothesis.__version__, "numpy": numpy.__version__,
      },
    }, handle, sort_keys=True, separators=(",", ":"))
'''

FOREIGN_FLOAT_CEILDIV_PROBE = r'''
import json
from tinygrad import Tensor, dtypes
result = Tensor.arange(0.1, 0.2, 1e-10, dtype=dtypes.float32)
# Shape and dtype are lazy graph metadata.  Do not realize or enumerate this
# billion-element tensor.
print(json.dumps({"shape": list(result.shape), "dtype": "float32"}))
'''

FOREIGN_REWORK_PROBE = r'''
import json, math
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ceildiv

def lazy(label, start, stop, step, dtype):
  try:
    tensor = Tensor.arange(start, stop, step, dtype=dtype)
    return {"label":label, "kind":"value", "shape":list(tensor.shape),
            "dtype":str(tensor.dtype)}
  except BaseException as exc:
    return {"label":label, "kind":"error", "type":type(exc).__name__,
            "message":str(exc)}

def realized_nonfinite(label, step, dtype):
  try:
    tensor = Tensor.arange(0.0, 1.0, step, dtype=dtype)
    values = tensor.tolist()
    return {"label":label, "kind":"value", "shape":list(tensor.shape),
            "dtype":str(tensor.dtype),
            "values":["nan" if isinstance(v, float) and math.isnan(v) else v for v in values]}
  except BaseException as exc:
    return {"label":label, "kind":"error", "type":type(exc).__name__,
            "message":str(exc)}

rows = [
  lazy("integral_index_safe", 0, 2**32, 1, dtypes.int64),
  lazy("integral_index_unsafe", 0, 2**32+1, 1, dtypes.int64),
  lazy("float_index_safe", 0, 2**32-1, 1, dtypes.float32),
  lazy("float_index_unsafe", 0, 2**32, 1, dtypes.float32),
  lazy("byte_product_refusal", 0, 2**61, 1, dtypes.int64),
  realized_nonfinite("positive_inf_f32", float("inf"), dtypes.float32),
  realized_nonfinite("negative_inf_f32", float("-inf"), dtypes.float32),
  realized_nonfinite("positive_inf_bf16", float("inf"), dtypes.bfloat16),
  realized_nonfinite("negative_inf_bf16", float("-inf"), dtypes.bfloat16),
  lazy("positive_inf_int8_float_endpoints", 0.0, 1.0, float("inf"), dtypes.int8),
  lazy("positive_inf_int32_float_endpoints", 0.0, 1.0, float("inf"), dtypes.int32),
  lazy("positive_inf_int64_float_endpoints", 0.0, 1.0, float("inf"), dtypes.int64),
  lazy("positive_inf_int8_int_endpoints", 0, 1, float("inf"), dtypes.int8),
  lazy("positive_inf_int32_int_endpoints", 0, 1, float("inf"), dtypes.int32),
  lazy("positive_inf_int64_int_endpoints", 0, 1, float("inf"), dtypes.int64),
  lazy("negative_inf_int8_float_endpoints", 0.0, 1.0, float("-inf"), dtypes.int8),
  lazy("negative_inf_int32_float_endpoints", 0.0, 1.0, float("-inf"), dtypes.int32),
  lazy("negative_inf_int64_float_endpoints", 0.0, 1.0, float("-inf"), dtypes.int64),
  lazy("negative_inf_int8_int_endpoints", 0, 1, float("-inf"), dtypes.int8),
  lazy("negative_inf_int32_int_endpoints", 0, 1, float("-inf"), dtypes.int32),
  lazy("negative_inf_int64_int_endpoints", 0, 1, float("-inf"), dtypes.int64),
  lazy("nan_step", 0.0, 1.0, float("nan"), dtypes.float32),
  lazy("infinite_result", 0.0, 1.0, 5e-324, dtypes.float32),
  lazy("huge_mixed", 10**400, 1.0, 1.0, dtypes.float32),
  lazy("signed_zero_positive_step", -0.0, 0.0, 1.0, dtypes.float32),
  lazy("signed_zero_negative_step", -0.0, 0.0, -1.0, dtypes.float32),
]
endpoint_cases = (
  ("positive_start", float("inf"), 1.0, 1.0),
  ("negative_start", float("-inf"), 1.0, 1.0),
  ("positive_stop", 0.0, float("inf"), 1.0),
  ("negative_stop", 0.0, float("-inf"), -1.0),
)
for orientation, start, stop, step in endpoint_cases:
  for dtype_name, dtype in (("bf16", dtypes.bfloat16), ("f32", dtypes.float32),
                            ("int8", dtypes.int8), ("int32", dtypes.int32),
                            ("int64", dtypes.int64)):
    rows.append(lazy(f"endpoint_{orientation}_{dtype_name}", start, stop, step, dtype))
print(json.dumps({"rows":rows, "ceildiv_inf":{
  "positive":ceildiv(1.0, float("inf")),
  "negative":ceildiv(1.0, float("-inf")),
}}, sort_keys=True))
'''


CANDIDATE_PROBE = r'''
import json, math, os, struct
import tgrad
from tinygrad import dtypes
from tinygrad.helpers import EMULATED_DTYPES

ROWS = json.load(open(os.environ["TGRAD_WAVE8_ROWS"], encoding="utf-8"))
CODE = {"bfloat16":0, "float32":1, "float16":2, "int32":3, "int8":6, "int64":11}
UINT64_MAX = (1 << 64) - 1

def scalar(encoded):
  if encoded["kind"] == "bool": return bool(encoded["value"])
  if encoded["kind"] == "int": return int(encoded["value"])
  if encoded["kind"] == "float":
    return struct.unpack("<d", struct.pack("<Q", encoded["bits"]))[0]
  raise RuntimeError(f"bad scalar {encoded!r}")

def observe(row):
  call, defaults = row["call"], row["defaults"]
  if not tgrad._dtype_set_default(0, CODE[defaults["int"]]):
    raise RuntimeError("candidate rejected observed integer default")
  if not tgrad._dtype_set_default(1, CODE[defaults["float"]]):
    raise RuntimeError("candidate rejected observed floating default")
  start, step = scalar(call["start"]), scalar(call["step"])
  stop = scalar(call["stop"]) if call["stop_present"] else None
  dtype_code = 255 if call["dtype_arg"] is None else CODE[call["dtype_arg"]]
  decision = tgrad._range_decision(start, stop, step, dtype_code)
  result = {"decision": decision}
  resolution = tgrad._range_resolution(start, stop, step, dtype_code)
  if resolution is not None:
    result["resolution"] = resolution
    result["stored_bits"] = [
      tgrad._range_value_bits(start, stop, step, dtype_code, index)
      for index in range(resolution["length"])
    ]
  return result

old_int, old_float = tgrad._dtype_default(0), tgrad._dtype_default(1)
try:
  observed = [observe(row) for row in ROWS]
finally:
  if not tgrad._dtype_set_default(0, old_int): raise RuntimeError("failed to restore default int")
  if not tgrad._dtype_set_default(1, old_float): raise RuntimeError("failed to restore default float")

# MSG 0123 dynamic calibration: metadata work is independent of output length.
original_query = tgrad._lib.tgrad_range_query
original_allocate = tgrad._lib.tgrad_tensor_arange
class Counted:
  def __init__(self, fn): self.fn, self.calls = fn, 0
  def __call__(self, *args): self.calls += 1; return self.fn(*args)
counted = Counted(original_query)
class AllocationTripwire(RuntimeError): pass
def allocation_tripwire(*args):
  raise AllocationTripwire("candidate allocating range path was reached")
tgrad._lib.tgrad_range_query = counted
tgrad._lib.tgrad_tensor_arange = allocation_tripwire
tgrad._range_resolution(0, 10, 1, 3)
small_calls = counted.calls
counted.calls = 0
tgrad._range_resolution(0, 1000000000, 1, 3)
large_calls = counted.calls

def public_preallocation_calls(stop):
  counted.calls = 0
  try: tgrad.Tensor.arange(0, stop, 1, dtype=dtypes.int32)
  except AllocationTripwire: return counted.calls
  raise RuntimeError("accepted public probe bypassed allocation tripwire")
public_small_calls = public_preallocation_calls(10)
public_large_calls = public_preallocation_calls(1000000000)

ceildiv_resolution = tgrad._range_resolution(0.1, 0.2, 1e-10, 1)
ceildiv_indices = [] if ceildiv_resolution is None else [
  0, 1, ceildiv_resolution["length"] - 1,
]
ceildiv_audit = {
  "resolution": ceildiv_resolution,
  "indices": ceildiv_indices,
  "stored_bits": [
    tgrad._range_value_bits(0.1, 0.2, 1e-10, 1, index)
    for index in ceildiv_indices
  ],
}

def summary(start, stop, step, code, indices=()):
  decision = tgrad._range_decision(start, stop, step, code)
  resolution = tgrad._range_resolution(start, stop, step, code)
  return {
    "decision": decision, "resolution": resolution, "indices": list(indices),
    "stored_bits": [tgrad._range_value_bits(start, stop, step, code, i) for i in indices],
  }

inherited = {
  "signed_boundary": summary(-(2**31), 2**31, 2**31-1, 255, (0, 1, 2)),
  "large_boundary_length": summary(0, 2**31, 1, 255, (0, 2**31-1)),
  "one_sided_int8_wrap": summary(125, 130, 3, 6, (0, 1)),
  "one_sided_int8_empty": summary(128, 0, 1, 6),
  "arbitrary_int_rejection": summary(2**70, 2**70+1, 1, 255),
  "float16_nonmaterialized": summary(0, 4, 1, 2),
}

rework = {
  "integral_index_safe": summary(0, 2**32, 1, 11, (0, 2**32-1)),
  "integral_index_unsafe": summary(0, 2**32+1, 1, 11, (0, 2**32)),
  "float_index_safe": summary(0, 2**32-1, 1, 1, (0, 2**32-2)),
  "float_index_unsafe": summary(0, 2**32, 1, 1, (0, 2**32-1)),
  "byte_product_refusal": summary(0, 2**61, 1, 11, (0, 2**61-1)),
  "positive_inf_f32": summary(0.0, 1.0, float("inf"), 1, (0,)),
  "negative_inf_f32": summary(0.0, 1.0, float("-inf"), 1),
  "positive_inf_bf16": summary(0.0, 1.0, float("inf"), 0, (0,)),
  "negative_inf_bf16": summary(0.0, 1.0, float("-inf"), 0),
  "positive_inf_int8": summary(0.0, 1.0, float("inf"), 6),
  "positive_inf_int32": summary(0.0, 1.0, float("inf"), 3),
  "positive_inf_int64": summary(0.0, 1.0, float("inf"), 11),
  "negative_inf_int8": summary(0.0, 1.0, float("-inf"), 6),
  "negative_inf_int32": summary(0.0, 1.0, float("-inf"), 3),
  "negative_inf_int64": summary(0.0, 1.0, float("-inf"), 11),
  "finite_float_int8": summary(0.0, 1.0, 0.25, 6),
  "nan_step": summary(0.0, 1.0, float("nan"), 1),
  "infinite_result": summary(0.0, 1.0, 5e-324, 1),
  "huge_mixed": summary(10**400, 1.0, 1.0, 1),
  "signed_zero_positive_step": summary(-0.0, 0.0, 1.0, 1),
  "signed_zero_negative_step": summary(-0.0, 0.0, -1.0, 1),
}
for orientation, start, stop, step in (
  ("positive_start", float("inf"), 1.0, 1.0),
  ("negative_start", float("-inf"), 1.0, 1.0),
  ("positive_stop", 0.0, float("inf"), 1.0),
  ("negative_stop", 0.0, float("-inf"), -1.0),
):
  for dtype_name, code in (("bf16", 0), ("f32", 1), ("int8", 6),
                           ("int32", 3), ("int64", 11)):
    rework[f"endpoint_{orientation}_{dtype_name}"] = summary(start, stop, step, code)
rework_kernels = {
  "positive_inf_f32": tgrad._range_kernel_observation(0.0, 1.0, float("inf"), 1),
  "positive_inf_bf16": tgrad._range_kernel_observation(0.0, 1.0, float("inf"), 0),
  "positive_inf_int8": tgrad._range_kernel_observation(0.0, 1.0, float("inf"), 6),
  "positive_inf_int32": tgrad._range_kernel_observation(0.0, 1.0, float("inf"), 3),
  "positive_inf_int64": tgrad._range_kernel_observation(0.0, 1.0, float("inf"), 11),
}

def public_outcome(label, start, stop, step, dtype):
  before = getattr(allocation_tripwire, "calls", 0)
  try:
    tgrad.Tensor.arange(start, stop, step, dtype=dtype)
    outcome = "returned"
  except BaseException as exc:
    outcome = type(exc).__name__
  after = getattr(allocation_tripwire, "calls", 0)
  return {"label":label, "outcome":outcome, "allocation_calls":after-before}

allocation_tripwire.calls = 0
def counted_allocation_tripwire(*args):
  allocation_tripwire.calls += 1
  raise AllocationTripwire("candidate allocating range path was reached")
tgrad._lib.tgrad_tensor_arange = counted_allocation_tripwire
public_rework = [
  public_outcome("integral_index_unsafe", 0, 2**32+1, 1, dtypes.int64),
  public_outcome("float_index_unsafe", 0, 2**32, 1, dtypes.float32),
  public_outcome("byte_product_refusal", 0, 2**61, 1, dtypes.int64),
  public_outcome("positive_inf_f32", 0.0, 1.0, float("inf"), dtypes.float32),
  public_outcome("negative_inf_f32", 0.0, 1.0, float("-inf"), dtypes.float32),
  public_outcome("positive_inf_int8_float_endpoints", 0.0, 1.0, float("inf"), dtypes.int8),
  public_outcome("positive_inf_int32_float_endpoints", 0.0, 1.0, float("inf"), dtypes.int32),
  public_outcome("positive_inf_int64_float_endpoints", 0.0, 1.0, float("inf"), dtypes.int64),
  public_outcome("positive_inf_int8_int_endpoints", 0, 1, float("inf"), dtypes.int8),
  public_outcome("positive_inf_int32_int_endpoints", 0, 1, float("inf"), dtypes.int32),
  public_outcome("positive_inf_int64_int_endpoints", 0, 1, float("inf"), dtypes.int64),
  public_outcome("negative_inf_int8_float_endpoints", 0.0, 1.0, float("-inf"), dtypes.int8),
  public_outcome("negative_inf_int32_float_endpoints", 0.0, 1.0, float("-inf"), dtypes.int32),
  public_outcome("negative_inf_int64_float_endpoints", 0.0, 1.0, float("-inf"), dtypes.int64),
  public_outcome("negative_inf_int8_int_endpoints", 0, 1, float("-inf"), dtypes.int8),
  public_outcome("negative_inf_int32_int_endpoints", 0, 1, float("-inf"), dtypes.int32),
  public_outcome("negative_inf_int64_int_endpoints", 0, 1, float("-inf"), dtypes.int64),
  public_outcome("finite_float_int8", 0.0, 1.0, 0.25, dtypes.int8),
  public_outcome("nan_step", 0.0, 1.0, float("nan"), dtypes.float32),
  public_outcome("infinite_result", 0.0, 1.0, 5e-324, dtypes.float32),
  public_outcome("huge_mixed", 10**400, 1.0, 1.0, dtypes.float32),
]
for orientation, start, stop, step in (
  ("positive_start", float("inf"), 1.0, 1.0),
  ("negative_start", float("-inf"), 1.0, 1.0),
  ("positive_stop", 0.0, float("inf"), 1.0),
  ("negative_stop", 0.0, float("-inf"), -1.0),
):
  for dtype_name, dtype in (("bf16", dtypes.bfloat16), ("f32", dtypes.float32),
                            ("int8", dtypes.int8), ("int32", dtypes.int32),
                            ("int64", dtypes.int64)):
    public_rework.append(public_outcome(
      f"endpoint_{orientation}_{dtype_name}", start, stop, step, dtype))

kernel_observations = {
  "int8": tgrad._range_kernel_observation(0, 4, 1, 6),
  "int64": tgrad._range_kernel_observation(0, 4, 1, 11),
  "float32": tgrad._range_kernel_observation(0.0, 2.0, 0.3, 1),
  "bfloat16": tgrad._range_kernel_observation(0.0, 2.0, 0.3, 0),
}
cache_pairs = {
  "signed_zero": [
    tgrad._range_kernel_observation(0.0, 2.0, 0.3, 1),
    tgrad._range_kernel_observation(-0.0, 2.0, 0.3, 1),
  ],
  "adjacent_float_bits": [
    tgrad._range_kernel_observation(0.0, 2.0, 0.3, 1),
    tgrad._range_kernel_observation(0.0, 2.0, math.nextafter(0.3, 1.0), 1),
  ],
  "numeric_mode": [
    tgrad._range_kernel_observation(0, 4, 1, 1),
    tgrad._range_kernel_observation(0.0, 4.0, 1.0, 1),
  ],
  "dtype": [
    tgrad._range_kernel_observation(0.0, 2.0, 0.3, 1),
    tgrad._range_kernel_observation(0.0, 2.0, 0.3, 0),
  ],
  "length": [
    tgrad._range_kernel_observation(0.0, 2.0, 0.3, 1),
    tgrad._range_kernel_observation(0.0, 2.2, 0.3, 1),
  ],
}
saved_int = tgrad._dtype_default(0)
try:
  if not tgrad._dtype_set_default(0, 11): raise RuntimeError("failed to set inherited int64 default")
  inherited["changed_default_int"] = summary(0, 4, 1, 255, (0, 3))
finally:
  if not tgrad._dtype_set_default(0, saved_int): raise RuntimeError("failed to restore inherited int default")

rejections = {}
for label, args in {
  "zero_step": (0, 10, 0, 255),
  "overflow_precedes_zero_i32": (2**40, 0, 0, 3),
  "overflow_precedes_zero_i8": (2**40, 0, 0, 6),
  "beyond_int64": (2**70, 2**70+1, 1, 255),
  "unsupported_float16_materializer": (0, 4, 1, 2),
}.items():
  start, stop, step, code = args
  try:
    dtype = None if code == 255 else getattr(dtypes, {2:"float16",3:"int32",6:"int8"}[code])
    tgrad.Tensor.arange(start, stop, step, dtype=dtype)
    rejections[label] = None
  except BaseException as exc:
    rejections[label] = type(exc).__name__

tgrad._lib.tgrad_range_query = original_query
tgrad._lib.tgrad_tensor_arange = original_allocate

print(json.dumps({
  "rows": observed,
  "global_supported": [code for code in range(255) if tgrad._dtype_query(code, 15) == 1],
  "range_supported": [
    code for code in range(255)
    if tgrad._dtype_query(code, 0) == 1
    and (resolution := tgrad._range_resolution(0, 2, 1, code)) is not None
    and resolution["materialize_admitted"]
  ],
  "emulated": [code for code in range(255) if tgrad._dtype_query(code, 17) == 1],
  "emulated_tolist": [item.code for item in EMULATED_DTYPES.tolist(dtypes)],
  "resolution_query_counts": [small_calls, large_calls],
  "public_preallocation_query_counts": [public_small_calls, public_large_calls],
  "float_ceildiv_audit": ceildiv_audit,
  "inherited_wave7": inherited,
  "wave8_rework": rework,
  "rework_kernels": rework_kernels,
  "public_rework": public_rework,
  "kernel_observations": kernel_observations,
  "cache_identity_pairs": cache_pairs,
  "public_rejections": rejections,
}, sort_keys=True))
'''


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True).strip()


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    merged.update(env)
    return subprocess.run(
        command, cwd=cwd, env=merged, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def function_node(source: str, name: str, class_name: str | None = None) -> ast.AST | None:
    tree = ast.parse(source)
    body = tree.body
    if class_name is not None:
        cls = next((node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
        body = cls.body if cls else []
    return next((node for node in body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name), None)


def architecture_failures(repo: Path) -> list[str]:
    failures: list[str] = []
    python_source = (repo / "python" / "tgrad.py").read_text()
    arange = function_node(python_source, "arange", "Tensor")
    resolution = function_node(python_source, "_range_resolution")
    value_bits = function_node(python_source, "_range_value_bits")
    if arange is None or resolution is None or value_bits is None:
        return ["missing public arange/resolution/indexed-observer boundary"]
    arange_text = ast.get_source_segment(python_source, arange) or ""
    resolution_text = ast.get_source_segment(python_source, resolution) or ""
    value_bits_text = ast.get_source_segment(python_source, value_bits) or ""

    # MSG 0123: neither public creation nor its pre-allocation resolver may do
    # work proportional to the result length.  The indexed query is separate.
    for label, node, text in (("Tensor.arange", arange, arange_text),
                              ("_range_resolution", resolution, resolution_text)):
        if any(isinstance(item, (ast.For, ast.AsyncFor, ast.While, ast.ListComp,
                                 ast.SetComp, ast.DictComp, ast.GeneratorExp))
               for item in ast.walk(node)):
            failures.append(f"{label} contains iterative/collection work")
        if "_range_value_bits" in text or re.search(r"\brange\s*\(\s*length\b", text):
            failures.append(f"{label} enumerates the resolved range")
        if any(token in text for token in ("np.arange", "np.full", "from_numpy", "value_bits = [")):
            failures.append(f"{label} builds progression values in Python")
    for helper_name in ("_range_scalar", "_range_boundary_args", "_creation_dtype_code",
                        "_range_decision"):
        helper = function_node(python_source, helper_name)
        if helper is None:
            failures.append(f"public pre-allocation helper {helper_name} is missing")
            continue
        if any(isinstance(item, (ast.For, ast.AsyncFor, ast.While, ast.ListComp,
                                 ast.SetComp, ast.DictComp, ast.GeneratorExp))
               for item in ast.walk(helper)):
            failures.append(f"public pre-allocation helper {helper_name} contains iterative work")
    if "Checker-only indexed observation" not in value_bits_text:
        failures.append("indexed stored-bit query is not marked checker-only")
    for required in ("_range_decision(", "_range_resolution(", "_lib.tgrad_tensor_arange"):
        if required not in arange_text:
            failures.append(f"public arange missing boundary step {required}")
    for required in ("if decision == 5", "raise OverflowError",
                     "if decision == 6", "raise ValueError"):
        if required not in arange_text:
            failures.append(f"public arange reason mapping missing {required}")

    lean_raw = (repo / "Tgrad" / "PythonFFI.lean").read_text()
    lean = re.sub(r"/-.*?-/", "", lean_raw, flags=re.S)
    lean = re.sub(r"--[^\n]*", "", lean)
    for required in (
        "def resolveRange ", "def rangeStoragePlan?", "def rangeStoredBits?",
        "def pythonFiniteFloorDivForCount?", "pythonFloatFmod left right",
        "pythonFiniteFloorDivForCount? (stop - start) (-step)",
        "def rangeKernelIndexFits", "length ≤ 4294967295",
        "length ≤ 4294967296", "def rangeMaterializationAdmitted",
        "rangeByteCountFits length dtype.sizeBytes",
        "Tgrad.Renderer.Metal.rangeKernelDecl",
        "length dtype start.kernelScalar step.kernelScalar",
        "let decision ← resolveRange", "let some plan := rangeStoragePlan? spec",
        "Tgrad.Tensor.ofEmpty? [0] spec.dtype", "| .empty tensor => TensorRegistry.register tensor",
    ):
        if required not in lean:
            failures.append(f"Lean range authority missing {required}")
    if lean.count("let decision ← resolveRange") != 2:
        failures.append("query and allocation do not share exactly one resolver authority")
    if lean.count("rangeStoragePlan? spec") < 2:
        failures.append("query and allocation do not share one storage-plan authority")
    query_decl = re.search(r"def rangeQuery\b(.*?)\ninitialize libCacheFill", lean, re.S)
    stored_decl = re.search(r"def rangeStoredBits\?(.*?)\ninductive RangeStoragePlan", lean, re.S)
    for label, declaration in (("rangeQuery", query_decl), ("rangeStoredBits", stored_decl)):
        if declaration is None:
            failures.append(f"could not isolate Lean {label} declaration")
            continue
        if any(token in declaration.group(1) for token in
               ("metalAlloc", "metalCompile", "metalDispatch", "compileOrCache", "TensorRegistry.register")):
            failures.append(f"allocation-free Lean {label} reaches allocation/dispatch")

    renderer_raw = (repo / "Tgrad" / "Renderer" / "Creation.lean").read_text()
    renderer = re.sub(r"/-.*?-/", "", renderer_raw, flags=re.S)
    renderer = re.sub(r"--[^\n]*", "", renderer)
    for required in (
        "inductive RangeKernelScalar", "value.toBits", "fma(((float)(gid.x + 1u))",
        "roundedBf16BitsExpr",
        'body := storeScalarStmts "result" ty outS "data0" outIdx rhs',
        "| .int8_ => some \"char\"", "| .int64_ => some \"long\"",
        "range_names_separate_float_bits", "range_names_separate_nearby_float_bits",
    ):
        if required not in renderer:
            failures.append(f"Lean renderer contract missing {required}")

    dtype_raw = (repo / "Tgrad" / "Dtype.lean").read_text()
    dtype = re.sub(r"/-.*?-/", "", dtype_raw, flags=re.S)
    dtype = re.sub(r"--[^\n]*", "", dtype)
    if "def Dtype.emulatedSet : List Dtype := []" not in dtype:
        failures.append("backend emulation relation is not Lean-owned empty data")

    c_source = (repo / "c" / "tgrad_python.c").read_text()
    for symbol, lean_symbol in (("tgrad_range_query", "tgrad_range_query_lean"),
                                ("tgrad_tensor_arange", "tgrad_tensor_arange_lean")):
        query_body = re.search(rf"uint64_t {symbol}\((.*?)\n}}\n", c_source, re.S)
        if query_body is None:
            failures.append(f"C range bridge missing {symbol}")
            continue
        body = query_body.group(1)
        for forbidden in ("for (", "while (", "rangeStored", "length *", "ceil(", "floor("):
            if forbidden in body:
                failures.append(f"C bridge {symbol} owns range semantics: {forbidden}")
        if "lean_cstr_to_int" not in body or lean_symbol not in body:
            failures.append(f"C bridge {symbol} does not narrowly forward arbitrary Lean Int payloads")
    return failures


def context_key(row: dict) -> str:
    return json.dumps({"call": row["call"], "defaults": row["defaults"]}, sort_keys=True, separators=(",", ":"))


def decode_candidate_bits(dtype: str, bits: int) -> dict:
    if dtype == "int8":
        value = bits & 0xff
        if value & 0x80: value -= 1 << 8
        return {"kind": "int", "value": str(value)}
    if dtype == "int32":
        value = bits & 0xffffffff
        if value & 0x80000000: value -= 1 << 32
        return {"kind": "int", "value": str(value)}
    if dtype == "int64":
        value = bits & ((1 << 64) - 1)
        if value & (1 << 63): value -= 1 << 64
        return {"kind": "int", "value": str(value)}
    if dtype == "float32": return {"kind": "f32", "bits": bits & 0xffffffff}
    if dtype == "bfloat16": return {"kind": "f32", "bits": (bits & 0xffff) << 16}
    raise AssertionError(dtype)


def validate_rows(foreign_rows: list[dict], candidate: dict) -> list[str]:
    failures: list[str] = []
    groups: dict[str, list[dict]] = {}
    for row in foreign_rows:
        groups.setdefault(context_key(row), []).append(row)
    if len(foreign_rows) != 67:
        failures.append(f"foreign execution denominator is {len(foreign_rows)}, expected 67")
    if len(groups) != 35:
        failures.append(f"foreign semantic denominator is {len(groups)}, expected 35")
    partitions = {
        target: [row for row in foreign_rows if row.get("nodeid") == target]
        for target in TARGETS
    }
    expected_partitions = {
        TARGETS[0]: 18, TARGETS[1]: 1, TARGETS[2]: 48,
    }
    for target, expected in expected_partitions.items():
        if len(partitions[target]) != expected:
            failures.append(f"foreign partition {target} has {len(partitions[target])}, expected {expected}")
    partition_contexts = {
        target: len({context_key(row) for row in rows})
        for target, rows in partitions.items()
    }
    if partition_contexts != {TARGETS[0]: 18, TARGETS[1]: 1, TARGETS[2]: 16}:
        failures.append(f"foreign semantic partitions changed: {partition_contexts!r}")
    overflow_count = sum("error" in row["result"] for row in foreign_rows)
    empty_count = sum(row["result"].get("shape") == [0] for row in foreign_rows)
    dtype_set = {row["result"].get("dtype") for row in foreign_rows if "error" not in row["result"]}
    if overflow_count != 5:
        failures.append(f"foreign overflow count is {overflow_count}, expected 5")
    if empty_count != 12:
        # Two semantic empty rows execute under six dtype-default aliases.
        failures.append(f"foreign empty execution count is {empty_count}, expected 12")
    if dtype_set != {"int8", "int32", "int64", "bfloat16", "float32"}:
        failures.append(f"foreign required dtype set changed: {dtype_set!r}")
    digest = hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()
    if digest != EXPECTED_CONTEXT_SHA256:
        failures.append(f"foreign normalized-context digest changed: {digest}")
    if len(candidate.get("rows", [])) != len(groups):
        failures.append("candidate row count differs from semantic denominator")
        return failures

    for index, ((_, executions), observed) in enumerate(zip(groups.items(), candidate["rows"])):
        foreign = executions[0]
        if any(item["result"] != foreign["result"] for item in executions[1:]):
            failures.append(f"foreign aliases disagree for semantic row {index}")
            continue
        result = foreign["result"]
        decision = observed.get("decision")
        if "error" in result:
            if result["error"] != "OverflowError" or decision != 4:
                failures.append(f"row {index}: foreign rejection differs from Lean decision")
            continue
        if decision != 0 or not isinstance(observed.get("resolution"), dict):
            failures.append(f"row {index}: Lean rejected foreign-successful range")
            continue
        resolution = observed["resolution"]
        dtype = result["dtype"]
        length = result["shape"][0]
        if resolution.get("dtype_code") != DTYPE_CODES[dtype]:
            failures.append(f"row {index}: dtype differs")
        if resolution.get("length") != length:
            failures.append(f"row {index}: length differs")
        if resolution.get("materialize_admitted") is not True:
            failures.append(f"row {index}: required range dtype is not admitted")
        plan = resolution.get("storage_plan")
        expected_plan = {
            "kind": 0 if length == 0 else 1, "rank": 1, "dim0": length,
            "dtype_code": DTYPE_CODES[dtype], "bytes": length * DTYPE_BYTES[dtype], "raw": 0,
        }
        if plan != expected_plan:
            failures.append(f"row {index}: storage plan differs: {plan!r}")
        bits = observed.get("stored_bits")
        if not isinstance(bits, list) or len(bits) != length:
            failures.append(f"row {index}: indexed stored-bit count differs")
            continue
        decoded = [decode_candidate_bits(dtype, value) for value in bits]
        if dtype not in ("float32", "bfloat16"):
            if decoded != result["values"]:
                failures.append(f"row {index}: Lean stored values differ from foreign")
        else:
            if foreign["nodeid"] == TARGETS[0]:
                atol, rtol = 1e-6, 1e-3
            elif foreign["nodeid"] == TARGETS[1]:
                atol, rtol = 1e-7, 1e-7
            else:
                atol, rtol = (0.0, 1e-2) if dtype == "bfloat16" else (0.0, 1e-7)
            for actual, expected in zip(decoded, result["values"]):
                actual_value = struct.unpack("<f", struct.pack("<I", actual["bits"]))[0]
                expected_value = struct.unpack("<f", struct.pack("<I", expected["bits"]))[0]
                if abs(actual_value - expected_value) > atol + rtol * abs(expected_value):
                    failures.append(f"row {index}: Lean stored float values exceed leaf tolerance")
                    break
    return failures


def validate_inherited(candidate: dict) -> list[str]:
    failures: list[str] = []
    rows = candidate.get("inherited_wave7")
    if not isinstance(rows, dict):
        return ["inherited Wave 7 observations are missing"]

    def accepted(label: str, dtype: int, length: int, bits: list[int]) -> dict | None:
        row = rows.get(label)
        resolution = row.get("resolution") if isinstance(row, dict) else None
        if not isinstance(row, dict) or row.get("decision") != 0 or not isinstance(resolution, dict):
            failures.append(f"inherited row {label} is not accepted")
            return None
        if resolution.get("dtype_code") != dtype or resolution.get("length") != length:
            failures.append(f"inherited row {label} dtype/length differs")
        if row.get("stored_bits") != bits:
            failures.append(f"inherited row {label} selected values differ")
        return resolution

    signed = accepted("signed_boundary", 3, 3,
                      [2147483648, 4294967295, 2147483646])
    if signed and signed.get("storage_plan", {}).get("bytes") != 12:
        failures.append("signed-boundary storage byte count differs")
    large = accepted("large_boundary_length", 3, 2147483648,
                     [0, 2147483647])
    if large and large.get("storage_plan", {}).get("bytes") != 8589934592:
        failures.append("large allocation-free byte-count boundary differs")
    accepted("one_sided_int8_wrap", 6, 2, [125, 128])
    empty = accepted("one_sided_int8_empty", 6, 0, [])
    if empty and empty.get("storage_plan") != {
            "kind": 0, "rank": 1, "dim0": 0, "dtype_code": 6,
            "bytes": 0, "raw": 0}:
        failures.append("one-sided int8 empty representation differs")
    changed = accepted("changed_default_int", 11, 4, [0, 3])
    if changed and changed.get("materialize_admitted") is not True:
        failures.append("changed int64 runtime default is not range-admitted")
    arbitrary = rows.get("arbitrary_int_rejection")
    if not isinstance(arbitrary, dict) or arbitrary.get("decision") != 4 \
            or arbitrary.get("resolution") is not None:
        failures.append("arbitrary-integer rejection/order differs")
    f16 = rows.get("float16_nonmaterialized")
    f16_resolution = f16.get("resolution") if isinstance(f16, dict) else None
    if not isinstance(f16_resolution, dict) or f16.get("decision") != 0 \
            or f16_resolution.get("dtype_code") != 2 \
            or f16_resolution.get("materialize_admitted") is not False:
        failures.append("float16 semantic acceptance/materialization boundary differs")
    return failures


def validate_rework(foreign: dict, candidate: dict) -> list[str]:
    failures: list[str] = []
    foreign_rows = foreign.get("rows") if isinstance(foreign, dict) else None
    if not isinstance(foreign_rows, list):
        return ["foreign Wave 8 rework rows are missing"]
    by_label = {row.get("label"): row for row in foreign_rows if isinstance(row, dict)}
    integer_infinity_labels = {
        f"{sign}_inf_{dtype}_{endpoints}_endpoints"
        for sign in ("positive", "negative")
        for dtype in ("int8", "int32", "int64")
        for endpoints in ("float", "int")
    }
    endpoint_labels = {
        f"endpoint_{orientation}_{dtype}"
        for orientation in ("positive_start", "negative_start",
                            "positive_stop", "negative_stop")
        for dtype in ("bf16", "f32", "int8", "int32", "int64")
    }
    expected_labels = {
        "integral_index_safe", "integral_index_unsafe", "float_index_safe",
        "float_index_unsafe", "byte_product_refusal", "positive_inf_f32",
        "negative_inf_f32", "positive_inf_bf16", "negative_inf_bf16",
        "nan_step", "infinite_result", "huge_mixed",
        "signed_zero_positive_step", "signed_zero_negative_step",
    } | integer_infinity_labels | endpoint_labels
    if set(by_label) != expected_labels:
        failures.append(f"foreign Wave 8 rework manifest changed: {sorted(by_label)}")
        return failures
    foreign_shapes = {
        "integral_index_safe": [2**32], "integral_index_unsafe": [2**32+1],
        "float_index_safe": [2**32-1], "float_index_unsafe": [2**32],
        "byte_product_refusal": [2**61],
        "positive_inf_f32": [1], "negative_inf_f32": [0],
        "positive_inf_bf16": [1], "negative_inf_bf16": [0],
        "signed_zero_positive_step": [0], "signed_zero_negative_step": [0],
    }
    for dtype in ("int8", "int32", "int64"):
        for endpoints in ("float", "int"):
            foreign_shapes[f"negative_inf_{dtype}_{endpoints}_endpoints"] = [0]
    for label, shape in foreign_shapes.items():
        row = by_label[label]
        if row.get("kind") != "value" or row.get("shape") != shape:
            failures.append(f"foreign rework row {label} shape/result changed: {row!r}")
    for label in ("positive_inf_f32", "positive_inf_bf16"):
        if by_label[label].get("values") != ["nan"]:
            failures.append(f"foreign rework row {label} is not one realized NaN")
    for label in ("negative_inf_f32", "negative_inf_bf16"):
        if by_label[label].get("values") != []:
            failures.append(f"foreign rework row {label} is not a realized empty tensor")
    expected_errors = {
        "nan_step": "ValueError",
        "infinite_result": "OverflowError",
        "huge_mixed": "OverflowError",
    }
    for dtype in ("int8", "int32", "int64"):
        for endpoints in ("float", "int"):
            # Pinned tinygrad computes ceildiv(..., +inf) == 1, then rejects
            # inside the constructor while creating the integer-valued full.
            expected_errors[f"positive_inf_{dtype}_{endpoints}_endpoints"] = "OverflowError"
    for orientation in ("positive_start", "negative_start", "positive_stop", "negative_stop"):
        for dtype in ("bf16", "f32"):
            expected_errors[f"endpoint_{orientation}_{dtype}"] = "ValueError"
        if orientation != "positive_start":
            for dtype in ("int8", "int32", "int64"):
                expected_errors[f"endpoint_{orientation}_{dtype}"] = "OverflowError"
    for dtype in ("int8", "int32", "int64"):
        expected_errors[f"endpoint_positive_start_{dtype}"] = "ValueError"
    for label, error in expected_errors.items():
        row = by_label[label]
        if row.get("kind") != "error" or row.get("type") != error:
            failures.append(f"foreign rework row {label} exception changed: {row!r}")
    if foreign.get("ceildiv_inf") != {"positive": 1, "negative": 0}:
        failures.append("foreign infinite-step ceildiv values changed")

    rows = candidate.get("wave8_rework")
    if not isinstance(rows, dict):
        return failures + ["candidate Wave 8 rework rows are missing"]
    sentinel_plan = {key: (1 << 64) - 1 for key in
                     ("kind", "rank", "dim0", "dtype_code", "bytes", "raw")}

    def accepted(label: str, dtype: int, length: int, admitted: bool,
                 plan: dict, selected_count: int | None = None) -> dict | None:
        row = rows.get(label)
        resolution = row.get("resolution") if isinstance(row, dict) else None
        if not isinstance(row, dict) or row.get("decision") != 0 or not isinstance(resolution, dict):
            failures.append(f"candidate rework row {label} is not semantically accepted")
            return None
        if resolution.get("dtype_code") != dtype or resolution.get("length") != length:
            failures.append(f"candidate rework row {label} dtype/length differs")
        if resolution.get("materialize_admitted") is not admitted:
            failures.append(f"candidate rework row {label} materialization admission differs")
        if resolution.get("storage_plan") != plan:
            failures.append(f"candidate rework row {label} storage plan differs")
        if selected_count is not None and len(row.get("stored_bits", [])) != selected_count:
            failures.append(f"candidate rework row {label} selected-bit count differs")
        return resolution

    accepted("integral_index_safe", 11, 2**32, True,
             {"kind":1, "rank":1, "dim0":2**32, "dtype_code":11,
              "bytes":(2**32)*8, "raw":0}, 2)
    accepted("integral_index_unsafe", 11, 2**32+1, False, sentinel_plan, 2)
    accepted("float_index_safe", 1, 2**32-1, True,
             {"kind":1, "rank":1, "dim0":2**32-1, "dtype_code":1,
              "bytes":(2**32-1)*4, "raw":0}, 2)
    accepted("float_index_unsafe", 1, 2**32, False, sentinel_plan, 2)
    accepted("byte_product_refusal", 11, 2**61, False, sentinel_plan, 2)
    f32 = accepted("positive_inf_f32", 1, 1, True,
                   {"kind":1, "rank":1, "dim0":1, "dtype_code":1,
                    "bytes":4, "raw":0}, 1)
    bf16 = accepted("positive_inf_bf16", 0, 1, True,
                    {"kind":1, "rank":1, "dim0":1, "dtype_code":0,
                     "bytes":2, "raw":0}, 1)
    for label, dtype in (("negative_inf_f32", 1), ("negative_inf_bf16", 0)):
        accepted(label, dtype, 0, True,
                 {"kind":0, "rank":1, "dim0":0, "dtype_code":dtype,
                  "bytes":0, "raw":0}, 0)
    for label in ("positive_inf_int8", "positive_inf_int32", "positive_inf_int64"):
        row = rows.get(label)
        if not isinstance(row, dict) or row.get("decision") != 4 or row.get("resolution") is not None:
            failures.append(f"candidate rework row {label} must reject before allocation")
    for label, dtype in (("negative_inf_int8", 6), ("negative_inf_int32", 3),
                         ("negative_inf_int64", 11)):
        accepted(label, dtype, 0, True,
                 {"kind":0, "rank":1, "dim0":0, "dtype_code":dtype,
                  "bytes":0, "raw":0}, 0)
    # The infinity rule must not become a semantic ban on ordinary finite
    # float-to-int ranges.  The current renderer still lacks this scalar/kernel
    # combination, so acceptance is semantic-only and public construction
    # fails closed before allocation.
    accepted("finite_float_int8", 6, 4, False, sentinel_plan, 0)
    if f32 is not None:
        bits = rows["positive_inf_f32"].get("stored_bits", [])
        if len(bits) != 1 or not math.isnan(struct.unpack("<f", struct.pack("<I", bits[0] & 0xffffffff))[0]):
            failures.append("candidate positive-inf f32 value is not NaN")
    if bf16 is not None:
        bits = rows["positive_inf_bf16"].get("stored_bits", [])
        if len(bits) != 1 or not math.isnan(struct.unpack("<f", struct.pack("<I", (bits[0] & 0xffff) << 16))[0]):
            failures.append("candidate positive-inf bf16 value is not NaN")
    expected_decisions = {
        "nan_step": 6, "infinite_result": 5, "huge_mixed": 5,
    }
    for orientation in ("positive_start", "negative_start", "positive_stop", "negative_stop"):
        for dtype in ("bf16", "f32"):
            expected_decisions[f"endpoint_{orientation}_{dtype}"] = 6
        for dtype in ("int8", "int32", "int64"):
            expected_decisions[f"endpoint_{orientation}_{dtype}"] = \
                6 if orientation == "positive_start" else 4
    for label, decision in expected_decisions.items():
        row = rows.get(label)
        if not isinstance(row, dict) or row.get("decision") != decision or row.get("resolution") is not None:
            failures.append(f"candidate rework row {label} reason differs")
    for label in ("signed_zero_positive_step", "signed_zero_negative_step"):
        resolution = accepted(label, 1, 0, True,
            {"kind":0, "rank":1, "dim0":0, "dtype_code":1, "bytes":0, "raw":0}, 0)
        if resolution is not None and resolution.get("start_identity_bits") != 1 << 63:
            failures.append(f"candidate rework row {label} lost signed-zero input identity")

    kernels = candidate.get("rework_kernels")
    if not isinstance(kernels, dict):
        failures.append("candidate non-finite kernel observations are missing")
    else:
        for label, flags in (("positive_inf_f32", 35), ("positive_inf_bf16", 49)):
            row = kernels.get(label)
            if not isinstance(row, dict) or row.get("source_flags") != flags:
                failures.append(f"candidate non-finite kernel {label} is not buildable")
        for label in ("positive_inf_int8", "positive_inf_int32", "positive_inf_int64"):
            if kernels.get(label) is not None:
                failures.append(f"candidate falsely builds non-finite integer kernel {label}")

    public_rows = candidate.get("public_rework")
    public = {row.get("label"): row for row in public_rows or [] if isinstance(row, dict)}
    expected_public = {
        "integral_index_unsafe": ("NotInLeanScope", 0),
        "float_index_unsafe": ("NotInLeanScope", 0),
        "byte_product_refusal": ("NotInLeanScope", 0),
        "positive_inf_f32": ("AllocationTripwire", 1),
        "negative_inf_f32": ("AllocationTripwire", 1),
        "nan_step": ("ValueError", 0),
        "infinite_result": ("OverflowError", 0),
        "huge_mixed": ("OverflowError", 0),
    }
    for dtype in ("int8", "int32", "int64"):
        for endpoints in ("float", "int"):
            expected_public[f"positive_inf_{dtype}_{endpoints}_endpoints"] = ("OverflowError", 0)
            expected_public[f"negative_inf_{dtype}_{endpoints}_endpoints"] = ("AllocationTripwire", 1)
    expected_public["finite_float_int8"] = ("NotInLeanScope", 0)
    for orientation in ("positive_start", "negative_start", "positive_stop", "negative_stop"):
        for dtype in ("bf16", "f32"):
            expected_public[f"endpoint_{orientation}_{dtype}"] = ("ValueError", 0)
        for dtype in ("int8", "int32", "int64"):
            expected_public[f"endpoint_{orientation}_{dtype}"] = \
                (("ValueError", 0) if orientation == "positive_start" else ("OverflowError", 0))
    if set(public) != set(expected_public):
        failures.append("public rework probe manifest differs")
    else:
        for label, (outcome, calls) in expected_public.items():
            row = public[label]
            if row.get("outcome") != outcome or row.get("allocation_calls") != calls:
                failures.append(f"public rework row {label} mapping/allocation differs")
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
    candidate_repo = (args.candidate_repo or repo).resolve()
    checkout = args.checkout.resolve()
    identity = {"revision": git(checkout, "rev-parse", "HEAD"),
                "tree": git(checkout, "rev-parse", "HEAD^{tree}")}
    failures = architecture_failures(candidate_repo)
    if identity != {"revision": PINNED_REVISION, "tree": PINNED_TREE}:
        failures.append("oracle identity mismatch")

    foreign_rows: list[dict] = []
    foreign_metadata: dict = {}
    foreign_ceildiv: dict = {}
    foreign_rework: dict = {}
    candidate_observation: dict = {}
    diagnostics: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="tgrad-wave8-arange-") as temp_name:
        temp = Path(temp_name)
        (temp / "wave8_probe.py").write_text(FOREIGN_PLUGIN)
        foreign_path = temp / "foreign.json"
        foreign_env = {
            "PYTHONPATH": os.pathsep.join((str(temp), str(checkout))),
            "TGRAD_WAVE8_FOREIGN_TRACE": str(foreign_path),
            "DEV": "CPU", "CACHELEVEL": "0", "DERANDOMIZE_CI": "1",
        }
        foreign_cp = run(
            [str(args.python), "-m", "pytest", "-q", "-p", "wave8_probe", *TARGETS],
            cwd=checkout, env=foreign_env)
        diagnostics["foreign_returncode"] = foreign_cp.returncode
        if foreign_cp.returncode != 0:
            failures.append("pinned foreign leaves did not execute successfully")
            diagnostics["foreign_stderr"] = foreign_cp.stderr[-4000:]
            diagnostics["foreign_stdout"] = foreign_cp.stdout[-4000:]
        if not foreign_path.is_file():
            failures.append("foreign observer did not produce a trace")
        else:
            foreign_payload = json.loads(foreign_path.read_text())
            if not isinstance(foreign_payload, dict) or not isinstance(foreign_payload.get("rows"), list):
                failures.append("foreign observer payload shape is invalid")
            else:
                foreign_rows = foreign_payload["rows"]
                foreign_metadata = {
                    "lifecycle": foreign_payload.get("lifecycle"),
                    "versions": foreign_payload.get("versions"),
                }

        ceildiv_cp = run(
            [str(args.python), "-c", FOREIGN_FLOAT_CEILDIV_PROBE], cwd=checkout,
            env={"PYTHONPATH": str(checkout), "DEV": "CPU", "CACHELEVEL": "0"})
        diagnostics["foreign_float_ceildiv_returncode"] = ceildiv_cp.returncode
        if ceildiv_cp.returncode != 0:
            failures.append("foreign float-ceildiv distinguisher failed")
            diagnostics["foreign_float_ceildiv_stderr"] = ceildiv_cp.stderr[-4000:]
        else:
            try:
                foreign_ceildiv = json.loads(ceildiv_cp.stdout)
            except json.JSONDecodeError as exc:
                failures.append(f"foreign float-ceildiv emitted invalid JSON: {exc}")

        rework_cp = run(
            [str(args.python), "-c", FOREIGN_REWORK_PROBE], cwd=temp,
            env={"PYTHONPATH": str(checkout), "DEV": "CPU", "CACHELEVEL": "0",
                 "CACHEDB": str(temp / "foreign-cache.db")})
        diagnostics["foreign_rework_returncode"] = rework_cp.returncode
        if rework_cp.returncode != 0:
            failures.append("foreign Wave 8 rework observer failed")
            diagnostics["foreign_rework_stderr"] = rework_cp.stderr[-4000:]
            diagnostics["foreign_rework_stdout"] = rework_cp.stdout[-4000:]
        else:
            try:
                foreign_rework = json.loads(rework_cp.stdout)
            except json.JSONDecodeError as exc:
                failures.append(f"foreign Wave 8 rework observer emitted invalid JSON: {exc}")

        groups: dict[str, dict] = {}
        for row in foreign_rows:
            groups.setdefault(context_key(row), row)
        rows_path = temp / "rows.json"
        rows_path.write_text(json.dumps(list(groups.values()), sort_keys=True))
        candidate_path = os.pathsep.join((
            str(candidate_repo / "scripts" / "parity" / "shim"),
            str(candidate_repo / "python"),
        ))
        candidate_cp = run(
            [str(args.python), "-c", CANDIDATE_PROBE], cwd=candidate_repo,
            env={"PYTHONPATH": candidate_path, "TGRAD_LIB": str(args.lib.resolve()),
                 "TGRAD_WAVE8_ROWS": str(rows_path),
                 "DYLD_LIBRARY_PATH": os.pathsep.join((
                     str(candidate_repo / ".lake" / "build" / "lib"),
                     os.environ.get("DYLD_LIBRARY_PATH", ""),
                 ))})
        diagnostics["candidate_returncode"] = candidate_cp.returncode
        if candidate_cp.returncode != 0:
            failures.append("candidate allocation-free observer failed")
            diagnostics["candidate_stderr"] = candidate_cp.stderr[-4000:]
        else:
            try:
                candidate_observation = json.loads(candidate_cp.stdout)
            except json.JSONDecodeError as exc:
                failures.append(f"candidate observer emitted invalid JSON: {exc}")

    if foreign_rows and candidate_observation:
        lifecycle = foreign_metadata.get("lifecycle")
        if not isinstance(lifecycle, dict) or lifecycle.get("collection_patch_applied") is not True \
                or lifecycle.get("collection_restore_called") is not True \
                or lifecycle.get("collection_supported") != ["int32", "bfloat16", "float32"] \
                or not {"int8", "int64"}.issubset(set(lifecycle.get("execution_supported") or [])):
            failures.append("foreign collection dtype restriction/restoration was not observed")
        failures.extend(validate_rows(foreign_rows, candidate_observation))
        failures.extend(validate_inherited(candidate_observation))
        failures.extend(validate_rework(foreign_rework, candidate_observation))
        if set(candidate_observation.get("global_supported", [])) != GLOBAL_SUPPORTED:
            failures.append("global compute dtype set widened or narrowed")
        if set(candidate_observation.get("range_supported", [])) != RANGE_SUPPORTED:
            failures.append("range-local dtype admission set widened or narrowed")
        if set(candidate_observation.get("emulated", [])) or candidate_observation.get("emulated_tolist") != []:
            failures.append("backend emulation relation is not empty")
        counts = candidate_observation.get("resolution_query_counts")
        if not isinstance(counts, list) or len(counts) != 2 or counts[0] != counts[1] or counts[0] > 20:
            failures.append("public resolution metadata work scales with range length")
        public_counts = candidate_observation.get("public_preallocation_query_counts")
        if not isinstance(public_counts, list) or len(public_counts) != 2 \
                or public_counts[0] != public_counts[1] or public_counts[0] > 24:
            failures.append("transitive public pre-allocation work scales with range length")
        expected_rejections = {
            "zero_step": "ZeroDivisionError",
            "overflow_precedes_zero_i32": "OverflowError",
            "overflow_precedes_zero_i8": "OverflowError",
            "beyond_int64": "OverflowError",
            "unsupported_float16_materializer": "NotInLeanScope",
        }
        if candidate_observation.get("public_rejections") != expected_rejections:
            failures.append("public rejection mapping differs from inherited Wave 7 contract")
        kernels = candidate_observation.get("kernel_observations")
        expected_flags = {"int8": 37, "int64": 41, "float32": 35, "bfloat16": 49}
        if not isinstance(kernels, dict):
            failures.append("actual Lean range kernel observations are missing")
        else:
            for name, flags in expected_flags.items():
                row = kernels.get(name)
                if not isinstance(row, dict) or row.get("source_flags") != flags \
                        or row.get("source_length", 0) <= 0 \
                        or row.get("source_hash") == (1 << 64) - 1:
                    failures.append(f"rendered {name} range kernel structure differs")
        cache_pairs = candidate_observation.get("cache_identity_pairs")
        if not isinstance(cache_pairs, dict) or set(cache_pairs) != {
                "signed_zero", "adjacent_float_bits", "numeric_mode", "dtype", "length"}:
            failures.append("cache-identity distinguishing pairs are missing")
        else:
            for name, pair in cache_pairs.items():
                if not isinstance(pair, list) or len(pair) != 2 \
                        or not all(isinstance(row, dict) for row in pair) \
                        or pair[0].get("name_hash") == pair[1].get("name_hash") \
                        or pair[0].get("source_hash") == pair[1].get("source_hash"):
                    failures.append(f"range cache identity merges {name}")
        ceildiv = candidate_observation.get("float_ceildiv_audit")
        resolution = ceildiv.get("resolution") if isinstance(ceildiv, dict) else None
        if foreign_ceildiv != {"shape": [1000000001], "dtype": "float32"}:
            failures.append("foreign float-ceildiv distinguisher changed")
        if not isinstance(resolution, dict) or resolution.get("length") != 1000000001 \
                or resolution.get("dtype_code") != 1:
            failures.append("Lean float ceildiv differs from pinned Python floor division")
        elif ceildiv.get("indices") != [0, 1, 1000000000] \
                or len(ceildiv.get("stored_bits", [])) != 3 \
                or any(bits == (1 << 64) - 1 for bits in ceildiv["stored_bits"]):
            failures.append("billion-scale indexed stored-bit audit is incomplete")

    result = {
        "status": "green" if not failures else "red",
        "oracle": identity,
        "targets": list(TARGETS),
        "foreign_execution_count": len(foreign_rows),
        "foreign_semantic_count": len({context_key(row) for row in foreign_rows}),
        "foreign_float_ceildiv": foreign_ceildiv,
        "foreign_rework": foreign_rework,
        "foreign_observer": foreign_metadata,
        "global_supported_codes": candidate_observation.get("global_supported"),
        "range_supported_codes": candidate_observation.get("range_supported"),
        "emulated_codes": candidate_observation.get("emulated"),
        "resolution_query_counts": candidate_observation.get("resolution_query_counts"),
        "public_preallocation_query_counts": candidate_observation.get("public_preallocation_query_counts"),
        "float_ceildiv_audit": candidate_observation.get("float_ceildiv_audit"),
        "kernel_observations": candidate_observation.get("kernel_observations"),
        "cache_identity_pairs": candidate_observation.get("cache_identity_pairs"),
        "wave8_rework": candidate_observation.get("wave8_rework"),
        "public_rework": candidate_observation.get("public_rework"),
        "rework_kernels": candidate_observation.get("rework_kernels"),
        "failures": failures,
        "diagnostics": diagnostics,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Foreign-grounded Wave 9 low-precision scalar authority observer.

The pinned tinygrad checkout supplies every expected conversion result.  The
candidate is observed twice for every row: through the strict public shim and
through the integer-bit Lean boundary.  This file deliberately contains no
FP8 descriptor, threshold, rounding rule, or expected result table.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
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
    "test/backend/test_dtype.py::TestFp8sConversions::test_float_to_fp8e4m3",
    "test/backend/test_dtype.py::TestFp8sConversions::test_float_to_fp8e5m2",
    "test/backend/test_dtype.py::TestFp8sConversions::test_float_to_fp8e5m2_extreme_values",
    "test/backend/test_dtype.py::TestFp8sConversions::test_fp8e4m3_to_float",
    "test/backend/test_dtype.py::TestFp8sConversions::test_fp8e5m2_to_float",
    "test/backend/test_dtype.py::TestFp8sConversions::test_float_to_fp8e4m3fnuz",
    "test/backend/test_dtype.py::TestFp8sConversions::test_float_to_fp8e4m3fnuz_extreme_values",
    "test/backend/test_dtype.py::TestFp8sConversions::test_float_to_fp8e5m2fnuz",
    "test/backend/test_dtype.py::TestFp8sConversions::test_float_to_fp8e5m2fnuz_extreme_values",
    "test/backend/test_dtype.py::TestFp8sConversions::test_fp8e4m3fnuz_to_float",
    "test/backend/test_dtype.py::TestFp8sConversions::test_fp8e5m2fnuz_to_float",
    "test/null/test_dtype_spec.py::TestHelpers::test_float_to_fp16",
    "test/null/test_dtype_spec.py::TestHelpers::test_truncate_fp8e4m3",
    "test/null/test_dtype_spec.py::TestHelpers::test_truncate_fp8e5m2",
)

FORMAT_NAMES = ("fp8e4m3", "fp8e5m2", "fp8e4m3fnuz", "fp8e5m2fnuz")
MISSING_AUTHORITY_DIAGNOSTIC = "missing Lean low-precision scalar authority"


FOREIGN_PROBE = r'''
import json, math, struct
from decimal import Decimal
from fractions import Fraction
import numpy as np
from tinygrad.dtype import dtypes, float_to_fp16, float_to_fp8, fp8_to_float, truncate

def bits(value): return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
def value(raw): return struct.unpack("<d", struct.pack("<Q", raw))[0]

formats = {name: getattr(dtypes, name) for name in
  ("fp8e4m3", "fp8e5m2", "fp8e4m3fnuz", "fp8e5m2fnuz")}
rows = []
decoded, encode_inputs = {}, {}
for name, dtype in formats.items():
  vals = []
  for payload in range(256):
    out = fp8_to_float(payload, dtype)
    out_bits = bits(out)
    rows.append({"op":"decode", "dtype":name, "input":payload, "expected":out_bits})
    if math.isfinite(out) and out >= 0: vals.append(out)
  decoded[name] = sorted(set(vals))

# Input selection is derived from foreign decoded values.  The answers still
# come only from executing the foreign helpers.  Midpoints and their adjacent
# binary64 values distinguish ties-to-even from unconditional rounding.
fixed_bits = {
  0x0000000000000000, 0x8000000000000000,
  0x0000000000000001, 0x8000000000000001,
  0x3ff0000000000000, 0xbff0000000000000,
  0x7ff0000000000000, 0xfff0000000000000,
  0x7ff8000000000000, 0xfff8000000000000,
  0x7ff0000000000001, 0xfff0000000000001,
  0x7fffffffffffffff, 0xffffffffffffffff,
}
for name, dtype in formats.items():
  inputs = set(fixed_bits)
  vals = decoded[name]
  for v in vals:
    inputs.add(bits(v)); inputs.add(bits(-v))
  for lo, hi in zip(vals, vals[1:]):
    mid = (lo + hi) / 2.0
    for v in (math.nextafter(mid, -math.inf), mid, math.nextafter(mid, math.inf)):
      inputs.add(bits(v)); inputs.add(bits(-v))
  maximum = max(vals)
  previous = vals[-2]
  overflow_midpoint = maximum + (maximum - previous) / 2.0
  for v in (maximum, math.nextafter(maximum, math.inf), maximum * 1.01,
            overflow_midpoint, math.nextafter(overflow_midpoint, -math.inf),
            math.nextafter(overflow_midpoint, math.inf),
            -maximum, math.nextafter(-maximum, -math.inf), -maximum * 1.01,
            -overflow_midpoint, math.nextafter(-overflow_midpoint, -math.inf),
            math.nextafter(-overflow_midpoint, math.inf)):
    inputs.add(bits(v))
  encode_inputs[name] = inputs
  for raw in sorted(inputs):
    x = value(raw)
    rows.append({"op":"encode", "dtype":name, "input":raw,
                 "expected":float_to_fp8(x, dtype)})

fp16_inputs = set(fixed_bits)
for x in (1, 1.1, 1e-8, -1e-8, 65504, 65519.999, 65520,
          -65504, -65519.999, -65520): fp16_inputs.add(bits(x))
# Selected adjacent half payloads cover zero/subnormal, min-normal, exponent,
# even/odd tie and overflow transitions without copying a conversion rule.
half_payloads = (0, 1, 2, 3, 0x3fe, 0x3ff, 0x400, 0x401, 0x7bfe, 0x7bff)
half_values = [struct.unpack("<e", struct.pack("<H", payload))[0] for payload in half_payloads]
for value_ in half_values:
  fp16_inputs.add(bits(value_)); fp16_inputs.add(bits(-value_))
for lo, hi in zip(half_values, half_values[1:]):
  mid = (lo + hi) / 2.0
  for value_ in (math.nextafter(mid, -math.inf), mid, math.nextafter(mid, math.inf)):
    fp16_inputs.add(bits(value_)); fp16_inputs.add(bits(-value_))
for raw in sorted(fp16_inputs):
  rows.append({"op":"fp16", "dtype":"float16", "input":raw,
               "expected":bits(float_to_fp16(value(raw)))})

for name in ("fp8e4m3", "fp8e5m2"):
  dtype = formats[name]
  for raw in sorted(encode_inputs[name]):
    rows.append({"op":"truncate", "dtype":name, "input":raw,
                 "expected":bits(truncate[dtype](value(raw)))})

foreign_rejections = {}
bad_dtypes = {"string":"fp8e4m3", "integer":10, "none":None,
              "float32":dtypes.float32, "float16":dtypes.float16}
calls = {}
for label, dtype in bad_dtypes.items():
  calls["encode_"+label] = lambda dtype=dtype: float_to_fp8(0.0, dtype)
  calls["decode_"+label] = lambda dtype=dtype: fp8_to_float(0, dtype)
for label, call in calls.items():
  try: call()
  except Exception as exc: foreign_rejections[label] = type(exc).__name__+":"+str(exc)
  else: foreign_rejections[label] = "accepted"
payload_rows = {}
public_value_rows = {}
for format_name, dtype in formats.items():
  for label, payload in {"string":"1", "float":1.5, "none":None, "bool":True,
                         "negative":-1, "huge":(1 << 130) + 1}.items():
    key = format_name+":"+label
    try: payload_rows[key] = {"value":bits(fp8_to_float(payload, dtype))}
    except Exception as exc: payload_rows[key] = {"error":type(exc).__name__+":"+str(exc)}
  for label, value_ in {"string":"1", "none":None, "float":1.5,
                        "int":1, "bool":True, "numpy_float16":np.float16(1.5),
                        "numpy_float32":np.float32(1.5), "numpy_float64":np.float64(1.5),
                        "numpy_int32":np.int32(1), "decimal":Decimal("1.5"),
                        "fraction":Fraction(3, 2), "huge_positive":1 << 2000,
                        "huge_negative":-(1 << 2000)}.items():
    for operation, call in {
      "encode":lambda value_=value_, dtype=dtype: float_to_fp8(value_, dtype),
      "truncate":lambda value_=value_, dtype=dtype: truncate[dtype](value_),
    }.items():
      key = format_name+":"+operation+":"+label
      try:
        out = call()
        public_value_rows[key] = {"value":int(out) if operation == "encode" else bits(out)}
      except Exception as exc:
        public_value_rows[key] = {"error":type(exc).__name__+":"+str(exc)}
try: public_value_rows["combined_invalid"] = {"value":float_to_fp8(None, dtypes.float32)}
except Exception as exc: public_value_rows["combined_invalid"] = {"error":type(exc).__name__+":"+str(exc)}
try: public_value_rows["combined_overflow"] = {"value":float_to_fp8(1 << 2000, dtypes.float32)}
except Exception as exc: public_value_rows["combined_overflow"] = {"error":type(exc).__name__+":"+str(exc)}
for label, value_ in {"positive":1 << 2000, "negative":-(1 << 2000)}.items():
  try: public_value_rows["fp16_overflow:"+label] = {"value":bits(float_to_fp16(value_))}
  except Exception as exc: public_value_rows["fp16_overflow:"+label] = {"error":type(exc).__name__+":"+str(exc)}
print(json.dumps({"rows":rows, "foreign_rejections":foreign_rejections,
                  "payload_rows":payload_rows, "public_value_rows":public_value_rows},
                 sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


CANDIDATE_PROBE = r'''
import json, os, struct
from decimal import Decimal
from fractions import Fraction
import numpy as np

rows = json.load(open(os.environ["TGRAD_LOW_PRECISION_ROWS"]))["rows"]
try:
  import tgrad
  from tinygrad.dtype import dtypes, float_to_fp16, float_to_fp8, fp8_to_float, truncate
except Exception as exc:
  print(json.dumps({"initialization_error":type(exc).__name__+":"+str(exc)}))
  raise SystemExit(0)

def bits(value): return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
def value(raw): return struct.unpack("<d", struct.pack("<Q", raw))[0]

op_codes = {"encode":0, "decode":1, "fp16":2, "truncate":3}
observed = []
for row in rows:
  dtype = getattr(dtypes, row["dtype"])
  try:
    if row["op"] == "encode": public = int(float_to_fp8(value(row["input"]), dtype))
    elif row["op"] == "decode": public = bits(fp8_to_float(row["input"], dtype))
    elif row["op"] == "fp16": public = bits(float_to_fp16(value(row["input"])))
    else: public = bits(truncate[dtype](value(row["input"])))
    public_error = None
  except Exception as exc:
    public, public_error = None, type(exc).__name__+":"+str(exc)
  try:
    direct = int(tgrad._low_precision_scalar(op_codes[row["op"]], dtype.code, row["input"]))
    direct_error = None
  except Exception as exc:
    direct, direct_error = None, type(exc).__name__+":"+str(exc)
  observed.append({"public":public, "public_error":public_error,
                   "direct":direct, "direct_error":direct_error})

public_rejections = {}
bad_dtypes = {"string":"fp8e4m3", "integer":10, "none":None,
              "float32":dtypes.float32, "float16":dtypes.float16}
calls = {}
for label, dtype in bad_dtypes.items():
  calls["encode_"+label] = lambda dtype=dtype: float_to_fp8(0.0, dtype)
  calls["decode_"+label] = lambda dtype=dtype: fp8_to_float(0, dtype)
for label, call in calls.items():
  try: call()
  except Exception as exc: public_rejections[label] = type(exc).__name__+":"+str(exc)
  else: public_rejections[label] = "accepted"
payload_rows = {}
public_value_rows = {}
formats = {name:getattr(dtypes, name) for name in
           ("fp8e4m3", "fp8e5m2", "fp8e4m3fnuz", "fp8e5m2fnuz")}
for format_name, dtype in formats.items():
  for label, payload in {"string":"1", "float":1.5, "none":None, "bool":True,
                         "negative":-1, "huge":(1 << 130) + 1}.items():
    key = format_name+":"+label
    try: payload_rows[key] = {"value":bits(fp8_to_float(payload, dtype))}
    except Exception as exc: payload_rows[key] = {"error":type(exc).__name__+":"+str(exc)}
  for label, value_ in {"string":"1", "none":None, "float":1.5,
                        "int":1, "bool":True, "numpy_float16":np.float16(1.5),
                        "numpy_float32":np.float32(1.5), "numpy_float64":np.float64(1.5),
                        "numpy_int32":np.int32(1), "decimal":Decimal("1.5"),
                        "fraction":Fraction(3, 2), "huge_positive":1 << 2000,
                        "huge_negative":-(1 << 2000)}.items():
    for operation, call in {
      "encode":lambda value_=value_, dtype=dtype: float_to_fp8(value_, dtype),
      "truncate":lambda value_=value_, dtype=dtype: truncate[dtype](value_),
    }.items():
      key = format_name+":"+operation+":"+label
      try:
        out = call()
        public_value_rows[key] = {"value":int(out) if operation == "encode" else bits(out)}
      except Exception as exc:
        public_value_rows[key] = {"error":type(exc).__name__+":"+str(exc)}
try: public_value_rows["combined_invalid"] = {"value":float_to_fp8(None, dtypes.float32)}
except Exception as exc: public_value_rows["combined_invalid"] = {"error":type(exc).__name__+":"+str(exc)}
try: public_value_rows["combined_overflow"] = {"value":float_to_fp8(1 << 2000, dtypes.float32)}
except Exception as exc: public_value_rows["combined_overflow"] = {"error":type(exc).__name__+":"+str(exc)}
for label, value_ in {"positive":1 << 2000, "negative":-(1 << 2000)}.items():
  try: public_value_rows["fp16_overflow:"+label] = {"value":bits(float_to_fp16(value_))}
  except Exception as exc: public_value_rows["fp16_overflow:"+label] = {"error":type(exc).__name__+":"+str(exc)}

unsupported = []
for op, dtype in ((0, dtypes.float32), (1, dtypes.float16), (2, dtypes.fp8e4m3)):
  try: tgrad._low_precision_scalar(op, dtype.code, 0)
  except Exception as exc: unsupported.append(type(exc).__name__)
  else: unsupported.append("accepted")
reason_sweep = []
valid_codes = [code for code in range(255) if tgrad._dtype_query(code, 0) == 1]
for code in valid_codes + [19, 253, 255]:
  reasons = [int(tgrad._lib.tgrad_low_precision_scalar(op, code, 0, 0)) for op in range(4)]
  reason_sweep.append({"code":code, "name":tgrad._dtype_public_name(code) if code in valid_codes else "invalid",
                       "reasons":reasons})
print(json.dumps({"observed":observed, "unsupported":unsupported,
                  "public_rejections":public_rejections,
                  "payload_rows":payload_rows,
                  "public_value_rows":public_value_rows,
                  "reason_sweep":reason_sweep,
                  "compute_supported":[code for code in valid_codes if tgrad._dtype_query(code, 15) == 1],
                  "emulated":[code for code in valid_codes if tgrad._dtype_query(code, 17) == 1]},
                 sort_keys=True, separators=(",", ":")))
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_hygiene_failures(repo: Path) -> tuple[list[str], list[str]]:
    """Check staged/unstaged diffs and directly scan every untracked packet file.

    ``git diff --check`` deliberately ignores untracked files.  Wave 9's new
    observer was therefore outside the check that its evidence claimed was
    green.  Keep Git's own checks for tracked content, then scan the union of
    changed, staged, and untracked working-tree paths plus this observer.
    """
    failures: list[str] = []
    checked: set[Path] = {Path(__file__).resolve()}
    for label, arguments in (
        ("unstaged", ("diff", "--check")),
        ("staged", ("diff", "--cached", "--check")),
    ):
        check = subprocess.run(["git", "-C", str(repo), *arguments], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check.returncode != 0:
            detail = (check.stdout + check.stderr).strip()
            failures.append(f"source hygiene {label} diff failed: {detail}")

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


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def isolated_env(temp: Path, **extra: str) -> dict[str, str]:
    home = temp / "home"
    scratch = temp / "tmp"
    home.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    env = {
        "HOME":str(home), "TMPDIR":str(scratch), "LANG":"C", "LC_ALL":"C",
        "PATH":os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED":"0",
        "PYTHONNOUSERSITE":"1", "PYTHONSAFEPATH":"1",
        "PYTHONDONTWRITEBYTECODE":"1", "CACHELEVEL":"0",
    }
    env.update(extra)
    return env


def snapshot_oracle(checkout: Path, destination: Path) -> None:
    archive = subprocess.run(["git", "-C", str(checkout), "archive", "--format=tar", EXPECTED.revision],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as stream:
        stream.extractall(destination, filter="data")


def function_node(source: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    return next((node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name), None)


def architecture_failures(repo: Path) -> list[str]:
    failures: list[str] = []
    lean_path = repo / "Tgrad" / "LowPrecision.lean"
    if not lean_path.is_file():
        lean = ""
    else:
        lean = re.sub(r"/-.*?-/", "", lean_path.read_text(), flags=re.S)
        lean = re.sub(r"--[^\n]*", "", lean)
    missing_components = [required for required in
                          ("private structure Fp8Descriptor", "def descriptorFor?",
                           "def convert", "descriptor_invariants_hold",
                           "supported_descriptor_dtypes_nodup")
                          if required not in lean]
    if missing_components:
        failures.append(MISSING_AUTHORITY_DIAGNOSTIC)
        failures.extend(f"Lean authority missing {required}"
                        for required in missing_components)
    if "Dtype.computeSupported" in lean:
        failures.append("scalar authority changes tensor compute admission")

    py_path = repo / "scripts" / "parity" / "shim" / "tinygrad" / "dtype.py"
    py_source = py_path.read_text()
    expected_delegate = {
        "float_to_fp16":"_low_precision_scalar",
        "float_to_fp8":"_low_precision_value_public",
        "fp8_to_float":"_low_precision_decode_public",
    }
    for name in ("float_to_fp16", "float_to_fp8", "fp8_to_float"):
        node = function_node(py_source, name)
        if node is None:
            failures.append(f"strict shim missing {name}")
            continue
        text = ast.get_source_segment(py_source, node) or ""
        if expected_delegate[name] not in text:
            failures.append(f"{name} does not delegate to Lean")
        forbidden_nodes = (ast.For, ast.AsyncFor, ast.While, ast.ListComp,
                           ast.SetComp, ast.DictComp, ast.GeneratorExp,
                           ast.BinOp, ast.BoolOp)
        if any(isinstance(item, forbidden_nodes) for item in ast.walk(node)):
            failures.append(f"{name} contains conversion semantics in Python")
        for token in ("numpy", "torch", "ldexp", "nextafter", "copysign", "isfinite"):
            if token in text:
                failures.append(f"{name} contains forbidden semantic helper {token}")
        if name in ("float_to_fp8", "fp8_to_float") and "to_dtype(dtype)" in text:
            failures.append(f"{name} normalizes non-DType public arguments")
        if name == "fp8_to_float" and "int(payload)" in text:
            failures.append("fp8_to_float coerces non-integral payloads")

    c_source = (repo / "c" / "tgrad_python.c").read_text()
    match = re.search(r"uint64_t\s+tgrad_low_precision_scalar\s*\([^)]*\)\s*\{(.*?)\n\}",
                      c_source, flags=re.S)
    if match is None:
        failures.append("missing C low-precision marshalling trampoline")
    else:
        body = match.group(1)
        for required in ("tgrad_low_precision_scalar_lean", "lean_unbox_uint64",
                         "lean_io_result_is_error"):
            if required not in body:
                failures.append(f"C low-precision trampoline missing {required}")
        for token in ("0x", "ldexp", "isfinite", "copysign", "memcpy", "struct"):
            if token in body:
                failures.append(f"C low-precision trampoline owns semantics via {token}")

    product_python = (repo / "python" / "tgrad.py").read_text()
    helper = function_node(product_python, "_low_precision_scalar")
    if helper is None:
        failures.append("product Python low-precision marshaller is missing")
    else:
        helper_text = ast.get_source_segment(product_python, helper) or ""
        if "_lib.tgrad_low_precision_scalar" not in helper_text:
            failures.append("product Python helper does not call the C trampoline")
        if any(isinstance(item, (ast.For, ast.AsyncFor, ast.While, ast.ListComp,
                                 ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.BinOp))
               for item in ast.walk(helper)):
            failures.append("product Python helper owns conversion work")
    decode_helper = function_node(product_python, "_low_precision_decode_public")
    if decode_helper is None:
        failures.append("product Python public-decode marshaller is missing")
    else:
        decode_text = ast.get_source_segment(product_python, decode_helper) or ""
        for required in ("isinstance(payload, int)", "str(payload).encode",
                         "_lib.tgrad_low_precision_decode_public"):
            if required not in decode_text:
                failures.append(f"public-decode marshaller missing {required}")
        for forbidden in ("payload &", "% 256", "0xff"):
            if forbidden in decode_text.lower():
                failures.append(f"public-decode marshaller owns payload semantics via {forbidden}")
    value_helper = function_node(product_python, "_low_precision_value_public")
    if value_helper is None:
        failures.append("product Python public-value marshaller is missing")
    else:
        value_text = ast.get_source_segment(product_python, value_helper) or ""
        for required in ("try:", "struct.pack(\"<d\", value)",
                         "except OverflowError", "operator.index(value)",
                         "value_tag, input_bits = 2, 0",
                         "_lib.tgrad_low_precision_value_public"):
            if required not in value_text:
                failures.append(f"public-value marshaller missing {required}")
        for forbidden in ("float(value)", "numpy", "torch"):
            if forbidden in value_text:
                failures.append(f"public-value marshaller owns value semantics via {forbidden}")
    combined_boundary = py_source + product_python + c_source
    for forbidden in ("_fp8_cfg", "min_denorm_half", "ovf_threshold",
                      "0x3F50000000000000", "0x407D000000000000"):
        if forbidden in combined_boundary:
            failures.append(f"non-Lean boundary contains descriptor semantics {forbidden}")

    dtype_source = (repo / "Tgrad" / "Dtype.lean").read_text()
    expected_compute = "| .bfloat16_ | .float32_ | .int32_ => true"
    if expected_compute not in dtype_source:
        failures.append("tensor compute dtype admission changed")
    return failures


def authority_diagnostic_contract(failures: list[str]) -> tuple[int, int, str | None]:
    """Require one aggregate diagnostic iff component-level authority is absent."""
    count = failures.count(MISSING_AUTHORITY_DIAGNOSTIC)
    expected = int(any(item.startswith("Lean authority missing ") for item in failures))
    issue = None if count == expected else \
        f"Lean authority aggregate diagnostic count {count}, expected {expected}"
    return count, expected, issue


def validate_rows(foreign: dict, candidate: dict) -> list[str]:
    failures: list[str] = []
    rows = foreign.get("rows")
    observed = candidate.get("observed")
    if not isinstance(rows, list) or not isinstance(observed, list) or len(rows) != len(observed):
        return ["candidate row manifest does not match foreign manifest"]
    decode_counts = {name: 0 for name in FORMAT_NAMES}
    for index, (row, got) in enumerate(zip(rows, observed)):
        if row["op"] == "decode": decode_counts[row["dtype"]] += 1
        expected = row["expected"]
        if len(failures) < 40 and (got.get("public_error") is not None or got.get("public") != expected):
            failures.append(f"public row {index} differs from foreign")
        if len(failures) < 40 and (got.get("direct_error") is not None or got.get("direct") != expected):
            failures.append(f"direct Lean row {index} differs from foreign")
    if decode_counts != {name: 256 for name in FORMAT_NAMES}:
        failures.append(f"full FP8 decode domain missing: {decode_counts}")
    if candidate.get("unsupported") != ["TgradTypeError"] * 3:
        failures.append("unsupported dtype/operation rejection differs")
    foreign_rejections = foreign.get("foreign_rejections")
    public_rejections = candidate.get("public_rejections")
    if not isinstance(foreign_rejections, dict) or not isinstance(public_rejections, dict) \
            or set(foreign_rejections) != set(public_rejections):
        failures.append("public dtype rejection manifest differs")
    else:
        for label, expected in foreign_rejections.items():
            if expected != "AssertionError:Only for fp8s" \
                    or public_rejections.get(label) != expected:
                failures.append(f"public dtype rejection differs for {label}")
    if candidate.get("payload_rows") != foreign.get("payload_rows"):
        failures.append("public decode payload admission/results differ")
    if candidate.get("public_value_rows") != foreign.get("public_value_rows"):
        failures.append("public encode/truncate value admission/results differ")
    expected_names = set(FORMAT_NAMES)
    sweep = candidate.get("reason_sweep")
    if not isinstance(sweep, list):
        failures.append("direct typed-reason sweep missing")
    else:
        for entry in sweep:
            name, reasons = entry.get("name"), entry.get("reasons")
            expected = [0, 0, 1, 0] if name in expected_names else \
                       [1, 1, 0, 1] if name == "float16" else [1, 1, 1, 1]
            if reasons != expected:
                failures.append(f"typed reason sweep differs for {name}")
    if candidate.get("compute_supported") != [0, 1, 3]:
        failures.append("compute-supported dtype set changed")
    if candidate.get("emulated") != []:
        failures.append("emulated dtype relation changed")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--candidate-head")
    parser.add_argument("--candidate-tree")
    parser.add_argument("--require-candidate-clean", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    candidate_repo = (args.candidate_repo or repo).resolve()
    checkout = args.checkout.resolve()
    # Preserve a virtualenv launcher symlink: resolving it selects the base
    # interpreter and silently loses that environment's installed observer
    # dependencies.  Only make a relative spelling absolute.
    python = args.python if args.python.is_absolute() else Path.cwd() / args.python
    failures = architecture_failures(candidate_repo)
    hygiene_failures, hygiene_paths = source_hygiene_failures(candidate_repo)
    failures.extend(hygiene_failures)
    generic_count, generic_expected, generic_issue = authority_diagnostic_contract(failures)
    if generic_issue is not None:
        failures.append(generic_issue)
    diagnostics: dict[str, object] = {
        "lean_authority_generic_count": generic_count,
        "lean_authority_generic_expected": generic_expected,
        "source_hygiene_checked_paths": hygiene_paths,
    }
    candidate_identity = {
        "top_level":git_text(candidate_repo, "rev-parse", "--show-toplevel"),
        "head":git_text(candidate_repo, "rev-parse", "HEAD"),
        "tree":git_text(candidate_repo, "rev-parse", "HEAD^{tree}"),
        "status":git_text(candidate_repo, "status", "--porcelain", "--untracked-files=all"),
    }
    if not os.path.samefile(candidate_identity["top_level"], candidate_repo):
        failures.append("candidate repository discovery escaped the subject directory")
    if args.candidate_head and candidate_identity["head"] != args.candidate_head:
        failures.append("candidate HEAD identity mismatch")
    if args.candidate_tree and candidate_identity["tree"] != args.candidate_tree:
        failures.append("candidate tree identity mismatch")
    if args.require_candidate_clean and candidate_identity["status"]:
        failures.append("candidate subject is not clean")
    ok, detail = verify(checkout)
    diagnostics["oracle_verification"] = detail
    if not ok:
        failures.append("pinned oracle source closure failed")

    foreign_payload: dict = {}
    candidate_payload: dict = {}
    with tempfile.TemporaryDirectory(prefix="tgrad-wave9-low-precision-") as name:
        temp = Path(name)
        snapshot = temp / "oracle"
        snapshot.mkdir()
        snapshot_oracle(checkout, snapshot)
        foreign_cp = run([str(python), "-c", FOREIGN_PROBE], cwd=temp,
                         env=isolated_env(temp, PYTHONPATH=str(snapshot), DEV="CPU"))
        diagnostics["foreign_probe_returncode"] = foreign_cp.returncode
        if foreign_cp.returncode != 0:
            failures.append("foreign low-precision probe failed")
            diagnostics["foreign_probe_stdout"] = foreign_cp.stdout[-4000:]
            diagnostics["foreign_probe_stderr"] = foreign_cp.stderr[-4000:]
        else:
            try: foreign_payload = json.loads(foreign_cp.stdout)
            except json.JSONDecodeError as exc: failures.append(f"foreign probe JSON invalid: {exc}")

        rows_path = temp / "foreign_rows.json"
        rows_path.write_text(json.dumps(foreign_payload, sort_keys=True, separators=(",", ":")))
        candidate_path = os.pathsep.join((
            str(candidate_repo / "scripts" / "parity" / "shim"),
            str(candidate_repo / "python"),
        ))
        candidate_cp = run([str(python), "-c", CANDIDATE_PROBE], cwd=candidate_repo,
                           env=isolated_env(temp, PYTHONPATH=candidate_path,
                                TGRAD_LIB=str(args.lib.resolve()),
                                TGRAD_LOW_PRECISION_ROWS=str(rows_path),
                                DYLD_LIBRARY_PATH=str(candidate_repo / ".lake" / "build" / "lib")))
        diagnostics["candidate_probe_returncode"] = candidate_cp.returncode
        if candidate_cp.returncode != 0:
            failures.append("candidate low-precision probe failed")
            diagnostics["candidate_probe_stdout"] = candidate_cp.stdout[-4000:]
            diagnostics["candidate_probe_stderr"] = candidate_cp.stderr[-4000:]
        else:
            try: candidate_payload = json.loads(candidate_cp.stdout)
            except json.JSONDecodeError as exc: failures.append(f"candidate probe JSON invalid: {exc}")

        pytest_args = ["-q", "-p", "no:cacheprovider", "--hypothesis-seed=0", *TARGETS]
        foreign_pytest = run([str(python), "-m", "pytest", *pytest_args], cwd=snapshot,
                             env=isolated_env(temp, PYTHONPATH=str(snapshot), DEV="CPU"))
        diagnostics["foreign_pytest_returncode"] = foreign_pytest.returncode
        diagnostics["foreign_pytest_tail"] = (foreign_pytest.stdout + foreign_pytest.stderr)[-2000:]
        if foreign_pytest.returncode != 0 or not re.search(r"\b14 passed\b", foreign_pytest.stdout):
            failures.append("exact pinned 14-leaf selection is not 14 passed")
        shim_runner = candidate_repo / "scripts" / "parity" / "shim" / "run_pytest.py"
        verification = run([str(python), str(shim_runner), "--verify-only"], cwd=snapshot,
                           env=isolated_env(temp, PYTHONPATH=str(candidate_repo / "python"),
                                TGRAD_LIB=str(args.lib.resolve()),
                                DYLD_LIBRARY_PATH=str(candidate_repo / ".lake" / "build" / "lib")))
        diagnostics["strict_verification_returncode"] = verification.returncode
        diagnostics["strict_verification"] = verification.stdout.strip()
        if verification.returncode != 0:
            failures.append("strict substitution verification failed")
        candidate_pytest = run([str(python), str(shim_runner), *pytest_args], cwd=snapshot,
                               env=isolated_env(temp, PYTHONPATH=str(candidate_repo / "python"),
                                    TGRAD_LIB=str(args.lib.resolve()),
                                    DYLD_LIBRARY_PATH=str(candidate_repo / ".lake" / "build" / "lib")))
        diagnostics["candidate_pytest_returncode"] = candidate_pytest.returncode
        diagnostics["candidate_pytest_tail"] = (candidate_pytest.stdout + candidate_pytest.stderr)[-2000:]
        if candidate_pytest.returncode != 0 or not re.search(r"\b14 passed\b", candidate_pytest.stdout):
            failures.append("strict substitution exact 14-leaf selection is not 14 passed")

        ok_after, detail_after = verify(checkout)
        diagnostics["oracle_reverification"] = detail_after
        if not ok_after:
            failures.append("live oracle changed during observation")

    if foreign_payload and candidate_payload:
        failures.extend(validate_rows(foreign_payload, candidate_payload))

    paths = [Path(__file__), candidate_repo / "Tgrad" / "Dtype.lean",
             candidate_repo / "Tgrad" / "PythonFFI.lean",
             candidate_repo / "c" / "tgrad_python.c",
             candidate_repo / "python" / "tgrad.py",
             candidate_repo / "scripts" / "parity" / "shim" / "tinygrad" / "dtype.py",
             candidate_repo / "scripts" / "parity" / "shim" / "run_pytest.py",
             args.lib.resolve(), candidate_repo / ".lake" / "build" / "lib" / "libtgrad_Tgrad.dylib",
             checkout / "test" / "backend" / "test_dtype.py",
             checkout / "test" / "null" / "test_dtype_spec.py"]
    if (candidate_repo / "Tgrad" / "LowPrecision.lean").exists():
        paths.append(candidate_repo / "Tgrad" / "LowPrecision.lean")
    report = {
        "schema":"tgrad.wave9.low_precision_scalar.v1",
        "oracle":{"revision":EXPECTED.revision, "tree":EXPECTED.tree},
        "candidate_identity":candidate_identity,
        "target_count":len(TARGETS),
        "foreign_row_count":len(foreign_payload.get("rows", [])),
        "foreign_rows_sha256":hashlib.sha256(json.dumps(foreign_payload, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "target_manifest_sha256":hashlib.sha256("\n".join(TARGETS).encode()).hexdigest(),
        "python_launcher":str(python),
        "python_resolved":str(python.resolve()),
        "python_resolved_sha256":sha256(python.resolve()),
        "hashes":{str(path.relative_to(candidate_repo) if path.is_relative_to(candidate_repo) else path):sha256(path)
                  for path in paths},
        "diagnostics":diagnostics,
        "failures":failures,
        "result":"pass" if not failures else "fail",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

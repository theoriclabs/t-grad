#!/usr/bin/env python3
"""Executable boundary check for Tgrad's exact compute-admission relation."""
from __future__ import annotations

import ctypes
import json

import tgrad


EXPECTED_PUBLIC = {"bfloat16", "float32", "int32"}
EXPECTED_AUTHORING = {"bf16", "f32", "i32"}
NEGATIVE_CODES = {
    "float16": 2,
    "fp8e4m3": 14,
    "fp8e5m2": 15,
    "fp8e4m3fnuz": 16,
    "fp8e5m2fnuz": 17,
}


def main() -> int:
    valid_codes = [code for code in range(255) if tgrad._dtype_query(code, 0) == 1]
    supported_codes = [code for code in valid_codes if tgrad._dtype_query(code, 15) == 1]
    observed_public = {tgrad._dtype_public_name(code) for code in supported_codes}
    observed_authoring = set(tgrad._SUPPORTED_DTYPES)

    shape = (ctypes.c_size_t * 1)(1)
    negatives = {
        name: int(tgrad._lib.tgrad_tensor_from_buffer(1, shape, 1, code))
        for name, code in NEGATIVE_CODES.items()
    }
    failures = []
    if observed_public != EXPECTED_PUBLIC:
        failures.append(
            f"Lean compute set {sorted(observed_public)} != {sorted(EXPECTED_PUBLIC)}")
    if observed_authoring != EXPECTED_AUTHORING:
        failures.append(
            f"Python projection {sorted(observed_authoring)} != {sorted(EXPECTED_AUTHORING)}")
    admitted_negatives = {name: handle for name, handle in negatives.items() if handle != 0}
    if admitted_negatives:
        failures.append(f"unsupported tensor registrations admitted: {admitted_negatives}")

    print(json.dumps({
        "status": "green" if not failures else "red",
        "lean_public": sorted(observed_public),
        "python_authoring": sorted(observed_authoring),
        "negative_handles": negatives,
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

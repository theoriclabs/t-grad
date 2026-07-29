#!/usr/bin/env python3
"""Extract the pinned tinygrad dtype semantics as a reproducible requirement.

This is a foreign observer: all semantic values are read from the pinned
tinygrad module, never from Tgrad.  The output is deliberately unpromoted and
is intended to live below ``var/``.  Extraction is fail-closed: identity or
semantic-shape disagreement leaves no output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any


PINNED_REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
REQUIREMENT_SCHEMA = "tgrad.foreign-dtype-core.v2"
EXPECTED_SEMANTIC_NAMES = (
    "weakint", "bool", "int8", "uint8", "int16", "uint16", "int32",
    "uint32", "int64", "uint64", "weakfloat", "fp8e4m3", "fp8e5m2",
    "fp8e4m3fnuz", "fp8e5m2fnuz", "float16", "bfloat16", "float32",
    "float64",
)
COLLECTIONS = (
    "fp8_ocp", "fp8_fnuz", "fp8s", "floats", "int8s", "int16s",
    "int32s", "int64s", "uints", "sints", "ints", "weaks", "all",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(checkout: Path, *args: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(checkout), *args], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return cp.stdout.strip()


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def json_range(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "decimal": str(value)}
    if isinstance(value, float) and math.isinf(value):
        return {"kind": "infinity", "sign": -1 if value < 0 else 1}
    raise TypeError(f"unsupported range value {value!r}")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def extract(checkout: Path) -> dict[str, Any]:
    checkout = checkout.resolve()
    revision = git(checkout, "rev-parse", "HEAD")
    if revision != PINNED_REVISION:
        raise RuntimeError(
            f"oracle revision mismatch: expected {PINNED_REVISION}, got {revision}")
    tree = git(checkout, "rev-parse", "HEAD^{tree}")
    source = checkout / "tinygrad" / "dtype.py"
    source_bytes = source.read_bytes()

    # Make the foreign checkout authoritative even if the caller has imported
    # another tinygrad.  Import failure aborts before any output is written.
    for name in tuple(sys.modules):
        if name == "tinygrad" or name.startswith("tinygrad."):
            del sys.modules[name]
    sys.path.insert(0, str(checkout))
    try:
        mod = importlib.import_module("tinygrad.dtype")
    finally:
        sys.path.pop(0)

    DType, dtypes = mod.DType, mod.dtypes
    ordered: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for name, value in dtypes.__dict__.items():
        if name.startswith("_") or not isinstance(value, DType):
            continue
        if id(value) not in seen:
            ordered.append((name, value))
            seen.add(id(value))
    canonical = {value: name for name, value in ordered}
    semantic_names = tuple(name for name, _ in ordered if name != "void")
    if semantic_names != EXPECTED_SEMANTIC_NAMES:
        raise RuntimeError(
            f"unexpected semantic dtype order: {semantic_names!r}")

    aliases: dict[str, str] = {}
    for name, value in dtypes.__dict__.items():
        if name.startswith("_") or not isinstance(value, DType):
            continue
        target = canonical[value]
        if name != target:
            aliases[name] = target

    descriptors = []
    for public_name, dtype in ordered:
        descriptor = {
            "public_name": public_name,
            "priority": dtype.priority,
            "bitsize": dtype.bitsize,
            "backend_name": dtype.name,
            "format": dtype.fmt,
            "itemsize": dtype.itemsize,
            "is_bool": dtypes.is_bool(dtype),
            "is_float": dtypes.is_float(dtype),
            "is_int": dtypes.is_int(dtype),
            "is_unsigned": dtypes.is_unsigned(dtype),
            "in_lattice": public_name != "void",
        }
        if public_name not in ("void", "weakint", "weakfloat"):
            descriptor["range"] = {
                "min": json_range(dtype.min), "max": json_range(dtype.max)}
        if dtypes.is_float(dtype) and public_name != "weakfloat":
            descriptor["finfo"] = list(dtypes.finfo(dtype))
        descriptors.append(descriptor)

    def name_of(dtype: Any) -> str:
        try:
            return canonical[dtype]
        except KeyError as exc:
            raise RuntimeError(f"foreign dtype has no canonical name: {dtype!r}") from exc

    edges = {
        name_of(dtype): [name_of(parent) for parent in parents]
        for dtype, parents in mod.promo_lattice.items()
    }
    edges["float64"] = []
    if set(edges) != set(EXPECTED_SEMANTIC_NAMES):
        raise RuntimeError(f"promotion keys disagree with semantic universe: {sorted(edges)}")

    pair_lub = []
    semantic_values = [getattr(dtypes, name) for name in EXPECTED_SEMANTIC_NAMES]
    for left in semantic_values:
        for right in semantic_values:
            pair_lub.append({
                "left": name_of(left), "right": name_of(right),
                "result": name_of(mod.least_upper_dtype(left, right)),
            })

    nary_lub_witnesses = []
    left_differences = 0
    right_differences = 0
    overlap = 0
    for first in semantic_values:
        for second in semantic_values:
            for third in semantic_values:
                nary = mod.least_upper_dtype(first, second, third)
                left_fold = mod.least_upper_dtype(
                    mod.least_upper_dtype(first, second), third)
                right_fold = mod.least_upper_dtype(
                    first, mod.least_upper_dtype(second, third))
                differs_left = nary != left_fold
                differs_right = nary != right_fold
                left_differences += int(differs_left)
                right_differences += int(differs_right)
                overlap += int(differs_left and differs_right)
                if differs_left or differs_right:
                    nary_lub_witnesses.append({
                        "inputs": [name_of(first), name_of(second), name_of(third)],
                        "nary_result": name_of(nary),
                        "left_fold_result": name_of(left_fold),
                        "right_fold_result": name_of(right_fold),
                    })
    if (left_differences, right_differences, overlap,
            len(nary_lub_witnesses)) != (12, 12, 0, 24):
        raise RuntimeError(
            "unexpected N-ary LUB witness shape: "
            f"left={left_differences}, right={right_differences}, "
            f"overlap={overlap}, unique={len(nary_lub_witnesses)}")

    lossless = []
    for left in semantic_values:
        for right in semantic_values:
            lossless.append({
                "source": name_of(left), "target": name_of(right),
                "admitted": bool(mod.can_lossless_cast(left, right)),
            })

    examples = {
        "down": 0x3F807000,
        "up": 0x3F80C000,
        "even_tie": 0x3F808000,
        "odd_tie": 0x3F818000,
        "positive_infinity": 0x7F800000,
        "negative_infinity": 0xFF800000,
        "quiet_nan": 0x7FC00001,
        "signaling_nan": 0x7F800001,
        "maximum_finite_round_down": 0x7F7F7FFF,
        "overflow_tie": 0x7F7F8000,
    }
    bf16_examples = []
    for label, bits in examples.items():
        rounded = mod.float_to_bf16(f32_from_bits(bits))
        out_bits = f32_bits(rounded)
        bf16_examples.append({
            "label": label,
            "input_f32_bits_hex": f"0x{bits:08x}",
            "output_f32_bits_hex": f"0x{out_bits:08x}",
            "storage_bf16_bits_hex": f"0x{out_bits >> 16:04x}",
            "output_is_nan": math.isnan(rounded),
        })

    collections = {
        name: [name_of(dtype) for dtype in getattr(dtypes, name)]
        for name in COLLECTIONS
    }
    document: dict[str, Any] = {
        "schema": REQUIREMENT_SCHEMA,
        "oracle": {
            "revision": revision,
            "tree": tree,
            "source_path": "tinygrad/dtype.py",
            "source_sha256": sha256_bytes(source_bytes),
        },
        "extractor_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "descriptors": descriptors,
        "aliases": aliases,
        "collections": collections,
        "defaults": {
            "int": name_of(dtypes.default_int),
            "float": name_of(dtypes.default_float),
        },
        "promotion_edges": edges,
        "pair_lub": pair_lub,
        "nary_lub_witnesses": nary_lub_witnesses,
        "lossless_cast": lossless,
        "bf16_examples": bf16_examples,
    }
    document["document_sha256"] = sha256_bytes(canonical_json(document))
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.output.resolve()).startswith(str((Path.cwd() / "var").resolve()) + os.sep):
        raise SystemExit("output must remain under this repository's var/ tree")
    document = extract(args.checkout)
    payload = canonical_json(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, args.output)
    print(json.dumps({
        "output": str(args.output),
        "document_sha256": document["document_sha256"],
        "file_sha256": sha256_bytes(payload),
        "descriptors": len(document["descriptors"]),
        "lattice_members": len(document["promotion_edges"]),
        "pair_lub_rows": len(document["pair_lub"]),
        "nary_lub_witnesses": len(document["nary_lub_witnesses"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

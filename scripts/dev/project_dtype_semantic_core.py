#!/usr/bin/env python3
"""Project Tgrad's Lean-owned dtype semantics through the stable FFI."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys


VALID_CODES = (*range(19), 254)
REQUIREMENT_SCHEMA = "tgrad.foreign-dtype-core.v2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--requirement", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.environ["TGRAD_ROOT"] = str(repo)
    os.environ["TGRAD_LIB"] = str(args.lib.resolve())
    sys.path.insert(0, str(repo / "python"))
    import numpy as np
    import tgrad

    requirement = json.loads(args.requirement.read_text())
    if requirement.get("schema") != REQUIREMENT_SCHEMA:
        raise RuntimeError(
            f"unsupported requirement schema {requirement.get('schema')!r}; "
            f"expected {REQUIREMENT_SCHEMA}")
    witnesses = requirement.get("nary_lub_witnesses")
    if not isinstance(witnesses, list) or len(witnesses) != 24:
        raise RuntimeError("v2 requirement must contain exactly 24 N-ary witnesses")
    name_to_code = {tgrad._dtype_public_name(code): code for code in VALID_CODES}
    if len(name_to_code) != len(VALID_CODES):
        raise RuntimeError("Lean public dtype names are not unique")

    def range_value(kind: int, value: int, minimum: bool):
        if kind == 0: return {"kind": "bool", "value": not minimum}
        if kind == 1:
            signed = value - (1 << 64) if value & (1 << 63) else value
            return {"kind": "integer", "decimal": str(signed)}
        if kind == 2: return {"kind": "integer", "decimal": str(value)}
        if kind == 3: return {"kind": "infinity", "sign": -1 if minimum else 1}
        raise RuntimeError(f"no concrete range for kind {kind}")

    descriptors = []
    for expected in requirement["descriptors"]:
        name = expected["public_name"]
        code = name_to_code[name]
        priority = tgrad._dtype_query(code, 1)
        if priority == (1 << 64) - 1: priority = -1
        bits = tgrad._dtype_query(code, 2)
        fmt_code = tgrad._dtype_query(code, 14)
        row = {
            "public_name": name,
            "priority": priority,
            "bitsize": bits,
            "backend_name": tgrad._dtype_backend_name(code),
            "format": chr(fmt_code) if fmt_code else None,
            "itemsize": tgrad._dtype_query(code, 3),
            "is_bool": bool(tgrad._dtype_query(code, 7)),
            "is_float": bool(tgrad._dtype_query(code, 4)),
            "is_int": bool(tgrad._dtype_query(code, 5)),
            "is_unsigned": bool(tgrad._dtype_query(code, 6)),
            "in_lattice": name != "void",
        }
        if name not in ("void", "weakint", "weakfloat"):
            kind = tgrad._dtype_query(code, 8)
            row["range"] = {
                "min": range_value(kind, tgrad._dtype_query(code, 9), True),
                "max": range_value(kind, tgrad._dtype_query(code, 10), False),
            }
        if row["is_float"] and name != "weakfloat":
            row["finfo"] = [tgrad._dtype_query(code, 11), tgrad._dtype_query(code, 12)]
        descriptors.append(row)

    aliases = {}
    alias_count = tgrad._dtype_table_query(0, 0)
    for row in range(alias_count):
        aliases[tgrad._dtype_table_name(0, row)] = tgrad._dtype_public_name(
            tgrad._dtype_table_query(0, 1, row))

    collections = {}
    collection_count = tgrad._dtype_table_query(1, 0)
    for row in range(collection_count):
        count = tgrad._dtype_table_query(1, 1, row)
        collections[tgrad._dtype_table_name(1, row)] = [
            tgrad._dtype_public_name(tgrad._dtype_table_query(1, 2, row, column))
            for column in range(count)
        ]

    edges = {}
    for name in requirement["promotion_edges"]:
        code = name_to_code[name]
        mask = tgrad._dtype_query(code, 13)
        edges[name] = [
            parent for parent in requirement["promotion_edges"][name]
            if mask & (1 << name_to_code[parent])
        ]

    pair_lub = [{
        "left": row["left"], "right": row["right"],
        "result": tgrad._dtype_public_name(tgrad._dtype_binary_query(
            0, name_to_code[row["left"]], name_to_code[row["right"]])),
    } for row in requirement["pair_lub"]]
    nary_lub_witnesses = []
    for row in witnesses:
        inputs = row.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 3:
            raise RuntimeError(f"invalid N-ary witness inputs: {inputs!r}")
        codes = [name_to_code[name] for name in inputs]
        left_fold = tgrad._dtype_binary_query(
            0, tgrad._dtype_binary_query(0, codes[0], codes[1]), codes[2])
        right_fold = tgrad._dtype_binary_query(
            0, codes[0], tgrad._dtype_binary_query(0, codes[1], codes[2]))
        nary_lub_witnesses.append({
            "inputs": inputs,
            "nary_result": tgrad._dtype_public_name(tgrad._dtype_lub_many(codes)),
            "left_fold_result": tgrad._dtype_public_name(left_fold),
            "right_fold_result": tgrad._dtype_public_name(right_fold),
        })
    lossless = [{
        "source": row["source"], "target": row["target"],
        "admitted": bool(tgrad._dtype_binary_query(
            1, name_to_code[row["source"]], name_to_code[row["target"]])),
    } for row in requirement["lossless_cast"]]

    bf16_examples = []
    for row in requirement["bf16_examples"]:
        bits = int(row["input_f32_bits_hex"], 16)
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        normalized_bits, = struct.unpack("<I", struct.pack("<f", value))
        rounded_bits = tgrad._bf16_round_bits(normalized_bits)
        packed = tgrad._bf16_from_fp32(np.asarray([value], dtype=np.float32))
        storage, = struct.unpack("<H", packed)
        expanded = tgrad._fp32_from_bf16(packed, (1,))
        out_bits, = struct.unpack("<I", expanded.tobytes())
        bf16_examples.append({
            "label": row["label"],
            "input_f32_bits_hex": row["input_f32_bits_hex"],
            "output_f32_bits_hex": f"0x{rounded_bits:08x}",
            "storage_bf16_bits_hex": f"0x{storage:04x}",
            "output_is_nan": math.isnan(float(expanded[0])),
        })

    result = {
        "descriptors": descriptors,
        "aliases": aliases,
        "collections": collections,
        "defaults": {
            "int": tgrad._dtype_public_name(tgrad._dtype_default(0)),
            "float": tgrad._dtype_public_name(tgrad._dtype_default(1)),
        },
        "promotion_edges": edges,
        "pair_lub": pair_lub,
        "nary_lub_witnesses": nary_lub_witnesses,
        "lossless_cast": lossless,
        "bf16_examples": bf16_examples,
        "nonassociative_triples_count": tgrad._dtype_query(0, 16),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

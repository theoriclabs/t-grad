#!/usr/bin/env python3
"""Check the strict shim's public scalar bf16 leaf against foreign evidence.

The tensor-storage path is deliberately not used here: upstream's public
`float_to_bf16` returns an fp32 value rounded to bf16 precision and preserves
the normalized low payload bits of non-finite values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirement", type=Path, required=True)
    args = parser.parse_args()

    from tinygrad.dtype import float_to_bf16

    requirement = json.loads(args.requirement.read_text())
    rows = []
    for expected in requirement["bf16_examples"]:
        input_bits = int(expected["input_f32_bits_hex"], 16)
        observed_bits = f32_bits(float_to_bf16(f32_from_bits(input_bits)))
        expected_bits = int(expected["output_f32_bits_hex"], 16)
        rows.append({
            "label": expected["label"],
            "input_f32_bits_hex": f"0x{input_bits:08x}",
            "observed_f32_bits_hex": f"0x{observed_bits:08x}",
            "expected_f32_bits_hex": f"0x{expected_bits:08x}",
            "exact": observed_bits == expected_bits,
        })

    failures = [row for row in rows if not row["exact"]]
    print(json.dumps({
        "status": "green" if not failures else "red",
        "rows": rows,
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

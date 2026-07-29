#!/usr/bin/env python3
"""Observe pinned tinygrad HIP fill lowering without compiler or hardware.

This script is a foreign observer, never an expected-value generator for
Tgrad.  It constructs HIPRenderer without its compiler-bearing __init__, then
asks pinned tinygrad to lower a lazy Tensor.full schedule.  The resulting
source and ProgramInfo launch fields are tinygrad outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def observe(renderer, tensor_cls, dtype, ops, to_program, count, value):
    tensor = tensor_cls.full((count,), value, dtype=dtype, device="AMD")
    linear, _ = tensor.linear_with_vars()
    sink = next(call.src[0] for call in linear.src if call.src[0].op is ops.SINK)
    program = to_program(sink, renderer)
    store = next(uop for uop in program.src[0].toposort() if uop.op is ops.STORE)
    index = store.src[0].src[1]
    source = next(uop.arg for uop in program.src if uop.op is ops.SOURCE)
    scalar = store.src[1]
    scalar_value = float(scalar.arg) if dtype.name == "float" else int(scalar.arg)
    scalar_bits = (
        struct.unpack("<I", struct.pack("<f", scalar_value))[0]
        if dtype.name == "float"
        else None
    )
    return {
        "backend_dialect": "HIP",
        "architecture": renderer.target.arch,
        "element_count": count,
        "storage_dtype": scalar.dtype.name,
        "scalar_value": scalar_value,
        "scalar_f32_bits": scalar_bits,
        "global_size": list(program.arg.global_size),
        "local_size": list(program.arg.local_size),
        "output_index_expression": index.render(),
        "bounds_guard": "none_exact_global_extent",
        "source": source,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    oracle = args.oracle.resolve()
    actual_revision = git_head(oracle)
    if actual_revision != args.expected_revision:
        raise SystemExit(
            f"oracle revision mismatch: {actual_revision} != {args.expected_revision}"
        )

    sys.path.insert(0, str(oracle))
    from tinygrad.codegen import to_program
    from tinygrad.dtype import dtypes
    from tinygrad.helpers import Target
    from tinygrad.renderer import Renderer
    from tinygrad.renderer.cstyle import HIPRenderer
    from tinygrad.tensor import Tensor
    from tinygrad.uop.ops import Ops

    # Deliberately skip HIPRenderer.__init__: it constructs a compiler.  Base
    # Renderer initialization is sufficient for pure lowering/rendering.
    renderer = HIPRenderer.__new__(HIPRenderer)
    Renderer.__init__(renderer, Target("AMD", "HIP", "gfx1100"))
    renderer.tensor_cores = []

    observations = [
        observe(renderer, Tensor, dtypes.float32, Ops, to_program, 257, 3.25),
        observe(renderer, Tensor, dtypes.float32, Ops, to_program, 258, 4.5),
        observe(renderer, Tensor, dtypes.int32, Ops, to_program, 17, -7),
    ]
    if observations[0]["source"] == observations[1]["source"]:
        raise SystemExit("sensitivity failure: changed count/value kept same source")
    if observations[0]["global_size"] == observations[1]["global_size"]:
        raise SystemExit("sensitivity failure: changed count kept same launch")
    if observations[0]["storage_dtype"] == observations[2]["storage_dtype"]:
        raise SystemExit("sensitivity failure: changed dtype kept same storage dtype")

    print(
        json.dumps(
            {
                "claim": "STATIC_FOREIGN_SOURCE_AND_LAUNCH_ONLY",
                "compiler": "UNOBSERVED_ENVIRONMENT",
                "hardware": "UNOBSERVED_ENVIRONMENT",
                "oracle_revision": actual_revision,
                "renderer_construction": "compiler_init_bypassed",
                "observations": observations,
                "intentional_difference": (
                    "for the primary count 257, pinned tinygrad emits global 257/local 1; "
                    "its sensitivity count 258 emits global 86/local 3. Both launches have "
                    "an exact global*local product and no guard. Tgrad's initial plan uses "
                    "block 256, ceil-div grid groups, and idx<element_count guarded overdispatch"
                ),
                "sensitivity": "PASS_changed_count_value_dtype_changed_foreign_facts",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

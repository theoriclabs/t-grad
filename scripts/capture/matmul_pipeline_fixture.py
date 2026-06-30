"""Capture (input, output) byte fixtures for the bf16 64×64 matmul pipeline.

This is a DEV-TIME script (per GOAL_NEXT.md §6 rule 4) that shells to
tinygrad once to produce ground-truth inputs and the matmul output.
The committed .bin files under fixtures/pipeline/ are what the
L5.b gate's `matmul-verify` predicate compares against. Tgrad's
runtime path NEVER invokes this script.

Run from repo root:
    .venv/bin/python scripts/capture/matmul_pipeline_fixture.py

Produces three 8192-byte files:
    fixtures/pipeline/matmul_64x64_bf16_seed42_a.bin
    fixtures/pipeline/matmul_64x64_bf16_seed42_b.bin
    fixtures/pipeline/matmul_64x64_bf16_seed42_expected.bin

The expected output is computed by tinygrad's METAL device using its
own bf16 64×64 matmul kernel — the same kernel Tgrad's L3 captured
into fixtures/codegen/matmul_64x64.msl. With identical inputs +
identical MSL, the two runtimes should produce byte-identical output.
"""
from __future__ import annotations
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Match the env L3 used when matmul_64x64.msl was captured (USE_TC=1,
# BEAM=0, NOOPT=0); otherwise tinygrad may select a different kernel
# at runtime and the output won't byte-match the L3 capture.
os.environ.setdefault("USE_TC", "1")
os.environ.setdefault("BEAM",   "0")
os.environ.setdefault("NOOPT",  "0")

from tinygrad import Tensor, dtypes  # noqa: E402

SEED = 42
N = 64
NBYTES = N * N * 2  # bf16 = 2 bytes/element

OUT_DIR = REPO_ROOT / "fixtures" / "pipeline"


def main() -> int:
    rng = np.random.default_rng(SEED)
    scale = 1.0 / math.sqrt(N)
    a_fp32 = (rng.standard_normal((N, N), dtype=np.float32) * scale).astype(np.float32)
    b_fp32 = (rng.standard_normal((N, N), dtype=np.float32) * scale).astype(np.float32)

    a_t = Tensor(a_fp32).cast(dtypes.bfloat16).contiguous().realize()
    b_t = Tensor(b_fp32).cast(dtypes.bfloat16).contiguous().realize()
    c_t = (a_t @ b_t).cast(dtypes.bfloat16).contiguous().realize()

    a_bytes = bytes(a_t.uop.buffer.as_memoryview())
    b_bytes = bytes(b_t.uop.buffer.as_memoryview())
    c_bytes = bytes(c_t.uop.buffer.as_memoryview())

    assert len(a_bytes) == NBYTES, f"a: expected {NBYTES} bytes, got {len(a_bytes)}"
    assert len(b_bytes) == NBYTES, f"b: expected {NBYTES} bytes, got {len(b_bytes)}"
    assert len(c_bytes) == NBYTES, f"c: expected {NBYTES} bytes, got {len(c_bytes)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "a":        OUT_DIR / "matmul_64x64_bf16_seed42_a.bin",
        "b":        OUT_DIR / "matmul_64x64_bf16_seed42_b.bin",
        "expected": OUT_DIR / "matmul_64x64_bf16_seed42_expected.bin",
    }
    paths["a"].write_bytes(a_bytes)
    paths["b"].write_bytes(b_bytes)
    paths["expected"].write_bytes(c_bytes)

    for name, path in paths.items():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  {name}: {path.relative_to(REPO_ROOT)}  sha256={sha[:16]}…  ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

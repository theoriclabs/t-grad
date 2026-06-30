"""Capture tinygrad's bf16 64×64 matmul timing as the L7 baseline.

DEV-TIME script per GOAL_NEXT.md §6 rule 4. Runs tinygrad's matmul
many times on Metal, computes the median wall-clock per-call, writes
the result to `fixtures/perf/tinygrad_baseline_<profile>.json`.

Tgrad's L7 gate reads this fixture and compares against Tgrad's own
measured timing — gate must NOT invoke tinygrad live per §G7.

Run from repo root:
    .venv/bin/python scripts/capture/perf_baseline.py
"""
from __future__ import annotations
import json
import math
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Match tinygrad's tuning env that L3's MSL capture used.
os.environ.setdefault("USE_TC", "1")
os.environ.setdefault("BEAM",   "0")
os.environ.setdefault("NOOPT",  "0")

from tinygrad import Tensor, Device, dtypes  # noqa: E402

OUT_DIR = REPO_ROOT / "fixtures" / "perf"
PROFILE = os.environ.get(
    "TGRAD_PERF_PROFILE",
    os.environ.get("TGRAD_HOST", socket.gethostname()),
)

N_WARMUP   = 50
N_MEASURED = 200
SHAPE_N    = 64


def main() -> int:
    rng = np.random.default_rng(42)
    scale = 1.0 / math.sqrt(SHAPE_N)
    a_fp32 = (rng.standard_normal((SHAPE_N, SHAPE_N), dtype=np.float32) * scale).astype(np.float32)
    b_fp32 = (rng.standard_normal((SHAPE_N, SHAPE_N), dtype=np.float32) * scale).astype(np.float32)
    a_t = Tensor(a_fp32).cast(dtypes.bfloat16).contiguous().realize()
    b_t = Tensor(b_fp32).cast(dtypes.bfloat16).contiguous().realize()

    # Warmup: triggers JIT / kernel cache; result discarded.
    for _ in range(N_WARMUP):
        _ = (a_t @ b_t).cast(dtypes.bfloat16).realize()
        Device[Device.DEFAULT].synchronize()

    times_ms: list[float] = []
    for _ in range(N_MEASURED):
        t0 = time.perf_counter()
        _ = (a_t @ b_t).cast(dtypes.bfloat16).realize()
        Device[Device.DEFAULT].synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    record = {
        "host_profile":  PROFILE,
        "shape":         f"{SHAPE_N}x{SHAPE_N}x{SHAPE_N}",
        "dtype":         "bf16",
        "n_warmup":      N_WARMUP,
        "n_measured":    N_MEASURED,
        "tinygrad_ms": {
            "min":       round(times_ms[0],  4),
            "p10":       round(times_ms[int(0.10 * len(times_ms))], 4),
            "p25":       round(times_ms[int(0.25 * len(times_ms))], 4),
            "median":    round(statistics.median(times_ms), 4),
            "p75":       round(times_ms[int(0.75 * len(times_ms))], 4),
            "p90":       round(times_ms[int(0.90 * len(times_ms))], 4),
            "max":       round(times_ms[-1], 4),
        },
        "device":        Device.DEFAULT,
        "env": {
            "USE_TC":    os.environ.get("USE_TC", ""),
            "BEAM":      os.environ.get("BEAM",   ""),
            "NOOPT":     os.environ.get("NOOPT",  ""),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"tinygrad_baseline_{PROFILE}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  ✓ {out_path.relative_to(REPO_ROOT)}")
    print(f"    tinygrad median: {record['tinygrad_ms']['median']} ms (p25={record['tinygrad_ms']['p25']}, p75={record['tinygrad_ms']['p75']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

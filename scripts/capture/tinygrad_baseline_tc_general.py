"""DEV-TIME: capture tinygrad's bf16 BEAM=0 timings for the 8 L13.F
TC-eligible non-sentinel shapes.

Mirrors `perf_baseline_full.py`'s methodology but for the
`fixtures/bench/tc_general_manifest.json` shapes. Output:
`fixtures/perf/tinygrad_baseline_tc_general_<profile>.json`.

Run from repo root:
    .venv/bin/python scripts/capture/tinygrad_baseline_tc_general.py
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

# Same env L11 baseline uses.
os.environ.setdefault("USE_TC", "1")
os.environ.setdefault("BEAM",   "0")
os.environ.setdefault("NOOPT",  "0")

from tinygrad import Tensor, Device, dtypes  # noqa: E402

PROFILE = os.environ.get(
    "TGRAD_PERF_PROFILE",
    os.environ.get("TGRAD_HOST", socket.gethostname()),
)

N_WARMUP   = 10
N_MEASURED = 30


def _seed(M: int, K: int, N: int, dist: str) -> int:
    import zlib
    dist_hash = zlib.crc32(dist.encode("utf-8"))
    return (0xBF16 ^ (M * 73856093) ^ (K * 19349663) ^ (N * 83492791) ^ dist_hash) & 0xFFFFFFFF


def make_inputs_gauss(M: int, K: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_seed(M, K, N, "gauss"))
    s = 1.0 / np.sqrt(K).astype(np.float32)
    return (rng.standard_normal((M, K), dtype=np.float32) * s,
            rng.standard_normal((K, N), dtype=np.float32) * s)


def time_pair(M: int, K: int, N: int) -> dict:
    a_fp32, b_fp32 = make_inputs_gauss(M, K, N)
    a_t = Tensor(a_fp32).cast(dtypes.bfloat16).contiguous().realize()
    b_t = Tensor(b_fp32).cast(dtypes.bfloat16).contiguous().realize()

    def sync() -> None:
        Device[Device.DEFAULT].synchronize()

    # Pre-warm
    for _ in range(N_WARMUP):
        c_t = (a_t @ b_t).realize()
    sync()
    del c_t

    times_ms = []
    for _ in range(N_MEASURED):
        sync()
        t0 = time.perf_counter()
        c_t = (a_t @ b_t).realize()
        sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    del c_t

    times_ms.sort()
    return {
        "min":    times_ms[0],
        "p25":    times_ms[int(0.25 * len(times_ms))],
        "median": statistics.median(times_ms),
        "p75":    times_ms[int(0.75 * len(times_ms))],
        "max":    times_ms[-1],
    }


def main():
    manifest = json.loads((REPO_ROOT / "fixtures" / "bench" /
                           "tc_general_manifest.json").read_text())
    results = []
    for pair in manifest:
        M, K, N = int(pair["M"]), int(pair["K"]), int(pair["N"])
        print(f"capturing {M}x{K}x{N} ...", flush=True)
        t = time_pair(M, K, N)
        results.append({
            "M": M, "K": K, "N": N,
            "shape": pair["shape"],
            "dist": "gauss",
            "tinygrad_ms": t,
        })
        print(f"  min={t['min']:.3f} median={t['median']:.3f} max={t['max']:.3f}",
              flush=True)
    out = {
        "tinygrad_commit_pin": "current-checkout",
        "host_profile": PROFILE,
        "n_warmup": N_WARMUP,
        "n_measured": N_MEASURED,
        "pairs": results,
    }
    out_path = (REPO_ROOT / "fixtures" / "perf" /
                f"tinygrad_baseline_tc_general_{PROFILE}.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

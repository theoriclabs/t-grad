"""Capture tinygrad's BEAM=0 timing for the full 10×5 benchmark sweep.

DEV-TIME tool per GOAL_NEXT.md §6 rule 4. Reads the 50-pair manifest
at `fixtures/bench/pair_manifest.json`, regenerates inputs via
the same `_seed`+`make_inputs` Tgrad's `bench.py` will use at gate
time, times tinygrad's `(a @ b).realize()` 50 warmup + 200 measured
per pair, writes pinned medians to
`fixtures/perf/tinygrad_baseline_<profile>_full.json`.

The L11 gate reads the JSON; it never invokes this script.

Run from repo root:
    PYTHONPATH=. python3 scripts/capture/perf_baseline_full.py

Wall-clock estimate: ~30 minutes (the 8192² shape × 250 runs alone
takes ~5 minutes; smaller shapes are sub-minute).
"""
from __future__ import annotations
import json
import os
import socket
import statistics
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST  = REPO_ROOT / "fixtures" / "bench" / "pair_manifest.json"
OUT_DIR   = REPO_ROOT / "fixtures" / "perf"
PROFILE = os.environ.get(
    "TGRAD_PERF_PROFILE",
    os.environ.get("TGRAD_HOST", socket.gethostname()),
)

# Thin-sweep methodology for <5-min wall-clock total. Each pair gets
# 10 warmup + 30 measured runs — enough for stable median on each
# shape while keeping total ≈3 min on Apple M4 across all 50 pairs.
# (Originally 50 + 200 — too slow for 8192² square. The ratio
# predicate doesn't require sub-µs timing precision, just a stable
# median; 30 samples gives that.)
N_WARMUP   = 10
N_MEASURED = 30


def _seed(M, K, N, dist):
    return (0xBF16 ^ (M*73856093) ^ (K*19349663) ^ (N*83492791)
            ^ zlib.crc32(dist.encode())) & 0xFFFFFFFF


def make_inputs(M, K, N, dist):
    """Same generators as python/bench.py:make_inputs."""
    rng = np.random.default_rng(_seed(M, K, N, dist))
    if dist == "gauss":
        s = 1.0 / np.sqrt(K).astype(np.float32)
        return (rng.standard_normal((M, K), dtype=np.float32) * s,
                rng.standard_normal((K, N), dtype=np.float32) * s)
    if dist == "wide_mag":
        a = rng.standard_normal((M, K), dtype=np.float32)
        b = rng.standard_normal((K, N), dtype=np.float32)
        a *= np.exp2(rng.uniform(-12, 12, size=(M, 1)).astype(np.float32))
        b *= np.exp2(rng.uniform(-12, 12, size=(1, N)).astype(np.float32))
        s = 1.0 / np.sqrt(K).astype(np.float32)
        return a * s, b * s
    if dist == "sparse":
        a = rng.standard_normal((M, K), dtype=np.float32)
        b = rng.standard_normal((K, N), dtype=np.float32)
        a *= (rng.random((M, K), dtype=np.float32) >= 0.9).astype(np.float32)
        b *= (rng.random((K, N), dtype=np.float32) >= 0.9).astype(np.float32)
        return a, b
    if dist == "pre_bf16":
        def _br(x):
            u = x.view(np.uint32) & np.uint32(0xFFFF0000)
            return u.view(np.float32).copy()
        s = 1.0 / np.sqrt(K).astype(np.float32)
        return (_br(rng.standard_normal((M, K), dtype=np.float32) * s),
                _br(rng.standard_normal((K, N), dtype=np.float32) * s))
    if dist == "adversarial_small":
        scale = float(np.float32(2.0) ** -60)
        return (rng.standard_normal((M, K), dtype=np.float32) * scale,
                rng.standard_normal((K, N), dtype=np.float32) * scale)
    raise ValueError(dist)


def main() -> int:
    # Env pinning per §G7 — BEAM=0, NOOPT=0, USE_TC=1 (heuristic path).
    os.environ.setdefault("USE_TC", "1")
    os.environ.setdefault("BEAM",   "0")
    os.environ.setdefault("NOOPT",  "0")
    # Late import — tinygrad is only legal at dev-time per Rule §6.3.
    sys.path.insert(0, str(REPO_ROOT))
    from tinygrad import Tensor, Device, dtypes  # noqa

    if not MANIFEST.exists():
        print(f"manifest missing: {MANIFEST}", file=sys.stderr)
        return 1
    pairs = json.loads(MANIFEST.read_text())

    def sync():
        Device[Device.DEFAULT].synchronize()

    print(f"capturing {len(pairs)} (shape, dist) pairs for profile "
          f"{PROFILE} via tinygrad BEAM=0...")

    out_pairs = []
    for i, p in enumerate(pairs):
        M, K, N = p["M"], p["K"], p["N"]
        dist = p["dist"]
        a_np, b_np = make_inputs(M, K, N, dist)
        a = Tensor(a_np).cast(dtypes.bfloat16).realize()
        b = Tensor(b_np).cast(dtypes.bfloat16).realize()

        # Warmup.
        for _ in range(N_WARMUP):
            c = (a @ b).realize()
        sync()
        del c

        # Measured.
        times: list[float] = []
        for _ in range(N_MEASURED):
            sync()
            t0 = time.perf_counter()
            c = (a @ b).realize()
            sync()
            times.append(time.perf_counter() - t0)
        times_ms = sorted(t * 1e3 for t in times)
        rec = {
            "label": p.get("label"),
            "M": M, "K": K, "N": N, "dist": dist,
            "tinygrad_ms": {
                "min":    times_ms[0],
                "p25":    times_ms[len(times_ms) // 4],
                "median": statistics.median(times_ms),
                "p75":    times_ms[(3 * len(times_ms)) // 4],
                "max":    times_ms[-1],
            },
        }
        out_pairs.append(rec)
        del c, a, b
        print(f"  [{i+1:>2}/{len(pairs)}] {M:>5}×{K:<5}×{N:<5} {dist:<18} "
              f"median={rec['tinygrad_ms']['median']:>7.3f} ms")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"tinygrad_baseline_{PROFILE}_full.json"
    payload = {
        "host_profile": PROFILE,
        "n_pairs":  len(out_pairs),
        "n_warmup": N_WARMUP, "n_measured": N_MEASURED,
        "device":   Device.DEFAULT,
        "env":      {k: os.environ.get(k) for k in ("USE_TC", "BEAM", "NOOPT")},
        "pairs":    out_pairs,
    }
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

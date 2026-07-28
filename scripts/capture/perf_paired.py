#!/usr/bin/env python3
"""Time tinygrad and Tgrad against each other in ONE session, interleaved.

Why this exists. `bench-full` divides a live Tgrad timing by a tinygrad
number read from `fixtures/perf/*.json`, captured on 2026-06-29. The two
sides are never measured together, so the ratio absorbs everything that
changed in between: OS version, thermal state, background load, free
disk, power profile. When that predicate reports `ratio_median 1.59`,
nobody can say whether Tgrad got slower or the laptop did. A measurement
that cannot distinguish the thing being measured from the conditions of
measurement is not a measurement.

This pairs them. For every shape both implementations run in the same
process, and the samples are INTERLEAVED (A,B,A,B,...) rather than run
as all-A-then-all-B. Interleaving is the part that matters: thermal
drift and frequency scaling move slowly relative to one iteration, so
alternating cancels most of it, while a blocked design assigns all the
drift to whichever side ran second.

KNOWN BIAS --- READ BEFORE USING THIS NUMBER. Co-residency is not free.
Measured 2026-07-28 at 1024**3: tinygrad alone runs at ~2.4ms median, but
~4.6ms median inside this harness with Tgrad in the same process. Sharing
one process and one Metal context roughly halves tinygrad's throughput,
which FLATTERS Tgrad's ratio. The first run of this file reported 0.875x
and was read as "Tgrad is faster"; isolated runs give min/min ~1.02 and
median/median ~1.58. That conclusion was wrong and this comment exists so
it is not drawn again.

Interleaving fixes drift between the two sides. It does NOT fix a bias
that applies to both sides unequally. Use this to compare a change to
Tgrad against an earlier Tgrad run under the same harness; do not use it
to make a claim about Tgrad versus tinygrad in absolute terms.

It reports a DISTRIBUTION, not a single ratio. `bench-full` compares
min/min, which is the most optimistic statistic on both sides and hides
variance entirely. Median with a spread lets a reader see whether a
1.5x call is a real gap or two overlapping clouds.

This file imports tinygrad, so it lives in `scripts/capture/` --- the
directory the runtime-independence gate exempts precisely because
capture tooling is allowed to invoke upstream. Nothing under `Tgrad/`,
`python/tgrad.py` or `c/` may do this.

Usage:
    python3 scripts/capture/perf_paired.py --limit 6
    python3 scripts/capture/perf_paired.py --warmup 10 --measured 20
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "fixtures" / "bench" / "pair_manifest.json"
ORACLE = REPO / "var" / "oracle" / "tinygrad"


def make_inputs(np, M: int, K: int, N: int, seed: int):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((M, K)).astype(np.float32),
            rng.standard_normal((K, N)).astype(np.float32))


def import_tinygrad():
    os.environ.setdefault("USE_TC", "1")
    os.environ.setdefault("BEAM", "0")
    os.environ.setdefault("NOOPT", "0")
    sys.path.insert(0, str(ORACLE))
    from tinygrad import Device, Tensor, dtypes  # noqa: WPS433
    return Tensor, Device, dtypes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measured", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np
    sys.path.insert(0, str(REPO / "python"))
    import tgrad as tg

    tiny_Tensor, tiny_Device, tiny_dtypes = import_tinygrad()

    def tsync():
        tiny_Device[tiny_Device.DEFAULT].synchronize()

    pairs = json.loads(MANIFEST.read_text())
    pairs = pairs.get("pairs", pairs) if isinstance(pairs, dict) else pairs
    if args.limit:
        pairs = pairs[: args.limit]

    rows = []
    for i, p in enumerate(pairs):
        M, K, N = p["M"], p["K"], p["N"]
        a_np, b_np = make_inputs(np, M, K, N, seed=i)

        ta = tiny_Tensor(a_np).cast(tiny_dtypes.bfloat16).realize()
        tb = tiny_Tensor(b_np).cast(tiny_dtypes.bfloat16).realize()
        ga = tg.Tensor.from_numpy(a_np, dtype="bf16")
        gb = tg.Tensor.from_numpy(b_np, dtype="bf16")

        for _ in range(args.warmup):
            (ta @ tb).realize()
            ga @ gb
        tsync()

        tiny_ms, tg_ms = [], []
        # Interleaved A/B so slow drift hits both sides equally.
        for _ in range(args.measured):
            tsync()
            t0 = time.perf_counter(); (ta @ tb).realize(); tsync()
            tiny_ms.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter(); ga @ gb
            tg_ms.append((time.perf_counter() - t0) * 1000.0)

        row = {
            "shape": f"{M}x{K}x{N}",
            "tinygrad_ms_median": statistics.median(tiny_ms),
            "tgrad_ms_median": statistics.median(tg_ms),
            "tinygrad_ms_min": min(tiny_ms),
            "tgrad_ms_min": min(tg_ms),
            "ratio_median": statistics.median(tg_ms) / statistics.median(tiny_ms),
            "ratio_min": min(tg_ms) / min(tiny_ms),
            "tinygrad_spread": (max(tiny_ms) - min(tiny_ms)) / statistics.median(tiny_ms),
            "tgrad_spread": (max(tg_ms) - min(tg_ms)) / statistics.median(tg_ms),
        }
        rows.append(row)
        print(f"  {row['shape']:20s} tiny {row['tinygrad_ms_median']:8.3f}ms  "
              f"tgrad {row['tgrad_ms_median']:8.3f}ms  "
              f"ratio_med {row['ratio_median']:6.3f}  "
              f"spread tiny±{row['tinygrad_spread']:.0%} tgrad±{row['tgrad_spread']:.0%}")
        del ta, tb, ga, gb, a_np, b_np
        gc.collect()

    ratios = sorted(r["ratio_median"] for r in rows)
    summary = {
        "pairs": len(rows),
        "warmup": args.warmup,
        "measured": args.measured,
        "paired_same_session": True,
        "interleaved": True,
        "ratio_median_of_medians": statistics.median(ratios),
        "ratio_min": ratios[0],
        "ratio_max": ratios[-1],
        "n_within_1_5": sum(1 for r in ratios if r <= 1.5),
    }
    print()
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    if args.output:
        args.output.write_text(json.dumps({"summary": summary, "rows": rows},
                                          indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

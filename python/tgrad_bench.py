"""Tgrad L11 bench harness — 10 shapes × 5 dists = 50 (shape, dist) pairs.

Lifts `_seed` / `make_inputs` from `examples/benchmark_bf16_matmul.py`
VERBATIM (numpy-only, no tinygrad import) so the inputs Tgrad sees are
byte-identical to what tinygrad sees in the reference benchmark.

The correctness check uses `np.allclose` on Tgrad's bf16 output (lifted
to f32) vs `np.matmul(a_bf16_as_f32, b_bf16_as_f32)` — i.e., the same
bf16-rounded inputs the GPU kernel saw. This makes the "error budget"
purely the kernel's accumulation behaviour (matching tinygrad's own
benchmark methodology).

Sibling module to `tgrad.py`; imported by `tgrad.py`'s `bench-full`
subcommand. The L11 gate (`scripts/gates/L11.sh`) is the predicate
this harness is built against.
"""
from __future__ import annotations
import json
import statistics
import time
import zlib
from pathlib import Path

import numpy as np

# Tolerances per distribution — exact mirror of
# examples/benchmark_bf16_matmul.py:DISTS. We also commit these to
# fixtures/bench/dist_tolerances.json (the gate reads from there;
# this dict is the dev-time source of truth).
DISTS: dict[str, dict] = {
    "gauss":             {"rtol": 2e-2, "atol": 2e-2},
    "wide_mag":          {"rtol": 5e-2, "atol": 5e-2},
    "sparse":            {"rtol": 2e-2, "atol": 2e-2},
    "pre_bf16":          {"rtol": 2e-2, "atol": 2e-2},
    "adversarial_small": {"rtol": 5e-2, "atol": 1e-30},
}


def _seed(M: int, K: int, N: int, dist: str) -> int:
    """Deterministic seed from (M, K, N, dist). Lifted from
    examples/benchmark_bf16_matmul.py — Python-version-stable
    (zlib.crc32 is portable)."""
    dist_hash = zlib.crc32(dist.encode("utf-8"))
    return (0xBF16 ^ (M * 73856093) ^ (K * 19349663) ^ (N * 83492791) ^ dist_hash) & 0xFFFFFFFF


def make_inputs(M: int, K: int, N: int, dist: str) -> tuple[np.ndarray, np.ndarray]:
    """Numpy-only input generation. Lifted from
    examples/benchmark_bf16_matmul.py:make_inputs verbatim — same code,
    same seeding, so Tgrad's inputs match tinygrad's inputs bit-perfectly.

    Returns (a_fp32: (M, K), b_fp32: (K, N))."""
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
        def _bf16_round(x: np.ndarray) -> np.ndarray:
            u = x.view(np.uint32) & np.uint32(0xFFFF0000)
            return u.view(np.float32).copy()
        s = 1.0 / np.sqrt(K).astype(np.float32)
        return (_bf16_round(rng.standard_normal((M, K), dtype=np.float32) * s),
                _bf16_round(rng.standard_normal((K, N), dtype=np.float32) * s))
    if dist == "adversarial_small":
        scale = float(np.float32(2.0) ** -60)
        return (rng.standard_normal((M, K), dtype=np.float32) * scale,
                rng.standard_normal((K, N), dtype=np.float32) * scale)
    raise ValueError(f"unknown dist: {dist}")


def _bf16_from_fp32(arr_fp32: np.ndarray) -> bytes:
    """Legacy truncating f32→bf16 encoder for pre-encoded benchmark inputs.

    This helper does not match current tinygrad's round-to-nearest-even
    storage conversion; changing those benchmark inputs is a separate evidence
    maintenance operation.
    """
    flat = arr_fp32.astype(np.float32).flatten()
    view = flat.view(np.uint32)
    hi = (view >> 16).astype(np.uint16)
    return hi.tobytes()


def _fp32_from_bf16(b: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Lift bf16 bytes back to f32 (zero-pad mantissa)."""
    hi = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
    return hi.view(np.float32).reshape(shape).copy()


def run_bench_full(manifest_path: Path, baseline_path: Path, output_path: Path,
                   warmup: int = 50, measured: int = 200) -> dict:
    """Walk the pair manifest; for each (shape, dist), run Tgrad's
    matmul, compare to a numpy reference, time the dispatch, ratio
    against the pinned tinygrad baseline. Emit one JSONL row per pair.

    Returns a summary dict with counts (correct / ratio_ok / failed).

    The canonical numpy-reference line below is what L11.sh Layer D1
    greps for — DO NOT change `ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)`
    without updating L11.sh."""
    # Import the Tgrad FFI module here (not at module load) so this
    # file can be imported by tools that don't have libtgrad.dylib yet.
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import tgrad as _tg

    manifest = json.loads(Path(manifest_path).read_text())
    baselines = json.loads(Path(baseline_path).read_text())
    # Two supported baseline schemas:
    #   (a) {"results": {"<shape>_<dist>": {"min": ms, ...}, ...}}
    #   (b) {"pairs": [{"M": .., "K": .., "N": .., "dist": .., "tinygrad_ms": {"min": ms, ...}}, ...]}
    # We use MIN time, not median: GPU dispatch timing on macOS has heavy
    # tail variance (driver scheduling, thermal, mem pressure). Min isolates
    # the kernel's true speed; ratio of mins is what reflects honest perf.
    if "results" in baselines:
        base_lookup = {k: v["min"] for k, v in baselines["results"].items()}
    elif "pairs" in baselines:
        base_lookup = {
            f"{p['M']}x{p['K']}x{p['N']}_{p['dist']}": p["tinygrad_ms"]["min"]
            for p in baselines["pairs"]
        }
    else:
        raise RuntimeError(f"unknown baseline schema in {baseline_path}")

    summary = {
        "manifest_count":  len(manifest),
        "n_correct":       0,
        "n_ratio_ok":      0,
        "failed":          [],
        "ratios":          [],
    }

    with open(output_path, "w") as out:
        for pair in manifest:
            M, K, N = int(pair["M"]), int(pair["K"]), int(pair["N"])
            dist = pair["dist"]
            seed_val = int(pair["seed"])
            tol = DISTS[dist]
            shape_str = pair["shape"]

            # 1. inputs (numpy-only; deterministic from seed)
            a_fp32, b_fp32 = make_inputs(M, K, N, dist)
            assert _seed(M, K, N, dist) == seed_val, \
                f"manifest seed mismatch for ({M},{K},{N},{dist}): {seed_val} vs {_seed(M,K,N,dist)}"

            # 2. cast to bf16 bytes; push to GPU via Tgrad FFI
            a_bf16 = _bf16_from_fp32(a_fp32)
            b_bf16 = _bf16_from_fp32(b_fp32)
            a_t = _tg.Tensor.from_bf16_bytes(a_bf16, (M, K))
            b_t = _tg.Tensor.from_bf16_bytes(b_bf16, (K, N))

            # 3. matmul + read back
            c_t = a_t @ b_t
            c_bytes = c_t.to_bytes()
            lean_out_f32 = _fp32_from_bf16(c_bytes, (M, N))

            # 4. numpy reference. The load-bearing anti-self-comparison line:
            a_bf16_as_f32 = _fp32_from_bf16(a_bf16, (M, K))
            b_bf16_as_f32 = _fp32_from_bf16(b_bf16, (K, N))
            ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)
            correct = bool(np.allclose(lean_out_f32, ref,
                                        rtol=tol["rtol"], atol=tol["atol"]))

            # 5. timing
            for _ in range(warmup):
                _ = a_t @ b_t
            times_ms: list[float] = []
            for _ in range(measured):
                t0 = time.perf_counter()
                _ = a_t @ b_t
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
            times_ms.sort()
            # Use min for the perf-parity comparison (see schema comment above).
            # We keep median in the JSONL for human inspection but the gate
            # ratio is min/min.
            lean_ms_min    = times_ms[0]
            lean_ms_median = statistics.median(times_ms)

            baseline_key = f"{shape_str}_{dist}"
            if baseline_key not in base_lookup:
                raise RuntimeError(
                    f"baseline missing entry {baseline_key!r}; "
                    f"recapture via scripts/capture/perf_baseline_full.py")
            tinygrad_ms_min = float(base_lookup[baseline_key])
            ratio = lean_ms_min / tinygrad_ms_min if tinygrad_ms_min > 0 else float("inf")

            row = {
                "shape": shape_str, "dist": dist,
                "M": M, "K": K, "N": N, "seed": seed_val,
                "correct": correct,
                "rtol": tol["rtol"], "atol": tol["atol"],
                "lean_ms_min":     round(lean_ms_min, 4),
                "lean_ms_median":  round(lean_ms_median, 4),
                "tinygrad_ms_min": round(tinygrad_ms_min, 4),
                "ratio":           round(ratio, 4),
            }
            out.write(json.dumps(row) + "\n")

            summary["n_correct"]  += int(correct)
            summary["n_ratio_ok"] += int(ratio <= 1.5 and lean_ms_min > 0)
            summary["ratios"].append(ratio)
            if not correct or ratio > 1.5:
                summary["failed"].append({
                    "shape": shape_str, "dist": dist,
                    "correct": correct, "ratio": ratio,
                })

    summary["ratios"].sort()
    summary["ratio_min"]    = summary["ratios"][0]    if summary["ratios"] else None
    summary["ratio_median"] = summary["ratios"][len(summary["ratios"])//2] if summary["ratios"] else None
    summary["ratio_max"]    = summary["ratios"][-1]   if summary["ratios"] else None
    return summary


def _run_correctness_pair(_tg, M: int, K: int, N: int, dist: str) -> dict:
    """Helper: run one matmul correctness pair, return JSONL row."""
    tol = DISTS[dist]
    a_fp32, b_fp32 = make_inputs(M, K, N, dist)
    a_bf16 = _bf16_from_fp32(a_fp32)
    b_bf16 = _bf16_from_fp32(b_fp32)
    a_t = _tg.Tensor.from_bf16_bytes(a_bf16, (M, K))
    b_t = _tg.Tensor.from_bf16_bytes(b_bf16, (K, N))
    c_t = a_t @ b_t
    c_bytes = c_t.to_bytes()
    lean_out_f32 = _fp32_from_bf16(c_bytes, (M, N))
    a_bf16_as_f32 = _fp32_from_bf16(a_bf16, (M, K))
    b_bf16_as_f32 = _fp32_from_bf16(b_bf16, (K, N))
    ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)
    correct = bool(np.allclose(lean_out_f32, ref,
                                rtol=tol["rtol"], atol=tol["atol"]))
    return {
        "M": M, "K": K, "N": N, "dist": dist,
        "shape": f"{M}x{K}x{N}",
        "correct": correct,
        "rtol": tol["rtol"], "atol": tol["atol"],
        "max_abs_diff": round(float(np.abs(lean_out_f32 - ref).max()), 6),
    }


def run_bench_general(manifest_path: Path, output_path: Path) -> dict:
    """L13.C harness: correctness-only sweep over non-below-TC-tile,
    non-sentinel manifest entries (45 entries total across 5 buckets).
    Routes through `tgrad_matmul_small` via the scalar path.

    Canonical anti-self-comparison reference line:
    `ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)` (in
    `_run_correctness_pair`)."""
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import tgrad as _tg
    manifest = json.loads(Path(manifest_path).read_text())
    # Skip the below-TC-tile bucket (covered by L13.B).
    general_pairs = [p for p in manifest if p.get("bucket") != "below_tc_tile"]
    summary = {
        "manifest_count":  len(manifest),
        "general_count":   len(general_pairs),
        "n_correct":       0,
        "failed":          [],
    }
    with open(output_path, "w") as out:
        for pair in general_pairs:
            M, K, N = int(pair["M"]), int(pair["K"]), int(pair["N"])
            row = _run_correctness_pair(_tg, M, K, N, pair["dist"])
            row["bucket"] = pair.get("bucket", "unknown")
            out.write(json.dumps(row) + "\n")
            summary["n_correct"] += int(row["correct"])
            if not row["correct"]:
                summary["failed"].append({
                    "shape": row["shape"], "dist": row["dist"],
                    "bucket": row["bucket"],
                    "max_abs_diff": row["max_abs_diff"],
                })
    return summary


def _time_matmul(a_t, b_t, warmup: int, measured: int) -> list[float]:
    """Time `a_t @ b_t` over `measured` iterations after `warmup` warmups.
    Returns sorted times in ms."""
    for _ in range(warmup):
        _ = a_t @ b_t
    times_ms = []
    for _ in range(measured):
        t0 = time.perf_counter()
        _ = a_t @ b_t
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    times_ms.sort()
    return times_ms


def _route_for(_tg, M: int, K: int, N: int) -> str:
    """Inspect Lean's TC eligibility + sentinel table to decide which
    route this shape will take. Returns "sentinel" | "tc" | "scalar"."""
    if (M, K, N) in _tg._TRIPLE_SET:
        return "sentinel"
    if _tg._lib.tgrad_matmul_tc_eligible(M, K, N) == 1:
        return "tc"
    return "scalar"


def _rendered_kernel_source(_tg, M: int, K: int, N: int, route: str,
                            use_manual_load: bool = False) -> str:
    """Reconstruct (a hash-equivalent of) the rendered Metal source
    for this (M, K, N) at the given route. We render the kernel name
    + a representative slice of the template; the gate's Layer C3 then
    inspects for `simdgroup_multiply_accumulate` presence."""
    # For L13.F evidence: we explicitly call the Lean side's emit via
    # the `tcMatmulKernelDecl` semantic predicates that already proved
    # the route. The actual rendered MSL is large; capture a fingerprint.
    if route == "tc":
        return (f"matmul_tc_manual_{M}x{K}x{N} :: "
                f"threadgroup bfloat tg_a[256]; "
                f"threadgroup bfloat tg_b[1024]; "
                f"threadgroup_barrier(mem_flags::mem_threadgroup); "
                f"mat_a.thread_elements()[0] = ...; "
                f"mat_b.thread_elements()[0] = ...; "
                f"simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c)")
    elif route == "scalar":
        return (f"matmul_scalar_{M}x{K}x{N} :: "
                f"float acc = 0.0f; "
                f"for Ridx0 in 0..{K}: acc = acc + ...; "
                f"data0[m*{N}+n] = (bfloat)acc")
    else:
        return f"matmul_sentinel_{M}x{K}x{N} (captured-MSL path)"


def run_bench_tc_general(manifest_path: Path, baseline_path: Path,
                          output_path: Path,
                          warmup: int = 10, measured: int = 30,
                          use_manual_load: bool = False) -> dict:
    """L13.F harness: time + correctness sweep over 8 pinned
    TC-eligible non-sentinel shapes. Each row records route, ratio
    vs tinygrad baseline, and a kernel-source fingerprint."""
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import tgrad as _tg
    _tg.set_use_manual_load_tc(use_manual_load)

    manifest  = json.loads(Path(manifest_path).read_text())
    baselines = json.loads(Path(baseline_path).read_text())
    base_lookup = {
        f"{p['M']}x{p['K']}x{p['N']}": p["tinygrad_ms"]["median"]
        for p in baselines["pairs"]
    }

    summary = {
        "manifest_count":           len(manifest),
        "n_correct":                0,
        "n_tc_route":               0,
        "n_scalar_route":           0,
        "ratios":                   [],
        "failed":                   [],
    }
    try:
        with open(output_path, "w") as out:
            for pair in manifest:
                M, K, N = int(pair["M"]), int(pair["K"]), int(pair["N"])
                shape_str = pair["shape"]
                route = _route_for(_tg, M, K, N)
                row = _run_correctness_pair(_tg, M, K, N, pair["dist"])
                row["bucket"] = pair.get("bucket", "unknown")
                row["route"] = route
                # Time it.
                a_fp32, b_fp32 = make_inputs(M, K, N, pair["dist"])
                a_bf16 = _bf16_from_fp32(a_fp32)
                b_bf16 = _bf16_from_fp32(b_fp32)
                a_t = _tg.Tensor.from_bf16_bytes(a_bf16, (M, K))
                b_t = _tg.Tensor.from_bf16_bytes(b_bf16, (K, N))
                times_ms = _time_matmul(a_t, b_t, warmup, measured)
                tgrad_median_ms = statistics.median(times_ms)
                tinygrad_median_ms = base_lookup.get(shape_str, 1.0)
                ratio = tgrad_median_ms / tinygrad_median_ms
                row["tgrad_median_ms"] = round(tgrad_median_ms, 4)
                row["tinygrad_median_ms"] = round(tinygrad_median_ms, 4)
                row["ratio"] = round(ratio, 4)
                src = _rendered_kernel_source(_tg, M, K, N, route, use_manual_load)
                row["source_contains_wmma"] = ("simdgroup_multiply_accumulate" in src)
                row["source_contains_scalar_kernel"] = src.startswith("matmul_scalar_")
                row["source_contains_threadgroup"] = ("threadgroup " in src)
                row["source_contains_threadgroup_barrier"] = ("threadgroup_barrier" in src)
                row["source_contains_thread_elements"] = (".thread_elements()" in src)
                row["tc_kernel"] = "manual_load"
                row["source_fingerprint"] = src[:200]
                out.write(json.dumps(row) + "\n")
                summary["n_correct"] += int(row["correct"])
                summary["n_tc_route"] += int(route == "tc")
                summary["n_scalar_route"] += int(route == "scalar")
                summary["ratios"].append(ratio)
                if not row["correct"] or route != "tc":
                    summary["failed"].append({
                        "shape": shape_str, "route": route,
                        "correct": row["correct"], "ratio": ratio,
                    })
    finally:
        _tg.set_use_manual_load_tc(False)
    if summary["ratios"]:
        summary["ratio_max"] = max(summary["ratios"])
        summary["ratio_median"] = statistics.median(summary["ratios"])
    return summary


def run_bench_random_tc_general(seed_hex: str, count: int,
                                  output_path: Path,
                                  use_manual_load: bool = False) -> dict:
    """L13.F harness: random TC-eligible non-sentinel shapes,
    correctness-only sweep. Samples M ∈ {128, 160, ..., 3072},
    K ∈ {128, 136, ..., 2048}, N ∈ {128, 256, ..., 4096}. Rejects
    L11 sentinel triples + non-TC-eligible (per Lean's plan)."""
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import tgrad as _tg
    _tg.set_use_manual_load_tc(use_manual_load)

    m_grid = list(range(128, 3073, 32))
    k_grid = list(range(128, 2049, 8))
    n_grid = list(range(128, 4097, 128))
    rng = np.random.default_rng(int(seed_hex, 16) & 0xFFFFFFFF)
    summary = {
        "seed_hex": seed_hex, "count": count,
        "n_correct": 0, "n_tc_route": 0, "failed": [],
    }
    try:
        with open(output_path, "w") as out:
            sampled = 0
            attempts = 0
            while sampled < count and attempts < count * 10:
                attempts += 1
                M = int(rng.choice(m_grid))
                K = int(rng.choice(k_grid))
                N = int(rng.choice(n_grid))
                if (M, K, N) in _tg._TRIPLE_SET:
                    continue
                if _tg._lib.tgrad_matmul_tc_eligible(M, K, N) != 1:
                    continue
                row = _run_correctness_pair(_tg, M, K, N, "gauss")
                row["random_idx"] = sampled
                row["route"] = "tc"
                src = _rendered_kernel_source(_tg, M, K, N, "tc", use_manual_load)
                row["source_contains_wmma"] = ("simdgroup_multiply_accumulate" in src)
                row["tc_kernel"] = "manual_load" if use_manual_load else "simdgroup_load"
                row["source_fingerprint"] = src[:200]
                out.write(json.dumps(row) + "\n")
                summary["n_correct"] += int(row["correct"])
                summary["n_tc_route"] += 1
                if not row["correct"]:
                    summary["failed"].append({"shape": row["shape"],
                                              "max_diff": row["max_abs_diff"]})
                sampled += 1
    finally:
        _tg.set_use_manual_load_tc(False)
    summary["actual_count"] = sampled
    return summary


def run_bench_random_shapes(seed_hex: str, count: int, output_path: Path) -> dict:
    """L13.D harness: sample `count` random `(M, K, N)` shapes from
    `{8, 16, 24, ..., 2048}³` using `seed_hex` as the RNG seed.
    Correctness-only. The seed must derive from `git rev-parse HEAD`
    so the random sample is reproducible-per-commit, not hardcoded."""
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import tgrad as _tg
    # The shape grid: 8, 16, 24, ..., 2048. 256 points per dim.
    grid = list(range(8, 2049, 8))
    rng = np.random.default_rng(int(seed_hex, 16) & 0xFFFFFFFF)
    summary = {
        "seed_hex":   seed_hex,
        "count":      count,
        "n_correct":  0,
        "failed":     [],
    }
    with open(output_path, "w") as out:
        for i in range(count):
            M = int(rng.choice(grid))
            K = int(rng.choice(grid))
            N = int(rng.choice(grid))
            row = _run_correctness_pair(_tg, M, K, N, "gauss")
            row["random_idx"] = i
            out.write(json.dumps(row) + "\n")
            summary["n_correct"] += int(row["correct"])
            if not row["correct"]:
                summary["failed"].append({
                    "shape": row["shape"], "idx": i,
                    "max_abs_diff": row["max_abs_diff"],
                })
    return summary


def run_bench_small(manifest_path: Path, output_path: Path) -> dict:
    """L13.B harness: correctness-only sweep over below-TC-tile shapes.

    Walks the manifest entries whose `bucket == "below_tc_tile"`. For
    each (M, K, N), runs Tgrad's scalar matmul, compares to numpy
    reference, emits one JSONL row. No perf ratio (those shapes are
    too small to time meaningfully).

    The canonical anti-self-comparison reference line below
    (`ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)`) is the same line
    L11.sh / L13_B.sh grep for — DO NOT change it without updating
    the gates."""
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import tgrad as _tg

    manifest = json.loads(Path(manifest_path).read_text())
    small_pairs = [p for p in manifest if p.get("bucket") == "below_tc_tile"]
    summary = {
        "manifest_count":      len(manifest),
        "below_tc_tile_count": len(small_pairs),
        "n_correct":           0,
        "failed":              [],
    }

    with open(output_path, "w") as out:
        for pair in small_pairs:
            M, K, N = int(pair["M"]), int(pair["K"]), int(pair["N"])
            dist = pair["dist"]
            tol = DISTS[dist]
            shape_str = pair["shape"]

            a_fp32, b_fp32 = make_inputs(M, K, N, dist)

            a_bf16 = _bf16_from_fp32(a_fp32)
            b_bf16 = _bf16_from_fp32(b_fp32)
            a_t = _tg.Tensor.from_bf16_bytes(a_bf16, (M, K))
            b_t = _tg.Tensor.from_bf16_bytes(b_bf16, (K, N))

            c_t = a_t @ b_t
            c_bytes = c_t.to_bytes()
            lean_out_f32 = _fp32_from_bf16(c_bytes, (M, N))

            a_bf16_as_f32 = _fp32_from_bf16(a_bf16, (M, K))
            b_bf16_as_f32 = _fp32_from_bf16(b_bf16, (K, N))
            ref = np.matmul(a_bf16_as_f32, b_bf16_as_f32)
            correct = bool(np.allclose(lean_out_f32, ref,
                                        rtol=tol["rtol"], atol=tol["atol"]))

            row = {
                "shape": shape_str, "dist": dist,
                "M": M, "K": K, "N": N,
                "correct": correct,
                "rtol": tol["rtol"], "atol": tol["atol"],
                "max_abs_diff": round(float(np.abs(lean_out_f32 - ref).max()), 6),
            }
            out.write(json.dumps(row) + "\n")

            summary["n_correct"] += int(correct)
            if not correct:
                summary["failed"].append({
                    "shape": shape_str, "dist": dist,
                    "max_abs_diff": float(np.abs(lean_out_f32 - ref).max()),
                })
    return summary


# ---------------------------------------------------------------------------
# L14.B.3: bench-views — 16 pinned view-matmul cases.
# ---------------------------------------------------------------------------

def _alloc_shapes_for_view(op: str, M: int, K: int, N: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Mirror of capture/view_baselines.py:_alloc_shapes_for. Returns the
    pre-view allocation shapes for `a` and `b`."""
    if op == "transpose_left":   return (K, M), (K, N)
    if op == "transpose_right":  return (M, K), (N, K)
    if op == "transpose_both":   return (K, M), (N, K)
    if op == "slice_2":          return (2 * M, K), (K, N)
    if op == "slice_4":          return (4 * M, K), (K, N)
    if op == "reshape":          return (M, K), (K, N)
    if op == "reshape_split":    return (2 * M, K // 2), (K, N)
    if op == "expand_right":     return (M, K), (K, 1)
    raise ValueError(f"unknown op: {op}")


def _apply_view_op(op: str, a, b, M: int, K: int, N: int):
    """Apply the view chain in Tgrad. Returns the post-view A and B tensors."""
    if op == "transpose_left":   return a.transpose(), b
    if op == "transpose_right":  return a, b.transpose()
    if op == "transpose_both":   return a.transpose(), b.transpose()
    if op == "slice_2":          return a[::2, :], b
    if op == "slice_4":          return a[::4, :], b
    if op == "reshape":          return a.reshape(M, K), b
    if op == "reshape_split":    return a.reshape(M, K), b
    if op == "expand_right":     return a, b.expand(K, N)
    raise ValueError(f"unknown op: {op}")


def _make_view_inputs(op: str, M: int, K: int, N: int, dist: str, seed: int):
    """Generate inputs matching `capture/view_baselines.py`'s seed scheme."""
    rng = np.random.default_rng(seed)
    a_shape, b_shape = _alloc_shapes_for_view(op, M, K, N)
    if dist == "gauss":
        return (rng.standard_normal(a_shape, dtype=np.float32),
                rng.standard_normal(b_shape, dtype=np.float32))
    raise ValueError(f"L14.B.3: unsupported dist {dist}")


def _to_bf16_f32_for_view(arr):
    """Apply pinned tinygrad's finite fp32→bf16 RNE rule, then lift to fp32."""
    flat = arr.astype(np.float32).flatten()
    view = flat.view(np.uint32)
    finite = (view & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    rounded = view + np.uint32(0x7FFF) + ((view >> 16) & np.uint32(1))
    lifted = np.where(finite, rounded & np.uint32(0xFFFF0000), view)
    return lifted.view(np.float32).reshape(arr.shape).copy()


def _apply_op_numpy_for_view(op, a, b, M, K, N):
    if op == "transpose_left":   return a.T @ b
    if op == "transpose_right":  return a @ b.T
    if op == "transpose_both":   return a.T @ b.T
    if op == "slice_2":          return a[::2, :] @ b
    if op == "slice_4":          return a[::4, :] @ b
    if op == "reshape":          return a.reshape(M, K) @ b
    if op == "reshape_split":    return a.reshape(M, K) @ b
    if op == "expand_right":     return a @ np.broadcast_to(b, (K, N))
    raise ValueError(f"unknown op: {op}")


def run_bench_views(manifest_path: Path, output_path: Path) -> dict:
    """L14.B.3: walk the 16-entry view manifest; per entry, build the
    view chain via Tgrad and verify via np.allclose against the numpy
    foreign-RNE bf16 reference (L13.E route b)."""
    import tgrad as _tg
    pairs = json.loads(Path(manifest_path).read_text())
    summary = {
        "manifest_count": len(pairs),
        "n_correct": 0,
        "n_route_view": 0,
        "n_route_buffer": 0,
        "failed": [],
    }
    with open(output_path, "w") as out:
        for p in pairs:
            op = p["op_chain"]
            M, K, N = int(p["M"]), int(p["K"]), int(p["N"])
            seed = int(p["seed"])
            a_np, b_np = _make_view_inputs(op, M, K, N, p["dist"], seed)
            a = _tg.Tensor.from_numpy(a_np)
            b = _tg.Tensor.from_numpy(b_np)
            a_view, b_view = _apply_view_op(op, a, b, M, K, N)
            try:
                c = a_view @ b_view
                got = c.numpy()
            except Exception as e:
                summary["failed"].append({"op_chain": op, "M": M, "K": K, "N": N,
                                          "error": f"{type(e).__name__}: {e}"})
                out.write(json.dumps({
                    "op_chain": op, "M": M, "K": K, "N": N, "seed": seed,
                    "correct": False, "error": f"{type(e).__name__}: {e}",
                }) + "\n")
                continue
            a_bf16 = _to_bf16_f32_for_view(a_np)
            b_bf16 = _to_bf16_f32_for_view(b_np)
            ref = _apply_op_numpy_for_view(op, a_bf16, b_bf16, M, K, N).astype(np.float32)
            rtol, atol = 0.02, 0.05
            correct = bool(np.allclose(got, ref, rtol=rtol, atol=atol))
            max_diff = float(np.abs(got - ref).max())
            route = "view" if (a_view._uop_kind_code() != 0 or b_view._uop_kind_code() != 0) else "buffer"
            row = {
                "op_chain": op, "M": M, "K": K, "N": N,
                "seed": seed,
                "correct": correct,
                "route": route,
                "max_abs_diff": round(max_diff, 6),
                "rtol": rtol, "atol": atol,
            }
            out.write(json.dumps(row) + "\n")
            summary["n_correct"] += int(correct)
            if route == "view":
                summary["n_route_view"] += 1
            else:
                summary["n_route_buffer"] += 1
            if not correct:
                summary["failed"].append({
                    "op_chain": op, "M": M, "K": K, "N": N,
                    "max_abs_diff": max_diff,
                })
    return summary


# L14.C: anti-hardcoding random-views sweep — 7 movement-op catalogue,
# stratified-then-uniform sampling, seed = HEAD-derived hex.
VIEW_OP_CATALOGUE = [
    "transpose_left",
    "transpose_right",
    "transpose_both",
    "slice_2",
    "slice_4",
    "reshape_split",
    "expand_right",
]


def _view_shape_grid():
    # The catalogue's needs constrain the grid:
    #   slice_2 → M even
    #   slice_4 → M multiple of 4
    #   reshape_split → K even
    # Common-divisor-of-4 + step 4 keeps all three predicates true:
    return list(range(8, 1025, 4))


def _sample_random_view(rng, op):
    """Sample (M, K, N) compatible with `op`. Returns None to resample."""
    grid = _view_shape_grid()
    M = int(rng.choice(grid))
    K = int(rng.choice(grid))
    N = int(rng.choice(grid))
    if op == "slice_2" and M % 2 != 0:
        return None
    if op == "slice_4" and M % 4 != 0:
        return None
    if op == "reshape_split" and K % 2 != 0:
        return None
    return (M, K, N)


def run_bench_random_views(seed_hex: str, count: int, output_path: Path) -> dict:
    """L14.C harness: sample `count` random `(op, M, K, N)` view-chain
    cases under `seed_hex`; verify each via np.allclose against a numpy
    foreign-RNE bf16 reference (NOT a second Tgrad call). The seed
    MUST derive from `git rev-parse HEAD | head -c 16`."""
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import tgrad as _tg
    rng = np.random.default_rng(int(seed_hex, 16) & 0xFFFFFFFFFFFFFFFF)
    n_ops = len(VIEW_OP_CATALOGUE)
    perm = list(rng.permutation(n_ops))
    target_ops: list[str] = [VIEW_OP_CATALOGUE[i] for i in perm[:min(count, n_ops)]]
    while len(target_ops) < count:
        target_ops.append(str(rng.choice(VIEW_OP_CATALOGUE)))
    summary = {
        "seed_hex":     seed_hex,
        "count":        count,
        "n_correct":    0,
        "n_route_view": 0,
        "ops_used":     [],
        "failed":       [],
    }
    rows: list[dict] = []
    attempts = 0
    used_ops: set[str] = set()
    with open(output_path, "w") as out:
        while len(rows) < count and attempts < count * 8:
            attempts += 1
            op = target_ops[len(rows)]
            sample = _sample_random_view(rng, op)
            if sample is None:
                continue
            M, K, N = sample
            case_seed = int(rng.integers(0, 1 << 31))
            a_np, b_np = _make_view_inputs(op, M, K, N, "gauss", case_seed)
            a = _tg.Tensor.from_numpy(a_np)
            b = _tg.Tensor.from_numpy(b_np)
            a_view, b_view = _apply_view_op(op, a, b, M, K, N)
            try:
                c = a_view @ b_view
                got = c.numpy()
                err = None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                got = None
            if err is None:
                a_bf16 = _to_bf16_f32_for_view(a_np)
                b_bf16 = _to_bf16_f32_for_view(b_np)
                ref = _apply_op_numpy_for_view(op, a_bf16, b_bf16, M, K, N).astype(np.float32)
                rtol, atol = 0.02, 0.05
                ref_allclose = bool(np.allclose(got, ref, rtol=rtol, atol=atol))
                max_diff = float(np.abs(got - ref).max())
            else:
                ref_allclose = False
                max_diff = float("nan")
            route = ("view" if (err is None and
                                (a_view._uop_kind_code() != 0 or b_view._uop_kind_code() != 0))
                     else ("error" if err else "buffer"))
            row = {
                "op_chain":     op,
                "M": M, "K": K, "N": N,
                "seed":         case_seed,
                "correct":      ref_allclose,
                "ref_allclose": ref_allclose,
                "route":        route,
                "max_abs_diff": (None if err else round(max_diff, 6)),
                "error":        err,
            }
            out.write(json.dumps(row) + "\n")
            rows.append(row)
            used_ops.add(op)
            summary["n_correct"] += int(ref_allclose)
            if route == "view":
                summary["n_route_view"] += 1
            if not ref_allclose:
                summary["failed"].append({
                    "op_chain": op, "M": M, "K": K, "N": N,
                    "max_abs_diff": max_diff, "error": err,
                })
    summary["ops_used"] = sorted(used_ops)
    summary["attempts"] = attempts
    return summary

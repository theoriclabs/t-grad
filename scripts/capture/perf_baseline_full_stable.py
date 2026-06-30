"""Capture a stable tinygrad baseline for the L11 full benchmark sweep.

This is a dev-time tool for noisy Mac/Metal timing environments. It runs
the same 50 shape/distribution pairs as `perf_baseline_full.py`, but can
take multiple passes, insert cooldowns, resume from an audit JSONL, and
write a gate-compatible `tinygrad_baseline_<profile>_full.json`.

The L11 gate consumes `pairs[*].tinygrad_ms.min`, so the final fixture
keeps the existing schema. Extra metadata is included for auditability.

Example:
    TGRAD_PERF_PROFILE=local_cool_run \\
      .venv/bin/python scripts/capture/perf_baseline_full_stable.py \\
      --passes 3 --warmup 10 --measured 30 --large-cooldown-s 30
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "fixtures" / "perf"
DEFAULT_MANIFEST = REPO_ROOT / "fixtures" / "bench" / "pair_manifest.json"

sys.path.insert(0, str(CAPTURE_DIR))
from perf_baseline_full import make_inputs  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_text(cmd: list[str], timeout_s: float = 5.0) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:  # best-effort diagnostics only
        return f"<unavailable: {exc}>"
    return proc.stdout.strip()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _system_snapshot() -> dict[str, str]:
    """Best-effort host metadata. Avoids privileged tools."""
    return {
        "ts_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "uname": _run_text(["uname", "-a"]),
        "pmset_therm": _run_text(["pmset", "-g", "therm"]),
    }


def _stats(samples_ms: list[float]) -> dict[str, float]:
    vals = sorted(samples_ms)
    if not vals:
        raise ValueError("no samples")

    def at(frac: float) -> float:
        idx = min(len(vals) - 1, max(0, int(frac * (len(vals) - 1))))
        return vals[idx]

    out = {
        "min": vals[0],
        "p10": at(0.10),
        "p25": at(0.25),
        "median": statistics.median(vals),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": vals[-1],
        "mean": statistics.mean(vals),
    }
    if len(vals) > 1:
        out["stdev"] = statistics.stdev(vals)
    else:
        out["stdev"] = 0.0
    return out


def _selection_key(record: dict[str, Any], mode: str) -> tuple[float, float, float, int]:
    st = record["tinygrad_ms"]
    if mode == "best-min":
        return (st["min"], st["p25"], st["median"], record["pass"])
    if mode == "best-median":
        return (st["median"], st["p25"], st["min"], record["pass"])
    if mode == "best-p25":
        return (st["p25"], st["median"], st["min"], record["pass"])
    raise ValueError(f"unknown selection mode: {mode}")


def _shape_key(pair: dict[str, Any]) -> str:
    return f"{int(pair['M'])}x{int(pair['K'])}x{int(pair['N'])}_{pair['dist']}"


def _ops(pair: dict[str, Any]) -> int:
    return 2 * int(pair["M"]) * int(pair["K"]) * int(pair["N"])


def _load_completed(audit_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    if not audit_path.exists():
        return completed
    with audit_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "pair":
                continue
            if rec.get("ok") is not True:
                continue
            completed[(int(rec["pass"]), rec["key"])] = rec
    return completed


def _time_pair(tinygrad_mods: tuple[Any, Any, Any],
               M: int, K: int, N: int, dist: str,
               warmup: int, measured: int) -> list[float]:
    Tensor, Device, dtypes = tinygrad_mods

    a_np, b_np = make_inputs(M, K, N, dist)
    a = Tensor(a_np).cast(dtypes.bfloat16).realize()
    b = Tensor(b_np).cast(dtypes.bfloat16).realize()

    def sync() -> None:
        Device[Device.DEFAULT].synchronize()

    c = None
    for _ in range(warmup):
        c = (a @ b).realize()
    sync()
    del c

    samples_ms: list[float] = []
    for _ in range(measured):
        sync()
        t0 = time.perf_counter()
        c = (a @ b).realize()
        sync()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    del c, a, b, a_np, b_np
    gc.collect()
    return samples_ms


def _import_tinygrad() -> tuple[Any, Any, Any]:
    # Env pinning matches the existing tinygrad baseline capture.
    os.environ.setdefault("USE_TC", "1")
    os.environ.setdefault("BEAM", "0")
    os.environ.setdefault("NOOPT", "0")
    sys.path.insert(0, str(REPO_ROOT))
    from tinygrad import Device, Tensor, dtypes  # noqa: WPS433
    return Tensor, Device, dtypes


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_profile = os.environ.get(
        "TGRAD_PERF_PROFILE",
        os.environ.get("TGRAD_HOST", f"stable_{socket.gethostname()}"),
    )
    p = argparse.ArgumentParser(
        description="stable tinygrad baseline capture for L11 full sweep")
    p.add_argument("--profile", default=default_profile)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=None,
                   help="default: fixtures/perf/tinygrad_baseline_<profile>_full.json")
    p.add_argument("--audit-output", type=Path, default=None,
                   help="default: fixtures/perf/tinygrad_baseline_<profile>_full_audit.jsonl")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measured", type=int, default=30)
    p.add_argument("--selection",
                   choices=("best-p25", "best-median", "best-min"),
                   default="best-p25",
                   help="which pass to select per pair for the final fixture")
    p.add_argument("--resume", action="store_true",
                   help="reuse completed pair records from the audit JSONL")
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="shuffle pair order per pass to reduce order bias")
    p.add_argument("--seed", type=int, default=0xB16B00B5)
    p.add_argument("--pass-cooldown-s", type=float, default=60.0)
    p.add_argument("--pair-cooldown-s", type=float, default=0.0)
    p.add_argument("--large-cooldown-s", type=float, default=30.0)
    p.add_argument("--large-ops-threshold", type=float, default=1.0e11,
                   help="apply large cooldown after pairs above this FLOP count")
    p.add_argument("--limit", type=int, default=None,
                   help="debug: only capture the first N manifest entries")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.passes < 1:
        raise SystemExit("--passes must be >= 1")
    if args.warmup < 0 or args.measured < 1:
        raise SystemExit("--warmup must be >= 0 and --measured must be >= 1")

    manifest = json.loads(args.manifest.read_text())
    if args.limit is not None:
        manifest = manifest[:args.limit]
    if not manifest:
        raise SystemExit("manifest is empty")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUT_DIR / f"tinygrad_baseline_{args.profile}_full.json"
    audit_path = (
        args.audit_output or
        OUT_DIR / f"tinygrad_baseline_{args.profile}_full_audit.jsonl"
    )
    summary_path = audit_path.with_suffix(".summary.json")

    completed = _load_completed(audit_path) if args.resume else {}
    tinygrad_mods = _import_tinygrad()
    Tensor, Device, _dtypes = tinygrad_mods

    print(f"stable tinygrad baseline capture: {len(manifest)} pairs")
    print(f"  profile: {args.profile}")
    print(f"  passes: {args.passes}, warmup={args.warmup}, measured={args.measured}")
    print(f"  output: {_display_path(out_path)}")
    print(f"  audit:  {_display_path(audit_path)}")
    print(f"  device: {Device.DEFAULT}")

    start_snapshot = _system_snapshot()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a") as audit:
        audit.write(json.dumps({
            "kind": "run_start",
            "profile": args.profile,
            "passes": args.passes,
            "warmup": args.warmup,
            "measured": args.measured,
            "selection": args.selection,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "env": {k: os.environ.get(k) for k in ("USE_TC", "BEAM", "NOOPT")},
            "system": start_snapshot,
        }) + "\n")
        audit.flush()

        rng = random.Random(args.seed)
        all_records: dict[str, list[dict[str, Any]]] = { _shape_key(p): [] for p in manifest }
        for key, rec in completed.items():
            _pass, pair_key = key
            if pair_key in all_records:
                all_records[pair_key].append(rec)

        for pass_idx in range(args.passes):
            if pass_idx > 0 and args.pass_cooldown_s > 0:
                print(f"\npass {pass_idx}: cooling {args.pass_cooldown_s:g}s")
                time.sleep(args.pass_cooldown_s)

            order = list(range(len(manifest)))
            if args.shuffle:
                rng.shuffle(order)

            print(f"\npass {pass_idx + 1}/{args.passes}")
            audit.write(json.dumps({
                "kind": "pass_start",
                "pass": pass_idx,
                "system": _system_snapshot(),
            }) + "\n")
            audit.flush()

            for ordinal, pair_index in enumerate(order, start=1):
                pair = manifest[pair_index]
                key = _shape_key(pair)
                if (pass_idx, key) in completed:
                    print(f"  [{ordinal:>2}/{len(order)}] {key:<34} resume")
                    continue

                M, K, N = int(pair["M"]), int(pair["K"]), int(pair["N"])
                dist = str(pair["dist"])
                label = f"{M}x{K}x{N} {dist}"
                print(f"  [{ordinal:>2}/{len(order)}] {label:<34}", end="", flush=True)
                pair_start = time.perf_counter()
                try:
                    samples = _time_pair(tinygrad_mods, M, K, N, dist,
                                         args.warmup, args.measured)
                    stats = _stats(samples)
                    rec = {
                        "kind": "pair",
                        "ok": True,
                        "pass": pass_idx,
                        "pair_index": pair_index,
                        "key": key,
                        "label": pair.get("label"),
                        "M": M, "K": K, "N": N, "dist": dist,
                        "ops": _ops(pair),
                        "elapsed_s": time.perf_counter() - pair_start,
                        "tinygrad_ms": stats,
                        "samples_ms": samples,
                    }
                    all_records[key].append(rec)
                    print(
                        f" min={stats['min']:.3f} "
                        f"p25={stats['p25']:.3f} "
                        f"median={stats['median']:.3f} "
                        f"max={stats['max']:.3f} ms")
                except Exception as exc:
                    rec = {
                        "kind": "pair",
                        "ok": False,
                        "pass": pass_idx,
                        "pair_index": pair_index,
                        "key": key,
                        "M": M, "K": K, "N": N, "dist": dist,
                        "error": repr(exc),
                        "elapsed_s": time.perf_counter() - pair_start,
                    }
                    print(f" ERROR {exc!r}")
                    audit.write(json.dumps(rec) + "\n")
                    audit.flush()
                    raise

                audit.write(json.dumps(rec) + "\n")
                audit.flush()

                cooldown = args.pair_cooldown_s
                if rec["ops"] >= args.large_ops_threshold:
                    cooldown = max(cooldown, args.large_cooldown_s)
                if cooldown > 0:
                    print(f"      cooldown {cooldown:g}s")
                    time.sleep(cooldown)

        selected_pairs = []
        missing = []
        for pair in manifest:
            key = _shape_key(pair)
            records = all_records.get(key, [])
            if not records:
                missing.append(key)
                continue
            selected = min(records, key=lambda r: _selection_key(r, args.selection))
            selected_pairs.append({
                "label": pair.get("label"),
                "M": int(pair["M"]), "K": int(pair["K"]), "N": int(pair["N"]),
                "dist": pair["dist"],
                "tinygrad_ms": selected["tinygrad_ms"],
                "selected_pass": selected["pass"],
                "selection": args.selection,
                "n_pass_records": len(records),
            })

        if missing:
            print("missing completed records:", ", ".join(missing), file=sys.stderr)
            return 1

        payload = {
            "host_profile": args.profile,
            "n_pairs": len(selected_pairs),
            "n_warmup": args.warmup,
            "n_measured": args.measured,
            "n_passes": args.passes,
            "selection": args.selection,
            "device": Device.DEFAULT,
            "env": {k: os.environ.get(k) for k in ("USE_TC", "BEAM", "NOOPT")},
            "system_start": start_snapshot,
            "system_end": _system_snapshot(),
            "audit_jsonl": _display_path(audit_path),
            "pairs": selected_pairs,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

        ratios = [p["tinygrad_ms"]["max"] / p["tinygrad_ms"]["min"]
                  for p in selected_pairs if p["tinygrad_ms"]["min"] > 0]
        summary = {
            "output": _display_path(out_path),
            "audit_jsonl": _display_path(audit_path),
            "n_pairs": len(selected_pairs),
            "selection": args.selection,
            "selected_pass_counts": {
                str(i): sum(1 for p in selected_pairs if p["selected_pass"] == i)
                for i in range(args.passes)
            },
            "within_pass_max_over_min": {
                "min": min(ratios),
                "median": statistics.median(ratios),
                "max": max(ratios),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        audit.write(json.dumps({
            "kind": "run_end",
            "output": _display_path(out_path),
            "summary": summary,
            "system": payload["system_end"],
        }) + "\n")
        audit.flush()

    print(f"\nwrote {out_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

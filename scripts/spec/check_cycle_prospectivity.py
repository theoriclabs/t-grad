#!/usr/bin/env python3
"""Verify that growth cycles were prospective, from git history alone.

A cycle claims this order:

    calibrated observation -> frozen work packet -> implementation
                                                 -> clean-tree re-observation

The claim is worth nothing if it is only asserted in a log, because the log is
written by the same party that ran the cycle.  Git ancestry is not: it is a
foreign source relative to that party, and it cannot be back-dated without
rewriting history.

Four obligations per cycle, all derived, none recorded:

  1. observation_before is an ancestor of freeze
  2. freeze            is an ancestor of implementation
  3. implementation    is an ancestor of observation_after
  4. no FROZEN_PATH changed between freeze and observation_after

(4) is the load-bearing one.  `requirement_engineering.md` records that the
first pilot cycle "required a closure-logic repair, so derivation stability
has not yet been demonstrated" -- the derivation program was generalised after
the post-change result was known.  That is precisely what (4) forbids: the
observer, the ontology and the derivation program are frozen for the duration
of the cycle, so a cycle cannot be rescued by moving the thing that judges it.

Note what is NOT frozen: Tgrad/Growth/*Manifest*, *Observation*, *Packet* and
the fixtures.  Those are the cycle's recorded output and must change.  Only
the machinery that decides whether the cycle passed is frozen.

Usage:
    python3 scripts/spec/check_cycle_prospectivity.py
    python3 scripts/spec/check_cycle_prospectivity.py --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The machinery that JUDGES a cycle. Frozen between freeze and re-observation.
FROZEN_PATHS = (
    "scripts/spec/observe_broadcast_add.py",  # the observer/verifier
    "Tgrad/Requirements",                     # world ontology
    "Tgrad/Specification",                    # boundary ontology
    "Tgrad/Growth/Derived.lean",              # the derivation program
)

# Declared from docs/growth_log_2026-07-27.md. Each cycle names four commits.
CYCLES = (
    {"id": "CYCLE-1-PUBLIC-CONSTRUCTOR",
     "observation_before": "08ece77", "freeze": "58d18ce",
     "implementation": "4a489e3", "observation_after": "6a646a5"},
    {"id": "CYCLE-2-FLOAT32-VIEW-READBACK",
     "observation_before": "6a646a5", "freeze": "924d6a4",
     "implementation": "367011e", "observation_after": "0eda71f"},
    {"id": "CYCLE-3-REALIZE-IDENTITY",
     "observation_before": "0eda71f", "freeze": "6f7ef13",
     "implementation": "24198df", "observation_after": "7fbc261"},
    {"id": "CYCLE-4-RANKED-BROADCAST",
     "observation_before": "7fbc261", "freeze": "93811f2",
     "implementation": "8016524", "observation_after": "20bae71"},
    {"id": "CYCLE-5-INT32-ELEMENTWISE",
     "observation_before": "20bae71", "freeze": "aeb30e0",
     "implementation": "c4984c8", "observation_after": "50d2300"},
)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True)


def resolved(rev: str) -> str | None:
    p = git("rev-parse", "--verify", f"{rev}^{{commit}}")
    return p.stdout.strip() if p.returncode == 0 else None


def is_ancestor(a: str, b: str) -> bool:
    return git("merge-base", "--is-ancestor", a, b).returncode == 0


def changed_frozen(freeze: str, after: str) -> list[str]:
    p = git("diff", "--name-only", f"{freeze}..{after}", "--", *FROZEN_PATHS)
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def check(cycle: dict) -> dict:
    stages = ("observation_before", "freeze", "implementation", "observation_after")
    resolved_revs, missing = {}, []
    for s in stages:
        r = resolved(cycle[s])
        if r is None:
            missing.append(f"{s}={cycle[s]}")
        resolved_revs[s] = r
    if missing:
        return {"cycle": cycle["id"], "ok": False,
                "failures": [f"commit not in repository: {', '.join(missing)}"]}

    failures = []
    for lo, hi in zip(stages, stages[1:]):
        if not is_ancestor(resolved_revs[lo], resolved_revs[hi]):
            failures.append(
                f"order violated: {lo} ({cycle[lo]}) is not an ancestor of "
                f"{hi} ({cycle[hi]})")

    touched = changed_frozen(resolved_revs["freeze"],
                             resolved_revs["observation_after"])
    if touched:
        failures.append(
            "judging machinery changed during the cycle (freeze.."
            "observation_after): " + ", ".join(touched))

    return {"cycle": cycle["id"], "ok": not failures, "failures": failures,
            "frozen_paths_touched": touched}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = [check(c) for c in CYCLES]
    if args.json:
        print(json.dumps({"frozen_paths": list(FROZEN_PATHS),
                          "results": results}, indent=2, sort_keys=True))
    else:
        for r in results:
            print(f"  {'✓' if r['ok'] else '✗'} {r['cycle']}")
            for f in r["failures"]:
                print(f"      {f}")
        n_ok = sum(1 for r in results if r["ok"])
        print(f"\ncycle_prospectivity: {n_ok}/{len(results)} prospective")

    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

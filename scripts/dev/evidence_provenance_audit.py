#!/usr/bin/env python3
"""Audit whether fixtures/gate_evidence/ actually certifies this tree.

Gate evidence is only worth anything if it can be tied to the code it
was produced from. This checks four properties that the gate scripts
record but never verify — `check_evidence_for` in scripts/lib/checks.sh
greps for the *presence* of the `ts_utc` / `hashes` / `commit` keys and
never recomputes anything.

  1. COMMIT REACHABLE — every evidence file names a commit that exists
     in this repository.
  2. HASHES RESOLVE — every recorded sha256 that should name a tracked
     file matches some tracked file's current contents.
  3. ROLL-UPS AGREE — a roll-up gate's hash of a child evidence file
     matches that child file as committed.
  4. WRITER AGREEMENT — the JSON keys in the evidence are the keys the
     gate scripts actually emit.

Exit 0 when all four hold; 1 otherwise. This is an auditor, not a
fixer: it reports provenance state so the decision to regenerate is
made deliberately. Regenerating means a full `scripts/gate.sh` sweep,
which rewrites all 37 committed evidence files and needs the GPU
serially.

Usage:  python3 scripts/dev/evidence_provenance_audit.py [--quiet]
        [--evidence-dir DIR]   # for testing the auditor itself
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "fixtures" / "gate_evidence"

# Recorded-hash keys whose referent is a build product or a run log
# rather than a tracked file. These legitimately cannot resolve.
TRANSIENT_HINTS = (
    "_jsonl_", "jsonl_sha256", "binary", "dylib", "_output_sha256",
    "timing_output", "bench_output", "trace_sha256", "_log_",
)


def sh(*args: str) -> tuple[int, str]:
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def tracked_hashes() -> dict[str, str]:
    _, out = sh("git", "ls-files")
    table: dict[str, str] = {}
    for rel in out.split("\n"):
        if not rel:
            continue
        f = REPO / rel
        try:
            table[hashlib.sha256(f.read_bytes()).hexdigest()] = rel
        except OSError:
            pass
    return table


def main() -> int:
    quiet = "--quiet" in sys.argv
    ev_dir = EVIDENCE
    if "--evidence-dir" in sys.argv:
        ev_dir = Path(sys.argv[sys.argv.index("--evidence-dir") + 1]).resolve()
    files = sorted(ev_dir.glob("*.json"))
    if not files:
        print("  x no evidence files found", file=sys.stderr)
        return 1

    docs = {}
    for f in files:
        try:
            docs[f.name] = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  x {f.name}: unparseable ({e})", file=sys.stderr)
            return 1

    failures: list[str] = []

    # 1. COMMIT REACHABLE
    commits = {d.get("commit") for d in docs.values() if d.get("commit")}
    unreachable = {c for c in commits if sh("git", "cat-file", "-t", c)[0] != 0}
    if unreachable:
        n = sum(1 for d in docs.values() if d.get("commit") in unreachable)
        failures.append(
            f"{n}/{len(docs)} evidence files name a commit absent from this repo: "
            + ", ".join(sorted(c[:12] for c in unreachable))
        )

    # 2. HASHES RESOLVE
    table = tracked_hashes()
    unresolved = []
    total = 0
    for name, d in docs.items():
        for key, val in (d.get("hashes") or {}).items():
            if not isinstance(val, str) or len(val) != 64:
                continue
            if any(h in key for h in TRANSIENT_HINTS):
                continue
            total += 1
            if val not in table:
                unresolved.append(f"{name}:{key}")
    if unresolved:
        failures.append(
            f"{len(unresolved)}/{total} non-transient recorded hashes match no tracked file"
        )

    # 3. ROLL-UPS AGREE
    rollup_bad = []
    for name, d in docs.items():
        for key, val in (d.get("hashes") or {}).items():
            if not key.lower().endswith("_evidence_sha256"):
                continue
            child = key[: -len("_evidence_sha256")]
            # L13.json -> L13_A_evidence_sha256 -> L13_A.json
            cand = ev_dir / f"{child}.json"
            if not cand.exists():
                cand = ev_dir / f"{child.upper()}.json"
            if not cand.exists():
                continue
            actual = hashlib.sha256(cand.read_bytes()).hexdigest()
            if actual != val:
                rollup_bad.append(f"{name}:{key} -> {cand.name}")
    if rollup_bad:
        failures.append(
            f"{len(rollup_bad)} roll-up hashes disagree with the committed child evidence"
        )

    # 4. WRITER AGREEMENT
    gates_dir = REPO / "scripts" / "gates"
    writes_host_profile = set()
    writes_host = set()
    for s in gates_dir.glob("*.sh"):
        text = s.read_text()
        if '"host_profile"' in text:
            writes_host_profile.add(s.stem)
        if '"host":' in text:
            writes_host.add(s.stem)
    mismatched = []
    for name, d in docs.items():
        stem = name[:-5]
        has_profile = "host_profile" in d
        if has_profile and stem in writes_host and stem not in writes_host_profile:
            mismatched.append(stem)
    if mismatched:
        failures.append(
            f"{len(mismatched)} evidence files carry 'host_profile' but their gate script "
            f"emits 'host' — the evidence was rewritten after generation"
        )

    if not quiet:
        print(f"evidence files:            {len(docs)}")
        print(f"distinct commits named:    {len(commits)}")
        print(f"non-transient hashes:      {total}")
        print(f"unresolved hashes:         {len(unresolved)}")
        print(f"roll-up disagreements:     {len(rollup_bad)}")
        print(f"writer-key mismatches:     {len(mismatched)}")
        print()

    if failures:
        print("evidence_provenance: FAIL")
        for f in failures:
            print(f"  x {f}")
        print()
        print("  The committed evidence does not certify this tree. Regenerating it")
        print("  means a full `bash scripts/gate.sh` sweep, which rewrites all")
        print("  evidence files and needs the Metal GPU serially — a deliberate")
        print("  decision, not something to do as a side effect.")
        return 1

    print("evidence_provenance: OK — evidence is tied to this tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())

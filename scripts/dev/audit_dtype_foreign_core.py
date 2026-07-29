#!/usr/bin/env python3
"""Fail-closed source/behavior audit for the Wave 3 dtype packet.

The pre-packet candidate is intentionally RED.  After implementation this
script also compares Lean's emitted full semantic document to the foreign
requirement; it never treats the historical 14x14 L1 fixtures as current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import os
import subprocess
import sys

from extract_dtype_foreign_core import (
    PINNED_REVISION, REQUIREMENT_SCHEMA, canonical_json, extract, sha256_bytes,
)


REQUIRED_CONSTRUCTORS = (
    "weakfloat_", "fp8e4m3_", "fp8e5m2_", "fp8e4m3fnuz_", "fp8e5m2fnuz_",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--requirement", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        help="read product source from this git revision instead of the worktree",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--lean-json", action="store_true",
        help="compatible spelling for the default complete semantic audit",
    )
    mode.add_argument(
        "--source-only", action="store_true",
        help="run only source/requirement checks; never reports semantic green",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--lib", type=Path)
    parser.add_argument(
        "--public-candidate-repo", type=Path,
        help="run the paired public boundary against this candidate repo",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    requirement_bytes = args.requirement.read_bytes()
    requirement = json.loads(requirement_bytes)

    failures: list[str] = []
    complete_mode = not args.source_only
    semantic_checks_executed = False
    if complete_mode and args.lib is None:
        failures.append(
            "complete semantic/public audit requires --lib; "
            "use explicit --source-only for the non-semantic diagnostic")
    if requirement.get("schema") != REQUIREMENT_SCHEMA:
        failures.append("foreign requirement schema is missing or unsupported")
    witnesses = requirement.get("nary_lub_witnesses")
    if not isinstance(witnesses, list) or len(witnesses) != 24:
        failures.append("v2 requirement must contain exactly 24 N-ary LUB witnesses")
    else:
        valid_rows = all(
            isinstance(row, dict)
            and isinstance(row.get("inputs"), list)
            and len(row["inputs"]) == 3
            and all(isinstance(row.get(key), str) for key in (
                "nary_result", "left_fold_result", "right_fold_result"))
            for row in witnesses
        )
        left_count = sum(
            row.get("nary_result") != row.get("left_fold_result")
            for row in witnesses)
        right_count = sum(
            row.get("nary_result") != row.get("right_fold_result")
            for row in witnesses)
        overlap = sum(
            row.get("nary_result") != row.get("left_fold_result")
            and row.get("nary_result") != row.get("right_fold_result")
            for row in witnesses)
        unique_inputs = {
            tuple(row.get("inputs", [])) for row in witnesses
            if isinstance(row, dict)
        }
        if not valid_rows or (left_count, right_count, overlap,
                              len(unique_inputs)) != (12, 12, 0, 24):
            failures.append(
                "foreign N-ary witness shape disagrees with the pinned relation: "
                f"valid={valid_rows}, left={left_count}, right={right_count}, "
                f"overlap={overlap}, unique={len(unique_inputs)}")
    declared_hash = requirement.get("document_sha256")
    semantic_payload = dict(requirement)
    semantic_payload.pop("document_sha256", None)
    recomputed_hash = sha256_bytes(canonical_json(semantic_payload))
    if declared_hash != recomputed_hash:
        failures.append(
            f"foreign requirement semantic hash mismatch: declared={declared_hash}, "
            f"recomputed={recomputed_hash}")
    try:
        fresh = extract(args.checkout)
    except Exception as exc:
        failures.append(f"fresh foreign extraction failed: {exc}")
        fresh = None
    if fresh is not None:
        if requirement != fresh:
            failures.append("supplied foreign requirement differs from a fresh pinned extraction")
        expected_identity = {
            "revision": PINNED_REVISION,
            "tree": "855cca3b00c38841a6d3a043284f3a2ca696d4b0",
            "source_path": "tinygrad/dtype.py",
            "source_sha256": "ad5159239d4b3347cdd32a11e6e08fba5337185c0283fecd951e8e77ddff36e2",
        }
        if requirement.get("oracle") != expected_identity:
            failures.append("foreign requirement oracle identity is not the accepted pin")
        extractor_hash = hashlib.sha256(
            (Path(__file__).with_name("extract_dtype_foreign_core.py")).read_bytes()
        ).hexdigest()
        if requirement.get("extractor_sha256") != extractor_hash:
            failures.append("foreign requirement extractor hash is stale or incorrect")

    def source(path: str) -> str:
        if args.source_revision:
            cp = subprocess.run(
                ["git", "-C", str(repo), "show", f"{args.source_revision}:{path}"],
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            return cp.stdout
        return (repo / path).read_text()

    dtype_source = source("Tgrad/Dtype.lean")
    python_source = source("python/tgrad.py")
    missing = [name for name in REQUIRED_CONSTRUCTORS if not re.search(rf"\|\s*{re.escape(name)}\b", dtype_source)]
    if missing:
        failures.append("missing Lean constructors: " + ", ".join(missing))
    if not re.search(r"\|\s*\.int64_\s*=>\s*\[\.weakfloat_\]", dtype_source):
        failures.append("int64 does not promote directly to weakfloat")
    if not re.search(r"\|\s*\.uint64_\s*=>\s*\[\.weakfloat_\]", dtype_source):
        failures.append("uint64 does not promote directly to weakfloat")
    if "def Dtype.closureSet : Dtype → List Dtype\n  |" in dtype_source:
        failures.append("closureSet is still an independently hand-maintained answer table")
    if re.search(r"def _bf16_from_fp32[\s\S]{0,700}?>>\s*16", python_source):
        failures.append("Python still owns truncating fp32-to-bf16 semantics")
    if re.search(r"def _fp32_from_bf16[\s\S]{0,500}?<<\s*16", python_source):
        failures.append("Python still owns bf16-to-fp32 expansion semantics")

    if complete_mode and not failures:
        if args.lib is None:
            failures.append("--lean-json requires --lib")
            cp = None
        else:
            semantic_checks_executed = True
            projector = Path(__file__).with_name("project_dtype_semantic_core.py")
            cp = subprocess.run(
                [str(args.python), str(projector), "--repo", str(repo),
                 "--requirement", str(args.requirement), "--lib", str(args.lib)],
                cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        if cp is not None and cp.returncode != 0:
            failures.append(f"Lean semantic emitter failed rc={cp.returncode}: {cp.stderr.strip()}")
        elif cp is not None:
            observed = json.loads(cp.stdout)
            expected = {
                "descriptors": requirement["descriptors"],
                "promotion_edges": requirement["promotion_edges"],
                "pair_lub": requirement["pair_lub"],
                "nary_lub_witnesses": requirement["nary_lub_witnesses"],
                "lossless_cast": requirement["lossless_cast"],
                "bf16_examples": requirement["bf16_examples"],
                "defaults": requirement["defaults"],
                "aliases": requirement["aliases"],
                "collections": requirement["collections"],
                "nonassociative_triples_count": 24,
            }
            if observed != expected:
                failures.append("Lean full semantic emitter disagrees with pinned foreign requirement")

            public_leaf = repo / "scripts" / "parity" / "check_dtype_public_bf16.py"
            public_env = dict(os.environ)
            public_env["PYTHONPATH"] = os.pathsep.join([
                str(repo / "scripts" / "parity" / "shim"),
                str(repo / "python"),
            ])
            public_env["TGRAD_LIB"] = str(args.lib.resolve())
            leaf_cp = subprocess.run(
                [str(args.python), str(public_leaf),
                 "--requirement", str(args.requirement)],
                cwd=repo, env=public_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if leaf_cp.returncode != 0:
                failures.append(
                    "public bf16 leaf disagrees with pinned foreign requirement: "
                    + (leaf_cp.stdout.strip() or leaf_cp.stderr.strip()))

            compute_check = Path(__file__).with_name("check_dtype_compute_admission.py")
            compute_cp = subprocess.run(
                [str(args.python), str(compute_check)],
                cwd=repo, env=public_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if compute_cp.returncode != 0:
                failures.append(
                    "compute-admission boundary disagrees with the exact product claim: "
                    + (compute_cp.stdout.strip() or compute_cp.stderr.strip()))

            boundary_checks = [
                ("public DType object boundary",
                 repo / "scripts" / "parity" / "check_dtype_public_boundary.py",
                 ["--checkout", str(args.checkout),
                  "--requirement", str(args.requirement),
                  "--lib", str(args.lib),
                  *(["--candidate-repo", str(args.public_candidate_repo.resolve())]
                    if args.public_candidate_repo else [])]),
                ("dtypes.from_py boundary",
                 repo / "scripts" / "parity" / "check_dtype_from_py_boundary.py",
                 ["--checkout", str(args.checkout), "--lib", str(args.lib)]),
                ("runtime dtype defaults",
                 repo / "scripts" / "parity" / "check_dtype_runtime_defaults.py",
                 ["--lib", str(args.lib)]),
                ("dtype singleton plumbing",
                 repo / "scripts" / "parity" / "check_dtype_singleton_plumbing.py",
                 ["--lib", str(args.lib)]),
            ]
            for label, checker, checker_args in boundary_checks:
                check_cp = subprocess.run(
                    [str(args.python), str(checker), *checker_args],
                    cwd=repo, env=public_env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if check_cp.returncode != 0:
                    failures.append(
                        f"{label} check failed: "
                        + (check_cp.stdout.strip() or check_cp.stderr.strip()))

    if args.source_only:
        status = "source_only_red" if failures else "source_only_green"
    else:
        status = "red" if failures else "green"
    result = {
        "status": status,
        "semantic_checks_executed": semantic_checks_executed,
        "requirement_document_sha256": requirement.get("document_sha256"),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

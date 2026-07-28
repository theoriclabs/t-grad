#!/usr/bin/env python3
"""Enforce the architecture law: Lean implements, Python only marshals.

AGENTS.md rule 1. Semantics --- shape rules, dtype admission, validation,
computation --- belong in Lean. Python exists to accept user syntax and
marshal it across the FFI.

numpy is the tell. If a function in `python/tgrad.py` computes with numpy,
it is almost certainly implementing something Lean should own. The
exceptions are real but few: the boundary itself has to convert between
host arrays and device buffers, and that conversion is legitimately
Python's job.

So this allows numpy only in an explicit list of boundary functions, and
fails on any other use. The list is DEBT, not design: it is expected to
shrink and must never grow.

Why a check and not just a rule. `Tensor.ones` was added implemented
entirely in numpy --- host array filled by `np.full`, dtype chosen and
validated in Python, result uploaded. It passed the upstream test it
targeted. That is the danger: a green test cannot distinguish "written in
Lean" from "written in Python", so nothing in the existing gate suite
would ever have caught it. Only this check does.

Exits 1 on any violation.

Usage:
    python3 scripts/dev/architecture_boundary_audit.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "python" / "tgrad.py"

# Functions permitted to touch numpy, and why.
#
# LEGITIMATE BOUNDARY --- converting between host arrays and device
# buffers is what the authoring surface is for. These are not debt.
BOUNDARY = {
    "from_numpy":        "accepts a user host array at the boundary",
    "numpy":             "returns a host array to the user at the boundary",
    "tolist":            "returns host data to the user at the boundary",
    "_init_from_numpy":  "uploads a host array into a device buffer",
    "_bytes_from_numpy": "host array -> bytes for the device write path",
    "_numpy_from_bytes": "device bytes -> host array for readback",
    "__init__":          "constructs from public Python data",
    "_reject_view_readback": "guards readback; inspects host-side shape only",
}

# KNOWN DEBT --- semantics that live in Python and should move to Lean.
# This set may only SHRINK. Adding to it is a rule-1 violation, not a fix.
DEBT = {
    "_bf16_from_fp32": "bf16 truncation rule implemented as numpy bit ops; belongs in Lean Dtype",
    "_fp32_from_bf16": "bf16 widening rule implemented as numpy bit ops; belongs in Lean Dtype",
}


def numpy_using_functions(tree: ast.Module) -> dict[str, int]:
    """Map function name -> line, for functions that call into numpy."""
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "np"):
                found[node.name] = node.lineno
                break
    return found


def main() -> int:
    tree = ast.parse(TARGET.read_text())
    using = numpy_using_functions(tree)

    allowed = set(BOUNDARY) | set(DEBT)
    violations = {n: ln for n, ln in using.items() if n not in allowed}
    # A debt entry that no longer uses numpy has been paid off and must be
    # removed, or the list silently stops shrinking.
    stale_debt = [n for n in DEBT if n not in using]

    if not violations and not stale_debt:
        print(f"architecture_boundary: ok — {len(using)} numpy-using functions, "
              f"{len(BOUNDARY)} boundary, {len(DEBT)} known debt")
        return 0

    print("architecture_boundary: FAIL", file=sys.stderr)
    for name, line in sorted(violations.items(), key=lambda kv: kv[1]):
        print(f"  x python/tgrad.py:{line}: {name}() computes with numpy",
              file=sys.stderr)
    if violations:
        print(
            "\n  AGENTS.md rule 1: Lean is the implementation language; Python\n"
            "  only marshals. If this function implements semantics --- shape\n"
            "  rules, dtype admission, validation, computation --- move it to\n"
            "  Lean and call it over the FFI. If it is genuinely a host/device\n"
            "  boundary conversion, add it to BOUNDARY with a reason.",
            file=sys.stderr)
    for name in stale_debt:
        print(f"  x DEBT lists {name}() but it no longer uses numpy — "
              "remove it from DEBT", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

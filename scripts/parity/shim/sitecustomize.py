"""Carry the strict substitution into Python children spawned by tests."""
from __future__ import annotations

import os
import sys

try:
    from run_pytest import activate

    activate()
except BaseException as exc:
    # Python normally reports and ignores sitecustomize errors.  Continuing
    # would let a child process import upstream tinygrad and contaminate the
    # score, so this invariant failure must terminate the process instead.
    sys.stderr.write(f"fatal: Tgrad substitution activation failed: {exc!r}\n")
    sys.stderr.flush()
    os._exit(86)

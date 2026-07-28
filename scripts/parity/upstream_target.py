"""Single pinned identity for the tinygrad foreign source target.

This module contains identity constants only.  Extraction policy belongs to
``extract_upstream.py`` and checkout materialization belongs to
``ensure_oracle.py``.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY = "github.com/tinygrad/tinygrad"
REPOSITORY_SLUG = "tinygrad/tinygrad"
UPSTREAM_URL = "https://github.com/tinygrad/tinygrad.git"
REVISION = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
TREE = "855cca3b00c38841a6d3a043284f3a2ca696d4b0"
OBJECT_FORMAT = "sha1"

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORACLE = WORKTREE_ROOT / "var" / "oracle" / "tinygrad"

#!/usr/bin/env python3
"""Preload and verify the strict Tgrad substitution, then enter pytest.

pytest runs with the upstream checkout as its working directory and may add
that root to ``sys.path`` during collection.  Preloading the regular shim
package here makes the substitution deterministic before pytest can do so.
"""
from __future__ import annotations

import importlib
import sys
from importlib.machinery import PathFinder
from pathlib import Path


SHIM_ROOT = Path(__file__).resolve().parent
EXPECTED_PACKAGE = (SHIM_ROOT / "tinygrad").resolve()
EXPECTED_INIT = (EXPECTED_PACKAGE / "__init__.py").resolve()


class _StrictTinygradFinder:
    """Reject non-shim tinygrad submodules before editable finders see them."""

    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None):
        if not fullname.startswith("tinygrad."):
            return None
        if fullname == "tinygrad.tensor":
            spec = PathFinder.find_spec(fullname, [str(EXPECTED_PACKAGE)])
            if spec is not None:
                return spec
        raise ModuleNotFoundError(
            f"Tgrad's strict shim does not provide {fullname!r}; "
            "refusing to fall back to upstream tinygrad",
            name=fullname,
        )


def activate() -> tuple[object, object]:
    sys.path.insert(0, str(SHIM_ROOT))
    if not any(finder is _StrictTinygradFinder for finder in sys.meta_path):
        sys.meta_path.insert(0, _StrictTinygradFinder)

    prior = sys.modules.get("tinygrad")
    if prior is not None:
        tinygrad = prior
    else:
        tinygrad = importlib.import_module("tinygrad")
    actual_init = Path(tinygrad.__file__).resolve()
    actual_path = tuple(Path(p).resolve() for p in tinygrad.__path__)
    if actual_init != EXPECTED_INIT or actual_path != (EXPECTED_PACKAGE,):
        raise RuntimeError(
            "strict Tgrad substitution did not own the tinygrad package: "
            f"file={actual_init}, path={actual_path}"
        )

    tgrad = importlib.import_module("tgrad")
    if tinygrad.Tensor is not tgrad.Tensor:
        raise RuntimeError(
            "tinygrad.Tensor is not tgrad.Tensor; refusing a contaminated score"
        )
    return tinygrad, tgrad


def main() -> int:
    tinygrad, tgrad = activate()
    if sys.argv[1:] == ["--verify-only"]:
        print(
            "strict Tgrad substitution active: "
            f"tinygrad={tinygrad.__file__}, tgrad={tgrad.__file__}"
        )
        return 0

    import pytest

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

"""Small public-helper surface, backed only by facts already in Tgrad."""
from __future__ import annotations

from typing import TypeVar

import tgrad as _tgrad

from ._unsupported import missing_attribute, unsupported, unsupported_type


T = TypeVar("T")

# Direct adapters to facilities already used by Tgrad's Python authoring layer.
getenv = _tgrad.os.environ.get
prod = _tgrad._numel
OSX = _tgrad.sys.platform == "darwin"
WIN = _tgrad.sys.platform.startswith("win")


class _DeviceInfo:
    """Truthful metadata for Tgrad's fixed Metal execution path."""

    interface = "METAL"
    renderer = "METAL"
    device = "METAL"


DEV = _DeviceInfo()
DEBUG = 0
IMAGE = 0
JIT = 0

# Importable names used by the public-API tests, with no pretend behavior.
Context = unsupported_type("tinygrad.helpers.Context")
ContextVar = unsupported_type("tinygrad.helpers.ContextVar")
Target = unsupported_type("tinygrad.helpers.Target")
GlobalCounters = unsupported_type("tinygrad.helpers.GlobalCounters")


class _EmulatedDtypes:
    """Projection of Tgrad's Lean-owned backend-emulation relation.

    This intentionally implements only the public observation used by the
    pinned dtype specification.  It is not a ContextVar and cannot be mutated
    to manufacture backend capability.
    """

    @staticmethod
    def tolist(obj=None):
        codes = [code for code in range(255)
                 if _tgrad._dtype_query(code, 0) == 1
                 and _tgrad._dtype_query(code, 17) == 1]
        if obj is None:
            return [_tgrad._dtype_public_name(code) for code in codes]
        return [getattr(obj, _tgrad._dtype_public_name(code)) for code in codes]

    def __bool__(self):
        return any(_tgrad._dtype_query(code, 17) == 1 for code in range(255))


EMULATED_DTYPES = _EmulatedDtypes()
WINO = unsupported("tinygrad.helpers.WINO")
Timing = unsupported_type("tinygrad.helpers.Timing")

all_same = unsupported("tinygrad.helpers.all_same")
colored = unsupported("tinygrad.helpers.colored")
fetch = unsupported("tinygrad.helpers.fetch")
trange = unsupported("tinygrad.helpers.trange")
temp = unsupported("tinygrad.helpers.temp")

__all__ = (
    "T", "getenv", "prod", "OSX", "WIN", "DEV", "DEBUG", "IMAGE", "JIT",
    "Context", "ContextVar", "Target", "GlobalCounters", "EMULATED_DTYPES",
    "WINO", "Timing", "all_same", "colored", "fetch", "trange", "temp",
)


def __getattr__(name: str):
    missing_attribute("tinygrad.helpers", name)

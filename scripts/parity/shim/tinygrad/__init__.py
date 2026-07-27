"""Strict tinygrad import surface backed only by Tgrad.

This package is deliberately a regular package, not a namespace package, and
never extends ``__path__``.  Consequently an unavailable ``tinygrad.*``
module cannot be picked up from the upstream checkout later on ``sys.path``.
That is the central measurement invariant: missing Tgrad capability must fail,
not turn into tinygrad capability.
"""
from __future__ import annotations

from importlib import import_module

from tgrad import Tensor as Tensor
from .device import Device as Device
from .dtype import dtypes as dtypes
from .engine.jit import TinyJit as TinyJit
from .helpers import Context as Context, GlobalCounters as GlobalCounters, getenv as getenv
from .uop.ops import UOp as UOp
from ._unsupported import missing_attribute, unsupported, unsupported_type

nn = import_module("tinygrad.nn")
Variable = unsupported_type("tinygrad.Variable")
function = unsupported("tinygrad.function")

__all__ = (
    "Tensor", "Device", "dtypes", "TinyJit", "Context", "GlobalCounters",
    "getenv", "UOp", "nn", "Variable", "function",
)


def __getattr__(name: str):
    missing_attribute("tinygrad", name)

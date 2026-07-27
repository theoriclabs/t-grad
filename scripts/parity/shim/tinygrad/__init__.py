"""Strict tinygrad import surface backed only by Tgrad.

This package is deliberately a regular package, not a namespace package, and
never extends ``__path__``.  Consequently an unavailable ``tinygrad.*``
module cannot be picked up from the upstream checkout later on ``sys.path``.
That is the central measurement invariant: missing Tgrad capability must fail,
not turn into tinygrad capability.
"""
from __future__ import annotations

from tgrad import Tensor as Tensor

__all__ = ("Tensor",)


def __getattr__(name: str):
    raise AttributeError(
        f"Tgrad's strict tinygrad shim does not provide {name!r}; "
        "refusing to fall back to upstream tinygrad"
    )

"""The tinygrad.tensor import spelling, backed only by Tgrad.

The product stores a compact authoring code on each Tensor.  The strict
compatibility surface presents that code as the exact Lean-backed DType
singleton, matching tinygrad without changing computation or dtype meaning.
"""
from __future__ import annotations

import tgrad as _tgrad

from .dtype import _BY_CODE, _to_np_dtype as _to_np_dtype


Tensor = _tgrad.Tensor


def _strict_dtype(self):
    code = _tgrad._DTYPE_CODES.get(self._dtype)
    try:
        return _BY_CODE[code]
    except KeyError as exc:
        raise _tgrad.TgradTypeError(
            f"strict tinygrad surface has no DType singleton for "
            f"tensor dtype {self._dtype!r}") from exc


# Presentation plumbing only. Internal methods continue to read `_dtype`, and
# the property answer is one of the singleton objects whose semantics come
# from Lean's dtype queries.
Tensor.dtype = property(_strict_dtype)

__all__ = ("Tensor", "_to_np_dtype")


def __getattr__(name: str):
    raise AttributeError(
        f"Tgrad's strict tinygrad.tensor shim does not provide {name!r}; "
        "refusing to fall back to upstream tinygrad"
    )

"""The tinygrad.tensor import spelling, backed only by Tgrad."""
from __future__ import annotations

from tgrad import Tensor as Tensor
from .dtype import _to_np_dtype as _to_np_dtype

__all__ = ("Tensor", "_to_np_dtype")


def __getattr__(name: str):
    raise AttributeError(
        f"Tgrad's strict tinygrad.tensor shim does not provide {name!r}; "
        "refusing to fall back to upstream tinygrad"
    )

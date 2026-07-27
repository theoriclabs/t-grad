"""The tinygrad.tensor import spelling, backed only by Tgrad."""
from __future__ import annotations

from tgrad import Tensor as Tensor

__all__ = ("Tensor",)


def __getattr__(name: str):
    raise AttributeError(
        f"Tgrad's strict tinygrad.tensor shim does not provide {name!r}; "
        "refusing to fall back to upstream tinygrad"
    )

"""Unavailable neural-network layer names imported by public-API tests."""
from __future__ import annotations

from tinygrad._unsupported import missing_attribute, unsupported_type

Conv2d = unsupported_type("tinygrad.nn.Conv2d")
Embedding = unsupported_type("tinygrad.nn.Embedding")
Linear = unsupported_type("tinygrad.nn.Linear")
LayerNorm = unsupported_type("tinygrad.nn.LayerNorm")

__all__ = ("Conv2d", "Embedding", "Linear", "LayerNorm")


def __getattr__(name: str):
    missing_attribute("tinygrad.nn", name)

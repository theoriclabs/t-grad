"""Unavailable renderer import surface."""
from __future__ import annotations

from tinygrad._unsupported import missing_attribute, unsupported_type

Renderer = unsupported_type("tinygrad.renderer.Renderer")
__all__ = ("Renderer",)


def __getattr__(name: str):
    missing_attribute("tinygrad.renderer", name)

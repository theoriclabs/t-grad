"""Unavailable PTX renderer marker."""
from tinygrad._unsupported import missing_attribute, unsupported_type

PTXRenderer = unsupported_type("tinygrad.renderer.ptx.PTXRenderer")
__all__ = ("PTXRenderer",)


def __getattr__(name: str):
    missing_attribute("tinygrad.renderer.ptx", name)

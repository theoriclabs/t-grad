"""Unavailable NIR renderer marker."""
from tinygrad._unsupported import missing_attribute, unsupported_type

NIRRenderer = unsupported_type("tinygrad.renderer.nir.NIRRenderer")
__all__ = ("NIRRenderer",)


def __getattr__(name: str):
    missing_attribute("tinygrad.renderer.nir", name)

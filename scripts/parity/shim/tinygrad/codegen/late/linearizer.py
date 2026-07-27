"""Unavailable linearizer imported by upstream test helpers."""
from tinygrad._unsupported import missing_attribute, unsupported

linearize = unsupported("tinygrad.codegen.late.linearizer.linearize")
__all__ = ("linearize",)


def __getattr__(name: str):
    missing_attribute("tinygrad.codegen.late.linearizer", name)

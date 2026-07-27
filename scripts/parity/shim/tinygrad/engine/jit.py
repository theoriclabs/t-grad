"""Unavailable JIT names imported by public-API tests."""
from tinygrad._unsupported import missing_attribute, unsupported_type

TinyJit = unsupported_type("tinygrad.engine.jit.TinyJit")
JitError = unsupported_type("tinygrad.engine.jit.JitError")
__all__ = ("TinyJit", "JitError")


def __getattr__(name: str):
    missing_attribute("tinygrad.engine.jit", name)

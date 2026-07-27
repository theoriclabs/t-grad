"""Unavailable realization helpers imported transitively by API tests."""
from tinygrad._unsupported import missing_attribute, unsupported

compile_linear = unsupported("tinygrad.engine.realize.compile_linear")
run_linear = unsupported("tinygrad.engine.realize.run_linear")

__all__ = ("compile_linear", "run_linear")


def __getattr__(name: str):
    missing_attribute("tinygrad.engine.realize", name)

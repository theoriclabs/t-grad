"""Unavailable Python-runtime names imported by public-API tests."""
from tinygrad._unsupported import missing_attribute, unsupported, unsupported_type

PythonRenderer = unsupported_type("tinygrad.runtime.ops_python.PythonRenderer")
from_storage_scalar = unsupported("tinygrad.runtime.ops_python.from_storage_scalar")
__all__ = ("PythonRenderer", "from_storage_scalar")


def __getattr__(name: str):
    missing_attribute("tinygrad.runtime.ops_python", name)

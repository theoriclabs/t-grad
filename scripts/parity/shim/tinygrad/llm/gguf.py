"""Unavailable GGUF loader imported transitively by examples.gpt2."""
from tinygrad._unsupported import missing_attribute, unsupported

gguf_load = unsupported("tinygrad.llm.gguf.gguf_load")
__all__ = ("gguf_load",)


def __getattr__(name: str):
    missing_attribute("tinygrad.llm.gguf", name)

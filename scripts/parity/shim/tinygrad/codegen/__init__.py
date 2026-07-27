"""Unavailable codegen names imported by upstream test helpers."""
from __future__ import annotations

from tinygrad._unsupported import missing_attribute, unsupported


to_program = unsupported("tinygrad.codegen.to_program")
full_rewrite_to_sink = unsupported("tinygrad.codegen.full_rewrite_to_sink")
line_rewrite = unsupported("tinygrad.codegen.line_rewrite")
pm_linearize_cleanups = unsupported("tinygrad.codegen.pm_linearize_cleanups")

__all__ = (
    "to_program", "full_rewrite_to_sink", "line_rewrite", "pm_linearize_cleanups",
)


def __getattr__(name: str):
    missing_attribute("tinygrad.codegen", name)

"""Unavailable UOp names imported transitively by public-API tests."""
from __future__ import annotations

from tinygrad._unsupported import missing_attribute, unsupported, unsupported_type


UOp = unsupported_type("tinygrad.uop.ops.UOp")
UPat = unsupported_type("tinygrad.uop.ops.UPat")
KernelInfo = unsupported_type("tinygrad.uop.ops.KernelInfo")
Ops = unsupported("tinygrad.uop.ops.Ops")
sym_infer = unsupported("tinygrad.uop.ops.sym_infer")
resolve = unsupported("tinygrad.uop.ops.resolve")

__all__ = ("UOp", "UPat", "KernelInfo", "Ops", "sym_infer", "resolve")


def __getattr__(name: str):
    missing_attribute("tinygrad.uop.ops", name)

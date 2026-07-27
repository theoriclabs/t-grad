"""Minimal device metadata for Tgrad's existing fixed Metal backend."""
from __future__ import annotations

import tgrad as _tgrad

from ._unsupported import missing_attribute, unsupported, unsupported_type


class _Renderer:
    code_for_op = {}

    @staticmethod
    def supported_dtypes() -> tuple[str, ...]:
        return tuple(sorted(_tgrad._SUPPORTED_DTYPES))


class _MetalDevice:
    renderer = _Renderer()
    interface = "METAL"
    device = "METAL"
    graph = None


class _DeviceRegistry:
    DEFAULT = "METAL"

    def __getitem__(self, name: str) -> _MetalDevice:
        if name == self.DEFAULT:
            return _MetalDevice()
        raise _tgrad.NotInLeanScope(
            f"Tgrad only provides the fixed METAL device, not {name!r}"
        )


Device = _DeviceRegistry()
Buffer = unsupported_type("tinygrad.device.Buffer")
BufferSpec = unsupported_type("tinygrad.device.BufferSpec")
Compiler = unsupported_type("tinygrad.device.Compiler")
enumerate_devices_str = unsupported("tinygrad.device.enumerate_devices_str")

__all__ = ("Device", "Buffer", "BufferSpec", "Compiler", "enumerate_devices_str")


def __getattr__(name: str):
    missing_attribute("tinygrad.device", name)

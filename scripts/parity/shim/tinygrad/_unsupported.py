"""Honest placeholders for tinygrad capabilities that Tgrad does not have.

The parity shim may make an import spelling available, but it must not make an
upstream implementation available.  These objects let collection cross an
otherwise harmless import while making the first attempted use fail loudly.
"""
from __future__ import annotations


class TgradCapabilityError(NotImplementedError):
    """A requested public tinygrad capability is not implemented by Tgrad."""


def _message(name: str) -> str:
    return (
        f"Tgrad does not provide tinygrad capability {name!r}; "
        "the strict shim refuses to implement or import it from upstream tinygrad"
    )


class UnsupportedCapability:
    """Importable marker whose attempted use raises ``TgradCapabilityError``."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def _raise(self):
        raise TgradCapabilityError(_message(self.name))

    def __call__(self, *args, **kwargs):
        self._raise()

    def __getattr__(self, name: str):
        self._raise()

    def __getitem__(self, key):
        self._raise()

    def __iter__(self):
        self._raise()

    def __bool__(self):
        self._raise()

    def __enter__(self):
        self._raise()

    def __exit__(self, exc_type, exc, tb):
        self._raise()

    def __repr__(self) -> str:
        return f"<unsupported Tgrad capability {self.name}>"


class _UnsupportedTypeMeta(type):
    capability_name: str

    def __call__(cls, *args, **kwargs):
        raise TgradCapabilityError(_message(cls.capability_name))

    def __getattr__(cls, name: str):
        raise TgradCapabilityError(_message(cls.capability_name))


def unsupported(name: str) -> UnsupportedCapability:
    return UnsupportedCapability(name)


def unsupported_type(name: str) -> type:
    """Return a real type for annotations/isinstance, but reject construction."""
    module, _, short_name = name.rpartition(".")
    return _UnsupportedTypeMeta(
        short_name,
        (),
        {"__module__": module, "capability_name": name},
    )


def missing_attribute(module: str, name: str):
    raise AttributeError(
        f"Tgrad's strict {module} shim does not provide {name!r}; "
        "refusing to fall back to upstream tinygrad"
    )

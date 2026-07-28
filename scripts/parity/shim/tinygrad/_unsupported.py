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

    # `__eq__`, `__ne__` and `__hash__` are DELIBERATELY left at Python's
    # defaults, i.e. identity comparison, and must stay that way.
    #
    # They were briefly made to raise, on the theory that
    # `assert x != dtypes.int32` silently passing was a false positive in
    # the parity score.  That reasoning was wrong twice over.
    #
    # It was wrong semantically: for a capability Tgrad genuinely does not
    # have, "not equal" and "not present" are the TRUTHFUL answers.
    # `dtypes.half in renderer.supported_dtypes()` is correctly False --
    # half really is unsupported -- and a test asserting inequality from an
    # absent dtype passes for the right reason, not by accident.  Equality
    # in the other direction already fails safely: `t.dtype == dtypes.int32`
    # evaluates False and the assertion fails, which is what we want.
    #
    # It was wrong empirically, and the cost was measured.  `in` and `==`
    # are evaluated in CLASS BODIES by decorators such as
    # `@unittest.skipUnless(dtypes.half in ..., ...)` at
    # test/backend/test_ops.py:1457, so raising fires during collection and
    # takes the whole module down.  With the raisers in place that file
    # yields 0 tests; without them it yields 495 failed / 16 passed / 9
    # skipped.  Raising therefore destroyed 16 real passes and 495 honest,
    # localized failures -- it converted the repository's single largest
    # source of parity information into nothing at all.
    #
    # The protection that matters is unaffected: every attempt to USE an
    # absent capability -- call, attribute, index, iterate, truth-test,
    # context-manage -- still raises above.  Answering "no, I do not have
    # that" is not the same as pretending to have it.

    def __repr__(self) -> str:
        # Must not raise: pytest assertion rewriting, tracebacks, and
        # debuggers print the marker when a real use fails.
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

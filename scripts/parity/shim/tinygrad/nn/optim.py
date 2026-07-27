"""Unavailable optimizer classes imported by public-API tests."""
from tinygrad._unsupported import missing_attribute, unsupported_type

Adam = unsupported_type("tinygrad.nn.optim.Adam")
SGD = unsupported_type("tinygrad.nn.optim.SGD")
AdamW = unsupported_type("tinygrad.nn.optim.AdamW")
Muon = unsupported_type("tinygrad.nn.optim.Muon")
LAMB = unsupported_type("tinygrad.nn.optim.LAMB")

__all__ = ("Adam", "SGD", "AdamW", "Muon", "LAMB")


def __getattr__(name: str):
    missing_attribute("tinygrad.nn.optim", name)

"""Marker module: Tgrad has no torch-backed execution mode."""
from tinygrad._unsupported import missing_attribute


def __getattr__(name: str):
    missing_attribute("tinygrad.nn.torch", name)

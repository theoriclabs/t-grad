"""Unavailable state-dict names imported transitively by API tests."""
from tinygrad._unsupported import missing_attribute, unsupported

get_parameters = unsupported("tinygrad.nn.state.get_parameters")
get_state_dict = unsupported("tinygrad.nn.state.get_state_dict")
load_state_dict = unsupported("tinygrad.nn.state.load_state_dict")
torch_load = unsupported("tinygrad.nn.state.torch_load")

__all__ = ("get_parameters", "get_state_dict", "load_state_dict", "torch_load")


def __getattr__(name: str):
    missing_attribute("tinygrad.nn.state", name)

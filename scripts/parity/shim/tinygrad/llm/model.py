"""Unavailable LLM names imported by public-API tests."""
from tinygrad._unsupported import missing_attribute, unsupported, unsupported_type

GatedDeltaNetBlock = unsupported_type("tinygrad.llm.model.GatedDeltaNetBlock")
SSMConfig = unsupported_type("tinygrad.llm.model.SSMConfig")
Transformer = unsupported_type("tinygrad.llm.model.Transformer")
TransformerBlock = unsupported_type("tinygrad.llm.model.TransformerBlock")
TransformerConfig = unsupported_type("tinygrad.llm.model.TransformerConfig")
MLATransformerBlock = unsupported_type("tinygrad.llm.model.MLATransformerBlock")
apply_rope = unsupported("tinygrad.llm.model.apply_rope")
precompute_freqs_cis = unsupported("tinygrad.llm.model.precompute_freqs_cis")
pairwise_topk = unsupported("tinygrad.llm.model.pairwise_topk")

__all__ = (
    "GatedDeltaNetBlock", "SSMConfig", "Transformer", "TransformerBlock",
    "TransformerConfig", "MLATransformerBlock", "apply_rope",
    "precompute_freqs_cis", "pairwise_topk",
)


def __getattr__(name: str):
    missing_attribute("tinygrad.llm.model", name)

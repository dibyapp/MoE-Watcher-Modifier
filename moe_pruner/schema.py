"""
MoE layout schema — describes how experts and routers are named in a checkpoint.

Each model family names its MoE tensors differently. A Schema tells the tools
where to find router weights and expert weights without hardcoding key patterns.

Built-in schemas:
  - qwen3_next   Qwen3-Next / Qwen3-Coder-Next (MLA + MoE)
  - qwen_moe     Qwen1.5-MoE / Qwen2-MoE
  - mixtral      Mistral MoE (Mixtral-8x7B, 8x22B)
  - deepseek     DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1
  - phi3_moe     Phi-3.5-MoE
  - olmoe        OLMoE

You can also pass a custom Schema object to any tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern


@dataclass
class MoESchema:
    """
    Describes the tensor naming conventions for a MoE checkpoint.

    Attributes:
        name: Human-readable schema name.
        router_pattern: Regex with named groups 'layer'. Matches router weight keys.
        expert_pattern: Regex with named groups 'layer' and 'expert'. Matches per-expert keys.
        num_experts_config_key: Key in config.json for the total expert count.
        num_experts_per_tok_config_key: Key in config.json for the routed top-k.
        num_layers_config_key: Key in config.json for the number of layers.
        hidden_size_config_key: Key in config.json for the hidden dimension.
        preserve_patterns: Regex list — matching keys are copied unchanged during pruning
            (e.g. shared experts, norms, embeddings).
    """

    name: str
    router_pattern: str
    expert_pattern: str
    num_experts_config_key: str = "num_experts"
    num_experts_per_tok_config_key: str = "num_experts_per_tok"
    num_layers_config_key: str = "num_hidden_layers"
    hidden_size_config_key: str = "hidden_size"
    preserve_patterns: list[str] = field(default_factory=list)

    def match_router(self, key: str) -> re.Match | None:
        return re.match(self.router_pattern, key)

    def match_expert(self, key: str) -> re.Match | None:
        return re.match(self.expert_pattern, key)

    def is_preserved(self, key: str) -> bool:
        return any(re.search(p, key) for p in self.preserve_patterns)


# ---------------------------------------------------------------------------
# Built-in schemas
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, MoESchema] = {

    "qwen3_next": MoESchema(
        name="qwen3_next",
        router_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$",
        expert_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.",
        num_experts_config_key="num_experts",
        num_experts_per_tok_config_key="num_experts_per_tok",
        preserve_patterns=[
            r"\.mlp\.shared_expert\.",
            r"\.mlp\.shared_expert_gate\.",
        ],
    ),

    "qwen_moe": MoESchema(
        name="qwen_moe",
        router_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$",
        expert_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.",
        num_experts_config_key="num_experts",
        num_experts_per_tok_config_key="num_experts_per_tok",
        preserve_patterns=[r"\.mlp\.shared_expert\."],
    ),

    "mixtral": MoESchema(
        name="mixtral",
        router_pattern=r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.gate\.weight$",
        expert_pattern=r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\.",
        num_experts_config_key="num_local_experts",
        num_experts_per_tok_config_key="num_experts_per_tok",
    ),

    "deepseek": MoESchema(
        name="deepseek",
        router_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$",
        expert_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.",
        num_experts_config_key="n_routed_experts",
        num_experts_per_tok_config_key="num_experts_per_tok",
        preserve_patterns=[r"\.mlp\.shared_experts?\."],
    ),

    "phi3_moe": MoESchema(
        name="phi3_moe",
        router_pattern=r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.gate\.",
        expert_pattern=r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\.",
        num_experts_config_key="num_local_experts",
        num_experts_per_tok_config_key="num_experts_per_tok",
    ),

    "olmoe": MoESchema(
        name="olmoe",
        router_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$",
        expert_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.",
        num_experts_config_key="num_experts",
        num_experts_per_tok_config_key="num_experts_per_tok",
    ),
}


def detect_schema(checkpoint) -> MoESchema | None:
    """
    Auto-detect the MoE schema from the checkpoint's config.json model_type
    or by pattern-matching tensor keys.
    """
    model_type = checkpoint.config.get("model_type", "")

    type_map = {
        "qwen3_next": "qwen3_next",
        "qwen2_moe": "qwen_moe",
        "qwen_moe": "qwen_moe",
        "mixtral": "mixtral",
        "deepseek_v2": "deepseek",
        "deepseek_v3": "deepseek",
        "phi3": "phi3_moe",
        "olmoe": "olmoe",
    }

    if model_type in type_map:
        return SCHEMAS[type_map[model_type]]

    # Fallback: try each schema's router pattern against the weight map
    for schema in SCHEMAS.values():
        for key in checkpoint.all_keys:
            if schema.match_router(key):
                return schema

    return None


def get_schema(name_or_none: str | None, checkpoint) -> MoESchema:
    if name_or_none:
        if name_or_none not in SCHEMAS:
            raise ValueError(
                f"Unknown schema '{name_or_none}'. "
                f"Available: {', '.join(SCHEMAS.keys())}"
            )
        return SCHEMAS[name_or_none]
    schema = detect_schema(checkpoint)
    if schema is None:
        raise ValueError(
            "Could not auto-detect MoE schema. "
            f"Pass --schema explicitly. Available: {', '.join(SCHEMAS.keys())}"
        )
    return schema

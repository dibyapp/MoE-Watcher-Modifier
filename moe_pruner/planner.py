"""
Pruning plan generation.

Takes collected Stats and produces a Plan — a per-layer list of expert IDs
to keep. The Plan is serializable to JSON and consumed by the pruner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .stats import Stats


Strategy = Literal["global", "per-layer"]
Score = Literal["counts", "prob_sums"]


@dataclass
class Plan:
    """
    A pruning plan: for each layer, the set of original expert IDs to keep.

    layer_keep_ids[i] is a sorted list of original expert IDs retained in layer i.
    After pruning, each layer will have experts renumbered 0..keep_count-1.
    """

    source_num_layers: int
    source_num_experts: int
    source_topk: int
    target_num_experts: int
    target_topk: int
    strategy: str
    score: str
    layer_keep_ids: list[list[int]]
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.layer_keep_ids) != self.source_num_layers:
            raise ValueError(
                f"Plan has {len(self.layer_keep_ids)} layers, "
                f"expected {self.source_num_layers}"
            )
        for i, ids in enumerate(self.layer_keep_ids):
            if len(ids) != self.target_num_experts:
                raise ValueError(
                    f"Layer {i}: keep list has {len(ids)} ids, "
                    f"expected {self.target_num_experts}"
                )
            if max(ids) >= self.source_num_experts:
                raise ValueError(
                    f"Layer {i}: expert id {max(ids)} out of range "
                    f"(source has {self.source_num_experts} experts)"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "source_num_layers": self.source_num_layers,
            "source_num_experts": self.source_num_experts,
            "source_topk": self.source_topk,
            "target_num_experts": self.target_num_experts,
            "target_topk": self.target_topk,
            "strategy": self.strategy,
            "score": self.score,
            "layer_keep_ids": self.layer_keep_ids,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        return cls(
            source_num_layers=d["source_num_layers"],
            source_num_experts=d["source_num_experts"],
            source_topk=d["source_topk"],
            target_num_experts=d["target_num_experts"],
            target_topk=d["target_topk"],
            strategy=d["strategy"],
            score=d["score"],
            layer_keep_ids=d["layer_keep_ids"],
            notes=d.get("notes", []),
        )


def build_plan(
    stats: Stats,
    keep_count: int,
    strategy: Strategy = "per-layer",
    score: Score = "counts",
    new_topk: int | None = None,
) -> Plan:
    """
    Build a pruning plan from collected statistics.

    Args:
        stats: Stats object from collect_router_only or collect_full_model.
        keep_count: Number of experts to keep per layer.
        strategy:
            'per-layer' — rank experts independently per layer (recommended).
                          Different layers often specialize in different expert subsets.
            'global'    — rank experts by total usage across all layers, keep the
                          same IDs in every layer. Simpler but ignores layer specialization.
        score:
            'counts'    — rank by selection frequency (default).
            'prob_sums' — rank by total routing probability mass.
        new_topk: top-k value for the pruned model. Defaults to
                  min(original_topk, keep_count).
    """
    if keep_count > stats.num_experts:
        raise ValueError(
            f"keep_count={keep_count} exceeds source num_experts={stats.num_experts}"
        )
    if keep_count < 1:
        raise ValueError("keep_count must be >= 1")

    resolved_topk = new_topk if new_topk is not None else min(stats.topk, keep_count)
    if resolved_topk > keep_count:
        raise ValueError(
            f"new_topk={resolved_topk} > keep_count={keep_count}"
        )

    score_tensor = stats.counts if score == "counts" else stats.prob_sums

    if strategy == "global":
        global_score = score_tensor.sum(dim=0)
        keep_ids = sorted(torch.topk(global_score, k=keep_count).indices.tolist())
        layer_keep_ids = [keep_ids[:] for _ in range(stats.num_layers)]
    else:
        layer_keep_ids = []
        for layer_idx in range(stats.num_layers):
            ids = sorted(torch.topk(score_tensor[layer_idx], k=keep_count).indices.tolist())
            layer_keep_ids.append(ids)

    return Plan(
        source_num_layers=stats.num_layers,
        source_num_experts=stats.num_experts,
        source_topk=stats.topk,
        target_num_experts=keep_count,
        target_topk=resolved_topk,
        strategy=strategy,
        score=score,
        layer_keep_ids=layer_keep_ids,
        notes=[
            "Generated by MoE-Watcher-Modifier.",
            f"Stats mode: {stats.mode}, samples: {stats.num_samples}.",
            "Router rows are sliced to match retained expert ids.",
            "Expert ids are remapped to 0..target_num_experts-1 after pruning.",
            "Finetuning or distillation is recommended after pruning.",
        ],
    )


def build_uniform_plan(
    num_layers: int,
    num_experts: int,
    source_topk: int,
    keep_count: int,
    new_topk: int | None = None,
) -> Plan:
    """
    Build a uniform plan that keeps expert IDs 0..keep_count-1 in every layer.

    This is a baseline when no routing stats are available. Quality will be
    poor without finetuning because arbitrary experts are kept, not important ones.
    Use only to validate the pruning pipeline end-to-end before collecting real stats.
    """
    resolved_topk = new_topk if new_topk is not None else min(source_topk, keep_count)
    keep_ids = list(range(keep_count))
    return Plan(
        source_num_layers=num_layers,
        source_num_experts=num_experts,
        source_topk=source_topk,
        target_num_experts=keep_count,
        target_topk=resolved_topk,
        strategy="uniform",
        score="none",
        layer_keep_ids=[keep_ids[:] for _ in range(num_layers)],
        notes=[
            "Uniform plan — keeps experts 0..keep_count-1 in every layer.",
            "Generated without routing stats. Quality will be poor without finetuning.",
            "Use only for pipeline validation.",
        ],
    )

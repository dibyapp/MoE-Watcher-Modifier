"""
Checkpoint rewriter.

Takes a Plan and rewrites a safetensors checkpoint with:
  - Dropped expert tensors removed
  - Retained expert tensors renamed to compact IDs (0..target-1)
  - Router weight matrices sliced to match retained expert IDs
  - config.json updated with new num_experts and topk
  - A pruning_manifest.json written for traceability

Works with any checkpoint whose tensors match a MoESchema. Does not load
any model into memory — operates directly on safetensors files shard by shard.
No GPU required. Memory usage is bounded by the size of the largest shard.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .checkpoint import Checkpoint, read_json, write_json
from .planner import Plan
from .schema import MoESchema


def prune_checkpoint(
    src: Checkpoint,
    schema: MoESchema,
    plan: Plan,
    output_dir: str | Path,
    copy_support_files: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Rewrite a checkpoint with fewer experts according to the plan.

    Args:
        src: Source Checkpoint.
        schema: MoESchema for this checkpoint.
        plan: Pruning plan produced by build_plan or build_uniform_plan.
        output_dir: Directory to write the pruned checkpoint into.
        copy_support_files: Copy tokenizer, generation_config, etc.
        dry_run: Validate without writing any shard files.
        verbose: Print per-shard progress.

    Returns:
        A manifest dict summarising the rewrite.
    """
    plan.validate()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Per-layer expert remap: old_id -> new_id
    expert_remaps: list[dict[int, int]] = []
    for keep_ids in plan.layer_keep_ids:
        expert_remaps.append({old: new for new, old in enumerate(keep_ids)})

    # Optionally copy tokenizer and auxiliary files
    if copy_support_files:
        _copy_support_files(src.root, output_dir)

    # Write updated config.json
    new_config = _rewrite_config(src.config, plan, schema)
    write_json(output_dir / "config.json", new_config)

    # Process shards
    new_weight_map: dict[str, str] = {}
    shard_stats: list[dict[str, int]] = []

    for shard_name in src.shard_names:
        shard_keys = src.keys_in_shard(shard_name)
        tensors = load_file(str(src.root / shard_name))
        rewritten: dict[str, torch.Tensor] = {}

        for key in shard_keys:
            tensor = tensors[key]

            # Expert tensor
            expert_m = schema.match_expert(key)
            if expert_m:
                layer_idx = int(expert_m.group("layer"))
                expert_idx = int(expert_m.group("expert"))
                remap = expert_remaps[layer_idx]
                if expert_idx not in remap:
                    continue  # drop this expert
                new_key = _remap_expert_key(key, schema, layer_idx, remap[expert_idx])
                rewritten[new_key] = tensor
                new_weight_map[new_key] = shard_name
                continue

            # Router tensor
            router_m = schema.match_router(key)
            if router_m:
                layer_idx = int(router_m.group("layer"))
                keep_ids = plan.layer_keep_ids[layer_idx]
                rewritten[key] = _slice_router(tensor, keep_ids, key)
                new_weight_map[key] = shard_name
                continue

            # Everything else (attention, norms, embeddings, shared experts)
            rewritten[key] = tensor
            new_weight_map[key] = shard_name

        shard_stats.append({
            "shard": shard_name,
            "source_tensors": len(shard_keys),
            "output_tensors": len(rewritten),
        })

        if not dry_run:
            metadata = {
                "format": "pt",
                "moe_pruner": "true",
                "source_shard": str(src.root / shard_name),
            }
            save_file(rewritten, str(output_dir / shard_name), metadata=metadata)

        if verbose:
            print(
                f"[prune] {shard_name}: "
                f"{len(shard_keys)} → {len(rewritten)} tensors"
                + (" [dry-run]" if dry_run else "")
            )

        del tensors, rewritten

    # Write new index
    if not dry_run:
        total_size = sum(
            (output_dir / s).stat().st_size
            for s in set(new_weight_map.values())
        )
    else:
        total_size = 0

    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    write_json(output_dir / "model.safetensors.index.json", new_index)

    elapsed = time.time() - t0
    manifest = {
        "format_version": 1,
        "dry_run": dry_run,
        "source_dir": str(src.root),
        "output_dir": str(output_dir),
        "source_num_experts": plan.source_num_experts,
        "target_num_experts": plan.target_num_experts,
        "source_topk": plan.source_topk,
        "target_topk": plan.target_topk,
        "strategy": plan.strategy,
        "score": plan.score,
        "elapsed_seconds": round(elapsed, 2),
        "shard_stats": shard_stats,
        "total_output_keys": len(new_weight_map),
        "total_size_bytes": total_size,
    }
    write_json(output_dir / "pruning_manifest.json", manifest)

    if verbose:
        size_gb = total_size / 1e9
        print(
            f"[prune] Done in {elapsed:.1f}s. "
            f"Output: {len(new_weight_map)} tensors, "
            + (f"{size_gb:.2f} GB" if not dry_run else "dry-run, no files written")
        )

    return manifest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _copy_support_files(src_dir: Path, dst_dir: Path) -> None:
    skip = {"model.safetensors.index.json", "config.json", "pruning_manifest.json"}
    for src_path in src_dir.iterdir():
        if src_path.name in skip or src_path.suffix == ".safetensors":
            continue
        dst_path = dst_dir / src_path.name
        if src_path.is_dir():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def _rewrite_config(
    config: dict[str, Any],
    plan: Plan,
    schema: MoESchema,
) -> dict[str, Any]:
    new_cfg = json.loads(json.dumps(config))
    new_cfg[schema.num_experts_config_key] = plan.target_num_experts
    new_cfg[schema.num_experts_per_tok_config_key] = plan.target_topk
    new_cfg["moe_pruner"] = {
        "source_num_experts": plan.source_num_experts,
        "target_num_experts": plan.target_num_experts,
        "strategy": plan.strategy,
        "experimental": True,
    }
    return new_cfg


def _remap_expert_key(
    key: str,
    schema: MoESchema,
    layer_idx: int,
    new_expert_idx: int,
) -> str:
    """Replace the expert index in a tensor key with the new compact index."""
    import re
    m = schema.match_expert(key)
    if not m:
        return key
    old_span = m.span("expert")
    return key[: old_span[0]] + str(new_expert_idx) + key[old_span[1] :]


def _slice_router(
    tensor: torch.Tensor,
    keep_ids: list[int],
    key: str,
) -> torch.Tensor:
    """Select rows from a router weight matrix corresponding to kept expert IDs."""
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected 2D router weight for {key}, got shape {tuple(tensor.shape)}"
        )
    if len(keep_ids) == tensor.shape[0]:
        return tensor
    if max(keep_ids) >= tensor.shape[0]:
        raise ValueError(
            f"Router keep id {max(keep_ids)} out of range for {key} "
            f"with shape {tuple(tensor.shape)}"
        )
    idx = torch.tensor(keep_ids, dtype=torch.long)
    return tensor.index_select(0, idx)

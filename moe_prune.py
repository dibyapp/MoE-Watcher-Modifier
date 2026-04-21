#!/usr/bin/env python3
"""
moe_prune.py — MoE checkpoint pruner.

Rewrites a safetensors MoE checkpoint with fewer experts using a pruning plan
produced by moe_monitor.py. No GPU required. Works with any model family.

Subcommands:
  inspect           Show checkpoint layout and auto-detect schema.
  dry-run           Validate a plan against a checkpoint without writing files.
  prune-checkpoint  Rewrite the checkpoint using a plan.
  uniform-plan      Generate a uniform (no-stats) plan for pipeline testing.

Examples:

  # Inspect a checkpoint:
  python3 moe_prune.py inspect --model-dir /path/to/checkpoint

  # Validate a plan without writing:
  python3 moe_prune.py dry-run \\
    --model-dir /path/to/checkpoint \\
    --plan ./stats/report-plan.json

  # Prune (run after dry-run passes):
  python3 moe_prune.py prune-checkpoint \\
    --model-dir /path/to/checkpoint \\
    --plan ./stats/report-plan.json \\
    --output-dir ./pruned-128experts \\
    --copy-support-files

  # Quick uniform plan (no stats needed, for pipeline testing only):
  python3 moe_prune.py uniform-plan \\
    --model-dir /path/to/checkpoint \\
    --keep-experts 128 --new-topk 4 \\
    --output ./stats/uniform-128-plan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_inspect(args: argparse.Namespace) -> None:
    from moe_pruner.checkpoint import Checkpoint
    from moe_pruner.schema import detect_schema, SCHEMAS

    ckpt = Checkpoint(args.model_dir)
    schema = detect_schema(ckpt) if not args.schema else None

    summary = ckpt.summary()
    summary["detected_schema"] = schema.name if schema else args.schema
    summary["available_schemas"] = list(SCHEMAS.keys())

    if schema or args.schema:
        from moe_pruner.schema import get_schema
        s = get_schema(args.schema, ckpt)
        router_keys = ckpt.find_keys(s.router_pattern)
        expert_keys = ckpt.find_keys(s.expert_pattern)
        summary["router_keys_found"] = len(router_keys)
        summary["expert_keys_found"] = len(expert_keys)
        summary["sample_router_key"] = router_keys[0] if router_keys else None
        summary["sample_expert_key"] = expert_keys[0] if expert_keys else None

    print(json.dumps(summary, indent=2))


def cmd_dry_run(args: argparse.Namespace) -> None:
    from moe_pruner.checkpoint import Checkpoint
    from moe_pruner.schema import get_schema
    from moe_pruner.planner import Plan
    from moe_pruner.pruner import prune_checkpoint

    ckpt = Checkpoint(args.model_dir)
    schema = get_schema(args.schema, ckpt)
    plan = Plan.from_dict(json.loads(Path(args.plan).read_text()))

    print(f"[dry-run] Validating plan: {args.plan}")
    print(f"[dry-run] Source: {plan.source_num_experts} experts → Target: {plan.target_num_experts} experts")

    plan.validate()
    print("[dry-run] Plan validation passed.")

    manifest = prune_checkpoint(
        ckpt, schema, plan,
        output_dir=args.output_dir or "/tmp/_moe_pruner_dryrun",
        copy_support_files=False,
        dry_run=True,
        verbose=True,
    )
    print(f"\n[dry-run] Complete. {manifest['total_output_keys']} output tensors.")


def cmd_prune(args: argparse.Namespace) -> None:
    from moe_pruner.checkpoint import Checkpoint
    from moe_pruner.schema import get_schema
    from moe_pruner.planner import Plan
    from moe_pruner.pruner import prune_checkpoint

    ckpt = Checkpoint(args.model_dir)
    schema = get_schema(args.schema, ckpt)
    plan_data = json.loads(Path(args.plan).read_text())

    # Accept both full report JSON (with nested "plan") and bare plan JSON
    if "plan" in plan_data and "layer_keep_ids" not in plan_data:
        plan_data = plan_data["plan"]
    plan = Plan.from_dict(plan_data)
    plan.validate()

    print(f"[prune] {plan.source_num_experts} → {plan.target_num_experts} experts  "
          f"topk {plan.source_topk} → {plan.target_topk}  strategy={plan.strategy}")

    manifest = prune_checkpoint(
        ckpt, schema, plan,
        output_dir=args.output_dir,
        copy_support_files=args.copy_support_files,
        dry_run=False,
        verbose=True,
    )

    size_gb = manifest["total_size_bytes"] / 1e9
    print(f"\n[prune] Output: {args.output_dir}")
    print(f"[prune] Size: {size_gb:.2f} GB  Tensors: {manifest['total_output_keys']}")
    print(f"[prune] Manifest: {args.output_dir}/pruning_manifest.json")


def cmd_uniform_plan(args: argparse.Namespace) -> None:
    from moe_pruner.checkpoint import Checkpoint
    from moe_pruner.schema import get_schema
    from moe_pruner.planner import build_uniform_plan
    from moe_pruner.checkpoint import write_json

    ckpt = Checkpoint(args.model_dir)
    schema = get_schema(args.schema, ckpt)
    cfg = ckpt.config

    num_layers = cfg[schema.num_layers_config_key]
    num_experts = cfg[schema.num_experts_config_key]
    topk = cfg[schema.num_experts_per_tok_config_key]

    plan = build_uniform_plan(
        num_layers=num_layers,
        num_experts=num_experts,
        source_topk=topk,
        keep_count=args.keep_experts,
        new_topk=args.new_topk,
    )

    output = Path(args.output).expanduser().resolve()
    write_json(output, plan.to_dict())
    print(f"[plan] Uniform plan written: {output}")
    print(f"[plan] {num_experts} → {args.keep_experts} experts, topk → {plan.target_topk}")
    print(f"[plan] WARNING: Uniform plans produce poor quality. Use moe_monitor.py for real stats.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE checkpoint pruner. Works with any safetensors MoE checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p = sub.add_parser("inspect", help="Show checkpoint layout.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--schema", default=None)
    p.set_defaults(func=cmd_inspect)

    # dry-run
    p = sub.add_parser("dry-run", help="Validate a plan without writing files.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--schema", default=None)
    p.set_defaults(func=cmd_dry_run)

    # prune-checkpoint
    p = sub.add_parser("prune-checkpoint", help="Rewrite checkpoint using a plan.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--plan", required=True, help="Plan JSON from moe_monitor.py.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--copy-support-files", action="store_true",
                   help="Copy tokenizer and auxiliary files to output dir.")
    p.add_argument("--schema", default=None)
    p.set_defaults(func=cmd_prune)

    # uniform-plan
    p = sub.add_parser(
        "uniform-plan",
        help="Generate a uniform plan (no stats). For pipeline testing only.",
    )
    p.add_argument("--model-dir", required=True)
    p.add_argument("--keep-experts", type=int, required=True)
    p.add_argument("--new-topk", type=int, default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--schema", default=None)
    p.set_defaults(func=cmd_uniform_plan)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)

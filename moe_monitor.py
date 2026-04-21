#!/usr/bin/env python3
"""
moe_monitor.py — MoE expert usage monitor.

Analyses which experts a Mixture-of-Experts model actually uses, ranks them,
and generates a pruning plan + ready-to-run command.

Works with any MoE checkpoint stored as safetensors. No specific model family,
framework, or hardware required.

Subcommands:
  inspect        Show checkpoint structure and detect MoE schema.
  router-only    Probe routers with random hidden states. No GPU, minimal RAM.
  full-model     Hook a transformers model on real prompts. Needs enough RAM/VRAM.
  report         Re-render or re-rank a saved report without re-probing.
  daemon         Transparent proxy — monitor live traffic from any runtime.

Examples:

  # Inspect a checkpoint and auto-detect its schema:
  python3 moe_monitor.py inspect --model-dir /path/to/checkpoint

  # Collect router-only stats and build a 128-expert plan (no GPU needed):
  python3 moe_monitor.py router-only \\
    --model-dir /path/to/checkpoint \\
    --output ./stats/report.json \\
    --keep-experts 128 --new-topk 4

  # Re-rank an existing report for a different keep count (instant):
  python3 moe_monitor.py report \\
    --input ./stats/report.json \\
    --keep-experts 64

  # Full-model mode on a small/pruned checkpoint that fits in memory:
  python3 moe_monitor.py full-model \\
    --model-dir /path/to/small-checkpoint \\
    --prompts ./prompts.jsonl \\
    --output ./stats/full-report.json \\
    --keep-experts 32 --new-topk 4

  # Daemon: transparent proxy in front of ollama (or any OpenAI-compatible server).
  # Point your app at :8080 instead of :11434. Stats accumulate from real traffic.
  python3 moe_monitor.py daemon \\
    --model-dir /path/to/checkpoint \\
    --backend http://localhost:11434 \\
    --listen-port 8080 \\
    --keep-experts 128 --new-topk 4 \\
    --output ./stats/live-report.json \\
    --report-every 50

  # Force an immediate report while daemon is running:
  kill -USR1 <daemon_pid>
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
    schema = detect_schema(ckpt)

    summary = ckpt.summary()
    summary["detected_schema"] = schema.name if schema else None
    summary["available_schemas"] = list(SCHEMAS.keys())

    if schema:
        router_keys = ckpt.find_keys(schema.router_pattern)
        expert_keys = ckpt.find_keys(schema.expert_pattern)
        summary["router_keys_found"] = len(router_keys)
        summary["expert_keys_found"] = len(expert_keys)
        summary["sample_router_key"] = router_keys[0] if router_keys else None
        summary["sample_expert_key"] = expert_keys[0] if expert_keys else None

    print(json.dumps(summary, indent=2))


def cmd_router_only(args: argparse.Namespace) -> None:
    import time
    from moe_pruner.checkpoint import Checkpoint, write_json
    from moe_pruner.schema import get_schema
    from moe_pruner.stats import collect_router_only
    from moe_pruner.planner import build_plan, build_uniform_plan
    from moe_pruner.report import build_report, print_report

    ckpt = Checkpoint(args.model_dir)
    schema = get_schema(args.schema, ckpt)

    cfg = ckpt.config
    num_layers = cfg.get(schema.num_layers_config_key, 0)
    num_experts = cfg.get(schema.num_experts_config_key, 0)
    topk = cfg.get(schema.num_experts_per_tok_config_key, 0)
    hidden_size = cfg.get(schema.hidden_size_config_key, 0)

    for name, val in [("num_layers", num_layers), ("num_experts", num_experts),
                      ("topk", topk), ("hidden_size", hidden_size)]:
        if not val:
            print(f"ERROR: Could not read {name} from config.json. "
                  "Try --schema or check your checkpoint.", file=sys.stderr)
            sys.exit(1)

    if args.keep_experts > num_experts:
        print(f"ERROR: --keep-experts {args.keep_experts} > checkpoint num_experts {num_experts}",
              file=sys.stderr)
        sys.exit(1)

    try:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
        from rich.console import Console
        _rich = True
        _console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} layers"),
            console=_console,
        )
        task = progress.add_task("Probing routers", total=num_layers)
        progress.start()

        def cb(done, total):
            progress.advance(task)

    except ImportError:
        _rich = False
        cb = None

    stats = collect_router_only(
        ckpt, schema, num_experts, num_layers, topk, hidden_size,
        num_samples=args.samples,
        verbose=not _rich if '_rich' in dir() else True,
        progress_callback=cb,
    )

    if _rich:
        progress.stop()

    plan = build_plan(
        stats,
        keep_count=args.keep_experts,
        strategy=args.strategy,
        score=args.score,
        new_topk=args.new_topk,
    )

    output_path = Path(args.output).expanduser().resolve()
    plan_path = output_path.parent / (output_path.stem + "-plan.json")

    report = build_report(stats, plan, str(ckpt.root), schema.name)
    write_json(output_path, report)
    write_json(plan_path, plan.to_dict())

    print_report(report, plan_path, str(ckpt.root))
    print(f"\nReport: {output_path}")
    print(f"Plan:   {plan_path}")


def cmd_full_model(args: argparse.Namespace) -> None:
    from moe_pruner.checkpoint import Checkpoint, write_json
    from moe_pruner.schema import get_schema
    from moe_pruner.stats import collect_full_model
    from moe_pruner.planner import build_plan
    from moe_pruner.report import build_report, print_report

    ckpt = Checkpoint(args.model_dir)
    schema = get_schema(args.schema, ckpt)

    cfg = ckpt.config
    num_layers = cfg[schema.num_layers_config_key]
    num_experts = cfg[schema.num_experts_config_key]
    topk = cfg[schema.num_experts_per_tok_config_key]

    prompts_path = Path(args.prompts).expanduser().resolve()
    prompts = []
    with prompts_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                prompts.append(json.loads(line)["prompt"])
            else:
                prompts.append(line)

    stats = collect_full_model(
        ckpt, schema, prompts, num_experts, num_layers, topk,
        device_map=args.device_map,
        dtype_str=args.dtype,
        max_length=args.max_length,
        verbose=True,
    )

    plan = build_plan(
        stats,
        keep_count=args.keep_experts,
        strategy=args.strategy,
        score=args.score,
        new_topk=args.new_topk,
    )

    output_path = Path(args.output).expanduser().resolve()
    plan_path = output_path.parent / (output_path.stem + "-plan.json")

    report = build_report(stats, plan, str(ckpt.root), schema.name)
    write_json(output_path, report)
    write_json(plan_path, plan.to_dict())

    print_report(report, plan_path, str(ckpt.root))
    print(f"\nReport: {output_path}")
    print(f"Plan:   {plan_path}")


def cmd_report(args: argparse.Namespace) -> None:
    import torch
    from moe_pruner.checkpoint import Checkpoint, write_json
    from moe_pruner.schema import get_schema
    from moe_pruner.stats import Stats
    from moe_pruner.planner import build_plan, Plan
    from moe_pruner.report import build_report, print_report

    input_path = Path(args.input).expanduser().resolve()
    data = json.loads(input_path.read_text())

    stats = Stats.from_dict({
        "counts": data["counts"],
        "prob_sums": data["prob_sums"],
        "num_layers": len(data["layer_stats"]),
        "num_experts": data["source_num_experts"],
        "topk": data["source_topk"],
        "mode": data["mode"],
        "num_samples": data["num_samples"],
        "elapsed_seconds": data.get("elapsed_seconds", 0),
    })

    keep_count = args.keep_experts or data["target_num_experts"]
    new_topk = args.new_topk or data["target_topk"]

    plan = build_plan(stats, keep_count=keep_count, new_topk=new_topk)

    output_path = input_path.parent / f"{input_path.stem}-{keep_count}experts.json"
    plan_path = input_path.parent / f"{input_path.stem}-{keep_count}experts-plan.json"

    report = build_report(stats, plan, data["model_dir"], data["schema"])
    write_json(output_path, report)
    write_json(plan_path, plan.to_dict())

    print_report(report, plan_path, data["model_dir"])
    print(f"\nReport: {output_path}")
    print(f"Plan:   {plan_path}")


def cmd_daemon(args: argparse.Namespace) -> None:
    from pathlib import Path
    from moe_pruner.daemon import run_daemon
    from moe_pruner.discovery import run_setup_wizard

    # Interactive wizard: validates model-dir, detects system, picks backend
    backend_url, model_dir = run_setup_wizard(
        backend_url=args.backend,
        model_dir=args.model_dir,
    )

    output_path = Path(args.output).expanduser().resolve()

    run_daemon(
        model_dir=model_dir,
        backend_url=backend_url,
        listen_port=args.listen_port,
        listen_host=args.listen_host,
        keep_experts=args.keep_experts,
        new_topk=args.new_topk,
        output_path=output_path,
        report_every=args.report_every,
        schema_name=args.schema,
        strategy=args.strategy,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE expert usage monitor. Works with any safetensors MoE checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p = sub.add_parser("inspect", help="Show checkpoint structure and detect schema.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--schema", default=None, help="Override auto-detected schema.")
    p.set_defaults(func=cmd_inspect)

    # router-only
    p = sub.add_parser(
        "router-only",
        help="Probe routers with random hidden states. No GPU, <200 MB RAM.",
    )
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True, help="Path to write report JSON.")
    p.add_argument("--keep-experts", type=int, required=True)
    p.add_argument("--new-topk", type=int, default=None)
    p.add_argument("--samples", type=int, default=512,
                   help="Random hidden states per router. Default: 512.")
    p.add_argument("--strategy", choices=("global", "per-layer"), default="per-layer")
    p.add_argument("--score", choices=("counts", "prob_sums"), default="counts")
    p.add_argument("--schema", default=None, help="Override auto-detected schema.")
    p.set_defaults(func=cmd_router_only)

    # full-model
    p = sub.add_parser(
        "full-model",
        help="Hook a transformers model on real prompts. Needs RAM/VRAM to load model.",
    )
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompts", required=True, help="JSONL or plain text prompts file.")
    p.add_argument("--output", required=True)
    p.add_argument("--keep-experts", type=int, required=True)
    p.add_argument("--new-topk", type=int, default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--strategy", choices=("global", "per-layer"), default="per-layer")
    p.add_argument("--score", choices=("counts", "prob_sums"), default="counts")
    p.add_argument("--schema", default=None)
    p.set_defaults(func=cmd_full_model)

    # report
    p = sub.add_parser(
        "report",
        help="Re-render or re-rank a saved report without re-probing.",
    )
    p.add_argument("--input", required=True, help="Existing report JSON.")
    p.add_argument("--keep-experts", type=int, default=None,
                   help="Re-rank with a different keep count.")
    p.add_argument("--new-topk", type=int, default=None)
    p.set_defaults(func=cmd_report)

    # daemon
    p = sub.add_parser(
        "daemon",
        help=(
            "Transparent proxy daemon. Sits in front of any OpenAI-compatible "
            "server (ollama, vLLM, llama.cpp, etc.) and silently accumulates "
            "real expert usage stats from live traffic."
        ),
    )
    p.add_argument("--model-dir", required=True,
                   help="Checkpoint directory (loads only embed_tokens + routers).")
    p.add_argument("--backend", default=None,
                   help="Backend model server URL, e.g. http://localhost:11434. "
                        "If omitted, auto-discovers running servers and asks you to pick one.")
    p.add_argument("--listen-port", type=int, default=8080,
                   help="Port to listen on. Default: 8080.")
    p.add_argument("--listen-host", default="0.0.0.0",
                   help="Host to bind. Default: 0.0.0.0.")
    p.add_argument("--keep-experts", type=int, required=True,
                   help="Target expert count for generated pruning plans.")
    p.add_argument("--new-topk", type=int, default=None,
                   help="Target top-k for pruning plan. Defaults to keep proportional.")
    p.add_argument("--output", default="./stats/live-report.json",
                   help="Path for periodic report JSON. Default: ./stats/live-report.json")
    p.add_argument("--report-every", type=int, default=100,
                   help="Emit a report every N analyzed prompts. Default: 100.")
    p.add_argument("--strategy", choices=("global", "per-layer"), default="per-layer",
                   help="Expert ranking strategy. Default: per-layer.")
    p.add_argument("--schema", default=None,
                   help="Override auto-detected schema.")
    p.set_defaults(func=cmd_daemon)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)

"""
Rich terminal report and JSON report generation.

Falls back to plain text if `rich` is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .stats import Stats
from .planner import Plan

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None


def build_report(
    stats: Stats,
    plan: Plan,
    model_dir: str,
    schema_name: str,
) -> dict[str, Any]:
    """Build the full JSON report combining stats and plan."""
    global_rank = stats.global_rank()
    counts = stats.counts

    layer_stats = []
    for i in range(stats.num_layers):
        layer_counts = counts[i]
        active = int((layer_counts > 0).sum().item())
        top5 = torch.topk(layer_counts, k=min(5, stats.num_experts)).indices.tolist()
        layer_stats.append({
            "layer": i,
            "active_experts": active,
            "top5_experts": top5,
            "keep_ids": plan.layer_keep_ids[i],
        })

    return {
        "format_version": 1,
        "schema": schema_name,
        "model_dir": model_dir,
        "mode": stats.mode,
        "num_samples": stats.num_samples,
        "elapsed_seconds": stats.elapsed_seconds,
        "source_num_experts": stats.num_experts,
        "source_topk": stats.topk,
        "target_num_experts": plan.target_num_experts,
        "target_topk": plan.target_topk,
        "total_router_selections": stats.total_selections,
        "expert_coverage_fraction": round(stats.coverage_fraction, 4),
        "global_expert_ranking": global_rank,
        "counts": stats.counts.tolist(),
        "prob_sums": stats.prob_sums.tolist(),
        "layer_stats": layer_stats,
        "plan": plan.to_dict(),
    }


def print_report(
    report: dict[str, Any],
    plan_path: Path,
    model_dir: str,
) -> None:
    if _RICH:
        _print_rich(report, plan_path, model_dir)
    else:
        _print_plain(report, plan_path, model_dir)


def _print_rich(report: dict[str, Any], plan_path: Path, model_dir: str) -> None:
    c = _console

    c.rule("[bold green]MoE-Watcher-Modifier — Expert Usage Report")

    summary = (
        f"[bold]Model:[/bold]           {report['model_dir']}\n"
        f"[bold]Schema:[/bold]          {report['schema']}\n"
        f"[bold]Mode:[/bold]            {report['mode']}\n"
        f"[bold]Samples/Prompts:[/bold] {report['num_samples']}\n"
        f"[bold]Source experts:[/bold]  {report['source_num_experts']} (top-k={report['source_topk']})\n"
        f"[bold]Target experts:[/bold]  {report['target_num_experts']} (top-k={report['target_topk']})\n"
        f"[bold]Total selections:[/bold] {report['total_router_selections']:,}\n"
        f"[bold]Coverage:[/bold]        {report['expert_coverage_fraction']*100:.1f}% of experts selected at least once"
    )
    c.print(Panel(summary, title="Summary", border_style="green"))

    # Top-20 globally
    table = Table(title="Top 20 Most Used Experts (Global)", border_style="cyan")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Expert ID", style="bold yellow")
    table.add_column("Selections", style="bold white")
    table.add_column("% of Total", style="green")

    counts = torch.tensor(report["counts"])
    global_counts = counts.sum(dim=0)
    total = global_counts.sum().item()
    top20 = torch.topk(global_counts, k=min(20, global_counts.shape[0])).indices.tolist()
    for rank, eid in enumerate(top20, 1):
        cnt = int(global_counts[eid].item())
        pct = cnt / total * 100 if total > 0 else 0
        table.add_row(str(rank), str(eid), f"{cnt:,}", f"{pct:.2f}%")
    c.print(table)

    # Per-layer sample
    layer_table = Table(
        title="Per-Layer Stats (first 8 layers)", border_style="blue"
    )
    layer_table.add_column("Layer", width=6)
    layer_table.add_column("Active", style="bold")
    layer_table.add_column("Top-5 Experts", style="yellow")
    layer_table.add_column("Keep IDs (first 5)", style="green")
    for ls in report["layer_stats"][:8]:
        layer_table.add_row(
            str(ls["layer"]),
            str(ls["active_experts"]),
            str(ls["top5_experts"]),
            str(ls["keep_ids"][:5]) + "...",
        )
    c.print(layer_table)

    # Prune command
    prune_cmd = (
        f"python3 moe_prune.py prune-checkpoint \\\n"
        f"  --model-dir {model_dir} \\\n"
        f"  --plan {plan_path} \\\n"
        f"  --output-dir ./pruned-{report['target_num_experts']}experts \\\n"
        f"  --copy-support-files"
    )
    c.print(Panel(
        f"[bold yellow]Plan written to:[/bold yellow] {plan_path}\n\n"
        f"[bold green]Run to prune:[/bold green]\n\n"
        f"[bold white]{prune_cmd}[/bold white]",
        title="Next Step",
        border_style="yellow",
    ))


def _print_plain(report: dict[str, Any], plan_path: Path, model_dir: str) -> None:
    print("=" * 60)
    print("MoE-Watcher-Modifier — Expert Usage Report")
    print("=" * 60)
    print(f"  Model:    {report['model_dir']}")
    print(f"  Schema:   {report['schema']}")
    print(f"  Mode:     {report['mode']}  samples={report['num_samples']}")
    print(f"  Source:   {report['source_num_experts']} experts, top-k={report['source_topk']}")
    print(f"  Target:   {report['target_num_experts']} experts, top-k={report['target_topk']}")
    print(f"  Coverage: {report['expert_coverage_fraction']*100:.1f}%")
    print()
    print("Top 10 experts globally:")
    counts = torch.tensor(report["counts"])
    global_counts = counts.sum(dim=0)
    total = global_counts.sum().item()
    top10 = torch.topk(global_counts, k=min(10, global_counts.shape[0])).indices.tolist()
    for rank, eid in enumerate(top10, 1):
        cnt = int(global_counts[eid].item())
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {rank:2}. expert {eid:4d} — {cnt:6,} selections ({pct:.2f}%)")
    print()
    print(f"Plan written to: {plan_path}")
    print()
    print("Run to prune:")
    print(
        f"  python3 moe_prune.py prune-checkpoint \\\n"
        f"    --model-dir {model_dir} \\\n"
        f"    --plan {plan_path} \\\n"
        f"    --output-dir ./pruned-{report['target_num_experts']}experts \\\n"
        f"    --copy-support-files"
    )

#!/usr/bin/env python3
"""
Generate Reddit-ready expert routing visualizations from a MoE report JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render heatmap + utilization charts from report JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("stats/report-512.json"),
        help="Path to report JSON with `counts`.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("stats/plots"),
        help="Output directory for generated PNG + summary JSON.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="qwen3_router",
        help="Filename prefix for output files.",
    )
    return parser.parse_args()


def _load_counts(report_path: Path) -> tuple[dict, np.ndarray]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "counts" not in report:
        raise ValueError(f"Missing `counts` in report: {report_path}")

    counts = np.asarray(report["counts"], dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(
            f"`counts` must be 2D [layers, experts], got shape {counts.shape}"
        )

    return report, counts


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError("Gini is undefined for negative values.")
    total = x.sum()
    if total <= 0:
        return 0.0

    sorted_x = np.sort(x)
    n = sorted_x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_x) / (n * total)) - ((n + 1) / n))


def _build_summary(report: dict, counts: np.ndarray) -> dict:
    global_counts = counts.sum(axis=0)
    sorted_counts = np.sort(global_counts)[::-1]
    total = float(sorted_counts.sum())
    cumulative = np.cumsum(sorted_counts) / total
    num_experts = int(sorted_counts.size)
    cold_per_layer = (counts == 0).sum(axis=1)

    def share_for_top_fraction(frac: float) -> float:
        k = max(1, int(round(num_experts * frac)))
        return float(cumulative[k - 1])

    return {
        "schema": report.get("schema"),
        "mode": report.get("mode"),
        "num_layers": int(counts.shape[0]),
        "num_experts": int(counts.shape[1]),
        "num_samples": report.get("num_samples"),
        "total_router_selections": int(total),
        "global_max_count": int(sorted_counts[0]),
        "global_min_count": int(sorted_counts[-1]),
        "global_gini": round(_gini(global_counts), 4),
        "top_1pct_share": round(share_for_top_fraction(0.01), 4),
        "top_5pct_share": round(share_for_top_fraction(0.05), 4),
        "top_10pct_share": round(share_for_top_fraction(0.10), 4),
        "top_20pct_share": round(share_for_top_fraction(0.20), 4),
        "top_50pct_share": round(share_for_top_fraction(0.50), 4),
        "avg_cold_experts_per_layer": round(float(cold_per_layer.mean()), 3),
        "max_cold_experts_in_layer": int(cold_per_layer.max()),
        "layers_with_any_cold_experts": int((cold_per_layer > 0).sum()),
    }


def _render_heatmap(
    report: dict, counts: np.ndarray, summary: dict, output_path: Path
) -> None:
    global_counts = counts.sum(axis=0)
    sort_idx = np.argsort(global_counts)[::-1]
    sorted_counts = counts[:, sort_idx]

    layer_totals = sorted_counts.sum(axis=1, keepdims=True)
    row_fraction = np.divide(
        sorted_counts,
        layer_totals,
        out=np.zeros_like(sorted_counts),
        where=layer_totals > 0,
    )
    vmax = float(np.percentile(row_fraction, 99.5))

    fig, ax = plt.subplots(figsize=(16, 7), dpi=220)
    image = ax.imshow(
        row_fraction,
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Per-layer selection share")

    num_layers, num_experts = counts.shape
    ax.set_title(
        "Qwen3-MoE Expert Routing Heatmap (Real Stats from report-512.json)",
        fontsize=14,
        pad=12,
    )
    ax.set_ylabel("Layer index")
    ax.set_xlabel("Expert rank (sorted by global utilization)")
    ax.set_yticks(np.arange(0, num_layers, 4))
    ax.set_xticks(np.arange(0, num_experts, 64))

    footer = (
        f"schema={report.get('schema')}  mode={report.get('mode')}  "
        f"samples={report.get('num_samples')}  "
        f"total selections={summary['total_router_selections']:,}  "
        f"top10% share={summary['top_10pct_share'] * 100:.1f}%"
    )
    fig.text(0.01, 0.01, footer, fontsize=9, ha="left")

    fig.tight_layout(rect=[0.0, 0.03, 1.0, 1.0])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_illustration(
    report: dict, counts: np.ndarray, summary: dict, output_path: Path
) -> None:
    global_counts = counts.sum(axis=0)
    sorted_counts = np.sort(global_counts)[::-1]
    cumulative_pct = np.cumsum(sorted_counts) / sorted_counts.sum() * 100.0
    num_experts = sorted_counts.size
    ranks = np.arange(1, num_experts + 1)
    cold_per_layer = (counts == 0).sum(axis=1)
    active_per_layer = counts.shape[1] - cold_per_layer

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        dpi=220,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    ax1.plot(ranks, cumulative_pct, color="#0ea5e9", linewidth=2.5)
    ax1.fill_between(ranks, cumulative_pct, color="#7dd3fc", alpha=0.25)

    for frac in (0.01, 0.05, 0.10, 0.20, 0.50):
        k = max(1, int(round(num_experts * frac)))
        y = cumulative_pct[k - 1]
        ax1.scatter([k], [y], color="#0369a1", s=24)
        ax1.text(k, y + 1.4, f"top {int(frac * 100)}%: {y:.1f}%", fontsize=9)

    ax1.set_xlim(1, num_experts)
    ax1.set_ylim(0, 102)
    ax1.set_title("Global Expert Utilization Concentration", fontsize=13, pad=8)
    ax1.set_xlabel("Top-N experts (ranked by global usage)")
    ax1.set_ylabel("Cumulative traffic share (%)")
    ax1.grid(alpha=0.25, linewidth=0.6)

    layer_idx = np.arange(counts.shape[0])
    ax2.bar(layer_idx, active_per_layer, color="#fb7185", alpha=0.9, width=0.8)
    ax2.plot(layer_idx, active_per_layer, color="#be123c", linewidth=1.4)
    ax2.set_ylim(0, counts.shape[1] + 5)
    ax2.set_title(
        "Per-layer active experts (higher means less collapse)",
        fontsize=12,
        pad=8,
    )
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Active experts")
    ax2.grid(axis="y", alpha=0.2, linewidth=0.6)

    footer = (
        f"mode={report.get('mode')}  gini={summary['global_gini']:.3f}  "
        f"avg cold experts/layer={summary['avg_cold_experts_per_layer']:.2f}  "
        f"max cold in one layer={summary['max_cold_experts_in_layer']}"
    )
    fig.text(0.01, 0.01, footer, fontsize=9, ha="left")

    fig.tight_layout(rect=[0.0, 0.03, 1.0, 1.0])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    report, counts = _load_counts(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(report, counts)

    heatmap_path = args.outdir / f"{args.prefix}_heatmap.png"
    illustration_path = args.outdir / f"{args.prefix}_illustration.png"
    summary_path = args.outdir / f"{args.prefix}_summary.json"

    _render_heatmap(report, counts, summary, heatmap_path)
    _render_illustration(report, counts, summary, illustration_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] Heatmap:      {heatmap_path}")
    print(f"[ok] Illustration: {illustration_path}")
    print(f"[ok] Summary:      {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dandi_analysis.visualisation.loaders import (
    article_figure_path,
    load_000718_pri_enrichment,
    load_000718_robustness,
)
from dandi_analysis.visualisation.style import PALETTE, apply_style


def _pair_label(entry: dict) -> str:
    return f"{entry['subject']}\n{entry['pair_label']}"


def _entry_colors(count: int) -> list:
    if count <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(count)]
    cmap = plt.get_cmap("viridis")
    return [cmap(value) for value in np.linspace(0.1, 0.9, count)]


def save_000718_enrichment_figure(output_path: Path | None = None) -> Path:
    apply_style()
    data = load_000718_pri_enrichment()
    output = output_path or article_figure_path("figure_6_open_data_000718_enrichment.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [_pair_label(entry) for entry in data]
    real = [entry["real_by_threshold"]["0.0"]["mean_enrichment"] for entry in data]
    shuffle_mean = [entry["c1_reg_shuffle"]["mean_enrichment"] for entry in data]
    shuffle_std = [entry["c1_reg_shuffle"]["std_enrichment"] for entry in data]
    deltas = [r - s for r, s in zip(real, shuffle_mean)]

    x = np.arange(len(labels))
    width = 0.34

    fig_width = max(9.2, 2.1 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(x - width / 2, real, width=width, color=PALETTE["real"], label="Real enrichment")
    ax.bar(
        x + width / 2,
        shuffle_mean,
        width=width,
        color=PALETTE["shuffle"],
        label="C1 registration shuffle",
        yerr=shuffle_std,
        capsize=5,
    )

    for idx, delta in enumerate(deltas):
        y = max(real[idx], shuffle_mean[idx] + shuffle_std[idx]) + 0.0015
        ax.text(x[idx], y, f"delta={delta:+.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean event minus inter-event enrichment")
    ax.set_title(
        "DANDI 000718: NeutralExposure core-unit enrichment exceeds shuffled registration baseline",
        pad=26,
    )
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(top=0.72, right=0.8)
    fig.savefig(output, format="png")
    plt.close(fig)
    return output


def save_000718_threshold_sweep_figure(output_path: Path | None = None) -> Path:
    apply_style()
    data = load_000718_pri_enrichment()
    output = output_path or article_figure_path("figure_7_open_data_000718_threshold_sweep.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [0.0, 0.5, 1.0]
    fig_width = max(8.4, 1.3 * len(data) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))

    colors = _entry_colors(len(data))
    for color, entry in zip(colors, data):
        y = [entry["real_by_threshold"][str(threshold)]["mean_enrichment"] for threshold in thresholds]
        ax.plot(
            thresholds,
            y,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=color,
            label=_pair_label(entry).replace("\n", " "),
        )

    ax.set_xticks(thresholds)
    ax.set_xlabel("Activity threshold (sigma)")
    ax.set_ylabel("Mean event minus inter-event enrichment")
    ax.set_title("DANDI 000718: enrichment remains positive across threshold choices", pad=40)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    legend_cols = min(4, max(1, len(data)))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=legend_cols,
        borderaxespad=0.2,
        handlelength=1.8,
    )
    fig.subplots_adjust(top=0.75)
    fig.savefig(output, format="png")
    fig.savefig(output.with_suffix(".svg"), format="svg")
    plt.close(fig)
    return output


def save_000718_robustness_heatmap(output_path: Path | None = None) -> Path:
    apply_style()
    data = load_000718_robustness()
    output = output_path or article_figure_path("figure_s1_open_data_000718_robustness_heatmap.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    signal_types = ["deconvolved", "denoised"]
    thresholds = sorted({float(entry["threshold_sigma"]) for entry in data})
    methods = ["nmf", "ica", "graph"]

    fig, axes = plt.subplots(1, len(methods), figsize=(12.0, 4.4), sharey=True)
    fig.suptitle(
        "DANDI 000718 supplement: robustness of event significance across signal types and thresholds",
        y=0.98,
    )

    vmin = min(
        entry["methods"][method]["fraction_significant"]
        for entry in data
        for method in methods
    )
    vmax = max(
        entry["methods"][method]["fraction_significant"]
        for entry in data
        for method in methods
    )

    last_im = None
    for axis, method in zip(axes, methods):
        matrix = np.zeros((len(signal_types), len(thresholds)))
        for row_idx, signal_type in enumerate(signal_types):
            for col_idx, threshold in enumerate(thresholds):
                match = next(
                    entry
                    for entry in data
                    if entry["signal_type"] == signal_type and float(entry["threshold_sigma"]) == threshold
                )
                matrix[row_idx, col_idx] = match["methods"][method]["fraction_significant"]

        last_im = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap="Blues", aspect="auto")
        axis.set_title(method.upper())
        axis.set_xticks(range(len(thresholds)))
        axis.set_xticklabels([f"{threshold:.1f}" for threshold in thresholds])
        axis.set_yticks(range(len(signal_types)))
        axis.set_yticklabels(signal_types)
        axis.set_xlabel("Threshold sigma")

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                axis.text(
                    col_idx,
                    row_idx,
                    f"{matrix[row_idx, col_idx]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

    axes[0].set_ylabel("Signal type")
    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=list(axes),
            shrink=0.85,
            fraction=0.032,
            pad=0.015,
            label="Fraction significant",
        )
    fig.subplots_adjust(top=0.82)
    fig.savefig(output, format="png")
    plt.close(fig)
    return output

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dandi_analysis.visualisation.style import PALETTE, apply_style


PRIMARY_CONDITIONS = ["spontaneous", "gratings", "fixed_gabors"]


def _conditions_map(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "conditions" in entry:
        return entry["conditions"]
    return {key: value for key, value in entry.items() if isinstance(value, dict)}


def _cross_summary(condition_data: dict[str, Any]) -> dict[str, Any]:
    if "cross" in condition_data:
        return condition_data["cross"]
    return condition_data["cross_plane"]


def _within_keys(condition_data: dict[str, Any]) -> tuple[str, str]:
    if "within_shallow" in condition_data:
        return "within_shallow", "within_deep"
    return "within_a", "within_b"


def _condition_triplets(entry: dict[str, Any], condition_order: list[str]) -> tuple[list[float], list[float], list[float]]:
    conditions = _conditions_map(entry)
    cross = []
    within_a = []
    within_b = []
    for condition in condition_order:
        summary = conditions[condition]
        key_a, key_b = _within_keys(summary)
        cross.append(_cross_summary(summary)["mean_r"])
        within_a.append(summary[key_a]["mean_r"])
        within_b.append(summary[key_b]["mean_r"])
    return cross, within_a, within_b


def _pairing_label(entry: dict[str, Any]) -> str:
    pairing = str(entry.get("pairing", "pair")).replace("_", "-")
    return pairing


def _entry_title(entry: dict[str, Any]) -> str:
    subject = entry.get("subject", entry.get("id", "pair"))
    return f"{subject}\n{_pairing_label(entry)}"


def _entry_color(entry: dict[str, Any]) -> str:
    pairing = _pairing_label(entry)
    if "area" in pairing:
        return PALETTE["supplementary"]
    return PALETTE["cross"]


def render_condition_coupling_figure(
    bundle: list[dict[str, Any]],
    output_path: Path,
    *,
    dataset_name: str,
    title: str,
) -> Path:
    apply_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(bundle)
    n_cols = min(3, max(1, n_panels))
    n_rows = math.ceil(n_panels / n_cols)
    fig_width = 4.5 * n_cols + 0.8
    fig_height = 4.6 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=True)
    axes_array = np.atleast_1d(axes).ravel()

    for axis, entry in zip(axes_array, bundle):
        conditions = [condition for condition in PRIMARY_CONDITIONS if condition in _conditions_map(entry)]
        cross, within_a, within_b = _condition_triplets(entry, conditions)
        x = np.arange(len(conditions))
        width = 0.25

        axis.bar(x - width, cross, width=width, color=PALETTE["cross"], label="Cross-plane")
        axis.bar(x, within_a, width=width, color=PALETTE["within_a"], label="Within plane A")
        axis.bar(x + width, within_b, width=width, color=PALETTE["within_b"], label="Within plane B")
        axis.set_title(_entry_title(entry))
        axis.set_xticks(x)
        axis.set_xticklabels([condition.replace("_", "\n") for condition in conditions])
        axis.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        axis.set_xlabel("Condition")

    for axis in axes_array[n_panels:]:
        axis.axis("off")

    axes_array[0].set_ylabel("Mean coupling (r)")
    axes_array[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.03))
    fig.suptitle(f"{dataset_name}: {title}", y=0.98)
    fig.subplots_adjust(top=0.85, wspace=0.25, hspace=0.35)
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path


def render_spontaneous_summary_figure(
    bundle: list[dict[str, Any]],
    output_path: Path,
    *,
    dataset_name: str,
    title: str,
) -> Path:
    apply_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [_entry_title(entry) for entry in bundle]
    spontaneous_sets = [_conditions_map(entry)["spontaneous"] for entry in bundle]
    mean_r = [_cross_summary(entry)["mean_r"] for entry in spontaneous_sets]
    z_scores = [_cross_summary(entry)["z_vs_null"] for entry in spontaneous_sets]
    colors = [_entry_color(entry) for entry in bundle]

    x = np.arange(len(labels))
    fig_width = max(10.8, 1.9 * len(labels) + 4.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.6))

    axes[0].bar(x, mean_r, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Cross-plane mean r")
    axes[0].set_title("Spontaneous coupling")

    axes[1].bar(x, z_scores, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("z vs null")
    axes[1].set_title("Null-separated effect size")

    for axis, values in zip(axes, [mean_r, z_scores]):
        max_value = max(values) if values else 0.0
        min_value = min(values) if values else 0.0
        value_span = max(max_value - min_value, abs(max_value), 1.0)
        offset = max(0.03 * value_span, 0.03)
        upper = max_value + 3.0 * offset
        lower = min(0.0, min_value - 0.08 * value_span)
        axis.set_ylim(lower, upper)
        for idx, value in enumerate(values):
            axis.text(idx, value + offset, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"{dataset_name}: {title}", y=0.98)
    fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.28)
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path

from __future__ import annotations

from pathlib import Path

from dandi_analysis.visualisation.crossplane_bundle import (
    render_condition_coupling_figure,
    render_spontaneous_summary_figure,
)
from dandi_analysis.visualisation.loaders import (
    article_figure_path,
    load_000336_full_bundle,
)


def save_000336_condition_coupling_figure(output_path: Path | None = None) -> Path:
    output = output_path or article_figure_path("figure_8_open_data_000336_coupling_by_condition.png")
    bundle = load_000336_full_bundle()
    return render_condition_coupling_figure(
        bundle,
        output,
        dataset_name="DANDI 000336",
        title="cross-plane coupling remains below within-plane coupling across bundle pairs",
    )


def save_000336_replication_figure(output_path: Path | None = None) -> Path:
    output = output_path or article_figure_path("figure_9_open_data_000336_replication.png")
    bundle = load_000336_full_bundle()
    return render_spontaneous_summary_figure(
        bundle,
        output,
        dataset_name="DANDI 000336",
        title="spontaneous cross-plane signature across all bundle pairs",
    )

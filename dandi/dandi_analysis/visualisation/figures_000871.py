from __future__ import annotations

from pathlib import Path

from dandi_analysis.visualisation.crossplane_bundle import (
    render_condition_coupling_figure,
    render_spontaneous_summary_figure,
)
from dandi_analysis.visualisation.loaders import (
    article_figure_path,
    load_000871_cross_area,
    load_000871_primary,
    load_000871_replication,
)


def _load_000871_bundle() -> list[dict]:
    return [
        {
            "id": "pair_a",
            "subject": "sub-644972",
            "pairing": "cross_depth",
            "conditions": load_000871_primary(),
        },
        {
            "id": "pair_b",
            "subject": "sub-656228",
            "pairing": "cross_depth",
            "conditions": load_000871_replication(),
        },
        {
            "id": "pair_c",
            "subject": "sub-656228",
            "pairing": "cross_area",
            "conditions": load_000871_cross_area(),
        },
    ]


def save_000871_condition_coupling_figure(output_path: Path | None = None) -> Path:
    output = output_path or article_figure_path("figure_8_open_data_000871_coupling_by_condition.png")
    bundle = _load_000871_bundle()
    return render_condition_coupling_figure(
        bundle,
        output,
        dataset_name="DANDI 000871",
        title="cross-plane coupling remains below within-plane coupling across legacy pairs",
    )


def save_000871_replication_figure(output_path: Path | None = None) -> Path:
    output = output_path or article_figure_path("figure_9_open_data_000871_replication.png")
    bundle = _load_000871_bundle()
    return render_spontaneous_summary_figure(
        bundle,
        output,
        dataset_name="DANDI 000871",
        title="spontaneous cross-plane signature across legacy pairs",
    )

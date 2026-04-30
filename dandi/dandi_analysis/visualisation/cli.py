from __future__ import annotations

import argparse
from pathlib import Path

from dandi_analysis.visualisation.figures_000718 import (
    save_000718_enrichment_figure,
    save_000718_robustness_heatmap,
    save_000718_threshold_sweep_figure,
)
from dandi_analysis.visualisation.figures_000336 import (
    save_000336_condition_coupling_figure,
    save_000336_replication_figure,
)
from dandi_analysis.visualisation.figures_000871 import (
    save_000871_condition_coupling_figure,
    save_000871_replication_figure,
)
from dandi_analysis.visualisation.loaders import ARTICLE_FIGURES_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render manuscript-facing DANDI figures from triage JSON outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTICLE_FIGURES_ROOT,
        help="Directory where SVG figures will be written.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        save_000718_enrichment_figure(output_dir / "figure_6_open_data_000718_enrichment.png"),
        save_000718_threshold_sweep_figure(output_dir / "figure_7_open_data_000718_threshold_sweep.png"),
        save_000336_condition_coupling_figure(output_dir / "figure_8_open_data_000336_coupling_by_condition.png"),
        save_000336_replication_figure(output_dir / "figure_9_open_data_000336_replication.png"),
        save_000871_condition_coupling_figure(output_dir / "figure_8_open_data_000871_coupling_by_condition.png"),
        save_000871_replication_figure(output_dir / "figure_9_open_data_000871_replication.png"),
        save_000718_robustness_heatmap(output_dir / "figure_s1_open_data_000718_robustness_heatmap.png"),
    ]

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

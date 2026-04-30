from __future__ import annotations

import matplotlib

matplotlib.use("Agg")


PALETTE = {
    "real": "#274c77",
    "shuffle": "#a3cef1",
    "cross": "#2a9d8f",
    "within_a": "#457b9d",
    "within_b": "#1d3557",
    "supplementary": "#8d99ae",
}


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )

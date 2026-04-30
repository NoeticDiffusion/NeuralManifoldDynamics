from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
TRIAGE_ROOT = REPO_ROOT / "data" / "dandi" / "triage"


def _default_article_figures_root() -> Path:
    candidates = [
        REPO_ROOT
        / ".article"
        / "A Cytoskeletal-Dendritic Accessibility Model of Associative Memory"
        / "figures",
        REPO_ROOT
        / ".article"
        / "A Cytoskeletal-Dendritic Key-Lock Model of Associative Memory"
        / "figures",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


ARTICLE_FIGURES_ROOT = _default_article_figures_root()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def triage_path(*parts: str) -> Path:
    return TRIAGE_ROOT.joinpath(*parts)


def article_figure_path(filename: str) -> Path:
    return ARTICLE_FIGURES_ROOT / filename


def load_000718_pri_enrichment(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or triage_path("000718", "h1_pri_enrichment.json")
    return load_json(target)


def load_000718_robustness(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or triage_path("000718", "h1_robustness.json")
    return load_json(target)


def load_000871_primary(path: Path | None = None) -> dict[str, Any]:
    target = path or triage_path("000871", "crossplane_coupling.json")
    return load_json(target)


def load_000871_replication(path: Path | None = None) -> dict[str, Any]:
    target = path or triage_path("000871", "crossplane_coupling_sub656228.json")
    return load_json(target)


def load_000871_cross_area(path: Path | None = None) -> dict[str, Any]:
    target = path or triage_path("000871", "crossplane_coupling_sub-656228_ses-1245548523.json")
    return load_json(target)


def load_000336_primary(path: Path | None = None) -> dict[str, Any]:
    target = path or triage_path("000336", "crossplane_coupling.json")
    return load_json(target)


def load_000336_replication(path: Path | None = None) -> dict[str, Any]:
    target = path or triage_path("000336", "crossplane_coupling_sub656228.json")
    return load_json(target)


def load_000336_cross_area(path: Path | None = None) -> dict[str, Any]:
    target = path or triage_path("000336", "crossplane_coupling_sub-656228_ses-1245548523.json")
    return load_json(target)


def load_000336_full_bundle(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or triage_path("000336", "full_bundle_coupling.json")
    return load_json(target)

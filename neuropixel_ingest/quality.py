"""Dataset-agnostic quality filtering for spike-sorted NWB Units tables."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .contracts import EphysReadConfig


def filter_units(units: pd.DataFrame, config: EphysReadConfig = EphysReadConfig()) -> pd.DataFrame:
    """Apply opt-in quality rules without assuming one lab's column names.

    ``quality_policy`` supports ``all``, ``good`` and ``acceptable``.  The
    latter two prefer string labels (``good``/``mua``), otherwise use IBL's
    numeric ``ibl_quality_score`` convention.  Explicit numeric thresholds are
    applied whenever their corresponding columns are available.
    """
    if units.empty:
        return units.copy()

    selected = units.copy()
    policy = config.quality_policy.lower().strip()
    if policy not in {"all", "good", "acceptable"}:
        raise ValueError("quality_policy must be one of: all, good, acceptable")

    if policy != "all":
        mask = _quality_mask(selected, policy)
        selected = selected.loc[mask].copy()
    selected = _threshold_filter(selected, "firing_rate", config.min_firing_rate_hz, np.greater_equal)
    selected = _threshold_filter(selected, "presence_ratio", config.min_presence_ratio, np.greater_equal)
    selected = _threshold_filter(
        selected, "isi_violations_ratio", config.max_isi_violations_ratio, np.less_equal
    )
    return selected.reset_index(drop=True)


def unit_quality_summary(units: pd.DataFrame) -> dict[str, int]:
    """Return disjoint normalized quality-tier counts."""
    if units.empty:
        return {"good": 0, "acceptable": 0, "poor": 0, "noise": 0, "unknown": 0, "total": 0}
    tiers = _quality_tiers(units)
    counts = {tier: int((tiers == tier).sum()) for tier in ("good", "acceptable", "poor", "noise", "unknown")}
    counts["total"] = len(units)
    return counts


def _quality_mask(units: pd.DataFrame, policy: str) -> pd.Series:
    tiers = _quality_tiers(units)
    if not (tiers == "unknown").all():
        allowed = {"good"} if policy == "good" else {"good", "acceptable"}
        return tiers.isin(allowed)

    warnings.warn(
        f"No recognized unit-quality column; quality_policy={policy!r} retains all units.",
        stacklevel=2,
    )
    return pd.Series(True, index=units.index)


def normalize_quality_tier(label: object | None = None, score: object | None = None) -> str:
    """Map common sorting labels and IBL scores to a portable quality tier."""
    if score is not None:
        try:
            value = float(score)
            if np.isclose(value, 1.0, atol=1e-3):
                return "good"
            if np.isclose(value, 2.0 / 3.0, atol=1e-3):
                return "acceptable"
            if np.isclose(value, 1.0 / 3.0, atol=1e-3):
                return "poor"
            if np.isclose(value, 0.0, atol=1e-3):
                return "noise"
        except (TypeError, ValueError):
            pass
    normalized = str(label or "").lower().strip()
    if normalized in {"excellent", "good", "single", "single_unit", "su"}:
        return "good"
    if normalized in {"acceptable", "mua", "multiunit", "multi-unit"}:
        return "acceptable"
    if normalized in {"poor", "bad"}:
        return "poor"
    if normalized in {"noise", "artifact", "artefact"}:
        return "noise"
    return "unknown"


def _quality_tiers(units: pd.DataFrame) -> pd.Series:
    label_col = next((name for name in ("quality", "label", "kilosort2_label") if name in units), None)
    labels = units[label_col] if label_col is not None else pd.Series(None, index=units.index)
    scores = units.get("ibl_quality_score", pd.Series(None, index=units.index))
    return pd.Series(
        [normalize_quality_tier(label, score) for label, score in zip(labels, scores, strict=True)],
        index=units.index,
        dtype="object",
    )


def _threshold_filter(
    units: pd.DataFrame, column: str, threshold: float | None, compare
) -> pd.DataFrame:
    if threshold is None or column not in units:
        return units
    numeric = pd.to_numeric(units[column], errors="coerce")
    return units.loc[numeric.isna() | compare(numeric, float(threshold))]

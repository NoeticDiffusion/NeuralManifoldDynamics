"""Portable unit-quality normalization tests."""

import numpy as np
import pandas as pd

from neuropixel_ingest.contracts import EphysReadConfig
from neuropixel_ingest.quality import filter_units, normalize_quality_tier, unit_quality_summary


def test_quality_labels_and_ibl_scores_are_normalized() -> None:
    assert normalize_quality_tier("Excellent") == "good"
    assert normalize_quality_tier("mua") == "acceptable"
    assert normalize_quality_tier(score=2.0 / 3.0 + 1e-5) == "acceptable"
    assert normalize_quality_tier(score=1.0 / 3.0) == "poor"


def test_summary_tiers_are_disjoint_and_good_policy_keeps_excellent() -> None:
    units = pd.DataFrame({"quality": ["Excellent", "mua", "poor", "noise", "unscored"]})
    summary = unit_quality_summary(units)
    assert summary == {"good": 1, "acceptable": 1, "poor": 1, "noise": 1, "unknown": 1, "total": 5}
    retained = filter_units(units, EphysReadConfig(quality_policy="good"))
    assert retained["quality"].tolist() == ["Excellent"]

"""Focused deterministic checks for VECTOR-PILOT-001 handoffs and controls."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mndm" / "src"))

from mndm.mnps.v3_2.experiments.vector_pilot_001 import (  # noqa: E402
    CHANNEL_COUNT,
    ED_REALIZATION_ID,
    NIE_REALIZATION_ID,
    OI_REALIZATION_ID,
    _make_row,
    construct_absolute_pearson_weights,
    control_seed,
    independent_ar1_sources,
    pairwise_associations,
)


def test_centered_pearson_constructs_exact_symmetric_absolute_weights_without_clipping() -> None:
    base = np.arange(100, dtype=np.float64)
    signals = np.vstack((base, 2.0 * base + 3.0, -4.0 * base + 2.0))

    weights, audit = construct_absolute_pearson_weights(signals)

    assert weights is not None
    assert np.array_equal(weights, weights.T)
    assert np.array_equal(np.diag(weights), np.zeros(3))
    assert weights[0, 1] == pytest.approx(1.0, abs=1e-14)
    assert weights[0, 2] == pytest.approx(1.0, abs=1e-14)
    assert audit["edge_estimator_id"] == "absolute_zero_lag_Pearson_r"
    assert audit["edge_rule_id"] == "no_threshold"
    assert audit["correlation_computation"] == "centered_gram_upper_triangle_pairwise_mirror_no_clipping"


def test_constant_channel_is_candidate_local_unsupported_edge_input() -> None:
    signals = np.vstack((np.arange(20, dtype=np.float64), np.ones(20)))

    weights, audit = construct_absolute_pearson_weights(signals)

    assert weights is None
    assert audit["status"] == "unsupported"
    assert audit["reason"] == "constant_channel_pearson_undefined"
    assert audit["constant_channel_indices"] == [1]


def test_unsupported_first_window_is_retained_as_three_null_family_rows(tmp_path: Path) -> None:
    labels = tuple(f"channel-{index:02d}" for index in range(CHANNEL_COUNT))
    window = SimpleNamespace(
        status="unsupported",
        reason="insufficient_effective_filter_edge_support",
        signals=None,
        channel_names=labels,
        support=None,
        provenance={
            "source_dependency": {
                "source_sample_start": -3376,
                "source_sample_end_exclusive": 19997,
            }
        },
    )

    row = _make_row(
        index=0,
        window=window,
        nominal=(0.0, 20.0),
        labels=labels,
        run_id="test-run",
        code_sha256="a" * 64,
        protocol_sha256="b" * 64,
        overlay_sha256="c" * 64,
        runtime={},
        edge_dir=tmp_path,
    )

    assert row["window_id"] == "ANPHY_EPCTL01_000_020s"
    assert row["source_dependency"] == {
        "source_sample_start": -3376,
        "source_sample_end_exclusive": 19997,
    }
    assert set(row["measurements"]) == {"ED", "OI", "NIE"}
    for family in row["measurements"].values():
        record = family["measurement_record"]
        assert record["value"] is None
        assert record["valid"] is False
        assert record["qualified"] is False
        assert record["status"] == "unsupported"
        assert record["reason"] == "insufficient_effective_filter_edge_support"
        assert record["support"]["support_status"] == "unsupported"
    assert row["measurements"]["ED"]["realization_id"] == ED_REALIZATION_ID
    assert row["measurements"]["OI"]["realization_id"] == OI_REALIZATION_ID
    assert row["measurements"]["NIE"]["realization_id"] == NIE_REALIZATION_ID


def test_pairwise_association_uses_pairwise_valid_rows_without_inference() -> None:
    rows = []
    for index in range(4):
        value = float(index + 1)
        rows.append({
            "window_id": f"window-{index}",
            "measurements": {
                "ED": {"measurement_record": {"valid": True, "value": value}},
                "OI": {"measurement_record": {"valid": index != 2, "value": value if index != 2 else None}},
                "NIE": {"measurement_record": {"valid": True, "value": 5.0 - value}},
            },
        })

    result = pairwise_associations(rows)

    assert result["inference"] == "none_descriptive_only"
    assert result["independent_window_assumption"] is False
    assert result["pairs"]["ED_OI"]["n"] == 3
    assert result["pairs"]["ED_OI"]["window_ids"] == ["window-0", "window-1", "window-3"]
    assert result["pairs"]["ED_OI"]["pearson_r"] == pytest.approx(1.0)
    assert result["pairs"]["ED_NIE"]["pearson_r"] == pytest.approx(-1.0)
    assert "p_value" not in result["pairs"]["ED_OI"]


def test_synthetic_ar1_control_seed_and_generation_are_reproducible() -> None:
    seed = control_seed(7)
    first = independent_ar1_sources(seed, channel_count=4, sample_count=100)
    second = independent_ar1_sources(seed, channel_count=4, sample_count=100)

    assert np.array_equal(first, second)
    assert first.shape == (4, 100)
    assert not np.array_equal(first[0], first[1])
    assert control_seed(7) != control_seed(8)

"""Synthetic checks for multi-surface, multi-window E3 execution."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1 import (
    CORE19_CHANNELS,
    MNPS31SensorSurface,
    MNPS31WindowSlice,
    MNPS31WindowSupport,
    compute_e3_surface_window_grid,
)


def _support(start: float, *, sample_count: int = 800, rate: float = 100.0, gap_free: bool = True):
    return MNPS31WindowSupport(
        window_start_sec=start,
        window_end_sec=start + sample_count / rate,
        sample_rate_hz=rate,
        sample_count=sample_count,
        segment_id="synthetic-segment-1",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=gap_free,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )


def _window(index: int, start_sample: int, *, gap_free: bool = True) -> MNPS31WindowSlice:
    return MNPS31WindowSlice(
        window_id=f"w{index:02d}",
        start_sample=start_sample,
        end_sample_exclusive=start_sample + 800,
        support=_support(index * 8.0, gap_free=gap_free),
    )


def _surface(surface_id: str, kind: str, labels: tuple[str, ...], **kwargs) -> MNPS31SensorSurface:
    return MNPS31SensorSurface(
        surface_id=surface_id,
        kind=kind,
        channel_names=labels,
        reference_mode="average_over_surface",
        **kwargs,
    )


def _rows_by_key(rows):
    return {(row["surface_id"], row["window_id"]): row["measurement"] for row in rows}


def test_e3_emits_each_window_for_native_core19_and_study_common_surfaces() -> None:
    rng = np.random.default_rng(42)
    labels = CORE19_CHANNELS + tuple(f"Extra{i:02d}" for i in range(13))
    signals = rng.normal(size=(len(labels), 2400))
    core19 = CORE19_CHANNELS
    study_common = labels[:24]
    surfaces = (
        _surface("native", "native", labels, allow_partial_channels=True),
        _surface("core19", "core19", core19),
        _surface("study_common_ds-example", "study_common", study_common),
    )
    windows = (_window(0, 0), _window(1, 800), _window(2, 1600))

    rows = compute_e3_surface_window_grid(
        signals,
        labels,
        surfaces,
        windows,
        preprocessing_id="synthetic-250hz-1to45hz-v1",
    )

    assert len(rows) == len(surfaces) * len(windows)
    keyed = _rows_by_key(rows)
    for surface in surfaces:
        for window in windows:
            result = keyed[(surface.surface_id, window.window_id)]
            assert result["computed"] is True
            assert result["valid"] is True
            assert result["qualified"] is False
            assert result["estimator_id"] == "E3.sensor_covariance_participation_ratio.v0"
            assert result["details"]["surface_id"] == surface.surface_id
            assert result["details"]["window_id"] == window.window_id
            assert result["details"]["cross_montage_equivalent"] == "unknown"
            assert result["details"]["preprocessing_id"] == "synthetic-250hz-1to45hz-v1"
            assert result["details"]["reference_mode"] == "average_over_effective_surface_channels"
            assert result["details"]["reference_channel_count"] == len(surface.channel_names)
    assert keyed[("native", "w00")]["details"]["effective_channel_count"] == 32
    assert keyed[("core19", "w00")]["details"]["effective_channel_count"] == 19
    assert keyed[("study_common_ds-example", "w00")]["details"]["effective_channel_count"] == 24


def test_one_bad_native_channel_reduces_coverage_but_does_not_block_other_surfaces() -> None:
    rng = np.random.default_rng(7)
    labels = CORE19_CHANNELS + tuple(f"Extra{i:02d}" for i in range(8))
    signals = rng.normal(size=(len(labels), 800))
    signals[0, :] = np.nan
    native = _surface("native", "native", labels, allow_partial_channels=True)
    core19 = _surface("core19", "core19", CORE19_CHANNELS)

    rows = compute_e3_surface_window_grid(
        signals,
        labels,
        (native, core19),
        (_window(0, 0),),
        preprocessing_id="synthetic-preprocessed-v1",
        bad_channel_names=("EEG_typo",),
    )
    keyed = _rows_by_key(rows)
    native_result = keyed[("native", "w00")]
    core_result = keyed[("core19", "w00")]

    assert native_result["computed"] is True
    assert native_result["details"]["effective_channel_count"] == len(labels) - 1
    assert native_result["details"]["channel_coverage_fraction"] == (len(labels) - 1) / len(labels)
    assert native_result["details"]["rejected_channel_labels"] == [CORE19_CHANNELS[0]]
    assert native_result["details"]["unmatched_bad_channel_labels"] == ["EEG_typo"]
    assert "bad_channel_metadata_label_not_found" in native_result["details"]["qc_flags"]
    assert core_result["computed"] is False
    assert core_result["reason"] == "required_surface_channel_unavailable"
    assert core_result["value"] is None


def test_grid_preserves_gap_window_as_unsupported_row_for_every_surface() -> None:
    rng = np.random.default_rng(11)
    labels = CORE19_CHANNELS
    signals = rng.normal(size=(len(labels), 1600))
    surface = _surface("core19", "core19", labels)
    windows = (_window(0, 0), _window(1, 800, gap_free=False))

    rows = compute_e3_surface_window_grid(
        signals,
        labels,
        (surface,),
        windows,
        preprocessing_id="synthetic-preprocessed-v1",
    )
    keyed = _rows_by_key(rows)

    assert len(rows) == 2
    assert keyed[("core19", "w00")]["computed"] is True
    assert keyed[("core19", "w01")]["computed"] is False
    assert keyed[("core19", "w01")]["reason"] == "gap_or_segment_boundary"
    assert keyed[("core19", "w01")]["value"] is None


def test_missing_core_channel_does_not_suppress_computable_native_rows() -> None:
    rng = np.random.default_rng(14)
    source_labels = CORE19_CHANNELS[1:] + tuple(f"Extra{i:02d}" for i in range(8))
    signals = rng.normal(size=(len(source_labels), 800))
    native = _surface("native", "native", source_labels, allow_partial_channels=True)
    core19 = _surface("core19", "core19", CORE19_CHANNELS)

    rows = compute_e3_surface_window_grid(
        signals,
        source_labels,
        (native, core19),
        (_window(0, 0),),
        preprocessing_id="synthetic-preprocessed-v1",
    )
    keyed = _rows_by_key(rows)

    assert keyed[("native", "w00")]["computed"] is True
    assert keyed[("core19", "w00")]["computed"] is False
    assert keyed[("core19", "w00")]["reason"] == "required_surface_channel_missing"
    assert keyed[("core19", "w00")]["details"]["missing_channel_labels"] == [CORE19_CHANNELS[0]]


def test_small_study_common_computes_like_custom_and_stays_strict() -> None:
    labels = CORE19_CHANNELS[:8]
    signals = np.random.default_rng(17).normal(size=(8, 800))
    surfaces = (_surface("study-small", "study_common", labels),
                _surface("custom-small", "custom", labels))
    rows = compute_e3_surface_window_grid(
        signals, labels, surfaces, (_window(0, 0),), preprocessing_id="synthetic-v1",
    )
    measurements = [row["measurement"] for row in rows]
    assert all(row["computed"] for row in measurements)
    assert measurements[0]["value"] == measurements[1]["value"]
    assert measurements[0]["details"]["effective_channel_count"] == 8
    missing = compute_e3_surface_window_grid(
        signals[1:], labels[1:], surfaces[:1], (_window(0, 0),), preprocessing_id="synthetic-v1",
    )
    assert missing[0]["measurement"]["reason"] == "required_surface_channel_missing"
    with pytest.raises(ValueError, match="core19_surface_must"):
        _surface("core19-small", "core19", labels)

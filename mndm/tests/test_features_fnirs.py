from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mndm.features.fnirs import FnirsFeatureContractError, compute_fnirs_features
from mndm.parallel import (
    _compute_features_from_preprocessed,
    _get_modality_handlers,
    _merge_feature_frames,
    _read_feature_table,
    write_merged_features,
    write_temp_features_csv,
)
from mndm.preprocess import PreprocessedSignals


def _payload(times: np.ndarray, *, data: np.ndarray | None = None) -> dict:
    if data is None:
        hbo = 1.0 + np.arange(times.size, dtype=float)
        hbr = -1.0 - np.arange(times.size, dtype=float)
        data = np.vstack([hbo, hbr])
    sfreq = 1.0 / float(np.median(np.diff(times)))
    gaps = [int(i) for i in np.flatnonzero(np.diff(times) > 1.0 / sfreq + 1e-6) + 1]
    return {
        "signals": {"fnirs": data},
        "sfreq": sfreq,
        "channels": {"fnirs": ["S1_D1 hbo", "S1_D1 hbr"]},
        "dataset_id": "synthetic",
        "meta": {
            "file": "run.snirf",
            "fnirs": {
                "channel_types": ["hbo", "hbr"],
                "channel_units": "M (molar; MNE FIFF_UNIT_MOL)",
                "preprocessing_provenance": {
                    "schema": "mndm.fnirs_preprocessing_provenance.v1",
                    "source": {"sha256": "a" * 64},
                    "input_classification": "cw_amplitude",
                    "software": {"mne": "1.6.1", "mne_nirs": "0.7.3"},
                    "operations": [{"name": "beer_lambert_law", "status": "applied"}],
                    "haemoglobin_units": "M (molar; MNE FIFF_UNIT_MOL)",
                    "measurement_validity": "not_assessed",
                    "claim_status": "no_biological_claim",
                },
            },
            "signal_support_provenance": {
                "schema": "mndm.fnirs_signal_support.v1",
                "sampling_frequency_hz": sfreq,
                "sample_count": int(times.size),
                "time_start_s": float(times[0]),
                "time_end_s": float(times[-1]),
                "time_gap_sample_indices": gaps,
                "sample_times_s": times.tolist(),
            },
        },
    }


def _config(*, length: float = 3.0, step: float = 3.0, min_support: float = 1.0) -> dict:
    return {
        "features": {
            "fnirs": {
                "window_length_s": length,
                "window_step_s": step,
                "min_valid_support_fraction": min_support,
            }
        }
    }


def test_fnirs_handler_is_registered_without_eeg_or_global_defaults():
    assert "fnirs" in _get_modality_handlers()


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({}, "features.fnirs must explicitly configure"),
        ({"features": {"fnirs": {"window_length_s": 3.0}}}, "missing required setting"),
        (_config(length=0), "window_length_s"),
        (_config(step=float("nan")), "window_step_s"),
        (_config(min_support=0), "min_valid_support_fraction"),
    ],
)
def test_fnirs_window_config_fails_closed(cfg: dict, message: str):
    with pytest.raises(FnirsFeatureContractError, match=message):
        compute_fnirs_features(_payload(np.arange(12, dtype=float)), cfg)


def test_fnirs_missing_source_hash_and_transform_provenance_fails_closed():
    payload = _payload(np.arange(12, dtype=float))
    payload["meta"]["fnirs"].pop("preprocessing_provenance")
    with pytest.raises(FnirsFeatureContractError, match="preprocessing_provenance is required"):
        compute_fnirs_features(payload, _config())


def test_fnirs_window_grid_uses_snirf_clock_and_records_gap_crossing(tmp_path: Path):
    times = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    frame = compute_fnirs_features(_payload(times), _config())

    assert frame["epoch_id"].tolist() == [0, 1, 2, 3]
    assert frame["t_start"].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert frame["t_end"].tolist() == [3.0, 6.0, 9.0, 12.0]
    assert frame["fnirs_clock_basis"].unique().tolist() == ["snirf_time_s"]
    assert frame["fnirs_window_status"].tolist() == ["clock_supported", "clock_rejected", "clock_supported", "clock_rejected"]
    assert frame["fnirs_rejection_reason"].iloc[1] == "sample_clock_gap"
    assert frame["fnirs_rejection_reason"].iloc[3] == "partial_recording_window"
    assert frame["fnirs_segment_id"].iloc[0] == 0
    assert pd.isna(frame["fnirs_segment_id"].iloc[1])
    assert frame["fnirs_segment_id"].iloc[2] == 1
    assert frame.loc[0, "fnirs_mean_M__S1_D1%20hbo"] == pytest.approx(2.0)
    assert np.isnan(frame.loc[1, "fnirs_mean_M__S1_D1%20hbo"])
    assert frame["qc_ok_fnirs"].isna().all()
    assert set(frame["fnirs_qc_status"]) == {"not_assessed"}
    assert frame.attrs["fnirs_feature_provenance"]["gap_policy"] == "global_grid_no_reset_and_reject_crossing_windows"

    csv_path = tmp_path / "fnirs.csv"
    frame.to_csv(csv_path, index=False)
    csv = pd.read_csv(csv_path)
    assert csv["epoch_id"].tolist() == frame["epoch_id"].tolist()
    assert csv["fnirs_rejection_reason"].fillna("").tolist() == frame["fnirs_rejection_reason"].astype("string").fillna("").tolist()
    assert csv["fnirs_mean_M__S1_D1%20hbo"].isna().tolist() == frame["fnirs_mean_M__S1_D1%20hbo"].isna().tolist()
    assert csv["fnirs_source_sha256"].tolist() == ["a" * 64] * len(frame)
    assert csv["fnirs_preprocessing_provenance_json"].tolist() == frame["fnirs_preprocessing_provenance_json"].tolist()
    assert csv["qc_ok_fnirs"].isna().all()
    assert csv["fnirs_qualification_status"].tolist() == ["not_assessed"] * len(frame)

    parquet_path = tmp_path / "fnirs.parquet"
    frame.to_parquet(parquet_path, index=False)
    parquet = pd.read_parquet(parquet_path)
    assert parquet.columns.tolist() == frame.columns.tolist()
    assert parquet["epoch_id"].tolist() == frame["epoch_id"].tolist()
    assert parquet["fnirs_rejection_reason"].fillna("").tolist() == frame["fnirs_rejection_reason"].astype("string").fillna("").tolist()
    assert parquet["fnirs_mean_M__S1_D1%20hbo"].isna().tolist() == frame["fnirs_mean_M__S1_D1%20hbo"].isna().tolist()
    assert parquet["fnirs_source_sha256"].tolist() == ["a" * 64] * len(frame)
    assert parquet["fnirs_preprocessing_provenance_json"].tolist() == frame["fnirs_preprocessing_provenance_json"].tolist()
    assert parquet["qc_ok_fnirs"].isna().all()
    assert parquet["fnirs_qualification_status"].tolist() == ["not_assessed"] * len(frame)


def test_fnirs_per_channel_missingness_is_support_not_qc_pass():
    times = np.arange(12, dtype=float)
    data = np.vstack([np.arange(12, dtype=float), np.arange(12, dtype=float)])
    data[0, 1] = np.nan
    frame = compute_fnirs_features(_payload(times, data=data), _config(min_support=0.9))

    assert frame.loc[0, "fnirs_rejection_reason__S1_D1%20hbo"] == "insufficient_finite_sample_support"
    assert frame.loc[0, "fnirs_valid_support_fraction__S1_D1%20hbo"] == pytest.approx(2 / 3)
    assert np.isnan(frame.loc[0, "fnirs_mean_M__S1_D1%20hbo"])
    assert frame.loc[0, "fnirs_mean_M__S1_D1%20hbr"] == pytest.approx(1.0)
    assert frame.loc[0, "qc_ok_fnirs"] is pd.NA or pd.isna(frame.loc[0, "qc_ok_fnirs"])


def test_fnirs_window_shorter_than_one_sample_interval_is_rejected():
    with pytest.raises(FnirsFeatureContractError, match="at least one nominal sample interval"):
        compute_fnirs_features(_payload(np.arange(12, dtype=float)), _config(length=0.25, step=0.25))


def test_fnirs_fractional_boundaries_use_duration_weighted_sample_support():
    times = np.arange(6, dtype=float)
    data = np.vstack([2.0 * times, -2.0 * times])
    frame = compute_fnirs_features(_payload(times, data=data), _config(length=1.0, step=0.5))

    # [0.5, 1.5) is half supported by sample 0 and half by sample 1.
    assert frame.loc[1, "fnirs_window_sample_count"] == 2
    assert frame.loc[1, "fnirs_window_timestamp_count"] == 1
    assert frame.loc[1, "fnirs_actual_covered_duration_s"] == pytest.approx(1.0)
    assert frame.loc[1, "fnirs_valid_support_fraction__S1_D1%20hbo"] == pytest.approx(1.0)
    assert frame.loc[1, "fnirs_valid_sample_count__S1_D1%20hbo"] == 2
    assert frame.loc[1, "fnirs_valid_timestamp_count__S1_D1%20hbo"] == 1
    assert frame.loc[1, "fnirs_mean_M__S1_D1%20hbo"] == pytest.approx(1.0)
    assert frame.loc[1, "fnirs_sd_M__S1_D1%20hbo"] == pytest.approx(1.0)
    assert frame.loc[1, "fnirs_rejection_reason__S1_D1%20hbo"] is pd.NA or pd.isna(
        frame.loc[1, "fnirs_rejection_reason__S1_D1%20hbo"]
    )


def test_fnirs_semantic_channel_and_chromophore_are_column_identity():
    frame = compute_fnirs_features(_payload(np.arange(12, dtype=float)), _config())

    assert "fnirs_mean_M__S1_D1%20hbo" in frame
    assert "fnirs_mean_M__S1_D1%20hbr" in frame
    assert frame.loc[0, "fnirs_chromophore__S1_D1%20hbo"] == "hbo"
    assert frame.loc[0, "fnirs_chromophore__S1_D1%20hbr"] == "hbr"


def test_fnirs_csv_schema_sidecar_preserves_nullable_dtypes(tmp_path: Path):
    frame = compute_fnirs_features(_payload(np.arange(12, dtype=float)), _config())
    paths = write_merged_features(
        frame,
        tmp_path,
        io_policy={"write_csv": True, "write_parquet": True},
    )

    assert "csv_schema" in paths
    csv = _read_feature_table(paths["csv"])
    parquet = _read_feature_table(paths["parquet"])

    assert str(csv["qc_ok_fnirs"].dtype) == "boolean"
    assert str(parquet["qc_ok_fnirs"].dtype) == "boolean"
    assert str(csv["fnirs_segment_id"].dtype) == str(parquet["fnirs_segment_id"].dtype) == "Int64"
    assert {name: str(dtype) for name, dtype in csv.dtypes.items()} == {
        name: str(dtype) for name, dtype in parquet.dtypes.items()
    }
    assert csv["qc_ok_fnirs"].isna().all() and parquet["qc_ok_fnirs"].isna().all()


def test_fnirs_temp_csv_reuse_removes_stale_schema_sidecar(tmp_path: Path):
    fnirs = compute_fnirs_features(_payload(np.arange(12, dtype=float)), _config())
    base = tmp_path / "features.csv"
    write_temp_features_csv(fnirs, base, "reuse", {"write_csv": True})
    temp_csv = tmp_path / "features_reuse.csv"
    schema = tmp_path / "features_reuse.csv.schema.json"
    assert schema.exists()

    write_temp_features_csv(pd.DataFrame({"ordinary_value": [1.0]}), base, "reuse", {"write_csv": True})

    assert not schema.exists()
    assert _read_feature_table(temp_csv)["ordinary_value"].tolist() == [1.0]


def test_fnirs_grid_preserves_nonzero_snirf_time_origin():
    times = 100.5 + np.arange(12, dtype=float) / 2.0
    frame = compute_fnirs_features(_payload(times), _config(length=2.0, step=2.0))

    assert frame.loc[0, "t_start"] == pytest.approx(100.5)
    assert frame.loc[0, "t_end"] == pytest.approx(102.5)
    assert frame.loc[0, "fnirs_run_relative_t_start_s"] == pytest.approx(0.0)
    assert frame.loc[0, "fnirs_run_relative_t_end_s"] == pytest.approx(2.0)
    assert frame.loc[0, "fnirs_window_status"] == "clock_supported"


def test_fnirs_handler_runs_through_normalized_preprocessed_payload():
    payload = _payload(np.arange(12, dtype=float))
    preprocessed = PreprocessedSignals(
        signals=payload["signals"],
        sfreq=payload["sfreq"],
        channels=payload["channels"],
        meta=payload["meta"],
    )

    frame, timings = _compute_features_from_preprocessed(preprocessed, _config())

    assert frame is not None
    assert len(frame) == 4
    assert "embodied_arousal_proxy" not in frame.columns
    assert "embodied_arousal_proxy_source" not in frame.columns
    assert frame.attrs["fnirs_feature_provenance"]["contract"] == "mndm.fnirs_features.v0.1"
    assert frame.attrs["signal_support_provenance"]["fnirs_feature_provenance"]["window_length_s"] == 3.0
    assert timings["feature_fnirs"] >= 0


def test_fnirs_feature_frames_refuse_epoch_only_alignment_with_eeg():
    f = compute_fnirs_features(_payload(np.arange(12, dtype=float)), _config())
    eeg = pd.DataFrame({"epoch_id": [0, 1], "t_start": [99.0, 102.0], "t_end": [102.0, 105.0], "eeg_x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="independent native clock"):
        _merge_feature_frames([f, eeg])
    with pytest.raises(ValueError, match="Multiple fNIRS feature frames"):
        _merge_feature_frames([f, f.copy()])

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))


def test_spectral_metrics_are_scale_invariant_and_zero_power_stays_explicit() -> None:
    from scipy.signal import get_window

    from neuropixel_ingest.lfp_mnps_v34 import compute_spectral_window_metrics

    fs = 500.0
    nperseg = 1000
    nfft = 2000
    n_samples = 2000
    time = np.arange(n_samples, dtype=np.float64) / fs
    signal = 0.4 * np.sin(2.0 * np.pi * 4.0 * time + 0.3)
    frequency = np.fft.rfftfreq(nfft, d=1.0 / fs)
    df = fs / nfft
    band = (frequency >= 3.0 - df / 2.0) & (frequency <= 5.0 + df / 2.0)

    metrics = compute_spectral_window_metrics(
        np.vstack([signal, 7.0 * signal, np.zeros_like(signal)]),
        fs=fs,
        window=get_window("hann", nperseg, fftbins=True),
        nperseg=nperseg,
        noverlap=500,
        nfft=nfft,
        frequencies=frequency,
        band_mask=band,
        df=df,
    )
    assert metrics["centroid_hz"][0] == pytest.approx(4.0, abs=0.02)
    assert metrics["centroid_hz"][1] == pytest.approx(metrics["centroid_hz"][0], abs=1e-12)
    assert metrics["bandpower_v2"][1] == pytest.approx(49.0 * metrics["bandpower_v2"][0], rel=1e-12)
    assert metrics["bandpower_v2"][2] == 0.0
    assert np.isnan(metrics["centroid_hz"][2])
    assert metrics["centroid_missing_reason"][2] == "zero_band_power"


def test_timestamp_scan_rejects_gap_without_flattening() -> None:
    from neuropixel_ingest.lfp_mnps_v34 import scan_explicit_timestamps

    timestamps = 13.0 + np.arange(100, dtype=np.float64) / 500.0
    result = scan_explicit_timestamps(timestamps)
    assert result["segment_count"] == 1
    assert result["sample_rate_hz"] == pytest.approx(500.0)

    timestamps[50:] += 0.01
    with pytest.raises(ValueError, match="discontinuities"):
        scan_explicit_timestamps(timestamps)


def test_envelope_guard_is_stable_at_batch_edges_and_amplitude_scaling() -> None:
    from scipy.signal import butter, hilbert, sosfiltfilt

    from neuropixel_ingest.lfp_mnps_v34 import compute_envelope_block, compute_envelope_cv_from_block

    fs = 500.0
    guard_samples = int(8.0 * fs)
    window_samples = int(4.0 * fs)
    step_samples = int(2.0 * fs)
    sos = butter(4, [3.0, 5.0], btype="bandpass", fs=fs, output="sos")
    padlen = 3 * (2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
    time = np.arange(int(180.0 * fs), dtype=np.float64) / fs
    signal = (1.0 + 0.5 * np.sin(2.0 * np.pi * 0.2 * time + 0.37)) * np.sin(
        2.0 * np.pi * 3.5 * time + 0.73
    )
    reference = np.abs(hilbert(sosfiltfilt(sos, signal, padtype="odd", padlen=padlen)))

    batch_start = int(52.0 * fs)
    batch_stop = int(130.0 * fs)
    block = signal[batch_start:batch_stop][None, :]
    envelope, finite = compute_envelope_block(block, sos=sos, padlen=padlen)
    scaled, scaled_finite = compute_envelope_block(block * 7.0, sos=sos, padlen=padlen)
    assert finite.tolist() == [True]
    assert scaled_finite.tolist() == [True]

    rmse_over_mean = []
    peak_over_mean = []
    cv_values = []
    for start_sec in range(60, 120, 2):
        start = int(start_sec * fs)
        local_start = start - batch_start
        assert start - batch_start >= guard_samples
        assert batch_stop - (start + window_samples) >= guard_samples
        window_envelope = envelope[0, local_start : local_start + window_samples]
        reference_window = reference[start : start + window_samples]
        rmse_over_mean.append(np.sqrt(np.mean((window_envelope - reference_window) ** 2)) / np.mean(reference_window))
        peak_over_mean.append(np.max(np.abs(window_envelope - reference_window)) / np.mean(reference_window))
        cv, reason = compute_envelope_cv_from_block(
            envelope,
            local_window_start=local_start,
            window_samples=window_samples,
        )
        scaled_cv, scaled_reason = compute_envelope_cv_from_block(
            scaled,
            local_window_start=local_start,
            window_samples=window_samples,
        )
        assert scaled_finite.tolist() == [True]
        assert reason[0] == scaled_reason[0] == ""
        assert scaled_cv[0] == pytest.approx(cv[0], rel=1e-10, abs=1e-12)
        cv_values.append(float(cv[0]))

    assert max(rmse_over_mean) <= 0.001
    assert max(peak_over_mean) <= 0.003
    assert np.isfinite(cv_values).all()


def test_group_aggregation_uses_contact_medians_and_frozen_support() -> None:
    from neuropixel_ingest.lfp_mnps_v34 import aggregate_contact_metrics

    contacts = [{"channel_index": idx} for idx in range(5)]
    group = {
        "group_id": "probeF__MOs__depth_bin_0",
        "ordered_channel_indices": [0, 1, 2, 3, 4],
        "support_rule": {
            "denominator": 5,
            "required_computable_contacts": 4,
        },
    }
    aggregate = aggregate_contact_metrics(
        np.asarray([[1.0, 2.0, 3.0, 4.0, np.nan], [1.0, 2.0, np.nan, np.nan, np.nan]]),
        contacts=contacts,
        groups=[group],
    )
    assert aggregate["values"][0, 0] == 2.5
    assert aggregate["support_count"][:, 0].tolist() == [4, 2]
    assert aggregate["support_fraction"][:, 0].tolist() == [0.8, 0.4]
    assert aggregate["support_pass"][:, 0].tolist() == [True, False]
    assert np.isnan(aggregate["values"][1, 0])


def test_candidate_hdf5_roundtrip_keeps_contract_and_support(tmp_path: Path) -> None:
    import h5py
    import pandas as pd

    from neuropixel_ingest.lfp_mnps_v34 import EXPORT_SCHEMA, write_candidate_export_h5

    estimator = {"spec_id": "synthetic-v1", "input": {"series_path": "acquisition/LFP"}}
    contact_spec = {"contact_set_sha256": "frozen-test-hash"}
    result = {
        "contacts": [
            {"contact_id": "c0", "channel_index": 0, "electrode_id": 10, "probe_id": "probeF", "location_raw": "MOs", "depth_bin_index": 0, "probe_vertical_position_um": 10.0, "probe_horizontal_position_um": 11.0, "is_data_valid": True, "reference_declaration": "tip", "group_id": "g0"},
            {"contact_id": "c1", "channel_index": 1, "electrode_id": 11, "probe_id": "probeF", "location_raw": "MOs", "depth_bin_index": 0, "probe_vertical_position_um": 20.0, "probe_horizontal_position_um": 27.0, "is_data_valid": False, "reference_declaration": "tip", "group_id": "g0"},
        ],
        "groups": [{"group_id": "g0"}],
        "windows": {"window_index": np.asarray([0]), "source_time_start_sec": np.asarray([13.0])},
        "contact_metrics": {"centroid_hz": np.asarray([[3.7, np.nan]])},
        "contact_missing_reasons": {"centroid_hz": np.asarray([["", "metadata_invalid_channel"]], dtype=object)},
        "group_metrics": {"centroid_hz_median": np.asarray([[3.7]]), "centroid_hz_support_count": np.asarray([[1]])},
        "annotations": pd.DataFrame({"lfp_behavioral_state": ["awake"], "lfp_stim_adjacent": [False]}),
    }
    target = tmp_path / "candidate_smoke.h5"
    write_candidate_export_h5(
        target,
        result=result,
        estimator_spec=estimator,
        contact_group_spec=contact_spec,
        provenance={"condition_comparisons_performed": False},
    )
    with h5py.File(target, "r") as handle:
        assert handle.attrs["schema"] == EXPORT_SCHEMA
        assert handle.attrs["contact_set_sha256"] == "frozen-test-hash"
        assert handle["contact_metrics/centroid_hz"].shape == (1, 2)
        assert np.isnan(handle["contact_metrics/centroid_hz"][0, 1])
        assert handle["group_metrics/centroid_hz_support_count"][0, 0] == 1
        assert handle["contacts/is_data_valid"][...].tolist() == [True, False]
        spec_json = handle["contracts/estimator_spec_json"][()].decode("utf-8")
        assert json.loads(spec_json)["spec_id"] == "synthetic-v1"

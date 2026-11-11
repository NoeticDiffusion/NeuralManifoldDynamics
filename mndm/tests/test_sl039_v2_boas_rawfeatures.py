from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "project/new_mnps_v.3.2/science_lead/sl039_v2_boas_rawfeatures_pilot.py"
SPEC = importlib.util.spec_from_file_location("sl039_v2_boas_rawfeatures_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_raw_v2_metrics_are_emitted_independently_and_proxy_is_separate() -> None:
    frame = pd.DataFrame({name: [float(index + 1)] for index, name in enumerate(MODULE.ALL_RAW_METRICS)})

    result = MODULE.raw_measurements_from_frame(frame)

    assert all(result[name]["computed"] and result[name]["valid"] for name in MODULE.REQUIRED_METRICS)
    assert result["eeg_beta"]["role"] == "ratio_supporting_bandpower"
    proxy = result[MODULE.EEG_PROXY_METRIC]
    assert proxy["valid"] is True
    assert proxy["role"] == "separate_eeg_high_frequency_spectral_proxy; not PPG"
    assert proxy["validity_scope"] == "finite numeric estimator output only; estimator quality not independently assessed"
    assert "ppg" not in result
    assert "embodied_arousal_proxy" not in result


def test_missing_raw_metric_does_not_suppress_other_computable_metrics() -> None:
    columns = {name: [1.0] for name in MODULE.ALL_RAW_METRICS if name != "eeg_permutation_entropy"}
    result = MODULE.raw_measurements_from_frame(pd.DataFrame(columns))

    assert result["eeg_permutation_entropy"]["computed"] is False
    assert result["eeg_permutation_entropy"]["valid"] is False
    assert result["eeg_permutation_entropy"]["value"] is None
    assert result["eeg_permutation_entropy"]["reason"] == "missing_single_window_feature_output"
    assert result["eeg_delta"]["valid"] is True


def test_extractor_failure_marks_each_raw_metric_unsupported_without_zero_fill() -> None:
    result = MODULE.raw_measurements_from_frame(None, "synthetic_missing_input")

    assert set(result) == set(MODULE.ALL_RAW_METRICS)
    assert all(not metric["computed"] and not metric["valid"] for metric in result.values())
    assert all(metric["value"] is None for metric in result.values())
    assert all(metric["reason"] == "synthetic_missing_input" for metric in result.values())


def test_entropy_provenance_uses_reported_metric_and_marks_missing_metadata_unknown() -> None:
    frame = pd.DataFrame({
        "eeg_entropy_construct": ["energetic_complexity"],
        "eeg_entropy_metric": ["spectral_entropy"],
        "eeg_entropy_backend": ["scipy_welch"],
        "eeg_entropy_degraded_mode": [True],
        "eeg_entropy_reason": ["permutation_entropy_failed"],
    })
    resolved = MODULE.entropy_provenance_from_frame(frame)
    unknown = MODULE.entropy_provenance_from_frame(pd.DataFrame({"eeg_permutation_entropy": [0.4]}))

    assert resolved["metric"] == "spectral_entropy"
    assert resolved["backend"] == "scipy_welch"
    assert resolved["degraded_mode"] is True
    assert resolved["metadata_status"] == "available"
    assert unknown["metric"] is None
    assert unknown["metadata_status"] == "unknown_or_incomplete"

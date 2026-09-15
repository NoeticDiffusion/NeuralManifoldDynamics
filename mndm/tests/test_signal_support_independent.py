"""Independent checks for signal-support provenance and bounded export.

These tests exercise the measurement contract at its boundaries.  They do not
reimplement the provenance builder or infer support from epoch timestamps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from mndm.features.ppg import compute_ppg_features
from mndm import parallel
from mndm.pipeline.signal_support_export import build_signal_support_export
from mndm.signal_support_provenance import (
    build_signal_support_provenance,
    refresh_source_hash,
)


def test_nominal_epoch_bounds_do_not_certify_raw_support() -> None:
    frame = pd.DataFrame(
        {
            "epoch_id": [0],
            "file": ["recording.edf"],
            "t_start": [30.0],
            "t_end": [60.0],
        }
    )
    exported = build_signal_support_export(
        frame,
        [
            {
                "raw_file": "recording.edf",
                "temporal_support": {
                    "status": "unknown",
                    "reason": "nonlocal_filter_and_unrecorded_sample_map",
                },
            }
        ],
    )
    extent = exported["per_epoch_input_extent"][0]
    assert extent["nominal_time_start_sec"] == 30.0
    assert extent["nominal_time_end_sec"] == 60.0
    assert extent["original_sample_start"] is None
    assert extent["original_sample_end"] is None
    assert extent["sample_mapping_status"] == "unknown"
    assert extent["support_status"] == "unknown"
    assert exported["temporal_support_status"] == "unknown"


def test_nonfinite_export_metadata_is_null_and_json_safe() -> None:
    frame = pd.DataFrame(
        {
            "epoch_id": [1],
            "file": ["recording.edf"],
            "t_start": [math.nan],
            "t_end": [math.inf],
        }
    )
    exported = build_signal_support_export(
        frame,
        [],
        feature_baselines={
            "feature": {
                "standardization_center": math.nan,
                "standardization_scale": math.inf,
            }
        },
    )
    extent = exported["per_epoch_input_extent"][0]
    assert extent["nominal_time_start_sec"] is None
    assert extent["nominal_time_end_sec"] is None
    fit = exported["reference_fit_population"]["features"]["feature"]
    assert fit["center"] is None
    assert fit["scale"] is None
    json.dumps(exported, allow_nan=False)


def test_source_hash_change_is_explicitly_detected(tmp_path: Path) -> None:
    source = tmp_path / "recording.edf"
    source.write_bytes(b"first source bytes")
    provenance = build_signal_support_provenance(source, input_sfreq=256.0)
    initial_hash = provenance["source"]["sha256"]

    source.write_bytes(b"changed source bytes")
    refresh_source_hash(provenance)

    assert provenance["source"]["sha256_after"] != initial_hash
    assert provenance["source"]["hash_stable"] is False


def test_missing_source_remains_unknown_after_final_hash_check(tmp_path: Path) -> None:
    source = tmp_path / "not_available.edf"
    provenance = build_signal_support_provenance(source, input_sfreq=256.0)
    refresh_source_hash(provenance)

    assert provenance["source"]["exists"] is False
    assert provenance["source"]["hash_status"] == "missing"
    assert provenance["source"]["sha256_after"] is None
    assert provenance["source"]["hash_stable"] is False
    assert provenance["temporal_support"]["status"] == "unknown"
    json.dumps(provenance, allow_nan=False)


def test_ppg_global_adaptive_dependency_is_recorded_and_not_finite_support() -> None:
    sfreq = 50.0
    time = np.arange(6000, dtype=float) / sfreq
    pulse = np.sin(2.0 * np.pi * 1.2 * time) + 0.05 * np.sin(2.0 * np.pi * 0.2 * time)
    result = compute_ppg_features(
        {
            "signals": {"ppg": pulse[None, :]},
            "sfreq": sfreq,
            "dataset_id": "bounded-reference",
            "meta": {},
        },
        {
            "epoching": {"length_s": 30.0, "step_s": 30.0},
            "features": {"ppg": {"bandpass_low_hz": 0.4, "bandpass_high_hz": 8.0}},
        },
    )
    provenance = result.attrs["signal_support_provenance"]
    operations = {item["name"]: item for item in provenance["operations"]}
    assert operations["ppg_bandpass_filtfilt"]["status"] in {"applied", "fallback"}
    peak = operations["ppg_peak_detection"]
    assert peak["parameters"]["threshold_fit_scope"] == "whole_stream_median_and_MAD"
    assert peak["result"]["global_threshold"] is True
    assert provenance["temporal_support"]["status"] == "unknown"


def test_actual_feature_path_preserves_modality_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    pulse_time = np.arange(3000, dtype=float) / 50.0
    pulse = np.sin(2.0 * np.pi * 1.2 * pulse_time)
    config = {
        "epoching": {"length_s": 30.0, "step_s": 30.0},
        "features": {"ppg": {"bandpass_low_hz": 0.4, "bandpass_high_hz": 8.0}},
    }
    monkeypatch.setattr(parallel, "_get_modality_handlers", lambda: {"ppg": compute_ppg_features})
    preprocessed = SimpleNamespace(
        signals={"ppg": pulse[None, :]},
        sfreq=50.0,
        channels={"ppg": ["PPG"]},
        meta={"dataset_id": "bounded-reference", "file": "bounded.edf"},
    )

    merged, _timings = parallel._compute_features_from_preprocessed(
        preprocessed, config, file_path_for_features="bounded.edf"
    )
    assert merged is not None
    merged_provenance = merged.attrs.get("signal_support_provenance")
    assert isinstance(merged_provenance, dict)
    assert any(op.get("name") == "ppg_peak_detection" for op in merged_provenance.get("operations", []))

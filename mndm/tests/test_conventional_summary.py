"""Unit tests for conventional EEG comparator summary provenance."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.conventional_summary import compute_conventional_eeg_summary


def _sub_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eeg_conventional_tier1_relative_delta": [0.1, 0.2, 0.15],
            "eeg_conventional_tier1_relative_theta": [0.2, 0.25, 0.22],
        }
    )


def _config() -> dict:
    return {"conventional_eeg": {"enabled": True, "packs": ["tier1"]}}


def test_conventional_eeg_summary_returns_none_when_disabled():
    result = compute_conventional_eeg_summary(
        sub_frame=_sub_frame(), config={"conventional_eeg": {"enabled": False}}, dataset_id="ds"
    )
    assert result is None


def test_conventional_eeg_summary_stamps_not_assessed_by_default():
    """No artifact_qc_applied argument -> explicit not_assessed, not a silent pass.

    Regression guard for ingest_jacobian_fidelity_handover_2.md items 9/12:
    the extension is still written (not withheld) when artifact QC evidence
    is absent, but must say so explicitly rather than implying an artifact
    detector was consulted.
    """
    result = compute_conventional_eeg_summary(sub_frame=_sub_frame(), config=_config(), dataset_id="ds")
    assert result is not None
    assert result["artifact_qc"] == {"status": "not_assessed", "reason": "no_artifact_qc_sidecar_evidence"}


def test_conventional_eeg_summary_stamps_not_confirmed_when_method_did_not_run():
    result = compute_conventional_eeg_summary(
        sub_frame=_sub_frame(), config=_config(), dataset_id="ds", artifact_qc_applied=False
    )
    assert result["artifact_qc"]["status"] == "not_confirmed"


def test_conventional_eeg_summary_stamps_confirmed_when_method_ran():
    result = compute_conventional_eeg_summary(
        sub_frame=_sub_frame(), config=_config(), dataset_id="ds", artifact_qc_applied=True
    )
    assert result["artifact_qc"]["status"] == "confirmed_applied"
    # The extension itself is never withheld by this flag.
    assert result["column_count"] == 2


def test_conventional_eeg_export_keeps_row_level_technical_and_proxy_status():
    frame = pd.DataFrame({
        "file": ["recording.hea"], "epoch_id": [7], "t_start": [4.0], "t_end": [12.0],
        "eeg_conventional_coma_suppression_ratio": [np.nan],
        "eeg_conventional_coma_amplitude_rms": [np.nan],
        "technical_eeg_status": ["valid"],
        "technical_eeg_reason": [None],
        "technical_eeg_clipping_status": ["not_assessed"],
        "technical_eeg_clipping_reason": ["verified_adc_rails_unavailable"],
        "coma_proxy_suppression_status": ["not_testable"],
        "coma_proxy_suppression_reason": ["amplitude_unit_unverified"],
        "coma_proxy_unit_status": ["unknown"],
        "coma_proxy_amplitude_unit": ["unknown"],
        "coma_proxy_reference_status": ["unknown"],
        "coma_proxy_reference_mode": ["unknown"],
        "coma_proxy_amplitude_status": ["not_testable"],
        "coma_proxy_amplitude_reason": ["amplitude_unit_unverified;amplitude_reference_unverified"],
    })
    result = compute_conventional_eeg_summary(
        sub_frame=frame,
        config={"conventional_eeg": {"enabled": True, "packs": ["coma"]}},
        dataset_id="physionet_icare_2_1",
    )
    assert result["schema_version"] == "mndm.conventional_eeg.v2"
    row = result["row_level_status"]["rows"][0]
    assert row["epoch_id"] == 7
    assert row["technical_eeg_status"] == "valid"
    assert row["technical_eeg_clipping_status"] == "not_assessed"
    assert row["coma_proxy_suppression_status"] == "not_testable"
    assert row["coma_proxy_suppression_reason"] == "amplitude_unit_unverified"
    definition = result["row_level_status"]["measure_definitions"]["coma_suppression_ratio"]
    assert definition["parameters"]["amplitude_floor"] == 0.01
    assert "not an ACNS adjudication" in definition["clinical_status"]
    assert result["row_level_status"]["measure_definitions"]["coma_amplitude_rms"]["status_field"] == "coma_proxy_amplitude_status"


def test_coma_summary_excludes_technically_suspect_windows_but_preserves_row_values():
    frame = pd.DataFrame({
        "file": ["valid.hea", "flatline.hea"],
        "epoch_id": [1, 2],
        "t_start": [0.0, 4.0],
        "t_end": [8.0, 12.0],
        "eeg_conventional_coma_suppression_ratio": [0.2, 1.0],
        "technical_eeg_status": ["valid", "suspect"],
        "technical_eeg_channel_status": ["computed", "computed"],
        "technical_eeg_reason": [None, "all_eeg_channels_exactly_flatlined"],
        "coma_proxy_suppression_status": ["computed", "computed"],
    })
    result = compute_conventional_eeg_summary(
        sub_frame=frame,
        config={"conventional_eeg": {"enabled": True, "packs": ["coma"]}},
        dataset_id="physionet_icare_2_1",
    )
    suppression = result["families"]["coma"]["suppression_ratio"]
    assert suppression["mean"] == 0.2
    assert result["families"]["coma"]["technical_eligibility"]["included_row_count"] == 1
    assert result["families"]["coma"]["technical_eligibility"]["excluded_row_count"] == 1
    exported_rows = result["row_level_status"]["rows"]
    assert exported_rows[1]["eeg_conventional_coma_suppression_ratio"] == 1.0
    assert exported_rows[1]["technical_eeg_status"] == "suspect"

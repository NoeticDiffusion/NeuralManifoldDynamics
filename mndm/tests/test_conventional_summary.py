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

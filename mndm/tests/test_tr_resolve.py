"""P0.3 fMRI TR resolver: BIDS JSON, NIfTI zooms, config fallback, fail-closed mismatch."""

from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.tr_resolve import (
    FmriTRNotTestable,
    TR_SOURCE_BIDS_JSON,
    TR_SOURCE_CONFIG_FALLBACK,
    TR_SOURCE_NIFTI_ZOOMS,
    bold_json_sidecar_path,
    fmri_tr_attrs_from_frame,
    resolve_fmri_tr,
)


def _write_sidecar(path: Path, repetition_time: float | None, extra: dict | None = None) -> Path:
    payload = dict(extra or {})
    if repetition_time is not None:
        payload["RepetitionTime"] = repetition_time
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_bold_json_sidecar_path_strips_nii_gz(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    assert bold_json_sidecar_path(bold) == tmp_path / "sub-001_task-rest_bold.json"


def test_resolve_fmri_tr_prefers_matching_bids_json(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    _write_sidecar(bold_json_sidecar_path(bold), 2.0)
    resolved = resolve_fmri_tr(bold, nifti_zooms=(2.0, 2.0, 2.0, 2.0))
    assert resolved.tr_source == TR_SOURCE_BIDS_JSON
    assert resolved.tr_sec == pytest.approx(2.0)
    assert resolved.sfreq == pytest.approx(0.5)
    assert resolved.nifti_tr_sec == pytest.approx(2.0)


def test_resolve_fmri_tr_json_vs_header_mismatch_is_not_testable(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    _write_sidecar(bold_json_sidecar_path(bold), 2.0)
    with pytest.raises(FmriTRNotTestable, match="NOT_TESTABLE"):
        resolve_fmri_tr(bold, nifti_zooms=(2.0, 2.0, 2.0, 1.0))


def test_resolve_fmri_tr_within_atol_uses_bids_json(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    _write_sidecar(bold_json_sidecar_path(bold), 2.0)
    resolved = resolve_fmri_tr(bold, nifti_zooms=(2.0, 2.0, 2.0, 2.0004))
    assert resolved.tr_source == TR_SOURCE_BIDS_JSON
    assert resolved.tr_sec == pytest.approx(2.0)


def test_resolve_fmri_tr_json_without_zooms(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    _write_sidecar(bold_json_sidecar_path(bold), 2.0)
    resolved = resolve_fmri_tr(bold, nifti_zooms=None)
    assert resolved.tr_source == TR_SOURCE_BIDS_JSON
    assert resolved.tr_sec == pytest.approx(2.0)


def test_resolve_fmri_tr_uses_nifti_zooms_without_sidecar(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    resolved = resolve_fmri_tr(bold, nifti_zooms=(2.0, 2.0, 2.0, 2.5))
    assert resolved.tr_source == TR_SOURCE_NIFTI_ZOOMS
    assert resolved.tr_sec == pytest.approx(2.5)
    assert resolved.sfreq == pytest.approx(0.4)


def test_resolve_fmri_tr_null_repetition_time_falls_through_to_zooms(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    bold_json_sidecar_path(bold).write_text(
        json.dumps({"RepetitionTime": None, "Method": "lightweight_norm_v1"}),
        encoding="utf-8",
    )
    resolved = resolve_fmri_tr(bold, nifti_zooms=(2.0, 2.0, 2.0, 2.0))
    assert resolved.tr_source == TR_SOURCE_NIFTI_ZOOMS
    assert resolved.tr_sec == pytest.approx(2.0)
    assert resolved.bids_tr_sec is None


def test_resolve_fmri_tr_uses_config_fallback_when_header_invalid(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    resolved = resolve_fmri_tr(
        bold,
        config={"preprocess": {"fallback_tr": 2.0}},
        nifti_zooms=(2.0, 2.0, 2.0, 0.0),
    )
    assert resolved.tr_source == TR_SOURCE_CONFIG_FALLBACK
    assert resolved.tr_sec == pytest.approx(2.0)


def test_resolve_fmri_tr_missing_all_sources_is_not_testable(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    with pytest.raises(FmriTRNotTestable, match="NOT_TESTABLE"):
        resolve_fmri_tr(bold, nifti_zooms=(2.0, 2.0, 2.0))


def test_resolve_fmri_tr_does_not_guess_one_second(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    with pytest.raises(FmriTRNotTestable, match="NOT_TESTABLE"):
        resolve_fmri_tr(bold, config={"preprocess": {}}, nifti_zooms=None)


def test_resolve_fmri_tr_corrupt_sidecar_is_not_testable(tmp_path: Path):
    bold = tmp_path / "sub-001_task-rest_bold.nii.gz"
    bold_json_sidecar_path(bold).write_text("{not-json", encoding="utf-8")
    with pytest.raises(FmriTRNotTestable, match="NOT_TESTABLE"):
        resolve_fmri_tr(bold, nifti_zooms=(2.0, 2.0, 2.0, 2.0))


def test_fmri_tr_attrs_from_frame_empty_for_eeg_like_table():
    frame = pd.DataFrame({"feat_a": [1.0, 2.0]})
    assert fmri_tr_attrs_from_frame(frame) == {}


def test_fmri_tr_attrs_from_frame_copies_resolved_fields():
    frame = pd.DataFrame(
        {
            "fmri_tr_sec": [2.0, 2.0],
            "fmri_sfreq": [0.5, 0.5],
            "fmri_tr_source": ["bids_json", "bids_json"],
        }
    )
    attrs = fmri_tr_attrs_from_frame(frame)
    assert attrs["tr_sec"] == pytest.approx(2.0)
    assert attrs["sfreq"] == pytest.approx(0.5)
    assert attrs["tr_source"] == "bids_json"

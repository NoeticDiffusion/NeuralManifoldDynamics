"""P0 fMRI preprocess contract on a synthetic NIfTI.

Unit coverage of nuisance, filter, TR, and cache identity also lives in
``test_fmri_nuisance.py``, ``test_fmri_filter_stage.py``,
``test_tr_resolve.py``, and ``test_fmri_p1_cache_parcellate.py``.
This module runs those contracts through ``preprocess_fmri``.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("nibabel")
import nibabel as nib  # noqa: E402

from mndm.features.fmri_continuous import FILTER_STAGE_DEFERRED
from mndm.pipeline.fmri_roi_cache import clear_atlas_cache
from mndm.pipeline.nuisance import NUISANCE_DISABLED
from mndm.pipeline.spatial_align import FmriAtlasSpaceNotTestable, SPACE_MATCHED, SPACE_RESAMPLED
from mndm.pipeline.tr_resolve import FmriTRNotTestable
from mndm.preprocess import preprocess_fmri

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0]).astype(float)
TR_SEC = 2.0


def _atlas_and_bold():
    atlas = np.zeros((4, 4, 4), dtype=np.int16)
    atlas[:2, :2, :] = 1
    atlas[2:, :2, :] = 2
    atlas[:2, 2:, :] = 3
    atlas[2:, 2:, :] = 4
    n_t = 12
    bold = np.zeros((4, 4, 4, n_t), dtype=np.float32)
    t = np.arange(n_t, dtype=np.float32)
    for lab in (1, 2, 3, 4):
        bold[atlas == lab, :] = lab + 0.01 * t
    return atlas, bold


def _write_nifti(path: Path, data: np.ndarray, *, affine=None, zooms=None) -> Path:
    img = nib.Nifti1Image(np.asarray(data), AFFINE if affine is None else affine)
    if zooms is None:
        zooms = (2.0, 2.0, 2.0, TR_SEC) if data.ndim == 4 else (2.0, 2.0, 2.0)
    img.header.set_zooms(zooms[: data.ndim])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def _write_sidecar(bold: Path, payload: dict) -> Path:
    side = bold.with_name(bold.name.replace(".nii.gz", ".json"))
    side.write_text(json.dumps(payload), encoding="utf-8")
    return side


def _config(tmp_path: Path, atlas: Path, **fmri_extra) -> dict:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    fmri = {
        "atlas_path": str(atlas),
        "min_regions_required": 1,
        "bandpass": [0.01, 0.1],
    }
    fmri.update(fmri_extra)
    return {
        "datasets": ["ds999999"],
        "paths": {"processed_dir": str(processed), "received_dir": str(tmp_path)},
        "preprocess": {"fallback_tr": TR_SEC, "fmri": fmri},
    }


def _expected_means(n_t: int = 12) -> np.ndarray:
    t = np.arange(n_t, dtype=np.float32)
    return np.stack([lab + 0.01 * t for lab in (1, 2, 3, 4)]).astype(np.float32)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_atlas_cache()
    yield
    clear_atlas_cache()


def test_synthetic_nifti_records_p0_provenance(tmp_path: Path):
    atlas, bold = _atlas_and_bold()
    ds = tmp_path / "ds999999"
    atlas_path = _write_nifti(ds / "atlas.nii.gz", atlas)
    bold_path = _write_nifti(ds / "sub-001_task-rest_bold.nii.gz", bold)
    _write_sidecar(bold_path, {"RepetitionTime": TR_SEC})
    out = preprocess_fmri(bold_path, _config(tmp_path, atlas_path))
    np.testing.assert_array_equal(out.signals["fmri"], _expected_means())
    assert out.meta["tr_source"] == "bids_json"
    assert out.meta["tr_sec"] == pytest.approx(TR_SEC)
    assert out.sfreq == pytest.approx(1.0 / TR_SEC)
    assert out.meta["filter_applied"] is False
    assert out.meta["filter_stage"] == FILTER_STAGE_DEFERRED
    assert out.meta["filter_ignored_preprocess_bandpass"] == [0.01, 0.1]
    assert out.meta["nuisance_status"] == NUISANCE_DISABLED
    assert out.meta["atlas_space_status"] == SPACE_MATCHED
    assert out.meta["atlas_affine_match"] == 1
    assert out.meta["atlas_affine_atol_mm"] == pytest.approx(1e-3)
    assert len(out.meta["roi_ts_cache_key"]) == 64
    assert out.meta["roi_ts_cache_hit"] == 0


def test_null_repetition_time_falls_through_to_zooms(tmp_path: Path):
    atlas, bold = _atlas_and_bold()
    ds = tmp_path / "ds999999"
    atlas_path = _write_nifti(ds / "atlas.nii.gz", atlas)
    bold_path = _write_nifti(ds / "sub-001_task-rest_bold.nii.gz", bold)
    _write_sidecar(bold_path, {"RepetitionTime": None})
    out = preprocess_fmri(bold_path, _config(tmp_path, atlas_path))
    assert out.meta["tr_source"] == "nifti_zooms"
    assert out.meta["tr_sec"] == pytest.approx(TR_SEC)
    assert out.meta["tr_bids_json_sec"] is None


def test_non_null_invalid_tr_is_not_testable(tmp_path: Path):
    atlas, bold = _atlas_and_bold()
    ds = tmp_path / "ds999999"
    atlas_path = _write_nifti(ds / "atlas.nii.gz", atlas)
    bold_path = _write_nifti(ds / "sub-001_task-rest_bold.nii.gz", bold)
    _write_sidecar(bold_path, {"RepetitionTime": -1})
    with pytest.raises(FmriTRNotTestable, match="NOT_TESTABLE"):
        preprocess_fmri(bold_path, _config(tmp_path, atlas_path))


def test_json_zoom_mismatch_is_not_testable(tmp_path: Path):
    atlas, bold = _atlas_and_bold()
    ds = tmp_path / "ds999999"
    atlas_path = _write_nifti(ds / "atlas.nii.gz", atlas)
    bold_path = _write_nifti(ds / "sub-001_task-rest_bold.nii.gz", bold)
    _write_sidecar(bold_path, {"RepetitionTime": TR_SEC + 0.05})
    with pytest.raises(FmriTRNotTestable, match="NOT_TESTABLE"):
        preprocess_fmri(bold_path, _config(tmp_path, atlas_path))


def test_same_shape_affine_mismatch_is_not_testable(tmp_path: Path):
    atlas, bold = _atlas_and_bold()
    shifted = AFFINE.copy()
    shifted[0, 3] = 10.0
    ds = tmp_path / "ds999999"
    atlas_path = _write_nifti(ds / "atlas.nii.gz", atlas, affine=shifted)
    bold_path = _write_nifti(ds / "sub-001_task-rest_bold.nii.gz", bold)
    _write_sidecar(bold_path, {"RepetitionTime": TR_SEC})
    with pytest.raises(FmriAtlasSpaceNotTestable, match="NOT_TESTABLE"):
        preprocess_fmri(bold_path, _config(tmp_path, atlas_path))


def test_resample_under_assume_same_space_keeps_integer_labels(tmp_path: Path):
    atlas, bold = _atlas_and_bold()
    shifted = AFFINE.copy()
    shifted[0, 3] = 0.1
    ds = tmp_path / "ds999999"
    atlas_path = _write_nifti(ds / "atlas.nii.gz", atlas, affine=shifted)
    bold_path = _write_nifti(ds / "sub-001_task-rest_bold.nii.gz", bold)
    _write_sidecar(bold_path, {"RepetitionTime": TR_SEC})
    out = preprocess_fmri(
        bold_path,
        _config(
            tmp_path,
            atlas_path,
            resample_atlas_to_bold=True,
            assume_same_space=True,
        ),
    )
    assert out.meta["atlas_space_status"] == SPACE_RESAMPLED
    assert out.meta["atlas_affine_match"] == 1
    assert out.signals["fmri"].shape[0] == 4
    np.testing.assert_array_equal(out.signals["fmri"], _expected_means())


def test_basename_collision_changes_cache_hash(tmp_path: Path):
    atlas, bold_a = _atlas_and_bold()
    bold_b = bold_a + 5.0
    atlas_path = _write_nifti(tmp_path / "ds999999" / "atlas.nii.gz", atlas)
    a_path = _write_nifti(tmp_path / "ds999999" / "runA" / "sub-001_task-rest_bold.nii.gz", bold_a)
    b_path = _write_nifti(tmp_path / "ds999999" / "runB" / "sub-001_task-rest_bold.nii.gz", bold_b)
    _write_sidecar(a_path, {"RepetitionTime": TR_SEC})
    _write_sidecar(b_path, {"RepetitionTime": TR_SEC})
    cfg = _config(tmp_path, atlas_path)
    out_a = preprocess_fmri(a_path, cfg)
    out_b = preprocess_fmri(b_path, cfg)
    assert a_path.name == b_path.name
    assert out_a.meta["roi_ts_cache_key"] != out_b.meta["roi_ts_cache_key"]
    assert not np.allclose(out_a.signals["fmri"], out_b.signals["fmri"])
    again = preprocess_fmri(a_path, cfg)
    assert again.meta["roi_ts_cache_hit"] == 1
    assert again.meta["roi_ts_cache_key"] == out_a.meta["roi_ts_cache_key"]

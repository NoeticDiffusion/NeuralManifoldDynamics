"""P0.5 fMRI nuisance status: explicit enum, no silent truncate/zero-fill, fail-closed when enabled."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.nuisance import (
    NUISANCE_APPLIED,
    NUISANCE_DISABLED,
    NUISANCE_FAILED_CLEAN,
    NUISANCE_SKIPPED_MISSING_FILE,
    NUISANCE_SKIPPED_NO_COLUMNS,
    NUISANCE_TRUNCATED_LENGTH,
    FmriNuisanceNotTestable,
    apply_fmri_nuisance_regression,
    confounds_path_for_bold,
    fmri_nuisance_attrs_from_frame,
    resolve_fmri_nuisance,
)

_COLUMNS = ["trans_x", "trans_y"]
_SUFFIX = "_desc-confoundsextended_timeseries.tsv"


def _bold(tmp_path: Path) -> Path:
    return tmp_path / "sub-001_task-rest_bold.nii.gz"


def _write_tsv(path: Path, n_times: int, columns=_COLUMNS, values=None) -> Path:
    data = {}
    for col in columns:
        if values is not None and col in values:
            data[col] = values[col]
        else:
            data[col] = np.arange(n_times, dtype=float)
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    return path


def _enabled_cfg(columns=None) -> dict:
    return {
        "enabled": True,
        "confounds_suffix": _SUFFIX,
        "columns": list(columns or _COLUMNS),
    }


def _fake_clean(signals, confounds=None, detrend=True, standardize=False, t_r=None):
    return np.asarray(signals, dtype=float) + 1.0


def test_confounds_path_strips_bold_nii_gz(tmp_path: Path):
    bold = _bold(tmp_path)
    assert confounds_path_for_bold(bold, _SUFFIX) == tmp_path / (
        "sub-001_task-rest_desc-confoundsextended_timeseries.tsv"
    )


def test_disabled_when_cfg_absent():
    array = np.ones((3, 10), dtype=float)
    out, meta = resolve_fmri_nuisance(array, "unused.nii.gz", {}, tr=2.0)
    assert meta["nuisance_status"] == NUISANCE_DISABLED
    assert np.array_equal(out, array)


def test_disabled_when_enabled_false(tmp_path: Path):
    array = np.ones((3, 10), dtype=float)
    out, meta = resolve_fmri_nuisance(
        array,
        _bold(tmp_path),
        {"nuisance_regression": {"enabled": False}},
        tr=2.0,
    )
    assert meta["nuisance_status"] == NUISANCE_DISABLED
    assert np.array_equal(out, array)


def test_missing_tsv_status_and_fail_closed(tmp_path: Path):
    array = np.ones((2, 8), dtype=float)
    bold = _bold(tmp_path)
    result = apply_fmri_nuisance_regression(array, bold, _enabled_cfg(), tr=2.0)
    assert result.status == NUISANCE_SKIPPED_MISSING_FILE
    assert np.array_equal(result.region_array, array)
    with pytest.raises(FmriNuisanceNotTestable, match="NOT_TESTABLE"):
        resolve_fmri_nuisance(
            array, bold, {"nuisance_regression": _enabled_cfg()}, tr=2.0
        )


def test_length_mismatch_does_not_truncate(tmp_path: Path):
    array = np.ones((2, 10), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    _write_tsv(conf, n_times=7)
    result = apply_fmri_nuisance_regression(array, bold, _enabled_cfg(), tr=2.0)
    assert result.status == NUISANCE_TRUNCATED_LENGTH
    assert result.region_array.shape == (2, 10)
    assert np.array_equal(result.region_array, array)
    with pytest.raises(FmriNuisanceNotTestable, match="truncated_length"):
        resolve_fmri_nuisance(
            array, bold, {"nuisance_regression": _enabled_cfg()}, tr=2.0
        )


def test_nan_column_is_failed_clean_not_zero_filled(tmp_path: Path):
    array = np.ones((2, 6), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    _write_tsv(
        conf,
        n_times=6,
        values={"trans_x": np.full(6, np.nan), "trans_y": np.arange(6, dtype=float)},
    )
    result = apply_fmri_nuisance_regression(array, bold, _enabled_cfg(), tr=2.0)
    assert result.status == NUISANCE_FAILED_CLEAN
    assert np.array_equal(result.region_array, array)
    with pytest.raises(FmriNuisanceNotTestable, match="failed_clean"):
        resolve_fmri_nuisance(
            array,
            bold,
            {"nuisance_regression": _enabled_cfg()},
            tr=2.0,
            clean_fn=_fake_clean,
        )


def test_no_usable_columns(tmp_path: Path):
    array = np.ones((2, 5), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    _write_tsv(conf, n_times=5, columns=["unrelated"])
    result = apply_fmri_nuisance_regression(array, bold, _enabled_cfg(), tr=2.0)
    assert result.status == NUISANCE_SKIPPED_NO_COLUMNS
    assert np.array_equal(result.region_array, array)


def test_applied_with_injected_clean(tmp_path: Path):
    array = np.ones((2, 8), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    _write_tsv(conf, n_times=8)
    out, meta = resolve_fmri_nuisance(
        array,
        bold,
        {"nuisance_regression": _enabled_cfg()},
        tr=2.0,
        clean_fn=_fake_clean,
    )
    assert meta["nuisance_status"] == NUISANCE_APPLIED
    assert meta["nuisance_columns_used"] == _COLUMNS
    assert np.allclose(out, array + 1.0)


def test_edge_nan_is_filled_and_applied(tmp_path: Path):
    array = np.ones((2, 5), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    trans_x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
    _write_tsv(conf, n_times=5, values={"trans_x": trans_x, "trans_y": np.arange(5.0)})
    out, meta = resolve_fmri_nuisance(
        array,
        bold,
        {"nuisance_regression": _enabled_cfg()},
        tr=2.0,
        clean_fn=_fake_clean,
    )
    assert meta["nuisance_status"] == NUISANCE_APPLIED
    assert np.allclose(out, array + 1.0)


def test_internal_nan_is_failed_clean_not_interpolated(tmp_path: Path):
    array = np.ones((2, 5), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    trans_x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    _write_tsv(conf, n_times=5, values={"trans_x": trans_x, "trans_y": np.arange(5.0)})
    result = apply_fmri_nuisance_regression(array, bold, _enabled_cfg(), tr=2.0)
    assert result.status == NUISANCE_FAILED_CLEAN
    assert np.array_equal(result.region_array, array)


def test_trailing_nan_is_edge_filled_and_applied(tmp_path: Path):
    array = np.ones((2, 5), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    trans_x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    _write_tsv(conf, n_times=5, values={"trans_x": trans_x, "trans_y": np.arange(5.0)})
    out, meta = resolve_fmri_nuisance(
        array,
        bold,
        {"nuisance_regression": _enabled_cfg()},
        tr=2.0,
        clean_fn=_fake_clean,
    )
    assert meta["nuisance_status"] == NUISANCE_APPLIED
    assert np.allclose(out, array + 1.0)


def test_non_numeric_confounds_are_failed_clean(tmp_path: Path):
    array = np.ones((2, 4), dtype=float)
    bold = _bold(tmp_path)
    conf = confounds_path_for_bold(bold, _SUFFIX)
    conf.write_text("trans_x\ttrans_y\nn/a\t1\nbad\t2\nn/a\t3\nbad\t4\n", encoding="utf-8")
    result = apply_fmri_nuisance_regression(array, bold, _enabled_cfg(), tr=2.0)
    assert result.status == NUISANCE_FAILED_CLEAN
    assert np.array_equal(result.region_array, array)


def test_fmri_nuisance_attrs_from_frame_and_eeg_empty():
    assert fmri_nuisance_attrs_from_frame(pd.DataFrame({"feat_a": [1.0]})) == {}
    frame = pd.DataFrame({"fmri_nuisance_status": ["applied", "applied"]})
    assert fmri_nuisance_attrs_from_frame(frame) == {"nuisance_status": "applied"}
    mixed = pd.DataFrame({"fmri_nuisance_status": ["applied", "disabled"]})
    assert fmri_nuisance_attrs_from_frame(mixed) == {}
    unknown = pd.DataFrame({"fmri_nuisance_status": ["partial", "partial"]})
    assert fmri_nuisance_attrs_from_frame(unknown) == {}

"""Tests for fail-closed ds003645 sensor-topography HDF5 lineage."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.tools.ds003645_sensor_topo import (
    load_simultaneous_feature_rows,
    require_exact_window_match,
    required_feature_columns,
)


def _write_h5(path: Path, *, row_order: np.ndarray | None = None, include_row_source: bool = True) -> None:
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)
    has_eeg = np.array([1, 1, 0], dtype=np.int8)
    has_meg = np.array([1, 0, 1], dtype=np.int8)
    if row_order is not None:
        values, has_eeg, has_meg = values[row_order], has_eeg[row_order], has_meg[row_order]
    with h5py.File(path, "w") as h5:
        group = h5.create_group("features_raw")
        group.create_dataset("values", data=values)
        group.create_dataset("names", data=np.asarray(["eeg_alpha", "meg_alpha"], dtype=h5py.string_dtype()))
        h5.create_dataset("epoch_id", data=np.arange(3))
        h5.create_dataset("window_start", data=np.arange(3, dtype=float))
        h5.create_dataset("window_end", data=np.arange(1, 4, dtype=float))
        if include_row_source:
            provenance = h5.create_group("row_source")
            provenance.create_dataset("has_eeg", data=has_eeg)
            provenance.create_dataset("has_meg", data=has_meg)
            provenance.create_dataset("raw_file", data=np.asarray(["x.fif", "y.set", "x.fif"], dtype=h5py.string_dtype()))
            provenance.create_dataset(
                "source_format",
                data=np.asarray(["neuromag_fif", "eeglab_set", "neuromag_fif"], dtype=h5py.string_dtype()),
            )


def test_load_simultaneous_rows_uses_provenance_not_half_position(tmp_path: Path):
    path = tmp_path / "rows.h5"
    _write_h5(path, row_order=np.array([2, 1, 0]))

    rows = load_simultaneous_feature_rows(path)

    assert len(rows.frame) == 1
    assert rows.frame["eeg_alpha"].iloc[0] == 1.0
    assert rows.frame["meg_alpha"].iloc[0] == 10.0


def test_load_simultaneous_rows_fails_closed_without_provenance(tmp_path: Path):
    path = tmp_path / "missing.h5"
    _write_h5(path, include_row_source=False)

    with pytest.raises(ValueError, match="row_source"):
        load_simultaneous_feature_rows(path)


def test_exact_window_match_rejects_positional_mismatch():
    base = pd.DataFrame(
        {
            "raw_file": ["x.fif", "x.fif"],
            "source_format": ["neuromag_fif", "neuromag_fif"],
            "epoch_id": [0, 1],
            "window_start": [0.0, 4.0],
            "window_end": [8.0, 12.0],
        }
    )
    altered = base.copy()
    altered.loc[1, "window_end"] = 13.0

    with pytest.raises(ValueError, match="do not align"):
        require_exact_window_match(base, altered, subject="012", run="1")


def test_required_feature_columns_requires_every_frozen_group():
    features = ["eeg_alpha", "eeg_alpha__g_eeg_left_anterior"]
    with pytest.raises(ValueError, match="Missing frozen"):
        required_feature_columns(
            features,
            modality="eeg",
            sector_names=("left_anterior", "right_posterior"),
            weighted_features=("eeg_alpha",),
        )

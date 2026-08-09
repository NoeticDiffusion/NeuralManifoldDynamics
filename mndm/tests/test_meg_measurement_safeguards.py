"""Tests for reusable MEG measurement safeguards."""

from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm import projection
from mndm.provenance.h5_lineage import load_simultaneous_feature_rows
from mndm.tools.meg_transform_replay import validate_meg_transform_contract
from mndm.tools.sensor_topography_qc import run_sensor_topography_qc


def _config():
    return {
        "mnps_projection": {
            "feature_standardization": {
                "meg_alpha": ["log10", "robust_z", "clip"],
                "meg_permutation_entropy": ["robust_z", "clip"],
            }
        },
        "meg_ensembles": {"groups": {"left": ["MEG0111"], "right": ["MEG0121"]}},
        "sensor_topography_qc": {"enabled": True},
    }


def _write_meg_h5(path: Path) -> None:
    names = ["meg_alpha", "meg_alpha__g_left", "meg_alpha__g_right", "meg_permutation_entropy"]
    raw = np.array([[1, 2, 4, .1], [2, 4, 8, .2], [4, 8, 16, .4], [8, 16, 32, .8]], dtype=np.float32)
    cfg = _config()
    z = np.column_stack(
        [projection._apply_column_pipeline(raw[:, i], projection._resolve_feature_pipeline(name.split("__g_")[0], cfg["mnps_projection"]["feature_standardization"])) for i, name in enumerate(names)]
    )
    with h5py.File(path, "w") as h5:
        for surface, values in (("features_raw", raw), ("features_projection_z", z)):
            grp = h5.create_group(surface)
            grp.create_dataset("values", data=values)
            grp.create_dataset("names", data=np.asarray(names, dtype=h5py.string_dtype()))
        h5.create_dataset("window_start", data=np.arange(4, dtype=float))
        h5.create_dataset("window_end", data=np.arange(1, 5, dtype=float))
        source = h5.create_group("row_source")
        source.create_dataset("has_eeg", data=np.ones(4, dtype=np.int8))
        source.create_dataset("has_meg", data=np.ones(4, dtype=np.int8))
        source.create_dataset("raw_file", data=np.asarray(["x.fif"] * 4, dtype=h5py.string_dtype()))
        source.create_dataset("source_format", data=np.asarray(["neuromag_fif"] * 4, dtype=h5py.string_dtype()))


def test_meg_transform_contract_replays_grouped_and_global_columns():
    raw = np.array([[1, 2, .1], [2, 4, .2], [4, 8, .4]], dtype=np.float32)
    names = ["meg_alpha", "meg_alpha__g_left", "meg_permutation_entropy"]
    cfg = _config()
    exported = np.column_stack(
        [projection._apply_column_pipeline(raw[:, i], projection._resolve_feature_pipeline(name.split("__g_")[0], cfg["mnps_projection"]["feature_standardization"])) for i, name in enumerate(names)]
    )
    report = validate_meg_transform_contract(raw, names, exported, cfg)
    assert report["status"] == "ok"
    assert report["n_grouped_features"] == 1


def test_generic_lineage_and_opt_in_qc(tmp_path: Path):
    path = tmp_path / "meg.h5"
    _write_meg_h5(path)
    assert len(load_simultaneous_feature_rows(path).frame) == 4
    report = run_sensor_topography_qc(path, _config(), dataset_id="test")
    assert report["status"] == "ok"
    assert report["frozen_group_coverage"] == {"left": 1, "right": 1}

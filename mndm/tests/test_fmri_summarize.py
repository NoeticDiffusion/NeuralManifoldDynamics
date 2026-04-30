"""Lightweight integration test for fMRI MNPS summarization."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mne")


def test_cmd_summarize_with_fmri_features(tmp_path, require_real_h5py):
    """cmd_summarize should run on a dataset with only fMRI-derived features."""
    from mndm import orchestrate

    ds_id = "dsFMRI"
    processed_dir = tmp_path

    # Minimal config needed by cmd_summarize
    config = {
        "paths": {
            "received_dir": str(tmp_path),  # no participants.tsv needed for this test
            "processed_dir": str(processed_dir),
        },
        "mnps_projection": {
            "normalize": None,
            "weights": {
                "m": {"fmri_entropy_global": 1.0},
                "d": {"fmri_modularity": 1.0},
                "e": {"fmri_entropy_global": 1.0},
            },
        },
        "mnps": {
            "fs_out": 1.0,
            "window_sec": 30.0,
            "overlap": 0.0,
            "derivative": {"method": "central", "window": 3, "polyorder": 1},
            "knn": {"k": 5, "metric": "euclidean"},
            "ridge": {"alpha": 1.0, "cv_folds": 3, "distance_weighted": True},
            "whiten": True,
            "super_window": 3,
            "embodied": {"enabled": False, "channels": []},
            "surrogates": {"enabled": False},
            "reliability": {"enabled": False},
            "stage_codebook": {},
        },
        "robustness": {
            "coverage": {"min_seconds": 0, "min_epochs": 0, "min_trials": 0},
        },
    }

    # Create a fake features.csv with fMRI features only
    ds_path = processed_dir / ds_id
    ds_path.mkdir(parents=True, exist_ok=True)
    features_df = pd.DataFrame(
        {
            "fmri_entropy_global": np.linspace(0.1, 0.3, 5),
            "fmri_modularity": np.linspace(0.4, 0.6, 5),
        }
    )
    (ds_path / "features.csv").write_text(features_df.to_csv(index=False), encoding="utf-8")

    rc = orchestrate.cmd_summarize(config, [ds_id], out_dir=processed_dir, subject=None)
    assert rc == 0

    # Check that an MNPS directory and HDF5 file were created
    mnps_dirs = list(ds_path.glob(f"neuralmanifolddynamics_{ds_id}_*"))
    assert mnps_dirs
    # h5 files are in subject subdirectories (e.g., mnps_*/sub-unknown_ses-unknown/*.h5)
    h5_files = list(mnps_dirs[0].glob("**/*.h5"))
    assert h5_files



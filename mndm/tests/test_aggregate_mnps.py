"""Tests for aggregate_mnps utility."""

from pathlib import Path
import sys
import json

import h5py
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _write_summary(path: Path, subject: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": "dsX",
        "subject": subject,
        "session": "ses-01",
        "samples": 12,
        "meta_indices": {"mean_trace": 0.1, "mean_rotation_fro": 0.2, "windows": 9},
        "participant_meta": {"Group": "Control"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_h5(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["x_definition"] = "direct_mde_v1"
        f.attrs["weights_hash_direct"] = "abc123"
        f.create_dataset("mnps_3d", data=np.array([[1.0, 2.0, 3.0], [np.nan, 4.0, 5.0]], dtype=np.float32))
        grp = f.create_group("coords_9d")
        grp.create_dataset("values", data=np.array([[0.1, np.nan], [0.2, 0.4]], dtype=np.float32))
        grp.create_dataset("names", data=np.array([b"m_a", b"e_s"], dtype="S3"))


def test_aggregate_uses_latest_run_and_matches_summary_h5(tmp_path: Path):
    from mndm.tools.aggregate_mnps import aggregate

    processed = tmp_path / "processed"
    ds = processed / "dsX"
    old_run = ds / "mnps_dsX_old"
    new_run = ds / "mnps_dsX_new"

    # Old run (should be ignored by default latest-run mode).
    _write_summary(old_run / "sub-01" / "summary.json", "sub-01")
    _write_h5(old_run / "sub-01" / "summary.h5")

    # New run.
    _write_summary(new_run / "sub-02" / "summary.json", "sub-02")
    _write_h5(new_run / "sub-02" / "summary.h5")

    # Ensure mtime order makes new_run latest.
    old_ts = old_run.stat().st_mtime - 100.0
    for p in (old_run, old_run / "sub-01", old_run / "sub-01" / "summary.json", old_run / "sub-01" / "summary.h5"):
        p.touch()
    for p in (new_run, new_run / "sub-02", new_run / "sub-02" / "summary.json", new_run / "sub-02" / "summary.h5"):
        p.touch()
    (old_run / "sub-01" / "summary.json").touch()
    # set old run dir older explicitly
    import os
    os.utime(old_run, (old_ts, old_ts))

    out_csv = ds / "mnps_aggregate.csv"
    rc = aggregate(processed, "dsX", out_csv)
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.iloc[0]["subject"] == "sub-02"
    assert "x_definition" in df.columns
    assert "v2_m_a_nan_frac" in df.columns


def test_aggregate_skips_corrupt_summary_and_continues(tmp_path: Path):
    from mndm.tools.aggregate_mnps import aggregate

    processed = tmp_path / "processed"
    ds = processed / "dsX"
    run = ds / "mnps_dsX_run"
    good = run / "sub-good"
    bad = run / "sub-bad"
    good.mkdir(parents=True, exist_ok=True)
    bad.mkdir(parents=True, exist_ok=True)

    _write_summary(good / "summary.json", "sub-good")
    _write_h5(good / "summary.h5")
    (bad / "summary.json").write_text("{not valid json", encoding="utf-8")
    _write_h5(bad / "summary.h5")

    out_csv = ds / "mnps_aggregate.csv"
    rc = aggregate(processed, "dsX", out_csv, run_dir=run)
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.iloc[0]["subject"] == "sub-good"


"""Structural probe coverage for nested continuous ephys series."""

from pathlib import Path

import h5py
import numpy as np

from dandi_ingest.contracts import AssetRecord
from dandi_ingest.probe import probe_local_asset


def test_probe_reports_nested_lfp_series_and_units_schema(tmp_path: Path) -> None:
    relative = "sub-1/session.nwb"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        lfp = handle.create_group("processing/ecephys/LFP")
        lfp.create_dataset("data", data=np.zeros((10, 4)))
        start = lfp.create_dataset("starting_time", data=0.0)
        start.attrs["rate"] = 1250.0
        lfp.create_dataset("electrodes", data=np.asarray([0, 1, 2, 3]))
        units = handle.create_group("units")
        units.create_dataset("id", data=np.asarray([1, 2]))
        units.create_dataset("spike_times_index", data=np.asarray([2, 4]))

    summary = probe_local_asset(
        AssetRecord(dandiset_id="test", version="draft", identifier="id", path=relative),
        raw_root=tmp_path,
    )
    series = summary.metadata["electrical_series"]
    assert series[0]["path"] == "/processing/ecephys/LFP"
    assert series[0]["rate_hz"] == 1250.0
    assert summary.metadata["units_summary"]["row_count"] == 2

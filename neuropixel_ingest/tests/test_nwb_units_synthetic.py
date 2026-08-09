"""Synthetic NWB coverage for schema-tolerant Units extraction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


def test_read_session_bundle_and_quality_filter(tmp_path: Path) -> None:
    pynwb = pytest.importorskip("pynwb")
    from neuropixel_ingest.contracts import EphysReadConfig
    from neuropixel_ingest.nwb_units import read_session_bundle

    path = tmp_path / "units.nwb"
    _write_units_nwb(pynwb, path)

    bundle = read_session_bundle(path, EphysReadConfig(quality_policy="good"), dataset_id="dandi_000006")
    assert len(bundle.units) == 1
    assert bundle.units["unit_id"].tolist() == [1]
    assert np.allclose(bundle.spike_times()[0], [0.1, 0.4, 0.9])
    assert len(bundle.trials) == 1
    assert bundle.trials["start_time"].iloc[0] == 0.0
    assert bundle.events["sample"].tolist() == [0.2]


def _write_units_nwb(pynwb, path: Path) -> None:
    nwbfile = pynwb.NWBFile(
        session_description="synthetic ephys session",
        identifier="synthetic-units",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.add_unit_column("quality", "sorting quality")
    nwbfile.add_unit_column("firing_rate", "reported rate")
    nwbfile.add_unit(id=1, spike_times=[0.1, 0.4, 0.9], quality="good", firing_rate=3.0)
    nwbfile.add_unit(id=2, spike_times=[0.2, 0.3], quality="mua", firing_rate=2.0)
    nwbfile.add_trial_column("cue_start_time", "sample cue time")
    nwbfile.add_trial(start_time=0.0, stop_time=1.0, cue_start_time=0.2)
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

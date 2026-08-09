"""Integration coverage for NWB Units -> mndm ephys feature path."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[2]))


def test_preprocess_nwb_units_and_compute_features(tmp_path: Path) -> None:
    pynwb = pytest.importorskip("pynwb")
    from mndm.parallel import process_single_file
    from mndm.preprocess import preprocess_nwb

    path = tmp_path / "sub-M1_ses-S1_task-delay_ecephys.nwb"
    _write_nwb(pynwb, path)
    config = {
        "datasets": ["synthetic_ephys"],
        "preprocess": {
            "nwb": {
                "units": {
                    "enabled": True,
                    "rate_bin_sec": 0.05,
                    "smoothing_sigma_sec": None,
                    "quality_policy": "all",
                }
            }
        },
        "epoching": {"length_s": 0.5, "step_s": 0.25},
    }

    preprocessed = preprocess_nwb(path, config)
    assert preprocessed.signals["ephys"].shape[0] == 2
    assert preprocessed.sfreq == 20.0
    assert preprocessed.meta["nwb_signal_kind"] == "units_rate"
    assert preprocessed.meta["nwb_unit_qc"]["n_units_raw"] == 2
    assert preprocessed.meta["nwb_unit_qc"]["n_units_retained"] == 2

    result = process_single_file(path, config)
    assert result.success
    assert result.features_df is not None
    assert "ephys_mean_rate_hz" in result.features_df


def _write_nwb(pynwb, path: Path) -> None:
    nwbfile = pynwb.NWBFile(
        session_description="synthetic ephys",
        identifier="test-ephys",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.add_unit(id=1, spike_times=[0.05, 0.15, 0.55, 0.95])
    nwbfile.add_unit(id=2, spike_times=[0.1, 0.3, 0.8])
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

"""Tests for DANDI/NWB ingest integration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[2]))


def test_dandi_ingest_config_and_missing_probe(tmp_path: Path):
    from dandi_ingest.config import resolve_dandi_config
    from dandi_ingest.contracts import AssetRecord
    from dandi_ingest.probe import probe_local_asset
    from dandi_ingest.registry import known_adapters

    cfg_path = tmp_path / "dandi.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  config_id: test_000718",
                "  adapter: dataset_000718",
                "  dandiset_id: '000718'",
                "storage:",
                f"  output_root: {tmp_path.as_posix()}/out",
                f"  raw_root: {tmp_path.as_posix()}/raw",
                "selection:",
                "  asset_limit: 1",
            ]
        ),
        encoding="utf-8",
    )

    config, _ = resolve_dandi_config(cfg_path)
    assert config.dataset.dandiset_id == "000718"
    assert "dataset_000718" in known_adapters()

    record = AssetRecord(
        dandiset_id="000718",
        version="draft",
        identifier="missing",
        path="sub-1/sub-1_ecephys.nwb",
    )
    summary = probe_local_asset(record, raw_root=config.storage.raw_root)
    assert summary.exists is False
    assert summary.error == "file_not_found"


def test_build_file_index_includes_nwb_metadata(tmp_path: Path, require_real_h5py):
    import h5py

    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "dandi_000718"
    nwb_path = dataset_root / "sub-M1" / "sub-M1_ses-S1_task-rest_ecephys.nwb"
    nwb_path.parent.mkdir(parents=True)
    with h5py.File(nwb_path, "w") as handle:
        handle.create_group("acquisition").create_group("ElectricalSeries")
        handle.create_group("processing").create_group("ecephys")
        handle.create_group("intervals").create_group("epochs")

    index = build_file_index(dataset_root, config={"datasets": ["dandi_000718"]}, dataset_id="dandi_000718")

    assert len(index) == 1
    row = index.iloc[0]
    assert row["modality"] == "nwb"
    assert row["subject"] == "M1"
    assert row["session"] == "S1"
    assert row["task"] == "rest"
    assert "ElectricalSeries" in str(row["nwb_acquisitions"])
    assert "ecephys" in str(row["nwb_modality_hints"])


def test_load_dandi_nwb_configs():
    from core.config_loader import load_config

    config_root = Path(__file__).resolve().parents[1] / "config"
    cfg = load_config(config_root / "config_ingest_dandi_000718.yaml")

    assert cfg["datasets"] == ["dandi_000718"]
    assert cfg["source"]["platform"] == "DANDI"
    assert cfg["source"]["format"] == "NWB"
    assert cfg["preprocess"]["nwb"]["enabled"] is True
    assert cfg["features"]["eeg_bands"]["theta"] == [4, 12]

    pilot_cfg = load_config(config_root / "config_ingest_dandi_000458.yaml")
    assert pilot_cfg["state_labels"]["enabled"] is True
    assert pilot_cfg["state_labels"]["candidate_intervals"] == ["epochs", "trials"]
    assert pilot_cfg["state_labels"]["include_custom_intervals"] is True
    assert pilot_cfg["within_run_labels"]["datasets"]["dandi_000458"]["enabled"] is True


def test_preprocess_nwb_and_process_single_file_smoke(tmp_path: Path):
    pynwb = pytest.importorskip("pynwb")
    pytest.importorskip("mne")

    import pandas as pd

    from mndm.parallel import process_single_file
    from mndm.preprocess import preprocess_nwb

    nwb_path = tmp_path / "sub-M1_ses-S1_task-rest_ecephys.nwb"
    _write_minimal_electrical_series_nwb(pynwb, nwb_path)

    config = {
        "datasets": ["dandi_test"],
        "preprocess": {
            "sfreq": 100,
            "sfreq_candidates": [100],
            "notch_hz": None,
            "eeg_bandpass": [1, 40],
            "reref": None,
            "nwb": {
                "enabled": True,
                "prefer_series_keywords": ["lfp", "electrical"],
                "channel_type": "eeg",
                "scale_to_microvolts": False,
            },
        },
        "epoching": {"length_s": 2.0, "step_s": 1.0},
        "state_labels": {
            "enabled": True,
            "source": "nwb_intervals",
            "output_name": "nwb_state",
            "candidate_intervals": ["epochs", "trials"],
            "include_custom_intervals": True,
            "label_candidates": ["state", "behavioral_epoch", "tags"],
            "codebook": {"awake": 0, "isoflurane": 1, "recovery": 2},
        },
        "features": {
            "eeg_psd": {"method": "welch", "fmin": 1.0, "fmax": 45.0},
            "eeg_bands": {
                "delta": [1, 4],
                "theta": [4, 8],
                "alpha": [8, 12],
                "beta": [13, 30],
                "gamma": [30, 45],
            },
            "ratios": {"alpha_theta": ["alpha", "theta"], "beta_alpha": ["beta", "alpha"]},
        },
        "robustness": {"ensembles": {"enabled": False}},
    }

    preprocessed = preprocess_nwb(nwb_path, config)
    assert preprocessed.meta["source_format"] == "NWB"
    assert preprocessed.meta["nwb_series_path"].endswith("LFP")
    assert preprocessed.signals["eeg"].shape[0] == 4

    result = process_single_file(nwb_path, config)
    assert result.success is True
    assert result.features_df is not None
    assert not result.features_df.empty
    assert "eeg_delta" in result.features_df.columns
    assert "nwb_state" in result.features_df.columns
    states = set(result.features_df["nwb_state"].dropna().astype(str))
    assert {"awake", "isoflurane", "recovery"}.issubset(states)
    stage_codes = set(pd.to_numeric(result.features_df["stage"], errors="coerce").dropna().astype(int))
    assert {0, 1, 2}.issubset(stage_codes)


def _write_minimal_electrical_series_nwb(pynwb, path: Path) -> None:
    from pynwb.ecephys import ElectricalSeries
    from pynwb.epoch import TimeIntervals

    nwbfile = pynwb.NWBFile(
        session_description="minimal nwb ephys",
        identifier="nwb-test",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        session_id="S1",
    )
    nwbfile.subject = pynwb.file.Subject(subject_id="M1", species="Mus musculus")
    device = nwbfile.create_device("probe")
    group = nwbfile.create_electrode_group(
        name="probe_group",
        description="synthetic probe",
        location="CA1",
        device=device,
    )
    for idx in range(4):
        nwbfile.add_electrode(id=idx, x=float(idx), y=0.0, z=0.0, imp=np.nan, location=f"CA1_{idx}", filtering="none", group=group)
    electrodes = nwbfile.create_electrode_table_region(region=[0, 1, 2, 3], description="all electrodes")

    sfreq = 100.0
    times = np.arange(0, 10.0, 1.0 / sfreq)
    data = np.column_stack(
        [
            np.sin(2 * np.pi * 3.0 * times),
            np.sin(2 * np.pi * 6.0 * times),
            np.sin(2 * np.pi * 10.0 * times),
            np.sin(2 * np.pi * 20.0 * times),
        ]
    )
    nwbfile.add_acquisition(ElectricalSeries(name="LFP", data=data, electrodes=electrodes, rate=sfreq))
    nwbfile.add_epoch_column(name="state", description="synthetic state label")
    nwbfile.add_epoch(start_time=0.0, stop_time=4.0, state="awake")
    nwbfile.add_epoch(start_time=4.0, stop_time=8.0, state="isoflurane")
    nwbfile.add_trial_column(name="behavioral_epoch", description="state label attached to point-like trials")
    nwbfile.add_trial(start_time=8.0, stop_time=8.01, behavioral_epoch="recovery")
    custom_states = TimeIntervals(name="sleep_states", description="custom state intervals")
    custom_states.add_column(name="state", description="synthetic custom state")
    custom_states.add_row(start_time=8.0, stop_time=10.0, state="recovery")
    nwbfile.add_time_intervals(custom_states)

    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

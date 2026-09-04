from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families import far_ext_002b_ds003670_signal_timebase as far_ext  # noqa: E402


def _write_config(repo_root: Path, *, valid: bool = True) -> Path:
    config = repo_root / "mndm" / "config" / "config_ingest_common_eeg.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        body = """
mnps:
  window_sec: 8.0
  overlap: 0.5
epoching:
  length_s: 8.0
  step_s: 4.0
"""
    else:
        body = """
mnps:
  window_sec: 4.0
  overlap: 0.0
epoching:
  length_s: 4.0
  step_s: 4.0
"""
    config.write_text(body, encoding="utf-8")
    return config


def _write_set(
    path: Path,
    *,
    rho: float,
    embedded_dtype: np.dtype = np.dtype("int16"),
    missing_fdt: bool = False,
) -> None:
    import scipy.io as sio

    path.parent.mkdir(parents=True, exist_ok=True)
    n_channels = 32
    n_samples = 140_000
    labels = np.empty(n_channels, dtype=object)
    for index in range(n_channels):
        labels[index] = {"labels": f"E{index + 1}"}
    native_events = np.empty(2, dtype=object)
    native_events[0] = {"type": "16", "latency": 60_001.0}
    native_events[1] = {"type": "32", "latency": 70_001.0}
    if missing_fdt:
        data: object = "recording.fdt"
    else:
        data = np.zeros((n_channels, n_samples), dtype=embedded_dtype)
    eeg = {
        "srate": 2000.0,
        "pnts": float(n_samples),
        "nbchan": float(n_channels),
        "xmin": 0.0,
        "data": data,
        "chanlocs": labels,
        "event": native_events,
    }
    sio.savemat(str(path), {"EEG": eeg}, do_compression=False)


def _write_bound_fixture(
    tmp_path: Path,
    *,
    config_valid: bool = True,
    float_data: bool = False,
    missing_fdt: bool = False,
) -> tuple[Path, Path, Path, Path]:
    metadata_root = tmp_path / "metadata" / "ds003670"
    eeg_root = tmp_path / "eeg" / "ds003670"
    repo_root = tmp_path / "repo"
    far002_path = tmp_path / "far_ext_002a.json"
    metadata_root.mkdir(parents=True)
    eeg_root.mkdir(parents=True)
    (metadata_root / "dataset_description.json").write_text("{}", encoding="utf-8")
    _write_config(repo_root, valid=config_valid)
    events = []
    for index, (subject, rho) in enumerate(
        (("001", 0.5), ("002", 0.5), ("003", 1.0), ("004", 1.0))
    ):
        relative_events = (
            f"sub-{subject}/ses-01/eeg/"
            f"sub-{subject}_ses-01_task-GXtESCTT_events.tsv"
        )
        relative_set = relative_events.replace("_events.tsv", "_eeg.set")
        relative_sidecar = relative_events.replace("_events.tsv", "_eeg.json")
        sidecar = metadata_root / Path(relative_sidecar)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps({"EEGChannelCount": 32}),
            encoding="utf-8",
        )
        set_path = eeg_root / Path(relative_set)
        _write_set(
            set_path,
            rho=rho,
            embedded_dtype=np.dtype("float32") if float_data else np.dtype("int16"),
            missing_fdt=missing_fdt,
        )
        events.append(
            {
                "subject": subject,
                "biological_unit": f"bio-{subject}",
                "experiment": 1,
                "session": "01",
                "block": 1,
                "trial_in_block": index + 1,
                "v": {
                    "target": "frontal",
                    "waveform": "sinusoidal",
                    "frequency_hz": 5,
                },
                "target": "frontal",
                "waveform": "sinusoidal",
                "frequency_hz": 5,
                "rho": rho,
                "rho_source": "publication_table_1",
                "rho_contrast_eligible": True,
                "trigger_start": 30.0,
                "trigger_max_end": 35.0,
                "ramp_down_end": 40.0,
                "next_stim_start": 200.0,
                "T_isolated": 160.0,
                "semantic_status": "PASS",
                "source_file": relative_events,
            }
        )
    far002_path.write_text(
        json.dumps(
            {
                "schema": "mndm.far_ext_002a_ds003670_semantic_join.v1",
                "protocol_id": "FAR-EXT-002A",
                "dataset_id": "ds003670",
                "source_root": str(metadata_root),
                "source_binding": {
                    "status": "PASS",
                    "signal_payloads": [],
                    "unexpected_files": [],
                },
                "global_status": "DS003670_LIMITED_CURVE_SEMANTICS_PASS",
                "events": events,
            }
        ),
        encoding="utf-8",
    )
    return metadata_root, eeg_root, repo_root, far002_path


def test_promoted_ledger_is_frozen_to_002a_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(far_ext, "PROMOTED_EVENT_COUNT", 1)
    payload = {
        "protocol_id": "FAR-EXT-002A",
        "global_status": "DS003670_LIMITED_CURVE_SEMANTICS_PASS",
        "events": [
            {
                "experiment": 1,
                "semantic_status": "PASS",
                "rho_contrast_eligible": True,
                "T_isolated": 9.0,
                "source_file": "sub-001/ses-01/eeg/a_events.tsv",
                "v": {"target": "frontal", "waveform": "DC", "frequency_hz": 0},
            },
            {
                "experiment": 1,
                "semantic_status": "PASS",
                "rho_contrast_eligible": True,
                "T_isolated": None,
                "source_file": "sub-001/ses-01/eeg/b_events.tsv",
                "v": {"target": "frontal", "waveform": "DC", "frequency_hz": 0},
            },
        ],
    }
    ledger = far_ext.promoted_event_ledger(payload)
    assert len(ledger) == 1
    assert ledger[0]["family_id"] == "frontal_dc_0hz"


def test_nmd_grid_uses_recording_origin_and_four_second_step() -> None:
    windows = far_ext._nmd_windows(
        trigger_start=30.0,
        t0=40.0,
        horizon=16.0,
    )
    assert windows["pre_starts"] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    assert windows["post_starts"] == [40.0, 44.0, 48.0]
    assert windows["post_count"] == 3


def test_audit_passes_integer_native_signal_with_common_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, repo_root, far002_path = _write_bound_fixture(tmp_path)
    monkeypatch.setattr(far_ext, "PROMOTED_EVENT_COUNT", 4)
    far002 = json.loads(far002_path.read_text(encoding="utf-8"))
    result = far_ext.audit_far_ext_002b(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002_payload=far002,
        repo_root=repo_root,
        far002_path=far002_path,
    )
    assert result["global_status"] == far_ext.SIGNAL_TIMEBASE_PASS
    assert result["promoted_event_count"] == 4
    family = result["family_ledger"][0]
    assert family["family_id"] == "frontal_sinusoidal_5hz"
    assert family["selected_post_horizon"] == 30.0
    assert family["nmd_post_window_count"] == 6
    assert family["subjects_per_rho"] == {
        "0.5": ["bio-001", "bio-002"],
        "1.0": ["bio-003", "bio-004"],
    }
    assert result["far_003b_authorized"] is False
    assert result["audit_scope"]["mnps_calculated"] is False
    assert result["audit_scope"]["far_calculated"] is False


def test_float_native_signal_without_rail_is_not_testable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, repo_root, far002_path = _write_bound_fixture(
        tmp_path,
        float_data=True,
    )
    monkeypatch.setattr(far_ext, "PROMOTED_EVENT_COUNT", 4)
    far002 = json.loads(far002_path.read_text(encoding="utf-8"))
    result = far_ext.audit_far_ext_002b(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002_payload=far002,
        repo_root=repo_root,
        far002_path=far002_path,
    )
    assert result["global_status"] == "NOT_TESTABLE"
    assert (
        result["family_ledger"][0]["eligibility_status"]
        == far_ext.POST_STIM_ARTIFACT_NOT_TESTABLE
    )


def test_nmd_mismatch_is_method_limited_after_native_qc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, repo_root, far002_path = _write_bound_fixture(
        tmp_path,
        config_valid=False,
    )
    monkeypatch.setattr(far_ext, "PROMOTED_EVENT_COUNT", 4)
    far002 = json.loads(far002_path.read_text(encoding="utf-8"))
    result = far_ext.audit_far_ext_002b(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002_payload=far002,
        repo_root=repo_root,
        far002_path=far002_path,
    )
    assert result["global_status"] == "METHOD_LIMITED"


def test_payload_binding_requires_declared_fdt(tmp_path: Path) -> None:
    metadata_root, eeg_root, repo_root, far002_path = _write_bound_fixture(
        tmp_path,
        missing_fdt=True,
    )
    del metadata_root, repo_root, far002_path
    binding = far_ext.bind_payload_root(
        eeg_root,
        ["sub-001/ses-01/eeg/sub-001_ses-01_task-GXtESCTT_eeg.set"],
    )
    assert binding["status"] == far_ext.SOURCE_BINDING_FAILED
    assert any(path.endswith("recording.fdt") for path in binding["missing"])

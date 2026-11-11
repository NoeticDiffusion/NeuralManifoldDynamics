"""Synthetic opt-in MNE source-grid integration checks."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import json

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1 import (
    CORE19_CHANNELS,
    SURFACE_GRID_MANIFEST_SCHEMA,
    run_e3_surface_grid_manifest,
)
from mndm.mnps.v3_1.source_grid import _read_manifest, _verify_mne_source_files


def _manifest(source_path: Path) -> dict[str, object]:
    labels = list(CORE19_CHANNELS)
    return {
        "schema": SURFACE_GRID_MANIFEST_SCHEMA,
        "manifest_id": "synthetic-surface-grid-v1",
        "source": {"path": source_path.name, "format": "fif"},
        "recording_identity": {
            "dataset_id": "synthetic",
            "dataset_version": "1",
            "subject_id": "sub-syn01",
            "session_id": "ses-01",
            "task_id": "rest",
            "run_id": "run-01",
        },
        "preprocessing": {
            "pipeline_id": "synthetic-mne-minphase-v1",
            "target_sfreq_hz": 100,
            "bandpass_hz": [10, 30],
            "resampler_id": "mndm.causal_polyphase_kaiser_beta5.v0",
            "filter_id": "mne.minimum_phase_firwin_hamming_constant.v0",
        },
        "windows": [
            {"window_id": "w10_30", "start_sec": 10, "end_sec": 30},
            {"window_id": "w32_52", "start_sec": 32, "end_sec": 52},
        ],
        "surfaces": [
            {
                "surface_id": "native",
                "kind": "native",
                "channel_names": "all_eeg",
                "reference_mode": "average_over_surface",
                "allow_partial_channels": True,
                "minimum_channels": 2,
            },
            {
                "surface_id": "core19",
                "kind": "core19",
                "channel_names": labels,
                "reference_mode": "average_over_surface",
            },
            {
                "surface_id": "study_common_synthetic",
                "kind": "study_common",
                "channel_names": labels,
                "reference_mode": "recorded",
                "reference_id": "synthetic-acquisition-reference",
                "reference_channel_names": labels,
            },
        ],
    }


class _FakeRaw:
    def __init__(self, data, channel_names, sfreq=100.0, annotations=()):
        self._data = np.asarray(data, dtype=np.float64)
        self.ch_names = list(channel_names)
        self.n_times = int(self._data.shape[1])
        self.first_samp = 0
        self._raw_lengths = [self.n_times]
        self.filenames = ()
        self.info = {
            "sfreq": sfreq,
            "bads": [],
            "ch_types": ["eeg"] * len(self.ch_names),
        }
        self.annotations = SimpleNamespace(
            onset=np.asarray([item[0] for item in annotations], dtype=float),
            duration=np.asarray([item[1] for item in annotations], dtype=float),
            description=np.asarray([item[2] for item in annotations], dtype=object),
        )

    def get_data(self, picks=None):
        return self._data if picks is None else self._data[np.asarray(picks, dtype=int)]

    def get_channel_types(self):
        return list(self.info["ch_types"])

    def filter(self, **kwargs):
        return self


def _install_fake_mne(monkeypatch, source_raw: _FakeRaw) -> None:
    import mndm.mnps.v3_1.source_grid as source_grid

    def pick_types(info, *, eeg, exclude):
        return [index for index, kind in enumerate(info["ch_types"]) if kind == "eeg"]

    def pick_info(info, picks, *, copy):
        return {
            "sfreq": info["sfreq"],
            "bads": [name for name in info.get("bads", []) if name in [source_raw.ch_names[i] for i in picks]],
            "ch_types": [info["ch_types"][index] for index in picks],
        }

    def raw_array(data, info, first_samp, verbose):
        names = [source_raw.ch_names[index] for index, kind in enumerate(source_raw.info["ch_types"]) if kind == "eeg"]
        return _FakeRaw(data, names, sfreq=info["sfreq"])

    fake_mne = ModuleType("mne")
    fake_mne.pick_types = pick_types
    fake_mne.pick_info = pick_info
    fake_mne.io = SimpleNamespace(RawArray=raw_array)
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setattr(source_grid, "_open_mne_raw", lambda path, source_format: source_raw)

    preprocess = ModuleType("mndm.preprocess")
    preprocess.__file__ = str(Path(__file__).resolve().parents[1] / "src" / "mndm" / "preprocess.py")
    preprocess._causal_polyphase_resample = lambda raw, target: (
        raw,
        {"up": 1, "down": 1, "tap_count": 1, "method": "synthetic_identity_resampler"},
    )
    preprocess._minimum_phase_eeg_filter_spec = lambda raw, picks, bandpass: {
        "method": "synthetic_filter_stub",
        "tap_count": 3,
        "bandpass_hz": list(bandpass),
    }
    monkeypatch.setitem(sys.modules, "mndm.preprocess", preprocess)


def test_manifest_reader_filters_source_and_emits_every_surface_window_row(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(21)
    labels = list(CORE19_CHANNELS) + ["Extra1"]
    raw = _FakeRaw(
        rng.normal(scale=1e-5, size=(len(labels), 6000)),
        labels,
        annotations=[(20.0, 0.0, "EDGE")],
    )
    source_path = tmp_path / "synthetic_raw.fif"
    source_path.write_bytes(b"synthetic-source-identity")
    raw.filenames = (str(source_path),)
    manifest_path = tmp_path / "surface_grid.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest(source_path), sort_keys=False), encoding="utf-8")
    _install_fake_mne(monkeypatch, raw)

    result = run_e3_surface_grid_manifest(manifest_path)
    document = result.to_dict()
    json.dumps(document, allow_nan=False)

    assert document["schema"] == "mndm.mnps_v3_1_surface_grid_run.v0"
    assert document["row_count"] == 6
    assert document["computed_count"] == 3
    assert document["unsupported_count"] == 3
    assert result.provenance["recording_identity"]["subject_id"] == "sub-syn01"
    assert result.provenance["reference_applied_before_measurement"] is False
    assert result.provenance["implementation_sha256"]["eeg_windows"]
    assert result.provenance["source_files"][str(source_path.resolve())]["sha256"]

    by_key = {(row["surface_id"], row["window_id"]): row["measurement"] for row in result.rows}
    for surface_id in ("native", "core19", "study_common_synthetic"):
        before_boundary = by_key[(surface_id, "w10_30")]
        after_boundary = by_key[(surface_id, "w32_52")]
        assert before_boundary["computed"] is False
        assert before_boundary["reason"] == "non_continuous_source"
        assert before_boundary["support"]["gap_free"] is False
        assert after_boundary["computed"] is True
        assert after_boundary["qualified"] is False
        assert after_boundary["details"]["source_sha256"] == result.provenance["source_files"][str(source_path.resolve())]["sha256"]
        assert after_boundary["details"]["window_provenance"]["source_dependency"]["source_sample_start"] >= 0


def test_brainvision_manifest_requires_associated_signal_and_marker_hashes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "synthetic.vhdr")
    manifest["source"] = {"path": "synthetic.vhdr", "format": "brainvision"}
    manifest_path = tmp_path / "invalid_brainvision.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "brainvision_manifest_must_hash_eeg_and_marker_files"):
        run_e3_surface_grid_manifest(manifest_path)


def test_brainvision_manifest_sidecars_must_match_header_references(tmp_path: Path) -> None:
    header = tmp_path / "synthetic.vhdr"
    data = tmp_path / "synthetic.eeg"
    marker = tmp_path / "synthetic.vmrk"
    header.write_text("Brain Vision Data Exchange Header File Version 1.0\nDataFile=synthetic.eeg\nMarkerFile=synthetic.vmrk\n", encoding="utf-8")
    data.write_bytes(b"eeg")
    marker.write_bytes(b"markers")
    manifest = _manifest(header)
    manifest["source"] = {
        "path": header.name,
        "format": "brainvision",
        "associated_files": [data.name, marker.name],
    }
    manifest_path = tmp_path / "brainvision.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    _read_manifest(manifest_path)
    header.write_text("Brain Vision Data Exchange Header File Version 1.0\nDataFile=other.eeg\nMarkerFile=synthetic.vmrk\n", encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "brainvision_manifest_sidecars_do_not_match_header"):
        _read_manifest(manifest_path)


def test_source_file_inventory_rejects_unhashed_split_files(tmp_path: Path) -> None:
    declared = tmp_path / "recording_raw.fif"
    split = tmp_path / "recording_raw-2.fif"
    declared.write_bytes(b"main")
    split.write_bytes(b"split")
    raw = SimpleNamespace(filenames=(str(declared), str(split)))

    with np.testing.assert_raises_regex(ValueError, "surface_grid_mne_source_file_not_hashed"):
        _verify_mne_source_files(raw, (declared,))
    _verify_mne_source_files(raw, (declared, split))


def test_surface_manifest_rejects_coercive_boolean_and_list_values(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_raw.fif"
    manifest = _manifest(source)
    manifest["surfaces"][0]["allow_partial_channels"] = "false"
    manifest_path = tmp_path / "invalid_surface.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "surface_grid_allow_partial_channels_must_be_boolean"):
        _read_manifest(manifest_path)

    manifest["surfaces"][0]["allow_partial_channels"] = True
    manifest["surfaces"][1]["reference_channel_names"] = "C3"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "surface_grid_reference_channel_labels_invalid"):
        _read_manifest(manifest_path)


def test_manifest_accepts_small_study_common_but_requires_declared_minimum(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "synthetic_raw.fif")
    surface = manifest["surfaces"][2]
    surface["channel_names"] = list(CORE19_CHANNELS[:8])
    surface["reference_channel_names"] = list(CORE19_CHANNELS[:8])
    path = tmp_path / "small_study.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _read_manifest(path)
    surface["minimum_channels"] = 9
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "minimum_channels_exceeds_declared_labels"):
        _read_manifest(path)


def test_surface_manifest_rejects_contradictory_reference_and_invalid_core(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_raw.fif"
    manifest = _manifest(source)
    manifest["surfaces"][1]["reference_mode"] = "average_over_surface"
    manifest["surfaces"][1]["reference_id"] = "recorded-reference"
    manifest_path = tmp_path / "invalid_surface.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with np.testing.assert_raises_regex(
        ValueError, "surface_grid_average_reference_cannot_declare_recorded_reference"
    ):
        _read_manifest(manifest_path)

    manifest["surfaces"][1].pop("reference_id")
    manifest["surfaces"][1]["reference_mode"] = "recorded"
    manifest["surfaces"][1]["channel_names"][0] = "NotFp1"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "core19_surface_must_use_the_declared_core19_labels_in_order"):
        _read_manifest(manifest_path)

    manifest["surfaces"][1]["channel_names"][0] = "Fp1"
    manifest["surfaces"][1]["minimum_channels"] = 20
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "surface_grid_minimum_channels_exceeds_declared_labels"):
        _read_manifest(manifest_path)

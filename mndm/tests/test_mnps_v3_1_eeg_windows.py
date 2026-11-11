"""Synthetic audit controls for the bounded v3.1 EDF window path."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import mne
import yaml

from mndm.mnps.v3_1.eeg_windows import (
    E1_G2_CROP_SEC,
    E1_G2_NOMINAL_WINDOWS_SEC,
    _frozen_g2_grid_reason,
    _intersects,
    _prepare_from_raw,
    load_audited_edf_eeg_grid,
    load_audited_edf_eeg_window,
)


def _fixture(
    tmp_path: Path,
    *,
    annotations: mne.Annotations | None = None,
    window_start_sec: float = 100.0,
    window_end_sec: float = 120.0,
):
    sfreq = 1000.0
    n_times = 200_000
    time = np.arange(n_times, dtype=np.float64) / sfreq
    rng = np.random.default_rng(7103)
    data = np.vstack(
        [np.sin(2.0 * np.pi * (8.0 + idx) * time) + 0.05 * rng.normal(size=n_times) for idx in range(4)]
    )
    info = mne.create_info(["F3", "C3", "P3", "O1"], sfreq, ch_types=["eeg"] * 4)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if annotations is not None:
        raw.set_annotations(annotations)
    source = tmp_path / "source.edf"
    source.write_bytes(b"synthetic-only-source-identity")
    config = tmp_path / "pilot.yaml"
    config.write_text("synthetic: true\n", encoding="utf-8")
    positions = tmp_path / "positions.pos"
    positions.write_text("4\n" + "".join(f"0 {name} 0 0 0\n" for name in raw.ch_names), encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    positions_hash = hashlib.sha256(positions.read_bytes()).hexdigest()
    return _prepare_from_raw(
        raw,
        source_path=source,
        source_sha256=source_hash,
        config_path=config,
        config_sha256=config_hash,
        positions_path=positions,
        positions_sha256=positions_hash,
        channel_names=tuple(raw.ch_names),
        window_start_sec=window_start_sec,
        window_end_sec=window_end_sec,
        target_sfreq_hz=250.0,
        bandpass_hz=(1.0, 45.0),
        source_hash_stable=True,
        code_hashes={"synthetic_test": "a" * 64},
    )


def test_audited_window_records_full_fir_dependency_and_fixed_clock(tmp_path: Path) -> None:
    result = _fixture(tmp_path)

    assert result.status == "ready"
    assert result.signals is not None and result.signals.shape == (4, 5000)
    assert result.support is not None
    assert result.support.source_is_continuous is True
    assert result.support.timestamps_uniform is True
    assert result.support.gap_free is True
    assert result.support.bad_interval_free is True
    assert result.support.within_recording_bounds is True
    assert result.support.effective_filter_support_valid is True
    assert result.provenance["source_dependency"]["source_sample_start"] == 96_624
    assert result.provenance["source_dependency"]["source_sample_end_exclusive"] == 119_997
    assert result.provenance["source_dependency"]["resampler"]["coefficient_sha256"]
    assert result.provenance["source_dependency"]["eeg_bandpass"]["coefficient_sha256"]
    assert result.provenance["support_audit"]["source_annotation_audit"] == "complete_mne_raw_annotations"
    assert result.provenance["support_audit"]["source_annotations"] == []


def test_audited_window_rejects_bad_interval_reaching_back_into_window(tmp_path: Path) -> None:
    bad = mne.Annotations([99.0], [0.1], ["BAD_artifact"])
    result = _fixture(tmp_path, annotations=bad)

    assert result.status == "unsupported"
    assert result.reason == "bad_annotation_overlaps_effective_source_support"
    assert result.signals is None
    assert result.support is not None
    assert result.support.bad_interval_free is False
    assert result.provenance["source_annotations"][0]["description"] == "BAD_artifact"
    assert result.provenance["source_dependency"]["source_sample_start"] == 96_624
    assert result.provenance["source_dependency"]["source_sample_end_exclusive"] == 119_997


def test_audited_window_rejects_gap_boundary_in_filter_dependency(tmp_path: Path) -> None:
    boundary = mne.Annotations([99.0], [0.0], ["EDGE boundary"])
    result = _fixture(tmp_path, annotations=boundary)

    assert result.status == "unsupported"
    assert result.reason == "segment_boundary_overlaps_effective_source_support"
    assert result.signals is None
    assert result.support is not None
    assert result.support.source_is_continuous is False
    assert result.support.gap_free is False


def test_audited_window_rejects_filter_context_that_reaches_recording_edge(tmp_path: Path) -> None:
    result = _fixture(tmp_path, window_start_sec=0.0, window_end_sec=20.0)

    assert result.status == "unsupported"
    assert result.reason == "insufficient_effective_filter_edge_support"
    assert result.signals is None
    assert result.provenance["effective_source_sample_start"] < 0


def test_point_annotation_at_half_open_endpoint_does_not_reject_window(tmp_path: Path) -> None:
    ready = _fixture(tmp_path)
    dependency = ready.provenance["source_dependency"]
    start = float(dependency["source_time_start_sec"])
    end = float(dependency["source_time_end_sec"])
    assert _intersects({"start_sec": start, "end_sec": start}, start, end) is True
    assert _intersects({"start_sec": end, "end_sec": end}, start, end) is False

    at_end = _fixture(tmp_path, annotations=mne.Annotations([end], [0.0], ["BAD_artifact"]))
    at_start = _fixture(tmp_path, annotations=mne.Annotations([start], [0.0], ["BAD_artifact"]))

    assert at_end.status == "ready"
    assert at_end.signals is not None
    assert at_start.status == "unsupported"
    assert at_start.reason == "bad_annotation_overlaps_effective_source_support"
    assert at_start.signals is None


def _pilot_overlay(
    tmp_path: Path,
    *,
    channel_count: int,
    bandpass: tuple[float, float],
    label_count: int | None = None,
) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.edf"
    source.write_bytes(b"synthetic-only-source-identity")
    count = channel_count if label_count is None else label_count
    positions = tmp_path / "positions.pos"
    positions.write_text(
        f"{count}\n" + "".join(f"0 CH{index:03d} 0 0 0\n" for index in range(count)),
        encoding="utf-8",
    )
    overlay = {
        "schema": "mndm.mnps_v3_1_eeg_pilot.v0",
        "eeg_pilot": {
            "dataset_id": "ANPHY",
            "dataset_version": "unresolved",
            "subject_id": "EPCTL01",
            "session_id": "unresolved",
            "task_id": "unresolved",
            "run_id": "unresolved",
            "source_edf": str(source.resolve()),
            "positions_file": str(positions.resolve()),
            "channel_count": channel_count,
            "window_sec": [120.0, 140.0],
            "target_sfreq_hz": 250.0,
            "resampler_id": "mndm.causal_polyphase_kaiser_beta5.v0",
            "eeg_bandpass_hz": [bandpass[0], bandpass[1]],
            "filter_phase": "minimum",
            "filter_pad": "constant",
            "notch_hz": None,
            "rereference": "average_over_fixed_channel_set",
            "csd": False,
            "coordinates_read": False,
            "candidate_id": "e_irregularity_channel_median_pe_v0",
            "estimator_id": "mndm.mnps_v3_1.e1_channel_median_pe.v0",
            "embedding_order": 5,
            "delay_samples": 1,
            "normalize": True,
            "tie_policy": "stable_mergesort",
        },
    }
    config = tmp_path / "pilot.yaml"
    config.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    return source, positions, config


def test_loader_rejects_wrong_channel_count_without_a_numeric_value(tmp_path: Path) -> None:
    source, positions, config = _pilot_overlay(tmp_path, channel_count=4, bandpass=(1.0, 45.0))

    result = load_audited_edf_eeg_window(source, positions, config)

    assert result.status == "unsupported"
    assert result.reason == "e1_fixed_channel_set_must_contain_83_labels"
    assert result.signals is None


def test_loader_rejects_position_file_that_is_not_the_83_label_set(tmp_path: Path) -> None:
    source, positions, config = _pilot_overlay(
        tmp_path,
        channel_count=83,
        bandpass=(1.0, 45.0),
        label_count=4,
    )

    result = load_audited_edf_eeg_window(source, positions, config)

    assert result.status == "unsupported"
    assert result.reason == "e1_fixed_channel_set_must_contain_83_labels"
    assert result.signals is None


def test_loader_and_transform_reject_wrong_bandpass_without_a_numeric_value(tmp_path: Path) -> None:
    source, positions, config = _pilot_overlay(tmp_path, channel_count=83, bandpass=(1.0, 40.0))
    loaded = load_audited_edf_eeg_window(source, positions, config)
    info = mne.create_info(["F3", "C3", "P3", "O1"], 1000.0, ch_types=["eeg"] * 4)
    transformed = _prepare_from_raw(
        mne.io.RawArray(np.zeros((4, 200_000)), info, verbose="ERROR"),
        source_path=source,
        source_sha256="ab" * 32,
        config_path=config,
        config_sha256="cd" * 32,
        positions_path=positions,
        positions_sha256="ef" * 32,
        channel_names=("F3", "C3", "P3", "O1"),
        window_start_sec=100.0,
        window_end_sec=120.0,
        target_sfreq_hz=250.0,
        bandpass_hz=(1.0, 40.0),
        source_hash_stable=True,
        code_hashes={"synthetic_test": "a" * 64},
    )

    assert loaded.status == "unsupported"
    assert loaded.reason == "e1_bandpass_must_be_1_45_hz"
    assert loaded.signals is None
    extra = tmp_path / "extra"
    extra.mkdir()
    extra_source, extra_positions, extra_config = _pilot_overlay(extra, channel_count=83, bandpass=(1.0, 45.0))
    extra_overlay = yaml.safe_load(extra_config.read_text(encoding="utf-8"))
    extra_overlay["eeg_pilot"]["eeg_bandpass_hz"] = [1.0, 45.0, 50.0]
    extra_config.write_text(yaml.safe_dump(extra_overlay), encoding="utf-8")
    extra_loaded = load_audited_edf_eeg_window(extra_source, extra_positions, extra_config)
    assert extra_loaded.status == "unsupported"
    assert extra_loaded.reason == "e1_bandpass_must_be_1_45_hz"
    assert extra_loaded.signals is None
    assert transformed.status == "unsupported"
    assert transformed.reason == "e1_bandpass_must_be_1_45_hz"
    assert transformed.signals is None


def test_frozen_g2_grid_rejects_a_shifted_nominal_window(tmp_path: Path) -> None:
    assert _frozen_g2_grid_reason(
        crop_sec=E1_G2_CROP_SEC,
        nominal_windows_sec=E1_G2_NOMINAL_WINDOWS_SEC,
    ) is None
    assert len(E1_G2_NOMINAL_WINDOWS_SEC) == 15
    source, positions, config = _pilot_overlay(tmp_path, channel_count=83, bandpass=(1.0, 45.0))
    shifted = list(E1_G2_NOMINAL_WINDOWS_SEC)
    shifted[0] = (1.0, 21.0)
    overlay = yaml.safe_load(config.read_text(encoding="utf-8"))
    pilot = overlay.pop("eeg_pilot")
    pilot.pop("window_sec")
    pilot["grid_id"] = "mnps_v3_1_anphy_epctl01_e1_g2_20260929"
    pilot["decision_status"] = "SELECT_FOR_G2_TECHNICAL_EVALUATION"
    pilot["crop_sec"] = [E1_G2_CROP_SEC[0], E1_G2_CROP_SEC[1]]
    pilot["nominal_windows_sec"] = [list(bounds) for bounds in shifted]
    pilot["output_root"] = str(tmp_path / "out")
    config.write_text(
        yaml.safe_dump({"schema": "mndm.mnps_v3_1_eeg_grid.v0", "eeg_grid": pilot}),
        encoding="utf-8",
    )

    result = load_audited_edf_eeg_grid(source, positions, config)

    assert result.status == "unsupported"
    assert result.reason == "e1_g2_nominal_grid_mismatch"
    assert result.windows == ()

    pilot["decision_status"] = "QUALIFIED"
    config.write_text(
        yaml.safe_dump({"schema": "mndm.mnps_v3_1_eeg_grid.v0", "eeg_grid": pilot}),
        encoding="utf-8",
    )
    mismatched_status = load_audited_edf_eeg_grid(source, positions, config)
    assert mismatched_status.status == "unsupported"
    assert mismatched_status.reason == "e1_g2_decision_status_mismatch"
    assert mismatched_status.windows == ()

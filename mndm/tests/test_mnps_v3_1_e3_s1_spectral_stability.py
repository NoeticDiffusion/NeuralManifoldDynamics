"""Synthetic tests for the bounded E3 split-half stability runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "project"
    / "new_mnps_v.3.1"
    / "run_anphy_epctl01_e3_s1_spectral_stability.py"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "project"
    / "new_mnps_v.3.1"
    / "anphy_epctl01_e3_s1_spectral_stability_protocol_v2_20260929.yaml"
)
spec = importlib.util.spec_from_file_location("mnps_v31_e3_s1_spectral_stability", RUNNER_PATH)
assert spec is not None and spec.loader is not None
e3_s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e3_s1)


def test_spectrum_recovers_known_rank_three_and_normalizes_spectrum() -> None:
    rng = np.random.default_rng(31_905)
    signals = e3_s1._known_rank_three_matrix(rng, 5000)

    spectrum = e3_s1._spectrum(signals)

    assert spectrum["participation_ratio_from_spectrum"] == pytest.approx(3.0, abs=1e-9)
    assert sum(spectrum["normalized_eigenvalues_descending"]) == pytest.approx(1.0, abs=1e-12)
    assert spectrum["top_1_variance_mass"] >= spectrum["top_5_variance_mass"] / 5


def test_spectrum_distances_have_frozen_edge_values() -> None:
    p = {"normalized_eigenvalues_descending": [1.0] + [0.0] * 82}
    q = {"normalized_eigenvalues_descending": [0.0, 1.0] + [0.0] * 81}

    same = e3_s1._spectrum_distance(p, p)
    orthogonal = e3_s1._spectrum_distance(p, q)

    assert same["total_variation"] == pytest.approx(0.0)
    assert same["jensen_shannon_nats"] == pytest.approx(0.0)
    assert orthogonal["total_variation"] == pytest.approx(1.0)
    assert orthogonal["jensen_shannon_nats"] == pytest.approx(np.log(2.0))


def test_half_support_keeps_audited_quality_and_exact_interval() -> None:
    from mndm.mnps.v3_1 import MNPS31WindowSupport

    parent = MNPS31WindowSupport(
        window_start_sec=20.0,
        window_end_sec=40.0,
        sample_rate_hz=250.0,
        sample_count=5000,
        segment_id="edf:abc:segment0",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=True,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )

    half = e3_s1._half_support(parent, start_sec=30.0, sample_count=2500)

    assert half.window_start_sec == 30.0
    assert half.window_end_sec == 40.0
    assert half.sample_count == 2500
    assert half.source_is_continuous is True
    assert half.effective_filter_support_valid is True


def test_half_dependency_is_derived_from_parent_tap_provenance() -> None:
    from mndm.mnps.v3_1 import MNPS31WindowSupport

    support = MNPS31WindowSupport(
        window_start_sec=20.0,
        window_end_sec=40.0,
        sample_rate_hz=250.0,
        sample_count=5000,
        segment_id="edf:abc:segment0",
        source_is_continuous=True,
        timestamps_uniform=True,
        gap_free=True,
        bad_interval_free=True,
        within_recording_bounds=True,
        effective_filter_support_valid=True,
    )
    parent = {
        "source_sample_start": 38944,
        "source_sample_end_exclusive": 79993,
        "resampler": {"up": 1, "down": 8, "tap_count": 257},
        "eeg_bandpass": {"tap_count": 101},
    }
    audit = SimpleNamespace(
        support=support,
        provenance={
            "source_dependency": parent,
            "source": {"mne_sample_rate_hz": 2000.0, "mne_sample_count": 1_000_000},
        },
    )

    first = e3_s1._half_dependency(audit, start_sec=20.0, sample_count=2500)
    second = e3_s1._half_dependency(audit, start_sec=30.0, sample_count=2500)

    assert first["source_sample_start"] == parent["source_sample_start"]
    assert first["source_sample_end_exclusive"] == 59993
    assert second["source_sample_start"] == 58944
    assert second["source_sample_end_exclusive"] == parent["source_sample_end_exclusive"]


def test_synthetic_controls_check_rank_duplicate_flatline_and_nonfinite_cases() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["synthetic_controls"]["replicates_per_scenario"] = 1

    controls = e3_s1._synthetic_controls(protocol)

    assert controls["scenarios"]["known_rank_3"]["median_full_participation_ratio"] == pytest.approx(3.0, abs=1e-8)
    duplicate = controls["scenarios"]["one_exactly_duplicated_sensor"]
    assert duplicate["empirical_covariance_rank_values"] == [81]
    assert duplicate["median_full_participation_ratio"] <= 81.0 + 1e-8
    assert controls["scenarios"]["all_channels_flatline"]["unsupported_reason_counts"]
    assert controls["scenarios"]["one_nonfinite_sample"]["unsupported_reason_counts"]


def test_protocol_split_mutation_is_rejected_before_data_loading() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    e3_s1._validate_protocol(protocol)

    protocol["estimator"]["split"]["first_half_samples"] = [0, 2400]
    with pytest.raises(ValueError, match="split_contract_mismatch"):
        e3_s1._validate_protocol(protocol)

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["synthetic_controls"]["scenarios"][2]["expected_covariance_rank"] = 82
    with pytest.raises(ValueError, match="synthetic_rank_contract_mismatch"):
        e3_s1._validate_protocol(protocol)

    mutations = (
        (0, "expected_participation_ratio", 2.5, "synthetic_expected_behavior_mismatch"),
        (1, "generator", "different AR(1) generator", "synthetic_generator_contract_mismatch"),
        (3, "generator", "different flatline generator", "synthetic_generator_contract_mismatch"),
        (3, "expected_result", "computed", "synthetic_expected_behavior_mismatch"),
        (4, "expected_result", "computed", "synthetic_expected_behavior_mismatch"),
    )
    for scenario_index, key, value, reason in mutations:
        protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["synthetic_controls"]["scenarios"][scenario_index][key] = value
        with pytest.raises(ValueError, match=reason):
            e3_s1._validate_protocol(protocol)


def test_run_rejects_mutated_protocol_before_calling_grid_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["estimator"]["split"]["second_half_samples"] = [2500, 4999]
    altered = tmp_path / "altered_e3_protocol.yaml"
    altered.write_text(yaml.safe_dump(protocol), encoding="utf-8")

    def fail_if_grid_load_is_reached(*args: object, **kwargs: object) -> None:
        pytest.fail("grid loader must not run for a protocol mutation")

    monkeypatch.setattr(e3_s1, "load_audited_edf_eeg_grid", fail_if_grid_load_is_reached)
    with pytest.raises(ValueError, match="split_contract_mismatch"):
        e3_s1.run(altered)


def test_unsupported_e3_half_stays_null_with_reason_and_suppresses_stability() -> None:
    measurements = {
        "full_20s": {"computed": True, "value": 3.0},
        "first_10s": {"computed": True, "value": 3.0},
        "second_10s": {"computed": False, "value": None, "reason": "undefined_covariance_spectrum"},
    }
    spectra = {"full_20s": {"p": [1]}, "first_10s": {"p": [1]}, "second_10s": None}

    result = e3_s1._stability_descriptors(measurements, spectra)

    assert result["status"] == "unsupported"
    assert result["unsupported_cells"] == [{"cell": "second_10s", "reason": "undefined_covariance_spectrum"}]
    assert result["values"]["half_normalized_spectrum_total_variation"] is None

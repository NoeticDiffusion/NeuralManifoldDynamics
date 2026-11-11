"""Focused tests for the frozen SL-006 feasibility runner."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mndm" / "src"))

from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio  # noqa: E402
from mndm.mnps.v3_2.experiments.vector_feasibility_002 import (  # noqa: E402
    CHANNEL_COUNT,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_BASE_OVERLAY_SHA256,
    EXPECTED_SELECTED_IDS,
    EXPECTED_SELECTION_SHA256,
    SAMPLE_COUNT,
    WINDOWS_SEC,
    _make_window_row,
    _kernel_results,
    _source_failure_rows,
    covariance_connectivity_diagnostics,
    participant_summaries,
    loader_identity_checks,
    shared_time_permutation_control,
    validate_frozen_preregistration,
)


def test_covariance_diagnostics_reconstruct_raw_ed_identity_and_standardized_ed() -> None:
    rng = np.random.Generator(np.random.PCG64(701))
    signals = rng.standard_normal((7, 1200), dtype=np.float64)
    signals[0] *= 3.0
    labels = tuple(f"sensor-{index}" for index in range(signals.shape[0]))
    ed = compute_ed_raw_sensor_covariance_participation_ratio(signals, labels)

    result = covariance_connectivity_diagnostics(signals, raw_ed_value=ed.value, nie_value=None)

    assert result["status"] == "computed"
    values = result["values"]
    assert values["raw_covariance_ED_identity_absolute_residual"] == pytest.approx(0.0, abs=1e-12)
    assert 1.0 <= values["ED_R"] <= signals.shape[0]
    assert values["mean_absolute_correlation"] >= 0.0
    assert values["mean_squared_correlation"] >= 0.0
    assert values["nie_graph_path_increment"] is None


def test_common_time_permutation_preserves_covariance_ed_and_nie() -> None:
    rng = np.random.Generator(np.random.PCG64(702))
    signals = rng.standard_normal((8, 1500), dtype=np.float64)
    signals[1] = 0.7 * signals[0] + 0.3 * signals[1]
    labels = tuple(f"sensor-{index}" for index in range(signals.shape[0]))
    from mndm.mnps.v3_2.experiments.vector_feasibility_002 import _kernel_results

    kernels, _, _ = _kernel_results(signals, labels)

    result = shared_time_permutation_control(
        signals,
        labels,
        subject_id="EPCTL-test",
        window_id="window-test",
        original=kernels,
    )

    assert result["status"] == "passed"
    assert result["same_permutation_for_all_channels"] is True
    assert result["covariance_preserved"] is True
    assert result["correlation_preserved"] is True
    assert result["ED_preserved"] is True
    assert result["NIE_preserved"] is True
    assert result["metric_deltas_permuted_minus_original"]["OI"] is not None
    assert result["qualified"] is False


def test_nie_kernel_exception_does_not_discard_ed_or_oi(monkeypatch: pytest.MonkeyPatch) -> None:
    import mndm.mnps.v3_2.nie as nie_module

    signals = np.random.Generator(np.random.PCG64(7021)).standard_normal((8, 900))
    labels = tuple(f"sensor-{index}" for index in range(signals.shape[0]))

    def fail_nie(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic NIE failure")

    monkeypatch.setattr(nie_module, "compute_nie_raw_weighted_global_efficiency", fail_nie)
    kernels, weights, audit = _kernel_results(signals, labels)

    assert weights is not None
    assert kernels["ED"].valid is True
    assert kernels["OI"].valid is True
    assert kernels["NIE"] is None
    assert audit["kernel_failures"]["NIE"] == "RuntimeError:synthetic NIE failure"


def test_frozen_preregistration_rejects_changed_protocol_hash() -> None:
    selection = {
        "schema": "mndm.mnps_v3_2.vector_feasibility_002_selection.v1",
        "random_selection": {"seed": 20261013, "selected_subject_ids_in_draw_order": list(EXPECTED_SELECTED_IDS)},
    }
    with pytest.raises(ValueError, match="frozen_protocol_sha256_mismatch"):
        validate_frozen_preregistration(
            selection,
            selection_sha256=EXPECTED_SELECTION_SHA256,
            protocol_sha256="0" * 64,
            base_overlay_sha256=EXPECTED_BASE_OVERLAY_SHA256,
        )


def test_frozen_preregistration_accepts_only_pinned_draw_and_hashes() -> None:
    selection = {
        "schema": "mndm.mnps_v3_2.vector_feasibility_002_selection.v1",
        "random_selection": {"seed": 20261013, "selected_subject_ids_in_draw_order": list(EXPECTED_SELECTED_IDS)},
    }
    validate_frozen_preregistration(
        selection,
        selection_sha256=EXPECTED_SELECTION_SHA256,
        protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        base_overlay_sha256=EXPECTED_BASE_OVERLAY_SHA256,
    )


def test_loader_identity_checks_use_each_window_provenance_record() -> None:
    source_hash = "a" * 64
    overlay_hash = "b" * 64
    positions_hash = "c" * 64
    one_window = {
        "source": {
            "sha256_before_read": source_hash,
            "sha256_after_read": source_hash,
            "sha256_stable_after_read": True,
        },
        "configuration": {"overlay_sha256": overlay_hash, "overlay_hash_stable": True},
        "channels": {
            "positions_sha256": positions_hash,
            "positions_sha256_after_read": positions_hash,
            "positions_hash_stable": True,
        },
    }
    assert all(loader_identity_checks(
        [one_window, one_window],
        payload_sha256_before_loader=source_hash,
        overlay_sha256=overlay_hash,
        positions_sha256=positions_hash,
    ).values())
    changed_second = {**one_window, "configuration": {"overlay_sha256": "d" * 64, "overlay_hash_stable": True}}
    checks = loader_identity_checks(
        [one_window, changed_second],
        payload_sha256_before_loader=source_hash,
        overlay_sha256=overlay_hash,
        positions_sha256=positions_hash,
    )
    assert checks["loader_source_hash_stable"] is True
    assert checks["overlay_hash_stable"] is False


def test_ready_window_emits_all_families_and_provenance_without_qualification(tmp_path: Path) -> None:
    rng = np.random.Generator(np.random.PCG64(703))
    signals = rng.standard_normal((CHANNEL_COUNT, SAMPLE_COUNT), dtype=np.float64) * 1e-5
    labels = tuple(f"sensor-{index:03d}" for index in range(CHANNEL_COUNT))
    audit_flags = {
        "source_is_continuous": True,
        "timestamps_uniform": True,
        "gap_free": True,
        "bad_interval_free": True,
        "within_recording_bounds": True,
        "effective_filter_support_valid": True,
    }
    support = SimpleNamespace(
        sample_rate_hz=250.0,
        sample_count=SAMPLE_COUNT,
        window_start_sec=20.0,
        window_end_sec=40.0,
        **audit_flags,
    )
    window = SimpleNamespace(
        status="ready",
        reason=None,
        signals=signals,
        support=support,
        provenance={
            "source": {"sha256_before_read": "a" * 64, "sha256_after_read": "a" * 64, "sha256_stable_after_read": True},
            "source_dependency": {
                "source_sample_start": 16624,
                "source_sample_end_exclusive": 39997,
                "eeg_bandpass": {"tap_count": 825},
            },
            "support_audit": audit_flags,
        },
    )

    row = _make_window_row(
        index=1,
        window=window,
        nominal=(20.0, 40.0),
        labels=labels,
        subject_id="EPCTL-test",
        run_id="test-run",
        code_sha256="b" * 64,
        protocol_sha256="c" * 64,
        config_sha256="d" * 64,
        runtime={"python": "test"},
        edge_dir=tmp_path,
    )

    assert row["window_audit_status"] == "ready"
    assert row["input_matrix_sha256_little_endian_float64"]
    assert set(row["measurements"]) == {"ED", "OI", "NIE"}
    assert row["time_permutation_control"]["status"] == "passed"
    for family in row["measurements"].values():
        record = family["measurement_record"]
        assert record["valid"] is True
        assert record["qualified"] is False
        assert family["actual_realization_id"] is None
        assert family["realization_identity_status"] == "incomplete"
        assert record["realization"]["actual_id"] is None
        assert record["support"]["support_status"] == "known"
        assert record["reproducibility"]["input_sha256"] == row["input_matrix_sha256_little_endian_float64"]
    assert row["covariance_connectivity_diagnostics"]["values"]["raw_covariance_ED_identity_absolute_residual"] < 1e-10
    assert row["amplitude_and_spectral_diagnostics"]["status"] == "computed"


def test_source_failure_retains_all_windows_and_null_family_values() -> None:
    rows = _source_failure_rows(
        subject_id="EPCTL-test",
        reason="selected_source_identity_changed_since_freeze",
        run_id="test-run",
        protocol_sha256="a" * 64,
        code_sha256="b" * 64,
    )

    assert len(rows) == len(WINDOWS_SEC) == 15
    assert all(row["subject_id"] == "EPCTL-test" for row in rows)
    for row in rows:
        assert set(row["measurements"]) == {"ED", "OI", "NIE"}
        for family in row["measurements"].values():
            record = family["measurement_record"]
            assert record["value"] is None
            assert record["valid"] is False
            assert record["qualified"] is False
            assert record["status"] == "unsupported"
            assert record["reason"] == "selected_source_identity_changed_since_freeze"


def _summary_row(window_id: str, values: tuple[float, float, float]) -> dict[str, object]:
    measurements = {}
    for family, value in zip(("ED", "OI", "NIE"), values, strict=True):
        measurements[family] = {
            "measurement_record": {"value": value, "valid": True},
        }
    return {
        "window_id": window_id,
        "measurements": measurements,
        "covariance_connectivity_diagnostics": {
            "status": "computed",
            "values": {
                "ED_R": values[0] - 0.1,
                "mean_absolute_correlation": values[2] - 0.1,
                "mean_squared_correlation": 0.2,
                "nie_graph_path_increment": 0.1,
            },
        },
        "amplitude_and_spectral_diagnostics": {"spectral": {"bands": {}}},
        "time_permutation_control": {
            "metric_deltas_permuted_minus_original": {"OI": 0.2},
        },
    }


def test_participant_associations_use_one_summary_per_participant() -> None:
    rows = {
        "p1": [_summary_row("p1-w1", (1.0, 0.3, 0.6)), _summary_row("p1-w2", (3.0, 0.7, 0.4))],
        "p2": [_summary_row("p2-w1", (2.0, 0.5, 0.5)), _summary_row("p2-w2", (4.0, 0.9, 0.3))],
        "p3": [_summary_row("p3-w1", (4.0, 0.9, 0.3)), _summary_row("p3-w2", (8.0, 0.9, 0.2))],
    }

    result = participant_summaries(rows)

    pair = result["participant_level_correlation_matrix"]["ED"]["OI"]
    assert pair["n_participants"] == 3
    assert pair["participant_ids"] == ["p1", "p2", "p3"]
    assert pair["pearson_r"] is not None
    assert result["participant_summaries"]["p1"]["valid_measure_summaries"]["ED"]["median"] == 2.0
    assert result["participant_summaries"]["p1"]["nominal_window_count"] == 2
    assert result["predeclared_review_triggers"]["automatic_coordinate_removal"] is False

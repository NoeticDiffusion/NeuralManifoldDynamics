"""Focused tests for the bounded Family B export tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

import h5py
import numpy as np

from mndm.tools.family_b_export import (
    FAMILY_B_FIDELITY_THRESHOLD,
    blocked_one_step_audit,
    build_family_b_export,
    load_family_b_source,
)


def _write_source(
    path: Path,
    *,
    source_status: str | None = None,
    time: np.ndarray | None = None,
    values: np.ndarray | None = None,
    rel_mse: float = 0.5,
) -> None:
    if time is None and values is not None:
        time = np.arange(np.asarray(values).shape[0], dtype=float) * 4.0
    t = np.asarray(time if time is not None else np.arange(48, dtype=float) * 4.0, dtype=float)
    x = np.asarray(
        values if values is not None else np.column_stack([np.sin(t / 15.0), np.cos(t / 18.0), t / 100.0]),
        dtype=np.float32,
    )
    x_dot = np.gradient(x, axis=0).astype(np.float32)
    centers = np.arange(1, t.size - 1, dtype=np.int32)
    jacobian = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], centers.size, axis=0)
    with h5py.File(path, "w") as h5:
        h5.attrs["dataset_id"] = "synthetic:sub-0001:run-001"
        h5.attrs["mndm_version"] = "3.0.0"
        h5.attrs["primary_coordinate_layer"] = "coords_3d_subject_anchored"
        h5.attrs["primary_coordinate_contract"] = "subject_anchored"
        h5.create_dataset("time", data=t)
        h5.create_dataset("mnps_3d", data=x)
        h5.create_dataset("mnps_3d_dot", data=x_dot)
        h5.create_dataset("window_start", data=t - 2.0)
        h5.create_dataset("window_end", data=t + 2.0)
        h5.create_dataset("epoch_id", data=np.arange(t.size, dtype=np.int64))
        coords = h5.create_group("coords_3d_subject_anchored")
        coords.attrs["coordinate_contract"] = "subject_anchored"
        coords.attrs["role"] = "within_subject_geometry"
        coords.create_dataset("values", data=x)
        coords.create_dataset("names", data=np.asarray([b"m", b"d", b"e"]))
        jac = h5.create_group("jacobian")
        jac.create_dataset("J_hat", data=jacobian)
        jac.create_dataset("centers", data=centers)
        diag = jac.create_group("diagnostics")
        diag.create_dataset("rel_mse_baseline_windows", data=np.full(centers.size, rel_mse, dtype=np.float32))
        diag.attrs["rel_mse_baseline_median"] = rel_mse
        diag.attrs["knn_k"] = 20
        diag.attrs["super_window"] = 3
        support = h5.create_group("provenance/signal_support_provenance")
        support.create_dataset("temporal_support_status", data=np.bytes_("unknown"))
        support.create_dataset("temporal_support_reason", data=np.bytes_("missing_qc_sidecar"))
        if source_status is not None:
            metrics = jac.create_group("derived_metrics/v1")
            metrics.create_dataset("computation_status", data=np.bytes_(source_status))
            metrics.create_dataset("failure_reason", data=np.bytes_("local_linear_fit_not_better_than_baseline"))


def test_export_writes_canonical_metrics_subset_and_audit(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    output = tmp_path / "family_b.h5"
    audit_path = tmp_path / "family_b_audit.json"
    _write_source(source)

    result = build_family_b_export(source, output, audit_json_path=audit_path)

    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["computation_status"] == "computed"
    assert result["audit_status"] == "computed"
    assert result["source_h5_sha256"] == expected_hash
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["qualification_status"] == "not_assessed"
    assert audit["support_scope"] == "nominal_windows_only"
    assert set(audit["models"]) == {"phi", "train_target_mean", "persistence"}

    with h5py.File(output, "r") as h5:
        assert h5["jacobian/J_hat"].shape == (46, 3, 3)
        assert np.array_equal(h5["jacobian/centers"][:], np.arange(1, 47, dtype=np.int32))
        assert np.array_equal(h5["time"][:], np.arange(48, dtype=float) * 4.0)
        assert h5["coords_3d_subject_anchored/values"].shape == (48, 3)
        assert h5["jacobian/diagnostics/rel_mse_baseline_windows"].shape == (46,)
        metrics = h5["jacobian/derived_metrics/v1"]
        assert metrics["computation_status"][()].decode() == "computed"
        assert metrics["provenance/source_h5_sha256"][()].decode() == expected_hash
        assert metrics["provenance/primary_coordinate_layer"][()].decode() == "coords_3d_subject_anchored"
        assert h5.attrs["family_b_fidelity_threshold"] == FAMILY_B_FIDELITY_THRESHOLD
        assert h5.attrs["mndm_version"] == "3.0.0"
        assert h5.attrs["source_mndm_version"] == "3.0.0"
        assert h5["provenance/family_b_export/source_attrs/mndm_version"][()].decode() == "3.0.0"


def test_source_refusal_remains_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "refused_source.h5"
    output = tmp_path / "refused_family_b.h5"
    _write_source(source, source_status="insufficient_support")

    result = build_family_b_export(source, output)

    assert result["computation_status"] == "insufficient_support"
    with h5py.File(output, "r") as h5:
        metrics = h5["jacobian/derived_metrics/v1"]
        assert metrics["computation_status"][()].decode() == "insufficient_support"
        assert metrics["failure_reason"][()].decode() == "local_linear_fit_not_better_than_baseline"
        assert metrics["measurement_validity"][()].decode() == "not_applicable"
        assert np.isnan(h5["jacobian/derived_metrics/v1/series/spectral_abscissa"][:]).all()
        assert np.allclose(h5["jacobian/derived_metrics/v1/series/rel_mse_baseline"][:], 0.5)
        assert np.all(h5["jacobian/derived_metrics/v1/series/stable_reactive_flag"][:] == -1)


def test_blocked_audit_reports_clock_gaps_without_bridging(tmp_path: Path) -> None:
    source_path = tmp_path / "gap_source.h5"
    times = np.arange(48, dtype=float) * 4.0
    times[24:] += 40.0
    _write_source(source_path, time=times)
    source = load_family_b_source(source_path)

    audit = blocked_one_step_audit(source)

    assert audit["n_pairs_rejected_clock_gap"] > 0
    assert audit["qualification_status"] == "not_assessed"
    assert audit["claim_status"] == "no_biological_claim"


def test_misaligned_gate_diagnostic_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "bad_diag.h5"
    output = tmp_path / "bad_diag_family_b.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        del h5["jacobian/diagnostics/rel_mse_baseline_windows"]
        h5["jacobian/diagnostics"].create_dataset("rel_mse_baseline_windows", data=np.ones(3, dtype=np.float32))
        h5["jacobian/diagnostics"].attrs["rel_mse_baseline_median"] = 0.1

    result = build_family_b_export(source, output)

    assert result["computation_status"] == "insufficient_support"
    assert result["failure_reason"] == "fit_fidelity_unknown"


def test_export_rejects_source_and_existing_output_collisions(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    output = tmp_path / "family_b.h5"
    _write_source(source)

    import pytest

    with pytest.raises(ValueError, match="separate from source"):
        build_family_b_export(source, source)
    build_family_b_export(source, output)
    with pytest.raises(FileExistsError, match="overwrite output"):
        build_family_b_export(source, output)


def test_known_affine_trajectory_has_a_computed_audit_but_no_qualification_claim(tmp_path: Path) -> None:
    source = tmp_path / "known.h5"
    t = np.arange(48, dtype=float) * 4.0
    values = np.zeros((t.size, 3), dtype=np.float32)
    values[0] = [0.2, -0.1, 0.4]
    transition = np.asarray([[0.8, 0.05, 0.0], [0.0, 0.7, 0.1], [0.0, 0.0, 0.9]], dtype=np.float32)
    offset = np.asarray([0.1, -0.03, 0.02], dtype=np.float32)
    for index in range(1, values.shape[0]):
        values[index] = values[index - 1] @ transition.T + offset
    _write_source(source, values=values)
    loaded = load_family_b_source(source)

    audit = blocked_one_step_audit(loaded)

    assert audit["status"] == "computed"
    assert audit["qualification_status"] == "not_assessed"
    assert audit["models"]["phi"]["mse"] < 1e-10
    assert audit["design_full_rank"] is True
    assert audit["finite_predictions"] is True


def test_noisy_fit_keeps_family_b_gate_closed(tmp_path: Path) -> None:
    source = tmp_path / "noise.h5"
    output = tmp_path / "noise_family_b.h5"
    _write_source(source, rel_mse=1.2)

    result = build_family_b_export(source, output)

    assert result["computation_status"] == "insufficient_support"
    assert result["failure_reason"] == "local_linear_fit_not_better_than_baseline"
    with h5py.File(output, "r") as h5:
        assert np.isnan(h5["jacobian/derived_metrics/v1/series/numerical_abscissa"][:]).all()


def test_9d_export_uses_matching_jacobian_without_3d_fallback(tmp_path: Path) -> None:
    source = tmp_path / "source_9d.h5"
    output = tmp_path / "family_b_9d.h5"
    t = np.arange(48, dtype=float) * 4.0
    mnps = np.column_stack([np.sin(t / 15.0), np.cos(t / 18.0), t / 100.0]).astype(np.float32)
    coords_9d = np.column_stack([mnps, mnps, mnps]).astype(np.float32)
    centers = np.arange(1, t.size - 1, dtype=np.int32)
    with h5py.File(source, "w") as h5:
        h5.attrs["dataset_id"] = "synthetic:sub-0001:run-001"
        h5.attrs["primary_coordinate_layer"] = "coords_9d_subject_anchored"
        h5.attrs["primary_coordinate_contract"] = "subject_anchored"
        h5.create_dataset("time", data=t)
        h5.create_dataset("mnps_3d", data=mnps)
        h5.create_dataset("mnps_3d_dot", data=np.gradient(mnps, axis=0).astype(np.float32))
        h5.create_dataset("window_start", data=t - 2.0)
        h5.create_dataset("window_end", data=t + 2.0)
        h5.create_dataset("epoch_id", data=np.arange(t.size, dtype=np.int64))
        coords = h5.create_group("coords_9d_subject_anchored")
        coords.create_dataset("values", data=coords_9d)
        coords.create_dataset("names", data=np.asarray([f"d{i}".encode() for i in range(9)]))
        jac = h5.create_group("jacobian_9D")
        jac.create_dataset("J_hat", data=np.repeat(np.eye(9, dtype=np.float32)[None], centers.size, axis=0))
        jac.create_dataset("centers", data=centers)
        diag = jac.create_group("diagnostics")
        diag.create_dataset("rel_mse_baseline_windows", data=np.full(centers.size, 0.5, dtype=np.float32))
        diag.attrs["rel_mse_baseline_median"] = 0.5
        diag.attrs["support_mode"] = "knn"

    result = build_family_b_export(source, output)

    assert result["jacobian_group"] == "jacobian_9D"
    with h5py.File(output, "r") as h5:
        assert h5["jacobian_9D/J_hat"].shape == (46, 9, 9)
        assert "jacobian_9D/derived_metrics/v1" in h5
        assert "jacobian/derived_metrics/v1" not in h5
        assert "jacobian_9D/diagnostics/rel_mse_baseline_windows" in h5
        assert "jacobian/diagnostics" not in h5
        assert h5["jacobian_9D/derived_metrics/v1/provenance/source_jacobian_diagnostics/rel_mse_baseline_windows"].shape == (46,)


def test_9d_source_does_not_fallback_to_3d_jacobian(tmp_path: Path) -> None:
    source = tmp_path / "missing_9d_jacobian.h5"
    t = np.arange(12, dtype=float) * 4.0
    _write_source(source, time=t)
    with h5py.File(source, "a") as h5:
        h5.attrs["primary_coordinate_layer"] = "coords_9d_subject_anchored"
        coords = h5.create_group("coords_9d_subject_anchored")
        coords.create_dataset("values", data=np.zeros((t.size, 9), dtype=np.float32))
        coords.create_dataset("names", data=np.asarray([f"d{i}".encode() for i in range(9)]))

    import pytest

    with pytest.raises(ValueError, match="source Jacobian is absent"):
        load_family_b_source(source)


def test_coordinate_prefix_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "wrong_coordinate_prefix.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        coords = h5["coords_3d_subject_anchored"]
        del coords["values"]
        del coords["names"]
        coords.create_dataset("values", data=np.zeros((48, 9), dtype=np.float32))
        coords.create_dataset("names", data=np.asarray([f"d{i}".encode() for i in range(9)]))

    import pytest

    with pytest.raises(ValueError, match="coords_3d layer must have exactly 3"):
        load_family_b_source(source)


def test_bad_window_bounds_and_epoch_file_boundaries_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad_bounds.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        h5["window_end"][3] = h5["window_start"][3]

    import pytest

    with pytest.raises(ValueError, match="window bounds"):
        load_family_b_source(source)

    source = tmp_path / "boundaries.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        h5["epoch_id"][24:] += 100
        rows = h5.create_group("row_source")
        rows.create_dataset("raw_file", data=np.asarray([b"a"] * 24 + [b"b"] * 24))
    audit = blocked_one_step_audit(load_family_b_source(source))
    assert audit["n_pairs_rejected_epoch_boundary"] == 1
    assert audit["n_pairs_rejected_source_file_boundary"] == 1

    source = tmp_path / "misaligned_raw_files.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        rows = h5.create_group("row_source")
        rows.create_dataset("raw_file", data=np.asarray([b"a"] * 47))
    with pytest.raises(ValueError, match="raw_file must align"):
        load_family_b_source(source)


def test_fractional_and_nonmonotone_centers_are_rejected_before_cast(tmp_path: Path) -> None:
    source = tmp_path / "fractional_centers.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        old = h5["jacobian/centers"][:]
        del h5["jacobian/centers"]
        h5["jacobian"].create_dataset("centers", data=old.astype(np.float64) + 0.5)

    import pytest

    with pytest.raises(ValueError, match="integer-valued"):
        load_family_b_source(source)

    source = tmp_path / "duplicate_centers.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        old = h5["jacobian/centers"][:]
        old[2] = old[1]
        h5["jacobian/centers"][:] = old
    with pytest.raises(ValueError, match="align with J_hat"):
        load_family_b_source(source)


def test_invalid_source_metric_schema_or_status_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "invalid_metric_contract.h5"
    _write_source(source, source_status="unknown")

    import pytest

    with pytest.raises(ValueError, match="unknown source Jacobian computation status"):
        load_family_b_source(source)

    source = tmp_path / "wrong_metric_schema.h5"
    _write_source(source, source_status="computed")
    with h5py.File(source, "a") as h5:
        h5["jacobian/derived_metrics/v1"].create_dataset("schema_version", data=np.bytes_("WRONG_SCHEMA"))
    with pytest.raises(ValueError, match="metrics schema is not canonical"):
        load_family_b_source(source)

    source = tmp_path / "missing_metric_status.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        metrics = h5["jacobian"].create_group("derived_metrics/v1")
        metrics.create_dataset("schema_version", data=np.bytes_("mndm.jacobian_metrics.v1"))
        summary = metrics.create_group("summary")
        summary.create_dataset("fit_identified", data=False)
    with pytest.raises(ValueError, match="computation status is required"):
        load_family_b_source(source)


def test_contradictory_computed_source_certificate_stays_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "contradictory_source.h5"
    output = tmp_path / "contradictory_export.h5"
    _write_source(source, source_status="computed")
    with h5py.File(source, "a") as h5:
        metrics = h5["jacobian/derived_metrics/v1"]
        metrics.create_dataset("schema_version", data=np.bytes_("mndm.jacobian_metrics.v1"))
        summary = metrics.create_group("summary")
        summary.create_dataset("fit_identified", data=False)

    result = build_family_b_export(source, output)

    assert result["computation_status"] == "insufficient_support"
    assert result["failure_reason"] == "source_fit_identified_false"
    with h5py.File(output, "r") as h5:
        assert h5["jacobian/derived_metrics/v1/measurement_validity"][()].decode() == "not_applicable"


def test_rank_deficient_holdout_is_not_reported_as_computed(tmp_path: Path) -> None:
    source = tmp_path / "rank_deficient.h5"
    values = np.ones((64, 3), dtype=np.float32)
    _write_source(source, values=values)

    audit = blocked_one_step_audit(load_family_b_source(source))

    assert audit["design_full_rank"] is False
    assert audit["status"] == "insufficient_support"
    assert audit["failure_reason"] == "blocked_holdout_rank_deficient_design"
    assert audit["models"] == {}


def test_missing_window_bounds_do_not_claim_audit_support(tmp_path: Path) -> None:
    source = tmp_path / "missing_bounds.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        del h5["window_start"]
        del h5["window_end"]

    audit = blocked_one_step_audit(load_family_b_source(source))

    assert audit["status"] == "insufficient_support"
    assert audit["failure_reason"] == "nominal_bounds_unavailable"
    assert audit["support_scope"] == "nominal_bounds_unavailable"


def test_irregular_window_widths_use_actual_pair_intervals_for_purge(tmp_path: Path) -> None:
    source = tmp_path / "irregular_bounds.h5"
    _write_source(source)
    with h5py.File(source, "a") as h5:
        # This training source window overlaps the held-out block even though
        # its target window ends before the held-out source window starts.
        h5["window_end"][34] = 150.0

    audit = blocked_one_step_audit(load_family_b_source(source))

    assert audit["status"] == "computed"
    assert audit["n_train_pairs"] == 34
    assert audit["n_pairs_excluded_by_block_gap"] == 3


def test_independent_noise_control_reports_null_scores_without_qualification(tmp_path: Path) -> None:
    source = tmp_path / "independent_noise.h5"
    rng = np.random.default_rng(12345)
    _write_source(source, values=rng.normal(size=(1024, 3)).astype(np.float32))

    audit = blocked_one_step_audit(load_family_b_source(source))

    assert audit["status"] == "computed"
    assert audit["qualification_status"] == "not_assessed"
    assert audit["claim_status"] == "no_biological_claim"
    phi = audit["models"]["phi"]
    mean = audit["models"]["train_target_mean"]
    persistence = audit["models"]["persistence"]
    assert np.isfinite([phi["mse"], mean["mse"], persistence["mse"]]).all()
    assert phi["mse"] >= mean["mse"]
    assert phi["mse"] < persistence["mse"]
    assert 0.8 <= phi["relative_to_test_mean"] <= 1.3

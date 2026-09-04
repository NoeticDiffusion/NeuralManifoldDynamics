"""Round-3 inferential-grain write/read contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamical_families.contracts import (
    DIFFUSION_GEOMETRY_SCHEMA_VERSION,
    unavailable_result,
)
from mndm.dynamics.finite_time_response import compute_finite_time_response
from mndm.dynamics.jacobian_metrics import compute_jacobian_metrics, flatten_metric_result
from mndm.dynamics.stochastic_reachability import (
    compute_stochastic_reachability,
    estimate_one_step_transition_covariance,
    estimate_transition_residual_covariance_proxy,
)
from mndm.dynamics.transition_residuals import compute_transition_residuals
from mndm.inferential_grain import (
    attach_grain,
    read_inferential_grain,
    validate_write_grain,
)
from mndm.schema import MNPSPayload


def test_attach_grain_rejects_unknown_native() -> None:
    with pytest.raises(ValueError, match="grain.native"):
        validate_write_grain(
            native="epoch",
            parent="recording",
            biological_unit="subject",
            repeated_measure="true",
            direct_between_subject_inference="forbidden",
        )
    with pytest.raises(ValueError, match="biological_unit"):
        attach_grain(
            {"computation_status": "computed"},
            native="window",
            parent="recording",
            repeated_measure="true",
            biological_unit="session",
        )


def test_legacy_reader_never_infers_window_from_series(tmp_path: Path) -> None:
    path = tmp_path / "legacy_no_grain.h5"
    with h5py.File(path, "w") as handle:
        group = handle.require_group("metrics")
        series = group.require_group("series")
        series.create_dataset("spectral_abscissa", data=np.array([1.0], dtype=np.float32))
        group.create_dataset("computation_status", data=np.bytes_("computed"))
    with h5py.File(path, "r") as handle:
        grain = read_inferential_grain(handle["metrics"])
    assert grain["native"] == "not_recorded"
    assert grain["parent"] == "not_recorded"
    assert grain["biological_unit"] == "not_recorded"
    assert grain["repeated_measure"] == "not_recorded"
    assert grain["direct_between_subject_inference"] == "not_recorded"
    assert grain["grain_origin"] == "legacy_absent"
    assert grain["native"] != "window"
    assert grain["biological_unit"] != "subject"


def test_canonical_reader_copies_grain_group(tmp_path: Path) -> None:
    path = tmp_path / "canonical_grain.h5"
    with h5py.File(path, "w") as handle:
        grain = handle.require_group("family/grain")
        grain.create_dataset("native", data=np.bytes_("window"))
        grain.create_dataset("parent", data=np.bytes_("recording"))
        grain.create_dataset("biological_unit", data=np.bytes_("subject"))
        grain.create_dataset("repeated_measure", data=np.bytes_("true"))
        grain.create_dataset(
            "direct_between_subject_inference",
            data=np.bytes_("forbidden"),
        )
    with h5py.File(path, "r") as handle:
        read = read_inferential_grain(handle["family"])
    assert read["native"] == "window"
    assert read["grain_origin"] == "canonical"


def test_unavailable_family_still_has_grain() -> None:
    result = unavailable_result(
        DIFFUSION_GEOMETRY_SCHEMA_VERSION,
        status="not_testable",
        failure_reason="mndm_diffusion_adapter_translation_qualification_required",
    )
    assert result["computation_status"] == "not_testable"
    assert result["grain"]["native"] == "window"
    assert result["grain"]["biological_unit"] == "subject"
    assert result["grain"]["direct_between_subject_inference"] == "forbidden"


def test_jacobian_metrics_grain_is_window_and_flatten_keeps_it(tmp_path: Path) -> None:
    j_hat = np.repeat(np.eye(3)[None], 2, axis=0)
    result = compute_jacobian_metrics(j_hat)
    assert result["grain"]["native"] == "window"
    assert result["grain"]["parent"] == "recording"
    assert result["grain"]["biological_unit"] == "subject"
    assert result["grain"]["repeated_measure"] == "true"
    assert result["grain"]["direct_between_subject_inference"] == "forbidden"
    flattened = flatten_metric_result(result)
    assert flattened["grain"]["native"] == "window"

    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        jacobian=j_hat[:1],
        jacobian_derived_metrics=compute_jacobian_metrics(j_hat[:1]),
    )
    out = write_h5(tmp_path / "metrics-grain.h5", "test", payload)
    with h5py.File(out, "r") as handle:
        grain = handle["/jacobian/derived_metrics/v1/grain"]
        assert grain["native"][()].decode() == "window"
        assert grain["biological_unit"][()].decode() == "subject"
        assert grain["direct_between_subject_inference"][()].decode() == "forbidden"
        assert np.array_equal(handle["jacobian/J_hat"][:], j_hat[:1])


def test_finite_time_response_grain_is_recording_horizon() -> None:
    result = compute_finite_time_response(
        np.repeat(np.array([[[-1.0]]]), 3, axis=0),
        horizon_steps=[1],
        time=np.array([0.0, 1.0, 2.0]),
        validation_level="model_derived",
    )
    assert result["computation_status"] == "computed"
    assert result["validation_level"] == "model_derived"
    assert result["measurement_validity"] == "not_assessed"
    assert result["grain"]["native"] == "recording_horizon"
    assert result["grain"]["parent"] == "recording"


def test_transition_residuals_grain_is_transition() -> None:
    x = np.arange(8, dtype=float)[:, None]
    nn = np.tile(np.arange(8, dtype=np.int32), (8, 1))
    result = compute_transition_residuals(
        x,
        np.ones_like(x),
        np.arange(8, dtype=float),
        np.array([2, 3, 5], dtype=np.int32),
        nn,
        super_window=1,
        ridge_alpha=1e-6,
        distance_weighted=False,
        crossfit_embargo_steps=0,
        coordinate_contract="subject_anchored",
        coordinate_layer="coords_3d_subject_anchored",
    )
    assert result["computation_status"] == "computed"
    assert result["grain"]["native"] == "transition"
    assert result["grain"]["parent"] == "recording"
    assert result["grain"]["repeated_measure"] == "true"


def test_q_proxy_grain_is_recording_not_repeated() -> None:
    state = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    proxy = estimate_one_step_transition_covariance(
        state, np.zeros_like(state), q_dt_sec=1.0
    )
    assert proxy["computation_status"] == "computed"
    assert proxy["grain"]["native"] == "recording"
    assert proxy["grain"]["parent"] == "subject"
    assert proxy["grain"]["repeated_measure"] == "false"
    reachability = compute_stochastic_reachability(
        [np.eye(2)],
        proxy["covariance"],
        q_contract=proxy,
    )
    assert reachability["grain"]["native"] == "recording_horizon"
    pooled = estimate_transition_residual_covariance_proxy(
        {
            "computation_status": "computed",
            "series": {
                "transition_residual": np.zeros((8, 2), dtype=np.float32),
                "dt_sec": np.ones(8, dtype=np.float32),
            },
            "provenance": {},
        }
    )
    assert pooled["grain"]["native"] == "recording"
    assert pooled["grain"]["repeated_measure"] == "false"

from __future__ import annotations

import numpy as np

from mndm.mnps.v3_4.dynamics_adapter import attach_meg_v3_4_dynamics, attach_v3_4_dynamics
from mndm.mnps.v3_4.coordinates import COORDINATE_ORDER


COORDINATE_NAMES = list(COORDINATE_ORDER)


def _config() -> dict:
    return {
        "measurement_surface": {"window_step_sec": 4.0},
        "v3_4_dynamics": {
            "enabled": True,
            "window_step_tolerance_sec": 1.0e-9,
            "coordinate_layers": {
                "raw": {"enabled": True},
                "anchored": {"enabled": True},
            },
            "drift": {
                "enabled": True,
                "neighborhood": {"k": 32},
                "min_neighborhood_samples": 28,
                "realized_velocity_level0": {"enabled": True},
                "conditional_mean_rate_level1": {
                    "pooled": {"enabled": True},
                    "blocked_crossfit": {"enabled": False},
                    "lag_diagnostics": {"enabled": False},
                },
            },
            "diffusion": {
                "enabled": True,
                "neighborhood": {"k": 32},
                "min_neighborhood_samples": 28,
            },
            "one_step": {
                "enabled": True,
                "declared_lags": [1],
                "n_blocks": 2,
                "embargo_steps": 4,
            },
        },
    }


def _payload(n_windows: int = 240) -> dict:
    rng = np.random.default_rng(17)
    values = np.zeros((n_windows, 9), dtype=np.float64)
    innovations = rng.normal(scale=0.12, size=values.shape)
    for idx in range(1, n_windows):
        values[idx] = 0.85 * values[idx - 1] + innovations[idx]
    starts = np.arange(n_windows, dtype=np.float64) * 4.0
    return {
        "coordinate_order": COORDINATE_NAMES,
        "window_start_sec": starts,
        "window_end_sec": starts + 8.0,
        "complete_9d_mask": np.ones(n_windows, dtype=np.int8),
        "raw_coordinates": {
            "values": values,
            "support": np.ones(values.shape, dtype=np.int8),
        },
        "coordinates": {"values": values / 2.0},
        "provenance": {
            "anchor_application_status": "computed",
            "anchor_id": "test-anchor",
            "anchor_hash": "anchor-content-hash",
            "anchor_manifest_sha256": "anchor-manifest-hash",
        },
    }


def test_v3_4_dynamics_keeps_realizations_separate_and_one_step_anchored() -> None:
    result = attach_v3_4_dynamics(_payload(), _config())

    assert result is not None
    dynamics = result["dynamics"]
    realizations = dynamics["coordinate_realizations"]
    raw = realizations["v3_4_raw"]
    anchored = realizations["v3_4_anchored"]

    assert raw["coordinate_unit_note"] == "mixed_source_units"
    assert anchored["coordinate_unit_note"] == "dimensionless_reference_scores"
    assert "one_step" not in raw["families"]
    assert "one_step" in anchored["families"]
    assert raw["families"]["diffusion"]["provenance"]["coordinate_layer"] == "v3_4_raw"
    assert anchored["families"]["drift"]["provenance"]["coordinate_layer"] == "v3_4_anchored"
    assert raw["provenance"]["input_coordinates_sha256"] != anchored["provenance"]["input_coordinates_sha256"]
    assert raw["provenance"]["effective_family_config_sha256"]
    assert raw["provenance"]["dynamics_implementation_hashes"]["v3_4_dynamics_adapter"]
    assert anchored["families"]["one_step"]["computation_status"] == "computed"
    assert anchored["families"]["drift"]["conditional_mean_rate_level1"]["pooled"]["provenance"]["settings"][
        "tie_break_neighbors_by_index"
    ] is True


def test_v3_4_dynamics_does_not_fallback_to_anchored_without_applied_anchor() -> None:
    payload = _payload()
    payload["complete_9d_mask"][:] = 0
    payload["provenance"]["anchor_application_status"] = "not_scaled_no_compatible_anchor"
    result = attach_v3_4_dynamics(payload, _config())

    assert result is not None
    realizations = result["dynamics"]["coordinate_realizations"]
    assert realizations["v3_4_raw"]["computation_status"] == "computed"
    assert realizations["v3_4_anchored"]["computation_status"] == "not_testable"
    assert realizations["v3_4_anchored"]["failure_reason"] == "external_anchor_not_applied"
    assert "one_step" not in realizations["v3_4_raw"]["families"]


def test_v3_4_dynamics_splits_segments_at_incomplete_windows() -> None:
    payload = _payload()
    payload["complete_9d_mask"][100] = 0
    payload["raw_coordinates"]["support"][100, 3] = 0
    payload["coordinates"]["values"][100, 3] = np.nan
    result = attach_v3_4_dynamics(payload, _config())

    assert result is not None
    raw = result["dynamics"]["coordinate_realizations"]["v3_4_raw"]
    assert raw["complete_9d_mask"][100] == 0
    assert raw["segment_id"][99] != raw["segment_id"][101]
    pooled = raw["families"]["drift"]["conditional_mean_rate_level1"]["pooled"]
    assert pooled["series"]["valid"][100] == 0
    assert np.isnan(pooled["series"]["b_hat"][100]).all()


def test_v3_4_one_step_records_temporal_block_support_floor() -> None:
    result = attach_v3_4_dynamics(_payload(126), _config())

    assert result is not None
    one_step = result["dynamics"]["coordinate_realizations"]["v3_4_anchored"]["families"]["one_step"]
    assert one_step["computation_status"] == "insufficient_support"
    assert one_step["summary"]["n_affine_parameters"] == 90
    assert one_step["summary"]["n_training_pairs"] < 90
    assert one_step["summary"]["below_declared_support_floor"] is True
    assert one_step["series"] == {}


def test_v3_4_dynamics_disabled_is_a_noop() -> None:
    payload = _payload(40)
    config = _config()
    config["v3_4_dynamics"]["enabled"] = False

    result = attach_v3_4_dynamics(payload, config)

    assert result is not None
    assert "dynamics" not in result


def test_v3_4_dynamics_parent_status_retains_insufficient_support() -> None:
    config = _config()
    block = config["v3_4_dynamics"]
    block["drift"]["min_samples"] = 1_000
    block["diffusion"]["min_samples"] = 1_000
    block["one_step"]["min_samples"] = 1_000

    result = attach_v3_4_dynamics(_payload(40), config)

    assert result is not None
    assert result["dynamics"]["computation_status"] == "insufficient_support"


def test_neighbor_ties_use_reference_row_order_when_requested() -> None:
    from mndm.dynamical_families.validity import chunked_nearest_neighbors

    queries = np.zeros((1, 2), dtype=np.float64)
    references = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)

    indices, _ = chunked_nearest_neighbors(
        queries,
        references,
        2,
        tie_break_by_index=True,
    )

    assert indices.tolist() == [[0, 1]]


def test_one_step_rejects_v3_4_raw_layer_with_specific_reason() -> None:
    from mndm.dynamical_families.one_step_operator import estimate_affine_one_step_family

    state = np.zeros((20, 9), dtype=np.float64)
    time = np.arange(20, dtype=np.float64)
    result = estimate_affine_one_step_family(
        state,
        time,
        coordinate_layer="v3_4_raw",
        coordinate_names=COORDINATE_NAMES,
    )

    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "one_step_requires_v3_4_anchored_layer"


def test_v3_4_dynamics_is_written_under_measurement_namespace(tmp_path) -> None:
    import h5py

    from core.io.h5_writer import write_measurement_export_sidecar

    payload = attach_v3_4_dynamics(_payload(), _config())
    assert payload is not None
    path = write_measurement_export_sidecar(
        tmp_path / "mnps_v3_4_measurements.h5",
        measurement_id="v3_4",
        measurement_payload=payload,
        recording_identity={"dataset_id": "synthetic"},
        skip_reason="test",
    )

    with h5py.File(path, "r") as handle:
        dynamics = handle["measurements/v3_4/dynamics"]
        assert dynamics["coordinate_realizations/v3_4_raw/coordinate_layer"][()].decode() == "v3_4_raw"
        assert dynamics["coordinate_realizations/v3_4_anchored/families/one_step/computation_status"][()].decode() == "computed"


def test_meg_dynamics_are_separate_by_sensor_family_and_support_7d_and_9d():
    base = _payload(240)
    values = np.asarray(base["raw_coordinates"]["values"], dtype=np.float64)
    support = np.ones_like(values, dtype=np.uint8)
    names_9d = [f"meg_coordinate_{idx}" for idx in range(9)]

    def variant(names: list[str], data: np.ndarray) -> dict:
        return {
            "coordinate_names": names,
            "coordinate_units": ["mixed_native"] * len(names),
            "raw_values": data,
            "coordinate_support": support[:, : len(names)],
            "complete_window_mask": np.ones(data.shape[0], dtype=np.uint8),
        }

    family_payload = {
        "variants": {
            "existing_coordinate_set": variant(names_9d, values),
            "removal_only": variant(names_9d[:7], values[:, :7]),
        }
    }
    payload = {
        "anchor_application_status": "not_applied",
        "window_axis": {
            "window_start_sec": base["window_start_sec"],
            "window_end_sec": base["window_end_sec"],
        },
        "sensor_families": {"mag": family_payload, "grad": family_payload},
    }
    result = attach_meg_v3_4_dynamics(payload, _config())

    mag = result["sensor_families"]["mag"]["variants"]["existing_coordinate_set"]["dynamics"]
    grad = result["sensor_families"]["grad"]["variants"]["existing_coordinate_set"]["dynamics"]
    mag_7d = result["sensor_families"]["mag"]["variants"]["removal_only"]["dynamics"]
    mag_raw = mag["coordinate_realizations"]["v3_4_raw"]
    assert mag["computation_status"] == "computed"
    assert grad["computation_status"] == "computed"
    assert mag["sensor_family"] == "mag"
    assert grad["sensor_family"] == "grad"
    assert mag_raw["coordinate_unit_note"] == "mixed_source_units"
    assert "complete_coordinate_mask" in mag_raw
    assert "one_step" not in mag_raw["families"]
    assert mag["one_step_status"]["computation_status"] == "not_testable"
    assert "not_applied" in mag["one_step_status"]["failure_reason"]
    assert mag_7d["coordinate_dimension"] == 7
    assert mag_7d["coordinate_realizations"]["v3_4_raw"]["computation_status"] == "computed"

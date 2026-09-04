"""OD-TQ2 adapter replay for the explicit 1-D O2b committor contract."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamical_families import committor as committor_module
from mndm.dynamical_families.committor import (
    estimate_committor_local_law_dense_grid_o2b,
)
from mndm.dynamical_families.validity import increment_pairs
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers
from mndm.schema import MNPSPayload


def _truth_known_local_ensembles(
    *, support_per_grid: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build zero-drift local ensembles with q(x)=(x-a)/(b-a)."""
    grid = np.linspace(-0.8, 0.8, 65)
    values: list[float] = []
    labels: list[int] = []
    segments: list[int] = []
    for grid_index, value in enumerate(grid):
        for replicate in range(support_per_grid):
            values.extend([float(value), float(value)])
            labels.extend([0, 1])
            segments.append(grid_index * support_per_grid + replicate)
            segments.append(grid_index * support_per_grid + replicate)
    reaction = np.asarray(values, dtype=np.float64)
    state = np.column_stack([reaction, np.zeros((reaction.size, 2), dtype=np.float64)])
    time = np.arange(reaction.size, dtype=np.float64) * 0.001
    regime = np.asarray(labels, dtype=np.int8)
    segment_id = np.asarray(segments, dtype=np.int32)
    return state, time, reaction, regime, segment_id


def _config(*, qualified: bool = True, grid_resolution: int = 65, min_support: int = 20) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "destination": {
                "enabled": True,
                "regime_source": "explicit_first_hit_labels",
                "label_key": "stage",
                "reaction_coordinate": {
                    "source": "explicit_column",
                    "key": "q_coordinate",
                    "name": "q_coordinate",
                    "boundaries": [-0.8, 0.8],
                },
                "estimator": "local_law_dense_grid_o2b",
                "translation_qualification": {
                    "qualified": qualified,
                    "qualification_id": "OD-TQ2-test-certificate" if qualified else None,
                    "qualification_contract_hash": "od-tq2-test-hash" if qualified else None,
                },
                "grid_resolution": grid_resolution,
                "diffusion_coefficient": 0.25,
                "set_A": [0],
                "set_B": [1],
                "min_samples": 100,
                "min_support_per_grid": min_support,
                "min_transition_segments": 5,
            },
        }
    }


def test_od_tq2_o2b_recovers_truth_known_zero_drift_committor() -> None:
    state, time, reaction, regime, segment_id = _truth_known_local_ensembles()
    result = estimate_committor_local_law_dense_grid_o2b(
        state,
        time,
        reaction,
        regime,
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=0.25,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate_name="q_coordinate",
        min_samples=100,
        min_support_per_grid=20,
    )
    assert result["computation_status"] == "computed"
    assert result["provenance"]["estimator"] == "local_law_dense_grid_o2b"
    assert result["summary"]["grid_resolution"] == 65
    assert result["summary"]["min_support_per_grid"] == 20
    assert result["summary"]["n_independent_A_B_transition_segments"] >= 5
    assert result["series"]["support_count"].shape == (state.shape[0],)
    assert result["series"]["grid_support_count"].shape == (65,)
    assert np.allclose(result["series"]["q_grid"], np.linspace(0.0, 1.0, 65), atol=1e-6)


def test_od_tq2_o2b_drift_binning_matches_reference_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bincount drift aggregation preserves non-zero, tied-bin means."""
    state, time, reaction, regime, segment_id = _truth_known_local_ensembles()
    query_grid = np.linspace(-1.0, 1.0, 65)
    for grid_index, value in enumerate(query_grid):
        pair_start = 2 * grid_index * 20
        reaction[pair_start : pair_start + 40] = value
    reaction[0] = 0.5 * (query_grid[0] + query_grid[1])
    state[:, 0] = reaction
    assert abs(reaction[0] - query_grid[0]) == abs(reaction[0] - query_grid[1])
    source_idx = np.arange(0, reaction.size, 2, dtype=np.int32)
    source_idx = np.concatenate([source_idx, np.asarray([0], dtype=np.int32)])
    increments = (
        0.001
        + 0.000001 * np.arange(source_idx.size, dtype=float)
    )[:, None]
    dts = np.ones(source_idx.size, dtype=float)

    def _synthetic_increment_pairs(*_args, **_kwargs):
        return source_idx, increments, dts

    monkeypatch.setattr(committor_module, "increment_pairs", _synthetic_increment_pairs)
    result = estimate_committor_local_law_dense_grid_o2b(
        state,
        time,
        reaction,
        regime,
        set_A=[0],
        set_B=[1],
        grid_min=-1.0,
        grid_max=1.0,
        diffusion_coefficient=0.25,
        segment_id=segment_id,
        min_samples=100,
        min_support_per_grid=19,
    )

    source_reaction = reaction[source_idx]
    in_bounds = (source_reaction >= -1.0) & (source_reaction <= 1.0)
    nearest_grid = np.argmin(
        np.abs(source_reaction[in_bounds, None] - query_grid[None, :]),
        axis=1,
    )
    drift_values = increments[in_bounds, 0] / dts[in_bounds]
    expected = np.asarray(
        [
            np.mean(drift_values[nearest_grid == grid_index])
            for grid_index in range(query_grid.size)
        ],
        dtype=np.float32,
    )

    assert result["series"]["grid_support_count"][0] == 21
    assert result["series"]["grid_support_count"][1] == 20
    np.testing.assert_allclose(
        result["series"]["drift_estimate_grid"],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_od_tq2_refuses_coarse_grid_and_under_supported_grid() -> None:
    state, time, reaction, regime, segment_id = _truth_known_local_ensembles()
    coarse = estimate_committor_local_law_dense_grid_o2b(
        state,
        time,
        reaction,
        regime,
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=0.25,
        segment_id=segment_id,
        grid_resolution=33,
        min_support_per_grid=20,
    )
    assert coarse["computation_status"] == "not_testable"
    assert coarse["failure_reason"] == "o2b_grid_resolution_below_minimum"

    under_supported = estimate_committor_local_law_dense_grid_o2b(
        state,
        time,
        reaction,
        regime,
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=0.25,
        segment_id=segment_id,
        grid_resolution=65,
        min_support_per_grid=21,
    )
    assert under_supported["computation_status"] == "insufficient_support"
    assert under_supported["failure_reason"] == "dense_query_grid_has_under_supported_points"


def test_od_tq2_adapter_payload_and_hdf5_chain(tmp_path: Path) -> None:
    state, time, reaction, regime, segment_id = _truth_known_local_ensembles()
    config = _config()
    layer_export = _build_dynamical_families_export_for_layers(
        config=config,
        x_subject_anchored=state.astype(np.float32),
        x_cohort_anchored=None,
        time=time,
        stage=regime,
        segment_id=segment_id,
        reaction_coordinate=reaction,
        reaction_coordinate_name="q_coordinate",
    )
    assert layer_export["destination"]["computation_status"] == "computed"
    assert (
        layer_export["destination"]["provenance"]["settings"]["reaction_coordinate_name"]
        == "q_coordinate"
    )
    export = build_dynamical_families_export(
        config=config,
        state=state,
        time=time,
        stage=regime,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=reaction,
        reaction_coordinate_name="q_coordinate",
    )
    result = export["destination"]
    assert result["computation_status"] == "computed"
    assert result["provenance"]["qualification_id"] == "OD-TQ2-test-certificate"
    assert result["provenance"]["validation_level"] == "mndm_translation_validated"

    payload = MNPSPayload(
        time=time,
        x=state.astype(np.float32),
        x_dot=np.zeros_like(state, dtype=np.float32),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "od_tq2.h5", "od_tq2", payload)
    with h5py.File(output, "r") as handle:
        family = handle["dynamical_families/destination/v1"]
        assert family.attrs["_schema_version"] == "mndm.committor.v1"
        assert family["computation_status"][()].decode() == "computed"
        assert family["series/q_grid"].shape == (65,)
        assert family["series/reaction_coordinate"].shape == (state.shape[0],)
        assert family["series/support_count"].shape == (state.shape[0],)
        assert family["series/grid_support_count"].shape == (65,)
        assert family["summary/grid_resolution"][()] == 65
        assert family["provenance/qualification_id"][()].decode() == "OD-TQ2-test-certificate"
        assert family["provenance/qualification_contract_hash"][()].decode() == "od-tq2-test-hash"
        assert family["provenance/validation_level"][()].decode() == "mndm_translation_validated"
        assert "jacobian/derived_metrics" not in handle


def test_od_tq2_requires_explicit_regime_and_reaction_coordinate() -> None:
    state, time, reaction, regime, segment_id = _truth_known_local_ensembles()
    config = _config(qualified=True)
    config["dynamical_families"]["destination"]["regime_source"] = "stage"
    missing_contract = build_dynamical_families_export(
        config=config,
        state=state,
        time=time,
        stage=regime,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=reaction,
    )
    assert missing_contract["destination"]["computation_status"] == "not_testable"
    assert (
        missing_contract["destination"]["failure_reason"]
        == "explicit_A_B_reaction_coordinate_contract_not_configured"
    )

    config = _config(qualified=True)
    missing_coordinate = _build_dynamical_families_export_for_layers(
        config=config,
        x_subject_anchored=state.astype(np.float32),
        x_cohort_anchored=None,
        time=time,
        stage=regime,
        segment_id=segment_id,
        reaction_coordinate=None,
    )
    assert missing_coordinate["destination"]["computation_status"] == "not_testable"
    assert (
        missing_coordinate["destination"]["failure_reason"]
        == "explicit_A_B_reaction_coordinate_contract_not_configured"
    )


def test_od_tq2_fail_closed_qualification_estimator_sets_and_timestep() -> None:
    state, time, reaction, regime, segment_id = _truth_known_local_ensembles()

    unqualified = build_dynamical_families_export(
        config=_config(qualified=False),
        state=state,
        time=time,
        stage=regime,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=reaction,
    )
    assert unqualified["destination"]["computation_status"] == "not_testable"
    assert (
        unqualified["destination"]["failure_reason"]
        == "mndm_committor_adapter_translation_qualification_required"
    )

    unsupported_config = _config()
    unsupported_config["dynamical_families"]["destination"]["estimator"] = (
        "local_first_hit_outcome_average"
    )
    unsupported = build_dynamical_families_export(
        config=unsupported_config,
        state=state,
        time=time,
        stage=regime,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=reaction,
    )
    assert unsupported["destination"]["computation_status"] == "not_testable"
    assert unsupported["destination"]["failure_reason"] == "unsupported_committor_estimator"

    missing_qualification_metadata = _config()
    missing_qualification_metadata["dynamical_families"]["destination"][
        "translation_qualification"
    ] = {"qualified": True}
    missing_metadata = build_dynamical_families_export(
        config=missing_qualification_metadata,
        state=state,
        time=time,
        stage=regime,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=reaction,
    )
    assert missing_metadata["destination"]["computation_status"] == "not_testable"
    assert (
        missing_metadata["destination"]["failure_reason"]
        == "committor_qualification_metadata_required"
    )

    malformed_boundary_config = _config()
    malformed_boundary_config["dynamical_families"]["destination"]["reaction_coordinate"][
        "boundaries"
    ] = [-0.8]
    malformed_boundaries = build_dynamical_families_export(
        config=malformed_boundary_config,
        state=state,
        time=time,
        stage=regime,
        segment_id=segment_id,
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        reaction_coordinate=reaction,
    )
    assert malformed_boundaries["destination"]["computation_status"] == "invalid"
    assert (
        malformed_boundaries["destination"]["failure_reason"]
        == "reaction_coordinate_boundaries_required"
    )

    invalid_sets = estimate_committor_local_law_dense_grid_o2b(
        state,
        time,
        reaction,
        regime,
        set_A=[],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=0.25,
        segment_id=segment_id,
        min_support_per_grid=20,
    )
    assert invalid_sets["computation_status"] == "invalid"
    assert invalid_sets["failure_reason"] == "invalid_or_overlapping_regime_sets"

    irregular_time = time.copy()
    irregular_time[1] += 0.001
    irregular = estimate_committor_local_law_dense_grid_o2b(
        state,
        irregular_time,
        reaction,
        regime,
        set_A=[0],
        set_B=[1],
        grid_min=-0.8,
        grid_max=0.8,
        diffusion_coefficient=0.25,
        segment_id=segment_id,
        min_support_per_grid=20,
    )
    assert irregular["computation_status"] == "not_testable"
    assert irregular["failure_reason"] == "materially_irregular_increment_timestep"

"""Final joint integration audit for standard orthogonal-dynamics v1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers
from mndm.schema import MNPSPayload


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "mndm" / "config"
AUDIT_ROOT = ROOT / "project" / "orthagonal_axis" / "results" / "od_v1_integration_audit"

DIFFUSION_ID = "OD-TQ1-v1-audit"
DIFFUSION_HASH = "od-tq1-v1-audit-hash"
COMMITTOR_ID = "OD-TQ2-v1-audit"
COMMITTOR_HASH = "od-tq2-v1-audit-hash"
RESILIENCE_ID = "OD-TQ3-v1-audit"
RESILIENCE_HASH = "od-tq3-v1-audit-hash"


def _fixture() -> dict[str, np.ndarray | dict[str, object]]:
    grid = np.linspace(-0.8, 0.8, 65)
    support_per_grid = 20
    reaction_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    segment_parts: list[np.ndarray] = []
    for grid_index, value in enumerate(grid):
        reaction_parts.append(np.repeat(value, 2 * support_per_grid))
        labels_parts.append(
            np.tile(np.asarray([0, 1], dtype=np.int8), support_per_grid)
        )
        segment_parts.append(
            np.repeat(
                np.arange(
                    grid_index * support_per_grid,
                    (grid_index + 1) * support_per_grid,
                    dtype=np.int32,
            ),
                2,
            )
        )
    reaction = np.concatenate(reaction_parts)
    labels = np.concatenate(labels_parts)
    segment_id = np.concatenate(segment_parts)
    time = np.arange(reaction.size, dtype=np.float64) * 0.001
    rng = np.random.default_rng(404)
    state = np.vstack(
        [
            np.zeros((1, 3), dtype=np.float64),
            np.cumsum(
                rng.normal(0.0, 0.3 * np.sqrt(0.001), (reaction.size - 1, 3)),
                axis=0,
            ),
        ]
    )

    amplitude_counts = [1300, 650, 650]
    amplitudes = np.repeat(np.asarray([0.0, 1.0, 2.0]), amplitude_counts)
    returned = np.concatenate(
        [
            np.ones(1300, dtype=np.int8),
            np.concatenate([np.ones(390, dtype=np.int8), np.zeros(260, dtype=np.int8)]),
            np.concatenate([np.ones(195, dtype=np.int8), np.zeros(455, dtype=np.int8)]),
        ]
    )
    recovery_time = np.where(returned.astype(bool), 0.5 + 0.25 * amplitudes, np.nan)
    protocol = {
        "perturbation_direction": "positive_x",
        "perturbation_time": "t0",
        "reference_attractor_or_state": "x_A",
        "return_criterion": "entered_reference_ball",
        "escape_criterion": "crossed_escape_boundary",
        "observation_horizon": 10.0,
        "non_return_is_escape": True,
    }
    return {
        "state": state.astype(np.float32),
        "time": time,
        "reaction": reaction,
        "labels": labels,
        "segment_id": segment_id,
        "amplitudes": amplitudes.astype(np.float64),
        "returned": returned,
        "recovery_time": recovery_time.astype(np.float64),
        "protocol": protocol,
    }


def _config(
    *,
    qualified: bool = True,
    include_committor_inputs: bool = True,
    include_resilience_inputs: bool = True,
) -> dict:
    committor = {
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
            "qualification_id": COMMITTOR_ID if qualified else None,
            "qualification_contract_hash": COMMITTOR_HASH if qualified else None,
        },
        "grid_resolution": 65,
        "set_A": [0],
        "set_B": [1],
        "diffusion_coefficient": 0.25,
        "min_samples": 100,
        "min_support_per_grid": 20,
        "min_transition_segments": 5,
    }
    if not include_committor_inputs:
        committor["regime_source"] = None
        committor["label_key"] = None
        committor["reaction_coordinate"]["key"] = None

    resilience = {
        "enabled": True,
        "protocol_source": (
            "explicit_perturbation_outcomes" if include_resilience_inputs else None
        ),
        "estimator": "observed_perturbation_outcome_summary",
        "amplitude_key": "perturbation_amplitude",
        "returned_key": "returned_to_reference",
        "recovery_time_key": "recovery_time_sec",
        "min_trials_per_amplitude": 20,
        "protocol": (
            {
                "perturbation_direction": "positive_x",
                "perturbation_time": "t0",
                "reference_attractor_or_state": "x_A",
                "return_criterion": "entered_reference_ball",
                "escape_criterion": "crossed_escape_boundary",
                "observation_horizon": 10.0,
                "non_return_is_escape": True,
            }
            if include_resilience_inputs
            else {}
        ),
        "translation_qualification": {
            "qualified": qualified,
            "qualification_id": RESILIENCE_ID if qualified else None,
            "qualification_contract_hash": RESILIENCE_HASH if qualified else None,
        },
    }

    return {
        "dynamical_families": {
            "enabled": True,
            "diffusion": {
                "enabled": True,
                "neighborhood": {"k": 200},
                "min_samples": 100,
                "min_neighborhood_samples": 20,
                "translation_qualification": {
                    "qualified": qualified,
                    "qualification_id": DIFFUSION_ID if qualified else None,
                    "qualification_contract_hash": DIFFUSION_HASH if qualified else None,
                },
            },
            "destination": committor,
            "resilience": resilience,
        }
    }


def _export(config: dict, fixture: dict) -> dict:
    return _build_dynamical_families_export_for_layers(
        config=config,
        x_subject_anchored=np.asarray(fixture["state"]),
        x_cohort_anchored=None,
        time=np.asarray(fixture["time"]),
        stage=np.asarray(fixture["labels"]),
        segment_id=np.asarray(fixture["segment_id"]),
        reaction_coordinate=np.asarray(fixture["reaction"]),
        reaction_coordinate_name="q_coordinate",
        perturbation_amplitudes=np.asarray(fixture["amplitudes"]),
        returned_to_reference=np.asarray(fixture["returned"]),
        recovery_time_sec=np.asarray(fixture["recovery_time"]),
        perturbation_protocol=dict(fixture["protocol"]),
    )


def _payload(fixture: dict, orthogonal_dynamics: dict) -> MNPSPayload:
    state = np.asarray(fixture["state"])
    return MNPSPayload(
        time=np.asarray(fixture["time"]),
        x=state,
        x_dot=np.zeros_like(state),
        jacobian=np.eye(3, dtype=np.float32)[None, :, :],
        jacobian_centers=np.asarray([0], dtype=np.int32),
        dynamical_families=orthogonal_dynamics,
    )


def _assert_family_provenance(
    family: h5py.Group,
    *,
    family_id: str,
    family_hash: str,
    measurement_validity: str = "translation_qualified",
) -> None:
    assert family["computation_status"][()].decode() == "computed"
    assert family["measurement_validity"][()].decode() == measurement_validity
    assert family["claim_status"][()].decode() == "no_biological_claim"
    assert (
        family["provenance/qualification_id"][()].decode() == family_id
    )
    assert (
        family["provenance/qualification_contract_hash"][()].decode()
        == family_hash
    )
    assert (
        family["provenance/validation_level"][()].decode()
        == "mndm_translation_validated"
    )


def test_od_v1_joint_three_family_hdf5_audit(tmp_path: Path) -> None:
    fixture = _fixture()
    export = _export(_config(), fixture)
    assert {
        family: export[family]["computation_status"]
        for family in (
            "diffusion",
            "destination",
            "resilience",
        )
    } == {
        "diffusion": "computed",
        "destination": "computed",
        "resilience": "computed",
    }
    assert export["diffusion"]["measurement_validity"] == "not_assessed"
    assert export["destination"]["measurement_validity"] == "translation_qualified"
    assert export["resilience"]["measurement_validity"] == "translation_qualified"

    joint_path = tmp_path / "od_v1_joint.h5"
    write_h5(joint_path, "od_v1_joint", _payload(fixture, export))
    with h5py.File(joint_path, "r") as handle:
        _assert_family_provenance(
            handle["dynamical_families/diffusion/v1"],
            family_id=DIFFUSION_ID,
            family_hash=DIFFUSION_HASH,
            measurement_validity="not_assessed",
        )
        _assert_family_provenance(
            handle["dynamical_families/destination/v1"],
            family_id=COMMITTOR_ID,
            family_hash=COMMITTOR_HASH,
        )
        _assert_family_provenance(
            handle["dynamical_families/resilience/v1"],
            family_id=RESILIENCE_ID,
            family_hash=RESILIENCE_HASH,
        )
        assert "experimental" not in handle
        assert "jacobian/derived_metrics" not in handle
        assert handle["mnps_3d"].shape == np.asarray(fixture["state"]).shape
        assert handle["jacobian/J_hat"].shape == (1, 3, 3)

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    archive_h5 = AUDIT_ROOT / "od_v1_joint.h5"
    write_h5(archive_h5, "od_v1_joint", _payload(fixture, export))
    (AUDIT_ROOT / "audit.json").write_text(
        json.dumps(
            {
                "stage": "OD-v1-integration-audit",
                "verdict": "PASS",
                "families": {
                    "diffusion": {
                        "status": "computed",
                        "measurement_validity": "not_assessed",
                        "validation_level": "mndm_translation_validated",
                        "qualification_id": DIFFUSION_ID,
                        "qualification_contract_hash": DIFFUSION_HASH,
                    },
                    "destination": {
                        "status": "computed",
                        "validation_level": "mndm_translation_validated",
                        "qualification_id": COMMITTOR_ID,
                        "qualification_contract_hash": COMMITTOR_HASH,
                    },
                    "resilience": {
                        "status": "computed",
                        "validation_level": "mndm_translation_validated",
                        "qualification_id": RESILIENCE_ID,
                        "qualification_contract_hash": RESILIENCE_HASH,
                    },
                },
                "hdf5_path": "/dynamical_families/<family>/v1",
                "experimental_namespace_absent": True,
                "jacobian_derived_metrics_absent": True,
                "failed_computation_not_translation_validated": True,
                "fixture_seed": 404,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_od_v1_eligibility_and_qualification_are_independent() -> None:
    fixture = _fixture()
    missing_inputs = _export(
        _config(include_committor_inputs=False, include_resilience_inputs=False),
        fixture,
    )
    assert missing_inputs["diffusion"]["computation_status"] == "computed"
    assert (
        missing_inputs["destination"]["failure_reason"]
        == "explicit_A_B_reaction_coordinate_contract_not_configured"
    )
    assert (
        missing_inputs["resilience"]["failure_reason"]
        == "no_perturbation_protocol"
    )

    unqualified = _export(_config(qualified=False), fixture)
    assert unqualified["diffusion"]["computation_status"] == "computed"
    assert unqualified["diffusion"]["measurement_validity"] == "not_assessed"
    assert not unqualified["diffusion"]["provenance"].get("qualification_id")
    assert unqualified["destination"]["computation_status"] == "not_testable"
    assert unqualified["resilience"]["computation_status"] == "not_testable"
    assert (
        unqualified["destination"]["failure_reason"]
        == "mndm_committor_adapter_translation_qualification_required"
    )
    assert (
        unqualified["resilience"]["failure_reason"]
        == "mndm_resilience_adapter_translation_qualification_required"
    )


def test_od_v1_does_not_change_mnps_or_jacobian_outputs(tmp_path: Path) -> None:
    fixture = _fixture()
    export = _export(_config(), fixture)
    with_od = write_h5(
        tmp_path / "with_od.h5",
        "with_od",
        _payload(fixture, export),
    )
    without_od = write_h5(
        tmp_path / "without_od.h5",
        "without_od",
        _payload(fixture, {}),
    )
    with h5py.File(with_od, "r") as with_handle, h5py.File(
        without_od, "r"
    ) as without_handle:
        for path in ("mnps_3d", "mnps_3d_dot", "jacobian/J_hat"):
            assert np.array_equal(with_handle[path][:], without_handle[path][:])


def test_od_v1_standard_profiles_do_not_enable_dynamical_families() -> None:
    profile_paths = [
        path
        for path in CONFIG_ROOT.rglob("*.yaml")
        if path.name != "config_ingest_common_dynamical_families.yaml"
        and "dynamical_families" not in path.name
        and "od_epi" not in path.name
    ]
    assert profile_paths
    for path in profile_paths:
        text = path.read_text(encoding="utf-8")
        assert "orthogonal_dynamics:" not in text, path
        assert "dynamical_families:" not in text, path
        assert "config_ingest_common_dynamical_families.yaml" not in text, path

"""OD-TQ1 pipeline-to-payload-to-HDF5 replay."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers
from mndm.schema import MNPSPayload


def _truth_known_state() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(404)
    n, dt, sigma = 900, 0.01, 0.3
    state = np.vstack(
        [np.zeros((1, 3)), np.cumsum(rng.normal(0.0, sigma * np.sqrt(dt), (n - 1, 3)), axis=0)]
    )
    return state.astype(np.float32), np.arange(n, dtype=np.float64) * dt


def _config(*, qualified: bool) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "diffusion": {
                "enabled": True,
                "neighborhood": {"k": 899},
                "min_samples": 100,
                "min_neighborhood_samples": 100,
                "translation_qualification": {
                    "qualified": qualified,
                    "qualification_id": "OD-TQ1-test-certificate" if qualified else None,
                    "qualification_contract_hash": "test-hash" if qualified else None,
                },
            },
        }
    }


def test_od_tq1_replays_adapter_payload_and_hdf5_chain(tmp_path: Path) -> None:
    state, time = _truth_known_state()
    segment_id = np.where(np.arange(time.size) < time.size // 2, 0, 1).astype(np.int32)
    export = _build_dynamical_families_export_for_layers(
        config=_config(qualified=True),
        x_subject_anchored=state,
        x_cohort_anchored=state + 100.0,
        time=time,
        stage=None,
        segment_id=segment_id,
    )
    result = export["diffusion"]
    assert result["computation_status"] == "computed"
    assert result["measurement_validity"] == "not_assessed"
    assert result["claim_status"] == "no_biological_claim"
    assert result["grain"]["native"] == "window"
    assert result["grain"]["biological_unit"] == "subject"
    assert result["grain"]["direct_between_subject_inference"] == "forbidden"
    assert result["provenance"]["qualification_id"] == "OD-TQ1-test-certificate"
    assert result["provenance"]["qualification_contract_hash"] == "test-hash"
    assert result["provenance"]["validation_level"] == "mndm_translation_validated"
    assert result["provenance"]["coordinate_layer"] == "coords_3d_subject_anchored"
    assert result["summary"]["n_increment_pairs"] == time.size - 2
    assert result["summary"]["A_bD_computation_status"] == "not_testable"
    assert result["summary"]["R_b_over_a_computation_status"] == "not_testable"
    assert result["summary"]["drift_alignment_failure_reason"] == "independent_drift_not_supplied"

    payload = MNPSPayload(
        time=time,
        x=state,
        x_dot=np.zeros((time.size, 3), dtype=np.float32),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "od_tq1.h5", "od_tq1", payload)
    with h5py.File(output, "r") as handle:
        family = handle["dynamical_families/diffusion/v1"]
        assert family.attrs["_schema_version"] == "mndm.diffusion_geometry.v1"
        assert "series/diffusion_tensor" in family
        assert "series/a_hat" in family
        assert family["series/diffusion_tensor"].shape == (time.size, 3, 3)
        assert np.isclose(
            np.nanmean(family["series/diffusion_total"][:]),
            3 * 0.3**2,
            rtol=0.25,
        )
        assert "provenance/qualification_id" in family
        assert family["provenance/qualification_id"][()].decode() == "OD-TQ1-test-certificate"
        assert family["measurement_validity"][()].decode() == "not_assessed"
        assert family["claim_status"][()].decode() == "no_biological_claim"
        assert family["grain/native"][()].decode() == "window"
        assert family["grain/direct_between_subject_inference"][()].decode() == "forbidden"
        assert family["summary/A_bD_computation_status"][()].decode() == "not_testable"
        assert family["summary/R_b_over_a_computation_status"][()].decode() == "not_testable"
        assert (
            family["summary/drift_alignment_failure_reason"][()].decode()
            == "independent_drift_not_supplied"
        )
        assert np.all(np.isnan(family["series/A_bD"][:]))
        assert np.all(np.isnan(family["series/R_b_over_a"][:]))
        assert "jacobian/derived_metrics" not in handle


def test_od_tq1_unqualified_adapter_still_computes_when_estimator_has_support(tmp_path: Path) -> None:
    state, time = _truth_known_state()
    export = _build_dynamical_families_export_for_layers(
        config=_config(qualified=False),
        x_subject_anchored=state,
        x_cohort_anchored=None,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
    )
    result = export["diffusion"]
    assert result["computation_status"] == "computed"
    assert result["measurement_validity"] == "not_assessed"
    assert result["claim_status"] == "no_biological_claim"
    assert result["grain"]["native"] == "window"
    assert not result["provenance"].get("qualification_id")
    payload = MNPSPayload(
        time=time,
        x=state,
        x_dot=np.zeros((time.size, 3), dtype=np.float32),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "od_tq1_unqualified.h5", "od_tq1", payload)
    with h5py.File(output, "r") as handle:
        family = handle["dynamical_families/diffusion/v1"]
        assert family["computation_status"][()].decode() == "computed"
        assert family["measurement_validity"][()].decode() == "not_assessed"
        assert family["claim_status"][()].decode() == "no_biological_claim"
        assert "series/a_hat" in family


def test_od_tq1_missing_qualification_metadata_still_computes() -> None:
    state, time = _truth_known_state()
    config = _config(qualified=True)
    config["dynamical_families"]["diffusion"][
        "translation_qualification"
    ] = {"qualified": True}
    export = _build_dynamical_families_export_for_layers(
        config=config,
        x_subject_anchored=state,
        x_cohort_anchored=None,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
    )
    result = export["diffusion"]
    assert result["computation_status"] == "computed"
    assert result["measurement_validity"] == "not_assessed"
    assert not result["provenance"].get("qualification_id")


def test_od_tq1_failed_computation_does_not_receive_translation_validation() -> None:
    state, time = _truth_known_state()
    irregular_time = time.copy()
    irregular_time[40:] += 1.0
    export = _build_dynamical_families_export_for_layers(
        config=_config(qualified=True),
        x_subject_anchored=state,
        x_cohort_anchored=None,
        time=irregular_time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
    )
    result = export["diffusion"]
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "materially_irregular_increment_timestep"
    assert result["provenance"]["validation_level"] == "simulator_validated"
    assert result["provenance"]["qualification_id"] is None
    assert result["provenance"]["qualification_contract_hash"] is None

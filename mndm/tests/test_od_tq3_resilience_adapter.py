"""OD-TQ3 explicit finite-amplitude resilience protocol qualification."""

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

from od_tq3_fixture import (
    QUALIFICATION_HASH,
    QUALIFICATION_ID,
    PROTOCOL,
    truth_outcomes,
)


def _config(
    *,
    qualified: bool = True,
    qualification_metadata: bool = True,
    estimator: str = "observed_perturbation_outcome_summary",
) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "resilience": {
                "enabled": True,
                "protocol_source": "explicit_perturbation_outcomes",
                "estimator": estimator,
                "amplitude_key": "perturbation_amplitude",
                "returned_key": "returned_to_reference",
                "recovery_time_key": "recovery_time_sec",
                "min_trials_per_amplitude": 20,
                "protocol": dict(PROTOCOL),
                "translation_qualification": {
                    "qualified": qualified,
                    "qualification_id": (
                        QUALIFICATION_ID if qualification_metadata else None
                    ),
                    "qualification_contract_hash": (
                        QUALIFICATION_HASH if qualification_metadata else None
                    ),
                },
            },
        }
    }


def _export(
    config: dict,
    *,
    amplitudes: np.ndarray | None = None,
    returned: np.ndarray | None = None,
    survived: np.ndarray | None = None,
    recovery_time_sec: np.ndarray | None = None,
    protocol: dict[str, object] | None = None,
) -> dict:
    fixture = truth_outcomes()
    return build_dynamical_families_export(
        config=config,
        state=np.asarray(fixture["state"]),
        time=np.asarray(fixture["time"]),
        stage=None,
        segment_id=np.zeros(len(fixture["time"]), dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        perturbation_amplitudes=(
            np.asarray(fixture["amplitudes"]) if amplitudes is None else amplitudes
        ),
        returned_to_reference=(
            np.asarray(fixture["returned"]) if returned is None else returned
        ),
        perturbation_survived=survived,
        recovery_time_sec=(
            np.asarray(fixture["recovery_time_sec"])
            if recovery_time_sec is None
            else recovery_time_sec
        ),
        perturbation_protocol=(
            dict(fixture["protocol"]) if protocol is None else protocol
        ),
    )


def test_od_tq3_explicit_outcomes_replay_adapter_payload_and_hdf5(tmp_path: Path) -> None:
    fixture = truth_outcomes()
    config = _config()
    layer_export = _build_dynamical_families_export_for_layers(
        config=config,
        x_subject_anchored=np.asarray(fixture["state"]),
        x_cohort_anchored=None,
        time=np.asarray(fixture["time"]),
        stage=None,
        segment_id=np.zeros(len(fixture["time"]), dtype=np.int32),
        perturbation_amplitudes=np.asarray(fixture["amplitudes"]),
        returned_to_reference=np.asarray(fixture["returned"]),
        recovery_time_sec=np.asarray(fixture["recovery_time_sec"]),
        perturbation_protocol=dict(fixture["protocol"]),
    )
    result = layer_export["resilience"]
    assert result["computation_status"] == "computed"
    assert result["grain"]["native"] == "event"
    assert result["grain"]["parent"] == "recording"
    assert result["grain"]["direct_between_subject_inference"] == "forbidden"
    curve = result["amplitude_curve"]
    assert [row["amplitude"] for row in curve] == [0.0, 1.0, 2.0]
    assert [row["return_fraction"] for row in curve] == [1.0, 0.6, 0.3]
    assert [row["escape_fraction"] for row in curve] == [0.0, 0.4, 0.7]
    assert result["summary"]["r50_discrete_first_bin_at_or_below_half"] == 2.0
    assert "r_escape" not in result["summary"]
    assert result["provenance"]["settings"]["protocol"] == PROTOCOL
    assert result["provenance"]["qualification_id"] == QUALIFICATION_ID
    assert result["provenance"]["qualification_contract_hash"] == QUALIFICATION_HASH
    assert result["provenance"]["validation_level"] == "mndm_translation_validated"

    payload = MNPSPayload(
        time=np.asarray(fixture["time"]),
        x=np.asarray(fixture["state"]),
        x_dot=np.zeros_like(np.asarray(fixture["state"])),
        dynamical_families=layer_export,
    )
    output = write_h5(tmp_path / "od_tq3.h5", "od_tq3", payload)
    with h5py.File(output, "r") as handle:
        family = handle["dynamical_families/resilience/v1"]
        assert family.attrs["_schema_version"] == "mndm.finite_amplitude_resilience.v1"
        assert family["computation_status"][()].decode() == "computed"
        assert family["grain/native"][()].decode() == "event"
        assert "amplitude_curve_json" in family
        curve = json.loads(family["amplitude_curve_json"][()].decode())
        assert [row["return_fraction"] for row in curve] == [1.0, 0.6, 0.3]
        assert family["summary/r50_discrete_first_bin_at_or_below_half"][()] == 2.0
        assert family["provenance/qualification_id"][()].decode() == QUALIFICATION_ID
        assert (
            family["provenance/qualification_contract_hash"][()].decode()
            == QUALIFICATION_HASH
        )
        assert (
            family["provenance/validation_level"][()].decode()
            == "mndm_translation_validated"
        )
        assert (
            family["provenance/settings/protocol/non_return_is_escape"][()] == 1
        )
        assert "jacobian/derived_metrics" not in handle


def test_od_tq3_observational_and_jacobian_only_data_fail_closed(tmp_path: Path) -> None:
    fixture = truth_outcomes()
    config = _config()
    result = build_dynamical_families_export(
        config=config,
        state=np.asarray(fixture["state"]),
        time=np.asarray(fixture["time"]),
        stage=None,
        segment_id=np.zeros(len(fixture["time"]), dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )["resilience"]
    assert result["computation_status"] == "not_testable"
    assert result["failure_reason"] == "no_perturbation_protocol"

    payload = MNPSPayload(
        time=np.asarray(fixture["time"]),
        x=np.asarray(fixture["state"]),
        x_dot=np.ones_like(np.asarray(fixture["state"])),
        dynamical_families={"resilience": result},
    )
    output = write_h5(tmp_path / "od_tq3_observational.h5", "od_tq3", payload)
    with h5py.File(output, "r") as handle:
        family = handle["dynamical_families/resilience/v1"]
        assert family["computation_status"][()].decode() == "not_testable"
        assert family["failure_reason"][()].decode() == "no_perturbation_protocol"

    incomplete = _export(_config(), protocol={})["resilience"]
    assert incomplete["computation_status"] == "not_testable"
    assert incomplete["failure_reason"] == "incomplete_perturbation_protocol"


def test_od_tq3_refuses_unqualified_or_incomplete_certificates() -> None:
    unqualified = _export(_config(qualified=False))["resilience"]
    assert unqualified["computation_status"] == "not_testable"
    assert (
        unqualified["failure_reason"]
        == "mndm_resilience_adapter_translation_qualification_required"
    )

    missing_metadata = _export(
        _config(qualified=True, qualification_metadata=False)
    )["resilience"]
    assert missing_metadata["computation_status"] == "not_testable"
    assert (
        missing_metadata["failure_reason"]
        == "resilience_qualification_metadata_required"
    )

    unsupported = _export(_config(estimator="jacobian_resilience_proxy"))[
        "resilience"
    ]
    assert unsupported["computation_status"] == "not_testable"
    assert unsupported["failure_reason"] == "unsupported_resilience_estimator"


def test_od_tq3_refuses_malformed_and_under_supported_outcomes() -> None:
    fixture = truth_outcomes()
    malformed = _export(
        _config(),
        amplitudes=np.asarray(fixture["amplitudes"])[:-1],
    )["resilience"]
    assert malformed["computation_status"] == "invalid"
    assert malformed["failure_reason"] == "amplitude_return_shape_mismatch"

    negative_recovery = _export(
        _config(),
        recovery_time_sec=-np.ones(len(fixture["amplitudes"])),
    )["resilience"]
    assert negative_recovery["computation_status"] == "invalid"
    assert negative_recovery["failure_reason"] == "negative_recovery_time"

    malformed_survival = _export(
        _config(),
        survived=np.ones(len(fixture["amplitudes"]) - 1),
    )["resilience"]
    assert malformed_survival["computation_status"] == "invalid"
    assert malformed_survival["failure_reason"] == "survival_shape_mismatch"

    malformed_recovery = _export(
        _config(),
        recovery_time_sec=np.ones(len(fixture["amplitudes"]) - 1),
    )["resilience"]
    assert malformed_recovery["computation_status"] == "invalid"
    assert malformed_recovery["failure_reason"] == "recovery_time_shape_mismatch"

    insufficient = _export(
        _config(),
        amplitudes=np.asarray([0.0] * 19),
        returned=np.asarray([1] * 19),
        recovery_time_sec=np.asarray([0.1] * 19),
    )["resilience"]
    assert insufficient["computation_status"] == "insufficient_support"
    assert (
        insufficient["failure_reason"] == "insufficient_valid_perturbation_trials"
    )

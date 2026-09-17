"""Grain is schema metadata; qualification_status is not computation_status.

There is no global ``qualified`` flag. YAML ``translation_qualification.qualified``
is a destination/resilience gate only.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import WRITABLE_FAMILY_IDS
from mndm.dynamical_families.contracts import (
    AFFINE_ONE_STEP_SCHEMA_VERSION,
    AMPLIFICATION_SCHEMA_VERSION,
    CHART_DRIFT_SCHEMA_VERSION,
    COMMITTOR_SCHEMA_VERSION,
    DIFFUSION_GEOMETRY_SCHEMA_VERSION,
    FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
    HISTORY_SCHEMA_VERSION,
    TURNING_SCHEMA_VERSION,
    unavailable_result,
)
from mndm.dynamical_families.measurement_register import QUALIFICATION_NOT_ASSESSED
from mndm.inferential_grain import GRAIN_BY_FAMILY_ID, GRAIN_BY_SCHEMA
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload, normalize_payload


def test_writable_families_have_registered_grain() -> None:
    assert set(GRAIN_BY_FAMILY_ID) == set(WRITABLE_FAMILY_IDS)
    for family_id, spec in GRAIN_BY_FAMILY_ID.items():
        assert spec["native"] in {
            "window",
            "recording",
            "event",
            "transition",
            "recording_horizon",
        }
        assert spec["parent"] in {"recording", "subject"}
        assert spec["repeated_measure"] in {"true", "false"}


def test_unavailable_result_has_grain_and_qualification_not_computation() -> None:
    for schema in (
        DIFFUSION_GEOMETRY_SCHEMA_VERSION,
        COMMITTOR_SCHEMA_VERSION,
        FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
        CHART_DRIFT_SCHEMA_VERSION,
        AFFINE_ONE_STEP_SCHEMA_VERSION,
        AMPLIFICATION_SCHEMA_VERSION,
        HISTORY_SCHEMA_VERSION,
        TURNING_SCHEMA_VERSION,
    ):
        result = unavailable_result(
            schema,
            status="not_testable",
            failure_reason="explicit_refusal",
        )
        assert result["computation_status"] == "not_testable"
        assert result["qualification_status"] == QUALIFICATION_NOT_ASSESSED
        assert result["qualification_status"] != result["computation_status"]
        assert "qualified" not in result
        assert result["grain"]["native"] == GRAIN_BY_SCHEMA[schema]["native"]
        assert result["grain"]["biological_unit"] == "subject"
        assert result["measurement_validity"] == "not_applicable"
        assert result["claim_status"] == "no_biological_claim"


def test_export_stamps_grain_and_qualification_without_global_qualified() -> None:
    state = np.zeros((24, 3), dtype=float)
    time = np.arange(24, dtype=float) * 0.01
    export = build_dynamical_families_export(
        config={
            "dynamical_families": {
                "enabled": True,
                "diffusion": {"enabled": True},
                "destination": {"enabled": True},
                "resilience": {"enabled": True},
                "drift": {"enabled": True},
                "one_step": {"enabled": True},
                "amplification": {"enabled": True},
                "history": {"enabled": True},
                "turning": {"enabled": True},
            }
        },
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    assert set(export) == set(WRITABLE_FAMILY_IDS)
    for family_id, payload in export.items():
        spec = GRAIN_BY_FAMILY_ID[family_id]
        grain = payload["grain"]
        assert grain["native"] == spec["native"]
        assert grain["parent"] == spec["parent"]
        assert grain["repeated_measure"] == spec["repeated_measure"]
        assert grain["biological_unit"] == "subject"
        assert grain["direct_between_subject_inference"] == "forbidden"
        assert str(payload.get("computation_status") or "")
        assert isinstance(payload.get("qualification_status"), str)
        assert payload["qualification_status"]
        assert "qualified" not in payload
        assert payload.get("measurement_validity") in {
            "not_assessed",
            "not_applicable",
            "translation_qualified",
        }


def test_payload_refuses_missing_grain_and_qualification_status() -> None:
    with pytest.raises(ValueError, match="requires nested grain"):
        normalize_payload(
            MNPSPayload(
                time=np.array([0.0, 1.0]),
                x=np.zeros((2, 3), dtype=np.float32),
                x_dot=np.zeros((2, 3), dtype=np.float32),
                dynamical_families={
                    "turning": {
                        "schema_version": TURNING_SCHEMA_VERSION,
                        "computation_status": "computed",
                        "qualification_status": "not_assessed",
                    }
                },
            )
        )
    with pytest.raises(ValueError, match="requires qualification_status"):
        normalize_payload(
            MNPSPayload(
                time=np.array([0.0, 1.0]),
                x=np.zeros((2, 3), dtype=np.float32),
                x_dot=np.zeros((2, 3), dtype=np.float32),
                dynamical_families={
                    "turning": {
                        "schema_version": TURNING_SCHEMA_VERSION,
                        "computation_status": "computed",
                    }
                },
            )
        )


def test_payload_refuses_boolean_qualified_sibling() -> None:
    with pytest.raises(ValueError, match="qualified"):
        normalize_payload(
            MNPSPayload(
                time=np.array([0.0, 1.0]),
                x=np.zeros((2, 3), dtype=np.float32),
                x_dot=np.zeros((2, 3), dtype=np.float32),
                dynamical_families={
                    "turning": {
                        "schema_version": TURNING_SCHEMA_VERSION,
                        "computation_status": "computed",
                        "qualified": True,
                    }
                },
            )
        )

"""SL-LEV-MES-002 FAR ladder under 003 corrections.

Existing amplitude_curve is a documented FAR identity, not spontaneous return.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamical_families import family_forbids
from mndm.dynamical_families.measurement_register import (
    FORBIDDEN_RESILIENCE_CONFIG_KEYS,
    LOGICAL_PERTURBATION_ROOT,
    LOGICAL_POSSIBLE_FUTURES_PERTURBATION,
    MEASUREMENT_ID_EXCURSION_RECOVERY,
    MEASUREMENT_ID_FAR_P50,
    MEASUREMENT_ID_FAR_P90,
    MEASUREMENT_ID_FAR_RECOVERY,
    MEASUREMENT_ID_MATCHED_RECOVERY_L1,
    MEASUREMENT_ID_MATCHED_RECOVERY_L2,
    MEASUREMENT_ID_R50_DISCRETE,
    MEASUREMENT_ID_SPONTANEOUS_RETURN,
    QUALIFICATION_NOT_ASSESSED,
    QUALIFICATION_NOT_FAR,
    RELATION_DIAGNOSTICS,
    RELATION_DOCUMENTED_IDENTITY,
    RELATION_LOGICAL,
    RELATION_SERIES_ALIAS,
    RELATION_SUPERSEDED,
    RELATION_WITHHELD,
    compatibility_entry,
    register_entry,
)
from mndm.dynamical_families.resilience import summarize_finite_amplitude_resilience
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers
from mndm.schema import MNPSPayload
from od_tq3_fixture import PROTOCOL, QUALIFICATION_HASH, QUALIFICATION_ID, truth_outcomes

ROOT = Path(__file__).resolve().parents[2]
COMMON_FAMILIES = ROOT / "mndm" / "config" / "config_ingest_common_dynamical_families.yaml"
ICARE_FAMILIES = (
    ROOT
    / "mndm"
    / "config"
    / "sources"
    / "other"
    / "config_ingest_physionet_i-care_2_1_dynamical_families.yaml"
)


def _protocol_config(*, qualified: bool = True) -> dict:
    return {
        "dynamical_families": {
            "enabled": True,
            "resilience": {
                "enabled": True,
                "protocol_source": "explicit_perturbation_outcomes",
                "estimator": "observed_perturbation_outcome_summary",
                "min_trials_per_amplitude": 20,
                "protocol": dict(PROTOCOL),
                "translation_qualification": {
                    "qualified": qualified,
                    "qualification_id": QUALIFICATION_ID if qualified else None,
                    "qualification_contract_hash": QUALIFICATION_HASH if qualified else None,
                },
            },
        }
    }


def test_computed_curve_is_far_recovery_not_spontaneous_return() -> None:
    fixture = truth_outcomes()
    result = summarize_finite_amplitude_resilience(
        fixture["amplitudes"],
        fixture["returned"],
        recovery_time_sec=fixture["recovery_time_sec"],
        min_trials_per_amplitude=20,
        protocol=PROTOCOL,
    )
    assert result["computation_status"] == "computed"
    assert result["measurement_id"] == MEASUREMENT_ID_FAR_RECOVERY
    assert result["interpretation_level"] == 4
    assert result["claim_class"] == "finite_amplitude_response"
    assert result["qualification_status"] == QUALIFICATION_NOT_ASSESSED
    assert result["summary"]["not_inferred_from_spontaneous_trajectory"] is True
    assert result["summary"]["estimand"] == "observed_return_fraction_by_perturbation_amplitude"
    assert MEASUREMENT_ID_SPONTANEOUS_RETURN not in result
    assert MEASUREMENT_ID_FAR_P50 not in result
    row = result["amplitude_curve"][0]
    assert row["return_fraction"] == row["basin_return_probability"]
    assert result["summary"]["r50_discrete_first_bin_at_or_below_half"] == 2.0
    assert register_entry(MEASUREMENT_ID_R50_DISCRETE)["relation"] == RELATION_DIAGNOSTICS


def test_far_compatibility_withholds_observational_names() -> None:
    curve = compatibility_entry("amplitude_curve")
    basin = compatibility_entry("basin_return_probability")
    assert curve["relation"] == basin["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert curve["measurement_id"] == basin["measurement_id"] == MEASUREMENT_ID_FAR_RECOVERY
    assert compatibility_entry("finite_amplitude_resilience_level4")["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert compatibility_entry(MEASUREMENT_ID_FAR_RECOVERY)["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert compatibility_entry("return_fraction")["relation"] == RELATION_SERIES_ALIAS
    r50 = compatibility_entry("r50_discrete_first_bin_at_or_below_half")
    assert r50["relation"] == RELATION_DIAGNOSTICS
    assert r50["measurement_id"] == MEASUREMENT_ID_R50_DISCRETE
    assert compatibility_entry(MEASUREMENT_ID_FAR_P50)["relation"] == RELATION_WITHHELD
    assert compatibility_entry(MEASUREMENT_ID_FAR_P90)["physical_path"] is None
    assert compatibility_entry(MEASUREMENT_ID_SPONTANEOUS_RETURN)["relation"] == RELATION_WITHHELD
    assert compatibility_entry(MEASUREMENT_ID_EXCURSION_RECOVERY)["relation"] == RELATION_WITHHELD
    assert register_entry(MEASUREMENT_ID_EXCURSION_RECOVERY)["default_qualification_status"] == QUALIFICATION_NOT_FAR
    superseded = compatibility_entry(MEASUREMENT_ID_MATCHED_RECOVERY_L2)
    assert superseded["relation"] == RELATION_SUPERSEDED
    assert compatibility_entry(MEASUREMENT_ID_MATCHED_RECOVERY_L1)["relation"] == RELATION_WITHHELD
    assert compatibility_entry(LOGICAL_PERTURBATION_ROOT)["relation"] == RELATION_LOGICAL
    assert compatibility_entry(LOGICAL_POSSIBLE_FUTURES_PERTURBATION)["relation"] == RELATION_LOGICAL
    assert family_forbids("resilience", "spontaneous_return_as_far")
    assert family_forbids("resilience", "matched_perturbation_as_level2_model")
    assert family_forbids("resilience", "discrete_r50_as_far_threshold_p50")
    assert family_forbids("resilience", "amplitude_curve_as_spontaneous_return")
    assert family_forbids("resilience", "r50_as_far_threshold_p50")
    assert family_forbids("resilience", MEASUREMENT_ID_SPONTANEOUS_RETURN)
    assert family_forbids("resilience", MEASUREMENT_ID_FAR_P50)
    for key in FORBIDDEN_RESILIENCE_CONFIG_KEYS:
        assert family_forbids("resilience", key)


def test_denied_protocol_keeps_far_identity_not_zeros() -> None:
    state = np.zeros((40, 3), dtype=float)
    time = np.arange(40, dtype=float) * 0.01
    export = build_dynamical_families_export(
        config={
            "dynamical_families": {
                "enabled": True,
                "resilience": {"enabled": True, "protocol_source": None},
            }
        },
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    resilience = export["resilience"]
    assert resilience["computation_status"] == "not_testable"
    assert resilience["failure_reason"] == "no_perturbation_protocol"
    assert resilience["measurement_id"] == MEASUREMENT_ID_FAR_RECOVERY
    assert resilience["series"] == {}
    assert MEASUREMENT_ID_SPONTANEOUS_RETURN not in resilience
    assert MEASUREMENT_ID_FAR_P50 not in resilience


def test_computed_far_export_writes_identity_to_hdf5(tmp_path: Path) -> None:
    fixture = truth_outcomes()
    export = build_dynamical_families_export(
        config=_protocol_config(),
        state=np.asarray(fixture["state"]),
        time=np.asarray(fixture["time"]),
        stage=None,
        segment_id=np.zeros(len(fixture["time"]), dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
        perturbation_amplitudes=np.asarray(fixture["amplitudes"]),
        returned_to_reference=np.asarray(fixture["returned"]),
        recovery_time_sec=np.asarray(fixture["recovery_time_sec"]),
        perturbation_protocol=PROTOCOL,
    )
    resilience = export["resilience"]
    assert resilience["computation_status"] == "computed"
    assert resilience["measurement_id"] == MEASUREMENT_ID_FAR_RECOVERY
    payload = MNPSPayload(
        time=np.asarray(fixture["time"]),
        x=np.asarray(fixture["state"]),
        x_dot=np.zeros_like(fixture["state"]),
        dynamical_families=export,
    )
    output = write_h5(tmp_path / "far_levels.h5", "far_levels", payload)
    with h5py.File(output, "r") as handle:
        family = handle["dynamical_families/resilience/v1"]
        assert family["measurement_id"][()].decode() == MEASUREMENT_ID_FAR_RECOVERY
        assert int(family["interpretation_level"][()]) == 4
        assert "far_threshold_p50_level4" not in family
        assert "spontaneous_return_fraction_level0" not in family


def test_forbidden_resilience_yaml_keys_are_refused() -> None:
    state = np.zeros((20, 3), dtype=float)
    time = np.arange(20, dtype=float) * 0.01
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    for key in FORBIDDEN_RESILIENCE_CONFIG_KEYS:
        config = {
            "dynamical_families": {
                "enabled": True,
                "resilience": {"enabled": False, key: {"enabled": True}},
            }
        }
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=config, **kwargs)


def test_resilience_overlays_are_not_flipped_by_this_gate() -> None:
    common = yaml.safe_load(COMMON_FAMILIES.read_text(encoding="utf-8"))
    assert common["dynamical_families"]["resilience"]["enabled"] is True
    icare = yaml.safe_load(ICARE_FAMILIES.read_text(encoding="utf-8"))
    resilience = (icare["dynamical_families"].get("resilience") or {})
    assert resilience.get("enabled", False) is True
    for key in FORBIDDEN_RESILIENCE_CONFIG_KEYS:
        assert key not in resilience


def test_missing_coordinate_layer_stamps_far_identity() -> None:
    export = _build_dynamical_families_export_for_layers(
        config={
            "dynamical_families": {
                "enabled": True,
                "coordinate_layer": "subject_anchored",
                "resilience": {"enabled": True},
            }
        },
        x_subject_anchored=None,
        x_cohort_anchored=None,
        time=np.arange(8, dtype=float),
        stage=None,
        segment_id=None,
    )
    resilience = export["resilience"]
    assert resilience["computation_status"] == "not_testable"
    assert resilience["failure_reason"] == "requested_coordinate_layer_not_available"
    assert resilience["measurement_id"] == MEASUREMENT_ID_FAR_RECOVERY

"""SL-LEV-MES-002 hysteresis/recovery ladder under 003 corrections.

Matching is not automatically level2. Hysteresis is not FAR. No physical family.
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
from mndm.dynamical_families import WRITABLE_FAMILY_IDS, family_forbids, get_family
from mndm.inferential_grain import GRAIN_BY_FAMILY_ID, attach_grain
from mndm.dynamical_families.io import canonical_group_path
from mndm.dynamical_families.measurement_register import (
    CLAIM_CONDITIONAL_HORIZON,
    CLAIM_DESCRIPTIVE,
    CLAIM_DISCRETE_ONE_STEP,
    FORBIDDEN_HYSTERESIS_CONFIG_KEYS,
    LOGICAL_PERTURBATION_ROOT,
    LOGICAL_POSSIBLE_FUTURES_HYSTERESIS,
    MEASUREMENT_ID_EXCURSION_RECOVERY_TIME,
    MEASUREMENT_ID_MATCHED_RETURN_DISTANCE,
    MEASUREMENT_ID_PATH_ASYMMETRY_L1,
    MEASUREMENT_ID_PATH_ASYMMETRY_L2,
    MEASUREMENT_ID_RECOVERY_TIME_L1,
    MEASUREMENT_ID_RETURN_DISTANCE,
    QUALIFICATION_NOT_FAR,
    RELATION_LOGICAL,
    RELATION_SUPERSEDED,
    RELATION_WITHHELD,
    compatibility_entry,
    register_entry,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.schema import MNPSPayload, normalize_payload

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


def test_hysteresis_names_are_withheld_and_not_far() -> None:
    level0 = register_entry(MEASUREMENT_ID_RETURN_DISTANCE)
    assert level0["interpretation_level"] == 0
    assert level0["claim_class"] == CLAIM_DESCRIPTIVE
    for measurement_id in (
        MEASUREMENT_ID_MATCHED_RETURN_DISTANCE,
        MEASUREMENT_ID_RECOVERY_TIME_L1,
        MEASUREMENT_ID_PATH_ASYMMETRY_L1,
    ):
        row = register_entry(measurement_id)
        mapped = compatibility_entry(measurement_id)
        assert row["interpretation_level"] == 1
        assert row["claim_class"] == CLAIM_CONDITIONAL_HORIZON
        assert row["relation"] == RELATION_WITHHELD
        assert row["physical_path"] is None
        assert row["default_qualification_status"] == QUALIFICATION_NOT_FAR
        assert row["not_far"] is True
        assert mapped["relation"] == RELATION_WITHHELD
        assert mapped["physical_path"] is None
    assert register_entry(MEASUREMENT_ID_RETURN_DISTANCE)["relation"] == RELATION_WITHHELD
    assert register_entry(MEASUREMENT_ID_RETURN_DISTANCE)["not_far"] is True
    superseded = compatibility_entry(MEASUREMENT_ID_PATH_ASYMMETRY_L2)
    assert superseded["relation"] == RELATION_SUPERSEDED
    assert superseded["measurement_id"] is None
    level2 = register_entry(MEASUREMENT_ID_PATH_ASYMMETRY_L2)
    assert level2["relation"] == RELATION_SUPERSEDED
    assert level2["interpretation_level"] == 2
    assert level2["claim_class"] == CLAIM_DISCRETE_ONE_STEP
    assert MEASUREMENT_ID_RECOVERY_TIME_L1 != MEASUREMENT_ID_EXCURSION_RECOVERY_TIME
    assert compatibility_entry(MEASUREMENT_ID_RECOVERY_TIME_L1)["measurement_id"] != (
        MEASUREMENT_ID_EXCURSION_RECOVERY_TIME
    )
    with pytest.raises(KeyError):
        register_entry("induction_recovery_path_asymmetry_level3")


def test_hysteresis_paths_are_logical_not_written() -> None:
    row = compatibility_entry(LOGICAL_POSSIBLE_FUTURES_HYSTERESIS)
    assert row["relation"] == RELATION_LOGICAL
    assert row["physical_path"] is None
    perturbation = compatibility_entry(LOGICAL_PERTURBATION_ROOT)
    assert perturbation["relation"] == RELATION_LOGICAL
    hysteresis = get_family("hysteresis")
    assert hysteresis["implementation"] == "withheld"
    assert hysteresis["namespace"] is None
    assert hysteresis["schema"] is None
    assert "hysteresis" not in WRITABLE_FAMILY_IDS
    assert "recovery" not in WRITABLE_FAMILY_IDS
    assert canonical_group_path("hysteresis") is None
    assert family_forbids("hysteresis", "hysteresis_as_far")
    assert family_forbids("hysteresis", "return_distance_as_far")
    assert family_forbids("hysteresis", "path_alignment_as_level2_model")
    assert family_forbids("hysteresis", "matched_return_as_level2_model")
    assert family_forbids("hysteresis", "recovery_time_as_far_recovery_time")
    assert family_forbids("hysteresis", "descriptive_return_as_intervention")
    for key in FORBIDDEN_HYSTERESIS_CONFIG_KEYS:
        assert family_forbids("hysteresis", key)
    assert "grain" not in hysteresis
    assert "requires" not in hysteresis
    from mndm.inferential_grain import GRAIN_BY_FAMILY_ID

    assert "hysteresis" not in GRAIN_BY_FAMILY_ID


def test_export_does_not_write_hysteresis_or_recovery() -> None:
    state = np.zeros((20, 3), dtype=float)
    time = np.arange(20, dtype=float) * 0.01
    export = build_dynamical_families_export(
        config={"dynamical_families": {"enabled": True}},
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    assert "hysteresis" not in export
    assert "recovery" not in export
    assert MEASUREMENT_ID_RETURN_DISTANCE not in export
    assert MEASUREMENT_ID_PATH_ASYMMETRY_L1 not in export
    assert MEASUREMENT_ID_PATH_ASYMMETRY_L2 not in export


def test_forbidden_hysteresis_yaml_keys_are_refused() -> None:
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
    for key in FORBIDDEN_HYSTERESIS_CONFIG_KEYS:
        config = {"dynamical_families": {"enabled": True, key: {"enabled": True}}}
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=config, **kwargs)
        nested = {
            "dynamical_families": {
                "enabled": True,
                "resilience": {key: {"enabled": True}},
            }
        }
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=nested, **kwargs)


def test_payload_refuses_withheld_hysteresis_family() -> None:
    for key in ("hysteresis", "recovery"):
        with pytest.raises(ValueError, match="withheld"):
            normalize_payload(
                MNPSPayload(
                    time=np.array([0.0, 1.0]),
                    x=np.zeros((2, 3), dtype=np.float32),
                    x_dot=np.zeros((2, 3), dtype=np.float32),
                    dynamical_families={
                        key: {"schema_version": f"mndm.{key}.v1"},
                    },
                )
            )


def test_hdf5_has_no_hysteresis_or_perturbation_root(tmp_path: Path) -> None:
    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3), dtype=np.float32),
        x_dot=np.zeros((2, 3), dtype=np.float32),
        dynamical_families={
            "diffusion": attach_grain(
                {
                    "schema_version": "mndm.diffusion_geometry.v1",
                    "computation_status": "not_testable",
                    "qualification_status": "not_assessed",
                },
                **GRAIN_BY_FAMILY_ID["diffusion"],
            )
        },
    )
    output = write_h5(tmp_path / "hysteresis_levels.h5", "hysteresis_levels", payload)
    with h5py.File(output, "r") as handle:
        assert "perturbation" not in handle
        assert "possible_futures" not in handle
        assert "dynamical_families/hysteresis" not in handle
        assert "dynamical_families/recovery" not in handle


def test_overlays_do_not_enable_hysteresis() -> None:
    common = yaml.safe_load(COMMON_FAMILIES.read_text(encoding="utf-8"))
    root = common["dynamical_families"]
    assert root["enabled"] is True
    for key in FORBIDDEN_HYSTERESIS_CONFIG_KEYS:
        assert key not in root
    icare = yaml.safe_load(ICARE_FAMILIES.read_text(encoding="utf-8"))
    icare_root = icare["dynamical_families"]
    for key in FORBIDDEN_HYSTERESIS_CONFIG_KEYS:
        assert key not in icare_root
    assert icare_root.get("resilience", {}).get("enabled", False) is True

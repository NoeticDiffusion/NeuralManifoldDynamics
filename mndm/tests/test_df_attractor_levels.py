"""SL-LEV-MES-002 attractor/basin ladder under 003 corrections.

Recurrence and region survival are not attractors. No physical family is written.
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
    FORBIDDEN_PERSISTENCE_CONFIG_KEYS,
    LOGICAL_POSSIBLE_FUTURES_PERSISTENCE,
    LOGICAL_POSSIBLE_FUTURES_RECURRENCE,
    LOGICAL_STATE_GEOMETRY_ROOT,
    MEASUREMENT_ID_BASIN_ATTRACTOR,
    MEASUREMENT_ID_ESCAPE_RATE,
    MEASUREMENT_ID_LOCAL_RECURRENCE,
    MEASUREMENT_ID_REGION_DWELL,
    MEASUREMENT_ID_REGION_SURVIVAL,
    MEASUREMENT_ID_SELF_RETENTION,
    MEASUREMENT_ID_STATE_RETURN,
    QUALIFICATION_NOT_ATTRACTOR,
    RELATION_LOGICAL,
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


def test_recurrence_and_survival_are_not_attractors() -> None:
    for measurement_id in (
        MEASUREMENT_ID_STATE_RETURN,
        MEASUREMENT_ID_LOCAL_RECURRENCE,
        MEASUREMENT_ID_REGION_SURVIVAL,
        MEASUREMENT_ID_REGION_DWELL,
        MEASUREMENT_ID_SELF_RETENTION,
        MEASUREMENT_ID_ESCAPE_RATE,
        MEASUREMENT_ID_BASIN_ATTRACTOR,
    ):
        row = register_entry(measurement_id)
        mapped = compatibility_entry(measurement_id)
        assert row["relation"] == RELATION_WITHHELD
        assert row["physical_path"] is None
        assert row["default_qualification_status"] == QUALIFICATION_NOT_ATTRACTOR
        assert row["not_attractor"] is True
        assert mapped["relation"] == RELATION_WITHHELD
        assert mapped["physical_path"] is None
    assert register_entry(MEASUREMENT_ID_ESCAPE_RATE)["not_escape_rate"] is True
    assert register_entry(MEASUREMENT_ID_ESCAPE_RATE)["estimand"] == (
        "minus_log_self_retention_over_dt"
    )
    assert compatibility_entry("state_recurrence_level0")["measurement_id"] == (
        MEASUREMENT_ID_STATE_RETURN
    )
    assert compatibility_entry("residence_persistence_level1")["measurement_id"] == (
        MEASUREMENT_ID_REGION_SURVIVAL
    )
    assert compatibility_entry("transition_metastability_level2")["measurement_id"] == (
        MEASUREMENT_ID_SELF_RETENTION
    )
    with pytest.raises(KeyError):
        register_entry("basin_attractor_geometry_level3")


def test_attractor_paths_are_logical_not_written() -> None:
    for path in (
        LOGICAL_STATE_GEOMETRY_ROOT,
        LOGICAL_POSSIBLE_FUTURES_PERSISTENCE,
        LOGICAL_POSSIBLE_FUTURES_RECURRENCE,
    ):
        row = compatibility_entry(path)
        assert row["relation"] == RELATION_LOGICAL
        assert row["physical_path"] is None
    persistence = get_family("persistence")
    assert persistence["implementation"] == "withheld"
    assert persistence["namespace"] is None
    assert persistence["schema"] is None
    assert "persistence" not in WRITABLE_FAMILY_IDS
    assert "attractor" not in WRITABLE_FAMILY_IDS
    assert canonical_group_path("persistence") is None
    assert family_forbids("persistence", "visual_clustering_as_attractor")
    assert family_forbids("persistence", "state_return_probability_as_attractor")
    assert family_forbids("persistence", "region_survival_as_attractor")
    assert family_forbids("persistence", "transformed_retention_as_escape_rate")
    assert family_forbids("persistence", MEASUREMENT_ID_STATE_RETURN)
    for key in FORBIDDEN_PERSISTENCE_CONFIG_KEYS:
        assert family_forbids("persistence", key)
    assert "grain" not in persistence
    assert "requires" not in persistence
    from mndm.inferential_grain import GRAIN_BY_FAMILY_ID

    assert "persistence" not in GRAIN_BY_FAMILY_ID


def test_export_does_not_write_attractor_or_persistence() -> None:
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
    assert "attractor" not in export
    assert "basin" not in export
    assert "persistence" not in export
    assert "recurrence" not in export
    assert MEASUREMENT_ID_STATE_RETURN not in export
    assert MEASUREMENT_ID_BASIN_ATTRACTOR not in export


def test_forbidden_persistence_yaml_keys_are_refused() -> None:
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
    for key in FORBIDDEN_PERSISTENCE_CONFIG_KEYS:
        config = {"dynamical_families": {"enabled": True, key: {"enabled": True}}}
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=config, **kwargs)


def test_payload_refuses_withheld_persistence_family() -> None:
    for key in ("attractor", "basin", "persistence", "recurrence"):
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


def test_hdf5_has_no_state_geometry_or_attractor_root(tmp_path: Path) -> None:
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
    output = write_h5(tmp_path / "attractor_levels.h5", "attractor_levels", payload)
    with h5py.File(output, "r") as handle:
        assert "state_geometry" not in handle
        assert "possible_futures" not in handle
        assert "dynamical_families/attractor" not in handle
        assert "dynamical_families/persistence" not in handle
        assert "dynamical_families/basin" not in handle


def test_overlays_do_not_enable_attractor_or_persistence() -> None:
    common = yaml.safe_load(COMMON_FAMILIES.read_text(encoding="utf-8"))
    root = common["dynamical_families"]
    assert root["enabled"] is True
    for key in FORBIDDEN_PERSISTENCE_CONFIG_KEYS:
        assert key not in root
    icare = yaml.safe_load(ICARE_FAMILIES.read_text(encoding="utf-8"))
    icare_root = icare["dynamical_families"]
    for key in FORBIDDEN_PERSISTENCE_CONFIG_KEYS:
        assert key not in icare_root
    assert icare_root.get("resilience", {}).get("enabled", False) is True

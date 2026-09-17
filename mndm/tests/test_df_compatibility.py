"""SL-LEV-MES-003 §8: compatibility mapping, not new estimators."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import WRITABLE_FAMILY_IDS
from mndm.dynamical_families.diffusion_geometry import estimate_local_diffusion_geometry
from mndm.dynamical_families.measurement_register import (
    COMPATIBILITY_RELATIONS,
    COMPATIBILITY_ROWS,
    FORBIDDEN_DRIFT_CONFIG_KEYS,
    LOGICAL_LOCAL_DYNAMICS_ROOT,
    MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
    MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
    MEASUREMENT_ID_DIFFUSION_DEFF,
    MEASUREMENT_ID_INCREMENT_COVARIANCE,
    MEASUREMENT_ID_ITO_DIFFUSION,
    MEASUREMENT_ID_ITO_DRIFT,
    MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
    MEASUREMENT_ID_REACH_DEFF_L4,
    MEASUREMENT_ID_REACTIVITY_GAP,
    MEASUREMENT_ID_REALIZED_VELOCITY,
    MEASUREMENT_ID_SMOOTHED_VELOCITY,
    PHYSICAL_DIFFUSION_A_HAT,
    PHYSICAL_MNPS_DOT,
    RELATION_ALIGNMENT,
    RELATION_DERIVED_SCALAR,
    RELATION_DIAGNOSTICS,
    RELATION_DOCUMENTED_IDENTITY,
    RELATION_EXISTING_DATASET,
    RELATION_LOGICAL,
    RELATION_NEW_MEASURE,
    RELATION_SAME_VARIANT,
    RELATION_SERIES_ALIAS,
    RELATION_SUPERSEDED,
    RELATION_SUPPORT_OBJECT,
    RELATION_WITHHELD,
    SUPPORT_OBJECT_LAG1,
    VARIANT_BLOCKED_CROSSFIT,
    compatibility_entry,
    register_entry,
    relation_for,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export


def test_every_compatibility_row_uses_a_closed_relation() -> None:
    frozen = {
        RELATION_DOCUMENTED_IDENTITY,
        RELATION_NEW_MEASURE,
        RELATION_EXISTING_DATASET,
        RELATION_SAME_VARIANT,
        RELATION_DIAGNOSTICS,
        RELATION_SERIES_ALIAS,
        RELATION_DERIVED_SCALAR,
        RELATION_ALIGNMENT,
        RELATION_SUPPORT_OBJECT,
        RELATION_WITHHELD,
        RELATION_SUPERSEDED,
        RELATION_LOGICAL,
    }
    assert COMPATIBILITY_RELATIONS == frozen
    assert len(COMPATIBILITY_RELATIONS) == 12
    for name, row in COMPATIBILITY_ROWS.items():
        assert row["historical_or_export_name"] == name
        assert row["relation"] in frozen
        assert relation_for(name) == row["relation"]
        copied = compatibility_entry(name)
        assert copied["relation"] == row["relation"]
        assert copied is not row
    assert compatibility_entry("b_hat")["relation"] == RELATION_NEW_MEASURE
    assert compatibility_entry(
        f"{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_BLOCKED_CROSSFIT}"
    )["relation"] == RELATION_SAME_VARIANT
    with pytest.raises(KeyError, match="blocked_crossfit"):
        compatibility_entry("blocked_crossfit")


def test_savgol_and_realized_velocity_are_not_aliases() -> None:
    savgol = compatibility_entry("/mnps_3d_dot")
    realized = compatibility_entry(MEASUREMENT_ID_REALIZED_VELOCITY)
    assert savgol["relation"] == RELATION_EXISTING_DATASET
    assert savgol["measurement_id"] == MEASUREMENT_ID_SMOOTHED_VELOCITY
    assert savgol["physical_path"] == PHYSICAL_MNPS_DOT
    assert realized["relation"] == RELATION_NEW_MEASURE
    assert realized["physical_path"] != savgol["physical_path"]
    assert realized["estimand"] != savgol["estimand"]
    register_savgol = register_entry(MEASUREMENT_ID_SMOOTHED_VELOCITY)
    register_realized = register_entry(MEASUREMENT_ID_REALIZED_VELOCITY)
    assert register_savgol["relation"] == RELATION_EXISTING_DATASET
    assert register_realized["relation"] == RELATION_NEW_MEASURE
    assert register_savgol["identity_relation"] == "existing_dataset_not_alias_of_realized_velocity"


def test_a_hat_is_documented_identity_not_a_rename() -> None:
    a_hat = compatibility_entry("a_hat")
    named = compatibility_entry(MEASUREMENT_ID_CONDITIONAL_COVARIANCE)
    assert a_hat["relation"] == named["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert a_hat["physical_path"] == named["physical_path"] == PHYSICAL_DIFFUSION_A_HAT
    assert a_hat["estimand"] == named["estimand"]
    stamped = register_entry(MEASUREMENT_ID_CONDITIONAL_COVARIANCE)
    assert stamped["identity_relation"] == "documented_identity_of_existing_a_hat_not_a_rename"
    assert stamped["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert MEASUREMENT_ID_CONDITIONAL_COVARIANCE not in stamped.get("physical_path", "")


def test_diffusion_tensor_is_same_array_alias_of_a_hat() -> None:
    alias = compatibility_entry("diffusion_tensor")
    assert alias["relation"] == RELATION_SERIES_ALIAS
    assert alias["measurement_id"] == MEASUREMENT_ID_CONDITIONAL_COVARIANCE
    rng = np.random.default_rng(103)
    n, dt, sigma = 400, 0.01, 0.3
    state = np.vstack(
        [np.zeros((1, 2)), np.cumsum(rng.normal(0, sigma * np.sqrt(dt), (n - 1, 2)), axis=0)]
    )
    result = estimate_local_diffusion_geometry(
        state,
        np.arange(n) * dt,
        neighborhood_k=n - 1,
        min_samples=100,
        min_neighborhood_samples=100,
    )
    assert result["computation_status"] == "computed"
    assert np.shares_memory(result["series"]["diffusion_tensor"], result["series"]["a_hat"])
    assert result["summary"]["identity_relation"] == "documented_identity_of_existing_a_hat_not_a_rename"


def test_diary_397_names_are_refused_not_aliases() -> None:
    diary_397 = ("finite_lag", "crossfit", "ito_candidate")
    for name in diary_397:
        row = compatibility_entry(name)
        assert row["relation"] == RELATION_SUPERSEDED
        assert row["physical_path"] is None
        assert row["measurement_id"] is None
    for name in ("ito_drift_level3", "ito_qualified"):
        assert name in FORBIDDEN_DRIFT_CONFIG_KEYS
        row = compatibility_entry(name)
        assert row["relation"] == RELATION_WITHHELD
        assert row["physical_path"] is None
    state = np.zeros((80, 3), dtype=float)
    time = np.arange(80, dtype=float) * 0.01
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    for key in FORBIDDEN_DRIFT_CONFIG_KEYS:
        config = {
            "dynamical_families": {
                "enabled": True,
                "drift": {"enabled": False, key: {"enabled": True}},
            }
        }
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=config, **kwargs)


def test_withheld_and_logical_paths_are_not_writable() -> None:
    for name in (
        MEASUREMENT_ID_ITO_DRIFT,
        MEASUREMENT_ID_ITO_DIFFUSION,
        MEASUREMENT_ID_REACTIVITY_GAP,
        MEASUREMENT_ID_DIFFUSION_DEFF,
        MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
        MEASUREMENT_ID_REACH_DEFF_L4,
    ):
        row = compatibility_entry(name)
        assert row["relation"] == RELATION_WITHHELD
        assert row["physical_path"] is None
        assert register_entry(name)["physical_path"] is None
    assert compatibility_entry("d_diff")["relation"] == RELATION_DERIVED_SCALAR
    assert compatibility_entry("reactivity_gap")["relation"] == RELATION_EXISTING_DATASET
    logical = compatibility_entry(LOGICAL_LOCAL_DYNAMICS_ROOT)
    assert logical["relation"] == RELATION_LOGICAL
    assert logical["physical_path"] is None
    assert "local_dynamics" not in WRITABLE_FAMILY_IDS
    assert SUPPORT_OBJECT_LAG1 in COMPATIBILITY_ROWS
    assert compatibility_entry(SUPPORT_OBJECT_LAG1)["measurement_id"] is None
    increment = compatibility_entry(MEASUREMENT_ID_INCREMENT_COVARIANCE)
    assert increment["relation"] == RELATION_NEW_MEASURE
    assert increment["physical_path"] == (
        "/dynamical_families/diffusion/v1/increment_covariance_level0"
    )


def test_package_does_not_export_diary_397_python_names() -> None:
    import mndm.dynamical_families as families
    import yaml

    assert "estimate_finite_lag_drift" not in families.__all__
    assert "estimate_crossfit_drift" not in families.__all__
    assert not hasattr(families, "estimate_finite_lag_drift")
    assert not hasattr(families, "estimate_crossfit_drift")
    common = Path(__file__).resolve().parents[1] / "config" / "config_ingest_common_dynamical_families.yaml"
    with common.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = cfg["dynamical_families"]
    assert root["enabled"] is True
    assert root["diffusion"]["enabled"] is True
    assert root["drift"]["enabled"] is True
    assert root["one_step"]["enabled"] is True
    assert root["amplification"]["enabled"] is True
    assert root["history"]["enabled"] is True
    assert root["turning"]["enabled"] is True
    assert "local_dynamics" not in root
    assert "possible_futures" not in root
    assert "perturbation" not in root
    assert "spread" not in root

"""SL-LEV-MES-002 destination ladder under 003 corrections.

Existing q_A_to_B is a documented identity, not a generator committor.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from mndm.dynamical_families import family_forbids
from mndm.dynamical_families.committor import (
    estimate_committor,
    estimate_committor_local_law_dense_grid_o2b,
)
from mndm.dynamical_families.measurement_register import (
    FORBIDDEN_DESTINATION_CONFIG_KEYS,
    LOGICAL_POSSIBLE_FUTURES_DESTINATION,
    LOGICAL_POSSIBLE_FUTURES_ROOT,
    MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0,
    MEASUREMENT_ID_DESTINATION_HIT_L1,
    MEASUREMENT_ID_DESTINATION_RESOLVED,
    MEASUREMENT_ID_DESTINATION_UNRESOLVED,
    MEASUREMENT_ID_GENERATOR_COMMITTOR,
    MEASUREMENT_ID_O2B_QUADRATURE,
    MEASUREMENT_ID_TRANSITION_HIT_L2,
    MEASUREMENT_ID_TRANSITION_HIT_L4,
    QUALIFICATION_NOT_GENERATOR_COMMITTOR,
    RELATION_DOCUMENTED_IDENTITY,
    RELATION_LOGICAL,
    RELATION_SERIES_ALIAS,
    RELATION_SUPERSEDED,
    RELATION_WITHHELD,
    RELATION_EXISTING_DATASET,
    RELATION_DIAGNOSTICS,
    compatibility_entry,
    register_entry,
)
from mndm.pipeline.dynamical_families_export import build_dynamical_families_export
from mndm.pipeline.summary import _build_dynamical_families_export_for_layers

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


def _labeled_segments(*, n_segments: int = 6, per_segment: int = 40):
    labeled = n_segments * per_segment
    extra = per_segment
    state = np.concatenate(
        [
            np.tile(np.linspace(-1.0, 1.0, per_segment), n_segments),
            np.linspace(-0.2, 0.2, extra),
        ]
    )[:, None]
    labels = np.full(labeled + extra, -1, dtype=np.int8)
    labels[:labeled:per_segment] = 0
    labels[per_segment - 1 : labeled : per_segment] = 1
    segment_id = np.concatenate(
        [
            np.repeat(np.arange(n_segments), per_segment),
            np.full(extra, n_segments, dtype=np.int32),
        ]
    )
    time = np.arange(state.shape[0], dtype=float) * 0.01
    return state, time, labels, segment_id


def _o2b_ensembles(*, support_per_grid: int = 20):
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


def test_first_hit_is_resolved_level1_not_generator_committor() -> None:
    state, time, labels, segment_id = _labeled_segments()
    result = estimate_committor(
        state,
        time,
        labels,
        set_A=[0],
        set_B=[1],
        segment_id=segment_id,
        neighborhood_k=30,
        min_support=20,
        min_transition_segments=2,
        min_valid_fraction=0.01,
    )
    assert result["computation_status"] == "computed"
    assert result["measurement_id"] == MEASUREMENT_ID_DESTINATION_RESOLVED
    assert result["interpretation_level"] == 1
    assert result["qualification_status"] == QUALIFICATION_NOT_GENERATOR_COMMITTOR
    assert result["summary"]["not_generator_committor"] is True
    assert result["summary"]["estimand"] == "local_mean_of_resolved_first_hit_outcomes"
    assert "q_A_to_B" in result["series"]
    assert MEASUREMENT_ID_GENERATOR_COMMITTOR not in result
    assert result["measurement_id"] != MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0
    outcomes = np.asarray(result["series"]["resolved_first_hit_outcome"], dtype=float)
    unresolved = ~np.isfinite(outcomes)
    assert int(np.sum(unresolved)) > 0
    observed = np.isfinite(outcomes)
    local_mean = float(np.mean(outcomes[observed]))
    valid_q = np.asarray(result["series"]["q_A_to_B"], dtype=float)
    interior = np.isfinite(valid_q) & (labels == -1)
    if int(np.sum(interior)) > 0:
        assert abs(float(np.nanmean(valid_q[interior])) - local_mean) < 0.6


def test_o2b_q_is_restricted_quadrature_not_generator_committor() -> None:
    state, time, reaction, regime, segment_id = _o2b_ensembles()
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
        min_samples=100,
        min_support_per_grid=20,
    )
    assert result["computation_status"] == "computed"
    assert result["measurement_id"] == MEASUREMENT_ID_O2B_QUADRATURE
    assert result["interpretation_level"] is None
    assert result["summary"]["interpretation_level_token"] == "not_numbered"
    assert result["qualification_status"] == QUALIFICATION_NOT_GENERATOR_COMMITTOR
    assert result["summary"]["not_a_3d_mnps_committor"] is True
    assert result["summary"]["estimand"] == (
        "one_d_constant_diffusion_quadrature_on_explicit_reaction_coordinate"
    )
    assert np.shares_memory(result["series"]["q_hat"], result["series"]["q_A_to_B"])
    assert MEASUREMENT_ID_GENERATOR_COMMITTOR not in result
    assert "generator_committor_level3" not in result["series"]


def test_destination_compatibility_and_withheld_names() -> None:
    first_hit = compatibility_entry("local_first_hit_outcome_average/q_A_to_B")
    o2b = compatibility_entry("local_law_dense_grid_o2b/q_A_to_B")
    assert first_hit["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert first_hit["measurement_id"] == MEASUREMENT_ID_DESTINATION_RESOLVED
    assert o2b["relation"] == RELATION_DOCUMENTED_IDENTITY
    assert o2b["measurement_id"] == MEASUREMENT_ID_O2B_QUADRATURE
    assert first_hit["measurement_id"] != o2b["measurement_id"]
    assert compatibility_entry("q_hat")["relation"] == RELATION_SERIES_ALIAS
    for name in (
        MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0,
        MEASUREMENT_ID_DESTINATION_HIT_L1,
        MEASUREMENT_ID_DESTINATION_UNRESOLVED,
        MEASUREMENT_ID_GENERATOR_COMMITTOR,
        MEASUREMENT_ID_TRANSITION_HIT_L4,
    ):
        row = compatibility_entry(name)
        assert row["relation"] == RELATION_WITHHELD
        assert row["physical_path"] is None
        assert register_entry(name)["physical_path"] is None
    superseded = compatibility_entry(MEASUREMENT_ID_TRANSITION_HIT_L2)
    assert superseded["relation"] == RELATION_SUPERSEDED
    assert superseded["physical_path"] is None
    assert compatibility_entry(LOGICAL_POSSIBLE_FUTURES_ROOT)["relation"] == RELATION_LOGICAL
    assert compatibility_entry(LOGICAL_POSSIBLE_FUTURES_DESTINATION)["relation"] == RELATION_LOGICAL
    assert family_forbids("destination", "computed_q_as_generator_committor")
    assert family_forbids("destination", "state_conditioned_first_hit_as_level0")
    assert family_forbids("destination", MEASUREMENT_ID_DESTINATION_HIT_L1)
    assert family_forbids("destination", MEASUREMENT_ID_DESTINATION_UNRESOLVED)
    assert family_forbids("destination", MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0)
    assert family_forbids("destination", MEASUREMENT_ID_TRANSITION_HIT_L4)
    assert family_forbids("destination", MEASUREMENT_ID_GENERATOR_COMMITTOR)
    assert family_forbids("destination", MEASUREMENT_ID_TRANSITION_HIT_L2)
    assert family_forbids("destination", "committor")
    from mndm.dynamical_families.registry import get_family

    destination_forbids = set(get_family("destination")["forbids"])
    for key in FORBIDDEN_DESTINATION_CONFIG_KEYS:
        assert key in destination_forbids
    outcome = compatibility_entry("resolved_first_hit_outcome")
    assert outcome["relation"] == RELATION_EXISTING_DATASET
    assert outcome["measurement_id"] is None
    assert outcome["measurement_id"] != MEASUREMENT_ID_DESTINATION_UNRESOLVED
    count = compatibility_entry("n_resolved_first_hit_outcomes")
    assert count["relation"] == RELATION_DIAGNOSTICS
    assert count["measurement_id"] is None


def test_denied_destination_qualification_keeps_o2b_identity_not_zeros() -> None:
    state = np.zeros((80, 3), dtype=float)
    time = np.arange(80, dtype=float) * 0.01
    export = build_dynamical_families_export(
        config={
            "dynamical_families": {
                "enabled": True,
                "destination": {
                    "enabled": True,
                    "regime_source": "explicit_first_hit_labels",
                    "label_key": "stage",
                    "reaction_coordinate": {
                        "source": "explicit_column",
                        "key": "q_coordinate",
                        "boundaries": [-0.8, 0.8],
                    },
                    "estimator": "local_law_dense_grid_o2b",
                    "translation_qualification": {"qualified": False},
                    "set_A": [0],
                    "set_B": [1],
                    "diffusion_coefficient": 0.25,
                },
            }
        },
        state=state,
        time=time,
        stage=np.zeros(time.size, dtype=np.int32),
        segment_id=np.zeros(time.size, dtype=np.int32),
        reaction_coordinate=state[:, 0],
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    destination = export["destination"]
    assert destination["computation_status"] == "not_testable"
    assert destination["measurement_id"] == MEASUREMENT_ID_O2B_QUADRATURE
    assert destination["qualification_status"] == QUALIFICATION_NOT_GENERATOR_COMMITTOR
    assert destination["series"] == {}
    assert MEASUREMENT_ID_GENERATOR_COMMITTOR not in destination


def test_forbidden_destination_yaml_keys_are_refused() -> None:
    state = np.zeros((40, 3), dtype=float)
    time = np.arange(40, dtype=float) * 0.01
    kwargs = dict(
        state=state,
        time=time,
        stage=None,
        segment_id=np.zeros(time.size, dtype=np.int32),
        coordinate_layer="coords_3d_subject_anchored",
        coordinate_names=["m", "d", "e"],
    )
    for key in FORBIDDEN_DESTINATION_CONFIG_KEYS:
        config = {
            "dynamical_families": {
                "enabled": True,
                "destination": {"enabled": False, key: {"enabled": True}},
            }
        }
        with pytest.raises(ValueError, match=key):
            build_dynamical_families_export(config=config, **kwargs)


def test_destination_overlays_are_not_flipped_by_this_gate() -> None:
    common = yaml.safe_load(COMMON_FAMILIES.read_text(encoding="utf-8"))
    assert common["dynamical_families"]["destination"]["enabled"] is True
    icare = yaml.safe_load(ICARE_FAMILIES.read_text(encoding="utf-8"))
    families = icare["dynamical_families"]
    destination = families.get("destination") or {}
    assert destination.get("enabled", False) is True
    assert "generator_committor_level3" not in destination
    assert destination.get("estimator") == "local_law_dense_grid_o2b"


def test_missing_coordinate_layer_stamps_o2b_identity() -> None:
    export = _build_dynamical_families_export_for_layers(
        config={
            "dynamical_families": {
                "enabled": True,
                "coordinate_layer": "subject_anchored",
                "destination": {"enabled": True},
            }
        },
        x_subject_anchored=None,
        x_cohort_anchored=None,
        time=np.arange(8, dtype=float),
        stage=None,
        segment_id=None,
    )
    destination = export["destination"]
    assert destination["computation_status"] == "not_testable"
    assert destination["failure_reason"] == "requested_coordinate_layer_not_available"
    assert destination["measurement_id"] == MEASUREMENT_ID_O2B_QUADRATURE
    assert destination["qualification_status"] == QUALIFICATION_NOT_GENERATOR_COMMITTOR
    assert destination["series"] == {}

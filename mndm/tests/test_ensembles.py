"""Tests for ensemble helper utilities."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_sanitize_group_name_basic():
    from core.ensembles import sanitize_group_name

    assert sanitize_group_name("Frontal") == "frontal"
    assert sanitize_group_name("Parietal/Occipital") == "parietal_occipital"
    assert sanitize_group_name("  Mixed  Name  ") == "mixed_name"


def test_realize_ensemble_groups_basic():
    from core.ensembles import EnsembleGroupDef, realize_ensemble_groups

    cfg = {
        "min_channels": 2,
        "groups": {
            "frontal": ["Fp1", "Fp2", "F3"],
            "central": ["C3", "Cz", "C4"],
        },
    }
    available = ["Fp1", "Fp2", "F3", "C3", "Cz", "C4"]

    groups = realize_ensemble_groups(cfg, "dsTEST", available)

    # We should get two realised groups with non-empty indices
    assert isinstance(groups, list)
    assert all(isinstance(g, EnsembleGroupDef) for g in groups)
    assert len(groups) == 2

    names = {g.safe_name for g in groups}
    assert "frontal" in names
    assert "central" in names

    for g in groups:
        assert len(g.indices) >= 2
        assert len(g.channels) == len(g.indices)
        assert all(0 <= idx < len(available) for idx in g.indices)


def test_compute_ensemble_summary_for_subject_mean_var(monkeypatch):
    """compute_ensemble_summary_for_subject should aggregate mean/var across groups."""
    import numpy as np
    import pandas as pd

    import mndm.pipeline.robustness_helpers as robustness_helpers_mod

    coords_9d_names = ["s1", "s2"]
    subcoords_spec = {"s1": {"feat1": 1.0}, "s2": {"feat2": 1.0}}

    # Dummy features frame (contents are irrelevant because we stub projection)
    sub_frame = pd.DataFrame({"feat1": np.zeros(10), "feat2": np.zeros(10)})

    # Stub ensemble group resolution to always return two groups
    def _fake_resolve_config_groups(cfg, dataset_id):
        return {"g1": ["Cz"], "g2": ["Pz"]}

    # Patch in the module where the names are looked up
    monkeypatch.setattr(robustness_helpers_mod.ensembles, "resolve_config_groups", _fake_resolve_config_groups)

    # Stub project_features_v2 to return deterministic coordinates per group call
    call_counter = {"i": 0}

    def _fake_project_features_v2(df, spec, normalize=None):
        call_counter["i"] += 1
        if call_counter["i"] == 1:
            coords = np.tile(np.array([1.0, 2.0], dtype=np.float32), (10, 1))
        else:
            coords = np.tile(np.array([3.0, 4.0], dtype=np.float32), (10, 1))
        return coords, list(coords_9d_names)

    monkeypatch.setattr(robustness_helpers_mod.projection, "project_features_v2", _fake_project_features_v2)

    config = {
        "robustness": {
            "ensembles": {
                "enabled": True,
                "groups": {"g1": ["Cz"], "g2": ["Pz"]},
            }
        }
    }

    summary = robustness_helpers_mod.compute_ensemble_summary_for_subject(
        config=config,
        dataset_id="dsTEST",
        sub_frame=sub_frame,
        coords_9d_names=list(coords_9d_names),
        subcoords_spec=subcoords_spec,
        normalize_mode=None,
    )

    assert summary is not None
    mean_map = summary["mean"]
    var_map = summary["var"]

    # For s1: group medians = [1, 3] → mean = 2.0, var = 1.0
    # For s2: group medians = [2, 4] → mean = 3.0, var = 1.0
    assert np.isclose(mean_map["s1"], 2.0)
    assert np.isclose(mean_map["s2"], 3.0)
    assert np.isclose(var_map["s1"], 1.0)
    assert np.isclose(var_map["s2"], 1.0)




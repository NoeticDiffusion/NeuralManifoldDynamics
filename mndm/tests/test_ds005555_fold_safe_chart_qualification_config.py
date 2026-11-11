from pathlib import Path

import pytest

from core.config_loader import load_config
from mndm.pipeline.fold_safe import FoldSplitError, load_fold_split


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "sources"
    / "openneuro"
    / "config_ingest_ds005555_continuous_c123_fold_safe_chart_qualification.yaml"
)
FEATURES_CONFIG = CONFIG.parent / "config_ingest_ds005555_continuous_c123_prediction_safe_full_cohort_features_20260929.yaml"


def test_fold_chart_overlay_preserves_frozen_chart_definitions_and_fails_closed():
    cfg = load_config(CONFIG)
    features_cfg = load_config(FEATURES_CONFIG)

    assert cfg["summarize_forbidden"] is False
    assert cfg["preprocess"]["temporal_support_mode"] == "causal_finite"
    assert cfg["features"]["ppg"]["estimator"] == "window_local_present_only_v1"
    assert cfg["mnps_projection"]["v1_mapping"] == features_cfg["mnps_projection"]["v1_mapping"]
    assert cfg["mnps_9d"] == features_cfg["mnps_9d"] | {"jacobian": {"enabled": False}}
    assert cfg["mnps_9d"]["definition_version"] == "2.0"
    assert cfg["mnps"]["jacobian"]["enabled"] is False
    assert cfg["mnps"]["stage_codebook"] == {}
    assert cfg["summarize"]["include_bids_event_annotations"] is False
    assert cfg["mnps_projection"]["normalize"] == "robust_z"
    assert cfg["mnps_projection"]["export_contracts"] == {
        "subject_anchored": False,
        "cohort_anchored": True,
    }

    fold_cfg = cfg["mnps_projection"]["fold_safe"]
    assert fold_cfg["enabled"] is True
    assert fold_cfg["manifest"] == ""
    assert fold_cfg["role"] == ""
    assert fold_cfg["signal_support_ledger"] == ""
    with pytest.raises(FoldSplitError, match="Could not read fold split manifest"):
        load_fold_split(fold_cfg["manifest"])

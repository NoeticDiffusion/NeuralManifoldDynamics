from pathlib import Path

from core.config_loader import load_config


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "sources"
    / "openneuro"
    / "config_ingest_ds005555_continuous_c123_prediction_safe_full_cohort_features_20260929.yaml"
)
PILOT_CONFIG = CONFIG.parent / "config_ingest_ds005555_continuous_c123_prediction_safe_pilot_20260929.yaml"
BASE_CONFIG = CONFIG.parent / "config_ingest_ds005555_continuous_c123_cautious_gap_safe_final.yaml"


def test_full_cohort_surface_config_is_isolated_and_measurement_only():
    cfg = load_config(CONFIG)
    pilot = load_config(PILOT_CONFIG)
    base = load_config(BASE_CONFIG)

    assert cfg["datasets"] == ["ds005555"]
    assert cfg["paths"]["processed_dir"].endswith(
        "ds005555_continuous_c123_prediction_safe_full_cohort_features_20260929"
    )
    assert cfg["paths"]["processed_dir"] not in {
        pilot["paths"]["processed_dir"],
        base["paths"]["processed_dir"],
    }
    epoching = cfg["epoching"]["datasets"]["ds005555"]
    assert epoching["length_s"] == 30.0
    assert epoching["step_s"] == 30.0
    assert epoching["sampling"]["enabled"] is False

    assert cfg["preprocess"]["temporal_support_mode"] == "causal_finite"
    assert cfg["preprocess"]["resample"]["method"] == "identity_sample_grid"
    assert cfg["preprocess"]["notch_hz"] is None
    assert cfg["preprocess"]["eeg_bandpass_phase"] == "minimum"
    assert cfg["features"]["ppg"]["estimator"] == "window_local_present_only_v1"
    assert cfg["summarize_forbidden"] is True

    # This is a feature-only overlay, not a redefinition of either chart.
    assert cfg["mnps_projection"] == base["mnps_projection"]
    assert cfg["mnps_9d"] == base["mnps_9d"]

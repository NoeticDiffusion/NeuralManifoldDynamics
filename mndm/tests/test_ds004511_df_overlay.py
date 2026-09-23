"""ds004511 dynamical-families overlay merge (coverage, not TQ)."""

from __future__ import annotations

from pathlib import Path

from core.config_loader import load_config


OPENNEURO = (
    Path(__file__).resolve().parents[1] / "config" / "sources" / "openneuro"
)
CONFIG = Path(__file__).resolve().parents[1] / "config"
COMMON_PROFILES = (
    CONFIG / "config_ingest_common_eeg.yaml",
    CONFIG / "config_ingest_common_fmri.yaml",
    CONFIG / "config_ingest_common_ephys.yaml",
)


def test_common_profiles_do_not_import_dynamical_families_overlay() -> None:
    for path in COMMON_PROFILES:
        text = path.read_text(encoding="utf-8")
        assert "config_ingest_common_dynamical_families.yaml" not in text
        assert "dynamical_families:" not in text


def test_ds004511_df_overlay_merge() -> None:
    cfg = load_config(OPENNEURO / "config_ingest_ds004511_dynamical_families.yaml")
    assert cfg["datasets"] == ["ds004511"]
    families = cfg["dynamical_families"]
    assert families["enabled"] is True
    assert families["coordinate_layer"] == "subject_anchored"
    assert families["diffusion"]["enabled"] is True
    assert families["diffusion"]["drift"]["enabled"] is False
    assert families["diffusion"]["drift"]["source"] == "not_supplied"
    assert families["diffusion"]["translation_qualification"]["qualified"] is False
    assert families["drift"]["enabled"] is True
    one_step = families["one_step"]
    assert one_step["enabled"] is True
    assert one_step["declared_lags"] == [1, 2]
    assert one_step["one_step_rel_mse_threshold"] == 0.9
    for name in ("amplification", "history", "turning"):
        assert families[name]["enabled"] is True
    assert families["destination"]["enabled"] is True
    assert families["destination"]["translation_qualification"]["qualified"] is False
    assert families["resilience"]["enabled"] is True
    assert families["resilience"]["translation_qualification"]["qualified"] is False
    assert "ito_drift_level3" not in one_step
    assert "ito_qualified" not in one_step
    assert "finite_time_peak_gain_level4" not in one_step
    assert "generator_committor_level3" not in families["destination"]
    assert "far_threshold_p50_level4" not in families["resilience"]
    local = cfg["local_dynamics"]
    assert local["finite_time_response"]["enabled"] is True
    assert local["transition_residuals"]["enabled"] is True
    assert local["stochastic_reachability"]["enabled"] is True
    assert cfg["regional_mnps"]["enabled"] is False
    assert cfg["robustness"]["ensembles"]["enabled"] is False
    assert cfg["paths"]["processed_dir"] == "K:/processed/ds004511_dynamical_families"
    assert cfg["paths"]["received_dir"] == "K:/ExternalReceivedDatasets/openneuro"
    assert (
        cfg["paths"]["dataset_received_dirs"]["ds004511"]
        == "K:/ExternalReceivedDatasets/openneuro/ds004511"
    )
    assert cfg["preprocess"]["datasets"]["ds004511"]["physio_tsv_inject"]["enabled"] is True

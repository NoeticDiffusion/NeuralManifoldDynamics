"""ds006036 / ds004504 dynamical-families overlay merge (coverage, not TQ)."""

from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

from core.config_loader import load_config
from mndm.pipeline.event_locked_runner import _filter_stage_block_intervals


OPENNEURO = (
    Path(__file__).resolve().parents[1] / "config" / "sources" / "openneuro"
)
CONFIG = Path(__file__).resolve().parents[1] / "config"
COMMON_PROFILES = (
    CONFIG / "config_ingest_common_eeg.yaml",
    CONFIG / "config_ingest_common_fmri.yaml",
    CONFIG / "config_ingest_common_ephys.yaml",
)


def _assert_df_overlay(cfg: dict, dataset_id: str) -> None:
    assert cfg["datasets"] == [dataset_id]
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
    assert one_step.get("one_step_rel_mse_threshold", 0.9) == 0.9
    for name in ("amplification", "history", "turning"):
        assert families[name]["enabled"] is True
    assert families["destination"]["translation_qualification"]["qualified"] is False
    assert families["resilience"]["translation_qualification"]["qualified"] is False
    local = cfg["local_dynamics"]
    assert local["finite_time_response"]["enabled"] is True
    assert local["transition_residuals"]["enabled"] is True
    assert local["stochastic_reachability"]["enabled"] is True
    assert cfg["regional_mnps"]["enabled"] is False
    assert cfg["robustness"]["ensembles"]["enabled"] is False


def test_common_profiles_do_not_import_dynamical_families_overlay() -> None:
    for path in COMMON_PROFILES:
        text = path.read_text(encoding="utf-8")
        assert "config_ingest_common_dynamical_families.yaml" not in text
        assert "dynamical_families:" not in text


def test_ds006036_df_overlay_merge() -> None:
    cfg = load_config(OPENNEURO / "config_ingest_ds006036_dynamical_families.yaml")
    _assert_df_overlay(cfg, "ds006036")
    assert cfg["paths"]["processed_dir"].endswith("ds006036_dynamical_families")
    blocking = cfg["epoching"]["datasets"]["ds006036"]["sampling"]["stage_blocking"]
    assert blocking["enabled"] is False
    assert cfg["event_locked"]["datasets"]["ds006036"]["enabled"] is False
    assert cfg["block_native"]["datasets"]["ds006036"]["enabled"] is False
    assert cfg["mnps_projection"]["export_contracts"]["cohort_anchored"] is False
    assert cfg["mnps_projection"]["anchor_auto_fit"]["enabled"] is False


def test_ds006036_frequency_block_overlay_merge() -> None:
    cfg = load_config(OPENNEURO / "config_ingest_ds006036_df_frequency_blocks.yaml")
    _assert_df_overlay(cfg, "ds006036")
    assert cfg["paths"]["processed_dir"].endswith("ds006036_df_frequency_blocks")
    assert cfg["mnps"]["window_sec"] == 8.0
    assert cfg["mnps"]["overlap"] == 0.5
    blocking = cfg["epoching"]["datasets"]["ds006036"]["sampling"]["stage_blocking"]
    assert blocking["enabled"] is True
    assert blocking["window_membership"]["mode"] == "overlap_frac_ge"
    assert float(blocking["window_membership"]["min_overlap_fraction"]) == 0.75
    event_locked = cfg["event_locked"]["datasets"]["ds006036"]
    assert event_locked["enabled"] is True
    assert event_locked["profile"] == "ds006036_frequency_block_end_sec_v1"
    assert event_locked["event_source"]["kind"] == "derived_stage_block_end"
    assert event_locked["event_source"]["block_parameters"] == [5, 10, 15, 20]
    assert event_locked["bins"] == {
        "in_block_tail_sec": [-8.0, 0.0],
        "post_block_early_sec": [0.0, 8.0],
        "post_block_late_sec": [8.0, 16.0],
    }
    codebook = cfg["mnps"]["stage_codebook"]
    assert int(codebook["PHOTO 5Hz"]) == 50
    assert int(codebook["PHOTO 10Hz"]) == 51
    assert int(codebook["PHOTO 15Hz"]) == 52
    assert int(codebook["PHOTO 20Hz"]) == 53
    block_native = cfg["block_native"]["datasets"]["ds006036"]
    assert block_native["enabled"] is True
    assert block_native["source"]["kind"] == "stage_blocking"
    profile = block_native["window_profile"]
    assert float(profile["window_length_sec"]) == 4.0
    assert float(profile["step_sec"]) == 2.0
    assert float(profile["min_block_sec"]) == 4.0
    assert int(profile["min_windows_per_block"]) == 2
    assert cfg["mnps_projection"]["anchor_auto_fit"]["enabled"] is False
    assert cfg["dynamical_families"]["diffusion"]["min_samples"] == 30


def test_ds006036_frequency_block_event_filter_keeps_target_hz() -> None:
    intervals = [
        SimpleNamespace(stage_code=50, block_parameter=5.0),
        SimpleNamespace(stage_code=57, block_parameter=3.0),
        SimpleNamespace(stage_code=51, block_parameter=10.0),
        SimpleNamespace(stage_code=58, block_parameter=7.0),
        SimpleNamespace(stage_code=52, block_parameter=15.0),
        SimpleNamespace(stage_code=53, block_parameter=20.0),
        SimpleNamespace(stage_code=55, block_parameter=25.0),
        SimpleNamespace(stage_code=52, block_parameter=float("nan")),
    ]
    kept = _filter_stage_block_intervals(intervals, block_parameters=[5, 10, 15, 20])
    assert [float(iv.block_parameter) for iv in kept] == [5.0, 10.0, 15.0, 20.0]


def test_ds004504_df_overlay_merge() -> None:
    cfg = load_config(OPENNEURO / "config_ingest_ds004504_dynamical_families.yaml")
    _assert_df_overlay(cfg, "ds004504")
    assert cfg["paths"]["processed_dir"].endswith("ds004504_dynamical_families")
    assert (
        cfg["paths"]["dataset_received_dirs"]["ds004504"]
        == "K:/ExternalReceivedDatasets/openneuro/ds004504"
    )
    assert "derivatives" not in cfg["paths"]["dataset_received_dirs"]["ds004504"]

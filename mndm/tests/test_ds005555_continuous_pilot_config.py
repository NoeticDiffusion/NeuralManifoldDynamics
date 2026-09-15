from pathlib import Path

from core.config_loader import load_config
from mndm.features.epoch_selection import resolve_stage_stratified_epoch_set


CONFIG = Path(__file__).resolve().parents[1] / "config" / "sources" / "openneuro" / "config_ingest_ds005555_dynamical_families_continuous_pilot.yaml"


def test_continuous_pilot_resolves_imports_and_isolated_output():
    cfg = load_config(CONFIG)
    parent = load_config(CONFIG.parent / "config_ingest_ds005555_dynamical_families.yaml")
    assert cfg["datasets"] == ["ds005555"]
    assert cfg["paths"]["processed_dir"].endswith("ds005555_dynamical_families_continuous_pilot")
    assert cfg["paths"]["processed_dir"] != parent["paths"]["processed_dir"]
    assert cfg["feature_storage"]["read_prefer"] == "parquet"
    assert cfg["feature_storage"]["write_both"] is True
    assert cfg["preprocess"]["crop"] == parent["preprocess"]["crop"]
    sampling = cfg["epoching"]["datasets"]["ds005555"]["sampling"]
    assert sampling["enabled"] is False
    assert cfg["epoching"]["datasets"]["ds005555"]["length_s"] == 30.0
    assert cfg["epoching"]["datasets"]["ds005555"]["step_s"] == 30.0


def test_continuous_pilot_preserves_family_and_coordinate_gates():
    cfg = load_config(CONFIG)
    parent = load_config(CONFIG.parent / "config_ingest_ds005555_dynamical_families.yaml")
    assert cfg["dynamical_families"] == parent["dynamical_families"]
    assert cfg["local_dynamics"] == parent["local_dynamics"]
    families = cfg["dynamical_families"]
    assert families["enabled"] is True
    assert families["coordinate_layer"] == "subject_anchored"
    assert families["diffusion"]["translation_qualification"]["qualified"] is False
    assert families["diffusion"]["drift"]["enabled"] is False
    assert families["destination"]["translation_qualification"]["qualified"] is False
    assert families["resilience"]["translation_qualification"]["qualified"] is False


def test_continuous_pilot_effectively_bypasses_stage_sampler():
    cfg = load_config(CONFIG)
    selected = resolve_stage_stratified_epoch_set(
        config=cfg, dataset_id="ds005555", raw_file_path=None, sfreq=100.0,
        n_samples=6000, step_s=30.0, epoch_length_samples=3000,
        epoch_step_samples=3000,
    )
    assert selected is None

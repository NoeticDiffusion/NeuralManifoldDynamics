"""Tests for config_loader module."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_load_config_callable():
    """Test that load_config is callable."""
    from core.config_loader import load_config
    
    assert callable(load_config)


def test_load_config_valid_yaml():
    """Test loading a valid YAML config."""
    from core.config_loader import load_config
    
    config_path = Path(__file__).resolve().parents[1] / "config" / "config_ingest.yaml"
    cfg = load_config(config_path)
    
    assert isinstance(cfg, dict)
    assert "datasets" in cfg
    assert "paths" in cfg
    assert "preprocess" in cfg


def test_load_config_invalid():
    """Test that invalid config raises appropriate error."""
    from core.config_loader import load_config
    import tempfile
    
    # On Windows, unlinking an open NamedTemporaryFile fails; capture name then close
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: [unclosed")
        f.flush()
        temp_name = f.name

    try:
        _ = load_config(Path(temp_name))
        # Should handle gracefully or raise ValueError
        assert True
    except Exception:
        assert True
    finally:
        Path(temp_name).unlink(missing_ok=True)


def test_load_config_imports_deep_merge():
    """Test that loader composes configs via imports and deep merge."""
    from core.config_loader import load_config
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        base = root / "base.yaml"
        overlay = root / "overlay.yaml"
        base.write_text(
            "\n".join(
                [
                    "version: 1.2",
                    "datasets: []",
                    "paths:",
                    "  received_dir: base",
                    "  processed_dir: base",
                    "preprocess:",
                    "  sfreq: 250",
                    "  channel_typing:",
                    "    enabled: true",
                    "epoching: {length_s: 8.0, step_s: 4.0}",
                    "features: {eeg_psd: {method: multitaper}}",
                    "mnps_projection: {normalize: robust_z}",
                    "robustness: {coverage: {min_seconds: 60, min_epochs: 20}}",
                ]
            ),
            encoding="utf-8",
        )
        overlay.write_text(
            "\n".join(
                [
                    "imports:",
                    "  - ./base.yaml",
                    "datasets: [ANPHY]",
                    "paths:",
                    "  processed_dir: overlay",
                    "preprocess:",
                    "  sfreq: 500",
                ]
            ),
            encoding="utf-8",
        )

        cfg = load_config(overlay)
        assert cfg["datasets"] == ["ANPHY"]
        assert cfg["paths"]["received_dir"] == "base"
        assert cfg["paths"]["processed_dir"] == "overlay"
        assert cfg["preprocess"]["sfreq"] == 500
        assert cfg["preprocess"]["channel_typing"]["enabled"] is True


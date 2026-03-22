"""Tests for pipeline context resolution."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.context import ResolvedConfig, SummarizeContext


def test_summarize_context_from_config_resolves_paths_and_overrides(tmp_path):
    """Test summarize context from config resolves paths and overrides."""
    config = {
        "paths": {
            "received_dir": str(tmp_path / "received_data"),
            "processed_dir": str(tmp_path / "processed_data"),
        },
        "epoching": {"length_s": 12.0},
        "mnps": {
            "fs_out": 8.0,
            "overlap": 0.25,
            "derivative": {"method": "sav_gol", "window": 5, "polyorder": 2},
            "knn": {"k": 12, "metric": "cityblock"},
            "ridge": {"alpha": 3.0, "distance_weighted": False},
            "stage_codebook": {"wake": 0},
            "embodied": {"enabled": False},
            "whiten": False,
        },
        "mnps_projection": {
            "normalize": "z",
            "weights": {
                "m": {"a": 0.5},
                "d": {"b": 1.0},
                "e": {"c": 2.0},
            },
        },
        "ndt_ingest": {
            "ingest_version": "1.2.3",
            "mnps_mapping": {
                "version": "2025-01",
                "normalize": "robust",
                "m_axis": {"weights": {"x": 1.0}},
            },
        },
        "robustness": {
            "coverage": {"min_seconds": 12.5, "min_epochs": 3},
        },
    }

    override_dir = tmp_path / "processed_override"
    overrides = {"knn_k": 7, "super_window": 9}
    ctx = SummarizeContext.from_config(config, override_dir, mnps_overrides=overrides)

    assert ctx.received_dir == Path(config["paths"]["received_dir"])
    assert ctx.processed_dir == override_dir
    assert ctx.normalize_override == "robust_z"
    assert ctx.weights["m"]["x"] == 1.0  # override from mapping spec
    assert ctx.mnps_cfg["knn_k"] == 7  # override applied
    assert ctx.mnps_cfg["super_window"] == 9
    assert ctx.mnps_cfg["whiten"] is False
    assert ctx.derivative_cfg["window"] == 5
    assert ctx.ingest_meta["ingest_version"] == "1.2.3"
    assert ctx.ingest_meta["mapping_source"] == "ndt_ingest.mnps_mapping"
    assert ctx.ingest_meta["normalize_mode_effective"] == "robust_z"
    assert ctx.coverage.min_seconds == 12.5
    assert ctx.coverage.min_epochs == 3


def test_resolved_config_from_mapping(tmp_path):
    """Test resolved config from mapping."""
    config = {
        "paths": {},
        "robustness": {"coverage": {"min_seconds": 5, "min_epochs": 2}},
    }

    resolved = ResolvedConfig.from_mapping(config, tmp_path)
    assert resolved.paths.processed_dir == tmp_path
    assert resolved.paths.received_dir.name == "received"
    assert resolved.coverage.min_epochs == 2


def test_resolve_mapping_spec_keeps_legacy_axes_when_partial_override():
    """Test resolve mapping spec keeps legacy axes when partial override."""
    from mndm.pipeline.context import resolve_mapping_spec

    config = {
        "mnps_projection": {
            "normalize": "z",
            "weights": {
                "m": {"a": 0.5},
                "d": {"b": 1.0},
                "e": {"c": 2.0},
            },
        },
        "ndt_ingest": {
            "mnps_mapping": {
                "m_axis": {"weights": {"x": 1.0}},
            }
        },
    }
    weights, normalize, ingest_meta = resolve_mapping_spec(config)
    assert weights["m"] == {"x": 1.0}
    assert weights["d"] == {"b": 1.0}
    assert weights["e"] == {"c": 2.0}
    assert normalize == "z"
    assert ingest_meta["mapping_source"] == "ndt_ingest.mnps_mapping"


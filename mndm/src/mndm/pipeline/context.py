"""
context.py
Shared configuration context for orchestrator commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from core.paths import resolve_paths
from ..reproducibility import resolve_reproducibility_policy

logger = logging.getLogger(__name__)


def _normalize_mode(value: Any) -> Optional[str]:
    """Normalize projection mode to canonical values used in pipeline code."""
    if value is None:
        return None
    if isinstance(value, str):
        mode = value.strip().lower()
        if not mode:
            return None
        if mode == "robust":
            return "robust_z"
        if mode in {"z", "robust_z"}:
            return mode
    logger.warning("Unknown normalize mode '%s'; disabling normalization", value)
    return None


def mnps_config_with_overrides(
    config: Mapping[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge MNPS configuration with optional CLI overrides."""
    base = config.get("mnps", {}) if isinstance(config, Mapping) else {}
    if not isinstance(base, Mapping):
        base = {}
    overrides = overrides or {}

    derivative = dict(base.get("derivative", {}))
    derivative.update(overrides.get("derivative", {}))

    knn_cfg = dict(base.get("knn", {}))
    if "knn_k" in overrides:
        try:
            knn_cfg["k"] = int(overrides["knn_k"])
        except Exception as exc:
            raise ValueError(f"knn_k must be an integer, got {overrides['knn_k']!r}") from exc
    metric = knn_cfg.get("metric", "euclidean")
    if str(metric).lower() != "euclidean":
        logger.warning("Only euclidean knn metric is supported; got '%s', forcing euclidean", metric)
        metric = "euclidean"
    epoching_cfg = config.get("epoching", {}) if isinstance(config, Mapping) else {}
    if not isinstance(epoching_cfg, Mapping):
        epoching_cfg = {}
    if "window_sec" not in base:
        logger.warning("mnps.window_sec not set; falling back to epoching.length_s")

    mnps_cfg = {
        # fs_out is retained as a configuration hint for downstream consumers.
        "fs_out": base.get("fs_out", 4.0),
        "window_sec": base.get("window_sec", epoching_cfg.get("length_s", 8.0)),
        "overlap": base.get("overlap", 0.5),
        "derivative": derivative,
        "knn_k": int(knn_cfg.get("k", 20)),
        "knn_metric": metric,
        "ridge_alpha": float(base.get("ridge", {}).get("alpha", 1.0)),
        "super_window": int(overrides.get("super_window", base.get("super_window", 3))),
        "stage_codebook": base.get("stage_codebook", {}),
        "embodied": base.get("embodied", {}),
        "surrogates": base.get("surrogates", {}),
        "reliability": base.get("reliability", {}),
        "whiten": bool(base.get("whiten", True)),
    }
    return mnps_cfg


def resolve_mapping_spec(config: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, float]], Optional[str], Dict[str, str]]:
    """Return MNPS axis weights, normalize mode, and ingest metadata."""
    legacy_proj = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
    if not isinstance(legacy_proj, Mapping):
        legacy_proj = {}
    legacy_weights = legacy_proj.get("weights", {})
    if not isinstance(legacy_weights, Mapping):
        legacy_weights = {}
    weights = {
        axis: dict(legacy_weights.get(axis, {})) if isinstance(legacy_weights.get(axis, {}), Mapping) else {}
        for axis in ("m", "d", "e")
    }
    normalize = _normalize_mode(legacy_proj.get("normalize"))

    ingest_meta: Dict[str, str] = {"mapping_source": "mnps_projection"}

    # fMRI configs use mnps_projection.v1_mapping; legacy EEG configs use
    # mnps_projection.weights.  When weights is completely empty (fMRI case),
    # fall back to v1_mapping so that project_features() has something to work
    # with in direct-feature contexts (e.g. per-network projection).
    if all(not v for v in weights.values()):
        v1_mapping_cfg = legacy_proj.get("v1_mapping", {})
        if isinstance(v1_mapping_cfg, Mapping) and v1_mapping_cfg:
            for axis in ("m", "d", "e"):
                row = v1_mapping_cfg.get(axis, {})
                if isinstance(row, Mapping) and row:
                    weights[axis] = {str(k): float(v) for k, v in row.items()}
            ingest_meta["mapping_source"] = "mnps_projection.v1_mapping"
    ndt_cfg = config.get("ndt_ingest", {}) if isinstance(config, Mapping) else {}
    if isinstance(ndt_cfg, Mapping) and ndt_cfg:
        ingest_version = ndt_cfg.get("ingest_version") or ndt_cfg.get("version")
        if ingest_version:
            ingest_meta["ingest_version"] = str(ingest_version)
        mapping_cfg = ndt_cfg.get("mnps_mapping", {})
        if isinstance(mapping_cfg, Mapping) and mapping_cfg:
            mapping_version = mapping_cfg.get("version") or ndt_cfg.get("mapping_version")
            if mapping_version:
                ingest_meta["mapping_version"] = str(mapping_version)
            ingest_meta["mapping_source"] = "ndt_ingest.mnps_mapping"
            normalize = _normalize_mode(mapping_cfg.get("normalize", normalize))
            for axis in ("m", "d", "e"):
                axis_weights = _extract_axis_weights(mapping_cfg.get(f"{axis}_axis", {}), axis=axis)
                if axis_weights:
                    weights[axis] = axis_weights

    for axis in ("m", "d", "e"):
        weights.setdefault(axis, {})
    ingest_meta["normalize_mode_effective"] = str(normalize) if normalize is not None else "none"

    return weights, normalize, ingest_meta


def _extract_axis_weights(axis_cfg: Mapping[str, Any], axis: Optional[str] = None) -> Dict[str, float]:
    if not isinstance(axis_cfg, Mapping):
        return {}
    weights_map = axis_cfg.get("weights")
    if isinstance(weights_map, Mapping) and weights_map:
        return {str(k): float(v) for k, v in weights_map.items()}
    inferred: Dict[str, float] = {}
    for key, value in axis_cfg.items():
        if isinstance(value, (int, float)):
            inferred[str(key)] = float(value)
    if inferred:
        logger.warning(
            "Inferred %d weight(s) for axis '%s' from numeric top-level keys; "
            "prefer explicit '<axis>_axis.weights' mapping to avoid ambiguity",
            len(inferred),
            axis or "unknown",
        )
    return inferred


@dataclass(frozen=True)
class PathsConfig:
    received_dir: Path
    processed_dir: Path


@dataclass(frozen=True)
class CoverageConfig:
    min_seconds: float
    min_epochs: int


@dataclass(frozen=True)
class ResolvedConfig:
    """Typed configuration shared across CLI commands."""

    raw: Mapping[str, Any]
    paths: PathsConfig
    coverage: CoverageConfig
    weights: Dict[str, Dict[str, float]]
    normalize_override: Optional[str]
    ingest_meta: Dict[str, str]
    mnps_cfg: Dict[str, Any]
    derivative_cfg: Dict[str, Any]
    extensions_cfg: Dict[str, Any]
    reproducibility: Dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        out_dir: Path | None,
        cli_data_dir: Path | None = None,
        mnps_overrides: Optional[Dict[str, Any]] = None,
    ) -> "ResolvedConfig":
        received_dir, processed_dir = resolve_paths(config, out_dir, cli_data_dir)
        robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
        if not isinstance(robustness_cfg, Mapping):
            robustness_cfg = {}
        coverage_cfg = robustness_cfg.get("coverage", {}) if isinstance(robustness_cfg, Mapping) else {}
        if not isinstance(coverage_cfg, Mapping):
            coverage_cfg = {}
        coverage = CoverageConfig(
            min_seconds=float(coverage_cfg.get("min_seconds", 0.0) or 0.0),
            min_epochs=int(coverage_cfg.get("min_epochs", 0) or 0),
        )
        weights, normalize_override, ingest_meta = resolve_mapping_spec(config)
        mnps_cfg = mnps_config_with_overrides(config, mnps_overrides)
        reproducibility = resolve_reproducibility_policy(config)
        derivative_cfg = {
            "method": mnps_cfg["derivative"].get("method", "sav_gol"),
            "window": int(mnps_cfg["derivative"].get("window", 7)),
            "polyorder": int(mnps_cfg["derivative"].get("polyorder", 3)),
        }
        extensions_cfg = config.get("mnps_extensions", {}) if isinstance(config, Mapping) else {}
        return cls(
            raw=config,
            paths=PathsConfig(received_dir=received_dir, processed_dir=processed_dir),
            coverage=coverage,
            weights=weights,
            normalize_override=normalize_override,
            ingest_meta=ingest_meta,
            mnps_cfg=mnps_cfg,
            derivative_cfg=derivative_cfg,
            extensions_cfg=extensions_cfg if isinstance(extensions_cfg, Mapping) else {},
            reproducibility=reproducibility,
        )


@dataclass(frozen=True)
class SummarizeContext:
    """Container for immutable summarization configuration.

    Wraps a ResolvedConfig and exposes convenience properties for
    backward compatibility with existing consumer code.
    """

    resolved: ResolvedConfig

    @classmethod
    def from_resolved(cls, resolved: ResolvedConfig) -> "SummarizeContext":
        return cls(resolved=resolved)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        out_dir: Path | None,
        cli_data_dir: Path | None = None,
        mnps_overrides: Optional[Dict[str, Any]] = None,
    ) -> "SummarizeContext":
        resolved = ResolvedConfig.from_mapping(config, out_dir, cli_data_dir, mnps_overrides)
        return cls.from_resolved(resolved)

    # --- Convenience properties for backward compatibility ---

    @property
    def config(self) -> Mapping[str, Any]:
        return self.resolved.raw

    @property
    def received_dir(self) -> Path:
        return self.resolved.paths.received_dir

    @property
    def processed_dir(self) -> Path:
        return self.resolved.paths.processed_dir

    @property
    def coverage(self) -> CoverageConfig:
        return self.resolved.coverage

    @property
    def weights(self) -> Dict[str, Dict[str, float]]:
        return self.resolved.weights

    @property
    def normalize_override(self) -> Optional[str]:
        return self.resolved.normalize_override

    @property
    def ingest_meta(self) -> Dict[str, str]:
        return self.resolved.ingest_meta

    @property
    def mnps_cfg(self) -> Dict[str, Any]:
        return self.resolved.mnps_cfg

    @property
    def derivative_cfg(self) -> Dict[str, Any]:
        return self.resolved.derivative_cfg

    @property
    def extensions_cfg(self) -> Dict[str, Any]:
        return self.resolved.extensions_cfg

    @property
    def reproducibility(self) -> Dict[str, Any]:
        return self.resolved.reproducibility

    @property
    def dt(self) -> float:
        """Effective step between MNPS samples from window overlap.

        Note:
            This is derived as `window_sec * (1-overlap)` and is independent of
            `mnps_cfg["fs_out"]` (which is treated as a configuration hint).
        """
        return self.mnps_cfg["window_sec"] * (1.0 - self.mnps_cfg["overlap"])


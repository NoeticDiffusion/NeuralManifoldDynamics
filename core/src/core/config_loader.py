"""Configuration loader and validator for the ingest pipeline.

Responsibilities
----------------
- Parse YAML configuration (datasets, defaults, feature sets, MNPS weights,
  robustness parameters, and IO paths).
- Support optional config composition via an ``imports`` list in YAML
  (deep-merged in order; local file overrides imported keys).
- Provide a typed dictionary (or dataclass in future) used by downstream steps.

Inputs
------
- path: filesystem path to `config_ingest.yaml`.

Outputs
-------
- Dict-like configuration object with keys such as:
  {"datasets": [...], "processing": {...}, "features": {...}, "mnps": {...}}.

Dependencies
------------
- None at import-time. Optional: `yaml` (PyYAML) during `load_config` execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries: override wins; nested mappings merge recursively."""
    out: Dict[str, Any] = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(override, Mapping):
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge_dict(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def _load_yaml_with_imports(path: Path, stack: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Load YAML with optional recursive ``imports`` composition."""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - import error surfaced at runtime
        raise RuntimeError("PyYAML is required to load configuration") from exc

    p = Path(path).resolve()
    stack = stack or []
    if p in stack:
        chain = " -> ".join(str(x) for x in [*stack, p])
        raise ValueError(f"Config import cycle detected: {chain}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping/dict: {p}")

    imports = raw.get("imports", [])
    if imports is None:
        imports = []
    if not isinstance(imports, list):
        raise ValueError(f"'imports' must be a list in config: {p}")

    merged: Dict[str, Any] = {}
    for imp in imports:
        imp_path = Path(str(imp))
        if not imp_path.is_absolute():
            imp_path = (p.parent / imp_path).resolve()
        imported_cfg = _load_yaml_with_imports(imp_path, stack=[*stack, p])
        merged = _deep_merge_dict(merged, imported_cfg)

    local_cfg = dict(raw)
    local_cfg.pop("imports", None)
    merged = _deep_merge_dict(merged, local_cfg)
    return merged


def load_config(path: Path) -> Dict[str, Any]:
    """Load the ingest configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration with minimal validation.

    Raises
    ------
    RuntimeError
        If PyYAML is not available.
    ValueError
        If configuration structure is invalid.
    """
    cfg = _load_yaml_with_imports(Path(path))

    # Validate required sections for *ingest* configs.
    #
    # Note: This loader is also used for non-ingest YAML specs (e.g. structure-check
    # specs) which intentionally do not contain the ingest sections below. We only
    # emit the "Missing config sections" warning when the YAML looks like an ingest
    # config (i.e., it declares either datasets/paths, or any other ingest section).
    required_sections = ["datasets", "paths", "preprocess", "epoching", "features", "mnps_projection", "robustness"]
    looks_like_ingest = any(k in cfg for k in ("datasets", "paths", "preprocess", "epoching", "features", "mnps_projection", "robustness"))
    if looks_like_ingest:
        missing = [s for s in required_sections if s not in cfg]
        if missing:
            logger.warning(f"Missing config sections: {missing}")

    return cfg



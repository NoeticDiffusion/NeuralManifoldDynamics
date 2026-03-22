"""JSON writer for MNPS tensor manifest outputs.

Builds compact JSON manifests summarizing samples, MNPS axes, Jacobian shapes,
and optional diagnostics; also writes arbitrary JSON summaries to disk.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np

from mndm.schema import MNPSPayload, compute_meta_indices, normalize_payload

logger = logging.getLogger(__name__)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable builtins."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return f if math.isfinite(f) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            # Fast-path for numeric arrays without non-finite values.
            if obj.dtype.kind in {"i", "u", "b"}:
                return obj.tolist()
            if obj.dtype.kind in {"f"} and np.isfinite(obj).all():
                return obj.tolist()
            # Fallback (object arrays / non-finite floats / mixed content)
            return _to_jsonable(obj.tolist())

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    return obj


def _is_scalar_manifest_value(value: Any) -> bool:
    """Internal helper: is scalar manifest value."""
    if isinstance(value, (str, bool, int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isfinite(float(value)))
    return False


def build_manifest(
    dataset_id: str,
    payload: MNPSPayload,
    diagnostics: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    """Build a JSON-serializable manifest dict for an MNPS payload.

    Args:
        dataset_id: Dataset identifier string.
        payload: MNPS payload (normalized internally).
        diagnostics: Optional diagnostics mapping; only scalar values are kept
            in the manifest (arrays stay in HDF5).
        extra: Optional extra keys merged into the manifest.

    Returns:
        Mutable mapping suitable for :func:`write_json_summary` or embedding in H5.
    """
    payload = normalize_payload(payload)
    events = payload.events or {}
    labels = payload.labels or {}
    jac = payload.jacobian
    jac_v2 = getattr(payload, "jacobian_9D", None)
    ws = getattr(payload, "window_start", None)
    we = getattr(payload, "window_end", None)
    manifest: MutableMapping[str, Any] = {
        "dataset_id": dataset_id,
        "samples": int(payload.time.shape[0]),
        "mnps": {
            "fs_out": payload.attrs.get("fs_out"),
            "window_sec": payload.attrs.get("window_sec"),
            "overlap": payload.attrs.get("overlap"),
            "window_bounds": bool(ws is not None and we is not None),
        },
        # Canonical MNPS axis definitions for consistency across the toolkit.
        "mnps_axes": {
            "names": ["m", "d", "e"],
            "meaning": {
                "m": "mobility",
                "d": "diffusivity",
                "e": "entropy",
            },
            "order": {"m": 0, "d": 1, "e": 2},
        },
        # Meta-indices for the primary MNPS Jacobian (typically 3D)
        "meta_indices": compute_meta_indices(jac) if jac is not None else {"windows": 0},
        "events": sorted(events.keys()),
        "labels": sorted(list(labels.keys()) + (["stage"] if payload.stage is not None else [])),
        "jacobian": {
            "windows": int(jac.shape[0]) if jac is not None else 0,
            "with_centers": bool(payload.jacobian_centers is not None),
        },
    }

    # Optional meta-indices for Stratified (v2) Jacobians when present
    if jac_v2 is not None and np.size(jac_v2) > 0:
        manifest["meta_indices_v2"] = compute_meta_indices(jac_v2)
        manifest["jacobian_9D"] = {
            "windows": int(jac_v2.shape[0]),
            "with_centers": bool(getattr(payload, "jacobian_9D_centers", None) is not None),
        }

    if diagnostics:
        # Keep summary JSON compact and stable; heavy per-window diagnostics go to H5.
        manifest["diagnostics"] = {
            str(k): _to_jsonable(v)
            for k, v in dict(diagnostics).items()
            if _is_scalar_manifest_value(v)
        }

    if payload.stage is not None:
        unique_codes = []
        for v in payload.stage.tolist():
            try:
                if v is None:
                    continue
                fv = float(v)
                if not math.isfinite(fv):
                    continue
                unique_codes.append(int(fv))
            except Exception:
                continue
        manifest["stage_codes"] = {
            "unique": sorted(set(unique_codes)),
            "codebook": payload.attrs.get("stage_codebook"),
        }

    # Include stratified MNPS (9D) info if available
    coords_9d_names = getattr(payload, "coords_9d_names", None)
    if coords_9d_names:
        m_names: list[str] = []
        d_names: list[str] = []
        e_names: list[str] = []
        for n in coords_9d_names:
            s = str(n)
            if s.startswith("m_"):
                m_names.append(s)
            elif s.startswith("d_"):
                d_names.append(s)
            elif s.startswith("e_"):
                e_names.append(s)
        manifest["coords_9d"] = {
            "names": list(coords_9d_names),
            "export_prefix": "v2_",
            "composites": {
                "m": "m_mean_from_v2",
                "d": "d_mean_from_v2",
                "e": "e_mean_from_v2",
            },
            "groups": {
                "m": m_names,
                "d": d_names,
                "e": e_names,
            },
        }

    # Include feature baselines if available
    feature_baselines = getattr(payload, "feature_baselines", None)
    if feature_baselines:
        manifest["feature_baselines"] = dict(feature_baselines)

    if extra:
        manifest.update({k: v for k, v in extra.items() if v is not None})

    return manifest


def write_json_summary(summary: Mapping[str, Any], out_path: Path) -> Path:
    """Write ``summary`` to ``out_path`` as UTF-8 JSON with indentation.

    Args:
        summary: JSON-serializable mapping (NumPy types are converted).
        out_path: Destination path.

    Returns:
        Resolved path that was written.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, ensure_ascii=False, indent=2, allow_nan=False)
    logger.info("Wrote JSON: %s", path)
    return path



"""Process-local atlas cache and persisted ROI time series (P1.1).

Atlas arrays are cached in-process by resolved path + mtime + size.
ROI-TS after parcellation/nuisance is written under
``processed_dir/<dataset>/intermediate/roi_ts/`` keyed by a content hash of
atlas identity, BOLD identity, TR, nuisance inputs, and space flags.
Summarize can reload that artifact without opening 4D BOLD.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .nuisance import DEFAULT_NUISANCE_COLUMNS

logger = logging.getLogger(__name__)

ROI_TS_CACHE_VERSION = "mndm.roi_ts.v2"

_ATLAS_CACHE: Dict[Tuple[str, int, int], "CachedAtlas"] = {}
_INDEX_NAME = "index.json"


@dataclass
class CachedAtlas:
    """Native atlas volume cached for reuse across BOLD files."""

    path: str
    affine: np.ndarray
    data: np.ndarray
    header_zooms: Tuple[float, ...]


def clear_atlas_cache() -> None:
    """Drop the process-local atlas cache (tests)."""
    _ATLAS_CACHE.clear()


def atlas_cache_size() -> int:
    """Number of native atlas volumes currently cached."""
    return len(_ATLAS_CACHE)


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = str(path.resolve()) if path.exists() else str(path)
    ident: Dict[str, Any] = {"path": resolved}
    try:
        st = path.stat()
        ident["mtime_ns"] = int(st.st_mtime_ns)
        ident["size"] = int(st.st_size)
    except OSError:
        ident["mtime_ns"] = None
        ident["size"] = None
    return ident


def _atlas_cache_key(path: Path) -> Tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))


def load_cached_atlas(atlas_path: Path) -> Tuple[Any, np.ndarray, bool]:
    """Load an atlas NIfTI, using the process-local cache when possible.

    Returns ``(nibabel image, data array, cache_hit)``.
    """
    import nibabel as nib  # type: ignore

    key = _atlas_cache_key(atlas_path)
    cached = _ATLAS_CACHE.get(key)
    if cached is not None:
        img = nib.Nifti1Image(cached.data, cached.affine)
        return img, cached.data, True
    img = nib.load(str(atlas_path))
    data = np.asarray(img.get_fdata())
    zooms = tuple(float(z) for z in img.header.get_zooms())
    _ATLAS_CACHE[key] = CachedAtlas(
        path=str(atlas_path.resolve()),
        affine=np.asarray(img.affine, dtype=float).copy(),
        data=data,
        header_zooms=zooms,
    )
    return img, data, False


def roi_ts_cache_dir(config: Mapping[str, Any], dataset_id: Optional[str]) -> Optional[Path]:
    """Return ``.../intermediate/roi_ts`` or None when processed_dir is unset."""
    paths = config.get("paths") if isinstance(config, Mapping) else None
    if not isinstance(paths, Mapping):
        return None
    processed = paths.get("processed_dir")
    if not processed:
        return None
    ds = str(dataset_id or "unknown")
    out = Path(processed) / ds / "intermediate" / "roi_ts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def roi_ts_identity_payload(
    *,
    bold_path: Path,
    atlas_path: Path,
    tr_sec: float,
    tr_source: str,
    nuisance_enabled: bool,
    confounds_path: Optional[Path],
    space_flags: Mapping[str, Any],
    atlas_labels_path: Optional[Path] = None,
    nuisance_columns: Optional[Sequence[str]] = None,
    confounds_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """Canonical identity for an ROI-TS artifact (hashed as the cache key)."""
    cols = [str(c) for c in (nuisance_columns if nuisance_columns is not None else DEFAULT_NUISANCE_COLUMNS)]
    return {
        "version": ROI_TS_CACHE_VERSION,
        "bold": _file_identity(bold_path),
        "atlas": _file_identity(atlas_path),
        "atlas_labels": _file_identity(atlas_labels_path) if atlas_labels_path is not None else None,
        "tr_sec": float(tr_sec),
        "tr_source": str(tr_source),
        "nuisance_enabled": bool(nuisance_enabled),
        "nuisance_columns": sorted(cols),
        "confounds_suffix": str(confounds_suffix or ""),
        "confounds": _file_identity(confounds_path) if confounds_path is not None else None,
        "space": {
            "resample_atlas_to_bold": bool(space_flags.get("resample_atlas_to_bold", False)),
            "assume_same_space": bool(space_flags.get("assume_same_space", False)),
            "atlas_affine_atol_mm": space_flags.get("atlas_affine_atol_mm"),
        },
    }


def roi_ts_config_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Identity fields that must match current config (excludes BOLD mtime/size)."""
    return {k: payload.get(k) for k in payload if k != "bold"}


def identities_match(stored: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """True when config-side cache identity matches (JSON-canonical)."""
    left = roi_ts_config_identity(stored)
    right = roi_ts_config_identity(current)
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def build_roi_ts_cache_key(
    *,
    bold_path: Path,
    atlas_path: Path,
    tr_sec: float,
    tr_source: str,
    nuisance_enabled: bool,
    confounds_path: Optional[Path],
    space_flags: Mapping[str, Any],
    atlas_labels_path: Optional[Path] = None,
    nuisance_columns: Optional[Sequence[str]] = None,
    confounds_suffix: Optional[str] = None,
) -> str:
    """Stable hex digest for one ROI-TS artifact."""
    payload = roi_ts_identity_payload(
        bold_path=bold_path,
        atlas_path=atlas_path,
        tr_sec=tr_sec,
        tr_source=tr_source,
        nuisance_enabled=nuisance_enabled,
        confounds_path=confounds_path,
        space_flags=space_flags,
        atlas_labels_path=atlas_labels_path,
        nuisance_columns=nuisance_columns,
        confounds_suffix=confounds_suffix,
    )
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / _INDEX_NAME


def _read_index(cache_dir: Path) -> Dict[str, Any]:
    path = _index_path(cache_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_index(cache_dir: Path, index: Mapping[str, Any]) -> None:
    path = _index_path(cache_dir)
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def _lookup_index_entry(index: Mapping[str, Any], bold_path: Path) -> Optional[Dict[str, Any]]:
    candidates = [str(bold_path)]
    try:
        candidates.append(str(bold_path.resolve()))
    except OSError:
        pass
    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        entry = index.get(key)
        if isinstance(entry, dict):
            return entry
    return None


def save_roi_ts(
    *,
    cache_dir: Path,
    key: str,
    bold_path: Path,
    roi_ts: np.ndarray,
    names: list[str],
    sfreq: float,
    meta: Mapping[str, Any],
    identity: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist ROI-TS + JSON sidecar and update the path index."""
    npz_path = cache_dir / f"{key}.npz"
    json_path = cache_dir / f"{key}.json"
    np.savez_compressed(
        npz_path,
        roi_ts=np.asarray(roi_ts, dtype=np.float32),
        sfreq=np.float64(sfreq),
    )
    sidecar = {
        "key": key,
        "version": ROI_TS_CACHE_VERSION,
        "bold_path": str(bold_path),
        "names": [str(n) for n in names],
        "meta": dict(meta),
        "identity": dict(identity) if identity is not None else None,
    }
    json_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    index = _read_index(cache_dir)
    ident = _file_identity(bold_path)
    entry = {"key": key, "npz": npz_path.name, "json": json_path.name, **ident}
    index[str(bold_path)] = entry
    try:
        index[str(bold_path.resolve())] = entry
    except OSError:
        pass
    _write_index(cache_dir, index)
    return npz_path


def try_load_roi_ts(
    *,
    bold_path: Path,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
    expected_key: Optional[str] = None,
    config_identity: Optional[Mapping[str, Any]] = None,
    require_bold_exists: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load a persisted ROI-TS artifact if the index and identity still match.

    Fail closed unless ``expected_key`` and/or ``config_identity`` is provided.
    When the NIfTI is missing, ``config_identity`` still has to match the
    sidecar (atlas, TR, nuisance, space, label TSV). BOLD mtime is checked
    only when the file exists.
    """
    if expected_key is None and config_identity is None:
        return None
    cache_dir = roi_ts_cache_dir(config, dataset_id)
    if cache_dir is None:
        return None
    index = _read_index(cache_dir)
    entry = _lookup_index_entry(index, bold_path)
    if entry is None:
        return None
    key = str(entry.get("key") or "")
    if not key:
        return None
    if expected_key is not None and key != expected_key:
        return None
    if bold_path.exists():
        current = _file_identity(bold_path)
        stored_mtime = entry.get("mtime_ns")
        stored_size = entry.get("size")
        if stored_mtime is not None and current.get("mtime_ns") != stored_mtime:
            return None
        if stored_size is not None and current.get("size") != stored_size:
            return None
    elif require_bold_exists:
        return None
    npz_path = cache_dir / str(entry.get("npz") or f"{key}.npz")
    json_path = cache_dir / str(entry.get("json") or f"{key}.json")
    if not npz_path.exists() or not json_path.exists():
        return None
    try:
        with np.load(npz_path) as payload:
            roi_ts = np.asarray(payload["roi_ts"], dtype=np.float32)
            sfreq = float(np.asarray(payload["sfreq"]))
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read ROI-TS cache %s", npz_path, exc_info=True)
        return None
    if config_identity is not None:
        stored_identity = sidecar.get("identity")
        if not isinstance(stored_identity, Mapping) or not identities_match(stored_identity, config_identity):
            return None
    names = [str(n) for n in sidecar.get("names", [])]
    meta = sidecar.get("meta") if isinstance(sidecar.get("meta"), Mapping) else {}
    meta = dict(meta)
    meta["roi_ts_cache_hit"] = 1
    meta["roi_ts_cache_key"] = key
    return {
        "roi_ts": roi_ts,
        "names": names,
        "sfreq": sfreq,
        "meta": meta,
    }

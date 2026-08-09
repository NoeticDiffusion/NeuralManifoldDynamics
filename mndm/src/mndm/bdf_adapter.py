"""Configurable adapter utilities for non-BIDS BioSemi BDF recordings.

The adapter deliberately separates reusable BDF mechanics from dataset policy:
YAML supplies filename/mapping tables, the active channel bank, canonical names,
and optional QC tables.  This keeps source-specific conventions out of the
generic BIDS indexer and preprocessing path.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import logging
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, or an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def resolve_bdf_adapter_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve global and dataset-specific BDF adapter policy."""
    preprocess = _as_mapping(config.get("preprocess"))
    base = dict(_as_mapping(preprocess.get("bdf_adapter")))
    datasets = _as_mapping(base.pop("datasets", {}))
    override = _as_mapping(datasets.get(str(dataset_id), {})) if dataset_id else {}
    merged = dict(base)
    merged.update(override)
    return merged


def _absolute_path(value: Any, dataset_root: Optional[Path]) -> Optional[Path]:
    """Resolve configured paths relative to the BDF dataset root."""
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute() and dataset_root is not None:
        path = dataset_root / path
    return path


@lru_cache(maxsize=16)
def _read_csv(path_str: str) -> pd.DataFrame:
    """Read and cache a small external BDF metadata CSV."""
    return pd.read_csv(path_str)


def _normalise_entity(value: Any, prefix: str, pad: int = 0) -> Optional[str]:
    """Return a BIDS-like entity token without its prefix."""
    if value is None or pd.isna(value):
        return None
    token = str(value).strip()
    if not token:
        return None
    if token.startswith(prefix):
        token = token[len(prefix) :]
    if pad and token.isdigit():
        token = token.zfill(pad)
    return token


def _mapping_row(
    stem: str,
    adapter_cfg: Mapping[str, Any],
    dataset_root: Optional[Path],
) -> Optional[Mapping[str, Any]]:
    """Look up one raw BDF stem in an optional mapping table."""
    mapping_path = _absolute_path(adapter_cfg.get("mapping_path"), dataset_root)
    if mapping_path is None or not mapping_path.exists():
        return None
    try:
        frame = _read_csv(str(mapping_path))
    except Exception as exc:
        logger.warning("Unable to read BDF mapping table %s: %s", mapping_path, exc)
        return None
    key_col = str(adapter_cfg.get("mapping_key_column", "old_id"))
    if key_col not in frame.columns:
        logger.warning("BDF mapping table %s lacks key column %s", mapping_path, key_col)
        return None
    hits = frame.loc[frame[key_col].astype(str) == stem]
    if len(hits) != 1:
        if len(hits) > 1:
            logger.warning("BDF mapping table has %d rows for %s", len(hits), stem)
        return None
    return hits.iloc[0].to_dict()


def parse_bdf_entities(
    file_path: Path,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
    dataset_root: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """Derive BIDS-style entities for a non-BIDS BDF filename.

    Mapping tables take precedence.  They permit sources to define arbitrary
    recording names while a named-group filename regex provides a portable
    fallback for simpler BDF collections.
    """
    adapter_cfg = resolve_bdf_adapter_config(config, dataset_id)
    result: Dict[str, Optional[str]] = {
        "subject": None,
        "session": None,
        "task": adapter_cfg.get("task"),
        "run": None,
        "acq": None,
        "mapping_id": None,
    }
    if not bool(adapter_cfg.get("enabled", False)) or file_path.suffix.lower() != ".bdf":
        return result

    row = _mapping_row(file_path.stem, adapter_cfg, dataset_root)
    if row:
        result["mapping_id"] = str(row.get(str(adapter_cfg.get("mapping_key_column", "old_id")), file_path.stem))
        subject_col = str(adapter_cfg.get("mapping_subject_column", "subject"))
        session_col = str(adapter_cfg.get("mapping_session_column", "session"))
        result["subject"] = _normalise_entity(
            row.get(subject_col),
            "sub-",
            int(adapter_cfg.get("subject_pad", 0) or 0),
        )
        result["session"] = _normalise_entity(
            row.get(session_col),
            "ses-",
            int(adapter_cfg.get("session_pad", 0) or 0),
        )

        # A mapping may have only ``sub-XX_ses-YY`` in one column.
        combined_col = str(adapter_cfg.get("mapping_subject_session_column", "subject_id"))
        combined = row.get(combined_col)
        if combined is not None and (result["subject"] is None or result["session"] is None):
            match = re.search(r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)", str(combined))
            if match:
                result["subject"] = result["subject"] or match.group("subject")
                result["session"] = result["session"] or match.group("session")
        return result

    regex = str(adapter_cfg.get("filename_regex", "") or "")
    if not regex:
        return result
    match = re.search(regex, file_path.name)
    if not match:
        return result
    groups = match.groupdict()
    result["subject"] = _normalise_entity(
        groups.get("subject"),
        "sub-",
        int(adapter_cfg.get("subject_pad", 0) or 0),
    )
    result["session"] = _normalise_entity(
        groups.get("session"),
        "ses-",
        int(adapter_cfg.get("session_pad", 0) or 0),
    )
    result["run"] = groups.get("run")
    result["acq"] = groups.get("acq")
    result["mapping_id"] = file_path.stem
    return result


def _bad_channels_for_recording(
    file_path: Path,
    adapter_cfg: Mapping[str, Any],
    dataset_root: Optional[Path],
) -> list[str]:
    """Read a list-like bad-channel field for one mapped recording."""
    path = _absolute_path(adapter_cfg.get("bad_channels_path"), dataset_root)
    if path is None or not path.exists():
        return []
    row = _mapping_row(file_path.stem, adapter_cfg, dataset_root)
    if not row:
        return []
    key = str(adapter_cfg.get("bad_channels_key_column", "subject_id"))
    if key not in row:
        return []
    try:
        table = _read_csv(str(path))
    except Exception as exc:
        logger.warning("Unable to read BDF bad-channel table %s: %s", path, exc)
        return []
    value_col = str(adapter_cfg.get("bad_channels_column", "bad_channels"))
    if key not in table.columns or value_col not in table.columns:
        logger.warning("BDF bad-channel table %s lacks %s or %s", path, key, value_col)
        return []
    hits = table.loc[table[key].astype(str) == str(row[key])]
    if len(hits) != 1:
        return []
    raw_value = hits.iloc[0][value_col]
    if pd.isna(raw_value):
        return []
    try:
        parsed = ast.literal_eval(str(raw_value))
        if isinstance(parsed, (list, tuple)):
            return [str(channel).strip() for channel in parsed if str(channel).strip()]
    except (SyntaxError, ValueError):
        pass
    return [token.strip() for token in str(raw_value).strip("[]").split(",") if token.strip()]


def _bad_segment_annotation_path(
    file_path: Path,
    adapter_cfg: Mapping[str, Any],
    dataset_root: Optional[Path],
) -> Optional[Path]:
    """Resolve an optional MNE annotation CSV for a mapped recording."""
    directory = _absolute_path(adapter_cfg.get("bad_segments_dir"), dataset_root)
    if directory is None:
        return None
    row = _mapping_row(file_path.stem, adapter_cfg, dataset_root)
    if not row:
        return None
    key = str(adapter_cfg.get("bad_segments_key_column", "subject_id"))
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    template = str(adapter_cfg.get("bad_segments_filename_template", "{subject_id}_annotations.csv"))
    try:
        file_name = template.format(**{key: str(value), "subject_id": str(value)})
    except (KeyError, ValueError):
        logger.warning("Invalid bad-segment filename template '%s'", template)
        return None
    path = directory / file_name
    return path if path.exists() else None


def _attach_bad_segment_annotations(
    raw: Any,
    file_path: Path,
    adapter_cfg: Mapping[str, Any],
    dataset_root: Optional[Path],
) -> Dict[str, Any]:
    """Attach public BAD_ annotations without altering the continuous signal."""
    path = _bad_segment_annotation_path(file_path, adapter_cfg, dataset_root)
    if path is None:
        return {"enabled": bool(adapter_cfg.get("mask_bad_segments", False)), "available": False}
    try:
        import mne

        annotations = mne.read_annotations(path)
        raw.set_annotations(annotations)
        descriptions = [str(value) for value in annotations.description]
        return {
            "enabled": bool(adapter_cfg.get("mask_bad_segments", False)),
            "available": True,
            "path": str(path),
            "count": int(len(annotations)),
            "duration_sec": float(np.sum(annotations.duration)),
            "descriptions": sorted(set(descriptions)),
        }
    except Exception as exc:
        logger.warning("Unable to load BDF bad-segment annotations for %s: %s", file_path.name, exc)
        return {
            "enabled": bool(adapter_cfg.get("mask_bad_segments", False)),
            "available": False,
            "load_error": str(exc),
        }


def apply_bdf_adapter_policy(
    raw: Any,
    file_path: Path,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
    dataset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Apply BDF channel/QC policy to an MNE Raw object.

    Returns auditable metadata for the preprocessing result.  Selection occurs
    before filtering/resampling so unused BioSemi banks and auxiliaries do not
    consume memory downstream.
    """
    adapter_cfg = resolve_bdf_adapter_config(config, dataset_id)
    if not bool(adapter_cfg.get("enabled", False)) or file_path.suffix.lower() != ".bdf":
        return {"applied": False}

    meta: Dict[str, Any] = {"applied": True, "original_channel_count": len(raw.ch_names)}
    selection = _as_mapping(adapter_cfg.get("channel_selection"))
    mode = str(selection.get("mode", "first_n")).lower()
    if mode == "first_n":
        n_channels = int(selection.get("n_channels", 32) or 32)
        selected = list(raw.ch_names[:n_channels])
    elif mode == "names":
        selected = [str(name) for name in selection.get("channels", []) if str(name) in raw.ch_names]
    else:
        raise ValueError(f"Unsupported BDF channel selection mode: {mode}")
    if not selected:
        raise ValueError(f"BDF adapter selected no channels for {file_path.name}")
    raw.pick(selected)
    meta["selected_channel_count"] = len(raw.ch_names)
    meta["channel_selection_mode"] = mode

    canonical = [str(name) for name in adapter_cfg.get("canonical_channel_names", [])]
    if canonical:
        if len(canonical) != len(raw.ch_names):
            raise ValueError(
                f"BDF canonical channel list has {len(canonical)} names but "
                f"{file_path.name} selected {len(raw.ch_names)} channels"
            )
        raw.rename_channels(dict(zip(raw.ch_names, canonical)))
    meta["canonical_channels"] = list(raw.ch_names)

    montage_name = adapter_cfg.get("montage")
    if montage_name:
        try:
            import mne

            raw.set_montage(mne.channels.make_standard_montage(str(montage_name)), on_missing="ignore")
            meta["montage"] = str(montage_name)
        except Exception as exc:
            logger.warning("BDF montage assignment failed for %s: %s", file_path.name, exc)
            meta["montage_error"] = str(exc)

    bad_channels = [name for name in _bad_channels_for_recording(file_path, adapter_cfg, dataset_root) if name in raw.ch_names]
    raw.info["bads"] = sorted(set(raw.info.get("bads", [])).union(bad_channels))
    meta["bad_channels"] = list(raw.info["bads"])
    meta["interpolate_bads"] = bool(adapter_cfg.get("interpolate_bads", False))
    meta["bad_segments"] = _attach_bad_segment_annotations(raw, file_path, adapter_cfg, dataset_root)

    crop_overrides = _as_mapping(adapter_cfg.get("recording_crops"))
    crop = _as_mapping(crop_overrides.get(file_path.stem))
    if crop:
        start = float(crop.get("start_sec", 0.0) or 0.0)
        stop_raw = crop.get("stop_sec")
        stop = float(stop_raw) if stop_raw is not None else None
        if stop is None or stop > start:
            raw.crop(tmin=max(0.0, start), tmax=stop)
            meta["recording_crop"] = {"start_sec": start, "stop_sec": stop}

    return meta


def interpolate_configured_bdf_bads(raw: Any, adapter_meta: Mapping[str, Any]) -> None:
    """Interpolate adapter-marked bad channels once raw samples are loaded."""
    if not adapter_meta.get("applied") or not adapter_meta.get("interpolate_bads") or not raw.info.get("bads"):
        return
    try:
        raw.interpolate_bads(reset_bads=False, verbose=False)
    except Exception as exc:
        logger.warning("BDF bad-channel interpolation failed: %s", exc)

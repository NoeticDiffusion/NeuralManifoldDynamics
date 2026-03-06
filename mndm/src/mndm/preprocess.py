"""Signal preprocessing utilities (resample, filter, rereference).

Responsibilities
----------------
- Apply per-modality bandpass and notch filters.
- Resample to a common sampling frequency.
- Implement average re-reference for EEG as configured.

Inputs
------
- file_path: path to raw signal file (e.g., EDF/BrainVision).
- config: dict with preprocessing settings (sfreq, filters, notch, reref).

Outputs
-------
- PreprocessedSignals dataclass with signals dict, sfreq, channels, meta.

Dependencies
------------
- mne, numpy, scipy for signal processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from uuid import uuid4

# Disable numba JIT to avoid long import stalls on Windows/venv
os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

try:
    import mne
except Exception:  # pragma: no cover - optional dependency
    mne = None  # type: ignore
import numpy as np
import pandas as pd

try:
    # Optional preprocessing utilities (may not be available in minimal envs)
    from mne.preprocessing import ICA
except Exception:  # pragma: no cover
    ICA = None  # type: ignore
try:
    from mne.preprocessing import compute_current_source_density
except Exception:  # pragma: no cover
    compute_current_source_density = None  # type: ignore

logger = logging.getLogger(__name__)


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries: override wins; nested mappings are merged recursively."""
    out: Dict[str, Any] = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(override, Mapping):
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge_dict(out.get(k, {}), v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=128)
def _load_channel_typing_policy(policy_dir: str, dataset_id: str) -> Dict[str, Any]:
    """Load optional per-dataset channel typing policy from YAML."""
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not policy_dir or not dataset_id:
        return {}

    root = Path(policy_dir)
    ds_path = root / f"{dataset_id}_channel_typing.yml"
    map_path = root / "datasets.yml"
    path = ds_path if ds_path.exists() else (map_path if map_path.exists() else None)
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    payload: Dict[str, Any] = {}
    if path.name == "datasets.yml":
        if isinstance(data, Mapping):
            # Supported shapes:
            # - {datasets: {<ds>: {...}}}
            # - {preprocess: {channel_typing: {datasets: {<ds>: {...}}}}}
            # - {channel_typing: {datasets: {<ds>: {...}}}}
            # - {<ds>: {...}}
            if isinstance(data.get("preprocess"), Mapping):
                p = data.get("preprocess") or {}
                ct = p.get("channel_typing", {}) if isinstance(p, Mapping) else {}
                ds_map = ct.get("datasets", {}) if isinstance(ct, Mapping) else {}
                if isinstance(ds_map, Mapping):
                    payload = dict(ds_map.get(dataset_id, {}) or {})
            if not payload and isinstance(data.get("channel_typing"), Mapping):
                ct = data.get("channel_typing") or {}
                ds_map = ct.get("datasets", {}) if isinstance(ct, Mapping) else {}
                if isinstance(ds_map, Mapping):
                    payload = dict(ds_map.get(dataset_id, {}) or {})
            if not payload and isinstance(data.get("datasets"), Mapping):
                payload = dict((data.get("datasets") or {}).get(dataset_id, {}) or {})
            if not payload and isinstance(data.get(dataset_id), Mapping):
                payload = dict(data.get(dataset_id) or {})
    else:
        if isinstance(data, Mapping) and isinstance(data.get("preprocess"), Mapping):
            p = data.get("preprocess") or {}
            ct = p.get("channel_typing", {}) if isinstance(p, Mapping) else {}
            if isinstance(ct, Mapping):
                payload = dict(ct)
        elif isinstance(data, Mapping) and isinstance(data.get("channel_typing"), Mapping):
            payload = dict(data.get("channel_typing") or {})
        else:
            payload = dict(data) if isinstance(data, Mapping) else {}

    for k in ("schema", "schema_version", "dataset_id", "description"):
        payload.pop(k, None)
    return payload


# -----------------------------------------------------------------------------
# Dataset-specific channel name helpers
# -----------------------------------------------------------------------------

# TruScan 128 numeric channel labels → 10-5 system labels.
# Derived from `openneuro/config/codebook/truescan_convert_channels.py`.
_TRUESCAN_10_5_MAP: Dict[str, str] = {
    "65": "FPz",
    "1": "FP1",
    "2": "FP2",
    "66": "FPAF1",
    "67": "FPAF2",
    "33": "AF7",
    "34": "AF3",
    "68": "AFz",
    "36": "AF4",
    "38": "AF8",
    "69": "AFF2",
    "70": "AFF1",
    "71": "AFF3",
    "72": "AFF4",
    "3": "F7",
    "41": "F5",
    "4": "F3",
    "43": "F1",
    "5": "Fz",
    "46": "F2",
    "6": "F4",
    "48": "F6",
    "7": "F8",
    "73": "FFT9",
    "74": "FFT7",
    "75": "FFC5",
    "76": "FFC3",
    "77": "FFC1",
    "78": "FFC2",
    "79": "FFC4",
    "80": "FFC6",
    "81": "FFT8",
    "82": "FFT10",
    "50": "FT9",
    "51": "FT7",
    "20": "FC5",
    "35": "FC3",
    "21": "FC1",
    "83": "FCz",
    "22": "FC2",
    "39": "FC4",
    "23": "FC6",
    "40": "FT8",
    "44": "FT10",
    "84": "FTT9",
    "85": "FTT7",
    "86": "FCC5",
    "87": "FCC3",
    "88": "FCC1",
    "89": "FCC2",
    "90": "FCC4",
    "91": "FCC6",
    "92": "FTT8",
    "93": "FTT10",
    "8": "T7",
    "45": "C5",
    "9": "C3",
    "49": "C1",
    "10": "Cz",
    "42": "C2",
    "11": "C4",
    "37": "C6",
    "12": "T8",
    "94": "TTP7",
    "95": "CCP5",
    "96": "CCP3",
    "97": "CCP1",
    "98": "CCP2",
    "99": "CCP4",
    "100": "CCP6",
    "101": "TTP8",
    "47": "TP7",
    "25": "CP5",
    "52": "CP3",
    "26": "CP1",
    "53": "CPz",
    "27": "CP2",
    "54": "CP4",
    "28": "CP6",
    "55": "TP8",
    "102": "TPP9",
    "103": "TPP7",
    "104": "CPP5",
    "105": "CPP3",
    "106": "CPP1",
    "107": "CPP2",
    "108": "CPP4",
    "109": "CPP6",
    "110": "TPP8",
    "111": "TPP10",
    "13": "P7",
    "56": "P5",
    "14": "P3",
    "57": "P1",
    "15": "Pz",
    "58": "P2",
    "16": "P4",
    "59": "P6",
    "17": "P8",
    "112": "P9",
    "119": "P10",
    "24": "TP9",
    "29": "TP10",
    "113": "PPO5",
    "114": "PPO3",
    "115": "PPO1",
    "116": "PPO2",
    "117": "PPO4",
    "118": "PPO6",
    "31": "PO9",
    "60": "PO7",
    "61": "PO3",
    "62": "POz",
    "63": "PO4",
    "64": "PO8",
    "32": "PO10",
    "121": "POO1",
    "122": "POO2",
    "120": "OPO3",
    "123": "OPO4",
    "18": "O1",
    "30": "Oz",
    "19": "O2",
    "124": "OI1",
    "125": "OI2",
    "126": "I1",
    "127": "Iz",
    "128": "I2",
}


def _apply_truescan_128_channel_renaming(raw: "mne.io.BaseRaw", dataset_id: Optional[str]) -> None:
    """Rename TruScan numeric channels to 10-5 labels when configured.

    Some datasets use a TruScan 128 system and often store channel names as numbers
    ("1".."128"). Downstream config (e.g., ROI pairs like F3/P3) assumes 10-20/10-5 labels.

    Special handling:
    - The dataset README states channels 124/125 were placed above/below the eyes for vEOG.
      We therefore:
        * type them as EOG (if present), and
        * rename them to VEOG1/VEOG2 (instead of OI1/OI2).
    """
    # Heuristic: only act when most channels are numeric labels.
    numeric = [ch for ch in raw.ch_names if str(ch).strip().isdigit()]
    if len(numeric) < 32:
        return

    rename_map: Dict[str, str] = {}

    # vEOG channels (dataset-specific note)
    if "124" in raw.ch_names:
        rename_map["124"] = "VEOG1"
    if "125" in raw.ch_names:
        rename_map["125"] = "VEOG2"

    # Apply 10-5 mapping for the rest
    for old in raw.ch_names:
        old_s = str(old).strip()
        if old_s in {"124", "125"}:
            continue
        new = _TRUESCAN_10_5_MAP.get(old_s)
        if new and new != old_s:
            rename_map[old] = new

    if not rename_map:
        return

    # Ensure uniqueness; if collisions exist, skip renaming to avoid hard failure.
    proposed = [rename_map.get(ch, ch) for ch in raw.ch_names]
    if len(set(proposed)) != len(proposed):
        logger.warning(
            "Skipping TruScan channel renaming for %s due to non-unique target names",
            dataset_id,
        )
        return

    # Type EOG channels before renaming so the type sticks with the channel object.
    eog_type_map: Dict[str, str] = {}
    if "124" in raw.ch_names:
        eog_type_map["124"] = "eog"
    if "125" in raw.ch_names:
        eog_type_map["125"] = "eog"
    if eog_type_map:
        raw.set_channel_types(eog_type_map, on_unit_change="ignore")

    raw.rename_channels(rename_map)


def _patch_edf_startdate_if_invalid(edf_path: Path) -> Optional[Path]:
    """Create a patched EDF copy with a valid startdate if the header is malformed.

    Some EDF files contain an invalid start date string (8 bytes) that
    MNE cannot parse (e.g., "..20" patterns). EDF startdate lives at byte offset
    168 (8 bytes) in the fixed header.

    Returns the path to the patched temp EDF, or None if patching was not needed
    or failed.
    """
    try:
        with edf_path.open("rb") as f:
            header = f.read(256)
        if len(header) < 176:
            return None
        startdate = header[168:176].decode("ascii", errors="ignore")
        # EDF spec: "dd.mm.yy"
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", startdate or ""):
            return None
        # Replace with a safe sentinel date (01.01.85 is commonly used in EDF examples)
        patched_date = b"01.01.85"
        if len(patched_date) != 8:
            return None

        # Copy to temp and patch in-place (only for the copy).
        tmp_dir = Path(tempfile.gettempdir()) / "noetic_ingest_edf_patch"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{edf_path.stem}.{uuid4().hex}.edf"

        # Stream copy to avoid loading the whole file into memory twice.
        with edf_path.open("rb") as src, tmp_path.open("wb") as dst:
            # Read and patch fixed header
            fixed = src.read(256)
            if len(fixed) < 256:
                return None
            fixed = fixed[:168] + patched_date + fixed[176:]
            dst.write(fixed)
            # Copy rest
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        logger.warning("Patched EDF invalid startdate for %s -> %s", edf_path.name, tmp_path.name)
        return tmp_path
    except Exception:
        return None



def _apply_channel_typing(raw: "mne.io.BaseRaw", dataset_id: Optional[str], config: Mapping[str, Any]) -> None:
    """Apply channel typing rules from config (if present).

    Config location:
      preprocess.channel_typing.enabled: bool
      preprocess.channel_typing.default_rules: list[rule]
      preprocess.channel_typing.datasets.<dataset_id>.rules: list[rule]

    Each rule supports:
      - regex:  regex string (matched from start; use ^...$ for exact)
      - prefix: string prefix match
      - type:   MNE channel type (eeg/eog/emg/misc/etc)
    Rules are applied in order; first match wins. Dataset-specific rules take
    precedence over default rules so that generic BIDS/OpenNeuro fallbacks do
    not override known dataset quirks.
    """
    if not dataset_id:
        return
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    ct_cfg = preprocess_cfg.get("channel_typing", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(ct_cfg, Mapping) or not ct_cfg.get("enabled", False):
        return
    ds_map = ct_cfg.get("datasets", {}) if isinstance(ct_cfg.get("datasets", {}), Mapping) else {}
    ds_cfg: Dict[str, Any] = dict(ds_map.get(dataset_id, {}) or {}) if isinstance(ds_map, Mapping) else {}
    policy_dir = ct_cfg.get("policy_dir")
    if isinstance(policy_dir, (str, Path)):
        policy_cfg = _load_channel_typing_policy(str(policy_dir), str(dataset_id))
        if policy_cfg:
            ds_cfg = _deep_merge_dict(ds_cfg, policy_cfg)
    default_rules = ct_cfg.get("default_rules", [])
    if not isinstance(default_rules, list):
        default_rules = []
    if not isinstance(ds_cfg, Mapping) and not default_rules:
        return
    ds_rules = ds_cfg.get("rules", []) if isinstance(ds_cfg, Mapping) else []
    if not isinstance(ds_rules, list):
        ds_rules = []
    rules = list(ds_rules) + list(default_rules)
    if not rules:
        return

    compiled: list[tuple[Optional[re.Pattern[str]], Optional[str], str]] = []
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        typ = rule.get("type")
        if not typ:
            continue
        regex = rule.get("regex")
        prefix = rule.get("prefix")
        pat: Optional[re.Pattern[str]] = None
        if regex:
            try:
                pat = re.compile(str(regex))
            except Exception:
                continue
        compiled.append((pat, str(prefix) if prefix else None, str(typ)))

    if not compiled:
        return

    type_map: Dict[str, str] = {}
    for ch in raw.ch_names:
        ch_str = str(ch)
        for pat, prefix, typ in compiled:
            if pat is not None and pat.match(ch_str):
                type_map[ch] = typ
                break
            if prefix is not None and ch_str.startswith(prefix):
                type_map[ch] = typ
                break

    if type_map:
        raw.set_channel_types(type_map, on_unit_change="ignore")


@dataclass
class PreprocessedSignals:
    """Container for preprocessed signals and metadata."""
    signals: Dict[str, np.ndarray]  # per modality
    sfreq: float
    channels: Optional[List[str]]
    meta: Dict[str, Any]


def _infer_dataset_id(file_path: Path, config: Mapping[str, Any]) -> Optional[str]:
    dataset_ids = config.get("datasets") if isinstance(config, Mapping) else None
    path_str = file_path.as_posix()
    if isinstance(dataset_ids, Iterable):
        for ds in dataset_ids:
            ds_str = str(ds)
            token = f"/{ds_str}/"
            if token in path_str:
                return ds_str
    for parent in file_path.parents:
        name = parent.name
        if name.startswith("ds") and name[2:].isdigit():
            return name
    return None


def _resolve_fmri_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve fMRI preprocessing configuration with optional per-dataset overrides."""
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    fmri_cfg = preprocess_cfg.get("fmri", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(fmri_cfg, Mapping):
        return {}
    merged: Dict[str, Any] = {k: fmri_cfg[k] for k in fmri_cfg if k != "datasets"}
    ds_overrides = fmri_cfg.get("datasets", {}) if isinstance(fmri_cfg.get("datasets", {}), Mapping) else {}
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _resolve_event_crop_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    event_cfg = preprocess_cfg.get("event_crop", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(event_cfg, Mapping):
        return {}
    merged: Dict[str, Any] = {k: event_cfg[k] for k in event_cfg if k != "datasets"}
    ds_overrides = event_cfg.get("datasets", {}) if isinstance(event_cfg.get("datasets", {}), Mapping) else {}
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _resolve_crop_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve static crop configuration with optional per-dataset overrides."""
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    crop_cfg = preprocess_cfg.get("crop", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(crop_cfg, Mapping):
        return {}
    merged: Dict[str, Any] = {k: crop_cfg[k] for k in crop_cfg if k != "datasets"}
    ds_overrides = crop_cfg.get("datasets", {}) if isinstance(crop_cfg.get("datasets", {}), Mapping) else {}
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _resolve_artifact_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve artifact configuration with optional per-dataset overrides."""
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    art_cfg = preprocess_cfg.get("artifacts", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(art_cfg, Mapping):
        return {}
    merged: Dict[str, Any] = {k: art_cfg[k] for k in art_cfg if k != "datasets"}
    ds_overrides = art_cfg.get("datasets", {}) if isinstance(art_cfg.get("datasets", {}), Mapping) else {}
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _resolve_eeg_csd_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve EEG CSD/Laplacian config with optional per-dataset overrides."""
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    csd_cfg = preprocess_cfg.get("eeg_csd", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(csd_cfg, Mapping):
        return {}
    merged: Dict[str, Any] = {k: csd_cfg[k] for k in csd_cfg if k != "datasets"}
    ds_overrides = csd_cfg.get("datasets", {}) if isinstance(csd_cfg.get("datasets", {}), Mapping) else {}
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _find_events_file(file_path: Path, cfg: Mapping[str, Any]) -> Optional[Path]:
    explicit = cfg.get("events_path")
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = file_path.parent / candidate
        if candidate.exists():
            return candidate
    suffix_candidates = cfg.get("suffix_candidates")
    if not isinstance(suffix_candidates, Iterable):
        suffix_candidates = ["_events.tsv"]
    base_stem = file_path.stem
    if base_stem.endswith("_eeg"):
        base_core = base_stem[:-4]
    elif base_stem.endswith("_ieeg"):
        base_core = base_stem[:-5]
    else:
        base_core = base_stem
    for suffix in suffix_candidates:
        candidate = file_path.parent / f"{base_core}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _compute_event_crop_window(
    file_path: Path,
    raw: "mne.io.BaseRaw",
    cfg: Mapping[str, Any],
) -> Optional[Tuple[float, Optional[float]]]:
    """Compute an event-aligned crop window (tmin, tmax) if configuration enables it."""
    if not cfg.get("enabled", False):
        return None

    events_file = _find_events_file(file_path, cfg)
    if events_file is None:
        logger.warning("Event crop enabled but no events file found for %s", file_path.name)
        return None

    try:
        events_df = pd.read_csv(events_file, sep="\t")
    except Exception as exc:  # pragma: no cover - runtime dependency
        logger.warning("Failed to read events file %s: %s", events_file, exc)
        return None

    onset_col = str(cfg.get("onset_column", "onset"))
    if onset_col not in events_df.columns:
        logger.warning("Events file %s missing onset column '%s'", events_file, onset_col)
        return None

    onsets = pd.to_numeric(events_df[onset_col], errors="coerce")

    match_col = str(cfg.get("match_column", "trial_type"))
    if match_col in events_df.columns:
        values = events_df[match_col].astype(str).fillna("")
    else:
        values = pd.Series([""] * len(events_df))

    mask = pd.Series(True, index=events_df.index, dtype=bool)

    include_values = cfg.get("include_values")
    if isinstance(include_values, Iterable):
        include_set = {str(v) for v in include_values}
        mask &= values.isin(include_set)

    include_regex = cfg.get("include_regex")
    if include_regex:
        mask &= values.str.contains(include_regex, regex=True, na=False)

    exclude_values = cfg.get("exclude_values")
    if isinstance(exclude_values, Iterable):
        exclude_set = {str(v) for v in exclude_values}
        mask &= ~values.isin(exclude_set)

    exclude_regex = cfg.get("exclude_regex")
    if exclude_regex:
        mask &= ~values.str.contains(exclude_regex, regex=True, na=False)

    filtered_onsets = onsets[mask].dropna()
    if filtered_onsets.empty:
        logger.warning("Event crop selection empty for %s (events file %s)", file_path.name, events_file.name)
        return None

    duration_col = cfg.get("duration_column", "duration")
    if duration_col in events_df.columns:
        durations = pd.to_numeric(events_df.loc[filtered_onsets.index, duration_col], errors="coerce").fillna(0.0)
    else:
        durations = pd.Series(0.0, index=filtered_onsets.index)

    start_offset = float(cfg.get("start_offset", 0.0) or 0.0)
    stop_offset = float(cfg.get("stop_offset", 0.0) or 0.0)
    tmin = max(0.0, float(filtered_onsets.min()) + start_offset)

    end_times = (filtered_onsets + durations).fillna(filtered_onsets)
    if end_times.empty or not np.isfinite(end_times.max()):
        tmax_candidate = float(filtered_onsets.max()) + stop_offset
    else:
        tmax_candidate = float(end_times.max()) + stop_offset

    raw_duration = float(raw.n_times) / float(raw.info["sfreq"])
    tmax = min(raw_duration, tmax_candidate) if np.isfinite(tmax_candidate) else raw_duration

    if tmax <= tmin:
        min_duration = float(cfg.get("min_duration", 0.0) or 0.0)
        if min_duration > 0:
            tmax = min(raw_duration, tmin + min_duration)

    if tmax <= tmin:
        logger.warning("Event crop produced invalid window for %s (tmin=%.2f, tmax=%.2f)", file_path.name, tmin, tmax)
        return None

    return tmin, tmax


def _detect_bad_eeg_channels(raw: "mne.io.BaseRaw", cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Detect obviously bad EEG channels using simple variance and correlation heuristics.

    Returns a dict with keys:
    - bad_channels: list of channel names
    - reasons: mapping ch_name -> reason dict (var, flat, high_var, low_corr, corr)
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    n_ch = len(eeg_picks)
    if n_ch == 0:
        return {"bad_channels": [], "reasons": {}}

    data = raw.get_data(picks=eeg_picks)
    if data.size == 0:
        return {"bad_channels": [], "reasons": {}}

    var = np.var(data, axis=1)
    finite_mask = np.isfinite(var)
    if not finite_mask.any():
        return {"bad_channels": [], "reasons": {}}

    positive_var = var[finite_mask & (var > 0)]
    if positive_var.size == 0:
        median_var = float(np.median(var[finite_mask]))
    else:
        median_var = float(np.median(positive_var))
    if not np.isfinite(median_var) or median_var <= 0:
        return {"bad_channels": [], "reasons": {}}

    # Thresholds (can be overridden under config['robustness']['bad_channels'])
    bad_cfg = cfg if isinstance(cfg, Mapping) else {}
    var_low_factor = float(bad_cfg.get("var_low_factor", 1e-4)) or 1e-4
    var_high_factor = float(bad_cfg.get("var_high_factor", 25.0)) or 25.0
    corr_thresh = float(bad_cfg.get("corr_thresh", 0.2)) or 0.2
    max_bad_fraction = float(bad_cfg.get("max_bad_fraction", 0.3)) or 0.3
    min_good_channels = int(bad_cfg.get("min_good_channels", 8)) or 8

    flat_thresh = max(1e-12, var_low_factor * median_var)
    high_thresh = var_high_factor * median_var

    flat_mask = var <= flat_thresh
    high_mask = var >= high_thresh

    # Correlation with global mean (very conservative threshold)
    global_mean = data.mean(axis=0)
    global_mean = global_mean - global_mean.mean()
    denom_global = np.linalg.norm(global_mean)
    corr = np.zeros(n_ch, dtype=float)
    if denom_global > 0:
        for idx in range(n_ch):
            ch_data = data[idx] - data[idx].mean()
            denom = np.linalg.norm(ch_data) * denom_global
            if denom == 0:
                corr[idx] = 0.0
            else:
                corr[idx] = float(np.dot(ch_data, global_mean) / denom)
    low_corr_mask = corr < corr_thresh

    # Always treat obvious non-EEG channels as bad by name (coordinates, EOG labels)
    eeg_names = [raw.ch_names[i] for i in eeg_picks]
    forced_bad_names = {"Z", "Y", "X", "VEOG", "HEOG", "HEO", "VEO"}
    forced_bad_mask = np.array([ch.upper() in forced_bad_names for ch in eeg_names], dtype=bool)

    # Final bad mask:
    # - Always drop flat or extremely high-variance channels
    # - Always drop forced_bad_names
    # - Only use low_corr as an additional hint when a channel is already flat/high_var
    bad_mask = flat_mask | high_mask | forced_bad_mask | (low_corr_mask & (flat_mask | high_mask))
    bad_indices = np.where(bad_mask)[0].tolist()

    # Limit how many channels we drop
    max_bad = int(max_bad_fraction * n_ch)
    if max_bad <= 0:
        max_bad = 1
    if len(bad_indices) > max_bad:
        severities = []
        for idx in bad_indices:
            score = 0.0
            if flat_mask[idx]:
                score += 2.0
            if high_mask[idx]:
                score += 1.0
            # low_corr contributes only mildly, and only when combined via bad_mask above
            if low_corr_mask[idx]:
                score += 0.5
            if forced_bad_mask[idx]:
                score += 3.0
            severities.append((score, idx))
        severities.sort(reverse=True)
        bad_indices = [idx for _, idx in severities[:max_bad]]

    # Ensure we retain at least a minimal number of channels
    if n_ch - len(bad_indices) < min_good_channels:
        # Fallback: only drop completely flat channels
        bad_indices = np.where(flat_mask)[0].tolist()

    bad_channels = [raw.ch_names[eeg_picks[idx]] for idx in bad_indices]
    reasons: Dict[str, Any] = {}
    for idx in bad_indices:
        ch_name = raw.ch_names[eeg_picks[idx]]
        reasons[ch_name] = {
            "var": float(var[idx]),
            "flat": bool(flat_mask[idx]),
            "high_var": bool(high_mask[idx]),
            "low_corr": bool(low_corr_mask[idx]),
            "corr": float(corr[idx]),
        }

    return {"bad_channels": bad_channels, "reasons": reasons}


def _read_text_guess_encoding(path: Path) -> str:
    """Read a small text file with a best-effort encoding strategy."""
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("latin-1", errors="replace")


def _brainvision_common_infos_bounds(lines: List[str]) -> Optional[Tuple[int, int]]:
    """Return (start_idx, end_idx) for the [Common Infos] section (end exclusive)."""
    start = None
    header_re = re.compile(r"^\s*\[(?P<name>.+?)\]\s*$")
    for i, line in enumerate(lines):
        if line.strip().lower() == "[common infos]":
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if header_re.match(lines[j]):
            return start, j
    return start, len(lines)


def _brainvision_get_key(lines: List[str], start: int, end: int, key: str) -> Optional[str]:
    key_l = key.strip().lower()
    kv_re = re.compile(r"^\s*(?P<k>[^=;]+?)\s*=\s*(?P<v>.*)\s*$")
    for i in range(start + 1, end):
        m = kv_re.match(lines[i])
        if not m:
            continue
        if m.group("k").strip().lower() == key_l:
            return m.group("v").strip()
    return None


def _brainvision_set_key(lines: List[str], start: int, end: int, key: str, value: str) -> None:
    """Set or insert key=value within [Common Infos]. Mutates lines in-place."""
    key_l = key.strip().lower()
    kv_re = re.compile(r"^\s*(?P<k>[^=;]+?)\s*=\s*(?P<v>.*)\s*$")
    for i in range(start + 1, end):
        m = kv_re.match(lines[i])
        if not m:
            continue
        if m.group("k").strip().lower() == key_l:
            # Preserve any original key formatting as much as possible.
            orig_key = m.group("k").strip()
            lines[i] = f"{orig_key}={value}"
            return
    # Insert near the end of the section (before the next section header).
    lines.insert(end, f"{key}={value}")


def _write_minimal_vmrk(marker_path: Path, data_file_abs: Path) -> None:
    """Write a minimal BrainVision .vmrk so MNE can load a dataset without events."""
    # Use POSIX separators to avoid escaping issues; Windows accepts these.
    data_str = data_file_abs.as_posix()
    content = "\n".join(
        [
            "Brain Vision Data Exchange Marker File, Version 1.0",
            "",
            "[Common Infos]",
            f"DataFile={data_str}",
            "",
            "[Marker Infos]",
            "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position>,<Size>,<Channel>,<Date>",
            "Mk1=New Segment,,1,1,0,0",
            "",
        ]
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(content, encoding="utf-8")


def _prepare_brainvision_vhdr_for_mne(vhdr_path: Path) -> Tuple[Path, List[Path]]:
    """If needed, create temp BrainVision header/marker so MNE can read it.

    Some OpenNeuro BrainVision headers omit MarkerFile= in [Common Infos].
    MNE treats this as an error (configparser.NoOptionError). For robustness we
    create a minimal .vmrk and a patched .vhdr that point to the real data file
    via absolute paths. We keep the originals untouched and clean up temps after.
    """
    created: List[Path] = []
    if vhdr_path.suffix.lower() != ".vhdr":
        return vhdr_path, created

    try:
        text = _read_text_guess_encoding(vhdr_path)
    except Exception:
        return vhdr_path, created

    lines = text.splitlines()
    bounds = _brainvision_common_infos_bounds(lines)
    if bounds is None:
        return vhdr_path, created
    start, end = bounds

    marker_val = _brainvision_get_key(lines, start, end, "MarkerFile")
    if marker_val:
        return vhdr_path, created

    data_val = _brainvision_get_key(lines, start, end, "DataFile")
    data_path = (vhdr_path.parent / data_val).resolve() if data_val else vhdr_path.with_suffix(".eeg").resolve()
    if not data_path.exists():
        logger.warning("BrainVision header %s missing MarkerFile and DataFile could not be resolved; leaving as-is", vhdr_path)
        return vhdr_path, created

    tmp_dir = Path(tempfile.gettempdir()) / "noetic_ingest_brainvision"
    uid = uuid4().hex
    tmp_vmrk = tmp_dir / f"{vhdr_path.stem}.{uid}.vmrk"
    tmp_vhdr = tmp_dir / f"{vhdr_path.stem}.{uid}.vhdr"

    try:
        _write_minimal_vmrk(tmp_vmrk, data_path)
        # Patch header to absolute paths so it can live in temp dir.
        _brainvision_set_key(lines, start, end, "DataFile", data_path.as_posix())
        # end may have shifted if DataFile was inserted above
        bounds2 = _brainvision_common_infos_bounds(lines)
        if bounds2 is None:
            return vhdr_path, created
        start2, end2 = bounds2
        _brainvision_set_key(lines, start2, end2, "MarkerFile", tmp_vmrk.as_posix())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_vhdr.write_text("\n".join(lines) + "\n", encoding="utf-8")
        created.extend([tmp_vhdr, tmp_vmrk])
        logger.warning("Repaired BrainVision header missing MarkerFile for %s using temporary files", vhdr_path.name)
        return tmp_vhdr, created
    except Exception as exc:
        logger.warning("Failed to repair BrainVision header %s: %s; leaving as-is", vhdr_path.name, exc)
        # Best-effort cleanup if we partially created temps.
        for p in (tmp_vhdr, tmp_vmrk):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        return vhdr_path, []


def _load_epoched_eeglab_as_raw(set_path: Path) -> Tuple["mne.io.BaseRaw", Dict[str, Any]]:
    """Load epoched EEGLAB .set and flatten epochs into a continuous RawArray."""
    if mne is None:  # pragma: no cover
        raise RuntimeError("mne is required for EEGLAB preprocessing")

    epochs = mne.read_epochs_eeglab(set_path, verbose=False)
    data = epochs.get_data(copy=True)
    if data.ndim != 3:
        raise RuntimeError(f"Unexpected EEGLAB epochs shape: {data.shape}")

    n_epochs, n_channels, n_times = data.shape
    # [E, C, T] -> [C, E*T]
    continuous = np.transpose(data, (1, 0, 2)).reshape(n_channels, n_epochs * n_times)
    ch_types = epochs.get_channel_types()
    info = mne.create_info(ch_names=epochs.ch_names, sfreq=float(epochs.info["sfreq"]), ch_types=ch_types)
    raw = mne.io.RawArray(continuous, info, verbose=False)
    meta = {
        "source": "eeglab_epoched_concat",
        "n_epochs": int(n_epochs),
        "epoch_samples": int(n_times),
        "original_shape": [int(n_epochs), int(n_channels), int(n_times)],
    }
    return raw, meta


def _resolve_eeglab_concat_policy(
    eeglab_cfg: Mapping[str, Any],
    dataset_id: Optional[str],
) -> Dict[str, bool]:
    """Resolve filtering/resampling policy for concatenated epoched EEGLAB streams."""
    policy = {
        "resample_concatenated_epochs": False,
        "filter_concatenated_epochs": False,
    }
    if not isinstance(eeglab_cfg, Mapping):
        return policy
    for key in policy.keys():
        if key in eeglab_cfg:
            policy[key] = bool(eeglab_cfg.get(key))
    ds_map = eeglab_cfg.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            for key in policy.keys():
                if key in ds_cfg:
                    policy[key] = bool(ds_cfg.get(key))
    return policy


def _resolve_dataset_preprocess_policy(
    preprocess_cfg: Mapping[str, Any],
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Resolve dataset-scoped preprocess policy switches."""
    policy: Dict[str, Any] = {
        "patch_invalid_edf_startdate": False,
        "channel_rename": None,
    }
    if not isinstance(preprocess_cfg, Mapping):
        return policy
    ds_map = preprocess_cfg.get("datasets", {})
    if not (dataset_id and isinstance(ds_map, Mapping)):
        return policy
    ds_cfg = ds_map.get(dataset_id)
    if not isinstance(ds_cfg, Mapping):
        return policy
    if "patch_invalid_edf_startdate" in ds_cfg:
        policy["patch_invalid_edf_startdate"] = bool(ds_cfg.get("patch_invalid_edf_startdate"))
    if "channel_rename" in ds_cfg:
        val = ds_cfg.get("channel_rename")
        policy["channel_rename"] = str(val) if val is not None else None
    return policy


def _resolve_target_sfreq(
    original_sfreq: float,
    preprocess_cfg: Mapping[str, Any],
) -> tuple[float, Dict[str, Any]]:
    """Resolve per-file target sfreq with optional integer-ratio candidates.

    Policy:
    - Prefer candidates that yield (approximately) integer downsampling ratios.
    - Among integer-ratio candidates, prefer the one closest to the configured
      base sfreq.
    - Fallback to the base sfreq when no candidate matches.
    """
    try:
        base_sfreq = float(preprocess_cfg.get("sfreq", 250.0))
    except Exception:
        base_sfreq = 250.0
    candidates_raw = preprocess_cfg.get("sfreq_candidates", [])
    candidates: List[float] = []
    if isinstance(candidates_raw, (list, tuple)):
        for c in candidates_raw:
            try:
                f = float(c)
                if np.isfinite(f) and f > 0:
                    candidates.append(f)
            except Exception:
                continue
    if not candidates:
        return base_sfreq, {
            "base_sfreq": base_sfreq,
            "candidates": [base_sfreq],
            "selected_sfreq": base_sfreq,
            "selection_mode": "base_only",
            "integer_ratio": bool(abs((original_sfreq / base_sfreq) - round(original_sfreq / base_sfreq)) <= 1e-9)
            if base_sfreq > 0
            else False,
            "ratio": (original_sfreq / base_sfreq) if base_sfreq > 0 else np.nan,
        }

    # Ensure base sfreq is considered as fallback candidate.
    all_candidates = sorted(set([base_sfreq] + candidates))
    integer_matches: List[tuple[float, float]] = []
    for cand in all_candidates:
        if cand <= 0:
            continue
        ratio = original_sfreq / cand
        # Only consider downsampling integer ratios >= 1.
        if ratio >= 1.0 and abs(ratio - round(ratio)) <= 1e-9:
            integer_matches.append((cand, ratio))

    if integer_matches:
        # Closest to base sfreq wins; tie-breaker picks the smaller ratio (less aggressive decimation).
        integer_matches.sort(key=lambda t: (abs(t[0] - base_sfreq), t[1]))
        chosen, ratio = integer_matches[0]
        return float(chosen), {
            "base_sfreq": base_sfreq,
            "candidates": all_candidates,
            "selected_sfreq": float(chosen),
            "selection_mode": "integer_ratio_candidate",
            "integer_ratio": True,
            "ratio": float(ratio),
        }

    return base_sfreq, {
        "base_sfreq": base_sfreq,
        "candidates": all_candidates,
        "selected_sfreq": base_sfreq,
        "selection_mode": "fallback_base",
        "integer_ratio": bool(abs((original_sfreq / base_sfreq) - round(original_sfreq / base_sfreq)) <= 1e-9)
        if base_sfreq > 0
        else False,
        "ratio": (original_sfreq / base_sfreq) if base_sfreq > 0 else np.nan,
    }


def preprocess_file(file_path: Path, config: Mapping[str, Any]) -> PreprocessedSignals:
    """Preprocess a single file (EEG or fMRI) and return signals + metadata.

    Parameters
    ----------
    file_path
        Path to raw EEG or fMRI file.
    config
        Configuration dict with preprocessing settings.

    Returns
    -------
    PreprocessedSignals
        Per-modality arrays, sfreq, channels, and metadata.
    """
    if mne is None:
        raise RuntimeError("mne is required for EEG preprocessing but is not installed.")
    suffixes = "".join(file_path.suffixes).lower()
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        return preprocess_fmri(file_path, config)
    # We can infer dataset id from the path (no file I/O) and use it for
    # dataset-specific loading quirks before MNE touches the file.
    dataset_id_hint = _infer_dataset_id(file_path, config)
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, dict) else {}
    target_sfreq = preprocess_cfg.get("sfreq", 250) if isinstance(preprocess_cfg, Mapping) else 250
    notch_hz = preprocess_cfg.get("notch_hz", None) if isinstance(preprocess_cfg, Mapping) else None
    eeg_bandpass = preprocess_cfg.get("eeg_bandpass", [1, 45]) if isinstance(preprocess_cfg, Mapping) else [1, 45]
    reref = preprocess_cfg.get("reref", "average") if isinstance(preprocess_cfg, Mapping) else "average"
    gpu_flag = preprocess_cfg.get("gpu", False) if isinstance(preprocess_cfg, Mapping) else False
    eeglab_cfg = preprocess_cfg.get("eeglab", {}) if isinstance(preprocess_cfg, Mapping) else {}
    dataset_preprocess_policy = _resolve_dataset_preprocess_policy(
        preprocess_cfg if isinstance(preprocess_cfg, Mapping) else {},
        dataset_id_hint,
    )
    allow_epoched_set = bool(eeglab_cfg.get("allow_epoched_set", False)) if isinstance(eeglab_cfg, Mapping) else False
    eeglab_concat_policy = _resolve_eeglab_concat_policy(eeglab_cfg if isinstance(eeglab_cfg, Mapping) else {}, dataset_id_hint)
    if isinstance(eeglab_cfg, Mapping):
        ds_map = eeglab_cfg.get("datasets", {})
        if isinstance(ds_map, Mapping):
            path_posix = file_path.as_posix()
            for ds_key, ds_cfg in ds_map.items():
                token = f"/{str(ds_key)}/"
                if token in path_posix and isinstance(ds_cfg, Mapping) and "allow_epoched_set" in ds_cfg:
                    allow_epoched_set = bool(ds_cfg.get("allow_epoched_set", allow_epoched_set))

    if gpu_flag:
        logger.info("GPU acceleration for preprocessing is not currently supported; proceeding on CPU")

    t_pre0 = time.perf_counter()
    preprocess_timings: Dict[str, float] = {}

    tmp_paths: List[Path] = []
    cleanup_after_load: List[Path] = []
    file_path_for_mne = file_path
    eeglab_import_meta: Optional[Dict[str, Any]] = None
    if suffixes.endswith(".vhdr"):
        file_path_for_mne, tmp_paths = _prepare_brainvision_vhdr_for_mne(file_path)
    elif suffixes.endswith(".edf") and bool(dataset_preprocess_policy.get("patch_invalid_edf_startdate", False)):
        # Dataset-configured: patch malformed EDF startdate pre-emptively.
        patched = _patch_edf_startdate_if_invalid(Path(file_path_for_mne))
        if patched is not None:
            cleanup_after_load.append(patched)
            file_path_for_mne = patched

    # Load raw with delayed data load to avoid reading entire file
    raw = None
    t_load0 = time.perf_counter()
    try:
        try:
            raw = mne.io.read_raw(file_path_for_mne, preload=False, verbose=False)
        except TypeError as exc:
            # EEGLAB .set can store epoched data. If enabled, flatten epochs to a
            # continuous surrogate stream for feature extraction.
            if (
                suffixes.endswith(".set")
                and allow_epoched_set
                and "number of trials" in str(exc).lower()
            ):
                raw, eeglab_import_meta = _load_epoched_eeglab_as_raw(Path(file_path_for_mne))
                logger.warning(
                    "Loaded epoched EEGLAB as concatenated continuous stream for %s",
                    file_path.name,
                )
            else:
                raise
        except ValueError as exc:
            # Dataset-configured EDF patch fallback on parse failures.
            dataset_id_for_retry = _infer_dataset_id(file_path, config)
            retry_policy = _resolve_dataset_preprocess_policy(
                preprocess_cfg if isinstance(preprocess_cfg, Mapping) else {},
                dataset_id_for_retry,
            )
            if suffixes.endswith(".edf") and bool(retry_policy.get("patch_invalid_edf_startdate", False)):
                patched = _patch_edf_startdate_if_invalid(Path(file_path_for_mne))
                if patched is not None:
                    cleanup_after_load.append(patched)
                    raw = mne.io.read_raw(patched, preload=False, verbose=False)
                else:
                    raise
            else:
                raise
    finally:
        # BrainVision temp headers/markers are safe to remove immediately:
        # MNE reads header paths and uses the referenced data file directly.
        for p in tmp_paths:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                # Best-effort cleanup; temp dir is shared across workers.
                pass

    assert raw is not None
    original_sfreq = float(raw.info["sfreq"])
    target_sfreq, sfreq_policy = _resolve_target_sfreq(
        original_sfreq=original_sfreq,
        preprocess_cfg=preprocess_cfg if isinstance(preprocess_cfg, Mapping) else {},
    )
    logger.info(
        "Resolved target sfreq for %s: %.3f Hz (base=%.3f, mode=%s, ratio=%s)",
        file_path.name,
        float(target_sfreq),
        float(sfreq_policy.get("base_sfreq", target_sfreq)),
        str(sfreq_policy.get("selection_mode", "unknown")),
        str(sfreq_policy.get("ratio", "n/a")),
    )

    # Determine crop window (event-based preferred, falling back to static crop)
    dataset_id = _infer_dataset_id(file_path, config)
    event_crop_cfg = _resolve_event_crop_config(config, dataset_id)
    crop_window: Optional[Tuple[float, Optional[float]]] = None
    if event_crop_cfg.get("enabled"):
        crop_window = _compute_event_crop_window(file_path, raw, event_crop_cfg)
        if crop_window:
            logger.info(
                "Event crop for %s (%s): %.2fs - %.2fs",
                file_path.name,
                dataset_id or "unknown",
                crop_window[0],
                crop_window[1] if crop_window[1] is not None else (float(raw.n_times) / float(raw.info["sfreq"])),
            )
    crop_window_info: Optional[Tuple[float, Optional[float]]] = crop_window
    if crop_window is None:
        crop_cfg = _resolve_crop_config(config, dataset_id)
        if crop_cfg:
            tmin = float(crop_cfg.get("start_sec", 0) or 0.0)
            tmax_val = crop_cfg.get("stop_sec", None)
            tmax = float(tmax_val) if tmax_val is not None else None
            crop_window = (tmin, tmax)

    if crop_window is not None:
        tmin, tmax = crop_window
        if tmax is not None and tmax <= tmin:
            logger.warning("Invalid crop window for %s (tmin=%.2f, tmax=%.2f); skipping crop", file_path.name, tmin, tmax)
        else:
            # Clamp tmax to the available data duration with a small epsilon to avoid boundary errors
            raw_duration = float(raw.times[-1])  # last sample time
            eps = max(1.0 / (float(raw.info["sfreq"]) * 2.0), 1e-6)
            tmin_clamped = max(tmin, 0.0)
            tmax_target = raw_duration if tmax is None else tmax
            tmax_clamped = min(raw_duration - eps, tmax_target)
            if tmax_clamped <= tmin_clamped:
                logger.warning(
                    "Clamped crop window invalid for %s (tmin=%.2f, tmax=%.2f); skipping crop",
                    file_path.name,
                    tmin_clamped,
                    tmax_clamped,
                )
            else:
                raw.crop(tmin=tmin_clamped, tmax=tmax_clamped)

    # Load only the cropped data
    raw.load_data(verbose=False)
    preprocess_timings["load_and_crop"] = float(time.perf_counter() - t_load0)

    # Dataset-specific channel renaming (must happen early so later feature extraction
    # can rely on canonical channel labels).
    t_channel_prep0 = time.perf_counter()
    rename_policy = str(dataset_preprocess_policy.get("channel_rename", "") or "").strip().lower()
    try:
        if rename_policy in {"truescan_128_numeric_to_10_5", "truescan_10_5"}:
            _apply_truescan_128_channel_renaming(raw, dataset_id)
        elif rename_policy:
            logger.warning(
                "Unknown preprocess.datasets.%s.channel_rename policy '%s'; skipping channel rename",
                dataset_id,
                rename_policy,
            )
    except Exception:
        logger.exception("Failed to apply configured channel renaming policy; continuing with original channel names")

    # Apply optional dataset-specific channel typing rules from config.
    try:
        _apply_channel_typing(raw, dataset_id, config)
    except Exception:
        logger.exception("Failed to apply channel typing rules for %s; continuing with MNE defaults", dataset_id)
    preprocess_timings["channel_prep"] = float(time.perf_counter() - t_channel_prep0)

    # Keep only biologically relevant channels before expensive resampling.
    # This reduces compute/memory without changing downstream feature semantics.
    t_pick0 = time.perf_counter()
    try:
        keep_picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True, seeg=True, ecog=True)
        n_total = len(raw.ch_names)
        n_keep = int(len(keep_picks))
        if n_keep > 0 and n_keep < n_total:
            raw.pick(keep_picks)
            logger.info(
                "Pre-resample channel prune for %s: keeping %d/%d channels",
                file_path.name,
                n_keep,
                n_total,
            )
    except Exception as exc:
        logger.warning("Pre-resample channel prune failed for %s (%s); continuing with all channels", file_path.name, exc)
    preprocess_timings["pre_resample_channel_prune"] = float(time.perf_counter() - t_pick0)

    # Resample immediately after channel pruning to reduce downstream cost.
    t_resample_early0 = time.perf_counter()
    is_eeglab_epoched_concat = bool(
        isinstance(eeglab_import_meta, Mapping) and eeglab_import_meta.get("source") == "eeglab_epoched_concat"
    )
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    resample_cfg = preprocess_cfg.get("resample", {}) if isinstance(preprocess_cfg, Mapping) else {}
    resample_n_jobs = resample_cfg.get("n_jobs", 1) if isinstance(resample_cfg, Mapping) else 1
    if isinstance(resample_n_jobs, str) and resample_n_jobs.strip().lower() == "auto":
        n_jobs_resample: Any = "auto"
    else:
        try:
            n_jobs_resample = max(1, int(resample_n_jobs))
        except Exception:
            n_jobs_resample = 1
    if is_eeglab_epoched_concat and not eeglab_concat_policy.get("resample_concatenated_epochs", False):
        logger.warning(
            "Skipping resample for concatenated epoched EEGLAB stream (%s) to avoid boundary artifacts",
            file_path.name,
        )
    elif raw.info["sfreq"] != target_sfreq:
        logger.info(
            "Early resample from %.1f Hz to %.1f Hz for %s (n_jobs=%s)",
            raw.info["sfreq"],
            target_sfreq,
            file_path.name,
            n_jobs_resample,
        )
        raw.resample(target_sfreq, n_jobs=n_jobs_resample, verbose=False)
    preprocess_timings["resample_early"] = float(time.perf_counter() - t_resample_early0)

    # If we patched an EDF copy, it is now safe to remove it (raw is in-memory).
    for p in cleanup_after_load:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # Optional artifact reduction (with per-dataset overrides)
    artifact_cfg = _resolve_artifact_config(config, dataset_id)
    art_cfg = artifact_cfg or {}
    art_method = str(art_cfg.get("method", "none")).lower()

    def _apply_eog_regression(r: mne.io.BaseRaw) -> None:
        eeg_picks = mne.pick_types(r.info, eeg=True)
        eog_picks = mne.pick_types(r.info, eog=True)
        if len(eeg_picks) == 0 or len(eog_picks) == 0:
            logger.info("EOG regression skipped: missing EEG or EOG channels")
            return
        data = r.get_data()  # shape: [n_channels, n_times]
        eeg_data = data[eeg_picks, :]
        eog_data = data[eog_picks, :].T  # [T, K]
        # Add intercept
        ones = np.ones((eog_data.shape[0], 1), dtype=eog_data.dtype)
        design = np.hstack([eog_data, ones])  # [T, K+1]
        try:
            # Solve per EEG channel: minimize ||design * beta - y||
            beta, *_ = np.linalg.lstsq(design, eeg_data.T, rcond=None)  # [K+1, C]
            fitted = design @ beta  # [T, C]
            corrected = (eeg_data.T - fitted).T
            # Write back
            r._data[eeg_picks, :] = corrected  # type: ignore[attr-defined]
            logger.info("Applied EOG regression to %d EEG channels using %d EOG predictors", len(eeg_picks), len(eog_picks))
        except Exception as exc:
            logger.warning("EOG regression failed (%s); continuing without regression", exc)

    def _apply_ica(r: mne.io.BaseRaw) -> None:
        if ICA is None:
            logger.info("ICA not available; skipping ICA artifacts removal")
            return
        eeg_picks = mne.pick_types(r.info, eeg=True)
        if len(eeg_picks) == 0:
            logger.info("ICA skipped: no EEG channels")
            return
        n_comp = int(art_cfg.get("ica_n_components", min(20, len(eeg_picks))))
        random_state = int(art_cfg.get("ica_random_state", 97))
        method = str(art_cfg.get("ica_method", "fastica"))
        fit_highpass_hz = float(art_cfg.get("ica_fit_highpass_hz", 1.0) or 1.0)
        try:
            ica = ICA(n_components=n_comp, method=method, random_state=random_state, max_iter="auto")
            raw_fit = r.copy()
            if fit_highpass_hz > 0:
                raw_fit.filter(l_freq=fit_highpass_hz, h_freq=None, picks=eeg_picks, verbose=False)
            ica.fit(raw_fit, picks=eeg_picks, verbose=False)
            # Try to find EOG-related components if EOG channels exist
            eog_picks = mne.pick_types(r.info, eog=True)
            exclude_idx: List[int] = []
            if len(eog_picks) > 0:
                for eog_idx in eog_picks:
                    eog_ch_name = raw_fit.ch_names[eog_idx]
                    this_idx, _ = ica.find_bads_eog(raw_fit, ch_name=eog_ch_name)
                    exclude_idx.extend(this_idx)
            ica.exclude = list(sorted(set(exclude_idx)))
            ica.apply(r)
            logger.info("Applied ICA (n_components=%d); excluded %d components", n_comp, len(ica.exclude))
        except Exception as exc:
            logger.warning("ICA artifact removal failed (%s); continuing without ICA", exc)

    t_artifact0 = time.perf_counter()
    if art_method == "eog_reg":
        _apply_eog_regression(raw)
    elif art_method == "ica":
        _apply_ica(raw)
    elif art_method not in {"", "none", "null"}:
        logger.info("Unknown artifact method '%s'; skipping artifact reduction", art_method)
    preprocess_timings["artifact_reduction"] = float(time.perf_counter() - t_artifact0)

    # For concatenated epoched EEGLAB streams, default is to skip time-domain filters
    # to avoid ringing at synthetic epoch boundaries.
    skip_time_filters = is_eeglab_epoched_concat and not eeglab_concat_policy.get("filter_concatenated_epochs", False)

    # Apply notch filter globally if configured (in-place)
    t_notch0 = time.perf_counter()
    if notch_hz is not None and not skip_time_filters:
        raw.notch_filter(freqs=notch_hz, verbose=False)
    preprocess_timings["notch"] = float(time.perf_counter() - t_notch0)

    # Treat intracranial EEG (sEEG/ECoG) as EEG for ingest purposes.
    # Many BIDS iEEG datasets (e.g. ds004100) label channels as "seeg"/"ecog",
    # which would otherwise result in empty EEG feature extraction.
    eeg_like_chans = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)

    # Apply EEG bandpass on EEG-like picks only
    t_bandpass0 = time.perf_counter()
    eeg_chans = eeg_like_chans
    if len(eeg_chans) > 0 and eeg_bandpass is not None and len(eeg_bandpass) == 2 and not skip_time_filters:
        try:
            raw.filter(l_freq=eeg_bandpass[0], h_freq=eeg_bandpass[1], picks=eeg_chans, verbose=False)
        except Exception:
            # Fallback to mne.filter.filter_data on array if raw.filter fails
            data = raw.get_data(picks=eeg_chans)
            filtered = mne.filter.filter_data(data, raw.info["sfreq"], eeg_bandpass[0], eeg_bandpass[1], verbose=False)
            raw._data[eeg_chans, :] = filtered  # type: ignore[attr-defined]
    preprocess_timings["bandpass"] = float(time.perf_counter() - t_bandpass0)

    # Resample once for all channels if needed
    # (Note: we already resampled early to save memory, so this is now usually a no-op).
    t_resample_late0 = time.perf_counter()
    if raw.info["sfreq"] != target_sfreq and not (is_eeglab_epoched_concat and not eeglab_concat_policy.get("resample_concatenated_epochs", False)):
        raw.resample(target_sfreq, verbose=False)
    preprocess_timings["resample_late"] = float(time.perf_counter() - t_resample_late0)

    # Detect and drop obviously bad EEG-like channels (simple heuristics) BEFORE CAR reref.
    bad_channel_cfg = (config.get("robustness", {}).get("bad_channels", {}) if isinstance(config, dict) else {}) or {}
    t_badch0 = time.perf_counter()
    bad_info = _detect_bad_eeg_channels(raw, bad_channel_cfg)
    bad_eeg_channels = bad_info.get("bad_channels", []) or []
    if bad_eeg_channels:
        try:
            raw.drop_channels(bad_eeg_channels)
            logger.info(
                "Dropped %d bad EEG channels for %s: %s",
                len(bad_eeg_channels),
                file_path.name,
                ", ".join(bad_eeg_channels),
            )
        except Exception as exc:
            logger.warning("Failed to drop bad EEG channels for %s: %s", file_path.name, exc)
    preprocess_timings["bad_channel_detection_drop"] = float(time.perf_counter() - t_badch0)

    # Optional scalp-EEG CSD / Surface Laplacian step to suppress broad
    # volume-conducted fields before downstream feature extraction.
    t_csd0 = time.perf_counter()
    eeg_csd_cfg = _resolve_eeg_csd_config(config, dataset_id)
    csd_applied = False
    csd_reason: Optional[str] = None
    if bool(eeg_csd_cfg.get("enabled", False)):
        if compute_current_source_density is None:
            csd_reason = "mne_csd_unavailable"
            logger.warning("EEG CSD requested but mne.compute_current_source_density is unavailable")
        else:
            eeg_scalp_picks = mne.pick_types(raw.info, eeg=True, seeg=False, ecog=False)
            min_eeg_channels = int(eeg_csd_cfg.get("min_eeg_channels", 16) or 16)
            if len(eeg_scalp_picks) < min_eeg_channels:
                csd_reason = f"insufficient_scalp_channels:{len(eeg_scalp_picks)}<{min_eeg_channels}"
                logger.warning(
                    "EEG CSD skipped for %s: scalp EEG channels %d < min_eeg_channels %d",
                    file_path.name,
                    len(eeg_scalp_picks),
                    min_eeg_channels,
                )
            else:
                lambda2 = float(eeg_csd_cfg.get("lambda2", 1e-5) or 1e-5)
                stiffness = float(eeg_csd_cfg.get("stiffness", 4.0) or 4.0)
                n_legendre_terms = int(eeg_csd_cfg.get("n_legendre_terms", 50) or 50)
                try:
                    raw = compute_current_source_density(
                        raw,
                        lambda2=lambda2,
                        stiffness=stiffness,
                        n_legendre_terms=n_legendre_terms,
                        copy=False,
                    )
                    csd_applied = True
                    csd_reason = "applied"
                    logger.info(
                        "Applied EEG CSD for %s (lambda2=%g, stiffness=%.2f, n_legendre_terms=%d)",
                        file_path.name,
                        lambda2,
                        stiffness,
                        n_legendre_terms,
                    )
                except Exception as exc:
                    csd_reason = f"failed:{exc}"
                    on_error = str(eeg_csd_cfg.get("on_error", "warn")).strip().lower()
                    if on_error in {"raise", "fail", "error"}:
                        raise RuntimeError(
                            f"EEG CSD failed for {file_path.name} with on_error={on_error}: {exc}"
                        ) from exc
                    logger.warning("EEG CSD failed for %s (%s); continuing without CSD", file_path.name, exc)
    else:
        csd_reason = "disabled"
    preprocess_timings["eeg_csd"] = float(time.perf_counter() - t_csd0)

    # Average re-reference for EEG-like channels if requested.
    # We do manual referencing so iEEG (seeg/ecog) is also covered.
    t_reref0 = time.perf_counter()
    eeg_chans = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)
    if reref == "average" and len(eeg_chans) > 0:
        if csd_applied:
            logger.info("Skipping average reref for %s because EEG CSD is already reference-free", file_path.name)
        else:
            try:
                eeg_data = raw.get_data(picks=eeg_chans)
                eeg_ref = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
                raw._data[eeg_chans, :] = eeg_ref  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("Average reference failed for %s (%s); continuing without reref", file_path.name, exc)
    preprocess_timings["reref"] = float(time.perf_counter() - t_reref0)

    # Collect per-modality arrays
    modality_signals: Dict[str, np.ndarray] = {}
    channels_dict: Dict[str, List[str]] = {}

    t_collect0 = time.perf_counter()
    # EEG (including iEEG: seeg/ecog)
    # MNE returns SI units (V). Convert to µV so PSD band-powers are on a
    # numerically stable scale (~1 µV²/Hz instead of ~1e-12 V²/Hz).
    eeg_chans = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)
    if len(eeg_chans) > 0:
        modality_signals["eeg"] = raw.get_data(picks=eeg_chans) * 1e6
        channels_dict["eeg"] = [raw.ch_names[i] for i in eeg_chans]

    # EOG
    eog_chans = mne.pick_types(raw.info, eog=True)
    if len(eog_chans) > 0:
        modality_signals["eog"] = raw.get_data(picks=eog_chans)
        channels_dict["eog"] = [raw.ch_names[i] for i in eog_chans]

    # EMG
    emg_chans = mne.pick_types(raw.info, emg=True)
    if len(emg_chans) > 0:
        modality_signals["emg"] = raw.get_data(picks=emg_chans)
        channels_dict["emg"] = [raw.ch_names[i] for i in emg_chans]

    # ECG
    ecg_chans = mne.pick_types(raw.info, ecg=True)
    if len(ecg_chans) > 0:
        modality_signals["ecg"] = raw.get_data(picks=ecg_chans)
        channels_dict["ecg"] = [raw.ch_names[i] for i in ecg_chans]
    preprocess_timings["collect_modalities"] = float(time.perf_counter() - t_collect0)
    preprocess_timings["total"] = float(time.perf_counter() - t_pre0)

    logger.info(f"Preprocessed {file_path.name}: {list(modality_signals.keys())}")

    artifact_meta: Dict[str, Any] = {
        "method": art_method,
        "bad_eeg_channels": bad_eeg_channels,
        "bad_eeg_reasons": bad_info.get("reasons", {}),
    }

    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "original_sfreq": original_sfreq,
        "target_sfreq_resolved": float(target_sfreq),
        "sfreq_policy": sfreq_policy,
        "crop_window": {
            "tmin": float(crop_window_info[0]) if crop_window_info is not None else None,
            "tmax": float(crop_window_info[1]) if (crop_window_info is not None and crop_window_info[1] is not None) else None,
        },
        "artifact": artifact_meta,
        "eeg_csd": {
            "enabled": bool(eeg_csd_cfg.get("enabled", False)),
            "applied": bool(csd_applied),
            "reason": csd_reason,
            "lambda2": float(eeg_csd_cfg.get("lambda2", 1e-5) or 1e-5),
            "stiffness": float(eeg_csd_cfg.get("stiffness", 4.0) or 4.0),
            "n_legendre_terms": int(eeg_csd_cfg.get("n_legendre_terms", 50) or 50),
        },
        "timings": preprocess_timings,
    }
    if eeglab_import_meta is not None:
        meta["eeglab_import"] = eeglab_import_meta
        meta["eeglab_concat_policy"] = {
            "skip_time_filters": bool(skip_time_filters),
            "resample_concatenated_epochs": bool(eeglab_concat_policy.get("resample_concatenated_epochs", False)),
            "filter_concatenated_epochs": bool(eeglab_concat_policy.get("filter_concatenated_epochs", False)),
        }

    return PreprocessedSignals(
        signals=modality_signals,
        sfreq=raw.info["sfreq"],
        channels=channels_dict,
        meta=meta,
    )


def preprocess_fmri(file_path: Path, config: Mapping[str, Any]) -> PreprocessedSignals:
    """Preprocess a single fMRI BOLD file into regional time series.

    This function performs minimal temporal cleaning and atlas-based
    parcellation to produce region-level BOLD time series suitable for
    downstream MNPS feature extraction.
    """
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("nibabel is required for fMRI preprocessing") from exc

    dataset_id = _infer_dataset_id(file_path, config)
    fmri_cfg = _resolve_fmri_config(config, dataset_id)

    atlas_path = fmri_cfg.get("atlas_path")
    if not atlas_path:
        raise ValueError("preprocess.fmri.atlas_path must be set in the config for fMRI preprocessing")

    atlas_path = Path(atlas_path)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    # Load BOLD and atlas images
    bold_img = nib.load(str(file_path))
    bold_data = bold_img.get_fdata()
    if bold_data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD data in {file_path}, got shape {bold_data.shape}")

    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()
    atlas_labels_expected = np.unique(atlas_data)
    atlas_labels_expected = atlas_labels_expected[atlas_labels_expected > 0]
    if atlas_data.shape != bold_data.shape[:3]:
        # Optionally resample atlas to the BOLD grid if requested in config.
        # This is useful when using a standard MNI atlas with native-space
        # BOLD data (e.g., ds000228 Pixar movie), but must be used with care
        # since it assumes both are in the same anatomical space.
        if bool(fmri_cfg.get("resample_atlas_to_bold", False)):
            try:
                from nibabel.processing import resample_from_to  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError(
                    "Atlas/BOLD shape mismatch and nibabel.processing.resample_from_to "
                    "is not available; cannot resample atlas to BOLD space."
                ) from exc

            logger.warning(
                "Resampling atlas %s from shape %s to match BOLD shape %s for %s",
                atlas_path,
                atlas_data.shape,
                bold_data.shape[:3],
                file_path.name,
            )
            # Use nearest-neighbour interpolation (order=0) to preserve integer labels.
            # Target is the 3D spatial grid of the BOLD image (shape + affine),
            # not the full 4D image, to avoid affine dimensionality mismatches.
            target = (bold_data.shape[:3], bold_img.affine)
            atlas_img_resampled = resample_from_to(atlas_img, target, order=0)
            atlas_img = atlas_img_resampled
            atlas_data = atlas_img_resampled.get_fdata()

            if atlas_data.shape != bold_data.shape[:3]:
                raise ValueError(
                    f"Resampled atlas shape {atlas_data.shape} still does not match BOLD spatial "
                    f"shape {bold_data.shape[:3]} for {file_path}"
                )
        else:
            raise ValueError(
                f"Atlas shape {atlas_data.shape} does not match BOLD spatial shape {bold_data.shape[:3]} for {file_path}"
            )

    # Derive TR and sampling frequency
    zooms = bold_img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else 0.0
    if not np.isfinite(tr) or tr <= 0:
        logger.warning("Invalid or missing TR for %s; defaulting to 1.0s", file_path.name)
        tr = 1.0
    sfreq = 1.0 / tr

    # Determine region labels
    labels = np.unique(atlas_data)
    labels = labels[labels > 0]  # ignore background
    n_expected_regions = int(atlas_labels_expected.size) if atlas_labels_expected.size else int(labels.size)
    n_present_regions = int(labels.size)

    min_regions_required_cfg = fmri_cfg.get("min_regions_required")
    if min_regions_required_cfg is None:
        min_region_fraction = float(fmri_cfg.get("min_region_fraction", 0.975) or 0.975)
        min_regions_required = int(np.ceil(max(0.0, min(1.0, min_region_fraction)) * max(1, n_expected_regions)))
    else:
        min_regions_required = int(min_regions_required_cfg)
    if min_regions_required > 0 and n_present_regions < min_regions_required:
        raise ValueError(
            "Insufficient atlas region coverage for robust graph/MNPS analysis: "
            f"present_regions={n_present_regions}, expected_regions={n_expected_regions}, "
            f"required_regions={min_regions_required}. "
            "Use spatially normalized derivatives (e.g., MNI/fMRIPrep) or fix atlas/BOLD alignment."
        )

    # Optional label names from TSV
    label_names: Dict[int, str] = {}
    atlas_labels = fmri_cfg.get("atlas_labels")
    if atlas_labels:
        labels_path = Path(atlas_labels)
        if labels_path.exists():
            try:
                df = pd.read_csv(labels_path, sep=None, engine="python")
                # Heuristic: look for id-like and name-like columns
                id_col = None
                name_col = None
                for col in df.columns:
                    c = str(col).lower()
                    if id_col is None and c in {"id", "label", "index", "roi_id"}:
                        id_col = col
                    if name_col is None and c in {"name", "label_name", "region", "roi_name"}:
                        name_col = col
                if id_col is not None and name_col is not None:
                    for _, row in df.iterrows():
                        try:
                            lab_id = int(row[id_col])
                        except Exception:
                            continue
                        label_names[lab_id] = str(row[name_col])
            except Exception as exc:
                logger.warning("Failed to read atlas_labels TSV %s: %s", labels_path, exc)

    # Parcellate: mean BOLD per atlas label
    n_times = bold_data.shape[3]
    region_ts: List[np.ndarray] = []
    region_names: List[str] = []
    for lab in labels.astype(int):
        mask = atlas_data == lab
        if not np.any(mask):
            continue
        # Extract all voxels for this label and average over space
        voxels = bold_data[mask, :]
        if voxels.ndim != 2 or voxels.shape[1] != n_times:
            voxels = voxels.reshape(-1, n_times)
        ts = np.nanmean(voxels, axis=0)
        region_ts.append(ts.astype(np.float32))
        region_names.append(label_names.get(lab, f"ROI_{lab}"))

    if not region_ts:
        raise ValueError(f"No non-empty atlas regions found for {file_path}")

    region_array = np.stack(region_ts, axis=0)  # [n_regions, n_times]

    # Optional temporal bandpass on regional time series
    bandpass = fmri_cfg.get("bandpass")
    if isinstance(bandpass, (list, tuple)) and len(bandpass) == 2:
        try:
            from scipy.signal import butter, filtfilt  # type: ignore

            low, high = float(bandpass[0]), float(bandpass[1])
            nyq = 0.5 * sfreq
            if 0.0 < low < high < nyq:
                b, a = butter(4, [low / nyq, high / nyq], btype="band")
                region_array = filtfilt(b, a, region_array, axis=1)
            else:
                logger.warning("Invalid fMRI bandpass [%s, %s] for sfreq=%.3f; skipping filter", low, high, sfreq)
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.warning("Failed to apply fMRI bandpass filter for %s: %s; continuing unfiltered", file_path.name, exc)

    modality_signals: Dict[str, np.ndarray] = {"fmri": region_array.astype(np.float32)}
    channels_dict: Dict[str, List[str]] = {"fmri": region_names}

    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "original_sfreq": sfreq,
        "atlas_path": str(atlas_path),
    }

    logger.info("Preprocessed fMRI %s: %d regions at sfreq=%.3f Hz", file_path.name, region_array.shape[0], sfreq)

    return PreprocessedSignals(
        signals=modality_signals,
        sfreq=sfreq,
        channels=channels_dict,
        meta=meta,
    )



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
import hashlib
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from .bdf_adapter import apply_bdf_adapter_policy, interpolate_configured_bdf_bads
from .reproducibility import resolve_component_seed

# Disable numba JIT to avoid long import stalls on Windows/venv
os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

try:
    import mne
except Exception:  # pragma: no cover - optional dependency
    mne = None  # type: ignore
try:
    import wfdb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wfdb = None  # type: ignore
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


# -----------------------------------------------------------------------------
# Physio TSV injection
# -----------------------------------------------------------------------------

def _inject_physio_tsv_channels(
    raw: "mne.io.BaseRaw",
    eeg_file_path: Path,
    physio_cfg: Mapping[str, Any],
) -> None:
    """Inject physiological channels from a BIDS *_physio.tsv.gz file into raw.

    The BioPac (or similar) physiological recording is stored separately from
    the EEG file in BIDS, typically under sub-*/ses-*/beh/*_physio.tsv.gz with
    a companion *_physio.json that lists column names and sampling frequency.

    Strategy ``beh_sibling`` (default): replace the modality folder (e.g. "eeg")
    with "beh" in the EEG file's parent path, then substitute the suffix
    ``_eeg.*`` with ``_physio``.

    Config keys (under ``preprocess.datasets.<id>.physio_tsv_inject``):
      enabled: bool
      path_strategy: "beh_sibling" (default)
      sampling_frequency: float  # fallback if physio.json is absent
      channels:
        - {column: "ECG A, X, ECG2-R", name: "ECG", type: "ecg", unit: "V"}
        - {column: "RSP A, X, RSP2-R", name: "RESP",  type: "resp", unit: "AU"}
        - {column: "EDA, Y, PPGED-R",  name: "EDA",   type: "gsr", unit: "uS"}
    """
    if not physio_cfg.get("enabled", False):
        return

    strategy = str(physio_cfg.get("path_strategy", "beh_sibling")).lower()
    channel_specs: List[Mapping[str, Any]] = list(physio_cfg.get("channels", []) or [])
    if not channel_specs:
        logger.debug("physio_tsv_inject: no channels specified; skipping")
        return

    # --- locate the physio file ---
    physio_tsv: Optional[Path] = None
    physio_json: Optional[Path] = None

    if strategy == "beh_sibling":
        eeg_dir = eeg_file_path.parent
        beh_dir = eeg_dir.parent / "beh"
        # Derive the physio stem: strip the last suffix extension then replace
        # "_eeg" suffix (or whatever modality suffix) with "_physio".
        stem = eeg_file_path.stem
        # Remove multi-extension suffix like ".set", ".vhdr", ".edf"
        for ext in (".set", ".vhdr", ".edf", ".bdf", ".fif"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        # Replace modality tag at end (e.g. _eeg, _meg, _ieeg) with _physio
        physio_stem = re.sub(r"_eeg$|_meg$|_ieeg$|_ecog$", "_physio", stem)
        if physio_stem == stem:
            # No modality tag found — just append _physio
            physio_stem = stem + "_physio"
        candidate_tsv = beh_dir / (physio_stem + ".tsv.gz")
        candidate_json = beh_dir / (physio_stem + ".json")
        if candidate_tsv.exists():
            physio_tsv = candidate_tsv
            physio_json = candidate_json if candidate_json.exists() else None
        else:
            logger.info(
                "physio_tsv_inject: physio file not found at %s; skipping",
                candidate_tsv,
            )
            return
    else:
        logger.warning("physio_tsv_inject: unknown path_strategy '%s'; skipping", strategy)
        return

    # --- read metadata ---
    fallback_sfreq = float(physio_cfg.get("sampling_frequency", 1000.0) or 1000.0)
    physio_sfreq = fallback_sfreq
    json_columns: List[str] = []
    if physio_json is not None:
        try:
            import json as _json
            meta = _json.loads(physio_json.read_text(encoding="utf-8"))
            physio_sfreq = float(meta.get("SamplingFrequency", fallback_sfreq))
            json_columns = list(meta.get("Columns", []) or [])
        except Exception as e:
            logger.warning("physio_tsv_inject: failed to read physio.json (%s); using fallback sfreq=%.0f", e, fallback_sfreq)

    # --- read TSV ---
    # BIDS convention: *_physio.tsv.gz files have NO header row.
    # Column names always come from the companion *_physio.json "Columns" field.
    try:
        phys_df = pd.read_csv(physio_tsv, sep="\t", header=None, compression="gzip")
        if json_columns and len(json_columns) == phys_df.shape[1]:
            phys_df.columns = json_columns
        else:
            # Fallback: use generic column names
            phys_df.columns = [f"col_{i}" for i in range(phys_df.shape[1])]
            if json_columns:
                logger.warning(
                    "physio_tsv_inject: column count mismatch (json=%d, tsv=%d) for %s; using generic names",
                    len(json_columns), phys_df.shape[1], physio_tsv.name,
                )
    except Exception as e:
        logger.warning("physio_tsv_inject: failed to read %s (%s); skipping", physio_tsv, e)
        return

    # --- extract and inject requested channels ---
    eeg_sfreq = raw.info["sfreq"]
    eeg_n_times = raw.n_times

    injected: List[str] = []
    ch_types: Dict[str, str] = {}
    arrays: List[np.ndarray] = []
    ch_names_new: List[str] = []
    ch_units: Dict[str, str] = {}

    for spec in channel_specs:
        col = str(spec.get("column", ""))
        ch_name = str(spec.get("name", col))
        ch_type = str(spec.get("type", "misc")).lower()
        ch_unit = str(spec.get("unit", "AU"))

        if col not in phys_df.columns:
            # Try case-insensitive match
            col_lower = {c.lower(): c for c in phys_df.columns}
            col = col_lower.get(col.lower(), "")
        if not col or col not in phys_df.columns:
            logger.debug("physio_tsv_inject: column '%s' not found in %s; skipping channel %s", spec.get("column"), physio_tsv.name, ch_name)
            continue
        if ch_name in raw.ch_names:
            logger.debug("physio_tsv_inject: channel '%s' already exists in raw; skipping injection", ch_name)
            continue

        signal = phys_df[col].to_numpy(dtype=np.float64, na_value=0.0)

        # Resample from physio_sfreq to eeg_sfreq
        if abs(physio_sfreq - eeg_sfreq) > 0.5:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                ratio_num = int(round(eeg_sfreq))
                ratio_den = int(round(physio_sfreq))
                g = gcd(ratio_num, ratio_den)
                signal = resample_poly(signal, ratio_num // g, ratio_den // g).astype(np.float64)
            except Exception as e:
                logger.warning("physio_tsv_inject: resampling failed for %s (%s); using nearest-sample fallback", ch_name, e)
                indices = np.round(np.linspace(0, len(signal) - 1, int(len(signal) * eeg_sfreq / physio_sfreq))).astype(int)
                indices = np.clip(indices, 0, len(signal) - 1)
                signal = signal[indices]

        # Align length to EEG (pad with zeros or trim)
        if len(signal) < eeg_n_times:
            signal = np.pad(signal, (0, eeg_n_times - len(signal)))
        elif len(signal) > eeg_n_times:
            signal = signal[:eeg_n_times]

        arrays.append(signal)
        ch_names_new.append(ch_name)
        ch_types[ch_name] = ch_type
        ch_units[ch_name] = ch_unit
        injected.append(ch_name)

    if not arrays:
        logger.info("physio_tsv_inject: no matching columns found in %s; skipping", physio_tsv.name)
        return

    # Build an MNE Info and RawArray for the new channels, then add to raw
    try:
        new_info = mne.create_info(ch_names_new, sfreq=eeg_sfreq, ch_types=[ch_types[n] for n in ch_names_new])
        new_raw = mne.io.RawArray(np.vstack(arrays), new_info, verbose=False)
        raw.add_channels([new_raw], force_update_info=True)
        logger.info(
            "physio_tsv_inject: injected %d channels from %s (%s) into raw (physio_sfreq=%.0f→%.0f Hz)",
            len(injected), physio_tsv.name, injected, physio_sfreq, eeg_sfreq,
        )
    except Exception as e:
        logger.warning("physio_tsv_inject: failed to add channels to raw (%s); skipping", e)


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


def _infer_path_datatype(file_path: Path) -> Optional[str]:
    """Infer a coarse BIDS datatype from path segments."""
    lowered = [str(part).strip().lower() for part in file_path.parts]
    for name in ("eeg", "meg", "ecg", "pupil", "beh", "func"):
        if name in lowered:
            return "fmri" if name == "func" else name
    return None


def _infer_dataset_id(file_path: Path, config: Mapping[str, Any]) -> Optional[str]:
    """Internal helper: infer dataset id."""
    dataset_ids = config.get("datasets") if isinstance(config, Mapping) else None
    path_str = file_path.as_posix()
    normalized_ids: List[str] = []
    if isinstance(dataset_ids, (str, bytes)):
        dataset_iterable: Iterable[Any] = [str(dataset_ids)]
    elif isinstance(dataset_ids, Iterable):
        dataset_iterable = dataset_ids
    else:
        dataset_iterable = []
    if dataset_iterable:
        for ds in dataset_iterable:
            ds_str = str(ds)
            normalized_ids.append(ds_str)
            token = f"/{ds_str}/"
            if token in path_str:
                return ds_str
        if len(normalized_ids) == 1:
            return normalized_ids[0]
    for parent in file_path.parents:
        name = parent.name
        if name.startswith("ds") and name[2:].isdigit():
            return name
    return None


def preprocess_pupil_table(file_path: Path, config: Mapping[str, Any]) -> PreprocessedSignals:
    """Preprocess a pupil TSV/CSV file into aligned pupil signals."""
    dataset_id = _infer_dataset_id(file_path, config)
    suffix = file_path.suffix.lower()
    sep = "\t" if suffix == ".tsv" else ","
    table = pd.read_csv(file_path, sep=sep)
    if table.empty:
        raise ValueError(f"Pupil table is empty: {file_path}")

    lower_map = {str(col).strip().lower(): str(col) for col in table.columns}
    time_candidates = ["time", "timestamp", "timestamps", "pupil_timestamp", "recording_timestamp"]
    time_col = next((lower_map[name] for name in time_candidates if name in lower_map), None)
    if time_col is not None:
        times = pd.to_numeric(table[time_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite_times = times[np.isfinite(times)]
        if finite_times.size >= 2:
            dt = np.diff(finite_times)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            sfreq = float(1.0 / np.median(dt)) if dt.size else 120.0
        else:
            sfreq = 120.0
    else:
        sfreq = 120.0

    pupil_cols: list[str] = []
    for col in table.columns:
        key = str(col).strip().lower()
        if any(token in key for token in ("diameter", "pupil")) and not any(
            token in key for token in ("confidence", "timestamp", "time", "blink", "gaze", "norm_pos", " x", " y")
        ):
            pupil_cols.append(str(col))
    if not pupil_cols:
        raise ValueError(f"Could not find pupil diameter columns in {file_path}")

    pupil_cols = pupil_cols[:2]
    pupil_values = [
        pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        for col in pupil_cols
    ]
    pupil_arr = np.vstack([np.asarray(v, dtype=np.float32) for v in pupil_values])
    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "pupil_time_column": time_col,
        "pupil_signal_columns": list(pupil_cols),
        "table_rows": int(len(table)),
        "original_sfreq": float(sfreq),
        "target_sfreq_resolved": float(sfreq),
    }
    return PreprocessedSignals(
        signals={"pupil": pupil_arr},
        sfreq=float(sfreq),
        channels={"pupil": list(pupil_cols)},
        meta=meta,
    )


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
    """Internal helper: resolve event crop config."""
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


def _resolve_nwb_config(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve NWB preprocessing configuration with optional per-dataset overrides."""
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    nwb_cfg = preprocess_cfg.get("nwb", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(nwb_cfg, Mapping):
        return {}
    merged: Dict[str, Any] = {k: nwb_cfg[k] for k in nwb_cfg if k != "datasets"}
    ds_overrides = nwb_cfg.get("datasets", {}) if isinstance(nwb_cfg.get("datasets", {}), Mapping) else {}
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _find_events_file(file_path: Path, cfg: Mapping[str, Any]) -> Optional[Path]:
    """Internal helper: find events file."""
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
    """Internal helper: brainvision get key."""
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


def _repair_eeglab_set_fdt_path(set_path: Path) -> Optional[Path]:
    """Return a temp-patched .set path when the internal FDT reference is stale.

    Some OpenNeuro datasets (e.g. ds003645, originally ds000117) were renamed
    to BIDS format after the EEGLAB export was done.  The .set files still
    reference the old FDT filename (e.g. ``ds000117_sub002_run-1.fdt``) but
    the on-disk file uses the BIDS name (e.g.
    ``sub-002_task-FacePerception_run-1_eeg.fdt``).  This function:

    1. Reads the mat structure with scipy.io.
    2. Checks whether the referenced FDT file exists.
    3. If not, locates the actual ``.fdt`` in the same directory.
    4. Writes a temp ``.mat`` file with the corrected ``datfile`` field and
       returns its path for MNE to load instead.

    Returns ``None`` when no repair is needed or possible.
    """
    try:
        import scipy.io as sio
    except ImportError:
        return None
    try:
        mat = sio.loadmat(str(set_path), squeeze_me=True)
    except Exception:
        return None

    datfile_raw = mat.get("datfile")
    if datfile_raw is None:
        return None
    datfile = str(datfile_raw).strip() if not isinstance(datfile_raw, np.ndarray) else str(datfile_raw.flat[0]).strip()
    if not datfile:
        return None

    fdt_referenced = set_path.parent / datfile
    if fdt_referenced.exists():
        return None  # already correct

    fdt_candidates = sorted(set_path.parent.glob("*.fdt"))
    if not fdt_candidates:
        return None

    run_m = re.search(r"run[_-](\d+)", set_path.stem, re.IGNORECASE)
    if run_m:
        # Use the full 'run-N' token (not just the digit) to avoid false
        # matches on subject IDs that share digits with run numbers
        # (e.g. run_num="2" would match "sub-002" in every filename).
        run_token = run_m.group(0).lower()  # e.g. "run-2" or "run_2"
        matched = [f for f in fdt_candidates if run_token in f.stem.lower()]
        actual_fdt = matched[0] if matched else fdt_candidates[0]
    else:
        actual_fdt = fdt_candidates[0]

    logger.info(
        "EEGLAB FDT mismatch for %s: referenced '%s' not found; using '%s'",
        set_path.name, datfile, actual_fdt.name,
    )
    try:
        # MNE reads EEG.data (not EEG.datfile) as the FDT path.
        # When EEG.data == 'in set file' (string with no .fdt extension) MNE
        # raises "Expected .fdt file format".  Patching EEG.data to the actual
        # FDT filename lets MNE find and load the binary data correctly.
        mat["datfile"] = np.array([actual_fdt.name], dtype=object)
        mat["data"] = np.array([actual_fdt.name], dtype=object)
        tmp_set = set_path.parent / f"_fdt_patched_{set_path.name}"
        sio.savemat(str(tmp_set), mat)
        return tmp_set
    except Exception as exc:
        logger.warning("FDT path repair failed for %s: %s", set_path.name, exc)
        return None


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
    # Pass through any remaining dataset-specific keys (e.g. physio_tsv_inject)
    # so callers can retrieve them without modifying this function for each new key.
    for key, val in ds_cfg.items():
        if key not in policy:
            policy[key] = val
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


def _iter_nwb_electrical_series(container: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield candidate NWB ElectricalSeries-like objects from a container."""
    if container is None:
        return
    try:
        items = container.items() if hasattr(container, "items") else []
    except Exception:
        items = []
    for name, obj in items:
        path = f"{prefix}/{name}" if prefix else str(name)
        class_name = obj.__class__.__name__.lower()
        if hasattr(obj, "data") and (hasattr(obj, "electrodes") or "electrical" in class_name):
            yield path, obj
        if hasattr(obj, "electrical_series"):
            try:
                for series_name, series_obj in obj.electrical_series.items():
                    yield f"{path}/{series_name}", series_obj
            except Exception:
                pass
        if hasattr(obj, "data_interfaces"):
            try:
                yield from _iter_nwb_electrical_series(obj.data_interfaces, path)
            except Exception:
                pass


def _find_nwb_electrical_series(nwbfile: Any, nwb_cfg: Mapping[str, Any]) -> Tuple[str, Any]:
    """Select the configured or first available ElectricalSeries-like object."""
    candidates: list[Tuple[str, Any]] = []
    try:
        candidates.extend(_iter_nwb_electrical_series(getattr(nwbfile, "acquisition", {}), "acquisition"))
    except Exception:
        pass
    try:
        candidates.extend(_iter_nwb_electrical_series(getattr(nwbfile, "processing", {}), "processing"))
    except Exception:
        pass
    if not candidates:
        raise ValueError("No ElectricalSeries-like NWB acquisition or processing object found.")

    requested = str(nwb_cfg.get("series_path", "") or "").strip()
    if requested:
        for path, obj in candidates:
            if path == requested or path.endswith(f"/{requested}") or requested in path:
                return path, obj
        raise ValueError(f"Configured preprocess.nwb.series_path={requested!r} was not found.")

    prefer = [str(x).lower() for x in nwb_cfg.get("prefer_series_keywords", ["lfp", "electrical", "eeg"]) or []]
    for keyword in prefer:
        for path, obj in candidates:
            haystack = f"{path} {obj.__class__.__name__}".lower()
            if keyword and keyword in haystack:
                return path, obj
    return candidates[0]


def _resolve_nwb_channel_selection(nwb_cfg: Mapping[str, Any]) -> tuple[list[int] | None, dict[str, Any]]:
    """Resolve a versioned LFP channel-selection artifact before data loading."""
    selection_cfg = nwb_cfg.get("channel_selection", {})
    if not isinstance(selection_cfg, Mapping):
        return None, {}
    path_value = selection_cfg.get("artifact_path")
    if not path_value:
        return None, {}
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Configured NWB channel-selection artifact does not exist: {path}")
    import json

    artifact = json.loads(path.read_text(encoding="utf-8"))
    selected = artifact.get("selected_channels", [])
    indices = [int(row["channel_index"]) for row in selected if isinstance(row, Mapping) and "channel_index" in row]
    if not indices:
        raise ValueError(f"NWB channel-selection artifact selected no channels: {path}")
    return sorted(set(indices)), artifact


def _nwb_series_sfreq(series: Any) -> float:
    rate = getattr(series, "rate", None)
    if rate is not None:
        try:
            return float(rate)
        except Exception:
            pass
    timestamps = getattr(series, "timestamps", None)
    if timestamps is not None:
        arr = np.asarray(timestamps[:], dtype=float)
        diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(1.0 / np.median(diffs))
    raise ValueError("NWB ElectricalSeries has neither a valid rate nor usable timestamps.")


def _nwb_series_channel_names(series: Any, n_channels: int) -> List[str]:
    names: List[str] = []
    electrodes = getattr(series, "electrodes", None)
    table = getattr(electrodes, "table", None)
    for col_name in ("label", "labels", "channel_name", "location", "group_name"):
        try:
            if table is not None and col_name in table.colnames:
                values = list(table[col_name].data[:])
                names = [str(values[i]) for i in range(min(n_channels, len(values)))]
                break
        except Exception:
            continue
    if len(names) < n_channels:
        names.extend([f"nwb_ch{idx:03d}" for idx in range(len(names), n_channels)])
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for name in names[:n_channels]:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count + 1}")
    return unique


def _nwb_data_to_channels_by_samples(
    series: Any,
    *,
    start_sample: int | None = None,
    stop_sample: int | None = None,
    max_channels: int | None = None,
    channel_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Load an ElectricalSeries, optionally bounding time/channel memory use."""
    source = series.data
    sample_slice = slice(max(0, start_sample or 0), stop_sample)
    if getattr(source, "ndim", 2) == 1:
        data = np.asarray(source[sample_slice], dtype=float)
    else:
        columns: Any
        if channel_indices is not None:
            columns = np.asarray(list(channel_indices), dtype=int)
        elif max_channels:
            columns = slice(0, max_channels)
        else:
            columns = slice(None)
        data = np.asarray(source[sample_slice, columns], dtype=float)
    conversion = getattr(series, "conversion", None)
    if conversion is not None:
        try:
            data = data * float(conversion)
        except Exception:
            pass
    offset = getattr(series, "offset", None)
    if offset is not None:
        try:
            data = data + float(offset)
        except Exception:
            pass
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:
        # NWB TimeSeries convention is usually time x channel.
        if data.shape[0] >= data.shape[1]:
            data = data.T
    else:
        raise ValueError(f"Expected 1D or 2D ElectricalSeries data, got shape {data.shape}.")
    return data


def _nwb_chunked_downsample(
    series: Any,
    *,
    source_sfreq: float,
    target_sfreq: float,
    channel_indices: Sequence[int],
    chunk_sec: float,
    start_sample: int = 0,
    stop_sample: int | None = None,
) -> np.ndarray:
    """Read selected channels in bounded chunks, downsampling before concatenation."""
    from scipy.signal import resample_poly

    total_samples = int(series.data.shape[0])
    stop = min(total_samples, stop_sample) if stop_sample is not None else total_samples
    ratio = float(source_sfreq) / float(target_sfreq)
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("NWB chunked downsampling requires an integer source/target rate ratio.")
    down = int(round(ratio))
    chunk_samples = max(down, int(round(chunk_sec * source_sfreq / down)) * down)
    chunks: list[np.ndarray] = []
    for first in range(start_sample, stop, chunk_samples):
        part = _nwb_data_to_channels_by_samples(
            series,
            start_sample=first,
            stop_sample=min(stop, first + chunk_samples),
            channel_indices=channel_indices,
        )
        chunks.append(resample_poly(part, up=1, down=down, axis=1).astype(np.float64, copy=False))
    return np.concatenate(chunks, axis=1) if chunks else np.empty((len(channel_indices), 0), dtype=np.float64)


def _nwb_unit_to_microvolt_factor(unit: Any, *, scale_to_microvolts: bool) -> float:
    """Return factor from the NWB physical unit to microvolts."""
    if not scale_to_microvolts:
        return 1.0
    unit_text = str(unit or "").strip().lower()
    if unit_text in {"v", "volt", "volts"}:
        return 1e6
    if unit_text in {"mv", "millivolt", "millivolts"}:
        return 1e3
    if unit_text in {"uv", "µv", "microvolt", "microvolts"}:
        return 1.0
    # Preserve previous behavior for sparse metadata, but make known units safer.
    return 1e6


def _wfdb_channel_type(name: str) -> str:
    """Infer a coarse MNE channel type from a WFDB signal name."""
    ch = str(name or "").strip().lower()
    if ch.startswith("ecg") or ch.startswith("ekg"):
        return "ecg"
    if "ppg" in ch or "pleth" in ch or "pulse" in ch or "photopleth" in ch:
        return "bio"
    if "eog" in ch:
        return "eog"
    if "emg" in ch:
        return "emg"
    if "resp" in ch or "thor" in ch or "abd" in ch:
        return "resp"
    if "eda" in ch or "gsr" in ch:
        return "gsr"
    return "eeg"


def _wfdb_unit_to_microvolt_factor(unit: Any, *, scale_to_microvolts: bool) -> float:
    """Return factor from WFDB unit to microvolts."""
    if not scale_to_microvolts:
        return 1.0
    text = str(unit or "").strip().lower()
    if text in {"v", "volt", "volts"}:
        return 1e6
    if text in {"mv", "millivolt", "millivolts"}:
        return 1e3
    if text in {"uv", "µv", "microvolt", "microvolts"}:
        return 1.0
    # "nu" in I-CARE headers denotes normalized/no-unit values; keep unchanged.
    if text in {"nu", "adu", "au", ""}:
        return 1.0
    return 1.0


def preprocess_wfdb(file_path: Path, config: Mapping[str, Any]) -> PreprocessedSignals:
    """Preprocess PhysioNet/WFDB recordings from .hea/.mat pairs."""
    if mne is None:
        raise RuntimeError("mne is required for WFDB preprocessing but is not installed.")
    if wfdb is None:
        raise RuntimeError("wfdb is required for WFDB preprocessing. Install with `pip install wfdb`.")

    t_pre0 = time.perf_counter()
    dataset_id = _infer_dataset_id(file_path, config)
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    wfdb_cfg = preprocess_cfg.get("wfdb", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(wfdb_cfg, Mapping):
        wfdb_cfg = {}
    preprocess_timings: Dict[str, float] = {}

    record_stem = str(file_path.with_suffix(""))
    try:
        record = wfdb.rdrecord(record_stem)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read WFDB record for {file_path}. "
            "Ensure matching signal files (e.g., .mat/.dat) are present."
        ) from exc

    data = np.asarray(getattr(record, "p_signal", None), dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise RuntimeError(f"WFDB record {file_path} has unexpected signal shape: {data.shape}")

    sig_names = list(getattr(record, "sig_name", []) or [])
    if len(sig_names) != data.shape[1]:
        sig_names = [f"wfdb_ch{idx:03d}" for idx in range(data.shape[1])]
    units = list(getattr(record, "units", []) or [])
    if len(units) != data.shape[1]:
        units = [""] * data.shape[1]

    ch_types = [_wfdb_channel_type(name) for name in sig_names]
    fs = float(getattr(record, "fs", 0.0) or 0.0)
    if fs <= 0:
        raise RuntimeError(f"WFDB record {file_path} has invalid sampling rate: {fs}")

    info = mne.create_info(ch_names=sig_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info, verbose=False)

    # Optional dataset-specific channel typing overrides.
    _apply_channel_typing(raw, dataset_id, config)

    crop_cfg = _resolve_crop_config(config, dataset_id)
    if crop_cfg:
        # NWB data were sliced before materialisation, so crop relative to the
        # in-memory segment rather than applying start_sec twice.
        tmin = 0.0
        tmax_val = crop_cfg.get("stop_sec", None)
        start_val = float(crop_cfg.get("start_sec", 0) or 0.0)
        tmax = float(tmax_val) - start_val if tmax_val is not None else None
        try:
            raw.crop(tmin=max(tmin, 0.0), tmax=tmax)
        except Exception as exc:
            logger.warning("WFDB crop failed for %s (%s); continuing without crop", file_path.name, exc)

    target_sfreq, sfreq_policy = _resolve_target_sfreq(
        original_sfreq=float(fs),
        preprocess_cfg=preprocess_cfg if isinstance(preprocess_cfg, Mapping) else {},
    )
    resample_cfg = preprocess_cfg.get("resample", {}) if isinstance(preprocess_cfg, Mapping) else {}
    resample_n_jobs = resample_cfg.get("n_jobs", 1) if isinstance(resample_cfg, Mapping) else 1
    if isinstance(resample_n_jobs, str) and resample_n_jobs.strip().lower() == "auto":
        n_jobs_resample: Any = "auto"
    else:
        try:
            n_jobs_resample = max(1, int(resample_n_jobs))
        except Exception:
            n_jobs_resample = 1
    t_resample0 = time.perf_counter()
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, n_jobs=n_jobs_resample, verbose=False)
    preprocess_timings["resample"] = float(time.perf_counter() - t_resample0)

    notch_hz = preprocess_cfg.get("notch_hz", None) if isinstance(preprocess_cfg, Mapping) else None
    eeg_picks = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)
    if notch_hz is not None and len(eeg_picks) > 0:
        raw.notch_filter(freqs=notch_hz, picks=eeg_picks, verbose=False)

    eeg_bandpass = preprocess_cfg.get("eeg_bandpass", [1, 45]) if isinstance(preprocess_cfg, Mapping) else [1, 45]
    meg_bandpass = (
        preprocess_cfg.get("meg_bandpass", eeg_bandpass)
        if isinstance(preprocess_cfg, Mapping)
        else eeg_bandpass
    )
    if eeg_bandpass is not None and len(eeg_bandpass) == 2 and len(eeg_picks) > 0:
        raw.filter(l_freq=eeg_bandpass[0], h_freq=eeg_bandpass[1], picks=eeg_picks, verbose=False)

    reref = preprocess_cfg.get("reref", None) if isinstance(preprocess_cfg, Mapping) else None
    if str(reref).lower() == "average" and len(eeg_picks) > 0:
        eeg_data = raw.get_data(picks=eeg_picks)
        raw._data[eeg_picks, :] = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)  # type: ignore[attr-defined]

    if len(eeg_picks) == 0:
        raise RuntimeError(f"No EEG channels detected in WFDB record: {file_path}")

    scale_to_microvolts = bool(wfdb_cfg.get("scale_to_microvolts", True))
    eeg_indices = set(int(i) for i in eeg_picks)
    eeg_scale = np.array(
        [
            _wfdb_unit_to_microvolt_factor(units[idx], scale_to_microvolts=scale_to_microvolts)
            for idx in range(len(sig_names))
            if idx in eeg_indices
        ],
        dtype=float,
    )
    eeg_values = raw.get_data(picks=eeg_picks)
    if eeg_scale.shape[0] == eeg_values.shape[0]:
        eeg_values = eeg_values * eeg_scale[:, np.newaxis]

    preprocess_timings["total"] = float(time.perf_counter() - t_pre0)
    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "source_format": "WFDB",
        "wfdb_record_name": str(getattr(record, "record_name", "") or ""),
        "wfdb_scale_to_microvolts": scale_to_microvolts,
        "original_sfreq": float(fs),
        "target_sfreq_resolved": float(raw.info["sfreq"]),
        "sfreq_policy": sfreq_policy,
        "timings": preprocess_timings,
    }
    channels = {"eeg": [raw.ch_names[i] for i in eeg_picks]}
    logger.info("Preprocessed WFDB %s with %d EEG channel(s)", file_path.name, len(channels["eeg"]))
    return PreprocessedSignals(signals={"eeg": eeg_values}, sfreq=float(raw.info["sfreq"]), channels=channels, meta=meta)


def _preprocess_nwb_units(
    file_path: Path,
    *,
    dataset_id: str,
    nwb_cfg: Mapping[str, Any],
) -> PreprocessedSignals:
    """Turn an NWB Units table into a binned population-rate matrix."""
    try:
        from neuropixel_ingest.binning import bin_spike_times
        from neuropixel_ingest.contracts import EphysReadConfig
        from neuropixel_ingest.nwb_units import read_session_bundle
    except ImportError as exc:
        raise RuntimeError(
            "NWB Units preprocessing requires the repository's neuropixel_ingest package."
        ) from exc

    units_cfg = nwb_cfg.get("units", {})
    units_cfg = units_cfg if isinstance(units_cfg, Mapping) else {}
    read_cfg = EphysReadConfig(
        quality_policy=str(units_cfg.get("quality_policy", "all")),
        min_firing_rate_hz=units_cfg.get("min_firing_rate_hz"),
        min_presence_ratio=units_cfg.get("min_presence_ratio"),
        max_isi_violations_ratio=units_cfg.get("max_isi_violations_ratio"),
    )
    bundle = read_session_bundle(file_path, read_cfg, dataset_id=dataset_id)
    spike_times = bundle.spike_times()
    if not spike_times:
        raise RuntimeError(f"NWB Units table has no readable spike trains: {file_path}")
    bin_sec = float(units_cfg.get("rate_bin_sec", 0.05))
    rates, bin_centers = bin_spike_times(
        spike_times,
        bin_sec=bin_sec,
        start_sec=units_cfg.get("time_start_sec"),
        stop_sec=units_cfg.get("time_stop_sec"),
        smoothing_sigma_sec=units_cfg.get("smoothing_sigma_sec"),
    )
    channels = {
        "ephys": [
            str(value) for value in bundle.units.get("unit_id", pd.Series(range(rates.shape[0]))).tolist()
        ]
    }
    geometry = bundle.probe_geometry
    raw_locations = bundle.units.get("location_raw", pd.Series(dtype=object))
    acronyms = bundle.units.get("location_acronym", pd.Series(dtype=object))
    probe_ids = geometry.get("probe_id", pd.Series(dtype=object)) if not geometry.empty else pd.Series(dtype=object)
    anatomy_coverage = {
        "location_raw": float(pd.Series(raw_locations).notna().mean()) if len(raw_locations) else 0.0,
        "location_acronym": float(pd.Series(acronyms).notna().mean()) if len(acronyms) else 0.0,
        "depth_um": float(bundle.units.get("depth_um", pd.Series(dtype=float)).notna().mean())
        if "depth_um" in bundle.units
        else 0.0,
    }
    return PreprocessedSignals(
        signals={"ephys": rates},
        sfreq=1.0 / bin_sec,
        channels=channels,
        meta={
            "file": str(file_path),
            "dataset_id": dataset_id,
            "source_format": "NWB",
            "nwb_signal_kind": "units_rate",
            "nwb_unit_count": int(rates.shape[0]),
            "nwb_rate_bin_sec": bin_sec,
            "nwb_rate_times_sec": bin_centers.tolist(),
            "nwb_trial_count": int(len(bundle.trials)),
            "nwb_quality_policy": read_cfg.quality_policy,
            "nwb_unit_qc": dict(bundle.unit_qc),
            "nwb_regions_present": sorted(str(value) for value in pd.Series(acronyms).dropna().unique()),
            "nwb_probe_ids": sorted(str(value) for value in pd.Series(probe_ids).dropna().unique()),
            "nwb_anatomy_coverage": anatomy_coverage,
            "nwb_probe_geometry": geometry.to_dict(orient="records"),
        },
    )


def preprocess_nwb(file_path: Path, config: Mapping[str, Any]) -> PreprocessedSignals:
    """Preprocess NWB continuous ephys as EEG or Units as population rates."""
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise RuntimeError("pynwb is required for NWB preprocessing. Install it with `pip install pynwb`.") from exc

    t_pre0 = time.perf_counter()
    dataset_id = _infer_dataset_id(file_path, config)
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    nwb_cfg = _resolve_nwb_config(config, dataset_id)
    source_cfg = config.get("source", {}) if isinstance(config, Mapping) else {}
    units_cfg = nwb_cfg.get("units", {}) if isinstance(nwb_cfg, Mapping) else {}
    if isinstance(units_cfg, Mapping) and bool(units_cfg.get("enabled", False)):
        return _preprocess_nwb_units(file_path, dataset_id=dataset_id, nwb_cfg=nwb_cfg)
    if mne is None:
        raise RuntimeError("mne is required for continuous NWB preprocessing but is not installed.")

    with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        series_path, series = _find_nwb_electrical_series(nwbfile, nwb_cfg)
        original_sfreq = _nwb_series_sfreq(series)
        crop_cfg = _resolve_crop_config(config, dataset_id)
        nwb_start = float(crop_cfg.get("start_sec", 0) or 0.0) if crop_cfg else 0.0
        nwb_stop_value = crop_cfg.get("stop_sec") if crop_cfg else None
        nwb_stop = float(nwb_stop_value) if nwb_stop_value is not None else None
        max_channels_value = nwb_cfg.get("max_channels")
        max_channels = int(max_channels_value) if max_channels_value is not None else None
        channel_indices, selection_artifact = _resolve_nwb_channel_selection(nwb_cfg)
        streamed_sfreq = nwb_cfg.get("streaming_sfreq")
        if channel_indices is not None and streamed_sfreq is not None:
            streamed_sfreq = float(streamed_sfreq)
            data = _nwb_chunked_downsample(
                series,
                source_sfreq=original_sfreq,
                target_sfreq=streamed_sfreq,
                channel_indices=channel_indices,
                chunk_sec=float(nwb_cfg.get("chunk_sec", 60.0) or 60.0),
                start_sample=int(round(nwb_start * original_sfreq)),
                stop_sample=int(round(nwb_stop * original_sfreq)) if nwb_stop is not None else None,
            )
            original_sfreq = streamed_sfreq
        else:
            data = _nwb_data_to_channels_by_samples(
                series,
                start_sample=int(round(nwb_start * original_sfreq)),
                stop_sample=int(round(nwb_stop * original_sfreq)) if nwb_stop is not None else None,
                max_channels=max_channels,
                channel_indices=channel_indices,
            )
        channel_names = (
            [f"lfp_ch{index:03d}" for index in channel_indices]
            if channel_indices is not None
            else _nwb_series_channel_names(series, data.shape[0])
        )
        nwb_unit = getattr(series, "unit", None)
        nwb_conversion = getattr(series, "conversion", None)
        nwb_offset = getattr(series, "offset", None)
        subject = getattr(nwbfile, "subject", None)
        species = str(getattr(subject, "species", "") or source_cfg.get("species") or nwb_cfg.get("species", "") or "")
        nwb_session_id = getattr(nwbfile, "session_id", None)

    ch_type = str(nwb_cfg.get("channel_type", "eeg") or "eeg").lower()
    if ch_type not in {"eeg", "seeg", "ecog"}:
        ch_type = "eeg"
    info = mne.create_info(ch_names=channel_names, sfreq=float(original_sfreq), ch_types=[ch_type] * len(channel_names))
    raw = mne.io.RawArray(data, info, verbose=False)

    preprocess_timings: Dict[str, float] = {}
    if crop_cfg:
        tmin = float(crop_cfg.get("start_sec", 0) or 0.0)
        tmax_val = crop_cfg.get("stop_sec", None)
        tmax = float(tmax_val) if tmax_val is not None else None
        try:
            raw.crop(tmin=max(tmin, 0.0), tmax=tmax)
        except Exception as exc:
            logger.warning("NWB crop failed for %s (%s); continuing without crop", file_path.name, exc)

    target_sfreq, sfreq_policy = _resolve_target_sfreq(
        original_sfreq=float(original_sfreq),
        preprocess_cfg=preprocess_cfg if isinstance(preprocess_cfg, Mapping) else {},
    )
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    preprocess_timings["resample"] = float(time.perf_counter() - t_pre0)

    notch_hz = preprocess_cfg.get("notch_hz", None) if isinstance(preprocess_cfg, Mapping) else None
    if notch_hz is not None:
        raw.notch_filter(freqs=notch_hz, verbose=False)

    eeg_bandpass = preprocess_cfg.get("eeg_bandpass", [1, 45]) if isinstance(preprocess_cfg, Mapping) else [1, 45]
    if eeg_bandpass is not None and len(eeg_bandpass) == 2:
        raw.filter(l_freq=eeg_bandpass[0], h_freq=eeg_bandpass[1], verbose=False)

    reref = preprocess_cfg.get("reref", None) if isinstance(preprocess_cfg, Mapping) else None
    reref_mode = str(reref).strip().lower() if reref is not None else "native"
    eeg_picks = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)
    if reref_mode == "average" and len(eeg_picks) > 0:
        eeg_data = raw.get_data(picks=eeg_picks)
        raw._data[eeg_picks, :] = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)  # type: ignore[attr-defined]
    elif reref_mode in {"median", "median_within_probe"} and len(eeg_picks) > 0:
        eeg_data = raw.get_data(picks=eeg_picks)
        raw._data[eeg_picks, :] = eeg_data - np.median(eeg_data, axis=0, keepdims=True)  # type: ignore[attr-defined]

    scale_to_microvolts = bool(nwb_cfg.get("scale_to_microvolts", True))
    unit_factor = _nwb_unit_to_microvolt_factor(nwb_unit, scale_to_microvolts=scale_to_microvolts)
    signals = {"eeg": raw.get_data(picks=eeg_picks) * unit_factor}
    channels = {"eeg": [raw.ch_names[i] for i in eeg_picks]}
    preprocess_timings["total"] = float(time.perf_counter() - t_pre0)

    selection_path_value = (
        nwb_cfg.get("channel_selection", {}).get("artifact_path")
        if isinstance(nwb_cfg.get("channel_selection", {}), Mapping)
        else None
    )
    selection_hash = None
    if selection_path_value:
        try:
            selection_hash = hashlib.sha256(Path(str(selection_path_value)).read_bytes()).hexdigest()
        except OSError:
            selection_hash = None
    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "source_format": "NWB",
        "species": species or None,
        "dandiset_id": source_cfg.get("dataset_id") if isinstance(source_cfg, Mapping) else None,
        "nwb_session_id": str(nwb_session_id) if nwb_session_id else None,
        "nwb_series_path": series_path,
        "nwb_channel_type": ch_type,
        "nwb_unit": str(nwb_unit) if nwb_unit is not None else None,
        "nwb_conversion": float(nwb_conversion) if nwb_conversion is not None else None,
        "nwb_offset": float(nwb_offset) if nwb_offset is not None else None,
        "nwb_scale_to_microvolts": scale_to_microvolts,
        "nwb_unit_to_output_factor": float(unit_factor),
        "nwb_reref_mode": reref_mode,
        "nwb_reref_channel_count": len(eeg_picks),
        "lfp_channel_selection": {
            "artifact_path": str(selection_path_value) if selection_path_value else None,
            "artifact_sha256": selection_hash,
            "n_selected": len(channel_indices) if channel_indices is not None else None,
            "probe_id": selection_artifact.get("probe_id") if selection_artifact else None,
            "selection_schema": selection_artifact.get("schema") if selection_artifact else None,
            "ensemble_groups": selection_artifact.get("ensemble_groups", {}) if selection_artifact else {},
        },
        "artifact": {
            "method": "lfp_streaming_qc_v1" if selection_artifact else None,
            "bad_eeg_channels": [],
            "bad_eeg_reasons": {},
        },
        "original_sfreq": float(original_sfreq),
        "target_sfreq_resolved": float(raw.info["sfreq"]),
        "sfreq_policy": sfreq_policy,
        "timings": preprocess_timings,
    }
    logger.info("Preprocessed NWB %s from %s: %d channels", file_path.name, series_path, len(channels["eeg"]))
    return PreprocessedSignals(signals=signals, sfreq=float(raw.info["sfreq"]), channels=channels, meta=meta)


def preprocess_file(file_path: Path, config: Mapping[str, Any]) -> PreprocessedSignals:
    """Preprocess a single file (EEG or fMRI) and return signals plus metadata.

    Args:
        file_path: Path to a raw EEG or fMRI file.
        config: Ingest configuration with ``preprocess`` and modality sections.

    Returns:
        :class:`PreprocessedSignals` with arrays, ``sfreq``, channel maps, and meta.
    """
    suffixes = "".join(file_path.suffixes).lower()
    source_datatype = _infer_path_datatype(file_path)
    if suffixes.endswith((".tsv", ".csv")) and source_datatype == "pupil":
        return preprocess_pupil_table(file_path, config)
    if mne is None:
        raise RuntimeError("mne is required for EEG preprocessing but is not installed.")
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        return preprocess_fmri(file_path, config)
    if suffixes.endswith(".nwb"):
        return preprocess_nwb(file_path, config)
    if suffixes.endswith(".hea"):
        return preprocess_wfdb(file_path, config)
    # We can infer dataset id from the path (no file I/O) and use it for
    # dataset-specific loading quirks before MNE touches the file.
    dataset_id_hint = _infer_dataset_id(file_path, config)
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, dict) else {}
    target_sfreq = preprocess_cfg.get("sfreq", 250) if isinstance(preprocess_cfg, Mapping) else 250
    notch_hz = preprocess_cfg.get("notch_hz", None) if isinstance(preprocess_cfg, Mapping) else None
    eeg_bandpass = preprocess_cfg.get("eeg_bandpass", [1, 45]) if isinstance(preprocess_cfg, Mapping) else [1, 45]
    meg_bandpass = preprocess_cfg.get("meg_bandpass", eeg_bandpass) if isinstance(preprocess_cfg, Mapping) else eeg_bandpass
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
        except OSError as exc:
            # EEGLAB .set files from migrated/renamed datasets may reference an
            # FDT binary with the old filename.  Attempt path repair and retry.
            if suffixes.endswith(".set") and "expected .fdt file format" in str(exc).lower():
                patched_set = _repair_eeglab_set_fdt_path(Path(file_path_for_mne))
                if patched_set is not None:
                    cleanup_after_load.append(patched_set)
                    raw = mne.io.read_raw(patched_set, preload=False, verbose=False)
                    logger.info("EEGLAB FDT repair succeeded for %s", file_path.name)
                else:
                    raise
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
    bdf_adapter_meta: Dict[str, Any] = {"applied": False}
    if suffixes.endswith(".bdf"):
        try:
            bdf_adapter_meta = apply_bdf_adapter_policy(
                raw,
                file_path,
                config,
                dataset_id_hint,
            )
        except Exception:
            logger.exception("BDF adapter policy failed for %s", file_path.name)
            raise
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
    interpolate_configured_bdf_bads(raw, bdf_adapter_meta)
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

    # Inject physiological channels from a companion BIDS *_physio.tsv.gz file,
    # if configured under preprocess.datasets.<id>.physio_tsv_inject.
    physio_inject_cfg = (dataset_preprocess_policy.get("physio_tsv_inject") or {}) if isinstance(dataset_preprocess_policy, Mapping) else {}
    if physio_inject_cfg.get("enabled"):
        try:
            _inject_physio_tsv_channels(raw, file_path, physio_inject_cfg)
        except Exception:
            logger.exception("physio_tsv_inject raised an unexpected error; continuing without physio channels")

    preprocess_timings["channel_prep"] = float(time.perf_counter() - t_channel_prep0)

    # Keep only biologically relevant channels before expensive resampling.
    # This reduces compute/memory without changing downstream feature semantics.
    t_pick0 = time.perf_counter()
    try:
        if source_datatype == "ecg":
            keep_picks = mne.pick_types(
                raw.info, eeg=True, ecg=True, eog=True, emg=True, resp=True, bio=True, misc=True, gsr=True
            )
        elif source_datatype == "meg":
            keep_picks = mne.pick_types(
                raw.info,
                meg=True,
                ref_meg=False,
                eeg=True,
                eog=True,
                ecg=True,
                emg=True,
                resp=True,
                bio=True,
                gsr=True,
                misc=True,
            )
        else:
            keep_picks = mne.pick_types(
                raw.info,
                eeg=True,
                eog=True,
                ecg=True,
                emg=True,
                seeg=True,
                ecog=True,
                resp=True,
                bio=True,
                gsr=True,
                misc=True,
            )
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
        """Internal helper: apply eog regression."""
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
        """Internal helper: apply ICA artifact removal.

        Supports EOG proxy channels (e.g. Fp1/Fp2) for datasets without
        dedicated EOG electrodes, and ECG component detection when a cardiac
        channel is present.
        """
        if ICA is None:
            logger.info("ICA not available; skipping ICA artifacts removal")
            return
        eeg_picks = mne.pick_types(r.info, eeg=True)
        if len(eeg_picks) == 0:
            logger.info("ICA skipped: no EEG channels")
            return
        n_comp = int(art_cfg.get("ica_n_components", min(20, len(eeg_picks))))
        random_state, _ = resolve_component_seed(
            config,
            dataset_id=dataset_id,
            fallback_seed=art_cfg.get("ica_random_state", 97),
            fallback_source="preprocess.artifacts.ica_random_state",
        )
        method = str(art_cfg.get("ica_method", "fastica"))
        fit_highpass_hz = float(art_cfg.get("ica_fit_highpass_hz", 1.0) or 1.0)
        eog_threshold = float(art_cfg.get("ica_eog_threshold", 3.0) or 3.0)
        ecg_threshold = float(art_cfg.get("ica_ecg_threshold", 3.0) or 3.0)
        max_exclude = int(art_cfg.get("ica_max_components_to_remove", 5) or 5)

        # EOG proxy: if no dedicated EOG channels, temporarily retype Fp1/Fp2
        # (or user-specified channels) as EOG for component detection.
        proxy_channels: List[str] = list(art_cfg.get("eog_proxy_channels", []) or [])
        retyped_as_eog: List[str] = []

        try:
            ica = ICA(n_components=n_comp, method=method, random_state=random_state, max_iter="auto")
            raw_fit = r.copy()
            if fit_highpass_hz > 0:
                raw_fit.filter(l_freq=fit_highpass_hz, h_freq=None, picks=eeg_picks, verbose=False)

            # Apply EOG proxy retyping on raw_fit before ICA fit
            if proxy_channels:
                eog_picks_existing = mne.pick_types(raw_fit.info, eog=True)
                if len(eog_picks_existing) == 0:
                    valid_proxy = [ch for ch in proxy_channels if ch in raw_fit.ch_names]
                    if valid_proxy:
                        raw_fit.set_channel_types(
                            {ch: "eog" for ch in valid_proxy}, on_unit_change="ignore"
                        )
                        retyped_as_eog = valid_proxy
                        logger.info("Using EOG proxy channels for ICA: %s", valid_proxy)

            ica.fit(raw_fit, picks=mne.pick_types(raw_fit.info, eeg=True), verbose=False)

            exclude_idx: List[int] = []

            # EOG component detection
            eog_picks_fit = mne.pick_types(raw_fit.info, eog=True)
            if len(eog_picks_fit) > 0:
                for eog_idx in eog_picks_fit:
                    eog_ch_name = raw_fit.ch_names[eog_idx]
                    try:
                        this_idx, _ = ica.find_bads_eog(
                            raw_fit, ch_name=eog_ch_name, threshold=eog_threshold
                        )
                        exclude_idx.extend(this_idx)
                    except Exception as eog_exc:
                        logger.debug("EOG component detection failed for %s: %s", eog_ch_name, eog_exc)

            # ECG component detection
            ecg_picks_fit = mne.pick_types(raw_fit.info, ecg=True)
            ecg_channel_cfg: Optional[str] = art_cfg.get("ecg_channel") or None
            ecg_ch_names: List[str] = []
            if ecg_channel_cfg and ecg_channel_cfg in raw_fit.ch_names:
                ecg_ch_names = [ecg_channel_cfg]
            elif len(ecg_picks_fit) > 0:
                ecg_ch_names = [raw_fit.ch_names[i] for i in ecg_picks_fit]
            for ecg_ch in ecg_ch_names:
                try:
                    this_idx, _ = ica.find_bads_ecg(
                        raw_fit, ch_name=ecg_ch, threshold=ecg_threshold
                    )
                    exclude_idx.extend(this_idx)
                except Exception as ecg_exc:
                    logger.debug("ECG component detection failed for %s: %s", ecg_ch, ecg_exc)

            # Apply safety cap
            unique_idx = list(sorted(set(exclude_idx)))
            ica.exclude = unique_idx[:max_exclude]
            ica.apply(r)
            logger.info(
                "Applied ICA (n_components=%d); excluded %d components (eog_proxy=%s, ecg=%s)",
                n_comp, len(ica.exclude), retyped_as_eog or "none", ecg_ch_names or "none",
            )
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
        if source_datatype == "ecg":
            notch_picks = mne.pick_types(
                raw.info,
                ecg=True,
                eog=True,
                emg=True,
                resp=True,
                bio=True,
                misc=True,
            )
        elif source_datatype == "meg":
            notch_picks = mne.pick_types(
                raw.info,
                meg=True,
                ref_meg=False,
                eeg=True,
                eog=True,
                ecg=True,
                emg=True,
                resp=True,
                bio=True,
            )
        else:
            notch_picks = mne.pick_types(
                raw.info,
                eeg=True,
                eog=True,
                ecg=True,
                emg=True,
                seeg=True,
                ecog=True,
                resp=True,
                bio=True,
            )
        if len(notch_picks) > 0:
            raw.notch_filter(freqs=notch_hz, picks=notch_picks, verbose=False)
        else:
            logger.info("Skipping notch filter for %s: no eligible channels after typing", file_path.name)
    preprocess_timings["notch"] = float(time.perf_counter() - t_notch0)

    # Treat intracranial EEG (sEEG/ECoG) as EEG for ingest purposes.
    # Many BIDS iEEG datasets (e.g. ds004100) label channels as "seeg"/"ecog",
    # which would otherwise result in empty EEG feature extraction.
    eeg_like_chans = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True) if source_datatype != "ecg" else np.array([], dtype=int)

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

    # Optional MEG bandpass for FIF-based M/EEG recordings.
    t_meg_bandpass0 = time.perf_counter()
    meg_chans = mne.pick_types(raw.info, meg=True, ref_meg=False)
    if len(meg_chans) > 0 and meg_bandpass is not None and len(meg_bandpass) == 2 and not skip_time_filters:
        try:
            raw.filter(l_freq=meg_bandpass[0], h_freq=meg_bandpass[1], picks=meg_chans, verbose=False)
        except Exception:
            data = raw.get_data(picks=meg_chans)
            filtered = mne.filter.filter_data(data, raw.info["sfreq"], meg_bandpass[0], meg_bandpass[1], verbose=False)
            raw._data[meg_chans, :] = filtered  # type: ignore[attr-defined]
    preprocess_timings["meg_bandpass"] = float(time.perf_counter() - t_meg_bandpass0)

    # Resample once for all channels if needed
    # (Note: we already resampled early to save memory, so this is now usually a no-op).
    t_resample_late0 = time.perf_counter()
    if raw.info["sfreq"] != target_sfreq and not (is_eeglab_epoched_concat and not eeglab_concat_policy.get("resample_concatenated_epochs", False)):
        raw.resample(target_sfreq, verbose=False)
    preprocess_timings["resample_late"] = float(time.perf_counter() - t_resample_late0)

    # Detect and drop obviously bad EEG-like channels (simple heuristics) BEFORE CAR reref.
    bad_channel_cfg = (config.get("robustness", {}).get("bad_channels", {}) if isinstance(config, dict) else {}) or {}
    t_badch0 = time.perf_counter()
    if source_datatype == "ecg":
        bad_info = {"bad_channels": [], "reasons": {}}
        bad_eeg_channels = []
    else:
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
    if source_datatype != "ecg" and bool(eeg_csd_cfg.get("enabled", False)):
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
    eeg_chans = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True) if source_datatype != "ecg" else np.array([], dtype=int)
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
    # EEG (including iEEG: seeg/ecog and scalp EEG transformed to CSD)
    # MNE returns SI units (V). Convert to µV so PSD band-powers are on a
    # numerically stable scale (~1 µV²/Hz instead of ~1e-12 V²/Hz).
    eeg_chans = (
        mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True, csd=True)
        if source_datatype != "ecg"
        else np.array([], dtype=int)
    )
    if len(eeg_chans) > 0:
        bad_segments_cfg = bdf_adapter_meta.get("bad_segments", {}) if isinstance(bdf_adapter_meta, Mapping) else {}
        mask_bad_segments = bool(
            isinstance(bad_segments_cfg, Mapping)
            and bad_segments_cfg.get("enabled")
            and bad_segments_cfg.get("available")
        )
        if mask_bad_segments:
            # Preserve the source time grid and mark BAD_ samples as NaN. The
            # EEG extractor rejects every window touching a non-finite sample,
            # avoiding synthetic adjacency across omitted artifact intervals.
            eeg_values = raw.get_data(picks=eeg_chans, reject_by_annotation="NaN")
            bad_sample_mask = ~np.isfinite(eeg_values).all(axis=0)
            bad_segments_cfg = dict(bad_segments_cfg)
            bad_segments_cfg["masked_samples"] = int(bad_sample_mask.sum())
            bad_segments_cfg["masked_duration_sec"] = float(bad_sample_mask.sum() / raw.info["sfreq"])
            bdf_adapter_meta["bad_segments"] = bad_segments_cfg
        else:
            eeg_values = raw.get_data(picks=eeg_chans)
        modality_signals["eeg"] = eeg_values * 1e6
        channels_dict["eeg"] = [raw.ch_names[i] for i in eeg_chans]

    # MEG
    meg_chans = mne.pick_types(raw.info, meg=True, ref_meg=False)
    if len(meg_chans) > 0:
        modality_signals["meg"] = raw.get_data(picks=meg_chans)
        channels_dict["meg"] = [raw.ch_names[i] for i in meg_chans]
        meg_mag_chans = mne.pick_types(raw.info, meg="mag", ref_meg=False)
        meg_grad_chans = mne.pick_types(raw.info, meg="grad", ref_meg=False)
        if len(meg_mag_chans) > 0:
            modality_signals["meg_mag"] = raw.get_data(picks=meg_mag_chans)
            channels_dict["meg_mag"] = [raw.ch_names[i] for i in meg_mag_chans]
        if len(meg_grad_chans) > 0:
            modality_signals["meg_grad"] = raw.get_data(picks=meg_grad_chans)
            channels_dict["meg_grad"] = [raw.ch_names[i] for i in meg_grad_chans]

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
    if len(ecg_chans) == 0 and source_datatype == "ecg" and len(raw.ch_names) > 0:
        ecg_chans = np.asarray([0], dtype=int)
    if len(ecg_chans) > 0:
        modality_signals["ecg"] = raw.get_data(picks=ecg_chans)
        channels_dict["ecg"] = [raw.ch_names[i] for i in ecg_chans]
    # Respiration
    resp_chans = mne.pick_types(raw.info, resp=True)
    if len(resp_chans) > 0:
        modality_signals["resp"] = raw.get_data(picks=resp_chans)
        channels_dict["resp"] = [raw.ch_names[i] for i in resp_chans]
    # PPG / pleth: prefer explicit bio channels or name-based matching.
    bio_chans = list(mne.pick_types(raw.info, bio=True))
    ppg_chans = [
        idx
        for idx, ch_name in enumerate(raw.ch_names)
        if re.search(r"(?i)(ppg|pleth|pulse|photopleth)", str(ch_name))
    ]
    if not ppg_chans and bio_chans:
        ppg_chans = [idx for idx in bio_chans if idx not in list(ecg_chans)]
    if not ppg_chans and source_datatype == "ecg":
        ppg_chans = [idx for idx in range(len(raw.ch_names)) if idx not in list(ecg_chans)][:1]
    if ppg_chans:
        modality_signals["ppg"] = raw.get_data(picks=ppg_chans)
        channels_dict["ppg"] = [raw.ch_names[i] for i in ppg_chans]
    # EDA / GSR: prefer MNE's native "gsr" channel type (available since
    # MNE >= 1.6). Also match by name among any remaining/misc-typed
    # channels for datasets whose config still types EDA as "misc" — this
    # keeps the extractor dataset-agnostic without requiring every config
    # to be migrated to the native type immediately.
    eda_chans = list(mne.pick_types(raw.info, gsr=True))
    if not eda_chans:
        eda_chans = [
            idx
            for idx, ch_name in enumerate(raw.ch_names)
            if re.search(r"(?i)(eda|gsr)", str(ch_name))
        ]
    if eda_chans:
        modality_signals["eda"] = raw.get_data(picks=eda_chans)
        channels_dict["eda"] = [raw.ch_names[i] for i in eda_chans]
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
        "bdf_adapter": bdf_adapter_meta,
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


_DEFAULT_NUISANCE_COLUMNS: Tuple[str, ...] = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "wm_mean",
    "csf_mean",
    "wm_pc1",
    "wm_pc2",
    "wm_pc3",
    "csf_pc1",
    "csf_pc2",
    "csf_pc3",
)


def _apply_fmri_nuisance_regression(
    region_array: np.ndarray,
    bold_path: Path,
    nuisance_cfg: Mapping[str, Any],
    tr: float,
) -> np.ndarray:
    """Regress lightweight motion + WM/CSF nuisance confounds out of regional BOLD.

    Looks for an "extended confounds" TSV next to `bold_path`, following the
    same "<bold-stem-minus-_bold><suffix>" convention as the existing
    FD-only confounds file (see mndm.pipeline.summary._resolve_confounds_path_for_bold
    and project/scripts/11_ds007216_fmri_lightweight_norm.py). On any failure
    to locate/parse the confounds file, logs a warning and returns
    `region_array` unchanged rather than failing the whole preprocessing run.
    """
    suffix = str(nuisance_cfg.get("confounds_suffix", "_desc-confoundsextended_timeseries.tsv"))
    columns = list(nuisance_cfg.get("columns") or _DEFAULT_NUISANCE_COLUMNS)

    bold_name = bold_path.name
    bold_base = bold_name.replace(".nii.gz", "").replace(".nii", "")
    conf_name = f"{bold_base[:-5]}{suffix}" if bold_base.endswith("_bold") else f"{bold_base}{suffix}"
    conf_path = bold_path.parent / conf_name

    if not conf_path.exists():
        logger.warning(
            "Nuisance regression enabled but confounds file not found at %s; skipping for %s",
            conf_path,
            bold_name,
        )
        return region_array

    try:
        conf_df = pd.read_csv(conf_path, sep="\t")
    except Exception as exc:
        logger.warning("Failed to read nuisance confounds %s: %s; skipping", conf_path, exc)
        return region_array

    use_cols = [c for c in columns if c in conf_df.columns]
    missing = [c for c in columns if c not in conf_df.columns]
    if missing:
        logger.warning("Nuisance confounds %s missing columns %s; using available subset", conf_path, missing)
    if not use_cols:
        logger.warning("No usable nuisance confound columns in %s; skipping regression", conf_path)
        return region_array

    conf_values = conf_df[use_cols].to_numpy(dtype=float)
    n_times = region_array.shape[1]
    if conf_values.shape[0] != n_times:
        n = min(conf_values.shape[0], n_times)
        logger.warning(
            "Nuisance confounds length %d != n_times %d for %s; truncating to %d",
            conf_values.shape[0],
            n_times,
            bold_name,
            n,
        )
        conf_values = conf_values[:n]
        region_array = region_array[:, :n]

    # nilearn.signal.clean requires finite confound values; the extended
    # confounds TSV can contain a leading NaN for near-zero-variance motion
    # frames (see script 21) or short leading/trailing PCA edge effects.
    conf_values = pd.DataFrame(conf_values).ffill().bfill().fillna(0.0).to_numpy()

    try:
        from nilearn.signal import clean as nilearn_clean

        cleaned = nilearn_clean(
            region_array.T,
            confounds=conf_values,
            detrend=True,
            standardize=False,
            t_r=tr if tr and np.isfinite(tr) and tr > 0 else None,
        )
    except Exception as exc:
        logger.warning("nilearn.signal.clean failed for %s: %s; skipping nuisance regression", bold_name, exc)
        return region_array

    logger.info(
        "Applied nuisance regression to %s using columns %s (n_times=%d)",
        bold_name,
        use_cols,
        region_array.shape[1],
    )
    return cleaned.T.astype(np.float32)


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

    # Optional lightweight motion + WM/CSF nuisance regression (config-gated;
    # see project/scripts/21_ds007216_fmri_confounds_extract.py for the
    # extended-confounds TSV producer). Runs before the bandpass filter below
    # so that confound-related variance is removed prior to any temporal
    # filtering, mirroring standard fMRIPrep-adjacent ordering.
    nuisance_cfg = fmri_cfg.get("nuisance_regression")
    if isinstance(nuisance_cfg, Mapping) and bool(nuisance_cfg.get("enabled", False)):
        region_array = _apply_fmri_nuisance_regression(
            region_array, file_path, nuisance_cfg, tr
        )

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



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
import ast
from dataclasses import dataclass
from fractions import Fraction
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
from .signal_support_provenance import (
    build_signal_support_provenance,
    callable_signature,
    file_sha256,
    finish_operation,
    map_preprocessed_window_to_source_samples,
    refresh_source_hash,
    quality_interval,
    original_raw_annotation_start,
    start_operation,
)

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


def _process_rss_mb() -> Optional[float]:
    """Current process RSS in MiB, or None if psutil is unavailable."""
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


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


class FnirsMetadataRejected(ValueError):
    """Raised when fNIRS metadata fails the structural or v0 input contract."""

    def __init__(self, report: Any):
        self.report = report.to_dict()
        super().__init__(f"fNIRS metadata rejected: {self.report}")


def _fnirs_index_value(record: Mapping[str, Any], key: str) -> Any:
    """Read one required fNIRS index value, treating CSV nulls as absent."""
    if key not in record:
        raise ValueError(f"fNIRS index record is missing required field {key!r}.")
    value = record[key]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "<na>"}:
        return None
    return value


def _fnirs_index_bool(record: Mapping[str, Any], key: str) -> bool:
    value = _fnirs_index_value(record, key)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(f"fNIRS index field {key!r} is missing or not a boolean.")


def _fnirs_index_path(root: Path, value: Any, field: str) -> Optional[Path]:
    if value is None:
        return None
    path = (root / str(value)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"fNIRS index path {field!r} escapes the dataset root.") from exc
    return path


def _fnirs_index_path_chain(root: Path, record: Mapping[str, Any], field: str) -> list[Path]:
    """Decode an ordered sidecar chain from an in-memory or CSV index record."""
    value = _fnirs_index_value(record, field)
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"fNIRS index field {field!r} is not a serialized path list.") from exc
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"fNIRS index field {field!r} must be an ordered list of paths.")
    paths = [_fnirs_index_path(root, item, field) for item in value]
    if not paths or any(path is None for path in paths):
        raise ValueError(f"fNIRS index field {field!r} is empty or incomplete.")
    return [path for path in paths if path is not None]


def _validate_fnirs_indexed_recording(
    file_path: Path,
    index_record: Optional[Mapping[str, Any]],
    dataset_root: Optional[Path],
) -> Any:
    """Validate one indexed SNIRF using the complete resolved sidecar contract."""
    if index_record is None or dataset_root is None:
        raise ValueError(
            "SNIRF input requires its BIDS index record and dataset root; "
            "metadata cannot be inferred safely from the signal path alone."
        )
    from .fnirs_metadata import validate_fnirs_recording

    root = Path(dataset_root).resolve()
    source_path = Path(file_path).resolve()
    try:
        source_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("fNIRS source file escapes the dataset root.") from exc
    json_chain = _fnirs_index_path_chain(root, index_record, "nirs_json_sidecars")
    coordsystem_chain = _fnirs_index_path_chain(root, index_record, "nirs_coordsystem_sidecars")
    sidecar_fields = {
        "channels_tsv": "nirs_channels_tsv",
        "optodes_tsv": "nirs_optodes_tsv",
        "events_tsv": "nirs_events_tsv",
    }
    sidecars = {
        field: _fnirs_index_path(root, _fnirs_index_value(index_record, key), key)
        for field, key in sidecar_fields.items()
    }
    ambiguity_fields = {
        "nirs_json_ambiguous": "nirs_json_sidecar_ambiguous",
        "channels_tsv_ambiguous": "nirs_channels_tsv_ambiguous",
        "events_tsv_ambiguous": "nirs_events_tsv_ambiguous",
        "optodes_tsv_ambiguous": "nirs_optodes_tsv_ambiguous",
        "coordsystem_json_ambiguous": "nirs_coordsystem_sidecar_ambiguous",
    }
    ambiguity = {name: _fnirs_index_bool(index_record, key) for name, key in ambiguity_fields.items()}
    report = validate_fnirs_recording(
        file_path,
        nirs_json=json_chain,
        channels_tsv=sidecars["channels_tsv"],
        optodes_tsv=sidecars["optodes_tsv"],
        coordsystem_json=coordsystem_chain,
        events_tsv=sidecars["events_tsv"],
        **ambiguity,
    )
    if not report.admissible:
        raise FnirsMetadataRejected(report)
    return report


def _preprocess_fnirs_file(
    file_path: Path,
    *,
    index_record: Optional[Mapping[str, Any]],
    dataset_root: Optional[Path],
) -> PreprocessedSignals:
    """Run the gated fNIRS metadata and optical path without EEG processing."""
    from . import preprocess_fnirs

    started = time.perf_counter()
    report = _validate_fnirs_indexed_recording(file_path, index_record, dataset_root)
    if mne is None:
        raise preprocess_fnirs.FnirsPreprocessingError(
            "preprocessing_dependency_unavailable",
            "MNE 1.6.1 is required to load the validated SNIRF recording.",
        )
    t_load = time.perf_counter()
    try:
        raw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
    except Exception as exc:
        raise preprocess_fnirs.FnirsPreprocessingError(
            "snirf_load_failed",
            f"Could not load the validated SNIRF file with MNE ({type(exc).__name__}).",
        ) from exc
    load_seconds = float(time.perf_counter() - t_load)
    t_transform = time.perf_counter()
    result = preprocess_fnirs.preprocess_fnirs(
        raw,
        validation=report,
        input_classification=str(report.classification),
        source_path=file_path,
    )
    transform_seconds = float(time.perf_counter() - t_transform)
    hb_raw = result.raw_haemoglobin
    channel_names = list(hb_raw.ch_names)
    channel_types = list(hb_raw.get_channel_types())
    signal = hb_raw.get_data()
    if signal.ndim != 2 or signal.shape != (len(channel_names), int(hb_raw.n_times)):
        raise preprocess_fnirs.FnirsPreprocessingError(
            "haemoglobin_output_shape_invalid",
            "MNE Hb output must retain channel-by-sample shape.",
        )
    meta: Dict[str, Any] = {
        "file": str(file_path),
        "source_format": "SNIRF",
        "modality": "fnirs",
        "fnirs": {
            "input_classification": report.classification,
            "output_state": "haemoglobin",
            "channel_names": channel_names,
            "channel_types": channel_types,
            "channel_units": result.provenance["haemoglobin_units"],
            "wavelengths_nm": list(report.summary.get("wavelengths_nm", [])),
            "metadata_validation": report.to_dict(),
            "diagnostics": dict(result.diagnostics),
            "preprocessing_provenance": dict(result.provenance),
            "scalp_coupling_index": dict(result.scalp_coupling_index),
            "computation_status": result.computation_status,
            "measurement_validity": result.measurement_validity,
            "claim_status": result.claim_status,
        },
        "signal_support_provenance": {
            "schema": "mndm.fnirs_signal_support.v1",
            "sampling_frequency_hz": float(hb_raw.info["sfreq"]),
            "sample_count": int(hb_raw.n_times),
            "time_start_s": report.summary["time_start_s"],
            "time_end_s": report.summary["time_end_s"],
            "median_sample_interval_s": report.summary["median_sample_interval_s"],
            "time_gap_sample_indices": list(report.summary.get("time_gap_sample_indices", [])),
            "resampling": "none",
            "filtering": "none",
        },
        "timings": {
            "load_snirf": load_seconds,
            "optical_preprocessing": transform_seconds,
            "total": float(time.perf_counter() - started),
        },
    }
    return PreprocessedSignals(
        signals={"fnirs": signal},
        sfreq=float(hb_raw.info["sfreq"]),
        channels={"fnirs": channel_names},
        meta=meta,
    )


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


def roi_ts_cache_lookup_args(
    file_path: Path,
    config: Mapping[str, Any],
    nifti_zooms: Optional[Sequence[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build ROI-TS cache key/identity args without materializing 4D BOLD."""
    from .pipeline.fmri_roi_cache import roi_ts_identity_payload
    from .pipeline.nuisance import DEFAULT_NUISANCE_COLUMNS, confounds_path_for_bold
    from .pipeline.tr_resolve import resolve_fmri_tr

    dataset_id = _infer_dataset_id(file_path, config)
    fmri_cfg = _resolve_fmri_config(config, dataset_id)
    atlas_path = fmri_cfg.get("atlas_path")
    if not atlas_path:
        return None
    atlas_path = Path(atlas_path)
    atlas_labels_raw = fmri_cfg.get("atlas_labels")
    atlas_labels_path = Path(atlas_labels_raw) if atlas_labels_raw else None
    tr_resolution = resolve_fmri_tr(
        file_path,
        config=config,
        nifti_zooms=nifti_zooms,
    )
    nuisance_cfg = fmri_cfg.get("nuisance_regression") if isinstance(fmri_cfg.get("nuisance_regression"), Mapping) else {}
    nuisance_enabled = bool(nuisance_cfg.get("enabled", False))
    confounds_suffix = str(nuisance_cfg.get("confounds_suffix") or "_desc-confoundsextended_timeseries.tsv")
    confounds_file = None
    if nuisance_enabled:
        candidate = confounds_path_for_bold(file_path, suffix=confounds_suffix)
        if candidate.exists():
            confounds_file = candidate
    columns = list(nuisance_cfg.get("columns") or DEFAULT_NUISANCE_COLUMNS)
    kwargs = {
        "bold_path": Path(file_path),
        "atlas_path": atlas_path,
        "tr_sec": float(tr_resolution.tr_sec),
        "tr_source": str(tr_resolution.tr_source),
        "nuisance_enabled": nuisance_enabled,
        "confounds_path": confounds_file,
        "space_flags": fmri_cfg,
        "atlas_labels_path": atlas_labels_path,
        "nuisance_columns": columns,
        "confounds_suffix": confounds_suffix,
    }
    identity = roi_ts_identity_payload(**kwargs)
    return {
        "dataset_id": dataset_id,
        "fmri_cfg": fmri_cfg,
        "tr_resolution": tr_resolution,
        "kwargs": kwargs,
        "identity": identity,
    }


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


def _resolve_quality_interval_policy(
    config: Mapping[str, Any], dataset_id: Optional[str]
) -> Dict[str, Any]:
    """Resolve opt-in handling for selected source quality intervals."""
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    policy_cfg = preprocess_cfg.get("quality_interval_policy", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(policy_cfg, Mapping):
        return {"enabled": False}
    merged: Dict[str, Any] = {key: policy_cfg[key] for key in policy_cfg if key != "datasets"}
    ds_overrides = policy_cfg.get("datasets", {})
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _resolve_raw_flatline_qc(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve the optional raw, pre-filter joint EEG flatline audit."""
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    qc_cfg = preprocess_cfg.get("raw_flatline_qc", {}) if isinstance(preprocess_cfg, Mapping) else {}
    if not isinstance(qc_cfg, Mapping):
        return {"enabled": False}
    merged: Dict[str, Any] = {key: qc_cfg[key] for key in qc_cfg if key != "datasets"}
    ds_overrides = qc_cfg.get("datasets", {})
    if dataset_id and isinstance(ds_overrides, Mapping):
        ds_cfg = ds_overrides.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged.update(ds_cfg)
    return merged


def _filter_skip_by_annotation(raw: Any, policy_record: Mapping[str, Any]) -> Optional[tuple[str, ...]]:
    """Return MNE skip prefixes for every BAD annotation under an opt-in policy."""
    if not bool(policy_record.get("enabled", False)):
        return None
    descriptions = getattr(getattr(raw, "annotations", None), "description", [])
    skip = ["edge", "bad_acq_skip"]
    seen = {value.casefold() for value in skip}
    for description in descriptions:
        text = str(description)
        if text.upper().startswith("BAD") and text.casefold() not in seen:
            skip.append(text)
            seen.add(text.casefold())
    return tuple(skip)


def _has_bad_raw_annotations(raw: Any) -> bool:
    """Whether the Raw contains any BAD_* annotation a fallback must not cross."""
    descriptions = getattr(getattr(raw, "annotations", None), "description", [])
    return any(str(description).upper().startswith("BAD") for description in descriptions)


def _apply_quality_interval_annotations(
    raw: Any,
    intervals: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    source_origin_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Attach selected source intervals as MNE BAD annotations, without masking data.

    ``source_origin_time_sec`` maps original-file-relative intervals to the
    annotation clock when this is called after a Raw has been cropped.
    """
    if not bool(policy.get("enabled", False)):
        return {"enabled": False, "status": "disabled", "annotations_added": 0}
    raw_kinds = policy.get("source_kinds", [])
    if isinstance(raw_kinds, str):
        raw_kinds = [raw_kinds]
    source_kinds = {str(value) for value in raw_kinds or []}
    description = str(policy.get("annotation_description", "BAD_acq_skip")).strip()
    if not source_kinds or not description.upper().startswith("BAD_"):
        raise ValueError(
            "preprocess.quality_interval_policy requires source_kinds and a BAD_* annotation_description"
        )

    selected: list[tuple[float, float]] = []
    for interval in intervals:
        if str(interval.get("source_kind", "")) not in source_kinds:
            continue
        if str(interval.get("clock", "original_raw_seconds")) != "original_raw_seconds":
            raise ValueError("quality interval annotation requires original_raw_seconds clock")
        start = float(interval["start_sec"])
        end = float(interval["end_sec"])
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError("quality interval has invalid source bounds")
        if end > start:
            selected.append((start, end))

    selected.sort()
    merged: list[list[float]] = []
    for start, end in selected:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    sfreq = float(raw.info["sfreq"])
    duration = float(raw.n_times / sfreq) if sfreq > 0 else 0.0
    first_time = float(getattr(raw, "first_time", 0.0) or 0.0)
    source_origin = first_time if source_origin_time_sec is None else float(source_origin_time_sec)
    if not np.isfinite([sfreq, duration, first_time, source_origin]).all() or sfreq <= 0:
        raise ValueError("quality interval annotation requires finite raw clock values and positive sfreq")
    current_source_start = first_time - source_origin
    current_source_end = current_source_start + duration
    added = 0
    for start, end in merged:
        clipped_start = max(current_source_start, start)
        clipped_end = min(current_source_end, end)
        if clipped_end <= clipped_start:
            continue
        # Map the original-file-relative source clock onto MNE's Raw annotation
        # clock, preserving a possible first_samp offset and prior crop.
        raw.annotations.append(
            float(clipped_start + source_origin),
            float(clipped_end - clipped_start),
            description,
        )
        added += 1
    if added:
        raw.annotations._sort()
    return {
        "enabled": True,
        "status": "applied" if added else "no_selected_intervals",
        "source_kinds": sorted(source_kinds),
        "annotation_description": description,
        "selected_source_interval_count": int(len(selected)),
        "annotations_added": int(added),
        "mask_bad_annotations": bool(policy.get("mask_bad_annotations", True)),
        "segment_resampling": bool(policy.get("segment_resampling", False)),
    }


def _joint_flatline_quality_intervals(
    data: np.ndarray,
    *,
    sfreq: float,
    first_time_sec: float,
    channel_names: Sequence[str],
    min_duration_sec: float = 1.0,
    raw_file: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Find intervals where every selected raw EEG channel is exactly unchanged."""
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        return []
    rate = float(sfreq)
    if not np.isfinite(rate) or rate <= 0:
        return []
    minimum_duration = float(min_duration_sec)
    if not np.isfinite(minimum_duration) or minimum_duration < 0:
        raise ValueError("min_duration_sec must be finite and nonnegative")
    adjacent_finite = np.isfinite(values[:, :-1]).all(axis=0) & np.isfinite(values[:, 1:]).all(axis=0)
    unchanged = adjacent_finite & np.all(values[:, :-1] == values[:, 1:], axis=0)
    edges = np.diff(np.r_[False, unchanged, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    min_samples = max(1, int(np.ceil(minimum_duration * rate)))
    results: list[Dict[str, Any]] = []
    for start_diff, stop_diff in zip(starts.tolist(), stops.tolist()):
        sample_start = int(start_diff)
        sample_end = int(stop_diff + 1)
        if sample_end - sample_start < min_samples:
            continue
        start_sec = float(first_time_sec) + sample_start / rate
        duration_sec = (sample_end - sample_start) / rate
        results.append(
            quality_interval(
                source_kind="raw_joint_eeg_flatline",
                start_sec=start_sec,
                duration_sec=duration_sec,
                raw_file=raw_file,
                raw_sample_start=sample_start,
                raw_sample_end_exclusive=sample_end,
                sfreq_hz=rate,
                channel_names=[str(name) for name in channel_names],
                detection="exact_all_channels_adjacent_sample_equality",
                min_duration_sec=minimum_duration,
                signal_stage="raw_before_temporal_preprocessing",
            )
        )
    return results


def _split_raw_at_bad_annotation_boundaries(
    raw: Any,
    *,
    target_sfreq: float,
    annotation_description: str = "BAD_acq_skip",
) -> tuple[Any, Dict[str, Any]]:
    """Split an MNE Raw at selected BAD annotation edges before FFT resampling.

    MNE Raw.resample resamples each underlying Raw segment independently, but
    does not split at annotations. This helper creates those segment boundaries
    so samples on opposite sides of a declared gap cannot mix during resampling.
    To preserve the downstream time grid, each boundary must align to the target
    sample grid; otherwise the opt-in operation fails closed.
    """
    sfreq = float(raw.info["sfreq"])
    target = float(target_sfreq)
    n_samples = int(raw.n_times)
    report: Dict[str, Any] = {
        "enabled": True,
        "status": "no_matching_intervals",
        "segment_count_before": int(len(getattr(raw, "_raw_lengths", [n_samples]))),
        "segment_count_after": int(len(getattr(raw, "_raw_lengths", [n_samples]))),
        "boundary_samples": [],
    }
    if not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(target) or target <= 0:
        raise ValueError("segment-aware resampling requires positive finite sampling frequencies")
    if abs(sfreq - target) <= max(1e-12, target * 1e-12):
        report["status"] = "skipped_target_rate_matches_input"
        return raw, report

    first_time = float(getattr(raw, "first_time", 0.0) or 0.0)
    cuts = {0, n_samples}
    matched = 0
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        if str(description).casefold() != annotation_description.casefold():
            continue
        local_start = float(onset) - first_time
        local_end = local_start + float(duration)
        start_idx = int(np.clip(np.rint(local_start * sfreq), 0, n_samples))
        end_idx = int(np.clip(np.rint(local_end * sfreq), 0, n_samples))
        if end_idx <= start_idx:
            continue
        for sample_idx in (start_idx, end_idx):
            target_position = (sample_idx / sfreq) * target
            if abs(target_position - round(target_position)) > 1e-7:
                raise ValueError(
                    "BAD annotation boundary does not align to the target sample grid; "
                    "refusing continuous FFT resampling across a time-shifting segment boundary"
                )
            if 0 < sample_idx < n_samples:
                cuts.add(sample_idx)
        matched += 1

    if len(cuts) <= 2 or matched == 0:
        return raw, report

    ordered_cuts = sorted(cuts)
    pieces = [
        raw.copy().crop(
            tmin=float(raw.times[start_idx]),
            tmax=float(raw.times[end_idx - 1]),
            include_tmax=True,
            verbose="ERROR",
        )
        for start_idx, end_idx in zip(ordered_cuts[:-1], ordered_cuts[1:])
        if end_idx > start_idx
    ]
    segmented = mne.concatenate_raws(pieces, preload=True, verbose="ERROR")
    if int(segmented.n_times) != n_samples:
        raise RuntimeError(
            f"segment-aware resampling changed the input sample count ({n_samples} -> {segmented.n_times})"
        )
    report.update(
        {
            "status": "segmented",
            "segment_count_after": int(len(getattr(segmented, "_raw_lengths", pieces))),
            "boundary_samples": [int(value) for value in ordered_cuts[1:-1]],
            "matching_annotation_count": int(matched),
        }
    )
    return segmented, report


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


def _nwb_series_timebase_summary(
    series: Any,
    *,
    source_sfreq: float,
    start_sample: int = 0,
    stop_sample: int | None = None,
) -> dict[str, Any]:
    """Summarize the selected NWB sample clock and reject hidden time gaps.

    The experimental overlap-trim path currently emits one continuous RawArray.
    It therefore accepts only one monotone timestamp segment. A gap is reported
    with its sample boundary instead of being flattened into that time grid.
    """
    n_samples = int(series.data.shape[0])
    start = max(0, int(start_sample))
    stop = min(n_samples, int(stop_sample)) if stop_sample is not None else n_samples
    if stop <= start:
        raise ValueError("NWB timebase selection contains no samples.")
    sample_rate = float(source_sfreq)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("NWB timebase requires a finite positive sample rate.")
    nominal_dt = 1.0 / sample_rate
    timestamps = getattr(series, "timestamps", None)

    if timestamps is not None:
        if len(timestamps) != n_samples:
            raise ValueError(
                f"NWB timestamp count {len(timestamps)} does not match data sample count {n_samples}."
            )
        first_time = float(timestamps[start])
        last_time = float(timestamps[stop - 1])
        if not np.isfinite(first_time) or not np.isfinite(last_time):
            raise ValueError("NWB selected timebase has a non-finite endpoint timestamp.")
        previous: float | None = None
        gap_indices: list[int] = []
        gap_count = 0
        invalid_indices: list[int] = []
        block_size = 1_000_000
        for block_start in range(start, stop, block_size):
            block_stop = min(stop, block_start + block_size)
            block = np.asarray(timestamps[block_start:block_stop], dtype=np.float64)
            if previous is not None:
                block = np.concatenate(([previous], block))
                right_sample_start = block_start
            else:
                right_sample_start = block_start + 1
            differences = np.diff(block)
            right_sample_indices = np.arange(
                right_sample_start,
                right_sample_start + len(differences),
                dtype=np.int64,
            )
            invalid = ~np.isfinite(differences) | (differences <= 0)
            gap_mask = np.isfinite(differences) & (differences > nominal_dt * 1.5)
            invalid_positions = np.flatnonzero(invalid)
            gap_positions = np.flatnonzero(gap_mask)
            invalid_indices.extend(
                int(right_sample_indices[i])
                for i in invalid_positions[: max(0, 10 - len(invalid_indices))]
            )
            gap_count += int(gap_positions.size)
            gap_indices.extend(
                int(right_sample_indices[i]) for i in gap_positions[: max(0, 10 - len(gap_indices))]
            )
            if len(block):
                previous = float(block[-1])
        if invalid_indices:
            raise ValueError(
                "NWB timestamp sequence is non-finite or non-monotone at sample indices "
                f"{invalid_indices}."
            )
        if gap_count:
            raise ValueError(
                "The experimental continuous NWB path found "
                f"{gap_count} timestamp discontinuity/discontinuities larger than "
                f"1.5 nominal sample intervals; first right-sample indices: {gap_indices}. "
                "Preserve and process the segments separately before enabling this path."
            )
        clock_source = "explicit_timestamps"
    else:
        starting_time = getattr(series, "starting_time", None)
        rate = getattr(series, "rate", None)
        if starting_time is None or rate is None:
            raise ValueError("NWB timebase needs explicit timestamps or starting_time plus rate.")
        try:
            origin = float(starting_time[()])
        except (TypeError, IndexError, KeyError):
            origin = float(starting_time)
        rate = float(rate)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("NWB starting_time/rate clock has an invalid rate.")
        first_time = origin + start / rate
        last_time = origin + (stop - 1) / rate
        clock_source = "starting_time_and_rate"

    return {
        "clock_source": clock_source,
        "source_sample_rate_hz": sample_rate,
        "nominal_sample_interval_sec": nominal_dt,
        "discontinuity_threshold_sec": 1.5 * nominal_dt,
        "selected_sample_start": start,
        "selected_sample_stop_exclusive": stop,
        "selected_sample_count": stop - start,
        "selected_time_start_sec": first_time,
        "selected_time_stop_sec": last_time,
        "segment_count": 1,
        "segments": [
            {
                "segment_id": "segment-0000",
                "sample_start": start,
                "sample_stop_exclusive": stop,
                "time_start_sec": first_time,
                "time_stop_sec": last_time,
            }
        ],
    }


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


def _nwb_chunked_downsample_overlap_trim(
    series: Any,
    *,
    source_sfreq: float,
    target_sfreq: float,
    channel_indices: Sequence[int],
    chunk_sec: float,
    start_sample: int = 0,
    stop_sample: int | None = None,
) -> np.ndarray:
    """Downsample bounded NWB chunks with context and trim to a continuous grid.

    ``resample_poly`` applies a finite symmetric FIR and pads each call at its
    edges. Resampling isolated chunks therefore creates artificial seams. This
    variant adds the default FIR half-support as context on both sides, keeps
    context starts aligned to the cropped source grid, and retains only each
    chunk's central output samples. The original independent-chunk behavior is
    left available through :func:`_nwb_chunked_downsample` for reproducibility.
    """
    from scipy.signal import resample_poly

    total_samples = int(series.data.shape[0])
    source_start = max(0, int(start_sample))
    stop = min(total_samples, int(stop_sample)) if stop_sample is not None else total_samples
    stop = max(source_start, stop)
    ratio = float(source_sfreq) / float(target_sfreq)
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("NWB chunked downsampling requires an integer source/target rate ratio.")
    if not np.isfinite(chunk_sec) or float(chunk_sec) <= 0:
        raise ValueError("NWB chunked downsampling requires chunk_sec > 0.")

    down = int(round(ratio))
    if down < 1:
        raise ValueError("NWB chunked downsampling requires target_sfreq <= source_sfreq.")
    chunk_samples = max(down, int(round(float(chunk_sec) * source_sfreq / down)) * down)
    # scipy.signal.resample_poly's default FIR has 10 * max(up, down) taps of
    # half-support on the upsampled input grid. Here up=1, so this is also the
    # source-sample support; round up to retain phase alignment at every chunk.
    context_samples = 10 * down
    total_source = stop - source_start
    total_output = (total_source + down - 1) // down
    chunks: list[np.ndarray] = []

    for first in range(source_start, stop, chunk_samples):
        chunk_stop = min(stop, first + chunk_samples)
        desired_context_start = max(source_start, first - context_samples)
        context_start = source_start + ((desired_context_start - source_start) // down) * down
        context_stop = min(stop, chunk_stop + context_samples)
        if context_stop < stop:
            context_stop = source_start + ((context_stop - source_start + down - 1) // down) * down
            context_stop = min(stop, context_stop)

        part = _nwb_data_to_channels_by_samples(
            series,
            start_sample=context_start,
            stop_sample=context_stop,
            channel_indices=channel_indices,
        )
        resampled = resample_poly(part, up=1, down=down, axis=1).astype(np.float64, copy=False)
        output_start = (first - source_start) // down
        output_stop = min(total_output, (chunk_stop - source_start + down - 1) // down)
        output_count = output_stop - output_start
        local_output_start = (first - context_start) // down
        retained = resampled[:, local_output_start : local_output_start + output_count]
        if retained.shape[1] != output_count:
            raise RuntimeError(
                "NWB overlap-trim resampling produced an unexpected output length "
                f"({retained.shape[1]} != {output_count})."
            )
        chunks.append(retained)

    result = np.concatenate(chunks, axis=1) if chunks else np.empty((len(channel_indices), 0), dtype=np.float64)
    if result.shape[1] != total_output:
        raise RuntimeError(
            f"NWB overlap-trim resampling produced {result.shape[1]} samples; expected {total_output}."
        )
    return result


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


def _causal_polyphase_resample(raw: Any, target_sfreq: float) -> tuple[Any, dict[str, Any]]:
    """Resample one continuous RawArray with a finite, causal FIR dependency.

    Unlike ``resample_poly``'s centered default, this deliberately retains the
    polyphase filter delay. It is intended for bounded opt-in pilots only.
    """
    from scipy.signal import firwin, upfirdn

    source_sfreq = float(raw.info["sfreq"])
    target_sfreq = float(target_sfreq)
    if not np.isfinite(source_sfreq) or not np.isfinite(target_sfreq) or min(source_sfreq, target_sfreq) <= 0:
        raise ValueError("causal polyphase resampling requires positive finite sample rates")
    if len(getattr(raw, "annotations", [])):
        raise ValueError("causal polyphase resampling refuses annotated boundaries")
    data = np.asarray(raw.get_data(), dtype=np.float64)
    if data.ndim != 2 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("causal polyphase resampling requires a finite, nonempty continuous array")

    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(100_000)
    up, down = int(ratio.numerator), int(ratio.denominator)
    realized_sfreq = source_sfreq * up / down
    if abs(realized_sfreq - target_sfreq) > max(1e-10, abs(target_sfreq) * 1e-12):
        raise ValueError("requested sample-rate ratio cannot be represented safely as a rational polyphase ratio")

    if up == down:
        resampled = data
        taps = np.array([1.0], dtype=np.float64)
    else:
        max_rate = max(up, down)
        half_len = 10 * max_rate
        taps = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)).astype(np.float64)
        taps *= up
        resampled = upfirdn(taps, data, up=up, down=down, axis=-1)
        output_count = (data.shape[-1] * up + down - 1) // down
        resampled = np.asarray(resampled[..., :output_count], dtype=np.float64)

    channel_types = list(raw.get_channel_types())
    info = mne.create_info(ch_names=list(raw.ch_names), sfreq=target_sfreq, ch_types=channel_types)
    info["bads"] = list(raw.info.get("bads", []))
    result = mne.io.RawArray(resampled, info, first_samp=int(getattr(raw, "first_samp", 0)), verbose=False)
    coefficient_hash = hashlib.sha256(np.asarray(taps, dtype="<f8").tobytes()).hexdigest()
    description = {
        "method": "scipy.signal.upfirdn_causal_fir",
        "up": up,
        "down": down,
        "tap_count": int(taps.size),
        "coefficient_sha256": coefficient_hash,
        "group_delay_input_samples": float((taps.size - 1) / (2 * up)),
        "group_delay_seconds": float((taps.size - 1) / (2 * up * source_sfreq)),
        "source_sfreq_hz": source_sfreq,
        "output_sfreq_hz": target_sfreq,
        "output_sample_count": int(resampled.shape[-1]),
        "phase": "causal_uncompensated",
        "boundary_extension": "zero_before_record_start; no future samples",
    }
    return result, description


def _identity_sample_grid_support_spec(raw: Any) -> dict[str, Any]:
    """Describe an unchanged, verified sample grid as a causal identity map."""
    sfreq = float(raw.info["sfreq"])
    if not np.isfinite(sfreq) or sfreq <= 0 or int(raw.n_times) < 1:
        raise ValueError("identity sample-grid support requires a positive clock and nonempty recording")
    return {
        "method": "identity_sample_grid",
        "up": 1,
        "down": 1,
        "tap_count": 1,
        "coefficient_sha256": hashlib.sha256(np.asarray([1.0], dtype="<f8").tobytes()).hexdigest(),
        "group_delay_input_samples": 0.0,
        "group_delay_seconds": 0.0,
        "source_sfreq_hz": sfreq,
        "output_sfreq_hz": sfreq,
        "output_sample_count": int(raw.n_times),
        "phase": "identity",
        "boundary_extension": "none; source and processed sample grids are identical",
    }


def _minimum_phase_eeg_filter_spec(raw: Any, picks: Sequence[int], bandpass: Sequence[float]) -> dict[str, Any]:
    """Describe the exact MNE FIR used by an opt-in causal finite-support path."""
    if mne is None:
        raise RuntimeError("MNE is required for the causal EEG filter")
    data = raw.get_data(picks=list(picks))
    taps = mne.filter.create_filter(
        data=data,
        sfreq=float(raw.info["sfreq"]),
        l_freq=float(bandpass[0]),
        h_freq=float(bandpass[1]),
        method="fir",
        phase="minimum",
        fir_design="firwin",
        fir_window="hamming",
        filter_length="auto",
        verbose=False,
    )
    taps = np.asarray(taps, dtype=np.float64)
    return {
        "method": "mne.filter.filter_data_fir",
        "phase": "minimum",
        "pad": "constant",
        "tap_count": int(taps.size),
        "coefficient_sha256": hashlib.sha256(np.asarray(taps, dtype="<f8").tobytes()).hexdigest(),
        "bandpass_hz": [float(bandpass[0]), float(bandpass[1])],
        "sfreq_hz": float(raw.info["sfreq"]),
        "support_interpretation": "output k depends on input [max(0,k-tap_count+1),k]",
    }


def _apply_minimum_phase_eeg_filter(
    raw: Any,
    picks: Sequence[int],
    bandpass: Sequence[float],
    *,
    skip_by_annotation: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Apply and describe one minimum-phase FIR with matching finite support."""
    spec = _minimum_phase_eeg_filter_spec(raw, picks, bandpass)
    kwargs: dict[str, Any] = {}
    if skip_by_annotation is not None:
        kwargs["skip_by_annotation"] = tuple(str(value) for value in skip_by_annotation)
    raw.filter(
        l_freq=float(bandpass[0]),
        h_freq=float(bandpass[1]),
        picks=list(picks),
        method="fir",
        phase="minimum",
        fir_design="firwin",
        fir_window="hamming",
        filter_length="auto",
        pad="constant",
        verbose=False,
        **kwargs,
    )
    return spec


def _verify_uniform_edf_sample_grid(raw: Any, source_path: Optional[Path] = None) -> dict[str, Any]:
    """Verify one uniform, continuous EDF source grid without EDF+D gaps."""
    extras = getattr(raw, "_raw_extras", None)
    if not isinstance(extras, (list, tuple)) or len(extras) != 1 or not isinstance(extras[0], Mapping):
        raise ValueError("causal_finite EDF mode requires one inspectable EDF signal header")
    header = extras[0]
    edf_path = Path(source_path) if source_path is not None else None
    if edf_path is None:
        filenames = getattr(raw, "filenames", None)
        if isinstance(filenames, (list, tuple)) and len(filenames) == 1 and filenames[0]:
            edf_path = Path(str(filenames[0]))
    if edf_path is None or not edf_path.is_file():
        raise ValueError("causal_finite EDF mode requires the original EDF header path for continuity verification")
    try:
        with edf_path.open("rb") as stream:
            fixed_header = stream.read(256)
        if len(fixed_header) != 256:
            raise ValueError("EDF fixed header is truncated")
        reserved = fixed_header[192:236].decode("ascii", errors="strict").strip(" \x00").upper()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("causal_finite EDF mode could not verify the EDF continuity marker") from exc
    if reserved.startswith("EDF+D"):
        raise ValueError("causal_finite EDF mode refuses EDF+D recordings until discontinuous clocks are mapped explicitly")
    if reserved and reserved != "EDF+C":
        raise ValueError(f"causal_finite EDF mode refuses unrecognized EDF continuity marker: {reserved!r}")
    annotations = getattr(raw, "annotations", None)
    if annotations is None:
        raise ValueError("causal_finite EDF mode requires inspectable MNE annotation metadata")
    descriptions = getattr(annotations, "description", None)
    if descriptions is None:
        raise ValueError("causal_finite EDF mode requires an MNE annotation description list")
    bad_acq_skip = [str(value) for value in descriptions if str(value).strip().upper().startswith("BAD_ACQ_SKIP")]
    if bad_acq_skip:
        raise ValueError("causal_finite EDF mode refuses recordings with MNE BAD_ACQ_SKIP annotations")
    edge_boundaries = [str(value) for value in descriptions if str(value).strip().upper().startswith("EDGE")]
    if edge_boundaries:
        raise ValueError("causal_finite EDF mode refuses MNE EDGE annotations without a qualified boundary map")
    bad_boundaries = [str(value) for value in descriptions if str(value).strip().upper().startswith("BAD BOUNDARY")]
    if bad_boundaries:
        raise ValueError("causal_finite EDF mode refuses MNE BAD boundary annotations without a qualified boundary map")
    try:
        samples_per_record = np.asarray(header["n_samps"], dtype=np.int64).reshape(-1)
        max_samples = int(header["max_samp"])
        record_count = int(header["n_records"])
        record_lengths = np.asarray(header["record_length"], dtype=float).reshape(-1)
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("causal_finite EDF mode requires complete per-channel sample-grid metadata") from exc
    if (
        samples_per_record.size != len(raw.ch_names)
        or samples_per_record.size == 0
        or np.any(samples_per_record <= 0)
        or np.any(samples_per_record != samples_per_record[0])
        or record_lengths.size == 0
        or np.any(~np.isfinite(record_lengths))
        or np.any(record_lengths <= 0)
        or np.any(record_lengths != record_lengths[0])
        or max_samples != int(samples_per_record[0])
        or record_count <= 0
        or n_times != record_count * max_samples
        or not np.isfinite(sfreq)
        or sfreq <= 0
        or not np.isclose(record_lengths[0], max_samples / sfreq, rtol=0.0, atol=max(1e-9, 1e-9 / sfreq))
    ):
        raise ValueError("causal_finite EDF mode refuses heterogeneous or inconsistent channel sample grids")
    return {
        "status": "verified_uniform_edf_grid",
        "continuity_status": "verified_continuous",
        "continuity_marker": reserved or "EDF",
        "continuity_verification": "EDF+D rejected; EDF/EDF+C record grid checked; no BAD_ACQ_SKIP, BAD boundary, or EDGE boundary annotations",
        "boundary_annotation_check": {
            "status": "verified_absent",
            "rejected_prefixes": ["BAD_ACQ_SKIP", "BAD BOUNDARY", "EDGE"],
        },
        "channel_count": int(samples_per_record.size),
        "samples_per_record_by_channel": samples_per_record.tolist(),
        "records": record_count,
        "samples_per_record": max_samples,
        "record_duration_seconds": float(record_lengths[0]),
        "recorded_continuous_duration_seconds": float(record_count * record_lengths[0]),
        "sample_rate_hz": sfreq,
        "sample_count": n_times,
        "verification": "all signal channels share one samples-per-record value and exact record-count grid",
    }


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

    # WFDB's signal matrix is stored in a companion file (typically .mat or
    # .dat), while the .hea file carries the channel and clock metadata. Hash
    # the header and the referenced signal payload before reading samples so
    # the later QC sidecar can identify the exact source pair.
    signal_support_provenance = build_signal_support_provenance(
        Path(file_path), source_datatype="eeg"
    )
    support_source = signal_support_provenance.get("source", {})
    header_file_names: list[str] = []
    is_multirecord = False
    source_sample_grid_supported = False
    source_sample_grid_reason = "wfdb_header_unverified"
    rdheader = getattr(wfdb, "rdheader", None)
    if callable(rdheader):
        try:
            header = rdheader(str(file_path.with_suffix("")))
            segment_names = getattr(header, "seg_name", None)
            is_multirecord = segment_names is not None
            if is_multirecord:
                source_sample_grid_reason = "wfdb_multirecord_segment_mapping_not_supported"
            else:
                raw_header_file_names = getattr(header, "file_name", []) or []
                if isinstance(raw_header_file_names, str):
                    raw_header_file_names = [raw_header_file_names]
                header_file_names = list(dict.fromkeys(str(name) for name in raw_header_file_names if str(name).strip()))
                source_sample_grid_reason = "wfdb_signal_payload_identity_unavailable"
            signal_support_provenance["wfdb_header_clock"] = {
                "base_date": str(getattr(header, "base_date", "") or "") or None,
                "base_time": str(getattr(header, "base_time", "") or "") or None,
                "recording_start_sample": 0,
                "sample_rate_hz": float(getattr(header, "fs", 0.0) or 0.0),
                "sample_count": int(getattr(header, "sig_len", 0) or 0),
                "channel_count": int(getattr(header, "n_sig", 0) or 0),
                "parser": "wfdb.rdheader",
            }
            from .pipeline.time_reference import parse_wfdb_header_clocks

            signal_support_provenance["recording_time_reference"] = parse_wfdb_header_clocks(Path(file_path))
        except Exception as exc:
            # rdrecord remains the authoritative load path. Preserve a clear
            # provenance gap if the preliminary header read is unavailable.
            signal_support_provenance["wfdb_header_clock"] = {
                "status": "unavailable",
                "reason": f"header_read_failed:{type(exc).__name__}",
                "parser": "wfdb.rdheader",
            }
            source_sample_grid_reason = "wfdb_header_unverified"
            signal_support_provenance["recording_time_reference"] = {
                "status": "unavailable",
                "reason": f"header_time_parse_failed:{type(exc).__name__}",
            }

    payload_files: list[dict[str, Any]] = []
    source_parent = Path(file_path).resolve().parent
    for name in header_file_names:
        candidate = (source_parent / name).resolve()
        try:
            candidate.relative_to(source_parent)
        except ValueError:
            payload_files.append({
                "path": str(candidate),
                "exists": False,
                "hash_status": "refused_path_outside_header_directory",
            })
            continue
        identity: dict[str, Any] = {"path": str(candidate), "exists": candidate.is_file()}
        if candidate.is_file():
            try:
                stat = candidate.stat()
                identity.update({
                    "size_bytes": int(stat.st_size),
                    "sha256": file_sha256(candidate),
                    "hash_status": "captured_before_read",
                })
            except Exception as exc:
                identity.update({"hash_status": "failed", "hash_error": str(exc)})
        else:
            identity["hash_status"] = "missing"
        payload_files.append(identity)
    if (
        not is_multirecord
        and payload_files
        and all(
            item.get("exists") is True
            and item.get("hash_status") == "captured_before_read"
            and bool(item.get("sha256"))
            for item in payload_files
        )
    ):
        source_sample_grid_supported = True
        source_sample_grid_reason = "single_record_source_payload_hashed_before_read"
    signal_support_provenance["source_sample_grid_supported"] = source_sample_grid_supported
    signal_support_provenance["source_sample_grid_reason"] = source_sample_grid_reason
    support_source["associated_signal_files"] = payload_files

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
    temporal_mode = str(preprocess_cfg.get("temporal_support_mode", "default")).strip().lower()
    causal_finite_mode = temporal_mode == "causal_finite"
    if temporal_mode not in {"default", "causal_finite"}:
        raise ValueError(f"unsupported WFDB temporal_support_mode: {temporal_mode}")
    if causal_finite_mode:
        header_clock = signal_support_provenance.get("wfdb_header_clock", {})
        try:
            header_sfreq = float(header_clock["sample_rate_hz"])
            header_sample_count = int(header_clock["sample_count"])
            header_channel_count = int(header_clock["channel_count"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("causal_finite WFDB mode requires a complete parsed header clock") from exc
        if (
            not np.isfinite(header_sfreq)
            or abs(header_sfreq - fs) > max(1e-9, fs * 1e-12)
            or header_sample_count != int(data.shape[0])
            or header_channel_count != int(data.shape[1])
        ):
            raise ValueError("causal_finite WFDB mode refuses a header grid that differs from the loaded record array")

    original_first_samp = int(getattr(raw, "first_samp", 0))
    original_first_time_sec = float(getattr(raw, "first_time", 0.0) or 0.0)
    signal_support_provenance["input_sfreq_hz"] = float(fs)
    signal_support_provenance["raw_clock_before_crop"] = {
        "sfreq_hz": float(fs),
        "first_time_sec": original_first_time_sec,
        "first_samp": original_first_samp,
        "n_times": int(data.shape[0]),
        "duration_sec": float(data.shape[0] / fs),
    }


    # Optional dataset-specific channel typing overrides.
    _apply_channel_typing(raw, dataset_id, config)

    crop_cfg = _resolve_crop_config(config, dataset_id)
    if causal_finite_mode:
        resample_cfg_for_mode = preprocess_cfg.get("resample", {})
        resample_method = (
            str(resample_cfg_for_mode.get("method", "")).strip().lower()
            if isinstance(resample_cfg_for_mode, Mapping)
            else ""
        )
        if not source_sample_grid_supported or is_multirecord:
            raise ValueError("causal_finite WFDB mode requires a verified single-record source sample grid")
        if crop_cfg:
            raise ValueError("causal_finite WFDB mode currently refuses configured crops")
        if len(getattr(raw, "annotations", [])):
            raise ValueError("causal_finite WFDB mode refuses annotated boundaries")
        if resample_method != "causal_polyphase":
            raise ValueError("causal_finite WFDB mode requires resample.method=causal_polyphase")
        if str(preprocess_cfg.get("eeg_bandpass_phase", "")).strip().lower() != "minimum":
            raise ValueError("causal_finite WFDB mode requires eeg_bandpass_phase=minimum")
        if preprocess_cfg.get("notch_hz") is not None:
            raise ValueError("causal_finite WFDB mode currently requires notch_hz=null")
        configured_bandpass = preprocess_cfg.get("eeg_bandpass", [1, 45])
        if not (isinstance(configured_bandpass, (list, tuple)) and len(configured_bandpass) == 2):
            raise ValueError("causal_finite WFDB mode requires a two-edge EEG bandpass")
    crop_record: Dict[str, Any] = {
        "requested": None,
        "applied": False,
        "actual_tmin_sec": original_first_time_sec,
        "actual_tmax_sec": original_first_time_sec + float(raw.n_times) / float(raw.info["sfreq"]),
        "actual_tmin_from_source_start_sec": 0.0,
        "reason": "not_configured",
    }
    if crop_cfg:
        # NWB data were sliced before materialisation, so crop relative to the
        # in-memory segment rather than applying start_sec twice.
        tmin = 0.0
        tmax_val = crop_cfg.get("stop_sec", None)
        start_val = float(crop_cfg.get("start_sec", 0) or 0.0)
        tmax = float(tmax_val) - start_val if tmax_val is not None else None
        crop_record["requested"] = {"tmin_sec": start_val, "tmax_sec": float(tmax_val) if tmax_val is not None else None}
        crop_record["reason"] = "configured"
        crop_operation = start_operation(
            signal_support_provenance,
            "crop",
            input_sfreq=float(raw.info["sfreq"]),
            parameters=dict(crop_record["requested"]),
            actual_kwargs={"tmin": max(tmin, 0.0), "tmax": tmax},
        )
        try:
            raw.crop(tmin=max(tmin, 0.0), tmax=tmax)
            crop_record.update({
                "applied": True,
                "actual_first_samp": int(getattr(raw, "first_samp", 0)),
                "actual_last_samp_exclusive": int(getattr(raw, "first_samp", 0)) + int(raw.n_times),
                "actual_tmin_sec": float(getattr(raw, "first_time", 0.0) or 0.0),
                "actual_tmax_sec": float(getattr(raw, "first_time", 0.0) or 0.0) + float(raw.n_times) / float(raw.info["sfreq"]),
                "actual_tmin_from_source_start_sec": float(getattr(raw, "first_time", 0.0) or 0.0) - original_first_time_sec,
                "actual_sfreq_hz": float(raw.info["sfreq"]),
                "actual_bounds_clock": "post_crop_raw_sample_grid",
            })
            finish_operation(crop_operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        except Exception as exc:
            crop_record.update({"applied": False, "reason": "failed", "error": str(exc)})
            finish_operation(crop_operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
            logger.warning("WFDB crop failed for %s (%s); continuing without crop", file_path.name, exc)
    signal_support_provenance["crop"] = crop_record

    target_sfreq, sfreq_policy = _resolve_target_sfreq(
        original_sfreq=float(fs),
        preprocess_cfg=preprocess_cfg if isinstance(preprocess_cfg, Mapping) else {},
    )
    signal_support_provenance["target_sfreq_hz"] = float(target_sfreq)
    signal_support_provenance["sampling_policy"] = sfreq_policy
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
    causal_resampler_spec: Optional[dict[str, Any]] = None
    if causal_finite_mode:
        operation = start_operation(
            signal_support_provenance,
            "resample",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"target_sfreq_hz": float(target_sfreq), "mode": "causal_finite"},
            actual_kwargs={"method": "scipy.signal.upfirdn", "phase": "causal_uncompensated"},
        )
        try:
            raw, causal_resampler_spec = _causal_polyphase_resample(raw, target_sfreq)
            operation["support"] = causal_resampler_spec
            operation["temporal_support_status"] = "finite_causal"
            operation["support_reason"] = "bounded FIR dependency with no future source samples"
            finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        except Exception as exc:
            finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
            raise
    elif raw.info["sfreq"] != target_sfreq:
        operation = start_operation(
            signal_support_provenance,
            "resample",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"target_sfreq_hz": float(target_sfreq)},
            actual_kwargs={"target_sfreq": target_sfreq, "n_jobs": n_jobs_resample, "verbose": False},
        )
        operation["actual_kwargs"]["call_signature"] = callable_signature(raw.resample)
        try:
            raw.resample(target_sfreq, n_jobs=n_jobs_resample, verbose=False)
            finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        except Exception as exc:
            finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
            raise
    else:
        operation = start_operation(
            signal_support_provenance,
            "resample",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"target_sfreq_hz": float(target_sfreq)},
            actual_kwargs={"skipped": True, "reason": "already_at_target_sfreq"},
        )
        finish_operation(operation, status="skipped", output_sfreq=float(raw.info["sfreq"]))
    preprocess_timings["resample"] = float(time.perf_counter() - t_resample0)

    notch_hz = preprocess_cfg.get("notch_hz", None) if isinstance(preprocess_cfg, Mapping) else None
    eeg_picks = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)
    if notch_hz is not None and len(eeg_picks) > 0:
        operation = start_operation(
            signal_support_provenance,
            "notch_filter",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"frequencies_hz": notch_hz},
            actual_kwargs={"freqs": notch_hz, "picks_count": int(len(eeg_picks)), "verbose": False},
        )
        try:
            raw.notch_filter(freqs=notch_hz, picks=eeg_picks, verbose=False)
            finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        except Exception as exc:
            finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
            raise
    else:
        operation = start_operation(
            signal_support_provenance,
            "notch_filter",
            input_sfreq=float(raw.info["sfreq"]),
            actual_kwargs={"skipped": True, "reason": "disabled_or_no_eeg_channels"},
        )
        finish_operation(operation, status="skipped", output_sfreq=float(raw.info["sfreq"]))

    eeg_bandpass = preprocess_cfg.get("eeg_bandpass", [1, 45]) if isinstance(preprocess_cfg, Mapping) else [1, 45]
    meg_bandpass = (
        preprocess_cfg.get("meg_bandpass", eeg_bandpass)
        if isinstance(preprocess_cfg, Mapping)
        else eeg_bandpass
    )
    causal_eeg_filter_spec: Optional[dict[str, Any]] = None
    if eeg_bandpass is not None and len(eeg_bandpass) == 2 and len(eeg_picks) > 0:
        operation = start_operation(
            signal_support_provenance,
            "eeg_bandpass",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"bandpass_hz": eeg_bandpass, "picked_channels": [raw.ch_names[int(i)] for i in eeg_picks]},
            actual_kwargs={"l_freq": eeg_bandpass[0], "h_freq": eeg_bandpass[1], "picks_count": int(len(eeg_picks)), "verbose": False},
        )
        operation["actual_kwargs"]["call_signature"] = callable_signature(raw.filter)
        try:
            if causal_finite_mode:
                causal_eeg_filter_spec = _minimum_phase_eeg_filter_spec(raw, eeg_picks, eeg_bandpass)
                operation["actual_kwargs"].update({
                    "method": "fir",
                    "phase": "minimum",
                    "fir_design": "firwin",
                    "fir_window": "hamming",
                    "filter_length": "auto",
                    "pad": "constant",
                })
                operation["support"] = causal_eeg_filter_spec
                operation["temporal_support_status"] = "finite_causal"
                operation["support_reason"] = "minimum-phase FIR with constant zero-state padding"
                raw.filter(
                    l_freq=eeg_bandpass[0],
                    h_freq=eeg_bandpass[1],
                    picks=eeg_picks,
                    method="fir",
                    phase="minimum",
                    fir_design="firwin",
                    fir_window="hamming",
                    filter_length="auto",
                    pad="constant",
                    verbose=False,
                )
            else:
                raw.filter(l_freq=eeg_bandpass[0], h_freq=eeg_bandpass[1], picks=eeg_picks, verbose=False)
            finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        except Exception as exc:
            finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
            raise
    else:
        operation = start_operation(
            signal_support_provenance,
            "eeg_bandpass",
            input_sfreq=float(raw.info["sfreq"]),
            actual_kwargs={"skipped": True, "reason": "disabled_or_no_eeg_channels"},
        )
        finish_operation(operation, status="skipped", output_sfreq=float(raw.info["sfreq"]))

    reref = preprocess_cfg.get("reref", None) if isinstance(preprocess_cfg, Mapping) else None
    if str(reref).lower() == "average" and len(eeg_picks) > 0:
        operation = start_operation(
            signal_support_provenance,
            "average_rereference",
            operation_class="spatial",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"requested": True, "channels": [raw.ch_names[int(i)] for i in eeg_picks]},
            actual_kwargs={"method": "mean_across_selected_channels", "copy": False},
        )
        eeg_data = raw.get_data(picks=eeg_picks)
        raw._data[eeg_picks, :] = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)  # type: ignore[attr-defined]
        finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
    else:
        operation = start_operation(
            signal_support_provenance,
            "average_rereference",
            operation_class="spatial",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"requested": str(reref).lower() == "average"},
            actual_kwargs={"skipped": True, "reason": "disabled_or_no_eeg_channels"},
        )
        finish_operation(operation, status="skipped", output_sfreq=float(raw.info["sfreq"]))

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
    wfdb_eeg_source_units = [str(units[int(idx)] or "").strip() for idx in eeg_picks]
    eeg_values = raw.get_data(picks=eeg_picks)
    if eeg_scale.shape[0] == eeg_values.shape[0]:
        operation = start_operation(
            signal_support_provenance,
            "wfdb_unit_scaling",
            operation_class="amplitude",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"scale_to_microvolts": scale_to_microvolts, "source_units": wfdb_eeg_source_units},
            actual_kwargs={"per_channel_factors": eeg_scale.tolist()},
        )
        eeg_values = eeg_values * eeg_scale[:, np.newaxis]
        finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))

    normalized_wfdb_units = {
        "v": "V", "volt": "V", "volts": "V",
        "mv": "mV", "millivolt": "mV", "millivolts": "mV",
        "uv": "microvolt", "µv": "microvolt", "microvolt": "microvolt", "microvolts": "microvolt",
    }
    source_unit_keys = [unit.lower() for unit in wfdb_eeg_source_units]
    all_physical_units_known = bool(source_unit_keys) and all(unit in normalized_wfdb_units for unit in source_unit_keys)
    if scale_to_microvolts and all_physical_units_known:
        eeg_amplitude_unit, eeg_amplitude_unit_status = "microvolt", "verified"
        eeg_amplitude_unit_reason = "wfdb_physical_channel_units_scaled_to_microvolts"
    elif not scale_to_microvolts and len(set(source_unit_keys)) == 1 and all_physical_units_known:
        eeg_amplitude_unit = normalized_wfdb_units[source_unit_keys[0]]
        eeg_amplitude_unit_status = "verified"
        eeg_amplitude_unit_reason = "uniform_wfdb_physical_channel_unit_preserved"
    else:
        eeg_amplitude_unit, eeg_amplitude_unit_status = None, "unknown"
        eeg_amplitude_unit_reason = (
            "wfdb_channel_units_not_physical_or_not_recorded"
            if not all_physical_units_known
            else "mixed_wfdb_channel_units_not_normalized"
        )
    wfdb_reference_mode = str(reref).strip().lower() if reref is not None else "native_reference_as_loaded"
    wfdb_reference_verified = wfdb_reference_mode == "average" and len(eeg_picks) > 0

    signal_support_provenance["raw_clock_after_crop"] = {
        "sfreq_hz": float(raw.info["sfreq"]),
        "first_time_sec": float(getattr(raw, "first_time", 0.0) or 0.0),
        "first_samp": int(getattr(raw, "first_samp", 0)),
        "n_times": int(raw.n_times),
        "duration_sec": float(raw.n_times / float(raw.info["sfreq"])) if float(raw.info["sfreq"]) > 0 else None,
    }
    if causal_finite_mode and causal_resampler_spec is not None and causal_eeg_filter_spec is not None:
        effective_support = {
            "status": "finite_causal",
            "mode": "causal_finite",
            "reason": "verified contiguous WFDB source plus finite causal polyphase resampler and minimum-phase EEG FIR",
            "resampler": causal_resampler_spec,
            "eeg_bandpass": causal_eeg_filter_spec,
            "excluded_operations": ["notch_filter", "crop", "annotations", "nonfinite_input"],
            "boundary_handling": "zero-state causal padding at record start; source contributors clipped to source array",
        }
        signal_support_provenance["effective_time_support"] = effective_support
        signal_support_provenance["temporal_support"] = {
            "status": "finite_causal_conservative_bound",
            "reason": effective_support["reason"],
            "epoch_bounds": "per-epoch source contributor interval emitted by map_preprocessed_window_to_source_samples",
        }
    else:
        signal_support_provenance["effective_time_support"] = {
            "status": "unknown",
            "reason": "filter_or_resampler_dependency_map_not_recorded",
        }
    # If a test double or unusual header lacked file references, retain a
    # clearly marked fallback identity for an existing conventional sibling.
    if not payload_files:
        fallback_payload = file_path.with_suffix(".mat")
        if fallback_payload.is_file():
            payload_files.append({
                "path": str(fallback_payload.resolve()),
                "exists": True,
                "size_bytes": int(fallback_payload.stat().st_size),
                "hash_status": "captured_after_wfdb_read",
                "sha256": file_sha256(fallback_payload),
            })
    for identity in payload_files:
        path = Path(str(identity.get("path", "")))
        before = identity.get("sha256")
        try:
            after = file_sha256(path)
            identity["sha256_after"] = after
            identity["hash_stable"] = bool(before and str(before).lower() == after.lower())
            if not before:
                identity["hash_status"] = "captured_after_wfdb_read"
        except Exception as exc:
            identity["sha256_after"] = None
            identity["hash_stable"] = False
            identity["hash_after_error"] = str(exc)
    support_source["associated_signal_files"] = payload_files
    support_source["associated_signal_files_status"] = (
        "verified"
        if source_sample_grid_supported and payload_files and all(item.get("hash_stable") is True for item in payload_files)
        else "unknown"
    )
    refresh_source_hash(signal_support_provenance)
    signal_support_provenance["status"] = "completed"

    preprocess_timings["total"] = float(time.perf_counter() - t_pre0)
    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "source_format": "WFDB",
        "wfdb_record_name": str(getattr(record, "record_name", "") or ""),
        "wfdb_scale_to_microvolts": scale_to_microvolts,
        "wfdb_eeg_source_units": wfdb_eeg_source_units,
        "eeg_amplitude_unit": eeg_amplitude_unit,
        "eeg_amplitude_unit_status": eeg_amplitude_unit_status,
        "eeg_amplitude_unit_reason": eeg_amplitude_unit_reason,
        "eeg_reference_mode": wfdb_reference_mode,
        "eeg_reference_status": "verified" if wfdb_reference_verified else "unknown",
        "eeg_reference_reason": "configured_average_reference_applied" if wfdb_reference_verified else "source_reference_not_verified",
        "original_sfreq": float(fs),
        "target_sfreq_resolved": float(raw.info["sfreq"]),
        "sfreq_policy": sfreq_policy,
        "signal_support_provenance": signal_support_provenance,
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
        nwb_timebase_summary = None
        if channel_indices is not None and streamed_sfreq is not None:
            streamed_sfreq = float(streamed_sfreq)
            chunked_resample_strategy = str(
                nwb_cfg.get("chunked_resample_strategy", "legacy_independent")
            ).strip().lower()
            chunked_resampler = {
                "legacy_independent": _nwb_chunked_downsample,
                "overlap_trim_v1": _nwb_chunked_downsample_overlap_trim,
            }.get(chunked_resample_strategy)
            if chunked_resampler is None:
                raise ValueError(
                    "Unsupported NWB chunked_resample_strategy="
                    f"{chunked_resample_strategy!r}. Expected 'legacy_independent' or 'overlap_trim_v1'."
                )
            chunk_start_sample = int(round(nwb_start * original_sfreq))
            chunk_stop_sample = int(round(nwb_stop * original_sfreq)) if nwb_stop is not None else None
            if chunked_resample_strategy == "overlap_trim_v1":
                nwb_timebase_summary = _nwb_series_timebase_summary(
                    series,
                    source_sfreq=original_sfreq,
                    start_sample=chunk_start_sample,
                    stop_sample=chunk_stop_sample,
                )
            data = chunked_resampler(
                series,
                source_sfreq=original_sfreq,
                target_sfreq=streamed_sfreq,
                channel_indices=channel_indices,
                chunk_sec=float(nwb_cfg.get("chunk_sec", 60.0) or 60.0),
                start_sample=chunk_start_sample,
                stop_sample=chunk_stop_sample,
            )
            if nwb_timebase_summary is not None:
                down = int(round(original_sfreq / streamed_sfreq))
                chunked_output_sfreq = original_sfreq / down
                grid_start = float(nwb_timebase_summary["selected_time_start_sec"])
                grid_stop = (
                    grid_start + (data.shape[1] - 1) / chunked_output_sfreq
                    if data.shape[1]
                    else grid_start
                )
                nwb_timebase_summary.update(
                    {
                        "requested_streaming_rate_hz": streamed_sfreq,
                        "chunked_output_sampling_rate_hz": chunked_output_sfreq,
                        "chunked_resampler": {
                            "id": "scipy.signal.resample_poly.overlap_trim_v1",
                            "up": 1,
                            "down": down,
                            "chunk_sec": float(nwb_cfg.get("chunk_sec", 60.0) or 60.0),
                            "context_source_samples_each_side": 10 * down,
                            "context_alignment": "relative_to_selected_source_start",
                        },
                        "resampled_sample_count": int(data.shape[1]),
                        "resampled_grid_start_sec": grid_start,
                        "resampled_grid_stop_sec": grid_stop,
                        "resampled_grid_stop_minus_source_stop_sec": (
                            grid_stop - float(nwb_timebase_summary["selected_time_stop_sec"])
                        ),
                    }
                )
                # Decimation by an integer factor retains the source clock at
                # source_sfreq / factor. Let the ordinary whole-recording
                # resample below convert that grid to the configured target,
                # rather than silently relabeling it as the requested rate.
                original_sfreq = chunked_output_sfreq
            else:
                original_sfreq = streamed_sfreq
        else:
            chunked_resample_strategy = None
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
    if nwb_timebase_summary is not None:
        grid_start = float(nwb_timebase_summary["selected_time_start_sec"])
        grid_stop = grid_start + (raw.n_times - 1) / target_sfreq if raw.n_times else grid_start
        nwb_timebase_summary.update(
            {
                "analysis_sampling_rate_hz": float(target_sfreq),
                "analysis_sample_count": int(raw.n_times),
                "analysis_grid_start_sec": grid_start,
                "analysis_grid_stop_sec": grid_stop,
                "analysis_grid_stop_minus_source_stop_sec": (
                    grid_stop - float(nwb_timebase_summary["selected_time_stop_sec"])
                ),
            }
        )
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
        **({"nwb_timebase": nwb_timebase_summary} if nwb_timebase_summary is not None else {}),
        **(
            {"nwb_chunked_resample_strategy": chunked_resample_strategy}
            if chunked_resample_strategy not in {None, "legacy_independent"}
            else {}
        ),
        "nwb_channel_type": ch_type,
        "nwb_unit": str(nwb_unit) if nwb_unit is not None else None,
        "nwb_conversion": float(nwb_conversion) if nwb_conversion is not None else None,
        "nwb_offset": float(nwb_offset) if nwb_offset is not None else None,
        "nwb_scale_to_microvolts": scale_to_microvolts,
        "nwb_unit_to_output_factor": float(unit_factor),
        "nwb_reref_mode": reref_mode,
        "nwb_reref_channel_count": len(eeg_picks),
        "eeg_reference_mode": reref_mode,
        "eeg_reference_status": "verified" if reref_mode in {"average", "median", "median_within_probe"} and len(eeg_picks) > 0 else "unknown",
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


def _read_raw_signal(file_path: Path | str, preprocess_cfg: Mapping[str, Any] | None):
    """Open one source file, with the CTF options this reader needs.

    ds006502 .hc files name coils ``LPA``, ``RPA``, and ``Nasion``. MNE's
    default matcher misses them. Its default ``system_clock='truncate'``
    also drops that dataset's rest recordings from the sidecar duration of
    360.0 s to 329.7 s. ``preprocess.ctf_system_clock: ignore`` keeps the
    samples that match the sidecar. Other files stay on ``mne.io.read_raw``.
    """
    import mne

    path = Path(file_path)
    if path.name.lower().endswith(".ds"):
        _register_ctf_hc_landmark_aliases()
        clock = "truncate"
        if isinstance(preprocess_cfg, Mapping):
            configured = preprocess_cfg.get("ctf_system_clock", "truncate")
            clock = str(configured or "truncate")
        if clock not in {"truncate", "ignore"}:
            raise ValueError("preprocess.ctf_system_clock must be truncate or ignore")
        return mne.io.read_raw_ctf(path, system_clock=clock, preload=False, verbose=False)
    return mne.io.read_raw(path, preload=False, verbose=False)


def _register_ctf_hc_landmark_aliases() -> None:
    """Teach the installed MNE CTF reader this dataset's .hc coil labels.

    MNE matches only ``left ear``, ``right ear``, and ``nasion``. ds006502
    writes ``LPA``, ``RPA``, and ``Nasion``, so every measured coil is stored
    with kind=-1 and ``read_raw_ctf`` raises missing HPI coordinates. Adding
    the aliases does not change channel data, only whether the file opens.
    """
    try:
        from mne.io.ctf.constants import CTF
        from mne.io.ctf.hc import _kind_dict
    except Exception:
        return
    _kind_dict.setdefault("LPA", CTF.CTFV_COIL_LPA)
    _kind_dict.setdefault("RPA", CTF.CTFV_COIL_RPA)
    _kind_dict.setdefault("Nasion", CTF.CTFV_COIL_NAS)


def preprocess_file(
    file_path: Path,
    config: Mapping[str, Any],
    *,
    index_record: Optional[Mapping[str, Any]] = None,
    dataset_root: Optional[Path] = None,
) -> PreprocessedSignals:
    """Preprocess a single file (currently EEG/fMRI and other supported types).

    Args:
        file_path: Path to a raw signal file.
        config: Ingest configuration with ``preprocess`` and modality sections.
        index_record: BIDS index row for modality-specific metadata contracts.
        dataset_root: Dataset root used to resolve index-relative sidecar paths.

    Returns:
        :class:`PreprocessedSignals` with arrays, ``sfreq``, channel maps, and meta.
    """
    suffixes = "".join(file_path.suffixes).lower()
    source_datatype = _infer_path_datatype(file_path)
    if suffixes.endswith(".snirf"):
        return _preprocess_fnirs_file(
            file_path,
            index_record=index_record,
            dataset_root=dataset_root,
        )
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

    # Load raw with delayed data load to avoid reading entire file.
    signal_support_provenance = build_signal_support_provenance(
        Path(file_path), target_sfreq=target_sfreq, source_datatype=source_datatype
    )
    raw = None
    t_load0 = time.perf_counter()
    try:
        try:
            raw = _read_raw_signal(file_path_for_mne, preprocess_cfg)
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
    temporal_support_mode = str(preprocess_cfg.get("temporal_support_mode", "default")).strip().lower()
    if temporal_support_mode not in {"default", "causal_finite"}:
        raise ValueError(f"unsupported MNE temporal_support_mode: {temporal_support_mode}")
    causal_finite_mode = temporal_support_mode == "causal_finite"
    causal_edf_sample_grid: Optional[dict[str, Any]] = None
    if causal_finite_mode:
        if not suffixes.endswith(".edf"):
            raise ValueError("causal_finite MNE mode currently requires one continuous EDF source file")
        causal_edf_sample_grid = _verify_uniform_edf_sample_grid(raw, Path(file_path_for_mne))
        original_rate = float(raw.info["sfreq"])
        if (
            int(getattr(raw, "first_samp", 0)) != 0
            or abs(float(getattr(raw, "first_time", 0.0) or 0.0)) > 1e-12
            or abs(float(target_sfreq) - original_rate) > max(1e-9, original_rate * 1e-12)
        ):
            raise ValueError("causal_finite EDF mode requires an unshifted source grid and no resampling")
        resample_cfg_for_mode = preprocess_cfg.get("resample", {})
        resample_method = (
            str(resample_cfg_for_mode.get("method", "")).strip().lower()
            if isinstance(resample_cfg_for_mode, Mapping)
            else ""
        )
        if resample_method != "identity_sample_grid":
            raise ValueError("causal_finite EDF mode requires resample.method=identity_sample_grid")
        if notch_hz is not None:
            raise ValueError("causal_finite EDF mode currently requires notch_hz=null")
        if str(preprocess_cfg.get("eeg_bandpass_phase", "")).strip().lower() != "minimum":
            raise ValueError("causal_finite EDF mode requires eeg_bandpass_phase=minimum")
        if not (isinstance(eeg_bandpass, (list, tuple)) and len(eeg_bandpass) == 2):
            raise ValueError("causal_finite EDF mode requires a two-edge EEG bandpass")
        if _resolve_crop_config(config, dataset_id_hint) or bool(
            _resolve_event_crop_config(config, dataset_id_hint).get("enabled", False)
        ):
            raise ValueError("causal_finite EDF mode currently refuses configured or event-based crops")
        robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
        bad_channel_cfg = robustness_cfg.get("bad_channels", {}) if isinstance(robustness_cfg, Mapping) else {}
        if isinstance(bad_channel_cfg, Mapping) and bool(bad_channel_cfg.get("enabled", False)):
            raise ValueError("causal_finite EDF mode refuses whole-recording bad-channel selection")
        heuristic_exclusion_cfg = preprocess_cfg.get("heuristic_bad_channel_exclusion", {}) if isinstance(preprocess_cfg, Mapping) else {}
        if isinstance(heuristic_exclusion_cfg, Mapping) and bool(heuristic_exclusion_cfg.get("enabled", False)):
            raise ValueError("causal_finite EDF mode refuses whole-recording heuristic EEG channel exclusion")
    signal_support_provenance["temporal_support_mode"] = temporal_support_mode
    if causal_edf_sample_grid is not None:
        signal_support_provenance["edf_sample_grid"] = causal_edf_sample_grid
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
    signal_support_provenance["input_sfreq_hz"] = original_sfreq
    signal_support_provenance["configured_target_sfreq_hz"] = signal_support_provenance.get("target_sfreq_hz")
    signal_support_provenance["target_sfreq_hz"] = float(target_sfreq)
    signal_support_provenance["sampling_policy"] = sfreq_policy
    original_first_samp = int(getattr(raw, "first_samp", 0))
    original_first_time_sec = float(getattr(raw, "first_time", 0.0) or 0.0)
    raw_file_str = str(file_path)
    signal_support_provenance["raw_clock_before_crop"] = {
        "sfreq_hz": original_sfreq,
        "first_time_sec": original_first_time_sec,
        "first_samp": int(getattr(raw, "first_samp", 0)),
        "n_times": int(raw.n_times),
        "duration_sec": float(raw.n_times / original_sfreq) if original_sfreq > 0 else None,
    }
    dataset_id = _infer_dataset_id(file_path, config)
    quality_interval_policy = _resolve_quality_interval_policy(config, dataset_id)
    raw_flatline_qc_cfg = _resolve_raw_flatline_qc(config, dataset_id)
    quality_interval_policy_record: Dict[str, Any] = {
        "enabled": bool(quality_interval_policy.get("enabled", False)),
        "status": "configured" if quality_interval_policy.get("enabled", False) else "disabled",
        "source_kinds": quality_interval_policy.get("source_kinds", []),
        "annotation_description": quality_interval_policy.get("annotation_description", "BAD_acq_skip"),
        "mask_bad_annotations": bool(quality_interval_policy.get("mask_bad_annotations", True)),
        "segment_resampling": bool(quality_interval_policy.get("segment_resampling", False)),
    }
    raw_flatline_qc_record: Dict[str, Any] = {
        "enabled": bool(raw_flatline_qc_cfg.get("enabled", False)),
        "status": "configured" if raw_flatline_qc_cfg.get("enabled", False) else "disabled",
        "interval_count": 0,
    }
    # Preserve source quality intervals before crop. Their source records remain
    # observational; an explicit opt-in policy may additionally mark selected
    # kinds BAD so temporal operations can respect those unavailable intervals.
    quality_intervals: List[Dict[str, Any]] = []
    annotations = getattr(raw, "annotations", None)
    annotation_status = "unavailable"
    if annotations is not None:
        try:
            annotation_status = "none_observed"
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description
            ):
                description_text = str(description)
                if not description_text.upper().startswith("BAD"):
                    continue
                # MNE annotation onset is in the raw time frame and includes
                # Raw.first_time for nonzero first_samp recordings.
                start_sec = original_raw_annotation_start(float(onset), original_first_time_sec)
                duration_sec = float(duration)
                quality_intervals.append(quality_interval(
                    source_kind="mne_raw_annotation",
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                    raw_file=raw_file_str,
                    description=description_text,
                    raw_first_samp=original_first_samp,
                ))
                annotation_status = "observed"
        except Exception as exc:
            annotation_status = "error"
            signal_support_provenance["source_quality_intervals_error"] = str(exc)

    event_status = "not_applicable"
    events_path = _find_events_file(file_path, {"suffix_candidates": ["_events.tsv"]}) if dataset_id == "ds005555" else None
    if dataset_id == "ds005555" and events_path is None:
        event_status = "unavailable"
    if events_path is not None:
        try:
            event_hash_before = file_sha256(events_path)
            events_df = pd.read_csv(events_path, sep="\t")
            event_hash_after = file_sha256(events_path)
            if event_hash_before != event_hash_after:
                raise RuntimeError("source_events_changed_during_read")
            if "stage_hum" in events_df.columns and "onset" in events_df.columns and "duration" in events_df.columns:
                event_status = "none_observed"
                stage_values = pd.to_numeric(events_df["stage_hum"], errors="coerce")
                onsets = pd.to_numeric(events_df["onset"], errors="coerce")
                durations = pd.to_numeric(events_df["duration"], errors="coerce")
                for row_idx in events_df.index[stage_values.eq(8) & onsets.notna() & durations.notna()]:
                    start_sec = float(onsets.loc[row_idx])
                    duration_sec = float(durations.loc[row_idx])
                    quality_intervals.append(quality_interval(
                        source_kind="source_stage_hum_code_8",
                        start_sec=start_sec,
                        duration_sec=duration_sec,
                        raw_file=raw_file_str,
                        source_row=int(row_idx),
                        stage_code=8,
                        source_path=str(events_path),
                        source_sha256=event_hash_before,
                        source_column="stage_hum",
                    ))
                    event_status = "observed"
            else:
                event_status = "error"
                signal_support_provenance["source_events_quality_error"] = "stage_hum_onset_duration_columns_missing"
        except Exception as exc:
            event_status = "error"
            signal_support_provenance["source_events_quality_error"] = str(exc)
    overall_quality_status = "observed" if quality_intervals else (
        "error" if "error" in {annotation_status, event_status} else (
            "unavailable" if "unavailable" in {annotation_status, event_status} else "none_observed"
        )
    )
    source_quality_record: Dict[str, Any] = {
        "status": overall_quality_status,
        "clock": "original_raw_seconds",
        "intervals": quality_intervals,
        "annotation_status": annotation_status,
        "source_stage_hum_code_8_status": event_status,
        "raw_file": raw_file_str,
        "analysis_admissibility": "not_assessed_here",
    }
    signal_support_provenance["source_quality_intervals"] = source_quality_record
    quality_interval_policy_record.update(
        _apply_quality_interval_annotations(raw, quality_intervals, quality_interval_policy)
    )
    source_quality_record["application_policy"] = dict(quality_interval_policy_record)

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
    crop_record: Dict[str, Any] = {
        "requested": None,
        "applied": False,
        "actual_tmin_sec": None,
        "actual_tmax_sec": None,
        "reason": "not_configured",
    }
    is_flattened_eeglab_epochs = bool(
        isinstance(eeglab_import_meta, Mapping)
        and str(eeglab_import_meta.get("source", "")).strip().lower() == "eeglab_epoched_concat"
    )
    if is_flattened_eeglab_epochs:
        crop_record["source_sample_mapping_status"] = "unknown"
        crop_record["source_sample_mapping_reason"] = "flattened_eeglab_epochs_have_no_continuous_source_sample_map"
    if crop_window is None:
        crop_cfg = _resolve_crop_config(config, dataset_id)
        if crop_cfg:
            tmin = float(crop_cfg.get("start_sec", 0) or 0.0)
            tmax_val = crop_cfg.get("stop_sec", None)
            tmax = float(tmax_val) if tmax_val is not None else None
            crop_window = (tmin, tmax)

    if crop_window is not None:
        tmin, tmax = crop_window
        crop_record["requested"] = {"tmin_sec": float(tmin), "tmax_sec": float(tmax) if tmax is not None else None}
        crop_record["reason"] = "configured"
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
                crop_record["reason"] = "invalid_after_clamp"
                logger.warning(
                    "Clamped crop window invalid for %s (tmin=%.2f, tmax=%.2f); skipping crop",
                    file_path.name,
                    tmin_clamped,
                    tmax_clamped,
                )
            else:
                pre_crop_sfreq = float(raw.info["sfreq"])
                pre_crop_first_time = float(getattr(raw, "first_time", 0.0) or 0.0)
                pre_crop_first_samp = int(getattr(raw, "first_samp", original_first_samp))
                raw.crop(tmin=tmin_clamped, tmax=tmax_clamped)
                actual_first_samp = int(getattr(raw, "first_samp", original_first_samp))
                actual_sfreq = float(raw.info["sfreq"])
                actual_first_time = float(getattr(raw, "first_time", 0.0) or 0.0)
                actual_n_times = int(raw.n_times)
                actual_last_samp_exclusive = actual_first_samp + actual_n_times
                crop_clock = {
                    "sfreq_hz": actual_sfreq,
                    "first_time_sec": actual_first_time,
                    "first_samp": actual_first_samp,
                    "n_times": actual_n_times,
                }
                crop_source_map = map_preprocessed_window_to_source_samples(
                    0,
                    actual_n_times,
                    preprocessed_sfreq_hz=actual_sfreq,
                    provenance={
                        **signal_support_provenance,
                        "raw_clock_after_crop": crop_clock,
                    },
                    allow_source_grid=not is_flattened_eeglab_epochs,
                    refusal_reason="flattened_eeglab_epochs_have_no_continuous_source_sample_map",
                )
                crop_record.update(
                    {
                        "applied": True,
                        "actual_first_samp": actual_first_samp,
                        "actual_last_samp_exclusive": actual_last_samp_exclusive,
                        "actual_tmin_sec": actual_first_time,
                        "actual_tmax_sec": actual_first_time + actual_n_times / actual_sfreq,
                        "actual_tmin_from_source_start_sec": actual_first_time - original_first_time_sec,
                        "actual_tmax_from_source_start_sec": actual_first_time - original_first_time_sec + actual_n_times / actual_sfreq,
                        "actual_sfreq_hz": actual_sfreq,
                        "actual_bounds_clock": "post_resample_raw_sample_grid",
                        "pre_crop_sfreq_hz": pre_crop_sfreq,
                        "pre_crop_first_time_sec": pre_crop_first_time,
                        "pre_crop_first_samp": pre_crop_first_samp,
                        "nominal_source_sample_start": crop_source_map.get("nominal_source_sample_start"),
                        "nominal_source_sample_end_exclusive": crop_source_map.get("nominal_source_sample_end_exclusive"),
                        "source_sample_mapping_status": crop_source_map.get("source_sample_mapping_status", "unknown"),
                        "source_sample_mapping_reason": crop_source_map.get("source_sample_mapping_reason"),
                    }
                )

    signal_support_provenance["crop"] = crop_record

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
    signal_support_provenance["channel_types_after_dataset_typing"] = {
        str(name): str(channel_type)
        for name, channel_type in zip(raw.ch_names, raw.get_channel_types())
    }

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

    # Optional, observational pre-filter QC: identify stretches where all
    # retained EEG-like channels have identical adjacent raw samples. This is
    # recorded separately and never changes the signal or selected BAD policy.
    if raw_flatline_qc_record["enabled"]:
        min_duration_sec = float(raw_flatline_qc_cfg.get("min_duration_sec", 1.0))
        flatline_picks = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)
        flatline_names = [raw.ch_names[int(idx)] for idx in flatline_picks]
        if len(flatline_picks) == 0:
            raw_flatline_qc_record.update({"status": "no_eeg_like_channels", "min_duration_sec": min_duration_sec})
        else:
            flatline_values = raw.get_data(picks=flatline_picks)
            flatline_intervals = _joint_flatline_quality_intervals(
                flatline_values,
                sfreq=float(raw.info["sfreq"]),
                first_time_sec=float(getattr(raw, "first_time", 0.0) or 0.0) - original_first_time_sec,
                channel_names=flatline_names,
                min_duration_sec=min_duration_sec,
                raw_file=raw_file_str,
            )
            raw_flatline_qc_record.update(
                {
                    "status": "observed" if flatline_intervals else "none_observed",
                    "min_duration_sec": min_duration_sec,
                    "channel_count": int(len(flatline_picks)),
                    "channel_names": flatline_names,
                    "interval_count": int(len(flatline_intervals)),
                    "intervals": flatline_intervals,
                    "signal_stage": "raw_before_temporal_preprocessing",
                    "detection": "exact_all_channels_adjacent_sample_equality",
                }
            )
            policy_source_kinds = quality_interval_policy_record.get("source_kinds", [])
            if isinstance(policy_source_kinds, str):
                policy_source_kinds = [policy_source_kinds]
            if (
                quality_interval_policy_record.get("enabled")
                and "raw_joint_eeg_flatline" in {str(value) for value in policy_source_kinds or []}
                and flatline_intervals
            ):
                flatline_annotation = _apply_quality_interval_annotations(
                    raw,
                    flatline_intervals,
                    quality_interval_policy,
                    source_origin_time_sec=original_first_time_sec,
                )
                quality_interval_policy_record["derived_qc_annotation"] = flatline_annotation
                quality_interval_policy_record["selected_source_interval_count"] = int(
                    quality_interval_policy_record.get("selected_source_interval_count", 0)
                ) + int(flatline_annotation.get("selected_source_interval_count", 0))
                quality_interval_policy_record["annotations_added"] = int(
                    quality_interval_policy_record.get("annotations_added", 0)
                ) + int(flatline_annotation.get("annotations_added", 0))
                quality_interval_policy_record["status"] = "applied"
                source_quality_record["application_policy"] = dict(quality_interval_policy_record)
        source_quality_record["raw_joint_eeg_flatline_qc"] = dict(raw_flatline_qc_record)
        signal_support_provenance["raw_joint_eeg_flatline_qc"] = dict(raw_flatline_qc_record)

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
        if quality_interval_policy_record.get("enabled") and quality_interval_policy_record.get("segment_resampling"):
            quality_interval_policy_record["resampling"] = {
                "enabled": True,
                "status": "not_run_eeglab_concat_policy",
            }
        logger.warning(
            "Skipping resample for concatenated epoched EEGLAB stream (%s) to avoid boundary artifacts",
            file_path.name,
        )
        skipped = start_operation(
            signal_support_provenance,
            "resample_early",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"target_sfreq_hz": target_sfreq},
            actual_kwargs={"skipped": True, "reason": "eeglab_concatenated_epoch_policy"},
        )
        finish_operation(skipped, status="skipped", output_sfreq=float(raw.info["sfreq"]))
    elif raw.info["sfreq"] != target_sfreq:
        operation = start_operation(
            signal_support_provenance,
            "resample_early",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"target_sfreq_hz": target_sfreq},
            actual_kwargs={"target_sfreq": target_sfreq, "n_jobs": n_jobs_resample, "verbose": False},
        )
        operation["actual_kwargs"]["call_signature"] = callable_signature(raw.resample)
        logger.info(
            "Early resample from %.1f Hz to %.1f Hz for %s (n_jobs=%s)",
            raw.info["sfreq"],
            target_sfreq,
            file_path.name,
            n_jobs_resample,
        )
        try:
            if quality_interval_policy_record.get("enabled") and quality_interval_policy_record.get("segment_resampling"):
                raw, segment_resampling_report = _split_raw_at_bad_annotation_boundaries(
                    raw,
                    target_sfreq=float(target_sfreq),
                    annotation_description=str(
                        quality_interval_policy_record.get("annotation_description", "BAD_acq_skip")
                    ),
                )
                quality_interval_policy_record["resampling"] = segment_resampling_report
                operation["actual_kwargs"]["quality_interval_resampling"] = segment_resampling_report
            raw.resample(target_sfreq, n_jobs=n_jobs_resample, verbose=False)
        except Exception as exc:
            finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
            raise
        finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
    else:
        if quality_interval_policy_record.get("enabled") and quality_interval_policy_record.get("segment_resampling"):
            quality_interval_policy_record["resampling"] = {
                "enabled": True,
                "status": "not_needed_target_rate_matches_input",
            }
        skipped = start_operation(
            signal_support_provenance,
            "resample_early",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"target_sfreq_hz": target_sfreq},
            actual_kwargs={"skipped": True, "reason": "already_at_target_sfreq"},
        )
        finish_operation(skipped, status="skipped", output_sfreq=float(raw.info["sfreq"]))
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
    if causal_finite_mode and art_method not in {"", "none", "null"}:
        raise ValueError("causal_finite EDF mode refuses recording-fitted temporal artifact correction")

    def _apply_eog_regression(r: mne.io.BaseRaw) -> bool:
        """Internal helper: apply eog regression. Returns True iff it actually applied."""
        eeg_picks = mne.pick_types(r.info, eeg=True)
        eog_picks = mne.pick_types(r.info, eog=True)
        if len(eeg_picks) == 0 or len(eog_picks) == 0:
            logger.info("EOG regression skipped: missing EEG or EOG channels")
            return False
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
            return True
        except Exception as exc:
            logger.warning("EOG regression failed (%s); continuing without regression", exc)
            return False

    def _apply_ica(r: mne.io.BaseRaw) -> bool:
        """Internal helper: apply ICA artifact removal.

        Supports EOG proxy channels (e.g. Fp1/Fp2) for datasets without
        dedicated EOG electrodes, and ECG component detection when a cardiac
        channel is present.
        """
        if ICA is None:
            logger.info("ICA not available; skipping ICA artifacts removal")
            return False
        eeg_picks = mne.pick_types(r.info, eeg=True)
        if len(eeg_picks) == 0:
            logger.info("ICA skipped: no EEG channels")
            return False
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
            # ica.apply(r) with an empty exclude list is a no-op: the signal
            # is byte-for-byte unmodified. "Applied" must mean a confirmed
            # artifact component was actually removed (e.g. no EOG/ECG
            # channels were present to detect against, or detection found
            # nothing), not merely that the ICA fit-and-apply call sequence
            # ran without raising. Per independent review of
            # ingest_jacobian_fidelity_handover_2.md item 9: qc_ok_eeg's
            # artifact_qc_applied signal must not be True for a window whose
            # signal was never actually touched.
            return len(ica.exclude) > 0
        except Exception as exc:
            logger.warning("ICA artifact removal failed (%s); continuing without ICA", exc)
            return False

    # Acquisition-side EEG QC is read before notch/bandpass and before the
    # configured rereference. It is descriptive only and never changes data.
    raw_eeg_acquisition_qc: Dict[str, Any] = {
        "status": "unsupported",
        "reason": "no_eeg_channels",
    }
    acquisition_eeg_picks = (
        mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True)
        if source_datatype != "ecg"
        else np.asarray([], dtype=int)
    )
    if len(acquisition_eeg_picks) > 0:
        try:
            from .mnps.v3_1.pipeline_measurements import compute_acquisition_qc

            qc_rate = float(raw.info["sfreq"])
            qc_samples = min(int(raw.n_times), max(8, int(round(20.0 * qc_rate))))
            qc_raw = raw.get_data(picks=acquisition_eeg_picks, start=0, stop=qc_samples)
            raw_eeg_acquisition_qc = compute_acquisition_qc(
                qc_raw,
                sample_rate_hz=qc_rate,
                stage="prefilter_before_artifact_reduction_and_rereference",
                window_start_sec=0.0,
                amplitude_unit="volt",
            )
            raw_eeg_acquisition_qc["status"] = (
                "computed"
                if raw_eeg_acquisition_qc.get("line_noise_48_52_ratio_status") == "computed"
                else "partial"
            )
        except Exception as exc:
            raw_eeg_acquisition_qc = {
                "status": "unsupported",
                "reason": f"acquisition_qc_failed:{type(exc).__name__}",
            }

    t_artifact0 = time.perf_counter()
    # Whether an artifact-reduction method actually executed and modified the
    # signal (not merely configured) -- see project/mnps_v3/tests/
    # ingest_jacobian_fidelity_handover_2.md item 9. `art_method` alone
    # records what was *configured*; a skip (missing EOG/EEG channels, ICA
    # unavailable) or exception still leaves art_method=="ica"/"eog_reg" even
    # though nothing ran.
    artifact_applied = False
    if art_method == "eog_reg":
        artifact_applied = _apply_eog_regression(raw)
    elif art_method == "ica":
        artifact_applied = _apply_ica(raw)
    elif art_method not in {"", "none", "null"}:
        logger.info("Unknown artifact method '%s'; skipping artifact reduction", art_method)
    preprocess_timings["artifact_reduction"] = float(time.perf_counter() - t_artifact0)

    # For concatenated epoched EEGLAB streams, default is to skip time-domain filters
    # to avoid ringing at synthetic epoch boundaries.
    skip_time_filters = is_eeglab_epoched_concat and not eeglab_concat_policy.get("filter_concatenated_epochs", False)

    # Apply notch filter globally if configured (in-place)
    t_notch0 = time.perf_counter()
    filter_annotation_kwargs: Dict[str, Any] = {}
    filter_skip_by_annotation = _filter_skip_by_annotation(raw, quality_interval_policy_record)
    if filter_skip_by_annotation is not None:
        filter_annotation_kwargs["skip_by_annotation"] = filter_skip_by_annotation
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
        operation = start_operation(
            signal_support_provenance,
            "notch_filter",
            input_sfreq=float(raw.info["sfreq"]),
            parameters={"freqs_hz": notch_hz, "picked_channels": int(len(notch_picks))},
            actual_kwargs={
                "freqs": notch_hz,
                "picks_count": int(len(notch_picks)),
                "verbose": False,
                **({"skip_by_annotation": list(filter_skip_by_annotation)} if filter_skip_by_annotation is not None else {}),
            },
        )
        operation["actual_kwargs"]["call_signature"] = callable_signature(raw.notch_filter)
        if len(notch_picks) > 0:
            try:
                raw.notch_filter(
                    freqs=notch_hz,
                    picks=notch_picks,
                    verbose=False,
                    **filter_annotation_kwargs,
                )
            except Exception as exc:
                finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
                raise
            finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        else:
            finish_operation(operation, status="skipped", output_sfreq=float(raw.info["sfreq"]), fallback="no_eligible_channels")
            logger.info("Skipping notch filter for %s: no eligible channels after typing", file_path.name)
    else:
        skipped = start_operation(signal_support_provenance, "notch_filter", input_sfreq=float(raw.info["sfreq"]), actual_kwargs={"skipped": True, "reason": "disabled_or_time_filter_policy"})
        finish_operation(skipped, status="skipped", output_sfreq=float(raw.info["sfreq"]))
    preprocess_timings["notch"] = float(time.perf_counter() - t_notch0)

    # Treat intracranial EEG (sEEG/ECoG) as EEG for ingest purposes.
    # Many BIDS iEEG datasets (e.g. ds004100) label channels as "seeg"/"ecog",
    # which would otherwise result in empty EEG feature extraction.
    eeg_like_chans = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True) if source_datatype != "ecg" else np.array([], dtype=int)

    # Apply EEG bandpass on EEG-like picks only
    t_bandpass0 = time.perf_counter()
    eeg_chans = eeg_like_chans
    causal_eeg_filter_spec: Optional[dict[str, Any]] = None
    if causal_finite_mode and (
        len(eeg_chans) == 0
        or eeg_bandpass is None
        or len(eeg_bandpass) != 2
        or skip_time_filters
    ):
        raise ValueError("causal_finite EDF mode requires eligible EEG channels and an applied bandpass")
    if len(eeg_chans) > 0 and eeg_bandpass is not None and len(eeg_bandpass) == 2 and not skip_time_filters:
        operation = start_operation(signal_support_provenance, "eeg_bandpass", input_sfreq=float(raw.info["sfreq"]), parameters={"bandpass_hz": eeg_bandpass, "picked_channels": [raw.ch_names[int(i)] for i in eeg_chans]}, actual_kwargs={"l_freq": eeg_bandpass[0], "h_freq": eeg_bandpass[1], "picks_count": int(len(eeg_chans)), "verbose": False, **({"skip_by_annotation": list(filter_skip_by_annotation)} if filter_skip_by_annotation is not None else {})})
        operation["actual_kwargs"]["call_signature"] = callable_signature(raw.filter)
        try:
            if causal_finite_mode:
                eeg_data = raw.get_data(picks=eeg_chans)
                if eeg_data.size == 0 or not np.isfinite(eeg_data).all():
                    raise ValueError("causal_finite EDF mode requires finite EEG input samples")
                operation["actual_kwargs"].update({
                    "method": "fir",
                    "phase": "minimum",
                    "fir_design": "firwin",
                    "fir_window": "hamming",
                    "filter_length": "auto",
                    "pad": "constant",
                })
                causal_eeg_filter_spec = _apply_minimum_phase_eeg_filter(
                    raw,
                    eeg_chans,
                    eeg_bandpass,
                    skip_by_annotation=filter_skip_by_annotation,
                )
                operation["support"] = causal_eeg_filter_spec
                operation["temporal_support_status"] = "finite_causal"
                operation["support_reason"] = "minimum-phase FIR with constant zero-state padding and finite past-only support"
            else:
                raw.filter(
                    l_freq=eeg_bandpass[0],
                    h_freq=eeg_bandpass[1],
                    picks=eeg_chans,
                    verbose=False,
                    **filter_annotation_kwargs,
                )
            finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        except Exception as primary_exc:
            if causal_finite_mode:
                finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=primary_exc)
                raise
            # Fallback to mne.filter.filter_data on array if raw.filter fails
            if quality_interval_policy_record.get("enabled") and _has_bad_raw_annotations(raw):
                finish_operation(
                    operation,
                    status="failed",
                    output_sfreq=float(raw.info["sfreq"]),
                    error=primary_exc,
                )
                raise RuntimeError(
                    "Refusing annotation-unaware EEG filter fallback while BAD intervals are active"
                ) from primary_exc
            try:
                data = raw.get_data(picks=eeg_chans)
                filtered = mne.filter.filter_data(data, raw.info["sfreq"], eeg_bandpass[0], eeg_bandpass[1], verbose=False)
                raw._data[eeg_chans, :] = filtered  # type: ignore[attr-defined]
                finish_operation(operation, status="fallback", output_sfreq=float(raw.info["sfreq"]), fallback="mne.filter.filter_data", error=primary_exc)
            except Exception as fallback_exc:
                finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), fallback="mne.filter.filter_data", error=fallback_exc)
                raise
    else:
        skipped = start_operation(signal_support_provenance, "eeg_bandpass", input_sfreq=float(raw.info["sfreq"]), actual_kwargs={"skipped": True, "reason": "no_eligible_channels_or_disabled"})
        finish_operation(skipped, status="skipped", output_sfreq=float(raw.info["sfreq"]))
    preprocess_timings["bandpass"] = float(time.perf_counter() - t_bandpass0)

    # Optional MEG bandpass for FIF-based M/EEG recordings.
    t_meg_bandpass0 = time.perf_counter()
    meg_chans = mne.pick_types(raw.info, meg=True, ref_meg=False)
    if causal_finite_mode and len(meg_chans) > 0:
        raise ValueError("causal_finite EDF mode currently supports EEG-only temporal preprocessing")
    if len(meg_chans) > 0 and meg_bandpass is not None and len(meg_bandpass) == 2 and not skip_time_filters:
        operation = start_operation(signal_support_provenance, "meg_bandpass", input_sfreq=float(raw.info["sfreq"]), parameters={"bandpass_hz": meg_bandpass, "picked_channels": [raw.ch_names[int(i)] for i in meg_chans]}, actual_kwargs={"l_freq": meg_bandpass[0], "h_freq": meg_bandpass[1], "picks_count": int(len(meg_chans)), "verbose": False, **({"skip_by_annotation": list(filter_skip_by_annotation)} if filter_skip_by_annotation is not None else {})})
        operation["actual_kwargs"]["call_signature"] = callable_signature(raw.filter)
        try:
            raw.filter(
                l_freq=meg_bandpass[0],
                h_freq=meg_bandpass[1],
                picks=meg_chans,
                verbose=False,
                **filter_annotation_kwargs,
            )
            finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
        except Exception as primary_exc:
            if quality_interval_policy_record.get("enabled") and _has_bad_raw_annotations(raw):
                finish_operation(
                    operation,
                    status="failed",
                    output_sfreq=float(raw.info["sfreq"]),
                    error=primary_exc,
                )
                raise RuntimeError(
                    "Refusing annotation-unaware MEG filter fallback while BAD intervals are active"
                ) from primary_exc
            data = raw.get_data(picks=meg_chans)
            try:
                filtered = mne.filter.filter_data(data, raw.info["sfreq"], meg_bandpass[0], meg_bandpass[1], verbose=False)
                raw._data[meg_chans, :] = filtered  # type: ignore[attr-defined]
                finish_operation(operation, status="fallback", output_sfreq=float(raw.info["sfreq"]), fallback="mne.filter.filter_data", error=primary_exc)
            except Exception as fallback_exc:
                finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), fallback="mne.filter.filter_data", error=fallback_exc)
                raise
    else:
        skipped = start_operation(signal_support_provenance, "meg_bandpass", input_sfreq=float(raw.info["sfreq"]), actual_kwargs={"skipped": True, "reason": "no_eligible_channels_or_disabled"})
        finish_operation(skipped, status="skipped", output_sfreq=float(raw.info["sfreq"]))
    preprocess_timings["meg_bandpass"] = float(time.perf_counter() - t_meg_bandpass0)

    # Resample once for all channels if needed
    # (Note: we already resampled early to save memory, so this is now usually a no-op).
    t_resample_late0 = time.perf_counter()
    if raw.info["sfreq"] != target_sfreq and not (is_eeglab_epoched_concat and not eeglab_concat_policy.get("resample_concatenated_epochs", False)):
        operation = start_operation(signal_support_provenance, "resample_late", input_sfreq=float(raw.info["sfreq"]), parameters={"target_sfreq_hz": target_sfreq}, actual_kwargs={"target_sfreq": target_sfreq, "verbose": False})
        try:
            raw.resample(target_sfreq, verbose=False)
        except Exception as exc:
            finish_operation(operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
            raise
        finish_operation(operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
    else:
        skipped = start_operation(signal_support_provenance, "resample_late", input_sfreq=float(raw.info["sfreq"]), actual_kwargs={"skipped": True, "reason": "already_at_target_or_eeglab_policy"})
        finish_operation(skipped, status="skipped", output_sfreq=float(raw.info["sfreq"]))
    preprocess_timings["resample_late"] = float(time.perf_counter() - t_resample_late0)

    # Detect and drop obviously bad EEG-like channels (simple heuristics) BEFORE CAR reref.
    # Respect the versioned heuristic exclusion switch. The legacy default
    # remains enabled for existing overlays that omit the new switch.
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config, Mapping) else {}
    heuristic_exclusion_cfg = (
        preprocess_cfg.get("heuristic_bad_channel_exclusion", {})
        if isinstance(preprocess_cfg, Mapping)
        else {}
    )
    heuristic_exclusion_cfg = (
        heuristic_exclusion_cfg if isinstance(heuristic_exclusion_cfg, Mapping) else {}
    )
    bad_channel_detection_enabled = bool(heuristic_exclusion_cfg.get("enabled", True))
    # Preserve the existing thresholds under robustness.bad_channels.
    robustness_cfg = config.get("robustness", {}) if isinstance(config, Mapping) else {}
    bad_channel_cfg = (
        robustness_cfg.get("bad_channels", {})
        if isinstance(robustness_cfg, Mapping)
        else {}
    )
    bad_channel_cfg = bad_channel_cfg if isinstance(bad_channel_cfg, Mapping) else {}
    t_badch0 = time.perf_counter()
    if source_datatype == "ecg":
        bad_info = {"bad_channels": [], "reasons": {}, "status": "not_applicable_ecg"}
        bad_eeg_channels = []
    elif not bad_channel_detection_enabled:
        bad_info = {"bad_channels": [], "reasons": {}, "status": "disabled_by_config"}
        bad_eeg_channels = []
    else:
        bad_info = _detect_bad_eeg_channels(raw, bad_channel_cfg)
        bad_info["status"] = "applied"
        bad_eeg_channels = bad_info.get("bad_channels", []) or []
        if causal_finite_mode and bad_eeg_channels:
            raise ValueError("causal_finite EDF mode refuses whole-recording bad-channel exclusion")
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
    csd_operation = start_operation(signal_support_provenance, "eeg_csd", operation_class="spatial", input_sfreq=float(raw.info["sfreq"]), parameters={"enabled": bool(eeg_csd_cfg.get("enabled", False)), "lambda2": eeg_csd_cfg.get("lambda2", 1e-5), "stiffness": eeg_csd_cfg.get("stiffness", 4.0), "n_legendre_terms": eeg_csd_cfg.get("n_legendre_terms", 50)}, actual_kwargs={"copy": False, "scalp_channels": []})
    if source_datatype != "ecg" and bool(eeg_csd_cfg.get("enabled", False)):
        if compute_current_source_density is None:
            csd_reason = "mne_csd_unavailable"
            finish_operation(csd_operation, status="skipped", output_sfreq=float(raw.info["sfreq"]), fallback=csd_reason)
            logger.warning("EEG CSD requested but mne.compute_current_source_density is unavailable")
        else:
            eeg_scalp_picks = mne.pick_types(raw.info, eeg=True, seeg=False, ecog=False)
            csd_operation["actual_kwargs"]["scalp_channels"] = [raw.ch_names[int(i)] for i in eeg_scalp_picks]
            csd_operation["actual_kwargs"]["call_signature"] = callable_signature(compute_current_source_density)
            min_eeg_channels = int(eeg_csd_cfg.get("min_eeg_channels", 16) or 16)
            if len(eeg_scalp_picks) < min_eeg_channels:
                csd_reason = f"insufficient_scalp_channels:{len(eeg_scalp_picks)}<{min_eeg_channels}"
                logger.warning(
                    "EEG CSD skipped for %s: scalp EEG channels %d < min_eeg_channels %d",
                    file_path.name,
                    len(eeg_scalp_picks),
                    min_eeg_channels,
                )
                finish_operation(csd_operation, status="skipped", output_sfreq=float(raw.info["sfreq"]), fallback=csd_reason)
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
                    finish_operation(csd_operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
                    logger.info(
                        "Applied EEG CSD for %s (lambda2=%g, stiffness=%.2f, n_legendre_terms=%d)",
                        file_path.name,
                        lambda2,
                        stiffness,
                        n_legendre_terms,
                    )
                except Exception as exc:
                    csd_reason = f"failed:{exc}"
                    finish_operation(csd_operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
                    on_error = str(eeg_csd_cfg.get("on_error", "warn")).strip().lower()
                    if on_error in {"raise", "fail", "error"}:
                        raise RuntimeError(
                            f"EEG CSD failed for {file_path.name} with on_error={on_error}: {exc}"
                        ) from exc
                    logger.warning("EEG CSD failed for %s (%s); continuing without CSD", file_path.name, exc)
    else:
        csd_reason = "disabled"
        finish_operation(csd_operation, status="skipped", output_sfreq=float(raw.info["sfreq"]), fallback=csd_reason)
    preprocess_timings["eeg_csd"] = float(time.perf_counter() - t_csd0)

    # Average re-reference for EEG-like channels if requested.
    # We do manual referencing so iEEG (seeg/ecog) is also covered.
    t_reref0 = time.perf_counter()
    eeg_chans = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True) if source_datatype != "ecg" else np.array([], dtype=int)
    reref_operation = start_operation(signal_support_provenance, "average_rereference", operation_class="spatial", input_sfreq=float(raw.info["sfreq"]), parameters={"requested": reref == "average", "channels": [raw.ch_names[int(i)] for i in eeg_chans]}, actual_kwargs={})
    if reref == "average" and len(eeg_chans) > 0:
        if csd_applied:
            finish_operation(reref_operation, status="skipped", output_sfreq=float(raw.info["sfreq"]), fallback="csd_reference_free")
            logger.info("Skipping average reref for %s because EEG CSD is already reference-free", file_path.name)
        else:
            try:
                eeg_data = raw.get_data(picks=eeg_chans)
                eeg_ref = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
                raw._data[eeg_chans, :] = eeg_ref  # type: ignore[attr-defined]
                finish_operation(reref_operation, status="applied", output_sfreq=float(raw.info["sfreq"]))
            except Exception as exc:
                finish_operation(reref_operation, status="failed", output_sfreq=float(raw.info["sfreq"]), error=exc)
                logger.warning("Average reference failed for %s (%s); continuing without reref", file_path.name, exc)
    else:
        finish_operation(reref_operation, status="skipped", output_sfreq=float(raw.info["sfreq"]), fallback="disabled_or_no_channels")
    preprocess_timings["reref"] = float(time.perf_counter() - t_reref0)

    # Collect per-modality arrays
    modality_signals: Dict[str, np.ndarray] = {}
    channels_dict: Dict[str, List[str]] = {}
    mask_quality_intervals = bool(
        quality_interval_policy_record.get("enabled")
        and quality_interval_policy_record.get("mask_bad_annotations")
    )

    def _get_modality_data(picks: Any) -> np.ndarray:
        if mask_quality_intervals:
            return raw.get_data(picks=picks, reject_by_annotation="NaN")
        return raw.get_data(picks=picks)

    t_collect0 = time.perf_counter()
    # EEG (including iEEG: seeg/ecog and scalp EEG transformed to CSD)
    # MNE returns SI units (V). Convert to µV so PSD band-powers are on a
    # numerically stable scale (~1 µV²/Hz instead of ~1e-12 V²/Hz).
    eeg_chans = (
        mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True, csd=True)
        if source_datatype != "ecg"
        else np.array([], dtype=int)
    )
    postfilter_eeg_acquisition_qc: Dict[str, Any] = {
        "status": "unsupported",
        "reason": "no_eeg_channels",
    }
    if len(eeg_chans) > 0:
        bad_segments_cfg = bdf_adapter_meta.get("bad_segments", {}) if isinstance(bdf_adapter_meta, Mapping) else {}
        mask_bdf_bad_segments = bool(
            isinstance(bad_segments_cfg, Mapping)
            and bad_segments_cfg.get("enabled")
            and bad_segments_cfg.get("available")
        )
        if mask_bdf_bad_segments or mask_quality_intervals:
            # Preserve the source time grid and mark BAD_ samples as NaN. The
            # EEG extractor rejects every window touching a non-finite sample,
            # avoiding synthetic adjacency across omitted artifact intervals.
            eeg_values = raw.get_data(picks=eeg_chans, reject_by_annotation="NaN")
            bad_sample_mask = ~np.isfinite(eeg_values).all(axis=0)
            masked_samples = int(bad_sample_mask.sum())
            masked_duration_sec = float(masked_samples / raw.info["sfreq"])
            if mask_bdf_bad_segments:
                bad_segments_cfg = dict(bad_segments_cfg)
                bad_segments_cfg["masked_samples"] = masked_samples
                bad_segments_cfg["masked_duration_sec"] = masked_duration_sec
                bdf_adapter_meta["bad_segments"] = bad_segments_cfg
            if mask_quality_intervals:
                quality_interval_policy_record["masking"] = {
                    "status": "applied",
                    "masked_samples": masked_samples,
                    "masked_duration_sec": masked_duration_sec,
                    "description": "MNE BAD_* annotations",
                }
        else:
            eeg_values = raw.get_data(picks=eeg_chans)
        modality_signals["eeg"] = eeg_values * 1e6
        channels_dict["eeg"] = [raw.ch_names[i] for i in eeg_chans]
        try:
            from .mnps.v3_1.pipeline_measurements import compute_acquisition_qc

            qc_rate = float(raw.info["sfreq"])
            qc_samples = min(int(eeg_values.shape[1]), max(8, int(round(20.0 * qc_rate))))
            postfilter_eeg_acquisition_qc = compute_acquisition_qc(
                eeg_values[:, :qc_samples],
                sample_rate_hz=qc_rate,
                stage="postfilter_postreference_first_20s",
                window_start_sec=0.0,
                amplitude_unit="volt",
            )
            postfilter_eeg_acquisition_qc["status"] = (
                "computed"
                if postfilter_eeg_acquisition_qc.get("line_noise_48_52_ratio_status") == "computed"
                else "partial"
            )
        except Exception as exc:
            postfilter_eeg_acquisition_qc = {
                "status": "unsupported",
                "reason": f"postfilter_qc_failed:{type(exc).__name__}",
            }

    # MEG
    meg_chans = mne.pick_types(raw.info, meg=True, ref_meg=False)
    if len(meg_chans) > 0:
        modality_signals["meg"] = _get_modality_data(meg_chans)
        channels_dict["meg"] = [raw.ch_names[i] for i in meg_chans]
        meg_mag_chans = mne.pick_types(raw.info, meg="mag", ref_meg=False)
        meg_grad_chans = mne.pick_types(raw.info, meg="grad", ref_meg=False)
        if len(meg_mag_chans) > 0:
            modality_signals["meg_mag"] = _get_modality_data(meg_mag_chans)
            channels_dict["meg_mag"] = [raw.ch_names[i] for i in meg_mag_chans]
        if len(meg_grad_chans) > 0:
            modality_signals["meg_grad"] = _get_modality_data(meg_grad_chans)
            channels_dict["meg_grad"] = [raw.ch_names[i] for i in meg_grad_chans]

    # EOG
    eog_chans = mne.pick_types(raw.info, eog=True)
    if len(eog_chans) > 0:
        modality_signals["eog"] = _get_modality_data(eog_chans)
        channels_dict["eog"] = [raw.ch_names[i] for i in eog_chans]

    # EMG
    emg_chans = mne.pick_types(raw.info, emg=True)
    if len(emg_chans) > 0:
        modality_signals["emg"] = _get_modality_data(emg_chans)
        channels_dict["emg"] = [raw.ch_names[i] for i in emg_chans]

    # ECG
    ecg_chans = mne.pick_types(raw.info, ecg=True)
    if len(ecg_chans) == 0 and source_datatype == "ecg" and len(raw.ch_names) > 0:
        ecg_chans = np.asarray([0], dtype=int)
    if len(ecg_chans) > 0:
        modality_signals["ecg"] = _get_modality_data(ecg_chans)
        channels_dict["ecg"] = [raw.ch_names[i] for i in ecg_chans]
    # Respiration
    resp_chans = mne.pick_types(raw.info, resp=True)
    if len(resp_chans) > 0:
        modality_signals["resp"] = _get_modality_data(resp_chans)
        channels_dict["resp"] = [raw.ch_names[i] for i in resp_chans]
    # PPG / pleth: prefer explicit bio channels or name-based matching.
    bio_chans = list(mne.pick_types(raw.info, bio=True))
    ppg_chans = [
        idx
        for idx, ch_name in enumerate(raw.ch_names)
        if re.search(r"(?i)(ppg|pleth|pulse|photopleth)", str(ch_name))
    ]
    ppg_channel_selection = "source_channel_name_regex"
    if not ppg_chans and bio_chans:
        ppg_chans = [idx for idx in bio_chans if idx not in list(ecg_chans)]
        ppg_channel_selection = "mne_bio_channel_type_fallback"
    if not ppg_chans and source_datatype == "ecg":
        ppg_chans = [idx for idx in range(len(raw.ch_names)) if idx not in list(ecg_chans)][:1]
        ppg_channel_selection = "ecg_source_non_ecg_channel_fallback"
    if ppg_chans:
        modality_signals["ppg"] = _get_modality_data(ppg_chans)
        channels_dict["ppg"] = [raw.ch_names[i] for i in ppg_chans]
    signal_support_provenance["ppg_source_selection"] = {
        "rule": ppg_channel_selection if ppg_chans else "no_ppg_source_selected",
        "channel_names": [raw.ch_names[i] for i in ppg_chans],
        "channel_types_after_dataset_typing": [raw.get_channel_types()[i] for i in ppg_chans],
    }
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
        modality_signals["eda"] = _get_modality_data(eda_chans)
        channels_dict["eda"] = [raw.ch_names[i] for i in eda_chans]
    preprocess_timings["collect_modalities"] = float(time.perf_counter() - t_collect0)
    preprocess_timings["total"] = float(time.perf_counter() - t_pre0)
    signal_support_provenance["raw_clock_after_crop"] = {
        "sfreq_hz": float(raw.info["sfreq"]),
        "first_time_sec": float(getattr(raw, "first_time", 0.0) or 0.0),
        "first_samp": int(getattr(raw, "first_samp", 0)),
        "n_times": int(raw.n_times),
        "duration_sec": float(raw.n_times / float(raw.info["sfreq"])) if float(raw.info["sfreq"]) > 0 else None,
    }
    if causal_finite_mode:
        before = signal_support_provenance.get("raw_clock_before_crop", {})
        after = signal_support_provenance["raw_clock_after_crop"]
        if (
            crop_record.get("applied") is True
            or int(before.get("first_samp", -1)) != 0
            or int(after.get("first_samp", -1)) != 0
            or int(before.get("n_times", -1)) != int(after.get("n_times", -2))
            or abs(float(before.get("sfreq_hz", float("nan"))) - float(after["sfreq_hz"])) > 1e-9
            or abs(float(before.get("first_time_sec", float("nan"))) - float(after["first_time_sec"])) > 1e-12
        ):
            raise ValueError("causal_finite EDF mode detected an unsupported crop or changed source sample grid")
        if causal_eeg_filter_spec is None:
            raise ValueError("causal_finite EDF mode completed without a recorded EEG FIR")
        applied_temporal = [
            operation.get("name")
            for operation in signal_support_provenance.get("operations", [])
            if isinstance(operation, Mapping)
            and operation.get("class", "temporal") == "temporal"
            and operation.get("status") == "applied"
        ]
        if applied_temporal != ["eeg_bandpass"]:
            raise ValueError(
                "causal_finite EDF mode observed unsupported applied temporal operations: "
                + ", ".join(str(name) for name in applied_temporal)
            )
        identity_support = _identity_sample_grid_support_spec(raw)
        signal_support_provenance["source_sample_grid_supported"] = True
        signal_support_provenance["source_sample_grid_reason"] = "verified_uniform_edf_sample_grid_no_resampling"
        signal_support_provenance["effective_time_support"] = {
            "status": "finite_causal",
            "mode": "causal_finite_edf_identity_grid",
            "resampler": identity_support,
            "eeg_bandpass": causal_eeg_filter_spec,
            "excluded_operations": ["notch_filter", "crop", "nonfinite_input", "recording-fitted_temporal_artifact_correction"],
            "support_scope": "all raw modalities conservatively bounded by the union of the EEG FIR and identity-grid dependencies",
        }
        signal_support_provenance["temporal_support"] = {
            "status": "finite_causal",
            "reason": "verified uniform EDF sample grid plus recorded minimum-phase EEG FIR tap extent",
            "epoch_bounds": "per-epoch source contributor interval emitted by map_preprocessed_window_to_source_samples",
        }
    source_quality_record["application_policy"] = dict(quality_interval_policy_record)
    signal_support_provenance["quality_interval_policy"] = dict(quality_interval_policy_record)
    signal_support_provenance["raw_joint_eeg_flatline_qc"] = dict(raw_flatline_qc_record)
    refresh_source_hash(signal_support_provenance)
    signal_support_provenance["status"] = "completed"

    logger.info(f"Preprocessed {file_path.name}: {list(modality_signals.keys())}")

    artifact_meta: Dict[str, Any] = {
        "method": art_method,
        # "applied" is True only when the configured method actually ran and
        # modified the signal; a skip (missing channels, ICA unavailable) or
        # an exception leaves this False even though "method" still names
        # the configured method. Downstream QC (qc_ok_eeg) must consult this
        # flag rather than treating a configured-but-not-executed method as
        # if artifact rejection had run.
        "applied": bool(artifact_applied),
        "bad_eeg_channels": bad_eeg_channels,
        "bad_eeg_reasons": bad_info.get("reasons", {}),
        "bad_channel_detection": {
            "method": "variance_correlation_heuristic",
            "enabled": bool(bad_channel_detection_enabled),
            "status": str(bad_info.get("status", "unknown")),
        },
    }

    final_eeg_labels = list(channels_dict.get("eeg", []))
    final_channel_types = list(raw.get_channel_types())
    channel_type_profile: Dict[str, int] = {}
    for channel_type in final_channel_types:
        key = str(channel_type).strip().lower() or "unknown"
        channel_type_profile[key] = channel_type_profile.get(key, 0) + 1
    channel_set_hash = hashlib.sha256(
        "\n".join(sorted(str(label).strip().casefold() for label in final_eeg_labels)).encode("utf-8")
    ).hexdigest() if final_eeg_labels else None
    reference_set_hash = (
        channel_set_hash
        if reref_operation.get("status") == "applied" and final_eeg_labels
        else None
    )
    raw_eeg_acquisition_qc["channel_type_profile"] = channel_type_profile
    raw_eeg_acquisition_qc["eeg_channel_count"] = int(len(final_eeg_labels))
    raw_eeg_acquisition_qc["misc_channel_count"] = int(channel_type_profile.get("misc", 0))
    raw_eeg_acquisition_qc["channel_set_hash"] = channel_set_hash
    raw_eeg_acquisition_qc["reference_set_hash"] = reference_set_hash
    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "original_sfreq": original_sfreq,
        "target_sfreq_resolved": float(target_sfreq),
        "sfreq_policy": sfreq_policy,
        "crop_window": {
            "tmin": crop_record["requested"]["tmin_sec"] if crop_record["requested"] else None,
            "tmax": crop_record["requested"]["tmax_sec"] if crop_record["requested"] else None,
        },
        "crop_window_applied": {
            "applied": bool(crop_record["applied"]),
            "tmin": crop_record["actual_tmin_sec"],
            "tmax": crop_record["actual_tmax_sec"],
            "reason": crop_record["reason"],
        },
        "artifact": artifact_meta,
        "bdf_adapter": bdf_adapter_meta,
        "quality_interval_policy": quality_interval_policy_record,
        "raw_joint_eeg_flatline_qc": raw_flatline_qc_record,
        "eeg_amplitude_unit": "microvolt" if len(eeg_chans) > 0 and not csd_applied else None,
        "eeg_amplitude_unit_status": "verified" if len(eeg_chans) > 0 and not csd_applied else "unknown",
        "eeg_amplitude_unit_reason": (
            "mne_si_volt_data_scaled_to_microvolts"
            if len(eeg_chans) > 0 and not csd_applied
            else "eeg_channel_missing_or_spatial_csd_changes_amplitude_unit"
        ),
        "eeg_reference_mode": "average" if reref_operation.get("status") == "applied" else (
            "csd_reference_free" if csd_applied else "source_reference_as_loaded"
        ),
        "eeg_reference_status": "verified" if reref_operation.get("status") == "applied" else "unknown",
        "eeg_reference_reason": str(reref_operation.get("status", "unknown")),
        "channel_type_profile": channel_type_profile,
        "eeg_channel_count": int(len(final_eeg_labels)),
        "misc_channel_count": int(channel_type_profile.get("misc", 0)),
        "channel_set_hash": channel_set_hash,
        "reference_set_hash": reference_set_hash,
        "eeg_acquisition_qc": {
            "schema": "mndm.eeg_acquisition_qc.v1",
            "prefilter": raw_eeg_acquisition_qc,
            "postfilter": postfilter_eeg_acquisition_qc,
            "descriptive_only": True,
            "measurement_adjustment_applied": False,
        },
        "eeg_csd": {
            "enabled": bool(eeg_csd_cfg.get("enabled", False)),
            "applied": bool(csd_applied),
            "reason": csd_reason,
            "lambda2": float(eeg_csd_cfg.get("lambda2", 1e-5) or 1e-5),
            "stiffness": float(eeg_csd_cfg.get("stiffness", 4.0) or 4.0),
            "n_legendre_terms": int(eeg_csd_cfg.get("n_legendre_terms", 50) or 50),
        },
        "timings": preprocess_timings,
        "signal_support_provenance": signal_support_provenance,
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

    from .pipeline.fmri_parcellate import parcellate_label_means
    from .pipeline.fmri_roi_cache import (
        build_roi_ts_cache_key,
        load_cached_atlas,
        roi_ts_cache_dir,
        save_roi_ts,
        try_load_roi_ts,
    )
    from .pipeline.nuisance import resolve_fmri_nuisance
    from .pipeline.spatial_align import (
        FmriAtlasSpaceNotTestable,
        orientation_code,
        resolve_atlas_space,
        spaces_match,
    )
    from .pipeline.tr_resolve import resolution_to_meta

    # Header-only BOLD load (no 4D materialization) for TR + cache key.
    bold_img = nib.load(str(file_path))
    lookup = roi_ts_cache_lookup_args(
        file_path, config, nifti_zooms=bold_img.header.get_zooms()
    )
    if lookup is None:
        raise ValueError("preprocess.fmri.atlas_path must be set in the config for fMRI preprocessing")
    tr_resolution = lookup["tr_resolution"]
    tr = float(tr_resolution.tr_sec)
    sfreq = float(tr_resolution.sfreq)
    tr_meta = resolution_to_meta(tr_resolution)
    cache_kwargs = lookup["kwargs"]
    cache_identity = lookup["identity"]
    cache_key = build_roi_ts_cache_key(**cache_kwargs)
    cached = try_load_roi_ts(
        bold_path=file_path,
        config=config,
        dataset_id=dataset_id,
        expected_key=cache_key,
        config_identity=cache_identity,
        require_bold_exists=True,
    )
    if cached is not None:
        meta = dict(cached["meta"])
        meta.update(tr_meta)
        meta["file"] = str(file_path)
        meta["dataset_id"] = dataset_id
        meta["original_sfreq"] = sfreq
        meta["atlas_path"] = str(atlas_path)
        meta["atlas_cache_hit"] = int(meta.get("atlas_cache_hit", 0) or 0)
        logger.info(
            "Preprocessed fMRI %s: ROI-TS cache hit (%d regions at sfreq=%.3f Hz) rss_mb=%s",
            file_path.name,
            int(np.asarray(cached["roi_ts"]).shape[0]),
            sfreq,
            None if (rss := _process_rss_mb()) is None else f"{rss:.1f}",
        )
        return PreprocessedSignals(
            signals={"fmri": np.asarray(cached["roi_ts"], dtype=np.float32)},
            sfreq=sfreq,
            channels={"fmri": list(cached["names"])},
            meta=meta,
        )

    bold_data = bold_img.get_fdata()
    if bold_data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD data in {file_path}, got shape {bold_data.shape}")
    rss_fdata = _process_rss_mb()
    logger.info(
        "fMRI RSS after get_fdata %s: rss_mb=%s header_dtype=%s array_dtype=%s shape=%s",
        file_path.name,
        None if rss_fdata is None else f"{rss_fdata:.1f}",
        bold_img.get_data_dtype(),
        bold_data.dtype,
        tuple(int(x) for x in bold_data.shape),
    )

    atlas_img, atlas_data, atlas_cache_hit = load_cached_atlas(atlas_path)
    atlas_labels_expected = np.unique(atlas_data)
    atlas_labels_expected = atlas_labels_expected[atlas_labels_expected > 0]

    space_decision = resolve_atlas_space(
        atlas_img.affine,
        bold_img.affine,
        atlas_data.shape,
        bold_data.shape[:3],
        fmri_cfg,
    )
    space_meta = space_decision.to_meta()
    if space_decision.should_resample:
        try:
            from nibabel.processing import resample_from_to  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Atlas/BOLD space mismatch and nibabel.processing.resample_from_to "
                "is not available; cannot resample atlas to BOLD space."
            ) from exc

        logger.warning(
            "Resampling atlas %s from shape %s affine/ornt %s to BOLD shape %s ornt %s for %s",
            atlas_path,
            atlas_data.shape,
            space_decision.atlas_ornt,
            bold_data.shape[:3],
            space_decision.bold_ornt,
            file_path.name,
        )
        # Nearest-neighbour (order=0) preserves integer labels. Target is the
        # 3D spatial grid of the BOLD image (shape + affine), not the 4D volume.
        target = (bold_data.shape[:3], bold_img.affine)
        atlas_img = resample_from_to(atlas_img, target, order=0)
        atlas_data = atlas_img.get_fdata()
        if atlas_data.shape != bold_data.shape[:3]:
            raise ValueError(
                f"Resampled atlas shape {atlas_data.shape} still does not match BOLD spatial "
                f"shape {bold_data.shape[:3]} for {file_path}"
            )
        if not spaces_match(atlas_img.affine, bold_img.affine):
            raise FmriAtlasSpaceNotTestable(
                "NOT_TESTABLE: resampled atlas still does not match BOLD affine/"
                f"orientation for {file_path}"
            )
        space_meta["atlas_ornt"] = orientation_code(atlas_img.affine)
        space_meta["bold_ornt"] = orientation_code(bold_img.affine)
        space_meta["atlas_affine_match"] = 1
        space_meta["atlas_ornt_match"] = 1

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

    # Parcellate: mean BOLD per atlas label (finite-weighted bincount).
    label_ids, region_array = parcellate_label_means(
        bold_data, atlas_data, labels=labels.astype(int)
    )
    rss_parcel = _process_rss_mb()
    logger.info(
        "fMRI RSS after parcellation %s: rss_mb=%s n_roi=%s n_times=%s",
        file_path.name,
        None if rss_parcel is None else f"{rss_parcel:.1f}",
        int(region_array.shape[0]),
        int(region_array.shape[1]) if region_array.ndim == 2 else None,
    )
    if region_array.shape[0] == 0:
        raise ValueError(f"No non-empty atlas regions found for {file_path}")
    region_names = [label_names.get(int(lab), f"ROI_{int(lab)}") for lab in label_ids]

    region_array, nuisance_meta = resolve_fmri_nuisance(
        region_array, file_path, fmri_cfg, tr
    )

    from .features.fmri_continuous import maybe_apply_preprocess_bandpass

    region_array, filter_meta = maybe_apply_preprocess_bandpass(region_array, fmri_cfg)
    if filter_meta.get("filter_ignored_preprocess_bandpass") is not None:
        logger.info(
            "P0.4: skipping preprocess_fmri bandpass %s for %s; canonical stage is fmri_continuous",
            filter_meta["filter_ignored_preprocess_bandpass"],
            file_path.name,
        )

    modality_signals: Dict[str, np.ndarray] = {"fmri": region_array.astype(np.float32)}
    channels_dict: Dict[str, List[str]] = {"fmri": region_names}

    meta: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_id": dataset_id,
        "original_sfreq": sfreq,
        "atlas_path": str(atlas_path),
        "atlas_cache_hit": int(bool(atlas_cache_hit)),
        "roi_ts_cache_hit": 0,
        "roi_ts_cache_key": cache_key,
        **tr_meta,
        **filter_meta,
        **nuisance_meta,
        **space_meta,
    }
    cache_dir = roi_ts_cache_dir(config, dataset_id)
    if cache_dir is not None:
        try:
            persist_meta = {k: v for k, v in meta.items() if k != "roi_ts_cache_hit"}
            save_roi_ts(
                cache_dir=cache_dir,
                key=cache_key,
                bold_path=file_path,
                roi_ts=np.asarray(region_array, dtype=np.float32),
                names=region_names,
                sfreq=sfreq,
                meta=persist_meta,
                identity=cache_identity,
            )
        except Exception:
            logger.debug("Failed to persist ROI-TS cache for %s", file_path, exc_info=True)

    logger.info("Preprocessed fMRI %s: %d regions at sfreq=%.3f Hz", file_path.name, region_array.shape[0], sfreq)

    return PreprocessedSignals(
        signals=modality_signals,
        sfreq=sfreq,
        channels=channels_dict,
        meta=meta,
    )



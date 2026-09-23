"""fMRI feature extraction with continuous preprocessing before epoching.

Ingest contract: session-level continuous transforms run first; a single epoch
loop defines temporal resolution; per-window metrics are computed from sliced
continuous outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import fmri_continuous
from . import fmri_epoch_metrics
from ..reproducibility import resolve_base_seed

logger = logging.getLogger(__name__)


def compute_fmri_features(
    signals: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    session_continuous: Optional[Mapping[str, Any]] = None,
    roi_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Compute per-window fMRI metrics from regional BOLD time series.

    Args:
        signals: Mapping with ``signals`` (``fmri`` array ``[n_regions, n_times]``),
            ``sfreq`` (1/TR in Hz), optional ``channels``/``dataset_id``, etc.
        config: Windowing from ``features.fmri`` / ``epoching``; continuous
            preprocessing from ``preprocessing`` or ``preprocess.fmri``.
        session_continuous: Optional precomputed ``process_session_signals``
            output (filtered_ts / phase_ts). When provided, Hilbert/bandpass
            are not rerun. Used by regional reuse (P1.3).
        roi_indices: Optional row index into the session arrays. Metrics are
            computed on that subset after session-level filtering.

    Returns:
        One row per window with epoch timing, global/regional metrics, FC
        summaries, and bookkeeping columns (window length, step, sample counts).
    """
    fmri_data = _validate_and_extract_data(signals)
    n_regions, n_times = fmri_data.shape
    if n_regions == 0 or n_times == 0:
        return pd.DataFrame()

    meta = signals.get("meta") if isinstance(signals.get("meta"), Mapping) else {}
    raw_sfreq = signals.get("sfreq")
    if raw_sfreq is None and isinstance(meta, Mapping):
        raw_sfreq = meta.get("sfreq")
    try:
        sfreq = float(raw_sfreq) if raw_sfreq is not None else float("nan")
    except (TypeError, ValueError):
        sfreq = float("nan")
    if not np.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError(
            "NOT_TESTABLE: fMRI feature extraction requires a positive sfreq from resolved TR"
        )
    dataset_id = signals.get("dataset_id")
    tr_sec_meta = meta.get("tr_sec") if isinstance(meta, Mapping) else None
    tr_source_meta = meta.get("tr_source") if isinstance(meta, Mapping) else None
    try:
        fmri_tr_sec = float(tr_sec_meta) if tr_sec_meta is not None else (1.0 / sfreq)
    except (TypeError, ValueError):
        fmri_tr_sec = 1.0 / sfreq
    fmri_tr_source = str(tr_source_meta) if tr_source_meta else None
    nuisance_status_meta = meta.get("nuisance_status") if isinstance(meta, Mapping) else None
    fmri_nuisance_status = str(nuisance_status_meta) if nuisance_status_meta else None
    space_status_meta = meta.get("atlas_space_status") if isinstance(meta, Mapping) else None
    fmri_atlas_space_status = str(space_status_meta) if space_status_meta else None
    atlas_ornt_meta = meta.get("atlas_ornt") if isinstance(meta, Mapping) else None
    fmri_atlas_ornt = str(atlas_ornt_meta) if atlas_ornt_meta else None
    bold_ornt_meta = meta.get("bold_ornt") if isinstance(meta, Mapping) else None
    fmri_bold_ornt = str(bold_ornt_meta) if bold_ornt_meta else None
    affine_match_meta = meta.get("atlas_affine_match") if isinstance(meta, Mapping) else None
    try:
        fmri_atlas_affine_match = int(affine_match_meta) if affine_match_meta is not None else None
    except (TypeError, ValueError):
        fmri_atlas_affine_match = None
    atol_meta = meta.get("atlas_affine_atol_mm") if isinstance(meta, Mapping) else None
    try:
        fmri_atlas_affine_atol_mm = float(atol_meta) if atol_meta is not None else None
    except (TypeError, ValueError):
        fmri_atlas_affine_atol_mm = None

    window_sec, step_sec, window_samples, step_samples = _compute_window_params(
        config=config,
        sfreq=sfreq,
        dataset_id=dataset_id,
    )
    if window_samples <= 1 or step_samples <= 0:
        logger.warning("Invalid fMRI windowing parameters; returning empty table")
        return pd.DataFrame()
    if n_times < window_samples:
        logger.warning(
            "fMRI run shorter than one window; returning empty feature table (n_times=%d, window_samples=%d)",
            n_times,
            window_samples,
        )
        return pd.DataFrame()

    if session_continuous is None:
        continuous_out = prepare_session_continuous(signals, config)
    else:
        continuous_out = dict(session_continuous)
    filtered_ts = np.asarray(continuous_out.get("filtered_ts", fmri_data), dtype=float)
    phase_ts = continuous_out.get("phase_ts")
    phase_arr = np.asarray(phase_ts, dtype=float) if phase_ts is not None else None
    if roi_indices is not None:
        idx = np.asarray(roi_indices, dtype=int)
        if idx.size == 0:
            return pd.DataFrame()
        fmri_data = fmri_data[idx]
        filtered_ts = filtered_ts[idx]
        if phase_arr is not None:
            phase_arr = phase_arr[idx]
        n_regions, n_times = fmri_data.shape
        if n_regions == 0 or n_times == 0:
            return pd.DataFrame()
    filter_band = continuous_out.get("filter_bandpass") or [0.01, 0.1]
    fmri_filter_applied = int(bool(continuous_out.get("filter_applied", True)))
    fmri_filter_skipped = int(bool(continuous_out.get("filter_skipped", False)))
    fmri_filter_stage = str(continuous_out.get("filter_stage") or fmri_continuous.FILTER_STAGE_CONTINUOUS)
    raw_order = continuous_out.get("filter_order")
    try:
        fmri_filter_order = int(raw_order) if raw_order is not None else None
    except (TypeError, ValueError):
        fmri_filter_order = None
    try:
        fmri_filter_low = float(filter_band[0])
        fmri_filter_high = float(filter_band[1])
    except (TypeError, ValueError, IndexError):
        fmri_filter_low = 0.01
        fmri_filter_high = 0.1

    roi_names = None
    channels_map = signals.get("channels") if isinstance(signals, Mapping) else None
    if isinstance(channels_map, Mapping):
        roi_names = channels_map.get("fmri")
    if roi_indices is not None and isinstance(roi_names, (list, tuple)):
        idx_names = np.asarray(roi_indices, dtype=int)
        roi_names = [roi_names[int(i)] for i in idx_names if 0 <= int(i) < len(roi_names)]

    metrics_cfg = {}
    if isinstance(config, Mapping):
        metrics_cfg = config.get("metrics", {}) or {}
        if not metrics_cfg:
            features_cfg = config.get("features", {}) or {}
            # Preferred v2 path: features.metrics
            if isinstance(features_cfg, Mapping):
                metrics_cfg = features_cfg.get("metrics", {}) or {}
            # Backward-compatible fallback for existing configs.
            if not metrics_cfg:
                metrics_cfg = ((features_cfg.get("fmri", {}) if isinstance(features_cfg, Mapping) else {}) or {}).get(
                    "metrics", {}
                ) or {}
            metrics_cfg = dict(metrics_cfg) if isinstance(metrics_cfg, Mapping) else {}
            base_seed, _ = resolve_base_seed(config, dataset_id=dataset_id)
            repro_cfg = metrics_cfg.get("reproducibility", {})
            repro_cfg = dict(repro_cfg) if isinstance(repro_cfg, Mapping) else {}
            repro_cfg.setdefault("seed", int(base_seed))
            metrics_cfg["reproducibility"] = repro_cfg

    # NMD-fMRI-v2.5 Stage 2 (sciencelead/003.md + 0004.md): dynamic-connectivity
    # delta features need the previous epoch's value, same pattern as the
    # existing fmri_lf_power_delta below. Opt-in only (default False), so no
    # existing dataset config changes output schema/cost unless it explicitly
    # enables compute_dynamic_deltas.
    compute_dynamic_deltas = bool(metrics_cfg.get("compute_dynamic_deltas", False))
    records: list[Dict[str, Any]] = []
    prev_power_global: Optional[float] = None
    prev_modularity: Optional[float] = None
    prev_fc_mean: Optional[float] = None

    # 2) Single epoch loop (no nested windows).
    for epoch_id, start_idx in enumerate(range(0, n_times - window_samples + 1, step_samples)):
        end_idx = start_idx + window_samples
        epoch_raw = fmri_data[:, start_idx:end_idx]
        epoch_filtered = filtered_ts[:, start_idx:end_idx]
        epoch_phase = phase_arr[:, start_idx:end_idx] if phase_arr is not None else None

        # 3) Local per-window metrics from already processed continuous signals.
        epoch_metrics = fmri_epoch_metrics.compute_local_metrics(
            epoch_raw=epoch_raw,
            epoch_filtered=epoch_filtered,
            epoch_phase=epoch_phase,
            sfreq=float(sfreq),
            config=metrics_cfg,
            roi_names=roi_names if isinstance(roi_names, (list, tuple)) else None,
        )

        # Hard scrub epochs with excessive motion proxy (DVARS) on raw BOLD.
        dvars_val = float(epoch_metrics.get("fmri_dvars", float("nan")))
        dvars_threshold = float(metrics_cfg.get("dvars_threshold", 5.0))
        if np.isfinite(dvars_val) and dvars_val > dvars_threshold:
            for key in list(epoch_metrics.keys()):
                if key != "fmri_dvars":
                    epoch_metrics[key] = float("nan")
            logger.debug(
                "Epoch %d scrubbed due to high DVARS (%.2f > %.2f)",
                int(epoch_id),
                dvars_val,
                dvars_threshold,
            )

        fmri_variance_global = float(epoch_metrics.get("fmri_variance_global", float("nan")))
        fmri_signal_power = float(epoch_metrics.get("fmri_signal_power", float("nan")))
        if prev_power_global is None or not np.isfinite(prev_power_global) or not np.isfinite(fmri_signal_power):
            lf_power_delta = float("nan")
            lf_power_delta_valid = 0
        else:
            lf_power_delta = float(abs(fmri_signal_power - prev_power_global))
            lf_power_delta_valid = 1

        dynamic_delta_metrics: Dict[str, Any] = {}
        if compute_dynamic_deltas:
            modularity_val = float(epoch_metrics.get("fmri_modularity", float("nan")))
            fc_mean_val = float(epoch_metrics.get("fmri_FC_mean", float("nan")))
            if prev_modularity is not None and np.isfinite(prev_modularity) and np.isfinite(modularity_val):
                dynamic_delta_metrics["fmri_modularity_delta"] = float(abs(modularity_val - prev_modularity))
            else:
                dynamic_delta_metrics["fmri_modularity_delta"] = float("nan")
            if prev_fc_mean is not None and np.isfinite(prev_fc_mean) and np.isfinite(fc_mean_val):
                dynamic_delta_metrics["fmri_FC_mean_change_rate"] = float(abs(fc_mean_val - prev_fc_mean))
            else:
                dynamic_delta_metrics["fmri_FC_mean_change_rate"] = float("nan")
            if np.isfinite(modularity_val):
                prev_modularity = modularity_val
            if np.isfinite(fc_mean_val):
                prev_fc_mean = fc_mean_val

        records.append(
            {
                "epoch_id": int(epoch_id),
                "t_start": start_idx / sfreq,
                "t_end": end_idx / sfreq,
                "fmri_variance_global": fmri_variance_global,
                "fmri_region_var_mean": fmri_variance_global,
                "fmri_entropy_global": fmri_variance_global,
                "fmri_signal_power": fmri_signal_power,
                "fmri_lf_power": fmri_signal_power,
                "fmri_lf_power_delta": lf_power_delta,
                "fmri_lf_power_delta_valid": int(lf_power_delta_valid),
                "fmri_window_sec": window_sec,
                "fmri_step_sec": step_sec,
                "fmri_window_samples": int(window_samples),
                "fmri_step_samples": int(step_samples),
                "fmri_sfreq": float(sfreq),
                "fmri_tr_sec": float(fmri_tr_sec),
                "fmri_tr_source": fmri_tr_source,
                "fmri_filter_stage": fmri_filter_stage,
                "fmri_filter_applied": fmri_filter_applied,
                "fmri_filter_skipped": fmri_filter_skipped,
                "fmri_filter_order": fmri_filter_order,
                "fmri_filter_bandpass_low": fmri_filter_low,
                "fmri_filter_bandpass_high": fmri_filter_high,
                "fmri_nuisance_status": fmri_nuisance_status,
                "fmri_atlas_space_status": fmri_atlas_space_status,
                "fmri_atlas_ornt": fmri_atlas_ornt,
                "fmri_bold_ornt": fmri_bold_ornt,
                "fmri_atlas_affine_match": fmri_atlas_affine_match,
                "fmri_atlas_affine_atol_mm": fmri_atlas_affine_atol_mm,
                "dataset_id": dataset_id,
                **epoch_metrics,
                **dynamic_delta_metrics,
            }
        )

        if np.isfinite(fmri_signal_power):
            prev_power_global = fmri_signal_power

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        logger.info("Computed %d fMRI epochs", len(df))
    return df


def _validate_and_extract_data(signals: Mapping[str, Any]) -> np.ndarray:
    """Internal helper: validate and extract data."""
    sig_dict = signals.get("signals", {})
    if "fmri" not in sig_dict:
        return np.zeros((0, 0), dtype=float)
    fmri_data = np.asarray(sig_dict["fmri"], dtype=float)
    if fmri_data.ndim != 2:
        raise ValueError("signals['fmri'] must have shape (n_regions, n_times)")
    return fmri_data


def _compute_window_params(config: Mapping[str, Any], sfreq: float, dataset_id: Any) -> Tuple[float, float, int, int]:
    """Internal helper: compute window params."""
    features_cfg = config.get("features", {}) if isinstance(config, Mapping) else {}
    fmri_cfg = features_cfg.get("fmri", {}) if isinstance(features_cfg, Mapping) else {}
    epoching_cfg = config.get("epoching", {}) if isinstance(config, Mapping) else {}

    ds_overrides: Dict[str, Any] = {}
    if isinstance(fmri_cfg, Mapping):
        ds_map = fmri_cfg.get("datasets", {})
        if isinstance(ds_map, Mapping) and dataset_id in ds_map and isinstance(ds_map[dataset_id], Mapping):
            ds_overrides = dict(ds_map[dataset_id])  # type: ignore[arg-type]

    window_sec = float(
        ds_overrides.get("window_sec", fmri_cfg.get("window_sec", epoching_cfg.get("length_s", 30.0))) or 30.0
    )
    step_sec = float(ds_overrides.get("step_sec", fmri_cfg.get("step_sec", epoching_cfg.get("step_s", 15.0))) or 15.0)
    if not np.isfinite(window_sec) or not np.isfinite(step_sec) or not np.isfinite(sfreq) or sfreq <= 0:
        return window_sec, step_sec, 0, 0

    window_samples = max(2, int(np.floor(window_sec * sfreq)))
    step_samples = max(1, int(np.floor(step_sec * sfreq)))
    return window_sec, step_sec, window_samples, step_samples


def prepare_session_continuous(signals: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run session-level bandpass/Hilbert once for reuse across ROI subsets.

    Rows that are not fully finite are left NaN (no imputation). Finite rows
    are filtered independently along time, so a later ROI-index slice matches
    filtering that subset alone.
    """
    fmri_data = _validate_and_extract_data(signals)
    meta = signals.get("meta") if isinstance(signals.get("meta"), Mapping) else {}
    raw_sfreq = signals.get("sfreq")
    if raw_sfreq is None and isinstance(meta, Mapping):
        raw_sfreq = meta.get("sfreq")
    sfreq = float(raw_sfreq)
    continuous_cfg = _resolve_continuous_cfg(config)
    if isinstance(meta, Mapping) and bool(meta.get("filter_applied")):
        continuous_cfg["skip_bandpass"] = True
        continuous_cfg["filter_applied"] = True
        if meta.get("filter_stage"):
            continuous_cfg["filter_stage"] = str(meta.get("filter_stage"))
        upstream_band = meta.get("filter_bandpass")
        if isinstance(upstream_band, (list, tuple)) and len(upstream_band) >= 2:
            continuous_cfg["bandpass"] = [float(upstream_band[0]), float(upstream_band[1])]
    complete = np.isfinite(fmri_data).all(axis=1)
    if not np.any(complete):
        filtered = np.full(fmri_data.shape, np.nan, dtype=float)
        return {
            "filtered_ts": filtered,
            "phase_ts": np.full(fmri_data.shape, np.nan, dtype=float),
            "filter_applied": False,
            "filter_skipped": True,
            "filter_stage": "skipped_nonfinite",
            "filter_bandpass": continuous_cfg.get("bandpass") or [0.01, 0.1],
        }
    subset_out = fmri_continuous.process_session_signals(
        roi_ts=fmri_data[complete],
        sfreq=sfreq,
        config=continuous_cfg,
    )
    filtered = np.full(fmri_data.shape, np.nan, dtype=float)
    phase = np.full(fmri_data.shape, np.nan, dtype=float)
    filtered[complete] = np.asarray(subset_out.get("filtered_ts"), dtype=float)
    phase_ts = subset_out.get("phase_ts")
    if phase_ts is not None:
        phase[complete] = np.asarray(phase_ts, dtype=float)
    out = dict(subset_out)
    out["filtered_ts"] = filtered
    out["phase_ts"] = phase
    return out


def _resolve_continuous_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Internal helper: resolve continuous cfg."""
    result: Dict[str, Any] = {}
    if not isinstance(config, Mapping):
        return result

    preprocess_cfg = config.get("preprocess", {})
    if isinstance(preprocess_cfg, Mapping):
        legacy_band = preprocess_cfg.get("fmri_bandpass")
        if isinstance(legacy_band, (list, tuple)) and len(legacy_band) >= 2:
            result["bandpass"] = [float(legacy_band[0]), float(legacy_band[1])]
        preprocess_fmri = preprocess_cfg.get("fmri", {})
        if isinstance(preprocess_fmri, Mapping):
            band = preprocess_fmri.get("bandpass")
            if isinstance(band, (list, tuple)) and len(band) >= 2:
                result["bandpass"] = [float(band[0]), float(band[1])]

    # Optional direct override (preferred when provided).
    user_cfg = config.get("preprocessing")
    if isinstance(user_cfg, Mapping):
        for key in ("f_low", "f_high", "bandpass", "compute_phase"):
            if key in user_cfg:
                result[key] = user_cfg[key]
    return result


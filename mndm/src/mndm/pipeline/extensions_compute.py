"""MNPS extension computations: E-Kappa, RFM, O-Koh, TIG."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .. import extensions

logger = logging.getLogger(__name__)


def compute_extensions(
    dataset_label: str,
    extensions_cfg: Mapping[str, Any],
    x: np.ndarray,
    sub_frame: pd.DataFrame,
    time: np.ndarray,
    dt: float,
    coords_9d: Optional[np.ndarray],
    coords_9d_names: list[str],
    regions_bold: Optional[np.ndarray],
    regions_sfreq: Optional[float],
    group_ts: Dict[str, np.ndarray],
    group_matrix: Optional[np.ndarray],
    group_names: List[str],
    region_groups: Dict[str, List[int]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute all enabled MNPS extensions.

    Returns
    -------
    extensions_payload : dict
        Data to be stored in HDF5 (arrays, times, etc.)
    extensions_summary : dict
        Summary statistics for JSON manifest
    """
    extensions_payload: Dict[str, Any] = {}
    extensions_summary: Dict[str, Any] = {}
    regional_available = bool(
        regions_bold is not None
        and regions_sfreq
        and group_ts
        and group_matrix is not None
        and group_names
    )
    extensions_summary["regional_data_available"] = regional_available

    def _run(fn, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Internal helper: run."""
        payload: Dict[str, Any] = {}
        summary: Dict[str, Any] = {}
        fn(*args, extensions_payload=payload, extensions_summary=summary, **kwargs)
        return payload, summary

    tasks = {
        "e_kappa": lambda: _run(
            _compute_e_kappa,
            dataset_label=dataset_label,
            extensions_cfg=extensions_cfg,
            sub_frame=sub_frame,
            time=time,
            dt=dt,
            regions_bold=regions_bold,
            regions_sfreq=regions_sfreq,
            group_ts=group_ts,
            region_groups=region_groups,
        ),
        "rfm": lambda: _run(
            _compute_rfm,
            dataset_label=dataset_label,
            extensions_cfg=extensions_cfg,
            sub_frame=sub_frame,
            dt=dt,
            regions_sfreq=regions_sfreq,
            group_matrix=group_matrix,
        ),
        "o_koh": lambda: _run(
            _compute_o_koh,
            dataset_label=dataset_label,
            extensions_cfg=extensions_cfg,
            sub_frame=sub_frame,
            group_matrix=group_matrix,
            group_names=group_names,
        ),
        "tig": lambda: _run(
            _compute_tig,
            dataset_label=dataset_label,
            extensions_cfg=extensions_cfg,
            x=x,
            time=time,
            dt=dt,
            regions_sfreq=regions_sfreq,
            group_ts=group_ts,
            group_matrix=group_matrix,
            group_names=group_names,
            region_groups=region_groups,
        ),
    }

    with ThreadPoolExecutor(max_workers=4) as ex:
        future_map = {ex.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(future_map):
            name = future_map[fut]
            try:
                payload_part, summary_part = fut.result()
                extensions_payload.update(payload_part)
                extensions_summary.update(summary_part)
            except Exception:
                logger.exception("Extension %s failed for %s", name, dataset_label)

    if "rfm" not in extensions_summary and not regional_available:
        extensions_summary["rfm"] = {"status": "skipped_no_regional_bold"}
    if "o_koh" not in extensions_summary and not regional_available:
        extensions_summary["o_koh"] = {"status": "skipped_no_regional_bold"}
    if "tig" not in extensions_summary and not regional_available:
        extensions_summary["tig"] = {"status": "skipped_no_regional_bold"}

    return extensions_payload, extensions_summary


def _extract_band_timeseries(sub_frame: pd.DataFrame, band_name: str) -> Optional[np.ndarray]:
    """Extract EEG band time series from sub_frame."""
    base = f"eeg_{band_name}"
    cols = [
        c
        for c in sub_frame.columns
        if (c == base or c.startswith(base + "__g_")) and "__psd_alt" not in str(c)
    ]
    if not cols:
        return None
    data = np.stack([sub_frame[c].astype(float).to_numpy() for c in cols], axis=0)
    if data.shape[0] < 2 or data.shape[1] < 2:
        return None
    return data


def _compute_e_kappa(
    dataset_label: str,
    extensions_cfg: Mapping[str, Any],
    sub_frame: pd.DataFrame,
    time: np.ndarray,
    dt: float,
    regions_bold: Optional[np.ndarray],
    regions_sfreq: Optional[float],
    group_ts: Dict[str, np.ndarray],
    region_groups: Dict[str, List[int]],
    extensions_payload: Dict[str, Any],
    extensions_summary: Dict[str, Any],
) -> None:
    """Compute E-Kappa (energetic curvature) extension."""
    e_kappa_cfg = extensions_cfg.get("e_kappa", {}) if isinstance(extensions_cfg, Mapping) else {}
    if not isinstance(e_kappa_cfg, Mapping) or not e_kappa_cfg.get("enabled", False):
        return

    default_features = [
        "eeg_delta",
        "eeg_theta",
        "eeg_alpha",
        "eeg_beta",
        "eeg_gamma",
        "fmri_signal_power",
        "fmri_variance_global",
    ]
    feature_keys = e_kappa_cfg.get("features", default_features)
    weights_map = e_kappa_cfg.get("weights") or {}

    def _collect_series(names: Sequence[str]) -> Tuple[List[np.ndarray], List[float]]:
        """Internal helper: collect series."""
        cols: List[np.ndarray] = []
        wts: List[float] = []
        for name in names or []:
            if name in sub_frame.columns:
                vals = sub_frame[name].astype(float).to_numpy()
                cols.append(vals)
                wts.append(float(weights_map.get(name, 1.0)) if isinstance(weights_map, Mapping) else 1.0)
        return cols, wts

    series_list, weights_list = _collect_series(feature_keys)
    if not series_list:
        fmri_feature_keys = e_kappa_cfg.get("fmri_features", ["fmri_signal_power", "fmri_variance_global"])
        series_list, weights_list = _collect_series(fmri_feature_keys)

    if series_list:
        band_mat = np.stack(series_list, axis=0)
        weights_arr = np.asarray(weights_list, dtype=float)
        weights_arr = weights_arr / (weights_arr.sum() or 1.0)
        energy = np.tensordot(weights_arr, band_mat, axes=(0, 0))
        try:
            kappa = extensions.compute_kappa_E_1d(energy, dt=dt)
            extensions_payload["e_kappa"] = {
                "time": time.astype(np.float32),
                "energy": energy.astype(np.float32),
                "kappa": kappa.astype(np.float32),
            }
            finite_mask = np.isfinite(kappa)
            mean_kappa = float(np.nanmean(kappa[finite_mask])) if finite_mask.any() else float("nan")
            extensions_summary["e_kappa"] = {"mean_kappa": mean_kappa}
        except Exception:
            logger.exception("Failed to compute E-Kappa for %s", dataset_label)

    # fMRI E-Kappa
    fmri_ext_cfg = e_kappa_cfg.get("fmri", {}) if isinstance(e_kappa_cfg.get("fmri", {}), Mapping) else {}
    fmri_enabled = bool(fmri_ext_cfg.get("enabled", False))
    if fmri_enabled and regions_bold is not None and regions_sfreq:
        try:
            dt_fmri = 1.0 / float(regions_sfreq)
            energy_global = extensions.compute_energy_from_regions(regions_bold)
            kappa_global = extensions.compute_kappa_E_1d(energy_global, dt=dt_fmri)
            bold_time = (np.arange(regions_bold.shape[1], dtype=float) / float(regions_sfreq)).astype(np.float32)
            ek_group = extensions_payload.setdefault("e_kappa_fmri", {})
            ek_group["global"] = {
                "time": bold_time,
                "energy": energy_global.astype(np.float32),
                "kappa": kappa_global.astype(np.float32),
            }
            finite_mask = np.isfinite(kappa_global)
            mean_kappa = float(np.nanmean(kappa_global[finite_mask])) if finite_mask.any() else float("nan")
            extensions_summary.setdefault("e_kappa_fmri", {})["global_mean_kappa"] = mean_kappa

            if fmri_ext_cfg.get("regional", True) and group_ts:
                min_regions = int(fmri_ext_cfg.get("min_regions", 1))
                regional_payload: Dict[str, Any] = {}
                for label, ts in group_ts.items():
                    region_count = len(region_groups.get(label, []))
                    if region_count < min_regions:
                        continue
                    energy_reg = extensions.compute_energy_from_regions(ts[None, :])
                    try:
                        kappa_reg = extensions.compute_kappa_E_1d(energy_reg, dt=dt_fmri)
                    except Exception:
                        continue
                    regional_payload[label] = {
                        "time": bold_time,
                        "energy": energy_reg.astype(np.float32),
                        "kappa": kappa_reg.astype(np.float32),
                        "n_regions": int(region_count),
                    }
                if regional_payload:
                    ek_group["regional"] = regional_payload
        except Exception:
            logger.exception("Failed to compute fMRI E-Kappa for %s", dataset_label)


def _compute_rfm(
    dataset_label: str,
    extensions_cfg: Mapping[str, Any],
    sub_frame: pd.DataFrame,
    dt: float,
    regions_sfreq: Optional[float],
    group_matrix: Optional[np.ndarray],
    extensions_payload: Dict[str, Any],
    extensions_summary: Dict[str, Any],
) -> None:
    """Compute RFM (resonant frequency modes) extension."""
    rfm_cfg = extensions_cfg.get("rfm", {}) if isinstance(extensions_cfg, Mapping) else {}
    if not isinstance(rfm_cfg, Mapping) or not rfm_cfg.get("enabled", False):
        return

    source_mode = str(rfm_cfg.get("source", "auto")).lower()
    rf_signals = None
    signal_source = None

    if source_mode in ("auto", "eeg"):
        band_for_rfm = rfm_cfg.get("band")
        band_name = str(band_for_rfm) if isinstance(band_for_rfm, str) else "alpha"
        rf_signals = _extract_band_timeseries(sub_frame, band_name) if band_name else None
        if rf_signals is not None:
            signal_source = "eeg"

    if rf_signals is None and source_mode in ("auto", "fmri"):
        if group_matrix is not None and regions_sfreq:
            rf_signals = group_matrix
            signal_source = "fmri"

    if rf_signals is not None and signal_source:
        try:
            if signal_source == "fmri":
                sfreq_rfm = float(regions_sfreq)
                window_sec = float(rfm_cfg.get("fmri_window_sec", 60.0))
                step_sec = float(rfm_cfg.get("fmri_step_sec", 30.0))
                apply_band = None
            else:
                sfreq_rfm = 1.0 / dt
                window_sec = float(rfm_cfg.get("window_sec", 32.0))
                step_sec = float(rfm_cfg.get("step_sec", 16.0))
                apply_band = None
            n_modes = int(rfm_cfg.get("n_modes", 3))
            rfm_res = extensions.compute_rfm(
                rf_signals,
                sfreq=sfreq_rfm,
                window_sec=window_sec,
                step_sec=step_sec,
                n_modes=n_modes,
                band=apply_band,
            )
            payload_key = "rfm_fmri" if signal_source == "fmri" else "rfm"
            extensions_payload[payload_key] = {
                "times": rfm_res.times.astype(np.float32),
                "eigvals": rfm_res.eigvals.astype(np.float32),
                "eigvecs": rfm_res.eigvecs.astype(np.float32),
                "dominance": rfm_res.dominance.astype(np.float32),
                "source": signal_source,
            }
            summary_key = "rfm_fmri" if signal_source == "fmri" else "rfm"
            extensions_summary[summary_key] = {
                "mean_dominance": float(np.nanmean(rfm_res.dominance)) if rfm_res.dominance.size else float("nan"),
                "source": signal_source,
            }
        except Exception:
            logger.exception("Failed to compute RFM for %s", dataset_label)


def _compute_o_koh(
    dataset_label: str,
    extensions_cfg: Mapping[str, Any],
    sub_frame: pd.DataFrame,
    group_matrix: Optional[np.ndarray],
    group_names: List[str],
    extensions_payload: Dict[str, Any],
    extensions_summary: Dict[str, Any],
) -> None:
    """Compute O-Koh (organizational coherence) extension."""
    o_koh_cfg = extensions_cfg.get("o_koh", {}) if isinstance(extensions_cfg, Mapping) else {}
    if not isinstance(o_koh_cfg, Mapping) or not o_koh_cfg.get("enabled", False):
        return

    band_for_ok = o_koh_cfg.get("band", "alpha")
    ok_signals = _extract_band_timeseries(sub_frame, str(band_for_ok))
    if ok_signals is not None:
        try:
            C = np.corrcoef(ok_signals)
            thresholds = o_koh_cfg.get("thresholds", None)
            ok_res = extensions.compute_OKoh(C, thresholds=thresholds, weights=None)
            extensions_payload["o_koh"] = {
                "thresholds": ok_res["thresholds"].astype(np.float32),
                "beta0": ok_res["beta0"].astype(np.float32),
                "beta1": ok_res["beta1"].astype(np.float32),
                "OKoh0": float(ok_res["OKoh0"]),
                "OKoh1": float(ok_res["OKoh1"]),
            }
            extensions_summary["o_koh"] = {
                "OKoh0": float(ok_res["OKoh0"]),
                "OKoh1": float(ok_res["OKoh1"]),
            }
        except Exception:
            logger.exception("Failed to compute O-Koh for %s", dataset_label)

    # fMRI O-Koh
    fmri_ok_cfg = o_koh_cfg.get("fmri", {}) if isinstance(o_koh_cfg.get("fmri", {}), Mapping) else {}
    if bool(fmri_ok_cfg.get("enabled", False)) and group_matrix is not None and group_matrix.shape[0] >= 2:
        try:
            C = np.corrcoef(group_matrix)
            thresholds = fmri_ok_cfg.get("thresholds")
            ok_res = extensions.compute_OKoh(C, thresholds=thresholds, weights=None)
            extensions_payload["o_koh_fmri"] = {
                "thresholds": ok_res["thresholds"].astype(np.float32),
                "beta0": ok_res["beta0"].astype(np.float32),
                "beta1": ok_res["beta1"].astype(np.float32),
                "OKoh0": float(ok_res["OKoh0"]),
                "OKoh1": float(ok_res["OKoh1"]),
                "networks": np.asarray(group_names, dtype="S"),
            }
            extensions_summary["o_koh_fmri"] = {
                "OKoh0": float(ok_res["OKoh0"]),
                "OKoh1": float(ok_res["OKoh1"]),
            }
        except Exception:
            logger.exception("Failed to compute fMRI O-Koh for %s", dataset_label)


def _compute_tig(
    dataset_label: str,
    extensions_cfg: Mapping[str, Any],
    x: np.ndarray,
    time: np.ndarray,
    dt: float,
    regions_sfreq: Optional[float],
    group_ts: Dict[str, np.ndarray],
    group_matrix: Optional[np.ndarray],
    group_names: List[str],
    region_groups: Dict[str, List[int]],
    extensions_payload: Dict[str, Any],
    extensions_summary: Dict[str, Any],
) -> None:
    """Compute TIG (temporal integrity grade) extension."""
    tig_cfg = extensions_cfg.get("tig", {}) if isinstance(extensions_cfg, Mapping) else {}
    if not isinstance(tig_cfg, Mapping) or not tig_cfg.get("enabled", False):
        return

    if x.size:
        try:
            max_lag_cfg = float(tig_cfg.get("max_lag_sec", 60.0))
            n_lags = int(tig_cfg.get("n_lags", 20))
            max_possible = max(dt * (max(len(time) - 1, 1)), dt)
            max_lag_sec = min(max_lag_cfg, max_possible)
            if max_lag_sec > 0:
                tig_res = extensions.compute_TIG_autocorr(
                    x,
                    dt=dt,
                    max_lag_sec=max_lag_sec,
                    n_lags=n_lags,
                    T_max=max_lag_cfg,
                )
                provisional_flag = bool(tig_res.get("provisional", False))
                extensions_payload["tig"] = {
                    "lags_sec": tig_res["lags_sec"].astype(np.float32),
                    "autocorr": tig_res["autocorr"].astype(np.float32),
                    "tau": float(tig_res["tau"]),
                    "TIG": float(tig_res["TIG"]),
                    "provisional": provisional_flag,
                }
                extensions_summary["tig"] = {
                    "tau": float(tig_res["tau"]),
                    "TIG": float(tig_res["TIG"]),
                    "provisional": provisional_flag,
                }
        except Exception:
            logger.exception("Failed to compute TIG for %s", dataset_label)

    # fMRI TIG (global)
    tig_cfg_fmri = tig_cfg.get("fmri", {}) if isinstance(tig_cfg.get("fmri", {}), Mapping) else {}
    if regions_sfreq and bool(tig_cfg_fmri.get("global", False)) and group_matrix is not None:
        try:
            dt_fmri = 1.0 / float(regions_sfreq)
            duration = dt_fmri * max(group_matrix.shape[1] - 1, 1)
            max_lag_sec_cfg = float(tig_cfg_fmri.get("max_lag_sec", tig_cfg.get("max_lag_sec", 60.0)))
            max_lag_sec = min(max_lag_sec_cfg, duration)
            n_lags = int(tig_cfg_fmri.get("n_lags", tig_cfg.get("n_lags", 20)))
            tig_res = extensions.compute_TIG_autocorr(
                group_matrix.T,
                dt=dt_fmri,
                max_lag_sec=max_lag_sec,
                n_lags=n_lags,
                T_max=float(tig_cfg_fmri.get("T_max", max_lag_sec)),
            )
            payload_key = extensions_payload.setdefault("tig_fmri", {})
            payload_key["global"] = {
                "lags_sec": tig_res["lags_sec"].astype(np.float32),
                "autocorr": tig_res["autocorr"].astype(np.float32),
                "tau": float(tig_res["tau"]),
                "TIG": float(tig_res["TIG"]),
                "provisional": bool(tig_res.get("provisional", False)),
                "networks": np.asarray(group_names, dtype="S"),
            }
            extensions_summary.setdefault("tig_fmri", {})["global_tau"] = float(tig_res["tau"])
        except Exception:
            logger.exception("Failed to compute fMRI TIG (global) for %s", dataset_label)

    # fMRI TIG (regional)
    if regions_sfreq and bool(tig_cfg_fmri.get("regional", False)) and group_ts:
        dt_fmri = 1.0 / float(regions_sfreq)
        max_lag_sec_cfg = float(tig_cfg_fmri.get("max_lag_sec", tig_cfg.get("max_lag_sec", 60.0)))
        n_lags = int(tig_cfg_fmri.get("n_lags", tig_cfg.get("n_lags", 20)))
        regional_payload: Dict[str, Any] = {}
        for label, ts in group_ts.items():
            if ts.size < 3:
                continue
            try:
                duration = dt_fmri * max(ts.shape[0] - 1, 1)
                max_lag_sec = min(max_lag_sec_cfg, duration)
                tig_res = extensions.compute_TIG_autocorr(
                    ts[:, None],
                    dt=dt_fmri,
                    max_lag_sec=max_lag_sec,
                    n_lags=n_lags,
                    T_max=float(tig_cfg_fmri.get("T_max", max_lag_sec)),
                )
                regional_payload[label] = {
                    "lags_sec": tig_res["lags_sec"].astype(np.float32),
                    "autocorr": tig_res["autocorr"].astype(np.float32),
                    "tau": float(tig_res["tau"]),
                    "TIG": float(tig_res["TIG"]),
                    "provisional": bool(tig_res.get("provisional", False)),
                    "n_regions": int(len(region_groups.get(label, []))),
                }
            except Exception:
                logger.exception("Failed TIG for %s network %s", dataset_label, label)
        if regional_payload:
            payload_key = extensions_payload.setdefault("tig_fmri", {})
            payload_key["regional"] = regional_payload


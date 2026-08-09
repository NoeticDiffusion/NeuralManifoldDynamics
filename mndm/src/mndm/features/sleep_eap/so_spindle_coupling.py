"""SO–spindle temporal coupling measures with explicit no-partner semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def couple_spindles_to_slow_oscillations(
    spindles: pd.DataFrame,
    slow_oscillations: pd.DataFrame,
    *,
    max_abs_latency_sec: float = 2.0,
    score_decay_sec: float = 0.5,
    so_phase: np.ndarray | None = None,
    sfreq: float | None = None,
) -> pd.DataFrame:
    """Attach nearest valid SO-up-state metrics to each spindle.

    ``so_spindle_latency_sec`` is spindle peak minus SO up-state time.  No
    partner is represented by NaN metrics and ``so_partner_missing=1`` rather
    than a numerical zero, so absence cannot be interpreted as synchrony.
    """
    if "onset_sec" not in spindles:
        raise ValueError("spindles must contain onset_sec")
    out = spindles.copy()
    columns = {
        "so_onset_sec": np.full(len(out), np.nan),
        "so_upstate_sec": np.full(len(out), np.nan),
        "so_amplitude": np.full(len(out), np.nan),
        "so_phase_at_spindle_peak": np.full(len(out), np.nan),
        "so_spindle_latency_sec": np.full(len(out), np.nan),
        "so_spindle_coupling_score": np.full(len(out), np.nan),
        "so_partner_missing": np.ones(len(out), dtype=np.int8),
        "so_coupling_qc_flag": np.full(len(out), "no_partner", dtype=object),
    }
    for name, values in columns.items():
        out[name] = values
    if slow_oscillations.empty:
        return out

    upstate_column = "upstate_sec" if "upstate_sec" in slow_oscillations else "peak_sec"
    if upstate_column not in slow_oscillations:
        return out
    so = slow_oscillations.copy()
    so["_upstate"] = pd.to_numeric(so[upstate_column], errors="coerce")
    so = so[np.isfinite(so["_upstate"])].sort_values("_upstate").reset_index(drop=True)
    if so.empty:
        return out
    upstates = so["_upstate"].to_numpy(dtype=float)
    peaks = pd.to_numeric(out.get("peak_sec", out["onset_sec"]), errors="coerce").to_numpy(dtype=float)

    for row_index, peak in enumerate(peaks):
        if not np.isfinite(peak):
            out.at[out.index[row_index], "so_coupling_qc_flag"] = "invalid_spindle_peak"
            continue
        nearest = int(np.argmin(np.abs(upstates - peak)))
        latency = float(peak - upstates[nearest])
        if abs(latency) > max_abs_latency_sec:
            out.at[out.index[row_index], "so_coupling_qc_flag"] = "partner_outside_tolerance"
            continue
        partner = so.iloc[nearest]
        target_index = out.index[row_index]
        out.at[target_index, "so_onset_sec"] = float(partner.get("onset_sec", np.nan))
        out.at[target_index, "so_upstate_sec"] = upstates[nearest]
        out.at[target_index, "so_amplitude"] = float(partner.get("amplitude", np.nan))
        out.at[target_index, "so_spindle_latency_sec"] = latency
        out.at[target_index, "so_spindle_coupling_score"] = float(np.exp(-abs(latency) / score_decay_sec))
        out.at[target_index, "so_partner_missing"] = 0
        out.at[target_index, "so_coupling_qc_flag"] = "ok"
        if so_phase is not None and sfreq is not None and sfreq > 0:
            phase_index = int(round(peak * sfreq))
            if 0 <= phase_index < len(so_phase):
                out.at[target_index, "so_phase_at_spindle_peak"] = float(so_phase[phase_index])
    return out

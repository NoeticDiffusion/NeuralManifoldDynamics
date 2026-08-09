"""Versioned event-level phase sidecars for N3 slow oscillations and REM theta.

These builders deliberately do not share the spindle event-phase schema.  N3
rows describe detected slow oscillations, while REM rows describe 30-second
stage epochs.  Neither product estimates spindle strength or SO--spindle
coupling.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import EVENT_PHASE_N3_SO_V1_CONTRACT, EVENT_PHASE_REM_THETA_V1_CONTRACT
from .phase_continuous import PhaseContinuousState, sample_phase_state


_N3_SO_COLUMNS = (
    "subject",
    "event_id",
    "onset_sec",
    "upstate_sec",
    "duration_sec",
    "amplitude",
    "stage",
    "event_type",
    "source",
    "so_qc_flag",
    "phi_so_trough",
    "phi_so_upstate",
    "so_phase_valid_trough",
    "so_phase_valid_upstate",
    "phi_cardiac_trough",
    "phi_resp_trough",
    "cardiac_phase_valid_trough",
    "resp_phase_valid_trough",
    "rr_interval_ms_at_trough",
    "inhale_at_trough",
    "autonomic_phase_eligible_trough",
    "phase_coupling_qc_flag_trough",
    "phi_cardiac_upstate",
    "phi_resp_upstate",
    "cardiac_phase_valid_upstate",
    "resp_phase_valid_upstate",
    "rr_interval_ms_at_upstate",
    "inhale_at_upstate",
    "autonomic_phase_eligible_upstate",
    "phase_coupling_qc_flag_upstate",
    "event_phase_contract",
    "phase_continuous_contract",
    "carrier_source",
    "carrier_band_low_hz",
    "carrier_band_high_hz",
    "phase_sample_hz",
)

_REM_THETA_COLUMNS = (
    "subject",
    "epoch_id",
    "onset_sec",
    "duration_sec",
    "reference_sec",
    "stage",
    "event_type",
    "phi_rem_theta",
    "theta_phase_valid",
    "phi_cardiac",
    "phi_resp",
    "cardiac_phase_valid",
    "resp_phase_valid",
    "rr_interval_ms_at_reference",
    "inhale_at_reference",
    "autonomic_phase_eligible",
    "phase_coupling_qc_flag",
    "event_phase_contract",
    "phase_continuous_contract",
    "carrier_source",
    "carrier_band_low_hz",
    "carrier_band_high_hz",
    "phase_sample_hz",
)


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """Return an empty schema-stable sidecar frame."""
    return pd.DataFrame(columns=list(columns))


def _sample_full_rate_phase(
    phase: np.ndarray,
    timestamps_sec: np.ndarray,
    *,
    sfreq: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-sample carrier-phase lookup with an explicit bounds validity flag."""
    values = np.asarray(phase, dtype=float).ravel()
    timestamps = np.asarray(timestamps_sec, dtype=float).ravel()
    sampled = np.full(timestamps.shape, np.nan, dtype=float)
    valid = np.zeros(timestamps.shape, dtype=bool)
    if values.size == 0 or sfreq <= 0:
        return sampled, valid
    indices = np.rint(timestamps * float(sfreq)).astype(np.int64)
    in_bounds = (indices >= 0) & (indices < values.size)
    if valid_mask is None:
        source_valid = np.isfinite(values)
    else:
        source_valid = np.asarray(valid_mask, dtype=bool).ravel() & np.isfinite(values)
        if source_valid.shape != values.shape:
            raise ValueError("carrier phase and validity mask must have the same length")
    valid[in_bounds] = source_valid[indices[in_bounds]]
    sampled[valid] = values[indices[valid]]
    return sampled, valid


def _sample_autonomic_state(
    state: PhaseContinuousState,
    timestamps_sec: np.ndarray,
) -> pd.DataFrame:
    """Sample the continuous phase state without allowing out-of-range clipping."""
    timestamps = np.asarray(timestamps_sec, dtype=float).ravel()
    sampled = sample_phase_state(state, timestamps)
    in_bounds = np.zeros(timestamps.shape, dtype=bool)
    if state.time_sec.size:
        in_bounds = (timestamps >= state.time_sec[0]) & (timestamps <= state.time_sec[-1])
    for value_col, valid_col in (
        ("phi_cardiac", "cardiac_phase_valid"),
        ("phi_resp", "resp_phase_valid"),
    ):
        valid = sampled[valid_col].to_numpy(dtype=bool) & in_bounds
        sampled[valid_col] = valid
        sampled.loc[~valid, value_col] = np.nan
    return sampled


def _rr_interval_ms_at(state: PhaseContinuousState, timestamps_sec: np.ndarray) -> np.ndarray:
    """Return enclosing RR intervals, with missing brackets represented by NaN."""
    timestamps = np.asarray(timestamps_sec, dtype=float).ravel()
    rpeaks = np.asarray(state.rpeaks_sec, dtype=float).ravel()
    output = np.full(timestamps.shape, np.nan, dtype=float)
    if rpeaks.size < 2:
        return output
    index = np.searchsorted(rpeaks, timestamps, side="right") - 1
    valid = (index >= 0) & (index < rpeaks.size - 1)
    output[valid] = (rpeaks[index[valid] + 1] - rpeaks[index[valid]]) * 1000.0
    return output


def _phase_qc_flag(
    *,
    carrier_valid: bool,
    cardiac_valid: bool,
    resp_valid: bool,
) -> str:
    """Encode the joint carrier/autonomic availability without sentinel values."""
    missing: list[str] = []
    if not carrier_valid:
        missing.append("carrier")
    if not cardiac_valid:
        missing.append("cardiac")
    if not resp_valid:
        missing.append("resp")
    return "ok" if not missing else "missing_" + "_and_".join(missing)


def _autonomic_fields(
    sampled: pd.DataFrame,
    state: PhaseContinuousState,
    timestamps_sec: np.ndarray,
    *,
    carrier_valid: np.ndarray,
    suffix: str,
) -> dict[str, np.ndarray]:
    """Build standard autonomic fields for a named N3 reference point."""
    cardiac_valid = sampled["cardiac_phase_valid"].to_numpy(dtype=bool)
    resp_valid = sampled["resp_phase_valid"].to_numpy(dtype=bool)
    eligible = cardiac_valid & resp_valid & carrier_valid
    phi_resp = sampled["phi_resp"].to_numpy(dtype=float)
    inhale = np.where(resp_valid, (phi_resp < 0.0).astype(float), np.nan)
    return {
        f"phi_cardiac_{suffix}": sampled["phi_cardiac"].to_numpy(dtype=float),
        f"phi_resp_{suffix}": phi_resp,
        f"cardiac_phase_valid_{suffix}": cardiac_valid.astype(np.int8),
        f"resp_phase_valid_{suffix}": resp_valid.astype(np.int8),
        f"rr_interval_ms_at_{suffix}": _rr_interval_ms_at(state, timestamps_sec),
        f"inhale_at_{suffix}": inhale,
        f"autonomic_phase_eligible_{suffix}": eligible.astype(np.int8),
        f"phase_coupling_qc_flag_{suffix}": np.asarray(
            [
                _phase_qc_flag(
                    carrier_valid=bool(carrier),
                    cardiac_valid=bool(cardiac),
                    resp_valid=bool(resp),
                )
                for carrier, cardiac, resp in zip(carrier_valid, cardiac_valid, resp_valid)
            ],
            dtype=object,
        ),
    }


def build_n3_slow_oscillation_event_phase(
    slow_oscillations: pd.DataFrame,
    *,
    phase_state: PhaseContinuousState,
    so_phase: np.ndarray,
    sfreq: float,
    subject: str,
    n3_stage_code: int = 3,
    carrier_band_hz: tuple[float, float] = (0.3, 1.25),
    contract: str = EVENT_PHASE_N3_SO_V1_CONTRACT,
) -> pd.DataFrame:
    """Build one N3-only row per detected slow oscillation.

    The two reference points are the detector's trough (``onset_sec``) and
    following up-state.  Missing autonomic phase remains NaN and is recorded in
    point-specific QC fields rather than being coerced to zero.
    """
    required = {"onset_sec", "upstate_sec", "stage"}
    if slow_oscillations.empty or not required.issubset(slow_oscillations.columns):
        return _empty_frame(_N3_SO_COLUMNS)

    rows = slow_oscillations.copy()
    stages = pd.to_numeric(rows["stage"], errors="coerce")
    rows = rows.loc[stages == int(n3_stage_code)].reset_index(drop=True)
    if rows.empty:
        return _empty_frame(_N3_SO_COLUMNS)

    trough = pd.to_numeric(rows["onset_sec"], errors="coerce").to_numpy(dtype=float)
    upstate = pd.to_numeric(rows["upstate_sec"], errors="coerce").to_numpy(dtype=float)
    phi_trough, valid_trough = _sample_full_rate_phase(so_phase, trough, sfreq=sfreq)
    phi_upstate, valid_upstate = _sample_full_rate_phase(so_phase, upstate, sfreq=sfreq)
    trough_state = _sample_autonomic_state(phase_state, trough)
    upstate_state = _sample_autonomic_state(phase_state, upstate)

    output = pd.DataFrame(
        {
            "subject": str(subject),
            "event_id": np.arange(rows.shape[0], dtype=np.int64),
            "onset_sec": trough,
            "upstate_sec": upstate,
            "duration_sec": pd.to_numeric(
                rows.get("duration_sec", pd.Series(np.nan, index=rows.index)),
                errors="coerce",
            ),
            "amplitude": pd.to_numeric(
                rows.get("amplitude", pd.Series(np.nan, index=rows.index)),
                errors="coerce",
            ),
            "stage": int(n3_stage_code),
            "event_type": "n3_slow_oscillation",
            "source": rows.get("source", pd.Series("detector:sleep_eap_so_v1", index=rows.index)).astype(str),
            "so_qc_flag": rows.get("so_qc_flag", pd.Series("ok", index=rows.index)).astype(str),
            "phi_so_trough": phi_trough,
            "phi_so_upstate": phi_upstate,
            "so_phase_valid_trough": valid_trough.astype(np.int8),
            "so_phase_valid_upstate": valid_upstate.astype(np.int8),
            "event_phase_contract": str(contract),
            "phase_continuous_contract": str(phase_state.contract),
            "carrier_source": "slow_oscillation_phase",
            "carrier_band_low_hz": float(carrier_band_hz[0]),
            "carrier_band_high_hz": float(carrier_band_hz[1]),
            "phase_sample_hz": float(phase_state.sample_hz),
        }
    )
    for name, values in _autonomic_fields(
        trough_state,
        phase_state,
        trough,
        carrier_valid=valid_trough,
        suffix="trough",
    ).items():
        output[name] = values
    for name, values in _autonomic_fields(
        upstate_state,
        phase_state,
        upstate,
        carrier_valid=valid_upstate,
        suffix="upstate",
    ).items():
        output[name] = values
    return output.loc[:, list(_N3_SO_COLUMNS)]


def build_rem_theta_event_phase(
    stage_codes: Sequence[int],
    *,
    phase_state: PhaseContinuousState,
    theta_phase: np.ndarray,
    theta_valid: np.ndarray,
    sfreq: float,
    subject: str,
    stage_epoch_sec: float = 30.0,
    rem_stage_code: int = 4,
    carrier_band_hz: tuple[float, float] = (4.0, 8.0),
    contract: str = EVENT_PHASE_REM_THETA_V1_CONTRACT,
) -> pd.DataFrame:
    """Build one REM-theta row per scored REM epoch at its 30-second midpoint."""
    stage_array = np.asarray(stage_codes, dtype=float).ravel()
    rem_indices = np.flatnonzero(stage_array == int(rem_stage_code))
    if rem_indices.size == 0:
        return _empty_frame(_REM_THETA_COLUMNS)

    onset = rem_indices.astype(float) * float(stage_epoch_sec)
    reference = onset + float(stage_epoch_sec) / 2.0
    sampled_theta, theta_is_valid = _sample_full_rate_phase(
        theta_phase,
        reference,
        sfreq=sfreq,
        valid_mask=theta_valid,
    )
    sampled_state = _sample_autonomic_state(phase_state, reference)
    cardiac_valid = sampled_state["cardiac_phase_valid"].to_numpy(dtype=bool)
    resp_valid = sampled_state["resp_phase_valid"].to_numpy(dtype=bool)
    eligible = theta_is_valid & cardiac_valid & resp_valid
    phi_resp = sampled_state["phi_resp"].to_numpy(dtype=float)

    output = pd.DataFrame(
        {
            "subject": str(subject),
            "epoch_id": rem_indices.astype(np.int64),
            "onset_sec": onset,
            "duration_sec": float(stage_epoch_sec),
            "reference_sec": reference,
            "stage": int(rem_stage_code),
            "event_type": "rem_theta_epoch",
            "phi_rem_theta": sampled_theta,
            "theta_phase_valid": theta_is_valid.astype(np.int8),
            "phi_cardiac": sampled_state["phi_cardiac"].to_numpy(dtype=float),
            "phi_resp": phi_resp,
            "cardiac_phase_valid": cardiac_valid.astype(np.int8),
            "resp_phase_valid": resp_valid.astype(np.int8),
            "rr_interval_ms_at_reference": _rr_interval_ms_at(phase_state, reference),
            "inhale_at_reference": np.where(resp_valid, (phi_resp < 0.0).astype(float), np.nan),
            "autonomic_phase_eligible": eligible.astype(np.int8),
            "phase_coupling_qc_flag": np.asarray(
                [
                    _phase_qc_flag(
                        carrier_valid=bool(carrier),
                        cardiac_valid=bool(cardiac),
                        resp_valid=bool(resp),
                    )
                    for carrier, cardiac, resp in zip(theta_is_valid, cardiac_valid, resp_valid)
                ],
                dtype=object,
            ),
            "event_phase_contract": str(contract),
            "phase_continuous_contract": str(phase_state.contract),
            "carrier_source": "rem_theta_phase",
            "carrier_band_low_hz": float(carrier_band_hz[0]),
            "carrier_band_high_hz": float(carrier_band_hz[1]),
            "phase_sample_hz": float(phase_state.sample_hz),
        }
    )
    return output.loc[:, list(_REM_THETA_COLUMNS)]


"""Shared helpers for interval/window overlap and membership decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

DEFAULT_WINDOW_MEMBERSHIP_MODE = "midpoint_in_interval"
DEFAULT_POINT_ATOL_SEC = 1e-3


@dataclass(frozen=True)
class TimeInterval:
    """Absolute time interval on the run clock."""

    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class WindowMembershipSpec:
    """How a window is considered part of an interval."""

    mode: str = DEFAULT_WINDOW_MEMBERSHIP_MODE
    min_overlap_fraction: float = 0.0
    point_atol_sec: float = DEFAULT_POINT_ATOL_SEC


def normalize_window_membership_mode(value: Any) -> str:
    """Return a canonical membership mode name."""
    raw = str(value or DEFAULT_WINDOW_MEMBERSHIP_MODE).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "midpoint": "midpoint_in_interval",
        "midpoint_in": "midpoint_in_interval",
        "midpoint_in_interval": "midpoint_in_interval",
        "contains_onset": "contains_onset",
        "contains_point": "contains_onset",
        "overlaps": "overlaps",
        "overlap": "overlaps",
        "overlap_frac_ge": "overlap_frac_ge",
        "min_overlap_fraction": "overlap_frac_ge",
        "fully_contained": "fully_contained",
        "contained": "fully_contained",
        "within_interval": "fully_contained",
    }
    return aliases.get(raw, DEFAULT_WINDOW_MEMBERSHIP_MODE)


def membership_spec_from_mapping(
    cfg: Mapping[str, Any] | None,
    *,
    default_mode: str = DEFAULT_WINDOW_MEMBERSHIP_MODE,
    default_min_overlap_fraction: float = 0.0,
    default_point_atol_sec: float = DEFAULT_POINT_ATOL_SEC,
) -> WindowMembershipSpec:
    """Build a membership spec from a config mapping."""
    if not isinstance(cfg, Mapping):
        return WindowMembershipSpec(
            mode=normalize_window_membership_mode(default_mode),
            min_overlap_fraction=float(default_min_overlap_fraction),
            point_atol_sec=float(default_point_atol_sec),
        )
    return WindowMembershipSpec(
        mode=normalize_window_membership_mode(cfg.get("mode", default_mode)),
        min_overlap_fraction=float(cfg.get("min_overlap_fraction", default_min_overlap_fraction) or 0.0),
        point_atol_sec=float(cfg.get("point_atol_sec", default_point_atol_sec) or default_point_atol_sec),
    )


def overlap_sec(
    window_start: np.ndarray | float,
    window_end: np.ndarray | float,
    interval_start: float,
    interval_end: float,
) -> np.ndarray:
    """Return overlap length in seconds between windows and one interval."""
    w_start = np.asarray(window_start, dtype=float)
    w_end = np.asarray(window_end, dtype=float)
    overlap_lo = np.maximum(w_start, float(interval_start))
    overlap_hi = np.minimum(w_end, float(interval_end))
    return np.maximum(0.0, overlap_hi - overlap_lo)


def overlap_frac(
    window_start: np.ndarray | float,
    window_end: np.ndarray | float,
    interval_start: float,
    interval_end: float,
) -> np.ndarray:
    """Return overlap fraction relative to each window duration."""
    w_start = np.asarray(window_start, dtype=float)
    w_end = np.asarray(window_end, dtype=float)
    dur = np.maximum(0.0, w_end - w_start)
    ov = overlap_sec(w_start, w_end, interval_start, interval_end)
    out = np.zeros_like(dur, dtype=float)
    valid = dur > 0
    out[valid] = ov[valid] / dur[valid]
    return out


def point_membership_mask(
    *,
    point_sec: float,
    t_start: np.ndarray,
    t_end: np.ndarray,
    t_mid: np.ndarray,
    atol_sec: float = DEFAULT_POINT_ATOL_SEC,
) -> np.ndarray:
    """Return windows that contain a point-event onset."""
    point = float(point_sec)
    mask = (t_start <= point) & (t_end > point)
    if mask.any():
        return mask
    return np.isclose(t_mid, point, atol=float(atol_sec))


def window_membership_mask(
    *,
    t_start: np.ndarray,
    t_end: np.ndarray,
    t_mid: np.ndarray,
    interval: TimeInterval,
    spec: WindowMembershipSpec | None = None,
) -> np.ndarray:
    """Return per-window membership mask for an interval."""
    spec = spec or WindowMembershipSpec()
    start = float(interval.start_sec)
    end = float(interval.end_sec)
    if end <= start:
        return point_membership_mask(
            point_sec=start,
            t_start=np.asarray(t_start, dtype=float),
            t_end=np.asarray(t_end, dtype=float),
            t_mid=np.asarray(t_mid, dtype=float),
            atol_sec=spec.point_atol_sec,
        )

    t_start_arr = np.asarray(t_start, dtype=float)
    t_end_arr = np.asarray(t_end, dtype=float)
    t_mid_arr = np.asarray(t_mid, dtype=float)
    mode = normalize_window_membership_mode(spec.mode)

    if mode == "contains_onset":
        return point_membership_mask(
            point_sec=start,
            t_start=t_start_arr,
            t_end=t_end_arr,
            t_mid=t_mid_arr,
            atol_sec=spec.point_atol_sec,
        )
    if mode == "overlaps":
        return overlap_sec(t_start_arr, t_end_arr, start, end) > 0.0
    if mode == "overlap_frac_ge":
        return overlap_frac(t_start_arr, t_end_arr, start, end) >= float(spec.min_overlap_fraction)
    if mode == "fully_contained":
        return (t_start_arr >= start) & (t_end_arr <= end)
    return (t_mid_arr >= start) & (t_mid_arr < end)


def event_window_mask(
    *,
    onset: float,
    duration: float,
    t_start: np.ndarray,
    t_end: np.ndarray,
    t_mid: np.ndarray,
    interval_spec: WindowMembershipSpec | None = None,
) -> np.ndarray:
    """Return windows touched by one event using shared default semantics."""
    start = float(onset)
    dur = float(duration) if np.isfinite(float(duration)) else 0.0
    end = start + (dur if dur > 0 else 0.0)
    return window_membership_mask(
        t_start=t_start,
        t_end=t_end,
        t_mid=t_mid,
        interval=TimeInterval(start_sec=start, end_sec=end),
        spec=interval_spec or WindowMembershipSpec(),
    )

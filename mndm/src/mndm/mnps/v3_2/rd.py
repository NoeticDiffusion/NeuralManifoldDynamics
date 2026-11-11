"""Versioned spectral-profile representational-dispersion estimator.

The formula accepts one already selected native sensor window. It constructs
centered log10 Welch-PSD profiles and averages their Euclidean pair distances;
this is a sensor-profile dispersion measure, not a neural-code distance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
from scipy import signal


RD_ESTIMATOR_ID = "RD.pipeline_native_window_centered_log_psd_dispersion.v1"
RD_SCHEMA_ID = "mndm.mnps_v3_2.rd_result.v1"


@dataclass(frozen=True)
class RDResult:
    estimator_id: str
    schema_id: str
    value: float | None
    units: str
    computed: bool
    valid: bool
    qualified: bool
    status: str
    reason: str | None
    channel_count: int
    sample_count: int
    frequency_count: int
    pair_count: int
    channel_labels: tuple[str, ...] | None
    qc: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["channel_labels"] = list(self.channel_labels) if self.channel_labels is not None else None
        payload["qc"] = list(self.qc)
        return payload


def _unsupported(reason: str, *, channels: int = 0, samples: int = 0,
                 labels: tuple[str, ...] | None = None, computed: bool = False,
                 frequency_count: int = 0) -> RDResult:
    return RDResult(RD_ESTIMATOR_ID, RD_SCHEMA_ID, None, "centered_log10_power",
                    computed, False, False, "unsupported", reason, channels,
                    samples, frequency_count, 0, labels, ())


def compute_rd_centered_log_psd_dispersion(
    signals: np.ndarray,
    channel_labels: Sequence[str] | None,
    *,
    sample_rate_hz: float,
    welch_segment_seconds: float = 4.0,
    frequency_min_hz: float = 1.0,
    frequency_max_hz: float = 45.0,
) -> RDResult:
    """Compute mean pairwise Euclidean distance of centered log10 PSD rows.

    Welch uses a Hann window, 50% overlap, constant detrending and density
    scaling, matching the existing v3.1 native-window RD estimator. Every
    supplied channel is required; nonpositive PSD support is a window-local
    null rather than an implicit floor or channel omission.
    """
    try:
        raw = np.asarray(signals)
        if np.iscomplexobj(raw):
            return _unsupported("complex_signal_not_supported")
        values = np.asarray(raw, dtype=np.float64)
        fs = float(sample_rate_hz)
        segment_seconds = float(welch_segment_seconds)
        low_hz = float(frequency_min_hz)
        high_hz = float(frequency_max_hz)
    except (TypeError, ValueError, OverflowError):
        return _unsupported("invalid_numeric_input")
    if values.ndim != 2:
        return _unsupported("signals_must_be_channels_by_samples")
    channels, samples = values.shape
    if channels < 2:
        return _unsupported("at_least_two_channels_required", channels=channels, samples=samples)
    if not np.isfinite(values).all():
        return _unsupported("non_finite_sample", channels=channels, samples=samples)
    if not math.isfinite(fs) or fs <= 0.0:
        return _unsupported("invalid_sample_rate", channels=channels, samples=samples)
    if not math.isfinite(segment_seconds) or segment_seconds <= 0.0:
        return _unsupported("invalid_welch_segment_seconds", channels=channels, samples=samples)
    if not math.isfinite(low_hz) or not math.isfinite(high_hz) or low_hz < 0.0 or high_hz <= low_hz:
        return _unsupported("invalid_frequency_range", channels=channels, samples=samples)
    if channel_labels is None or isinstance(channel_labels, (str, bytes)):
        return _unsupported("channel_labels_required", channels=channels, samples=samples)
    labels = tuple(str(label).strip() for label in channel_labels)
    if len(labels) != channels or any(not label for label in labels):
        return _unsupported("channel_label_count_mismatch_or_empty", channels=channels, samples=samples, labels=labels)
    if len({label.casefold() for label in labels}) != len(labels):
        return _unsupported("duplicate_channel_labels", channels=channels, samples=samples, labels=labels)
    nperseg = int(round(segment_seconds * fs))
    if nperseg < 8 or samples < nperseg:
        return _unsupported("insufficient_samples_for_welch_segment", channels=channels, samples=samples, labels=labels)

    try:
        frequencies, psd = signal.welch(
            values,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return _unsupported(f"welch_failed:{type(exc).__name__}", channels=channels, samples=samples, labels=labels, computed=True)
    upper_hz = min(high_hz, fs / 2.0)
    keep = (frequencies >= low_hz) & (frequencies <= upper_hz)
    frequency_count = int(np.count_nonzero(keep))
    if frequency_count < 2:
        return _unsupported("insufficient_frequency_bins", channels=channels, samples=samples, labels=labels,
                            computed=True, frequency_count=frequency_count)
    selected = psd[:, keep]
    if not np.isfinite(selected).all():
        return _unsupported("non_finite_power_spectral_density", channels=channels, samples=samples, labels=labels,
                            computed=True, frequency_count=frequency_count)
    if np.any(selected <= 0.0):
        return _unsupported("nonpositive_log_psd_frequency_support", channels=channels, samples=samples, labels=labels,
                            computed=True, frequency_count=frequency_count)
    profiles = np.log10(selected)
    profiles = profiles - np.mean(profiles, axis=1, keepdims=True, dtype=np.float64)
    differences = profiles[:, None, :] - profiles[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=-1, dtype=np.float64))
    if not np.isfinite(distances).all():
        return _unsupported("non_finite_profile_distance", channels=channels, samples=samples, labels=labels,
                            computed=True, frequency_count=frequency_count)
    upper = np.triu(np.ones((channels, channels), dtype=bool), k=1)
    value = float(np.mean(distances[upper], dtype=np.float64))
    valid = math.isfinite(value)
    return RDResult(
        RD_ESTIMATOR_ID, RD_SCHEMA_ID, value if valid else None,
        "centered_log10_power", True, valid, False,
        "computed_exploratory" if valid else "unsupported",
        None if valid else "non_finite_pairwise_mean", channels, samples,
        frequency_count, int(np.count_nonzero(upper)), labels,
        ("sensor_profile_dispersion_only",),
    )


__all__ = ["RD_ESTIMATOR_ID", "RD_SCHEMA_ID", "RDResult", "compute_rd_centered_log_psd_dispersion"]

"""Versioned temporal-persistence estimators for MNPS v3.2.

The envelope estimator is intentionally distinct from v3.1's one-sample
lag-1-decorrelation candidate. Both functions here consume already prepared
window matrices; filtering and time-support auditing belong to the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


TP_ENVELOPE_ESTIMATOR_ID = "TP.alpha_log_rms_envelope_pearson_persistence.v1"
TP_RAW_COMPARATOR_ESTIMATOR_ID = "TP.raw_signal_pearson_persistence.v1"
TP_RESULT_SCHEMA_ID = "mndm.mnps_v3_2.tp_result.v1"


@dataclass(frozen=True)
class TPResult:
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
    pair_count: int
    lag_samples: int
    lag_seconds: float
    channel_labels: tuple[str, ...] | None
    channel_correlations: tuple[float | None, ...]
    channel_reasons: tuple[str | None, ...]
    qc: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["channel_labels"] = list(self.channel_labels) if self.channel_labels is not None else None
        payload["channel_correlations"] = list(self.channel_correlations)
        payload["channel_reasons"] = list(self.channel_reasons)
        payload["qc"] = list(self.qc)
        return payload


def _validate_matrix(
    signals: np.ndarray,
    channel_labels: Sequence[str] | None,
    sample_rate_hz: float,
    lag_samples: int,
    minimum_pair_count: int,
) -> tuple[np.ndarray | None, tuple[str, ...] | None, str | None]:
    try:
        raw = np.asarray(signals)
        if np.iscomplexobj(raw):
            return None, None, "complex_signal_not_supported"
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, None, "invalid_signal_matrix"
    if values.ndim != 2:
        return None, None, "signals_must_be_channels_by_samples"
    channels, samples = values.shape
    if channels < 1:
        return None, None, "no_channels"
    if not np.isfinite(values).all():
        return None, None, "non_finite_sample"
    if not math.isfinite(float(sample_rate_hz)) or sample_rate_hz <= 0.0:
        return None, None, "invalid_sample_rate"
    if isinstance(lag_samples, bool) or not isinstance(lag_samples, (int, np.integer)) or lag_samples < 1:
        return None, None, "lag_samples_must_be_positive_integer"
    if isinstance(minimum_pair_count, bool) or not isinstance(minimum_pair_count, (int, np.integer)) or minimum_pair_count < 2:
        return None, None, "minimum_pair_count_must_be_integer_at_least_2"
    if samples - lag_samples < minimum_pair_count:
        return None, None, "insufficient_lagged_pairs"
    if channel_labels is None or isinstance(channel_labels, (str, bytes)):
        return None, None, "channel_labels_required"
    labels = tuple(str(label).strip() for label in channel_labels)
    if len(labels) != channels or any(not label for label in labels):
        return None, labels, "channel_label_count_mismatch_or_empty"
    normalized = tuple(label.casefold() for label in labels)
    if len(set(normalized)) != len(normalized):
        return None, labels, "duplicate_channel_labels"
    return values, labels, None


def _pearson_paired(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    """Pearson correlation with separate means for the two paired slices."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 2:
        return None, "insufficient_lagged_pairs"
    a = a - float(np.mean(a, dtype=np.float64))
    b = b - float(np.mean(b, dtype=np.float64))
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    if not math.isfinite(aa) or not math.isfinite(bb):
        return None, "non_finite_lag_variance"
    if aa <= 0.0 or bb <= 0.0:
        return None, "zero_lagged_series_variance"
    value = float(np.dot(a, b) / math.sqrt(aa * bb))
    if not math.isfinite(value) or abs(value) > 1.0 + 1e-12:
        return None, "non_finite_or_out_of_range_correlation"
    return float(np.clip(value, -1.0, 1.0)), None


def _unsupported(
    estimator_id: str,
    reason: str,
    *,
    channels: int = 0,
    samples: int = 0,
    lag_samples: int = 0,
    lag_seconds: float = math.nan,
    labels: tuple[str, ...] | None = None,
    qc: tuple[str, ...] = (),
) -> TPResult:
    return TPResult(
        estimator_id, TP_RESULT_SCHEMA_ID, None, "1", False, False, False,
        "unsupported", reason, channels, samples, 0, lag_samples, lag_seconds,
        labels, (), (), qc,
    )


def compute_tp_log_rms_envelope_persistence(
    bandpassed_signals: np.ndarray,
    channel_labels: Sequence[str] | None,
    *,
    sample_rate_hz: float,
    envelope_samples: int,
    lag_samples: int,
    epsilon_amplitude: float = 1e-12,
    minimum_pair_count: int = 20,
) -> TPResult:
    """Compute median channel Pearson persistence of log trailing-RMS envelopes.

    The caller supplies the already band-limited window and its complete
    ordered sensor set. For each channel, the envelope is a trailing RMS over
    ``envelope_samples`` and is evaluated only where the complete envelope is
    observed. Pearson correlation compares independently demeaned, lagged
    envelope slices. If any required channel has undefined support, the window
    is null rather than silently aggregating a reduced channel set.
    """
    try:
        fs = float(sample_rate_hz)
        eps = float(epsilon_amplitude)
    except (TypeError, ValueError, OverflowError):
        return _unsupported(TP_ENVELOPE_ESTIMATOR_ID, "invalid_numeric_parameter")
    if not math.isfinite(fs) or fs <= 0.0:
        return _unsupported(TP_ENVELOPE_ESTIMATOR_ID, "invalid_sample_rate")
    if not math.isfinite(eps) or eps <= 0.0:
        return _unsupported(TP_ENVELOPE_ESTIMATOR_ID, "epsilon_amplitude_must_be_finite_positive")
    if isinstance(envelope_samples, bool) or not isinstance(envelope_samples, (int, np.integer)) or envelope_samples < 1:
        return _unsupported(TP_ENVELOPE_ESTIMATOR_ID, "envelope_samples_must_be_positive_integer")
    values, labels, error = _validate_matrix(
        bandpassed_signals, channel_labels, fs, lag_samples, minimum_pair_count
    )
    if error is not None:
        n_channels = int(np.asarray(bandpassed_signals).shape[0]) if np.asarray(bandpassed_signals).ndim >= 1 else 0
        n_samples = int(np.asarray(bandpassed_signals).shape[1]) if np.asarray(bandpassed_signals).ndim == 2 else 0
        return _unsupported(TP_ENVELOPE_ESTIMATOR_ID, error, channels=n_channels, samples=n_samples,
                            lag_samples=int(lag_samples) if isinstance(lag_samples, (int, np.integer)) else 0,
                            lag_seconds=float(lag_samples) / fs if isinstance(lag_samples, (int, np.integer)) and fs > 0 else math.nan,
                            labels=labels)
    assert values is not None and labels is not None
    channels, samples = values.shape
    if samples - int(envelope_samples) + 1 - int(lag_samples) < int(minimum_pair_count):
        return _unsupported(
            TP_ENVELOPE_ESTIMATOR_ID, "insufficient_complete_envelope_pairs",
            channels=channels, samples=samples, lag_samples=int(lag_samples),
            lag_seconds=int(lag_samples) / fs, labels=labels,
        )

    correlations: list[float | None] = []
    reasons: list[str | None] = []
    for row in values:
        scale = float(np.max(np.abs(row)))
        if scale == 0.0:
            correlations.append(None)
            reasons.append("zero_bandpassed_channel")
            continue
        scaled = row / scale
        squared = scaled * scaled
        cumulative = np.empty(squared.size + 1, dtype=np.float64)
        cumulative[0] = 0.0
        np.cumsum(squared, dtype=np.float64, out=cumulative[1:])
        means = (cumulative[int(envelope_samples):] - cumulative[:-int(envelope_samples)]) / float(envelope_samples)
        envelope = np.sqrt(np.maximum(means, 0.0)) * scale
        if not np.isfinite(envelope).all():
            correlations.append(None)
            reasons.append("non_finite_rms_envelope")
            continue
        log_envelope = np.log(envelope + eps)
        lag = int(lag_samples)
        correlation, reason = _pearson_paired(log_envelope[:-lag], log_envelope[lag:])
        correlations.append(correlation)
        reasons.append(reason)

    if any(value is None for value in correlations):
        first_reason = next(reason for value, reason in zip(correlations, reasons, strict=True) if value is None)
        return TPResult(
            TP_ENVELOPE_ESTIMATOR_ID, TP_RESULT_SCHEMA_ID, None, "1", True, False,
            False, "unsupported", first_reason or "one_or_more_channels_undefined",
            channels, samples, max(0, samples - int(envelope_samples) + 1 - int(lag_samples)),
            int(lag_samples), int(lag_samples) / fs, labels, tuple(correlations),
            tuple(reasons), ("complete_channel_set_required",),
        )
    value = float(np.median(np.asarray(correlations, dtype=np.float64)))
    valid = math.isfinite(value)
    return TPResult(
        TP_ENVELOPE_ESTIMATOR_ID, TP_RESULT_SCHEMA_ID, value if valid else None,
        "1", True, valid, False, "computed_exploratory" if valid else "unsupported",
        None if valid else "non_finite_median_correlation", channels, samples,
        max(0, samples - int(envelope_samples) + 1 - int(lag_samples)),
        int(lag_samples), int(lag_samples) / fs, labels, tuple(correlations),
        tuple(reasons), ("complete_channel_set_required",),
    )


def compute_tp_raw_signal_persistence(
    signals: np.ndarray,
    channel_labels: Sequence[str] | None,
    *,
    sample_rate_hz: float,
    lag_samples: int,
    minimum_pair_count: int = 20,
) -> TPResult:
    """Compute the separate direct raw-signal lagged Pearson comparator."""
    try:
        fs = float(sample_rate_hz)
    except (TypeError, ValueError, OverflowError):
        return _unsupported(TP_RAW_COMPARATOR_ESTIMATOR_ID, "invalid_sample_rate")
    values, labels, error = _validate_matrix(
        signals, channel_labels, fs, lag_samples, minimum_pair_count
    )
    if error is not None:
        return _unsupported(TP_RAW_COMPARATOR_ESTIMATOR_ID, error, labels=labels,
                            lag_samples=int(lag_samples) if isinstance(lag_samples, (int, np.integer)) else 0,
                            lag_seconds=int(lag_samples) / fs if isinstance(lag_samples, (int, np.integer)) and fs > 0 else math.nan)
    assert values is not None and labels is not None
    channels, samples = values.shape
    lag = int(lag_samples)
    correlations: list[float | None] = []
    reasons: list[str | None] = []
    for row in values:
        correlation, reason = _pearson_paired(row[:-lag], row[lag:])
        correlations.append(correlation)
        reasons.append(reason)
    if any(value is None for value in correlations):
        first_reason = next(reason for value, reason in zip(correlations, reasons, strict=True) if value is None)
        return TPResult(
            TP_RAW_COMPARATOR_ESTIMATOR_ID, TP_RESULT_SCHEMA_ID, None, "1", True,
            False, False, "unsupported", first_reason or "one_or_more_channels_undefined",
            channels, samples, samples - lag, lag, lag / fs, labels,
            tuple(correlations), tuple(reasons), ("complete_channel_set_required",),
        )
    value = float(np.median(np.asarray(correlations, dtype=np.float64)))
    valid = math.isfinite(value)
    return TPResult(
        TP_RAW_COMPARATOR_ESTIMATOR_ID, TP_RESULT_SCHEMA_ID, value if valid else None,
        "1", True, valid, False, "computed_exploratory" if valid else "unsupported",
        None if valid else "non_finite_median_correlation", channels, samples,
        samples - lag, lag, lag / fs, labels, tuple(correlations), tuple(reasons),
        ("complete_channel_set_required",),
    )


__all__ = [
    "TP_ENVELOPE_ESTIMATOR_ID", "TP_RAW_COMPARATOR_ESTIMATOR_ID",
    "TP_RESULT_SCHEMA_ID", "TPResult", "compute_tp_log_rms_envelope_persistence",
    "compute_tp_raw_signal_persistence",
]

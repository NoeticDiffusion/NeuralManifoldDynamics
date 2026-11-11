"""Experimental single-sensor M2 lag-one decorrelation measurement.

The estimator reports ``1 - rho(1)`` for one audited, contiguous sensor
window. It is a dimensionless, sample-step mobility index, not an inferred
long-timescale or biological quantity. Its physical lag is one sample, so
results are only directly comparable when sample spacing and signal
realization are compatible. The input is not filtered, detrended, or otherwise
modified here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Literal

import numpy as np

from .candidates import MNPS31WindowSupport


M2_LAG1_DECORRELATION_ESTIMATOR_ID = "M2.sensor_lag1_decorrelation.v0"
M2_SURFACE_LAG1_DECORRELATION_ESTIMATOR_ID = (
    "M2.surface_channel_median_lag1_decorrelation.v0"
)


@dataclass(frozen=True)
class M2InputBranch:
    """Explicit signal realization metadata required for surface output.

    ``filter_band_hz=None`` declares no fixed band filter. Any additional
    preprocessing must be captured in ``branch_id``; the estimator itself
    never modifies the signal.
    """

    branch_id: str
    sample_rate_hz: float
    filter_band_hz: tuple[float, float] | None
    reference: str
    reference_set_sha256: str | None


@dataclass(frozen=True)
class MNPS31M2ChannelValue:
    """Per-channel value or reason it could not be computed."""

    channel_label: str
    value: float | None
    lag1_autocorrelation: float | None
    status: Literal["computed_exploratory", "unsupported"]
    reason: str | None


@dataclass(frozen=True)
class MNPS31M2SurfaceMeasurement:
    """Median M2 surface value and support/provenance for one input realization."""

    candidate_id: str
    estimator_id: str
    value: float | None
    units: str
    computed: bool
    valid: bool
    qualified: bool
    status: Literal["computed_exploratory", "unsupported"]
    reason: str | None
    sample_count: int
    sample_rate_hz: float
    lag_seconds: float | None
    channel_labels: tuple[str, ...]
    channel_count: int
    valid_channel_count: int
    valid_channel_coverage: float
    channel_order_sha256: str
    channel_set_sha256: str
    input_branch: M2InputBranch
    support: MNPS31WindowSupport
    per_channel: tuple[MNPS31M2ChannelValue, ...]
    qc: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return an auditable payload with the actual sensor realization."""
        payload = asdict(self)
        payload["support"] = asdict(self.support)
        payload["input_branch"] = asdict(self.input_branch)
        payload["claim_status"] = "no_biological_claim"
        payload["coordinate_assignment"] = None
        return payload


@dataclass(frozen=True)
class MNPS31M2Measurement:
    """One attempted M2 value, with computation and qualification kept apart."""

    candidate_id: str
    estimator_id: str
    value: float | None
    units: str
    lag1_autocorrelation: float | None
    lag_seconds: float | None
    sample_count: int
    computed: bool
    valid: bool
    qualified: bool
    status: Literal["computed_exploratory", "unsupported"]
    reason: str | None
    support: MNPS31WindowSupport
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return an auditable JSON-compatible payload with no qualification claim."""
        payload = asdict(self)
        payload["support"] = asdict(self.support)
        payload["claim_status"] = "no_biological_claim"
        payload["coordinate_assignment"] = None
        return payload


def _unsupported(
    *, support: MNPS31WindowSupport, reason: str, sample_count: int = 0
) -> MNPS31M2Measurement:
    return MNPS31M2Measurement(
        candidate_id="M2",
        estimator_id=M2_LAG1_DECORRELATION_ESTIMATOR_ID,
        value=None,
        units="1",
        lag1_autocorrelation=None,
        lag_seconds=None,
        sample_count=sample_count,
        computed=False,
        valid=False,
        qualified=False,
        status="unsupported",
        reason=reason,
        support=support,
        details={
            "definition": "mobility = 1 - rho(1)",
            "acf": "mean_centered_lag1_product_sum_divided_by_full_record_centered_energy",
        },
    )


def _support_error(
    values: np.ndarray, support: MNPS31WindowSupport
) -> str | None:
    if values.ndim != 1:
        return "signal_must_be_one_dimensional"
    if values.size < 3:
        return "at_least_three_samples_required"
    if not np.isfinite(values).all():
        return "non_finite_sample"
    if (
        isinstance(support.sample_count, (bool, np.bool_))
        or not isinstance(support.sample_count, (int, np.integer))
        or support.sample_count != values.size
    ):
        return "sample_count_mismatch"

    audit_fields = (
        support.source_is_continuous,
        support.timestamps_uniform,
        support.gap_free,
        support.bad_interval_free,
        support.within_recording_bounds,
        support.effective_filter_support_valid,
    )
    if any(field is None for field in audit_fields):
        return "support_audit_incomplete"
    if any(not isinstance(field, (bool, np.bool_)) for field in audit_fields):
        return "support_audit_flag_not_boolean"
    failed_support_reasons = (
        (support.source_is_continuous, "non_continuous_source"),
        (support.timestamps_uniform, "irregular_sample_times"),
        (support.gap_free, "gap_or_segment_boundary"),
        (support.bad_interval_free, "bad_interval_overlap"),
        (support.within_recording_bounds, "recording_boundary_overrun"),
        (support.effective_filter_support_valid, "insufficient_filter_edge_support"),
    )
    for passed, reason in failed_support_reasons:
        if not bool(passed):
            return reason

    if not isinstance(support.segment_id, str) or not support.segment_id.strip():
        return "missing_segment_id"
    try:
        sample_rate_hz = float(support.sample_rate_hz)
        window_start_sec = float(support.window_start_sec)
        window_end_sec = float(support.window_end_sec)
    except (TypeError, ValueError):
        return "invalid_timebase"
    if (
        not math.isfinite(sample_rate_hz)
        or sample_rate_hz <= 0.0
        or not math.isfinite(window_start_sec)
        or not math.isfinite(window_end_sec)
    ):
        return "invalid_timebase"
    duration_seconds = window_end_sec - window_start_sec
    if duration_seconds <= 0.0 or not math.isclose(
        duration_seconds,
        values.size / sample_rate_hz,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        return "window_time_support_mismatch"
    return None


def compute_m2_sensor_lag1_decorrelation(
    signal: np.ndarray, support: MNPS31WindowSupport
) -> MNPS31M2Measurement:
    """Compute the experimental lag-one mobility index for one sensor window.

    For ``x`` with its full-window mean removed, this uses
    ``rho(1) = sum(x[:-1] * x[1:]) / sum(x * x)`` and returns ``1 - rho(1)``.
    The finite result is marked valid for this estimator's support contract;
    ``qualified`` always remains false. No montage or minimum channel count is
    imposed because this function computes one channel at a time.
    """
    try:
        values = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _unsupported(support=support, reason="invalid_numeric_input")

    sample_count = int(values.size) if values.ndim == 1 else 0
    reason = _support_error(values, support)
    if reason:
        return _unsupported(
            support=support, reason=reason, sample_count=sample_count
        )

    # Scaling first protects centering and dot products from overflow while
    # preserving the correlation coefficient, including for tiny amplitudes.
    amplitude_scale = float(np.max(np.abs(values)))
    if amplitude_scale == 0.0:
        return _unsupported(
            support=support, reason="zero_variance_signal", sample_count=sample_count
        )
    normalized = values / amplitude_scale
    centered = normalized - float(np.mean(normalized))
    energy = float(np.dot(centered, centered))
    if not math.isfinite(energy) or energy <= 0.0:
        return _unsupported(
            support=support, reason="zero_variance_signal", sample_count=sample_count
        )

    lag1_autocorrelation = float(
        np.dot(centered[:-1], centered[1:]) / energy
    )
    if not math.isfinite(lag1_autocorrelation):
        return _unsupported(
            support=support,
            reason="undefined_lag1_autocorrelation",
            sample_count=sample_count,
        )
    # The full-energy ACF is bounded by [-1, 1]; allow only floating-point
    # roundoff at the boundary, then enforce that mathematical range.
    if abs(lag1_autocorrelation) > 1.0 + 1e-12:
        return _unsupported(
            support=support,
            reason="lag1_autocorrelation_out_of_bounds",
            sample_count=sample_count,
        )
    lag1_autocorrelation = min(1.0, max(-1.0, lag1_autocorrelation))
    value = 1.0 - lag1_autocorrelation
    sample_rate_hz = float(support.sample_rate_hz)

    return MNPS31M2Measurement(
        candidate_id="M2",
        estimator_id=M2_LAG1_DECORRELATION_ESTIMATOR_ID,
        value=float(value),
        units="1",
        lag1_autocorrelation=lag1_autocorrelation,
        lag_seconds=1.0 / sample_rate_hz,
        sample_count=sample_count,
        computed=True,
        valid=True,
        qualified=False,
        status="computed_exploratory",
        reason=None,
        support=support,
        details={
            "definition": "mobility = 1 - rho(1)",
            "acf": "mean_centered_lag1_product_sum_divided_by_full_record_centered_energy",
            "sample_lag": 1,
            "sample_rate_hz": sample_rate_hz,
            "lag_seconds": 1.0 / sample_rate_hz,
            "theoretical_value_range": [0.0, 2.0],
            "input_transform": "none",
            "known_limits": [
                "sampling_step_specific; comparisons require compatible sample spacing and signal realizations",
                "summarizes adjacent-sample dependence only, not a long-timescale or multiscale decay",
                "periodicity, drift, filtering, reference, and modality can affect lag-one correlation",
            ],
        },
    )


def _labels_hash(labels: tuple[str, ...], *, ordered: bool) -> str:
    normalized = tuple(label.strip().casefold() for label in labels)
    if not ordered:
        normalized = tuple(sorted(normalized))
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _input_branch_error(branch: M2InputBranch, support: MNPS31WindowSupport) -> str | None:
    if not isinstance(branch, M2InputBranch):
        return "invalid_input_branch"
    if not isinstance(branch.branch_id, str) or not branch.branch_id.strip():
        return "missing_input_branch_id"
    if not isinstance(branch.reference, str) or not branch.reference.strip():
        return "missing_reference_description"
    try:
        branch_rate = float(branch.sample_rate_hz)
        support_rate = float(support.sample_rate_hz)
    except (TypeError, ValueError):
        return "invalid_input_branch_sample_rate"
    if (
        not math.isfinite(branch_rate)
        or not math.isclose(branch_rate, support_rate, rel_tol=0.0, abs_tol=1e-9)
    ):
        return "input_branch_sample_rate_mismatch"
    if branch.filter_band_hz is not None:
        try:
            low_hz, high_hz = (float(value) for value in branch.filter_band_hz)
        except (TypeError, ValueError):
            return "invalid_filter_band"
        if (
            not math.isfinite(low_hz)
            or not math.isfinite(high_hz)
            or low_hz <= 0.0
            or high_hz <= low_hz
            or high_hz > branch_rate / 2.0
        ):
            return "invalid_filter_band"
    if branch.reference_set_sha256 is not None:
        digest = str(branch.reference_set_sha256)
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            return "invalid_reference_set_sha256"
    return None


def compute_m2_surface_lag1_decorrelation(
    signals: np.ndarray,
    channel_names: tuple[str, ...] | list[str],
    support: MNPS31WindowSupport,
    input_branch: M2InputBranch,
) -> MNPS31M2SurfaceMeasurement:
    """Compute a per-channel lag-one index and its supported-channel median.

    Rows are used in the exact input order and no montage, channel-count floor,
    or channel repair is applied. Invalid channels remain in ``per_channel``
    with their reasons; the median uses every computable channel. The output
    preserves ordered labels and label-set hashes, reference metadata, the
    caller-declared sample/filter branch, and valid-channel coverage.
    """
    try:
        values = np.asarray(signals, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        values = np.empty((0, 0), dtype=np.float64)
        input_error = "invalid_numeric_input"
    else:
        input_error = None

    try:
        raw_labels = tuple(channel_names)
    except TypeError:
        raw_labels = ()
        input_error = input_error or "channel_labels_must_be_a_sequence"
    labels = tuple(
        label.strip() if isinstance(label, str) else "" for label in raw_labels
    )
    n_channels = len(labels)
    n_samples = int(values.shape[1]) if values.ndim == 2 else 0
    order_hash = _labels_hash(labels, ordered=True) if labels else ""
    set_hash = _labels_hash(labels, ordered=False) if labels else ""

    reason = input_error
    if reason is None and values.ndim != 2:
        reason = "signals_must_be_channels_by_samples"
    if reason is None and values.shape[0] == 0:
        reason = "at_least_one_channel_required"
    label_keys = tuple(label.casefold() for label in labels)
    if reason is None and (
        len(labels) != values.shape[0]
        or any(not isinstance(label, str) for label in raw_labels)
        or any(not label for label in labels)
        or len(set(label_keys)) != len(label_keys)
    ):
        reason = "channel_labels_must_be_nonempty_unique_and_match_rows"
    if reason is None:
        reason = _input_branch_error(input_branch, support)

    if reason is not None:
        try:
            sample_rate_hz = float(support.sample_rate_hz)
        except (TypeError, ValueError):
            sample_rate_hz = 0.0
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            sample_rate_hz = 0.0
        return MNPS31M2SurfaceMeasurement(
            candidate_id="M2",
            estimator_id=M2_SURFACE_LAG1_DECORRELATION_ESTIMATOR_ID,
            value=None,
            units="1",
            computed=False,
            valid=False,
            qualified=False,
            status="unsupported",
            reason=reason,
            sample_count=n_samples,
            sample_rate_hz=sample_rate_hz,
            lag_seconds=(1.0 / sample_rate_hz if sample_rate_hz > 0.0 else None),
            channel_labels=labels,
            channel_count=n_channels,
            valid_channel_count=0,
            valid_channel_coverage=0.0,
            channel_order_sha256=order_hash,
            channel_set_sha256=set_hash,
            input_branch=input_branch,
            support=support,
            per_channel=(),
            qc={"unsupported_reason": reason},
        )

    measurements = tuple(
        compute_m2_sensor_lag1_decorrelation(row, support) for row in values
    )
    channel_results = tuple(
        MNPS31M2ChannelValue(
            channel_label=label,
            value=measurement.value,
            lag1_autocorrelation=measurement.lag1_autocorrelation,
            status=measurement.status,
            reason=measurement.reason,
        )
        for label, measurement in zip(labels, measurements, strict=True)
    )
    valid_values = [
        float(measurement.value)
        for measurement in measurements
        if measurement.computed and measurement.value is not None
    ]
    valid_count = len(valid_values)
    coverage = valid_count / n_channels
    reasons: dict[str, int] = {}
    for measurement in measurements:
        if measurement.reason is not None:
            reasons[measurement.reason] = reasons.get(measurement.reason, 0) + 1
    computed = valid_count > 0
    value = float(np.median(valid_values)) if computed else None
    sample_rate_hz = float(support.sample_rate_hz)
    return MNPS31M2SurfaceMeasurement(
        candidate_id="M2",
        estimator_id=M2_SURFACE_LAG1_DECORRELATION_ESTIMATOR_ID,
        value=value,
        units="1",
        computed=computed,
        valid=computed,
        qualified=False,
        status="computed_exploratory" if computed else "unsupported",
        reason=None if computed else "no_computable_channels",
        sample_count=n_samples,
        sample_rate_hz=sample_rate_hz,
        lag_seconds=1.0 / sample_rate_hz,
        channel_labels=labels,
        channel_count=n_channels,
        valid_channel_count=valid_count,
        valid_channel_coverage=coverage,
        channel_order_sha256=order_hash,
        channel_set_sha256=set_hash,
        input_branch=input_branch,
        support=support,
        per_channel=channel_results,
        qc={
            "channel_failure_counts_by_reason": reasons,
            "aggregation": "median_of_computable_channel_values",
            "minimum_coverage_threshold": None,
        },
    )

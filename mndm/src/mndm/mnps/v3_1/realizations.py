"""Experimental multi-surface, multi-window EEG measurement execution.

This module keeps an estimator identity separate from the sensor realization
used to observe it. Inputs must already be on the declared sampled and
filtered grid. Each requested surface is selected independently, optionally
average-referenced over that exact surface, and evaluated for every supplied
window. It reads no source files and does not qualify measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from typing import Literal, Sequence

import numpy as np

from .candidates import MNPS31WindowSupport
from .eeg_metrics import EEGCandidateMeasurement, compute_e3_sensor_participation_ratio


CORE19_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)


@dataclass(frozen=True)
class MNPS31SensorSurface:
    """A named ordered sensor realization for one recording.

    ``channel_names`` is the exact intended surface. Core19 and study-common
    surfaces require all declared channels; study-common has no Core19 count
    floor. Native surfaces may opt into available-channel
    computation; the minimum count is then only the estimator's computability
    floor, and channel coverage is returned for downstream decisions.
    """

    surface_id: str
    kind: Literal["native", "core19", "study_common", "custom"]
    channel_names: tuple[str, ...]
    reference_mode: Literal["recorded", "average_over_surface"]
    reference_id: str | None = None
    reference_channel_names: tuple[str, ...] | None = None
    allow_partial_channels: bool = False
    minimum_channels: int = 2

    def __post_init__(self) -> None:
        labels = tuple(str(label).strip() for label in self.channel_names)
        keys = tuple(label.casefold() for label in labels)
        if not self.surface_id.strip():
            raise ValueError("surface_id_must_be_nonempty")
        if self.kind not in {"native", "core19", "study_common", "custom"}:
            raise ValueError("surface_kind_unsupported")
        if self.reference_mode not in {"recorded", "average_over_surface"}:
            raise ValueError("reference_mode_unsupported")
        if self.reference_mode == "average_over_surface" and (
            self.reference_id is not None or self.reference_channel_names is not None
        ):
            raise ValueError("average_reference_cannot_declare_recorded_reference")
        if not labels or any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("surface_labels_must_be_nonempty_and_unique")
        if self.kind in {"core19", "study_common"} and self.allow_partial_channels:
            raise ValueError("fixed_surface_cannot_allow_partial_channels")
        if self.kind == "core19" and keys != tuple(label.casefold() for label in CORE19_CHANNELS):
            raise ValueError("core19_surface_must_use_the_declared_core19_labels_in_order")
        if (
            isinstance(self.minimum_channels, bool)
            or not isinstance(self.minimum_channels, int)
            or self.minimum_channels < 2
            or self.minimum_channels > len(labels)
        ):
            raise ValueError("minimum_channels_out_of_range")
        if self.reference_mode == "recorded" and self.reference_channel_names is not None:
            reference_keys = [str(label).strip().casefold() for label in self.reference_channel_names]
            if not reference_keys or any(not label for label in reference_keys) or len(set(reference_keys)) != len(reference_keys):
                raise ValueError("reference_labels_must_be_nonempty_and_unique")
        object.__setattr__(self, "channel_names", labels)
        if self.reference_channel_names is not None:
            object.__setattr__(
                self,
                "reference_channel_names",
                tuple(str(label).strip() for label in self.reference_channel_names),
            )


@dataclass(frozen=True)
class MNPS31WindowSlice:
    """One requested half-open sample interval and its independently audited support."""

    window_id: str
    start_sample: int
    end_sample_exclusive: int
    support: MNPS31WindowSupport
    sample_time_origin_sec: float = 0.0
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.window_id.strip():
            raise ValueError("window_id_must_be_nonempty")
        if (
            isinstance(self.start_sample, bool)
            or isinstance(self.end_sample_exclusive, bool)
            or not isinstance(self.start_sample, (int, np.integer))
            or not isinstance(self.end_sample_exclusive, (int, np.integer))
            or self.start_sample < 0
            or self.end_sample_exclusive <= self.start_sample
        ):
            raise ValueError("window_sample_bounds_invalid")
        if not np.isfinite(self.support.sample_rate_hz) or self.support.sample_rate_hz <= 0:
            raise ValueError("window_sample_rate_invalid")
        if self.end_sample_exclusive - self.start_sample != self.support.sample_count:
            raise ValueError("window_sample_bounds_do_not_match_support")
        expected_start = self.sample_time_origin_sec + self.start_sample / self.support.sample_rate_hz
        expected_end = self.sample_time_origin_sec + self.end_sample_exclusive / self.support.sample_rate_hz
        if (
            not np.isfinite(self.sample_time_origin_sec)
            or not np.isclose(expected_start, self.support.window_start_sec, rtol=0.0, atol=1e-6)
            or not np.isclose(expected_end, self.support.window_end_sec, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("window_sample_bounds_do_not_match_support_timebase")


def _ordered_hash(labels: Sequence[str]) -> str:
    payload = "\n".join(str(label).strip().casefold() for label in labels)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_hash(labels: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(label).strip().casefold() for label in labels))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unavailable_e3(
    support: MNPS31WindowSupport,
    *,
    reason: str,
    details: dict[str, object],
) -> EEGCandidateMeasurement:
    return EEGCandidateMeasurement(
        candidate_id="E3",
        estimator_id="E3.sensor_covariance_participation_ratio.v0",
        value=None,
        units="sensor_directions",
        observation_space="sensor_only",
        computed=False,
        valid=False,
        qualified=False,
        status="unsupported",
        reason=reason,
        support=support,
        details=details,
    )


def compute_e3_surface_window_grid(
    signals: np.ndarray,
    source_channel_names: Sequence[str],
    surfaces: Sequence[MNPS31SensorSurface],
    windows: Sequence[MNPS31WindowSlice],
    *,
    preprocessing_id: str,
    bad_channel_names: Sequence[str] = (),
) -> tuple[dict[str, object], ...]:
    """Compute raw E3 for every requested ``surface × window`` pair.

    The input matrix is channels by samples and must already be sampled and
    filtered. Core19/study-common require every declared channel for a window.
    Native/custom surfaces may use the finite, non-bad subset when explicitly
    configured to allow it; their effective channel count and coverage are
    recorded. Each row is returned, including missing-surface, bad-channel,
    and unsupported-window cases. Average reference is applied independently
    within each surface/window and its exact labels are recorded.
    """
    try:
        values = np.asarray(signals, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("signals_must_be_numeric") from exc
    labels = tuple(str(label).strip() for label in source_channel_names)
    keys = tuple(label.casefold() for label in labels)
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("signals_must_match_source_channel_labels")
    if not labels or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("source_labels_must_be_nonempty_and_unique")
    source_key_set = set(keys)
    if not preprocessing_id.strip():
        raise ValueError("preprocessing_id_must_be_nonempty")
    surface_ids = [surface.surface_id for surface in surfaces]
    window_ids = [window.window_id for window in windows]
    if len(set(surface_ids)) != len(surface_ids) or len(set(window_ids)) != len(window_ids):
        raise ValueError("surface_and_window_ids_must_be_unique")
    lookup = {key: index for index, key in enumerate(keys)}
    bad_names = tuple(str(label).strip() for label in bad_channel_names)
    bad_keys = {label.casefold() for label in bad_names}
    if any(not label for label in bad_keys):
        raise ValueError("bad_channel_labels_must_be_nonempty")
    unmatched_bad_labels = tuple(label for label in bad_names if label.casefold() not in source_key_set)

    rows: list[dict[str, object]] = []
    for surface in surfaces:
        requested = surface.channel_names
        requested_keys = tuple(label.casefold() for label in requested)
        requested_key_set = set(requested_keys)
        missing = tuple(label for label, key in zip(requested, requested_keys, strict=True) if key not in lookup)
        extras = tuple(label for label, key in zip(labels, keys, strict=True) if key not in requested_key_set)
        for window in windows:
            base_details: dict[str, object] = {
                "surface_id": surface.surface_id,
                "surface_kind": surface.kind,
                "declared_channel_count": len(requested),
                "declared_channel_labels": list(requested),
                "declared_channel_order_sha256": _ordered_hash(requested),
                "declared_channel_set_sha256": _set_hash(requested),
                "source_channel_order_sha256": _ordered_hash(labels),
                "window_id": window.window_id,
                "window_start_sec": window.support.window_start_sec,
                "window_end_sec": window.support.window_end_sec,
                "start_sample": int(window.start_sample),
                "end_sample_exclusive": int(window.end_sample_exclusive),
                "sample_time_origin_sec": window.sample_time_origin_sec,
                "window_provenance": dict(window.provenance),
                "preprocessing_id": preprocessing_id,
                "cross_montage_equivalent": "unknown",
                "extra_source_channel_labels": list(extras),
                "unmatched_bad_channel_labels": list(unmatched_bad_labels),
                "qc_flags": [],
            }
            qc_flags = base_details["qc_flags"]
            assert isinstance(qc_flags, list)
            if extras:
                qc_flags.append("source_channels_outside_requested_surface")
            if unmatched_bad_labels:
                qc_flags.append("bad_channel_metadata_label_not_found")
            if surface.reference_mode == "average_over_surface":
                base_details["reference_mode"] = "average_over_effective_surface_channels"
                base_details["reference_id"] = "common_average"
                base_details["reference_channel_labels"] = list(requested)
                base_details["reference_channel_count"] = len(requested)
                base_details["reference_channel_order_sha256"] = _ordered_hash(requested)
                base_details["reference_channel_set_sha256"] = _set_hash(requested)
            else:
                base_details["reference_mode"] = "recorded_input_reference"
                base_details["reference_id"] = surface.reference_id
                if surface.reference_channel_names is not None:
                    base_details["reference_channel_labels"] = list(surface.reference_channel_names)
                    base_details["reference_channel_count"] = len(surface.reference_channel_names)
                    base_details["reference_channel_order_sha256"] = _ordered_hash(surface.reference_channel_names)
                    base_details["reference_channel_set_sha256"] = _set_hash(surface.reference_channel_names)
                else:
                    base_details["reference_channel_labels"] = None
                    base_details["reference_channel_count"] = None
                    base_details["reference_channel_order_sha256"] = None
                    base_details["reference_channel_set_sha256"] = None
                    qc_flags.append("reference_set_provenance_missing")
                if not surface.reference_id:
                    qc_flags.append("reference_identity_missing")

            if missing and not surface.allow_partial_channels:
                result = _unavailable_e3(
                    window.support,
                    reason="required_surface_channel_missing",
                    details={
                        **base_details,
                        "missing_channel_labels": list(missing),
                        "available_channel_count": len(requested) - len(missing),
                        "channel_availability_fraction": (len(requested) - len(missing)) / len(requested),
                    },
                )
                rows.append({"measurement": result.to_dict(), "surface_id": surface.surface_id, "window_id": window.window_id})
                continue
            if window.end_sample_exclusive > values.shape[1]:
                result = _unavailable_e3(
                    window.support,
                    reason="window_outside_signal_bounds",
                    details={**base_details, "signal_sample_count": int(values.shape[1])},
                )
                rows.append({"measurement": result.to_dict(), "surface_id": surface.surface_id, "window_id": window.window_id})
                continue

            present_positions = [
                (label, key, lookup[key])
                for label, key in zip(requested, requested_keys, strict=True)
                if key in lookup
            ]
            window_values = values[
                [position for _, _, position in present_positions],
                window.start_sample : window.end_sample_exclusive,
            ]
            finite_rows = np.isfinite(window_values).all(axis=1)
            present_labels = tuple(label for label, _, _ in present_positions)
            present_keys = tuple(key for _, key, _ in present_positions)
            explicit_bad_rows = np.asarray([key in bad_keys for key in present_keys], dtype=bool)
            usable_rows = finite_rows & ~explicit_bad_rows
            rejected_labels = [
                label
                for label, finite, is_bad in zip(present_labels, finite_rows, explicit_bad_rows, strict=True)
                if not finite or is_bad
            ]
            rejected_labels = [*missing, *rejected_labels]
            usable_labels = tuple(label for label, usable in zip(present_labels, usable_rows, strict=True) if usable)
            coverage = len(usable_labels) / len(requested)
            base_details.update(
                {
                    "effective_channel_count": len(usable_labels),
                    "effective_channel_labels": list(usable_labels),
                    "effective_channel_order_sha256": _ordered_hash(usable_labels),
                    "effective_channel_set_sha256": _set_hash(usable_labels),
                    "channel_coverage_fraction": coverage,
                    "rejected_channel_labels": rejected_labels,
                }
            )
            if rejected_labels:
                qc_flags.append("one_or_more_surface_channels_unavailable")
            strict_failure = bool(rejected_labels) and not surface.allow_partial_channels
            too_few = len(usable_labels) < surface.minimum_channels
            if strict_failure or too_few:
                result = _unavailable_e3(
                    window.support,
                    reason=("required_surface_channel_unavailable" if strict_failure else "insufficient_usable_channels"),
                    details=base_details,
                )
                rows.append({"measurement": result.to_dict(), "surface_id": surface.surface_id, "window_id": window.window_id})
                continue

            selected = window_values[usable_rows, :]
            if surface.reference_mode == "average_over_surface":
                selected = selected - np.mean(selected, axis=0, keepdims=True)
                reference_labels = usable_labels
                base_details["reference_channel_labels"] = list(reference_labels)
                base_details["reference_channel_count"] = len(reference_labels)
                base_details["reference_channel_order_sha256"] = _ordered_hash(reference_labels)
                base_details["reference_channel_set_sha256"] = _set_hash(reference_labels)
            measurement = compute_e3_sensor_participation_ratio(selected, usable_labels, window.support)
            details = {**measurement.details, **base_details}
            result = replace(measurement, details=details)
            rows.append({"measurement": result.to_dict(), "surface_id": surface.surface_id, "window_id": window.window_id})
    return tuple(rows)


__all__ = [
    "CORE19_CHANNELS",
    "MNPS31SensorSurface",
    "MNPS31WindowSlice",
    "compute_e3_surface_window_grid",
]

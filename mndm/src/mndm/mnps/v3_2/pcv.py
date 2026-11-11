"""Experimental, low-level Phase Coordination Variability (PCV) kernel.

Given an already estimated channel-by-sample phase matrix, PCV is the
population standard deviation of the sensor phase order parameter. This
module does not choose the phase estimator, band, sensor set, reference,
preprocessing, or source/time support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from .contracts import MeasurementRecord, RealizationRecord, ReproducibilityRecord, TimeSupportRecord


PCV_ESTIMATOR_ID = "PCV.sensor_phase_order_parameter_population_sd.v1"
PCV_SCHEMA_ID = "nmdm.mnps_v3_2.pcv_matrix_result.v1"


def _label_metadata(labels: Sequence[str] | None, channel_count: int):
    if labels is None:
        return None, None, None, ("channel_labels_unavailable",)
    if isinstance(labels, (str, bytes, bytearray)):
        return None, None, None, ("channel_labels_must_be_sequence",)
    try:
        result = tuple(labels)
    except (TypeError, ValueError):
        return None, None, None, ("channel_labels_unreadable",)
    if any(not isinstance(label, str) or not label.strip() for label in result):
        return None, None, None, ("channel_labels_must_be_nonempty_strings",)
    if len(result) != channel_count:
        return result, None, None, ("channel_label_count_mismatch",)
    normalized = tuple(label.strip().casefold() for label in result)
    if len(set(normalized)) != len(normalized):
        return result, None, None, ("duplicate_channel_label",)
    order = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    unordered = json.dumps(sorted(result), ensure_ascii=False, separators=(",", ":"))
    return (
        result,
        hashlib.sha256(order.encode("utf-8")).hexdigest(),
        hashlib.sha256(unordered.encode("utf-8")).hexdigest(),
        (),
    )


@dataclass(frozen=True)
class PCVResult:
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
    sample_count_per_channel: int
    phase_estimator_id: str | None
    band_id: str | None
    channel_labels: tuple[str, ...] | None
    channel_order_sha256: str | None
    channel_set_sha256: str | None
    qc: tuple[str, ...]

    def to_measurement_record(self) -> MeasurementRecord:
        return MeasurementRecord(
            family_id="PCV",
            estimator_id=self.estimator_id,
            schema_id=self.schema_id,
            value=self.value,
            units=self.units,
            computed=self.computed,
            valid=self.valid,
            qualified=False,
            status="computed_exploratory" if self.valid else "unsupported",
            reason=self.reason,
            support=TimeSupportRecord(
                support_status="unknown",
                sample_count=self.sample_count_per_channel or None,
            ),
            realization=RealizationRecord(
                audit_status="complete" if self.channel_order_sha256 else "incomplete",
                kind="phase_matrix_rows_unresolved_sensor_realization",
                requested_id=None,
                actual_id=None,
                ordered_labels=self.channel_labels,
                order_sha256=self.channel_order_sha256,
                set_sha256=self.channel_set_sha256,
                count=self.channel_count,
            ),
            qc=self.qc,
            reproducibility=ReproducibilityRecord(status="not_supplied"),
            method_parameters=(
                ("formula", "population_sd_over_time(abs(mean_c(exp(i*phase[c,t]))))"),
                ("phase_estimator_id", self.phase_estimator_id),
                ("band_id", self.band_id),
                ("sensor_aggregation", "mean_complex_phase_vectors_per_sample"),
                ("time_aggregation", "population_standard_deviation_ddof0"),
            ),
            direction="higher_more_phase_order_parameter_variability",
            claim_status="no_biological_claim",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_labels"] = list(self.channel_labels) if self.channel_labels is not None else None
        payload["qc"] = list(self.qc)
        payload["measurement_record"] = self.to_measurement_record().to_dict()
        return payload


def compute_pcv_phase_order_parameter_sd(
    phase_radians: np.ndarray,
    *,
    phase_estimator_id: str | None = None,
    band_id: str | None = None,
    channel_labels: Sequence[str] | None = None,
) -> PCVResult:
    """Compute ``SD_t(abs(mean_c(exp(i*phase[c,t]))))`` with ``ddof=0``.

    ``phase_radians`` is a finite real matrix with shape ``(channels,
    samples)``. One or more rows and one or more samples are mathematically
    computable. Missing phase/band IDs or row labels remain provenance QC and
    do not suppress a finite formula value. This is an estimator kernel, not a
    filter or phase-extraction pipeline.
    """
    try:
        raw = np.asarray(phase_radians)
        if np.iscomplexobj(raw):
            return PCVResult(
                PCV_ESTIMATOR_ID, PCV_SCHEMA_ID, None, "1", False, False, False,
                "unsupported", "phase_matrix_must_be_real_2d",
                int(raw.shape[0]) if raw.ndim else 0,
                int(raw.shape[1]) if raw.ndim == 2 else 0,
                None, None, None, None, None, (),
            )
        phase = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return PCVResult(
            PCV_ESTIMATOR_ID, PCV_SCHEMA_ID, None, "1", False, False, False,
            "unsupported", "invalid_phase_matrix", 0, 0,
            None, None, None, None, None, (),
        )
    if phase.ndim != 2:
        return PCVResult(
            PCV_ESTIMATOR_ID, PCV_SCHEMA_ID, None, "1", False, False, False,
            "unsupported", "phase_matrix_must_be_real_2d", int(phase.shape[0]) if phase.ndim else 0,
            int(phase.shape[1]) if phase.ndim == 2 else 0, None, None, None, None, None, (),
        )
    channels, samples = phase.shape
    if channels < 1 or samples < 1:
        return PCVResult(
            PCV_ESTIMATOR_ID, PCV_SCHEMA_ID, None, "1", False, False, False,
            "unsupported", "phase_matrix_must_contain_channel_and_sample", channels, samples,
            None, None, None, None, None, (),
        )
    labels, order_hash, set_hash, label_qc = _label_metadata(channel_labels, channels)
    if not np.isfinite(phase).all():
        return PCVResult(
            PCV_ESTIMATOR_ID, PCV_SCHEMA_ID, None, "1", False, False, False,
            "unsupported", "non_finite_phase", channels, samples,
            phase_estimator_id, band_id, labels, order_hash, set_hash, label_qc,
        )
    order_parameter = np.abs(np.mean(np.exp(1j * phase), axis=0))
    value = float(np.std(order_parameter, ddof=0))
    qc = list(label_qc)
    if not isinstance(phase_estimator_id, str) or not phase_estimator_id.strip():
        qc.append("phase_estimator_provenance_missing")
    if not isinstance(band_id, str) or not band_id.strip():
        qc.append("phase_band_provenance_missing")
    if np.any(np.abs(phase) > np.pi):
        qc.append("phase_values_outside_principal_angle_range")
    return PCVResult(
        estimator_id=PCV_ESTIMATOR_ID,
        schema_id=PCV_SCHEMA_ID,
        value=value,
        units="1",
        computed=True,
        valid=bool(np.isfinite(value)),
        qualified=False,
        status="computed_exploratory" if np.isfinite(value) else "unsupported",
        reason=None if np.isfinite(value) else "non_finite_phase_order_parameter_sd",
        channel_count=channels,
        sample_count_per_channel=samples,
        phase_estimator_id=phase_estimator_id.strip() if isinstance(phase_estimator_id, str) and phase_estimator_id.strip() else None,
        band_id=band_id.strip() if isinstance(band_id, str) and band_id.strip() else None,
        channel_labels=labels,
        channel_order_sha256=order_hash,
        channel_set_sha256=set_hash,
        qc=tuple(qc),
    )


__all__ = ["PCV_ESTIMATOR_ID", "PCV_SCHEMA_ID", "PCVResult", "compute_pcv_phase_order_parameter_sd"]

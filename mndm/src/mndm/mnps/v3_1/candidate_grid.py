"""Opt-in, per-recording MNPS v3.1 candidate execution over named surfaces.

This module only consumes arrays prepared by a caller. The standard candidate
branch receives a 250 Hz, 1–45 Hz filtered window before reference; the module
selects each realization and applies CAR over that exact effective surface.
M1 uses a separate unfiltered, resampled, recorded-reference context window for
its existing 8–13 Hz phase definition. No file I/O or qualification occurs
here, and every requested candidate/surface output is retained when unsupported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Callable, Mapping, Sequence

import numpy as np

from ...features.eeg_entropy import compute_permutation_entropy
from . import eeg_metrics
from .candidates import MNPS31WindowSupport, construct_metadata
from .eeg_metrics import (
    EEGCandidateMeasurement,
    compute_d1_sensor_weighted_global_efficiency,
    compute_d3_sensor_rdm_dispersion,
    compute_e3_sensor_participation_ratio,
    compute_m1_sensor_phase_variability,
    compute_m3_sensor_activity_reconfiguration,
)
from .realizations import CORE19_CHANNELS, MNPS31SensorSurface, MNPS31WindowSlice


CANDIDATE_GRID_SCHEMA = "mndm.mnps_v3_1_candidate_surface_grid.v1"
DS004148_WINDOW_START_SEC = 20.0
DS004148_WINDOW_END_SEC = 40.0
DS004148_SAMPLE_RATE_HZ = 250.0
DS004148_WINDOW_DURATION_SEC = 20.0
DS004148_WINDOW_SAMPLES = 5000
M1_CONTEXT_SAMPLES_EACH_SIDE = 94
M1_CONTEXT_SAMPLE_COUNT = DS004148_WINDOW_SAMPLES + 2 * M1_CONTEXT_SAMPLES_EACH_SIDE
E2_MASTER_SEED = 310930
E2_REFERENCE_COUNT = 32
E2_SEED_DERIVATION_ID = "mndm.mnps_v3_1.e2_seed_derivation.v1"

# Frozen from header_preflight_v3.json for the bounded ds004148 pilot.
DS004148_CORE60_CHANNELS: tuple[str, ...] = (
    "af3", "af4", "af7", "af8", "c1", "c2", "c3", "c4", "c5", "c6",
    "cp1", "cp2", "cp3", "cp4", "cp5", "cp6", "cpz", "cz", "f1", "f2",
    "f3", "f4", "f5", "f6", "f7", "f8", "fc1", "fc2", "fc3", "fc4",
    "fc5", "fc6", "fp1", "fp2", "ft7", "ft8", "fz", "o1", "o2", "oz",
    "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "po3", "po4",
    "po7", "po8", "poz", "pz", "t7", "t8", "tp10", "tp7", "tp8", "tp9",
)
DS004148_CORE60_SET_SHA256 = "c15b4d16a84d4af409e11553db438ed90b3e1ac1e6ac3801ba547f88a99c13ad"


def _ordered_hash(labels: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(label).strip().casefold() for label in labels).encode("utf-8")).hexdigest()


def _set_hash(labels: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(label).strip().casefold() for label in labels)).encode("utf-8")
    ).hexdigest()


def ds004148_candidate_surfaces(native_channel_names: Sequence[str]) -> tuple[MNPS31SensorSurface, ...]:
    """Return the fixed native, ds004148 Core60, and Core19 realizations."""
    native = tuple(str(label).strip() for label in native_channel_names)
    if not native or any(not label for label in native) or len({x.casefold() for x in native}) != len(native):
        raise ValueError("native_channel_labels_must_be_nonempty_and_unique")
    if _set_hash(DS004148_CORE60_CHANNELS) != DS004148_CORE60_SET_SHA256:
        raise RuntimeError("pinned_ds004148_core60_hash_mismatch")
    if not set(label.casefold() for label in CORE19_CHANNELS) <= set(
        label.casefold() for label in DS004148_CORE60_CHANNELS
    ):
        raise RuntimeError("pinned_core19_is_not_a_subset_of_ds004148_core60")
    return (
        MNPS31SensorSurface(
            surface_id="native",
            kind="native",
            channel_names=native,
            reference_mode="average_over_surface",
            allow_partial_channels=True,
            minimum_channels=2,
        ),
        MNPS31SensorSurface(
            surface_id="core60",
            kind="study_common",
            channel_names=DS004148_CORE60_CHANNELS,
            reference_mode="average_over_surface",
            allow_partial_channels=False,
            minimum_channels=2,
        ),
        MNPS31SensorSurface(
            surface_id="core19",
            kind="core19",
            channel_names=CORE19_CHANNELS,
            reference_mode="average_over_surface",
            allow_partial_channels=False,
            minimum_channels=2,
        ),
    )


@dataclass(frozen=True)
class CandidateGridValue:
    """One candidate output returned by a built-in or extension evaluator."""

    value: object = None
    units: str = "1"
    reason: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    metric_estimator_id: str | None = None


@dataclass(frozen=True)
class CandidateGridContext:
    """Surface-selected inputs shared by candidate and extension evaluators."""

    surface: MNPS31SensorSurface
    window: MNPS31WindowSlice
    support: MNPS31WindowSupport
    source_identity: str
    recording_identity: Mapping[str, str]
    preprocessing_id: str
    declared_channel_names: tuple[str, ...]
    effective_channel_names: tuple[str, ...]
    filtered_surface_signals_unreferenced: np.ndarray | None
    filtered_car_signals: np.ndarray | None
    standard_reason: str | None
    recorded_reference_context_signals: np.ndarray | None
    m1_effective_channel_names: tuple[str, ...]
    m1_support: MNPS31WindowSupport | None
    m1_reason: str | None
    recorded_reference_id: str | None
    recorded_reference_channel_names: tuple[str, ...] | None
    bad_channel_names: tuple[str, ...]
    source_provenance: Mapping[str, object]
    surface_details: Mapping[str, object]


@dataclass(frozen=True)
class CandidateGridPlugin:
    """A candidate family that emits one or more realization-specific rows.

    ``estimator_id_templates`` must contain ``{surface_id}``. ``evaluate`` is
    called once for each requested surface and returns a mapping keyed by the
    corresponding resolved estimator IDs. Missing entries become explicit
    unsupported rows.
    """

    candidate_id: str
    estimator_id_templates: tuple[str, ...]
    evaluate: Callable[[CandidateGridContext], Mapping[str, CandidateGridValue]]
    support_source: str = "standard"

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.estimator_id_templates:
            raise ValueError("candidate_grid_plugin_identity_missing")
        if any("{surface_id}" not in template for template in self.estimator_id_templates):
            raise ValueError("candidate_grid_estimator_templates_must_include_surface_id")
        if len(set(self.estimator_id_templates)) != len(self.estimator_id_templates):
            raise ValueError("candidate_grid_estimator_templates_must_be_unique")
        if not callable(self.evaluate):
            raise ValueError("candidate_grid_plugin_evaluator_must_be_callable")
        if self.support_source not in {"standard", "m1"}:
            raise ValueError("candidate_grid_plugin_support_source_invalid")


@dataclass(frozen=True)
class MNPS31CandidateGridRun:
    """Serializable measurements and provenance for one recording/window."""

    rows: tuple[dict[str, object], ...]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        computed = sum(row["status"] == "computed_exploratory" for row in self.rows)
        return {
            "schema": CANDIDATE_GRID_SCHEMA,
            "status": "completed" if computed else "completed_no_supported_values",
            "row_count": len(self.rows),
            "computed_count": computed,
            "unsupported_count": len(self.rows) - computed,
            "qualified": False,
            "claim_status": "no_biological_claim",
            "provenance": self.provenance,
            "rows": list(self.rows),
        }


def _to_json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_to_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    return value


def _finite_json_value(value: object) -> bool:
    if isinstance(value, (bool, str)) or value is None:
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, (tuple, list, np.ndarray)):
        return all(_finite_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(_finite_json_value(item) for item in value.values())
    return False


def _finite_metric_value(value: object) -> bool:
    """Accept finite real scalars or finite nonempty numeric vectors only."""
    if value is None or isinstance(value, (bool, str, bytes)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, np.ndarray):
        return bool(
            value.ndim == 1
            and value.size > 0
            and value.dtype.kind in "iuf"
            and np.isfinite(value).all()
        )
    if isinstance(value, (tuple, list)):
        return bool(value) and all(
            not isinstance(item, (bool, str, bytes, tuple, list, np.ndarray))
            and isinstance(item, (int, float, np.integer, np.floating))
            and bool(np.isfinite(item))
            for item in value
        )
    return False


def _plugin_support_reason(
    plugin: CandidateGridPlugin, context: CandidateGridContext
) -> str | None:
    if plugin.support_source == "m1":
        if context.m1_reason is not None:
            return context.m1_reason
        if context.m1_support is None:
            return "m1_support_audit_missing"
        return eeg_metrics._support_reason(context.m1_support)
    return context.standard_reason or eeg_metrics._support_reason(context.support)


def _measurement_value(measurement: EEGCandidateMeasurement) -> CandidateGridValue:
    return CandidateGridValue(
        value=measurement.value,
        units=measurement.units,
        reason=measurement.reason,
        details=measurement.details,
        metric_estimator_id=measurement.estimator_id,
    )


def _unsupported(reason: str, *, metric_estimator_id: str | None = None, details: Mapping[str, object] | None = None) -> CandidateGridValue:
    return CandidateGridValue(
        value=None,
        reason=reason,
        metric_estimator_id=metric_estimator_id,
        details={} if details is None else dict(details),
    )


def _candidate_ids(candidate_id: str, surface_id: str) -> tuple[str, ...]:
    """Preserve the bounded pilot's v1 IDs for its two named common cores."""
    specs = {
        "E1": ("E1.normalized_PE5_delay1.median.{surface_id}.v1",),
        "E2": ("E2.median_LZ78_exactMultiset.{surface_id}.v1",),
        "E3": (
            "E3.covariance_PR.{surface_id}.v1",
            "E3.covariance_PR_over_channel_count.{surface_id}.v1",
        ),
        "M1": ("M1.alpha_phase_order_parameter_SD.{surface_id}.v1",),
        "M3": (
            "M3.activity_RMS_adjacent_change_vector9.{surface_id}.v1",
            "M3.activity_RMS_adjacent_change_mean9.{surface_id}.v1",
        ),
        "D1": ("D1.absPearson_efficiency_noThreshold.{surface_id}.v1",),
        "D3": ("D3.logPSD_withinSensorCentered_meanDistance.{surface_id}.v1",),
    }
    return tuple(template.format(surface_id=surface_id) for template in specs[candidate_id])


def _e1(context: CandidateGridContext) -> Mapping[str, CandidateGridValue]:
    estimator_id = _candidate_ids("E1", context.surface.surface_id)[0]
    metric_id = "mndm.mnps_v3_1.e1_channel_median_pe.v0"
    if context.standard_reason:
        value = _unsupported(context.standard_reason, metric_estimator_id=metric_id)
    else:
        assert context.filtered_car_signals is not None
        per_channel = np.asarray(
            [compute_permutation_entropy(signal, order=5, delay=1, normalize=True)
             for signal in context.filtered_car_signals],
            dtype=np.float64,
        )
        if per_channel.shape != (len(context.effective_channel_names),) or not np.isfinite(per_channel).all():
            value = _unsupported("permutation_entropy_non_finite_or_wrong_shape", metric_estimator_id=metric_id)
        else:
            value = CandidateGridValue(
                value=float(np.median(per_channel)),
                units="normalized_entropy",
                metric_estimator_id=metric_id,
                details={
                    "per_channel_values": per_channel.tolist(),
                    "channel_count": len(context.effective_channel_names),
                    "order": 5,
                    "delay_samples": 1,
                    "normalized": True,
                    "tie_handling": "stable_mergesort_ordinal_order",
                    "aggregation": "median_over_every_effective_surface_channel",
                    "fixed_83_channel_gate": False,
                    "selection_status": "exploratory_variable_channel_surface_variant",
                },
            )
    return {estimator_id: value}


def _e2_exact_multiset(context: CandidateGridContext, estimator_id: str) -> CandidateGridValue:
    # This variable-channel ds004148 protocol variant is identified by the
    # existing row estimator_id; the stable v3.1 API's E2 ID is specifically
    # fixed to the ANPHY 83-channel realization and does not apply here.
    metric_id = None
    if context.standard_reason:
        return _unsupported(context.standard_reason, metric_estimator_id=metric_id)
    if not context.source_identity.strip():
        return _unsupported("source_identity_missing", metric_estimator_id=metric_id)
    assert context.filtered_car_signals is not None
    ratios: list[float] = []
    per_channel: list[dict[str, object]] = []
    for row, channel_name in zip(context.filtered_car_signals, context.effective_channel_names, strict=True):
        threshold = float(np.median(row))
        symbols = np.asarray(row > threshold, dtype=np.uint8)
        zero_count = int(symbols.size - np.count_nonzero(symbols))
        one_count = int(np.count_nonzero(symbols))
        if zero_count == 0 or one_count == 0:
            return _unsupported(
                "symbol_sequence_has_fewer_than_two_symbols",
                metric_estimator_id=metric_id,
                details={"failed_channel": channel_name},
            )
        seed_material = "\0".join((
            E2_SEED_DERIVATION_ID,
            str(E2_MASTER_SEED),
            context.source_identity,
            context.support.segment_id,
            float(context.support.window_start_sec).hex(),
            float(context.support.window_end_sec).hex(),
            channel_name,
            estimator_id,
        )).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)
        rng = np.random.Generator(np.random.PCG64(seed))
        observed = eeg_metrics._lz78_phrase_count(tuple(int(item) for item in symbols))
        references = [
            eeg_metrics._lz78_phrase_count(tuple(int(item) for item in symbols[rng.permutation(symbols.size)]))
            for _ in range(E2_REFERENCE_COUNT)
        ]
        reference_median = float(np.median(np.asarray(references, dtype=np.float64)))
        if not math.isfinite(reference_median) or reference_median <= 0:
            return _unsupported("undefined_exact_multiset_reference", metric_estimator_id=metric_id,
                                details={"failed_channel": channel_name})
        ratio = float(observed / reference_median)
        ratios.append(ratio)
        per_channel.append({
            "channel_name": channel_name,
            "threshold": threshold,
            "tie_count": int(np.count_nonzero(row == threshold)),
            "symbol_counts": {"0": zero_count, "1": one_count},
            "observed_phrase_count": observed,
            "reference_phrase_counts": references,
            "reference_phrase_count_median": reference_median,
            "ratio": ratio,
            "derived_seed": seed,
        })
    return CandidateGridValue(
        value=float(np.median(np.asarray(ratios, dtype=np.float64))),
        units="dimensionless_ratio",
        metric_estimator_id=metric_id,
        details={
            "construct": "temporal_order_compressibility_relative_to_exact_symbol_multiset_shuffles",
            "channel_count": len(context.effective_channel_names),
            "channel_order_sha256": _ordered_hash(context.effective_channel_names),
            "symbolization": "per_channel_arithmetic_median_strictly_greater_is_1_ties_are_0",
            "reference_count_per_channel": E2_REFERENCE_COUNT,
            "master_seed": E2_MASTER_SEED,
            "seed_derivation_id": E2_SEED_DERIVATION_ID,
            "reference_rng": "numpy.Generator(PCG64)",
            "numpy_version": np.__version__,
            "aggregation": "median_of_all_required_effective_channel_ratios",
            "per_channel": per_channel,
            "selection_status": "existing_ds004148_exploratory_variant",
        },
    )


def _e2(context: CandidateGridContext) -> Mapping[str, CandidateGridValue]:
    estimator_id = _candidate_ids("E2", context.surface.surface_id)[0]
    return {estimator_id: _e2_exact_multiset(context, estimator_id)}


def _e3(context: CandidateGridContext) -> Mapping[str, CandidateGridValue]:
    raw_id, normalized_id = _candidate_ids("E3", context.surface.surface_id)
    if context.standard_reason:
        raw = _unsupported(context.standard_reason, metric_estimator_id="E3.sensor_covariance_participation_ratio.v0")
    else:
        assert context.filtered_car_signals is not None
        raw = _measurement_value(compute_e3_sensor_participation_ratio(
            context.filtered_car_signals, context.effective_channel_names, context.support
        ))
    if raw.reason is not None or raw.value is None:
        normalized = _unsupported(raw.reason or "e3_value_missing", metric_estimator_id=raw.metric_estimator_id,
                                  details={"value_definition_status": "separate_unselected_sensitivity"})
    else:
        normalized = CandidateGridValue(
            value=float(raw.value) / len(context.effective_channel_names),
            units="participation_ratio_per_channel",
            reason=None,
            metric_estimator_id=raw.metric_estimator_id,
            details={
                **dict(raw.details),
                "normalization": "raw_participation_ratio_divided_by_named_surface_channel_count",
                "raw_PR": float(raw.value),
                "channel_count": len(context.effective_channel_names),
                "value_definition_status": "separate_unselected_sensitivity",
            },
        )
    return {raw_id: raw, normalized_id: normalized}


def _m1(context: CandidateGridContext) -> Mapping[str, CandidateGridValue]:
    estimator_id = _candidate_ids("M1", context.surface.surface_id)[0]
    metric_id = "M1.sensor_phase_order_parameter_sd.v0"
    if context.m1_reason:
        return {estimator_id: _unsupported(context.m1_reason, metric_estimator_id=metric_id)}
    assert context.recorded_reference_context_signals is not None
    assert context.m1_support is not None
    from scipy import signal

    sos = signal.butter(4, [8.0, 13.0], btype="bandpass", fs=DS004148_SAMPLE_RATE_HZ, output="sos")
    alpha = signal.sosfiltfilt(
        sos,
        context.recorded_reference_context_signals,
        axis=-1,
        padtype=None,
        padlen=0,
    )
    analytic = signal.hilbert(alpha, axis=-1)
    central = analytic[:, M1_CONTEXT_SAMPLES_EACH_SIDE:M1_CONTEXT_SAMPLES_EACH_SIDE + DS004148_WINDOW_SAMPLES]
    amplitude = np.abs(central)
    if not np.isfinite(amplitude).all() or np.any(amplitude == 0.0):
        return {estimator_id: _unsupported("invalid_or_zero_analytic_amplitude", metric_estimator_id=metric_id)}
    phases = np.asarray(np.angle(central), dtype=np.float64)
    measurement = compute_m1_sensor_phase_variability(
        phases,
        context.m1_effective_channel_names,
        context.m1_support,
        phase_estimator_id="M1.alpha_8_13Hz_order4_SOS_sosfiltfilt_Hilbert.core_input.v1",
        band_id=f"alpha_8_13_hz_{context.surface.surface_id}_acquisition_reference",
    )
    result = _measurement_value(measurement)
    return {estimator_id: CandidateGridValue(
        value=result.value,
        units=result.units,
        reason=result.reason,
        metric_estimator_id=result.metric_estimator_id,
        details={
            **dict(result.details),
            "source_reference": "recorded_acquisition_reference_no_core_CAR",
            "source_reference_id": context.recorded_reference_id,
            "reference_channel_names": context.recorded_reference_channel_names,
            "reference_channel_count": (None if context.recorded_reference_channel_names is None else len(context.recorded_reference_channel_names)),
            "reference_channel_order_sha256": (None if context.recorded_reference_channel_names is None else _ordered_hash(context.recorded_reference_channel_names)),
            "reference_channel_set_sha256": (None if context.recorded_reference_channel_names is None else _set_hash(context.recorded_reference_channel_names)),
            "context_samples_each_side": M1_CONTEXT_SAMPLES_EACH_SIDE,
            "context_seconds_each_side": M1_CONTEXT_SAMPLES_EACH_SIDE / DS004148_SAMPLE_RATE_HZ,
            "context_support_claim": "operational_margin_not_exact_filter_or_Hilbert_finite_support",
            "phase_filter": "butterworth_order4_SOS_8_13_Hz_sosfiltfilt_padtype_null_padlen_0",
            "analytic_signal": "scipy.signal.hilbert",
            "average_reference_applied": False,
        },
    )}


def _m3(context: CandidateGridContext) -> Mapping[str, CandidateGridValue]:
    vector_id, mean_id = _candidate_ids("M3", context.surface.surface_id)
    if context.standard_reason:
        unavailable = _unsupported(context.standard_reason, metric_estimator_id="M3.sensor_activity_adjacent_pearson_change.v0")
        return {vector_id: unavailable, mean_id: unavailable}
    assert context.filtered_car_signals is not None
    blocks = context.filtered_car_signals.reshape(len(context.effective_channel_names), 10, 500).transpose(1, 0, 2)
    activity = np.sqrt(np.mean(blocks * blocks, axis=2))
    intervals = np.asarray(
        [[context.window.support.window_start_sec + 2 * index, context.window.support.window_start_sec + 2 * (index + 1)] for index in range(10)],
        dtype=np.float64,
    )
    measurement = compute_m3_sensor_activity_reconfiguration(
        activity,
        intervals,
        context.effective_channel_names,
        context.support,
        vector_construction_id="RMS_per_sensor_per_adjacent_2s_block.core.v1",
        vector_scaling_id="metric_internal_maxabs_then_centered_Pearson_r",
    )
    result = _measurement_value(measurement)
    if result.reason is not None or result.value is None:
        return {vector_id: result, mean_id: result}
    vector = tuple(float(value) for value in result.value)  # type: ignore[arg-type]
    if len(vector) != 9 or not all(math.isfinite(value) for value in vector):
        invalid = _unsupported("m3_vector_not_finite_nine_changes", metric_estimator_id=result.metric_estimator_id)
        return {vector_id: invalid, mean_id: invalid}
    mean = CandidateGridValue(
        value=float(np.mean(vector)),
        units=result.units,
        metric_estimator_id=result.metric_estimator_id,
        details={
            **dict(result.details),
            "source_vector": list(vector),
            "aggregation": "arithmetic_mean_of_all_nine_adjacent_changes",
            "selection_status": "unselected_hypothesis_sensitivity",
        },
    )
    return {vector_id: result, mean_id: mean}


def _d1(context: CandidateGridContext) -> Mapping[str, CandidateGridValue]:
    estimator_id = _candidate_ids("D1", context.surface.surface_id)[0]
    if context.standard_reason:
        return {estimator_id: _unsupported(context.standard_reason, metric_estimator_id="D1.sensor_weighted_global_efficiency_invweight.v0")}
    assert context.filtered_car_signals is not None
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(context.filtered_car_signals)
    if correlation.shape != (len(context.effective_channel_names), len(context.effective_channel_names)) or not np.isfinite(correlation).all():
        return {estimator_id: _unsupported("non_finite_or_constant_channel_coupling", metric_estimator_id="D1.sensor_weighted_global_efficiency_invweight.v0")}
    weights = np.abs(correlation)
    np.fill_diagonal(weights, 0.0)
    measurement = compute_d1_sensor_weighted_global_efficiency(
        weights,
        context.support,
        node_order_sha256=_ordered_hash(context.effective_channel_names),
        edge_estimator_id="absolute_zero_lag_Pearson_r_full_20s_core.v1",
        edge_rule_id="no_threshold",
    )
    return {estimator_id: _measurement_value(measurement)}


def _d3(context: CandidateGridContext) -> Mapping[str, CandidateGridValue]:
    estimator_id = _candidate_ids("D3", context.surface.surface_id)[0]
    metric_id = "D3.sensor_rdm_mean_euclidean.v0"
    if context.standard_reason:
        return {estimator_id: _unsupported(context.standard_reason, metric_estimator_id=metric_id)}
    assert context.filtered_car_signals is not None
    from scipy import signal

    frequencies, psd = signal.welch(
        context.filtered_car_signals,
        fs=DS004148_SAMPLE_RATE_HZ,
        window=signal.get_window("hann", 1000, fftbins=True),
        nperseg=1000,
        noverlap=0,
        nfft=1000,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    keep = (frequencies >= 1.0) & (frequencies <= 45.0)
    if not np.any(keep) or not np.isfinite(psd[:, keep]).all() or np.any(psd[:, keep] <= 0.0):
        return {estimator_id: _unsupported("invalid_log_psd_feature_profile", metric_estimator_id=metric_id)}
    profiles = np.log10(psd[:, keep] * 1e12)
    centered = profiles - np.mean(profiles, axis=1, keepdims=True)
    measurement = compute_d3_sensor_rdm_dispersion(
        centered,
        context.effective_channel_names,
        context.support,
        feature_construction_id="within_sensor_centered_log10_Welch_PSD_1_to_45Hz.core.v1",
        scaling_id="PSD_times_1e12_microvolt_squared_per_Hz",
        feature_units="log10_microvolt_squared_per_hz_centered_within_sensor",
    )
    result = _measurement_value(measurement)
    return {estimator_id: CandidateGridValue(
        value=result.value,
        units=result.units,
        reason=result.reason,
        metric_estimator_id=result.metric_estimator_id,
        details={
            **dict(result.details),
            "retained_frequency_bin_count": int(np.count_nonzero(keep)),
            "welch_nperseg": 1000,
            "welch_noverlap": 0,
            "welch_nfft": 1000,
        },
    )}


def builtin_candidate_grid_plugins() -> tuple[CandidateGridPlugin, ...]:
    """Return the nine outputs from the existing ds004148 candidate pilot."""
    return (
        CandidateGridPlugin("E1", ("E1.normalized_PE5_delay1.median.{surface_id}.v1",), _e1),
        CandidateGridPlugin("E2", ("E2.median_LZ78_exactMultiset.{surface_id}.v1",), _e2),
        CandidateGridPlugin("E3", (
            "E3.covariance_PR.{surface_id}.v1",
            "E3.covariance_PR_over_channel_count.{surface_id}.v1",
        ), _e3),
        CandidateGridPlugin(
            "M1",
            ("M1.alpha_phase_order_parameter_SD.{surface_id}.v1",),
            _m1,
            support_source="m1",
        ),
        CandidateGridPlugin("M3", (
            "M3.activity_RMS_adjacent_change_vector9.{surface_id}.v1",
            "M3.activity_RMS_adjacent_change_mean9.{surface_id}.v1",
        ), _m3),
        CandidateGridPlugin("D1", ("D1.absPearson_efficiency_noThreshold.{surface_id}.v1",), _d1),
        CandidateGridPlugin("D3", ("D3.logPSD_withinSensorCentered_meanDistance.{surface_id}.v1",), _d3),
    )


def _validate_window(window: MNPS31WindowSlice) -> None:
    support = window.support
    if (
        not math.isclose(support.sample_rate_hz, DS004148_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1e-12)
        or support.sample_count != DS004148_WINDOW_SAMPLES
        or not math.isclose(window.sample_time_origin_sec, 0.0, rel_tol=0.0, abs_tol=1e-12)
        or window.end_sample_exclusive - window.start_sample != DS004148_WINDOW_SAMPLES
        or not math.isclose(
            support.window_end_sec - support.window_start_sec,
            DS004148_WINDOW_DURATION_SEC,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            support.window_start_sec,
            window.start_sample / DS004148_SAMPLE_RATE_HZ,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            support.window_end_sec,
            window.end_sample_exclusive / DS004148_SAMPLE_RATE_HZ,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("candidate_grid_requires_a_sample_aligned_20_second_250hz_window")


def _validate_surfaces(
    surfaces: Sequence[MNPS31SensorSurface], source_channel_names: Sequence[str]
) -> None:
    if tuple(surface.surface_id for surface in surfaces) != ("native", "core60", "core19"):
        raise ValueError("candidate_grid_requires_native_core60_core19_surfaces_in_order")
    native, core60, core19 = surfaces
    if native.kind != "native" or not native.allow_partial_channels:
        raise ValueError("native_surface_must_allow_partial_channels")
    if tuple(label.strip().casefold() for label in native.channel_names) != tuple(
        label.strip().casefold() for label in source_channel_names
    ):
        raise ValueError("native_surface_must_match_source_channel_order")
    if core60.kind != "study_common" or tuple(label.casefold() for label in core60.channel_names) != tuple(
        label.casefold() for label in DS004148_CORE60_CHANNELS
    ):
        raise ValueError("core60_surface_must_match_frozen_ds004148_core60_order")
    if core60.allow_partial_channels:
        raise ValueError("core60_surface_must_be_strict")
    if core19.kind != "core19" or tuple(label.casefold() for label in core19.channel_names) != tuple(
        label.casefold() for label in CORE19_CHANNELS
    ):
        raise ValueError("core19_surface_must_match_frozen_core19_order")
    if core19.allow_partial_channels:
        raise ValueError("core19_surface_must_be_strict")


def _prepare_surface_context(
    *,
    surface: MNPS31SensorSurface,
    source_values: np.ndarray,
    source_labels: tuple[str, ...],
    context_values: np.ndarray | None,
    context_labels: tuple[str, ...],
    window: MNPS31WindowSlice,
    source_identity: str,
    recording_identity: Mapping[str, str],
    preprocessing_id: str,
    m1_support: MNPS31WindowSupport | None,
    recorded_reference_id: str | None,
    recorded_reference_channel_names: tuple[str, ...] | None,
    bad_channel_names: tuple[str, ...],
    source_provenance: Mapping[str, object],
) -> CandidateGridContext:
    support = window.support
    lookup = {label.casefold(): index for index, label in enumerate(source_labels)}
    bad = {label.casefold() for label in bad_channel_names}
    requested = surface.channel_names
    missing = tuple(label for label in requested if label.casefold() not in lookup)
    extra = tuple(label for label in source_labels if label.casefold() not in {x.casefold() for x in requested})
    present = tuple((label, lookup[label.casefold()]) for label in requested if label.casefold() in lookup)
    finite = {index: bool(np.isfinite(source_values[index]).all()) for _, index in present}
    rejected = tuple(
        label for label, index in present
        if label.casefold() in bad or not finite[index]
    )
    usable = tuple(
        (source_labels[index], index) for label, index in present
        if label.casefold() not in bad and finite[index]
    )
    selected_labels = tuple(label for label, _ in usable)
    standard_reason = eeg_metrics._support_reason(support)
    if standard_reason is None and surface.reference_mode != "average_over_surface":
        standard_reason = "standard_candidates_require_average_reference_over_exact_surface"
    elif standard_reason is None and missing and not surface.allow_partial_channels:
        standard_reason = "required_surface_channel_missing"
    elif standard_reason is None and rejected and not surface.allow_partial_channels:
        standard_reason = "required_surface_channel_unavailable"
    elif standard_reason is None and len(usable) < surface.minimum_channels:
        standard_reason = "insufficient_usable_channels"

    unreferenced = None
    referenced = None
    if standard_reason is None:
        picked = np.asarray(source_values[[index for _, index in usable]], dtype=np.float64)
        unreferenced = picked.copy()
        referenced = picked - np.mean(picked, axis=0, keepdims=True)
        unreferenced.setflags(write=False)
        referenced.setflags(write=False)

    m1_lookup = {label.casefold(): index for index, label in enumerate(context_labels)}
    m1_missing = (
        tuple(requested)
        if context_values is None
        else tuple(label for label in requested if label.casefold() not in m1_lookup)
    )
    m1_present = (
        ()
        if context_values is None
        else tuple((label, m1_lookup[label.casefold()]) for label in requested if label.casefold() in m1_lookup)
    )
    m1_finite = ({index: bool(np.isfinite(context_values[index]).all()) for _, index in m1_present}
                 if context_values is not None else {})
    m1_rejected = tuple(
        label for label, index in m1_present
        if label.casefold() in bad or not m1_finite[index]
    )
    m1_usable = tuple(
        (context_labels[index], index) for label, index in m1_present
        if label.casefold() not in bad and m1_finite[index]
    )
    m1_labels = tuple(label for label, _ in m1_usable)
    m1_reason = None
    m1_selected = None
    if context_values is None:
        m1_reason = "recorded_reference_context_missing"
    elif not recorded_reference_id or not recorded_reference_id.strip():
        m1_reason = "recorded_reference_identity_missing"
    elif m1_support is None:
        m1_reason = "m1_support_audit_missing"
    elif m1_missing and not surface.allow_partial_channels:
        m1_reason = "m1_required_surface_channel_missing"
    elif m1_rejected and not surface.allow_partial_channels:
        m1_reason = "m1_required_surface_channel_unavailable"
    elif len(m1_usable) < surface.minimum_channels:
        m1_reason = "m1_insufficient_usable_channels"
    elif eeg_metrics._support_reason(m1_support) is not None:
        m1_reason = eeg_metrics._support_reason(m1_support)
    elif (
        m1_support.window_start_sec != support.window_start_sec
        or m1_support.window_end_sec != support.window_end_sec
        or m1_support.sample_rate_hz != DS004148_SAMPLE_RATE_HZ
        or m1_support.sample_count != DS004148_WINDOW_SAMPLES
    ):
        m1_reason = "m1_support_window_mismatch"
    else:
        assert context_values is not None
        m1_selected = np.asarray(context_values[[index for _, index in m1_usable]], dtype=np.float64)
        m1_selected.setflags(write=False)
        if m1_selected.shape[1] != M1_CONTEXT_SAMPLE_COUNT:
            m1_reason = "recorded_reference_context_sample_count_mismatch"
            m1_selected = None

    if context_values is not None and context_values.shape[1] != M1_CONTEXT_SAMPLE_COUNT and m1_reason is None:
        m1_reason = "recorded_reference_context_sample_count_mismatch"
        m1_selected = None

    surface_details: dict[str, object] = {
        "surface_id": surface.surface_id,
        "surface_kind": surface.kind,
        "declared_channel_labels": list(requested),
        "declared_channel_count": len(requested),
        "declared_channel_order_sha256": _ordered_hash(requested),
        "declared_channel_set_sha256": _set_hash(requested),
        "effective_channel_labels": list(selected_labels),
        "effective_channel_count": len(selected_labels),
        "effective_channel_order_sha256": _ordered_hash(selected_labels),
        "effective_channel_set_sha256": _set_hash(selected_labels),
        "channel_coverage_fraction": len(selected_labels) / len(requested),
        "missing_channel_labels": list(missing),
        "rejected_channel_labels": list(rejected),
        "extra_source_channel_labels": list(extra),
        "unmatched_bad_channel_labels": [label for label in bad_channel_names if label.casefold() not in {x.casefold() for x in source_labels}],
        "qc_flags": [],
        "reference_mode": "average_over_effective_surface_channels",
        "reference_id": "common_average",
        "reference_channel_labels": list(selected_labels),
        "reference_channel_count": len(selected_labels),
        "reference_channel_order_sha256": _ordered_hash(selected_labels),
        "reference_channel_set_sha256": _set_hash(selected_labels),
        "cross_montage_equivalent": "unknown",
        "m1_source_reference_mode": "recorded_input_reference",
        "m1_source_reference_id": recorded_reference_id,
        "m1_reference_channel_labels": (None if recorded_reference_channel_names is None else list(recorded_reference_channel_names)),
        "m1_reference_channel_count": (None if recorded_reference_channel_names is None else len(recorded_reference_channel_names)),
        "m1_reference_channel_order_sha256": (None if recorded_reference_channel_names is None else _ordered_hash(recorded_reference_channel_names)),
        "m1_reference_channel_set_sha256": (None if recorded_reference_channel_names is None else _set_hash(recorded_reference_channel_names)),
        "m1_effective_channel_labels": list(m1_labels),
        "m1_effective_channel_count": len(m1_labels),
        "m1_channel_coverage_fraction": len(m1_labels) / len(requested),
        "m1_missing_channel_labels": list(m1_missing),
        "m1_rejected_channel_labels": list(m1_rejected),
    }
    if extra:
        surface_details["qc_flags"].append("source_channels_outside_requested_surface")  # type: ignore[union-attr]
    if rejected or m1_rejected:
        surface_details["qc_flags"].append("one_or_more_surface_channels_unavailable")  # type: ignore[union-attr]
    if standard_reason:
        surface_details["qc_flags"].append("standard_surface_unsupported")  # type: ignore[union-attr]
    if m1_reason:
        surface_details["qc_flags"].append("m1_surface_unsupported")  # type: ignore[union-attr]
    if recorded_reference_channel_names is None:
        surface_details["qc_flags"].append("m1_recorded_reference_set_provenance_missing")  # type: ignore[union-attr]

    return CandidateGridContext(
        surface=surface,
        window=window,
        support=support,
        source_identity=source_identity,
        recording_identity=dict(recording_identity),
        preprocessing_id=preprocessing_id,
        declared_channel_names=requested,
        effective_channel_names=selected_labels,
        filtered_surface_signals_unreferenced=unreferenced,
        filtered_car_signals=referenced,
        standard_reason=standard_reason,
        recorded_reference_context_signals=m1_selected,
        m1_effective_channel_names=m1_labels,
        m1_support=m1_support,
        m1_reason=m1_reason,
        recorded_reference_id=recorded_reference_id,
        recorded_reference_channel_names=recorded_reference_channel_names,
        bad_channel_names=bad_channel_names,
        source_provenance=dict(source_provenance),
        surface_details=surface_details,
    )


def _runtime_versions() -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "mndm": version("mndm"),
    }


def _implementation_hashes(plugins: Sequence[CandidateGridPlugin]) -> dict[str, str]:
    """Hash built-ins, optional candidate implementations, and plugin code."""
    paths = {
        "candidate_grid": Path(__file__),
        "eeg_metrics": Path(eeg_metrics.__file__),
    }
    from ...features import eeg_entropy
    paths["eeg_entropy"] = Path(eeg_entropy.__file__)
    plugin_ids = {plugin.candidate_id for plugin in plugins}
    if "M2" in plugin_ids:
        from . import m2_acf_mobility

        paths["m2_acf_mobility"] = Path(m2_acf_mobility.__file__)
    if "D2" in plugin_ids:
        from . import d2_experimental

        paths["d2_experimental"] = Path(d2_experimental.__file__)
    for plugin in plugins:
        module_name = getattr(plugin.evaluate, "__module__", None)
        module = sys.modules.get(module_name) if isinstance(module_name, str) else None
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ValueError(f"candidate_grid_plugin_source_unavailable:{plugin.candidate_id}")
        paths[f"plugin_{plugin.candidate_id}_evaluator"] = Path(module_file)
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"candidate_grid_implementation_file_missing:{name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes


def run_mnps31_candidate_surface_grid(
    filtered_window_signals: np.ndarray,
    source_channel_names: Sequence[str],
    surfaces: Sequence[MNPS31SensorSurface],
    window: MNPS31WindowSlice,
    *,
    source_identity: str,
    recording_identity: Mapping[str, str],
    preprocessing_id: str,
    recorded_reference_context: np.ndarray | None = None,
    recorded_reference_channel_names: Sequence[str] | None = None,
    recorded_reference_id: str | None = None,
    recorded_reference_channels: Sequence[str] | None = None,
    m1_support: MNPS31WindowSupport | None = None,
    bad_channel_names: Sequence[str] = (),
    source_provenance: Mapping[str, object] | None = None,
    extension_plugins: Sequence[CandidateGridPlugin] = (),
) -> MNPS31CandidateGridRun:
    """Run existing v3.1 candidate definitions for native/Core60/Core19.

    ``filtered_window_signals`` must be channel-by-5000 for a sample-aligned
    20-second interval at 250 Hz, filtered 1–45 Hz, and still in its input
    reference. Candidate-specific CAR is applied after exact surface
    selection. ``recorded_reference_context`` must be the unfiltered but
    resampled 250 Hz context matrix with 94 real samples before and after the
    central window; it is used only by M1.
    """
    _validate_window(window)
    if not isinstance(preprocessing_id, str) or not preprocessing_id.strip():
        raise ValueError("candidate_grid_preprocessing_id_missing")
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise ValueError("candidate_grid_source_identity_missing")
    if not isinstance(recording_identity, Mapping) or not recording_identity:
        raise ValueError("candidate_grid_recording_identity_missing")
    if any(not isinstance(value, str) or not value.strip() for value in recording_identity.values()):
        raise ValueError("candidate_grid_recording_identity_invalid")
    try:
        json.dumps(_to_json_value(source_provenance or {}), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("candidate_grid_source_provenance_not_finite_json") from exc

    try:
        source_values = np.asarray(filtered_window_signals, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("filtered_window_signals_must_be_numeric") from exc
    source_labels = tuple(str(label).strip() for label in source_channel_names)
    source_keys = tuple(label.casefold() for label in source_labels)
    if (
        source_values.ndim != 2
        or source_values.shape != (len(source_labels), DS004148_WINDOW_SAMPLES)
        or not source_labels
        or any(not label for label in source_labels)
        or len(set(source_keys)) != len(source_keys)
    ):
        raise ValueError("filtered_window_signals_must_match_unique_labels_and_5000_samples")
    _validate_surfaces(surfaces, source_labels)

    try:
        context_values = None if recorded_reference_context is None else np.asarray(recorded_reference_context, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("recorded_reference_context_must_be_numeric") from exc
    context_labels = tuple(
        str(label).strip() for label in (
            recorded_reference_channel_names if recorded_reference_channel_names is not None else source_labels
        )
    )
    if context_values is not None and (
        context_values.ndim != 2
        or context_values.shape != (len(context_labels), M1_CONTEXT_SAMPLE_COUNT)
        or not context_labels
        or any(not label for label in context_labels)
        or len({label.casefold() for label in context_labels}) != len(context_labels)
    ):
        raise ValueError("recorded_reference_context_must_match_unique_labels_and_5188_samples")
    if context_values is None:
        context_labels = source_labels

    bad_labels = tuple(str(label).strip() for label in bad_channel_names)
    if any(not label for label in bad_labels) or len({label.casefold() for label in bad_labels}) != len(bad_labels):
        raise ValueError("bad_channel_names_must_be_nonempty_and_unique")
    reference_labels = None if recorded_reference_channels is None else tuple(
        str(label).strip() for label in recorded_reference_channels
    )
    if reference_labels is not None and (not reference_labels or any(not label for label in reference_labels)):
        raise ValueError("recorded_reference_channel_labels_invalid")

    plugins = (*builtin_candidate_grid_plugins(), *tuple(extension_plugins))
    plugin_ids = [plugin.candidate_id for plugin in plugins]
    if len(set(plugin_ids)) != len(plugin_ids):
        raise ValueError("candidate_grid_plugin_candidate_ids_must_be_unique")
    provenance = {
        "grid_schema": CANDIDATE_GRID_SCHEMA,
        "recording_identity": dict(recording_identity),
        "source_identity": source_identity,
        "window": {
            "window_id": window.window_id,
            "start_sec": window.support.window_start_sec,
            "end_sec": window.support.window_end_sec,
            "sample_rate_hz": window.support.sample_rate_hz,
            "sample_count": window.support.sample_count,
            "segment_id": window.support.segment_id,
            "provenance": dict(window.provenance),
        },
        "preprocessing": {
            "pipeline_id": preprocessing_id,
            "filtered_candidate_branch": {
                "target_sample_rate_hz": DS004148_SAMPLE_RATE_HZ,
                "bandpass_hz": [1.0, 45.0],
                "input_reference": "recorded_source_reference_before_surface_selection",
                "candidate_reference": "instantaneous_average_over_exact_effective_surface_after_filter",
            },
            "m1_branch": {
                "target_sample_rate_hz": DS004148_SAMPLE_RATE_HZ,
                "input_reference": "recorded_acquisition_reference_no_core_CAR",
                "reference_id": recorded_reference_id,
                "bandpass_hz": [8.0, 13.0],
                "phase_estimator_id": "M1.alpha_8_13Hz_order4_SOS_sosfiltfilt_Hilbert.core_input.v1",
                "context_samples_each_side": M1_CONTEXT_SAMPLES_EACH_SIDE,
                "context_support_claim": "operational_margin_not_exact_filter_or_Hilbert_finite_support",
            },
        },
        "source_provenance": {} if source_provenance is None else dict(source_provenance),
        "plugins": [
            {
                "candidate_id": plugin.candidate_id,
                "estimator_id_templates": list(plugin.estimator_id_templates),
                "support_source": plugin.support_source,
            }
            for plugin in plugins
        ],
        "implementation_sha256": _implementation_hashes(plugins),
        "runtime": _runtime_versions(),
        "qualified": False,
        "claim_status": "no_biological_claim",
    }

    rows: list[dict[str, object]] = []
    for surface in surfaces:
        context = _prepare_surface_context(
            surface=surface,
            source_values=source_values,
            source_labels=source_labels,
            context_values=context_values,
            context_labels=context_labels,
            window=window,
            source_identity=source_identity,
            recording_identity=recording_identity,
            preprocessing_id=preprocessing_id,
            m1_support=m1_support,
            recorded_reference_id=recorded_reference_id,
            recorded_reference_channel_names=reference_labels,
            bad_channel_names=bad_labels,
            source_provenance={} if source_provenance is None else source_provenance,
        )
        for plugin in plugins:
            expected_ids = tuple(template.format(surface_id=surface.surface_id) for template in plugin.estimator_id_templates)
            support_reason = _plugin_support_reason(plugin, context)
            if support_reason is not None:
                results = {}
                unexpected = ()
                exception_reason = support_reason
                exception_details: Mapping[str, object] = {
                    "support_source": plugin.support_source,
                    "plugin_evaluation_skipped": True,
                }
            else:
                try:
                    results = plugin.evaluate(context)
                    if not isinstance(results, Mapping):
                        raise TypeError("candidate_plugin_must_return_estimator_mapping")
                    unexpected = tuple(key for key in results if key not in expected_ids)
                except Exception as exc:
                    results = {}
                    unexpected = ()
                    exception_reason = f"candidate_plugin_exception:{type(exc).__name__}"
                    exception_details = {"exception_message": str(exc)}
                else:
                    exception_reason = ""
                    exception_details = {}
            if unexpected:
                exception_reason = "candidate_plugin_estimator_ids_mismatch"
                exception_details = {"unexpected_estimator_ids": list(unexpected)}
                results = {}
            for estimator_id in expected_ids:
                value = results.get(estimator_id)
                if value is None:
                    value = _unsupported(
                        exception_reason or "candidate_plugin_result_missing",
                        details=exception_details,
                    )
                if not isinstance(value, CandidateGridValue):
                    value = _unsupported("candidate_plugin_value_type_invalid")
                if not isinstance(value.details, Mapping):
                    value = _unsupported(
                        "candidate_plugin_details_invalid",
                        metric_estimator_id=value.metric_estimator_id,
                    )
                if not isinstance(value.units, str) or not value.units.strip():
                    value = _unsupported(
                        "candidate_plugin_units_invalid",
                        metric_estimator_id=value.metric_estimator_id,
                        details=value.details,
                    )
                metric_value_is_valid = _finite_metric_value(value.value)
                computed = value.reason is None and metric_value_is_valid
                if value.reason is not None:
                    reason = value.reason
                elif value.value is None:
                    reason = "candidate_returned_no_value"
                elif not metric_value_is_valid:
                    reason = "candidate_returned_nonnumeric_or_nonfinite_value"
                else:
                    reason = None
                is_m1 = plugin.support_source == "m1"
                branch = "m1_recorded_reference_8_13_hz" if is_m1 else "surface_CAR_after_1_45_hz_filter"
                row_surface_details = dict(context.surface_details)
                if is_m1:
                    m1_labels = context.m1_effective_channel_names
                    row_surface_details.update({
                        "standard_effective_channel_labels": list(context.effective_channel_names),
                        "standard_effective_channel_count": len(context.effective_channel_names),
                        "effective_channel_labels": list(m1_labels),
                        "effective_channel_count": len(m1_labels),
                        "effective_channel_order_sha256": _ordered_hash(m1_labels),
                        "effective_channel_set_sha256": _set_hash(m1_labels),
                        "channel_coverage_fraction": len(m1_labels) / len(context.declared_channel_names),
                        "reference_mode": "recorded_input_reference",
                        "reference_id": recorded_reference_id,
                    })
                serialized_value = _to_json_value(value.value) if computed else None
                reference = {
                    "mode": "recorded_input_reference" if is_m1 else "average_over_effective_surface_channels",
                    "reference_id": recorded_reference_id if is_m1 else "common_average",
                    "reference_channel_labels": (
                        None if is_m1 and reference_labels is None
                        else list(reference_labels or ()) if is_m1
                        else list(context.effective_channel_names)
                    ),
                    "reference_channel_count": (
                        None if is_m1 and reference_labels is None
                        else len(reference_labels or ()) if is_m1
                        else len(context.effective_channel_names)
                    ),
                    "reference_channel_order_sha256": (
                        None if is_m1 and reference_labels is None
                        else _ordered_hash(reference_labels or ()) if is_m1
                        else _ordered_hash(context.effective_channel_names)
                    ),
                    "reference_channel_set_sha256": (
                        None if is_m1 and reference_labels is None
                        else _set_hash(reference_labels or ()) if is_m1
                        else _set_hash(context.effective_channel_names)
                    ),
                }
                # Keep flat and nested reference metadata on the same branch.
                row_surface_details.update({
                    "reference_mode": reference["mode"],
                    **{key: item for key, item in reference.items() if key != "mode"},
                })
                row_details = {
                    **row_surface_details,
                    "branch": branch,
                    "reference": reference,
                    "support": asdict(context.m1_support if is_m1 and context.m1_support is not None else context.support),
                    "metric_details": dict(value.details),
                }
                try:
                    serialized_details = _to_json_value(row_details)
                    json.dumps(serialized_details, allow_nan=False)
                except (TypeError, ValueError, OverflowError):
                    computed = False
                    reason = "candidate_details_not_finite_json"
                    serialized_value = None
                    row_details["metric_details"] = {
                        "serialization_error": "candidate_details_not_finite_json"
                    }
                    serialized_details = _to_json_value(row_details)
                rows.append({
                    "candidate": plugin.candidate_id,
                    **construct_metadata(plugin.candidate_id),
                    "estimator_id": estimator_id,
                    "metric_estimator_id": value.metric_estimator_id,
                    "surface_id": surface.surface_id,
                    "surface_kind": surface.kind,
                    "recording_identity": dict(recording_identity),
                    "source_identity": source_identity,
                    "preprocessing_id": preprocessing_id,
                    "window": {
                        "window_id": window.window_id,
                        "start_sec": window.support.window_start_sec,
                        "end_sec": window.support.window_end_sec,
                        "sample_rate_hz": window.support.sample_rate_hz,
                        "sample_count": window.support.sample_count,
                        "segment_id": window.support.segment_id,
                    },
                    "value": serialized_value if computed else None,
                    "units": value.units,
                    "computed": computed,
                    "valid": computed,
                    "qualified": False,
                    "status": "computed_exploratory" if computed else "unsupported",
                    "reason": reason,
                    "details": serialized_details,
                    "source_provenance": _to_json_value(context.source_provenance),
                    "claim_status": "no_biological_claim",
                })

    json.dumps({"provenance": provenance, "rows": rows}, allow_nan=False)
    return MNPS31CandidateGridRun(tuple(rows), provenance)


__all__ = [
    "CANDIDATE_GRID_SCHEMA",
    "DS004148_CORE60_CHANNELS",
    "DS004148_CORE60_SET_SHA256",
    "DS004148_WINDOW_START_SEC",
    "DS004148_WINDOW_END_SEC",
    "DS004148_SAMPLE_RATE_HZ",
    "DS004148_WINDOW_DURATION_SEC",
    "DS004148_WINDOW_SAMPLES",
    "M1_CONTEXT_SAMPLES_EACH_SIDE",
    "M1_CONTEXT_SAMPLE_COUNT",
    "CandidateGridContext",
    "CandidateGridPlugin",
    "CandidateGridValue",
    "MNPS31CandidateGridRun",
    "builtin_candidate_grid_plugins",
    "ds004148_candidate_surfaces",
    "run_mnps31_candidate_surface_grid",
]

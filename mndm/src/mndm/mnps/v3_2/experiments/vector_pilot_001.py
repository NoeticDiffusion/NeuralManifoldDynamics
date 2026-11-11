"""Bounded ANPHY EPCTL01 ED-OI-NIE engineering pilot.

This runner uses one audited E1-G2 source-grid load and passes each ready
float64 window matrix unchanged to the three v3.2 matrix kernels. It preserves
source support and family-local failure states in the output records. It does
not integrate with the production pipeline or make scientific qualification
claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[6]
for _source_root in (REPO_ROOT / "core" / "src", REPO_ROOT / "mndm" / "src"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))
DEFAULT_OVERLAY = REPO_ROOT / "mndm" / "config" / "mnps_v3_2_anphy_epctl01_e1_g2_grid_20260929.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "VECTOR-PILOT-001"
PROTOCOL_PATH = REPO_ROOT / "docs" / "api" / "mnps_v3_2_vector_pilot_001_protocol.rst"
RUNNER_PATH = Path(__file__).resolve()
EXPERIMENT_ID = "nmdm.mnps_v3_2.vector_pilot_001.v1"
RESULT_SCHEMA_ID = "mndm.mnps_v3_2.vector_pilot_window_result.v1"
MANIFEST_SCHEMA_ID = "mndm.mnps_v3_2.vector_pilot_run_manifest.v1"

ED_REALIZATION_ID = "ED.anphy_epctl01_E1_G2_fixed83_avgref.v1"
OI_REALIZATION_ID = "OI.anphy_epctl01_G2_fixed83_PE5_delay1_median.v1"
NIE_REALIZATION_ID = "NIE.anphy_epctl01_E1_G2_fixed83_avgref_abs_pearson_no_threshold.v1"
NIE_EDGE_ESTIMATOR_ID = "absolute_zero_lag_Pearson_r"
NIE_EDGE_RULE_ID = "no_threshold"

EXPECTED_SOURCE_SHA256 = "a16ddc068edcc4ce2ff8c499cca51b9a890b76b7ddb569f9a96978013f8b5e8c"
EXPECTED_POSITIONS_SHA256 = "b5bbc85ef1e840e0ca703c62c22a1000dd3252f32210e251b8d0e81123fa1bd2"
EXPECTED_OVERLAY_SHA256 = "3f1d3135d45ad43648ef1f1c24b0732970ad1492209a414804c35c6eed240824"
EXPECTED_LEGACY_ORDER_SHA256 = "5a3d09949a04c1a1483e8077e7e51ce96773c3092e0b92c9343255e9c1eeaad4"
EXPECTED_ORDER_SHA256 = "9559b64ead401c7399f6218720bc47fe9f7b7c8b785c27c48526542412e2e1e7"
EXPECTED_SET_SHA256 = "9819377836ecdb54673e8eb5e84a9dc4c19ee0a42e0ff665c9f56794e914fcce"
EXPECTED_RESAMPLER_SHA256 = "33a200a29b1076712d73b299c490abad2e3d6096e6f75a67fb8b4476af66ce73"
EXPECTED_FILTER_SHA256 = "fb33689327b6b81f630bf36cdf04b794ba66a022d3d1eb40b1b8a9dc2d081537"
PREPROCESSING_ID = "anphy_eeg_causal_fir_resample_bandpass_average_reference.v0"
SAMPLE_RATE_HZ = 250.0
WINDOW_SECONDS = 20.0
SAMPLE_COUNT = 5_000
CHANNEL_COUNT = 83
WINDOWS_SEC = tuple((float(start), float(start + 20)) for start in range(0, 300, 20))
SUPPORT_FLAGS = (
    "source_is_continuous",
    "timestamps_uniform",
    "gap_free",
    "bad_interval_free",
    "within_recording_bounds",
    "effective_filter_support_valid",
)

CONTROL_VERSION = "VECTOR-PILOT-001.independent_ar1_reference_control.v1"
CONTROL_MASTER_SEED_FIELDS = (CONTROL_VERSION, "20261012")
CONTROL_REPLICATES = 64
CONTROL_AR1_RHO = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def labels_hashes(labels: Sequence[str]) -> tuple[str, str]:
    exact = tuple(str(label) for label in labels)
    ordered = json.dumps(exact, ensure_ascii=False, separators=(",", ":"))
    unordered = json.dumps(sorted(exact), ensure_ascii=False, separators=(",", ":"))
    return (
        hashlib.sha256(ordered.encode("utf-8")).hexdigest(),
        hashlib.sha256(unordered.encode("utf-8")).hexdigest(),
    )


def legacy_order_hash(labels: Sequence[str]) -> str:
    payload = "\n".join(str(label).strip().casefold() for label in labels)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key)
    return value if isinstance(value, Mapping) else {}


def _object_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(vars(value))
    except TypeError:
        return {}


def independent_ar1_sources(seed: int, *, channel_count: int = CHANNEL_COUNT, sample_count: int = SAMPLE_COUNT) -> np.ndarray:
    """Generate independent stationary Gaussian AR(1) channel rows."""
    from scipy import signal

    if channel_count < 2 or sample_count < 2:
        raise ValueError("control_dimensions_must_be_at_least_two")
    rng = np.random.Generator(np.random.PCG64(seed))
    innovations = rng.standard_normal((channel_count, sample_count), dtype=np.float64)
    sources = np.empty_like(innovations)
    sources[:, 0] = innovations[:, 0]
    scale = math.sqrt(1.0 - CONTROL_AR1_RHO * CONTROL_AR1_RHO)
    tail, _ = signal.lfilter(
        [scale],
        [1.0, -CONTROL_AR1_RHO],
        innovations[:, 1:],
        axis=-1,
        zi=(CONTROL_AR1_RHO * innovations[:, 0])[:, None],
    )
    sources[:, 1:] = tail
    if not np.isfinite(sources).all():
        raise FloatingPointError("control_source_nonfinite")
    return sources


def control_seed(replicate_index: int) -> int:
    if not isinstance(replicate_index, int) or replicate_index < 0:
        raise ValueError("replicate_index_must_be_nonnegative_integer")
    serialized = "\0".join((*CONTROL_MASTER_SEED_FIELDS, str(replicate_index)))
    digest = hashlib.sha256(serialized.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def construct_absolute_pearson_weights(signals: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Construct exact-symmetric absolute Pearson weights from one window.

    The centered Gram matrix computes each ordinary Pearson coefficient. The
    upper-triangle coefficient for each undirected pair is copied to its
    matching lower-triangle cell as part of constructing the symmetric graph;
    this is not a post-hoc repair. Values are never clipped.
    """
    try:
        values = np.asarray(signals, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, {"status": "unsupported", "reason": "invalid_signal_matrix"}
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return None, {"status": "unsupported", "reason": "invalid_signal_matrix_shape"}
    if not np.isfinite(values).all():
        return None, {"status": "unsupported", "reason": "non_finite_sample"}

    centered = values - np.mean(values, axis=1, keepdims=True)
    gram = centered @ centered.T
    variances = np.diag(gram).copy()
    if not np.isfinite(gram).all() or not np.isfinite(variances).all():
        return None, {"status": "unsupported", "reason": "non_finite_centered_gram"}
    if np.any(variances <= 0.0):
        return None, {
            "status": "unsupported",
            "reason": "constant_channel_pearson_undefined",
            "constant_channel_indices": np.flatnonzero(variances <= 0.0).astype(int).tolist(),
        }

    denominator = np.sqrt(variances[:, None] * variances[None, :])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        correlations = gram / denominator
    if not np.isfinite(correlations).all():
        return None, {"status": "unsupported", "reason": "non_finite_pearson_correlation"}

    weights = np.zeros(correlations.shape, dtype=np.float64)
    upper = np.triu_indices(values.shape[0], k=1)
    pair_weights = np.abs(correlations[upper])
    weights[upper] = pair_weights
    weights[(upper[1], upper[0])] = pair_weights
    if not np.array_equal(weights, weights.T):
        return None, {"status": "unsupported", "reason": "constructed_edge_matrix_not_exactly_symmetric"}
    if not np.all(np.diag(weights) == 0.0):
        return None, {"status": "unsupported", "reason": "constructed_edge_matrix_diagonal_nonzero"}
    if not np.isfinite(weights).all():
        return None, {"status": "unsupported", "reason": "non_finite_edge_weight"}

    off_diagonal = weights[upper]
    return weights, {
        "status": "constructed",
        "reason": None,
        "edge_estimator_id": NIE_EDGE_ESTIMATOR_ID,
        "edge_rule_id": NIE_EDGE_RULE_ID,
        "edge_count_positive": int(np.count_nonzero(off_diagonal)),
        "possible_undirected_edges": int(len(off_diagonal)),
        "density_positive": float(np.count_nonzero(off_diagonal) / len(off_diagonal)),
        "weight_min_including_zeros": float(np.min(off_diagonal)),
        "weight_max": float(np.max(off_diagonal)),
        "matrix_exactly_symmetric": True,
        "diagonal_exactly_zero": True,
        "weights_finite": True,
        "weight_bounds_valid": bool(np.all((off_diagonal >= 0.0) & (off_diagonal <= 1.0))),
        "correlation_computation": "centered_gram_upper_triangle_pairwise_mirror_no_clipping",
    }


def _support_audit_complete(window: Any, provenance: Mapping[str, Any]) -> bool:
    if getattr(window, "status", None) != "ready" or getattr(window, "support", None) is None:
        return False
    support = window.support
    audit = _nested(provenance, "support_audit")
    try:
        support_matches = (
            float(support.sample_rate_hz) == SAMPLE_RATE_HZ
            and int(support.sample_count) == SAMPLE_COUNT
            and np.isclose(
                float(support.window_end_sec) - float(support.window_start_sec),
                WINDOW_SECONDS,
                rtol=0.0,
                atol=1e-9,
            )
            and all(getattr(support, flag) is True for flag in SUPPORT_FLAGS)
        )
    except (AttributeError, TypeError, ValueError):
        return False
    dependency = _nested(provenance, "source_dependency")
    return bool(
        support_matches
        and all(audit.get(flag) is True for flag in SUPPORT_FLAGS)
        and dependency.get("source_sample_start") is not None
        and dependency.get("source_sample_end_exclusive") is not None
    )


def _identity_complete(labels: Sequence[str], provenance: Mapping[str, Any]) -> bool:
    source = _nested(provenance, "source")
    config = _nested(provenance, "configuration")
    channels = _nested(provenance, "channels")
    preprocessing = _nested(provenance, "preprocessing")
    dependency = _nested(provenance, "source_dependency")
    order_hash, set_hash = labels_hashes(labels) if labels else (None, None)
    return bool(
        len(labels) == CHANNEL_COUNT
        and order_hash == EXPECTED_ORDER_SHA256
        and set_hash == EXPECTED_SET_SHA256
        and legacy_order_hash(labels) == EXPECTED_LEGACY_ORDER_SHA256
        and source.get("sha256_stable_after_read") is True
        and source.get("sha256_before_read") == source.get("sha256_after_read") == EXPECTED_SOURCE_SHA256
        and config.get("overlay_hash_stable") is True
        and config.get("overlay_sha256") == EXPECTED_OVERLAY_SHA256
        and channels.get("positions_hash_stable") is True
        and channels.get("positions_sha256") == EXPECTED_POSITIONS_SHA256
        and tuple(str(item) for item in channels.get("labels", ())) == tuple(str(item) for item in labels)
        and preprocessing.get("pipeline_id") == PREPROCESSING_ID
        and preprocessing.get("target_sfreq_hz") == SAMPLE_RATE_HZ
        and preprocessing.get("bandpass_hz") == [1.0, 45.0]
        and preprocessing.get("notch_hz") is None
        and preprocessing.get("reference") == "average_over_fixed_channel_set"
        and preprocessing.get("csd") is False
        and _nested(dependency, "resampler").get("coefficient_sha256") == EXPECTED_RESAMPLER_SHA256
        and _nested(dependency, "eeg_bandpass").get("coefficient_sha256") == EXPECTED_FILTER_SHA256
    )


def _time_support(
    window: Any,
    nominal: tuple[float, float],
    provenance: Mapping[str, Any],
    support_status: str,
) -> Any:
    from mndm.mnps.v3_2.contracts import TimeSupportRecord

    window_meta = _nested(provenance, "window")
    dependency = _nested(provenance, "source_dependency")
    source = _nested(provenance, "source")
    audit = _nested(provenance, "support_audit")
    start = window_meta.get("output_sample_start", int(round(nominal[0] * SAMPLE_RATE_HZ)))
    end = window_meta.get("output_sample_end_exclusive", int(round(nominal[1] * SAMPLE_RATE_HZ)))
    filter_spec = _nested(dependency, "eeg_bandpass")
    identity = _nested(provenance, "recording_identity")
    source_hash = source.get("sha256_after_read") or source.get("sha256_before_read")
    return TimeSupportRecord(
        support_status=support_status,
        source_id=f"{identity.get('dataset_id', 'unknown')}/{identity.get('subject_id', 'unknown')}",
        source_sha256=source_hash,
        segment_id=f"edf:{str(source_hash or '')[:16]}:segment0",
        sample_grid_id="ANPHY_EPCTL01_E1_G2_output_grid_250Hz",
        sample_rate_hz=SAMPLE_RATE_HZ,
        start_sample=int(start),
        end_sample_exclusive=int(end),
        sample_count=int(end) - int(start),
        nominal_window_seconds=nominal[1] - nominal[0],
        dependency_start_sample=(
            None if dependency.get("source_sample_start") is None else int(dependency["source_sample_start"])
        ),
        dependency_end_sample_exclusive=(
            None
            if dependency.get("source_sample_end_exclusive") is None
            else int(dependency["source_sample_end_exclusive"])
        ),
        gap_free=audit.get("gap_free"),
        coverage=1.0 if support_status == "known" else None,
        availability=(
            "all nominal output samples and source dependencies audited"
            if support_status == "known"
            else "unsupported: effective filter dependency outside recording"
            if support_status == "unsupported"
            else "source/time support audit incomplete"
        ),
        filter_warmup_samples=(None if filter_spec.get("tap_count") is None else int(filter_spec["tap_count"]) - 1),
        filter_delay_samples=None,
    )


def _family_record(
    *,
    family_id: str,
    estimator_id: str,
    schema_id: str,
    realization_id: str,
    units: str,
    kernel_result: Any | None,
    reason: str | None,
    window: Any,
    nominal: tuple[float, float],
    labels: Sequence[str],
    provenance: Mapping[str, Any],
    run_id: str,
    input_sha256: str | None,
    code_sha256: str,
    protocol_sha256: str,
    overlay_sha256: str,
    runtime: Mapping[str, str],
    method_parameters: tuple[tuple[str, str | int | float | bool | None], ...],
) -> Any:
    from mndm.mnps.v3_2.contracts import (
        MeasurementRecord,
        RealizationRecord,
        ReproducibilityRecord,
    )

    supported = _support_audit_complete(window, provenance)
    support_status = (
        "known"
        if supported
        else "unsupported"
        if getattr(window, "status", None) != "ready"
        else "unknown"
    )
    identity = _identity_complete(labels, provenance)
    order_hash, set_hash = labels_hashes(labels) if labels else (None, None)
    actual_id = realization_id if identity else None
    if kernel_result is None:
        computed = False
        valid = False
        value = None
        kernel_qc: tuple[str, ...] = ()
        status = "unsupported"
        final_reason = reason or "kernel_not_attempted"
    else:
        computed = bool(kernel_result.computed)
        valid = bool(kernel_result.valid)
        value = kernel_result.value
        kernel_qc = tuple(getattr(kernel_result, "qc", ()))
        status = "computed_exploratory" if valid else "unsupported"
        final_reason = None if valid else (getattr(kernel_result, "reason", None) or reason or "kernel_unsupported")

    qc = tuple(item for item in kernel_qc if item not in {"temporal_and_source_support_not_audited"})
    if not supported and getattr(window, "status", None) == "ready":
        qc = tuple(dict.fromkeys((*qc, "end_to_end_support_audit_incomplete")))
    elif getattr(window, "status", None) != "ready":
        qc = tuple(dict.fromkeys((*qc, f"window_unsupported:{final_reason}")))
    return MeasurementRecord(
        family_id=family_id,
        estimator_id=estimator_id,
        schema_id=schema_id,
        value=value,
        units=units,
        computed=computed,
        valid=valid,
        qualified=False,
        status=status,
        reason=final_reason,
        support=_time_support(window, nominal, provenance, support_status),
        realization=RealizationRecord(
            audit_status="complete" if identity else "incomplete" if labels else "unknown",
            kind="fixed_83_ANPHY_EPCTL01_E1_G2_surface",
            requested_id=realization_id,
            actual_id=actual_id,
            ordered_labels=tuple(str(label) for label in labels) if labels else None,
            order_sha256=order_hash,
            set_sha256=set_hash,
            count=len(labels) if labels else None,
            coverage=1.0 if supported else None,
            reference_id="average_over_fixed_channel_set" if labels else None,
            reference_set_sha256=set_hash if labels else None,
        ),
        qc=qc,
        reproducibility=ReproducibilityRecord(
            status="complete" if provenance else "partial",
            run_id=run_id,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            protocol_sha256=protocol_sha256,
            config_id="mnps_v3_1_anphy_epctl01_e1_g2_20260929",
            config_sha256=overlay_sha256,
            dependency_versions=tuple(sorted((str(key), str(value)) for key, value in runtime.items())),
        ),
        method_parameters=method_parameters,
        direction=("higher_more_ordinally_irregular" if family_id == "OI" else None),
        claim_status="no_biological_claim",
    )


def _correlation_summary(x: Sequence[float], y: Sequence[float]) -> tuple[float | None, str | None]:
    if len(x) != len(y) or len(x) < 2:
        return None, "fewer_than_two_pairwise_valid_windows"
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None, "non_finite_pairwise_value"
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None, "constant_metric_values"
    return float(np.corrcoef(left, right)[0, 1]), None


def pairwise_associations(window_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report descriptive Pearson coefficients on pairwise-valid windows."""
    pairs = (("ED", "OI"), ("ED", "NIE"), ("OI", "NIE"))
    result: dict[str, Any] = {
        "schema": "mndm.mnps_v3_2.vector_pilot_pairwise_associations.v1",
        "method": "pearson_product_moment",
        "inference": "none_descriptive_only",
        "overlapping_effective_source_dependencies": True,
        "independent_window_assumption": False,
        "qualified": False,
        "pairs": {},
    }
    pair_results: dict[str, Any] = {}
    for left_name, right_name in pairs:
        window_ids: list[str] = []
        left_values: list[float] = []
        right_values: list[float] = []
        for row in window_rows:
            family_records = _mapping(row.get("measurements"))
            left = _mapping(family_records.get(left_name))
            right = _mapping(family_records.get(right_name))
            left_record = _mapping(left.get("measurement_record"))
            right_record = _mapping(right.get("measurement_record"))
            a = left_record.get("value")
            b = right_record.get("value")
            if left_record.get("valid") is True and right_record.get("valid") is True and a is not None and b is not None:
                window_ids.append(str(row["window_id"]))
                left_values.append(float(a))
                right_values.append(float(b))
        coefficient, reason = _correlation_summary(left_values, right_values)
        pair_results[f"{left_name}_{right_name}"] = {
            "left_family": left_name,
            "right_family": right_name,
            "window_ids": window_ids,
            "n": len(window_ids),
            "pearson_r": coefficient,
            "reason": reason,
            "interpretation": "exploratory_single_recording_only",
        }
    result["pairs"] = pair_results
    return result


def _control_metric_values(signals: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
    from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency
    from mndm.mnps.v3_2.oi import compute_oi_matrix

    ed = compute_ed_raw_sensor_covariance_participation_ratio(signals, labels)
    oi = compute_oi_matrix(signals, labels)
    weights, edge_audit = construct_absolute_pearson_weights(signals)
    if weights is None:
        return {
            "ED": ed.value,
            "OI": oi.value,
            "NIE": None,
            "NIE_reason": edge_audit["reason"],
            "edge_audit": edge_audit,
        }
    order_hash, _ = labels_hashes(labels)
    nie = compute_nie_raw_weighted_global_efficiency(
        weights,
        node_labels=labels,
        node_order_sha256=order_hash,
        edge_estimator_id=NIE_EDGE_ESTIMATOR_ID,
        edge_rule_id=NIE_EDGE_RULE_ID,
    )
    return {
        "ED": ed.value,
        "OI": oi.value,
        "NIE": nie.value,
        "NIE_reason": nie.reason,
        "edge_audit": edge_audit,
    }


def run_synthetic_controls() -> dict[str, Any]:
    """Run 64 paired colored-source controls before and after exact average reference."""
    labels = tuple(f"synthetic-{index + 1:04d}" for index in range(CHANNEL_COUNT))
    rows: list[dict[str, Any]] = []
    for replicate_index in range(CONTROL_REPLICATES):
        seed = control_seed(replicate_index)
        sources = independent_ar1_sources(seed)
        referenced = sources - np.mean(sources, axis=0, keepdims=True)
        rows.append(
            {
                "replicate_index": replicate_index,
                "seed": seed,
                "source_sha256": sha256_array(sources),
                "unreferenced": _control_metric_values(sources, labels),
                "average_referenced": _control_metric_values(referenced, labels),
                "qualified": False,
            }
        )

    family_summaries: dict[str, Any] = {}
    for family in ("ED", "OI", "NIE"):
        unreferenced = np.asarray([row["unreferenced"][family] for row in rows if row["unreferenced"][family] is not None], dtype=np.float64)
        referenced = np.asarray([row["average_referenced"][family] for row in rows if row["average_referenced"][family] is not None], dtype=np.float64)
        paired = np.asarray(
            [row["average_referenced"][family] - row["unreferenced"][family] for row in rows if row["average_referenced"][family] is not None and row["unreferenced"][family] is not None],
            dtype=np.float64,
        )
        def distribution(values: np.ndarray) -> dict[str, float | int | None]:
            if not values.size:
                return {"n": 0, "mean": None, "min": None, "q25": None, "median": None, "q75": None, "max": None}
            return {
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }
        family_summaries[family] = {
            "unreferenced": distribution(unreferenced),
            "average_referenced": distribution(referenced),
            "paired_reference_minus_unreferenced": distribution(paired),
        }
    return {
        "schema": "mndm.mnps_v3_2.vector_pilot_independent_ar1_control.v1",
        "control_version": CONTROL_VERSION,
        "source_model": "independent_stationary_Gaussian_AR1",
        "rho": CONTROL_AR1_RHO,
        "initialization": "x[0]=epsilon[0]; x[t]=rho*x[t-1]+sqrt(1-rho^2)*epsilon[t]",
        "replicates": CONTROL_REPLICATES,
        "channel_count": CHANNEL_COUNT,
        "sample_count": SAMPLE_COUNT,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "seed_fields": list(CONTROL_MASTER_SEED_FIELDS),
        "seed_serialization": "UTF8_NUL_separated_fields_then_SHA256_first_8_bytes_big_endian",
        "conditions": ["unreferenced", "instantaneous_average_reference_over_same_83_channels"],
        "preprocessing": "synthetic_target_grid_only_no_EDF_resampling_or_filtering",
        "family_summaries": family_summaries,
        "replicate_rows": rows,
        "interpretation": "finite_sample_and_reference_mechanism_control_only_not_EEG_model_or_empirical_null",
        "baseline_subtraction_from_EEG": False,
        "qualified": False,
    }


def _validate_grid(grid: Any, overlay_sha256: str) -> tuple[tuple[str, ...], dict[str, Any]]:
    provenance = _mapping(grid.provenance)
    if grid.status != "completed" or len(grid.windows) != len(WINDOWS_SEC):
        raise RuntimeError(f"audited_grid_incomplete:{grid.status}:{grid.reason}:{len(grid.windows)}")
    if provenance.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("frozen_source_sha256_mismatch")
    if provenance.get("positions_sha256") != EXPECTED_POSITIONS_SHA256:
        raise RuntimeError("frozen_positions_sha256_mismatch")
    if provenance.get("overlay_sha256") != EXPECTED_OVERLAY_SHA256 or overlay_sha256 != EXPECTED_OVERLAY_SHA256:
        raise RuntimeError("frozen_overlay_sha256_mismatch")

    first = grid.windows[0]
    first_provenance = _mapping(first.provenance)
    first_dependency = _nested(first_provenance, "source_dependency")
    if first.status != "unsupported" or first.reason != "insufficient_effective_filter_edge_support":
        raise RuntimeError("first_window_unsupported_contract_mismatch")
    if first_dependency.get("source_sample_start") != -3376 or first_dependency.get("source_sample_end_exclusive") != 19997:
        raise RuntimeError("first_window_effective_dependency_mismatch")

    labels = tuple(str(label) for label in _nested(grid.windows[1].provenance, "channels").get("labels", ()))
    order_hash, set_hash = labels_hashes(labels)
    if len(labels) != CHANNEL_COUNT or order_hash != EXPECTED_ORDER_SHA256 or set_hash != EXPECTED_SET_SHA256:
        raise RuntimeError("frozen_ordered_83_channel_identity_mismatch")
    if legacy_order_hash(labels) != EXPECTED_LEGACY_ORDER_SHA256:
        raise RuntimeError("frozen_legacy_order_identity_mismatch")

    for index, window in enumerate(grid.windows):
        window_provenance = _mapping(window.provenance)
        dependency = _nested(window_provenance, "source_dependency")
        if index == 0:
            continue
        if window.status != "ready" or not _support_audit_complete(window, window_provenance):
            raise RuntimeError(f"supported_window_audit_not_reproduced:{index}:{window.status}:{window.reason}")
        if not isinstance(window.signals, np.ndarray) or window.signals.shape != (CHANNEL_COUNT, SAMPLE_COUNT):
            raise RuntimeError(f"supported_window_matrix_shape_mismatch:{index}")
        if np.asarray(window.signals).dtype != np.float64 or not np.isfinite(window.signals).all():
            raise RuntimeError(f"supported_window_matrix_dtype_or_finiteness_mismatch:{index}")
        if _nested(dependency, "resampler").get("coefficient_sha256") != EXPECTED_RESAMPLER_SHA256:
            raise RuntimeError(f"frozen_resampler_coefficient_identity_mismatch:{index}")
        if _nested(dependency, "eeg_bandpass").get("coefficient_sha256") != EXPECTED_FILTER_SHA256:
            raise RuntimeError(f"frozen_filter_coefficient_identity_mismatch:{index}")
    return labels, dict(provenance)


def _make_row(
    *,
    index: int,
    window: Any,
    nominal: tuple[float, float],
    labels: Sequence[str],
    run_id: str,
    code_sha256: str,
    protocol_sha256: str,
    overlay_sha256: str,
    runtime: Mapping[str, str],
    edge_dir: Path,
) -> dict[str, Any]:
    from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
    from mndm.mnps.v3_2.nie import NIE_ESTIMATOR_ID, NIE_SCHEMA_ID, compute_nie_raw_weighted_global_efficiency
    from mndm.mnps.v3_2.oi import OI_MATRIX_ESTIMATOR_ID, OI_MATRIX_SCHEMA_ID, compute_oi_matrix

    provenance = _mapping(window.provenance)
    input_matrix = None
    input_hash = None
    matrix_error: str | None = None
    if window.status == "ready" and window.signals is not None:
        input_matrix = np.asarray(window.signals, dtype=np.float64)
        if input_matrix.shape != (CHANNEL_COUNT, SAMPLE_COUNT):
            matrix_error = "audited_matrix_shape_or_label_count_mismatch"
            input_matrix = None
        else:
            input_hash = sha256_array(input_matrix)
    elif window.status == "ready":
        matrix_error = "ready_window_missing_matrix"
    else:
        matrix_error = str(window.reason or "upstream_window_not_ready")

    ed_kernel = compute_ed_raw_sensor_covariance_participation_ratio(input_matrix, labels) if input_matrix is not None else None
    oi_kernel = compute_oi_matrix(input_matrix, labels) if input_matrix is not None else None
    weights: np.ndarray | None = None
    edge_audit: dict[str, Any] = {"status": "not_attempted", "reason": matrix_error}
    nie_kernel = None
    edge_relative_path: str | None = None
    edge_sha256: str | None = None
    if input_matrix is not None:
        weights, edge_audit = construct_absolute_pearson_weights(input_matrix)
        if weights is not None:
            node_order_hash, _ = labels_hashes(labels)
            nie_kernel = compute_nie_raw_weighted_global_efficiency(
                weights,
                node_labels=labels,
                node_order_sha256=node_order_hash,
                edge_estimator_id=NIE_EDGE_ESTIMATOR_ID,
                edge_rule_id=NIE_EDGE_RULE_ID,
            )
            edge_audit = {
                **edge_audit,
                "matrix_kernel_reason": nie_kernel.reason,
                "matrix_kernel_valid": nie_kernel.valid,
            }
            edge_path = edge_dir / f"window_{index:02d}_{int(nominal[0]):03d}_{int(nominal[1]):03d}s.npy"
            with edge_path.open("wb") as stream:
                np.save(stream, weights, allow_pickle=False)
            edge_relative_path = str(edge_path.relative_to(edge_dir.parent)).replace("\\", "/")
            edge_sha256 = sha256_file(edge_path)

    family_inputs = (
        ("ED", "ED.sensor_covariance_participation_ratio_raw_matrix.v1", "nmdm.mnps_v3_2.ed_raw_matrix_result.v1", ED_REALIZATION_ID, "sensor_directions", ed_kernel,
         "centered_sample_covariance_ddof1_no_shrinkage_no_channel_normalization"),
        ("OI", OI_MATRIX_ESTIMATOR_ID, OI_MATRIX_SCHEMA_ID, OI_REALIZATION_ID, "normalized_entropy", oi_kernel,
         "order5_delay1_stable_mergesort_median_all_83_channels"),
        ("NIE", NIE_ESTIMATOR_ID, NIE_SCHEMA_ID, NIE_REALIZATION_ID, "global_efficiency", nie_kernel,
         "abs_pearson_no_threshold_inverse_positive_weight_edge_length"),
    )
    measurements: dict[str, Any] = {}
    for family, estimator, schema, realization, units, kernel, method in family_inputs:
        family_reason = matrix_error
        if family == "NIE" and input_matrix is not None and weights is None:
            family_reason = str(edge_audit.get("reason") or "edge_construction_unsupported")
        if kernel is not None and not kernel.valid:
            family_reason = kernel.reason or family_reason
        if kernel is None:
            kernel_json = None
            kernel_estimator = estimator
        else:
            kernel_json = kernel.to_dict()
            kernel_estimator = kernel.estimator_id
        record = _family_record(
            family_id=family,
            estimator_id=kernel_estimator,
            schema_id=schema,
            realization_id=realization,
            units=units,
            kernel_result=kernel,
            reason=family_reason,
            window=window,
            nominal=nominal,
            labels=labels,
            provenance=provenance,
            run_id=run_id,
            input_sha256=input_hash,
            code_sha256=code_sha256,
            protocol_sha256=protocol_sha256,
            overlay_sha256=overlay_sha256,
            runtime=runtime,
            method_parameters=(("definition", method),),
        )
        measurements[family] = {
            "realization_id": realization,
            "matrix_kernel_result": kernel_json,
            "measurement_record": record.to_dict(),
            "qualified": False,
        }
        if family == "NIE":
            measurements[family]["edge_audit"] = edge_audit
            measurements[family]["edge_matrix_path"] = edge_relative_path
            measurements[family]["edge_matrix_sha256"] = edge_sha256

    output_sample_start = int(round(nominal[0] * SAMPLE_RATE_HZ))
    output_sample_end = int(round(nominal[1] * SAMPLE_RATE_HZ))
    return {
        "schema": RESULT_SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "recording_id": "ANPHY/EPCTL01",
        "window_id": f"ANPHY_EPCTL01_{int(nominal[0]):03d}_{int(nominal[1]):03d}s",
        "window_index": index,
        "nominal_window_sec_half_open": list(nominal),
        "output_grid": {
            "sample_grid_id": "ANPHY_EPCTL01_E1_G2_output_grid_250Hz",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "start_sample": output_sample_start,
            "end_sample_exclusive": output_sample_end,
            "sample_count": output_sample_end - output_sample_start,
        },
        "window_audit_status": str(window.status),
        "window_audit_reason": window.reason,
        "input_matrix_sha256_little_endian_float64": input_hash,
        "source_dependency": provenance.get("source_dependency"),
        "support_audit": provenance.get("support_audit"),
        "window_audit_provenance": provenance,
        "realization_identity": {
            "ordered_labels": list(labels),
            "channel_count": len(labels),
            "order_sha256": labels_hashes(labels)[0],
            "set_sha256": labels_hashes(labels)[1],
            "reference_id": "average_over_fixed_channel_set",
            "reference_set_sha256": labels_hashes(labels)[1],
        },
        "measurements": measurements,
        "qualified": False,
        "claim_status": "no_biological_claim",
    }


def _write_long_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import csv

    fieldnames = (
        "recording_id", "window_id", "window_index", "nominal_start_sec", "nominal_end_sec",
        "family_id", "estimator_id", "realization_id", "value", "units", "computed", "valid",
        "status", "reason", "qualified", "support_status", "dependency_start_sample",
        "dependency_end_sample_exclusive", "source_sha256", "input_matrix_sha256",
        "channel_count", "ordered_labels_json", "channel_order_sha256", "channel_set_sha256",
        "reference_id", "reference_set_sha256", "edge_matrix_sha256", "qc_json",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            measurements = _mapping(row.get("measurements"))
            dependency = _mapping(row.get("source_dependency"))
            provenance = _mapping(row.get("window_audit_provenance"))
            source = _nested(provenance, "source")
            identity = _mapping(row.get("realization_identity"))
            for family in ("ED", "OI", "NIE"):
                family_row = _mapping(measurements.get(family))
                record = _mapping(family_row.get("measurement_record"))
                support = _mapping(record.get("support"))
                realization = _mapping(record.get("realization"))
                writer.writerow({
                    "recording_id": row.get("recording_id"),
                    "window_id": row.get("window_id"),
                    "window_index": row.get("window_index"),
                    "nominal_start_sec": row.get("nominal_window_sec_half_open", [None, None])[0],
                    "nominal_end_sec": row.get("nominal_window_sec_half_open", [None, None])[1],
                    "family_id": family,
                    "estimator_id": record.get("estimator_id"),
                    "realization_id": family_row.get("realization_id"),
                    "value": record.get("value"),
                    "units": record.get("units"),
                    "computed": record.get("computed"),
                    "valid": record.get("valid"),
                    "status": record.get("status"),
                    "reason": record.get("reason"),
                    "qualified": record.get("qualified"),
                    "support_status": support.get("support_status"),
                    "dependency_start_sample": dependency.get("source_sample_start"),
                    "dependency_end_sample_exclusive": dependency.get("source_sample_end_exclusive"),
                    "source_sha256": source.get("sha256_after_read") or source.get("sha256_before_read"),
                    "input_matrix_sha256": row.get("input_matrix_sha256_little_endian_float64"),
                    "channel_count": realization.get("count", identity.get("channel_count")),
                    "ordered_labels_json": json.dumps(realization.get("ordered_labels", identity.get("ordered_labels")), ensure_ascii=False, separators=(",", ":")),
                    "channel_order_sha256": realization.get("order_sha256", identity.get("order_sha256")),
                    "channel_set_sha256": realization.get("set_sha256", identity.get("set_sha256")),
                    "reference_id": realization.get("reference_id", identity.get("reference_id")),
                    "reference_set_sha256": realization.get("reference_set_sha256", identity.get("reference_set_sha256")),
                    "edge_matrix_sha256": family_row.get("edge_matrix_sha256"),
                    "qc_json": json.dumps(record.get("qc", []), ensure_ascii=False, separators=(",", ":")),
                })


def _control_code_summary(controls: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in controls.items() if key != "replicate_rows"}


def run(
    *,
    overlay_path: Path = DEFAULT_OVERLAY,
    output_root: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Run one audited 15-window grid, plus fixed synthetic controls."""
    overlay_path = overlay_path.resolve()
    if not overlay_path.is_file():
        raise FileNotFoundError(f"frozen E1-G2 overlay not found: {overlay_path}")
    overlay_sha256 = sha256_file(overlay_path)
    if overlay_sha256 != EXPECTED_OVERLAY_SHA256:
        raise RuntimeError("frozen_overlay_sha256_mismatch")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_sl005")
    run_root = ((output_root or DEFAULT_OUTPUT_ROOT) / run_id).resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite VECTOR-PILOT-001 output root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    edge_dir = run_root / "edge_matrices"
    edge_dir.mkdir()
    numba_cache = run_root / ".numba_cache"
    numba_cache.mkdir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    # The tracked byte-for-byte overlay preserves the previously audited config
    # hash while removing the execution dependency on ignored project files.
    import yaml
    from mndm.mnps.v3_1 import load_audited_edf_eeg_grid

    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    config = overlay["eeg_grid"]
    source_path = Path(str(config["source_edf"]))
    positions_path = Path(str(config["positions_file"]))
    if not source_path.is_file() or not positions_path.is_file():
        raise FileNotFoundError("frozen G2 EDF and positions file must both be accessible")
    grid = load_audited_edf_eeg_grid(source_path, positions_path, overlay_path)
    labels, grid_provenance = _validate_grid(grid, overlay_sha256)

    runtime = _mapping(grid_provenance.get("runtime"))
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    code_paths = {
        "vector_pilot_runner": RUNNER_PATH,
        "ed_kernel": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/ed.py",
        "oi_kernel": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/oi.py",
        "nie_kernel": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/nie.py",
        "contracts": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/contracts.py",
        "eeg_window_loader": REPO_ROOT / "mndm/src/mndm/mnps/v3_1/eeg_windows.py",
        "preprocess": REPO_ROOT / "mndm/src/mndm/preprocess.py",
        "tracked_overlay": overlay_path,
    }
    code_components = {name: sha256_file(path) for name, path in code_paths.items()}
    code_sha256 = sha256_json(code_components)

    rows = [
        _make_row(
            index=index,
            window=window,
            nominal=nominal,
            labels=labels,
            run_id=run_id,
            code_sha256=code_sha256,
            protocol_sha256=protocol_sha256,
            overlay_sha256=overlay_sha256,
            runtime=runtime,
            edge_dir=edge_dir,
        )
        for index, (window, nominal) in enumerate(zip(grid.windows, WINDOWS_SEC, strict=True))
    ]
    for row in rows:
        target = run_root / f"window_{row['window_index']:02d}.json"
        target.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    long_path = run_root / "measurements_long.csv"
    _write_long_csv(rows, long_path)
    associations = pairwise_associations(rows)
    (run_root / "exploratory_pairwise_associations.json").write_text(
        json.dumps(associations, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    controls = run_synthetic_controls()
    (run_root / "synthetic_controls.json").write_text(
        json.dumps(controls, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    family_counts: dict[str, Any] = {}
    for family in ("ED", "OI", "NIE"):
        family_records = [
            _mapping(_mapping(row.get("measurements")).get(family)).get("measurement_record", {})
            for row in rows
        ]
        family_counts[family] = {
            "nominal_rows": len(family_records),
            "computed_valid_rows": sum(_mapping(record).get("valid") is True for record in family_records),
            "unsupported_or_invalid_rows": sum(_mapping(record).get("valid") is not True for record in family_records),
            "all_qualified_false": all(_mapping(record).get("qualified") is False for record in family_records),
        }

    manifest = {
        "schema": MANIFEST_SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256,
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": code_components["vector_pilot_runner"],
        "code_component_sha256": code_components,
        "code_sha256_composite": code_sha256,
        "source_path": str(source_path),
        "source_sha256": grid_provenance.get("source_sha256"),
        "positions_path": str(positions_path),
        "positions_sha256": grid_provenance.get("positions_sha256"),
        "overlay_path": str(overlay_path),
        "overlay_sha256": overlay_sha256,
        "source_identity": grid_provenance.get("recording_identity"),
        "ordered_channel_labels": list(labels),
        "channel_count": len(labels),
        "channel_order_sha256": labels_hashes(labels)[0],
        "channel_set_sha256": labels_hashes(labels)[1],
        "legacy_order_sha256_casefolded_newline": legacy_order_hash(labels),
        "reference_id": "average_over_fixed_channel_set",
        "reference_set_sha256": labels_hashes(labels)[1],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "sample_count_per_channel": SAMPLE_COUNT,
        "preprocessing_id": PREPROCESSING_ID,
        "preprocessing_hashes": {
            "resampler_coefficient_sha256": EXPECTED_RESAMPLER_SHA256,
            "eeg_bandpass_coefficient_sha256": EXPECTED_FILTER_SHA256,
        },
        "window_count": len(rows),
        "supported_common_candidate_windows": sum(row["window_audit_status"] == "ready" for row in rows),
        "unsupported_windows": [
            {"window_id": row["window_id"], "reason": row["window_audit_reason"], "source_dependency": row["source_dependency"]}
            for row in rows if row["window_audit_status"] != "ready"
        ],
        "family_counts": family_counts,
        "window_json_sha256": {
            f"window_{row['window_index']:02d}.json": sha256_file(run_root / f"window_{row['window_index']:02d}.json")
            for row in rows
        },
        "long_csv_sha256": sha256_file(long_path),
        "pairwise_associations_sha256": sha256_file(run_root / "exploratory_pairwise_associations.json"),
        "synthetic_controls_sha256": sha256_file(run_root / "synthetic_controls.json"),
        "edge_matrix_sha256": {
            str(_mapping(_mapping(row.get("measurements")).get("NIE")).get("edge_matrix_path")): _mapping(_mapping(row.get("measurements")).get("NIE")).get("edge_matrix_sha256")
            for row in rows if _mapping(_mapping(row.get("measurements")).get("NIE")).get("edge_matrix_path") is not None
        },
        "synthetic_control_contract": _control_code_summary(controls),
        "runtime": runtime,
        "all_empirical_results_qualified_false": all(
            _mapping(_mapping(_mapping(row.get("measurements")).get(family)).get("measurement_record")).get("qualified") is False
            for row in rows for family in ("ED", "OI", "NIE")
        ),
        "pipeline_export_integration": "not_performed",
        "jacobian_or_manifold_inference": "not_performed",
        "claim_status": "no_biological_claim",
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    arguments = parser.parse_args()
    print(run(overlay_path=arguments.overlay, output_root=arguments.output_root, run_id=arguments.run_id))

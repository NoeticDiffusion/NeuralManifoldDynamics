"""Outcome-blind twelve-participant ANPHY-Sleep feasibility and redundancy pilot.

The runner applies the unchanged v3.2 ED/OI/NIE matrix kernels to one common
audited E1-G2 grid, retains every nominal window, and adds the diagnostics and
paired time-index control frozen in the SL-006 protocol. It does not qualify
measurements or make biological claims.
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

from mndm.mnps.v3_2.experiments.vector_pilot_001 import (  # noqa: E402
    CHANNEL_COUNT,
    EXPECTED_ORDER_SHA256,
    EXPECTED_POSITIONS_SHA256,
    EXPECTED_SET_SHA256,
    NIE_EDGE_ESTIMATOR_ID,
    NIE_EDGE_RULE_ID,
    SAMPLE_COUNT,
    SAMPLE_RATE_HZ,
    WINDOW_SECONDS,
    WINDOWS_SEC,
    _mapping,
    _nested,
    _support_audit_complete,
    construct_absolute_pearson_weights,
    labels_hashes,
    sha256_array,
    sha256_file,
    sha256_json,
)


EXPERIMENT_ID = "nmdm.mnps_v3_2.vector_feasibility_002.v1"
RESULT_SCHEMA_ID = "mndm.mnps_v3_2.vector_feasibility_002_window.v1"
MANIFEST_SCHEMA_ID = "mndm.mnps_v3_2.vector_feasibility_002_run_manifest.v1"
SELECTION_PATH = REPO_ROOT / "mndm" / "config" / "mnps_v3_2_vector_feasibility_002_selection_20261013.json"
BASE_OVERLAY_PATH = REPO_ROOT / "mndm" / "config" / "mnps_v3_2_anphy_epctl01_e1_g2_grid_20260929.yaml"
PROTOCOL_PATH = REPO_ROOT / "docs" / "api" / "mnps_v3_2_vector_feasibility_002_protocol.rst"
DEFAULT_SOURCE_ROOT = Path("K:/ExternalReceivedDatasets/ANPHY")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "VECTOR-FEASIBILITY-002"
EXPECTED_SELECTION_SHA256 = "c43b5bdc08e211d708c4b283b1ce5ae8a85f30ea4c50586e0884a4409545e4f9"
EXPECTED_PROTOCOL_SHA256 = "422976eeaddcee4349aa8ee01d6badf5a5707470394a1045578eb4477521ce0c"
EXPECTED_BASE_OVERLAY_SHA256 = "3f1d3135d45ad43648ef1f1c24b0732970ad1492209a414804c35c6eed240824"
EXPECTED_SELECTED_IDS = (
    "EPCTL08", "EPCTL12", "EPCTL15", "EPCTL17", "EPCTL20", "EPCTL19",
    "EPCTL23", "EPCTL22", "EPCTL13", "EPCTL24", "EPCTL03", "EPCTL09",
)

ED_REALIZATION_ID = "ED.anphy_sleep_fixed83_E1_G2_avgref.v1"
OI_REALIZATION_ID = "OI.anphy_sleep_fixed83_PE5_delay1_median.v1"
NIE_REALIZATION_ID = "NIE.anphy_sleep_fixed83_avgref_abs_pearson_no_threshold.v1"
PREPROCESSING_ID = "anphy_eeg_causal_fir_resample_bandpass_average_reference.v0"
CONTROL_ID = "VECTOR-FEASIBILITY-002.shared_time_index_permutation.v1"
INVARIANCE_RTOL = 1e-10
INVARIANCE_ATOL = 1e-12
PSD_BANDS_HZ = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "gamma_30_45": (30.0, 45.0),
}
PARTICIPANT_SUMMARY_METRICS = (
    "ED", "OI", "NIE", "ED_R", "mean_absolute_correlation",
    "mean_squared_correlation", "nie_graph_path_increment",
)


def _selection_rows(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = selection.get("source_roster_metadata_only", ())
    return [row for row in rows if isinstance(row, Mapping)]


def _selected_ids(selection: Mapping[str, Any]) -> tuple[str, ...]:
    random_selection = _mapping(selection.get("random_selection"))
    return tuple(str(value) for value in random_selection.get("selected_subject_ids_in_draw_order", ()))


def validate_frozen_preregistration(
    selection: Mapping[str, Any],
    *,
    selection_sha256: str,
    protocol_sha256: str,
    base_overlay_sha256: str,
) -> None:
    """Refuse to run if the frozen draw, protocol, or base method changed."""
    if selection_sha256 != EXPECTED_SELECTION_SHA256:
        raise ValueError("frozen_selection_manifest_sha256_mismatch")
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frozen_protocol_sha256_mismatch")
    if base_overlay_sha256 != EXPECTED_BASE_OVERLAY_SHA256:
        raise ValueError("frozen_base_overlay_sha256_mismatch")
    if selection.get("schema") != "mndm.mnps_v3_2.vector_feasibility_002_selection.v1":
        raise ValueError("selection_manifest_schema_mismatch")
    if _selected_ids(selection) != EXPECTED_SELECTED_IDS:
        raise ValueError("frozen_selected_subject_draw_mismatch")
    if _mapping(selection.get("random_selection")).get("seed") != 20261013:
        raise ValueError("frozen_selection_seed_mismatch")


def loader_identity_checks(
    window_provenances: Sequence[Mapping[str, Any]],
    *,
    payload_sha256_before_loader: str | None,
    overlay_sha256: str,
    positions_sha256: str,
) -> dict[str, bool]:
    """Check the loader's per-window source/config/channel provenance records."""
    sources = [_mapping(item.get("source")) for item in window_provenances]
    configurations = [_mapping(item.get("configuration")) for item in window_provenances]
    channels = [_mapping(item.get("channels")) for item in window_provenances]
    return {
        "loader_source_sha256_matches_preload": bool(sources) and all(
            item.get("sha256_before_read") == payload_sha256_before_loader for item in sources
        ),
        "loader_source_hash_stable": bool(sources) and all(
            item.get("sha256_stable_after_read") is True
            and item.get("sha256_before_read") == item.get("sha256_after_read")
            for item in sources
        ),
        "overlay_hash_stable": bool(configurations) and all(
            item.get("overlay_hash_stable") is True and item.get("overlay_sha256") == overlay_sha256
            for item in configurations
        ),
        "positions_hash_stable": bool(channels) and all(
            item.get("positions_hash_stable") is True
            and item.get("positions_sha256") == positions_sha256
            and item.get("positions_sha256_after_read") == positions_sha256
            for item in channels
        ),
    }


def _source_header_identity(path: Path) -> dict[str, Any]:
    """Read only fixed EDF header bytes; never scans a signal data record."""
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError("edf_fixed_header_short")
        try:
            header_bytes = int(fixed[184:192].decode("ascii").strip())
            n_records = int(fixed[236:244].decode("ascii").strip())
            record_duration = float(fixed[244:252].decode("ascii").strip())
            n_signals = int(fixed[252:256].decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("edf_fixed_header_invalid") from exc
        rest = stream.read(header_bytes - 256)
    if header_bytes < 256 or len(rest) != n_signals * 256:
        raise ValueError("edf_signal_header_length_invalid")
    header = fixed + rest
    labels_raw = [rest[index * 16 : (index + 1) * 16].decode("ascii", "replace").strip() for index in range(n_signals)]
    samples_offset = n_signals * 216
    samples_per_record = [
        int(rest[samples_offset + index * 8 : samples_offset + (index + 1) * 8].decode("ascii").strip())
        for index in range(n_signals)
    ]
    dimension_offset = n_signals * 96
    physical_dimensions = [
        rest[dimension_offset + index * 8 : dimension_offset + (index + 1) * 8].decode("ascii", "replace").strip()
        for index in range(n_signals)
    ]
    return {
        "header_bytes": header_bytes,
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "edf_reserved_format": fixed[192:236].decode("ascii", "replace").strip(),
        "recording_date": fixed[168:176].decode("ascii", "replace").strip(),
        "recording_time": fixed[176:184].decode("ascii", "replace").strip(),
        "record_count": n_records,
        "record_duration_sec": record_duration,
        "signal_count": n_signals,
        "labels": labels_raw,
        "samples_per_record": samples_per_record,
        "physical_dimensions": physical_dimensions,
    }


def validate_frozen_source(path: Path, subject_id: str, selection: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the selected local source has not changed since the metadata freeze."""
    frozen = next((row for row in _selection_rows(selection) if row.get("subject_id") == subject_id), None)
    if frozen is None:
        return {"valid": False, "reason": "subject_not_in_frozen_roster"}
    if frozen.get("local_edf_status") != "present":
        return {"valid": False, "reason": "frozen_source_not_present"}
    stat = path.stat() if path.is_file() else None
    if stat is None:
        return {"valid": False, "reason": "selected_edf_missing"}
    observed = _source_header_identity(path)
    payload_sha256 = sha256_file(path)
    checks = {
        "relative_path_matches": path.name == frozen.get("file_name"),
        "size_matches": stat.st_size == frozen.get("file_size_bytes"),
        "mtime_matches": stat.st_mtime_ns == frozen.get("mtime_ns_at_freeze"),
        "header_sha256_matches": observed["header_sha256"] == frozen.get("edf_header_sha256"),
        "edf_plus_format_matches": observed["edf_reserved_format"] == frozen.get("edf_header_reserved_format"),
    }
    return {
        "valid": all(checks.values()),
        "reason": None if all(checks.values()) else "selected_source_identity_changed_since_freeze",
        "checks": checks,
        "observed_header": {key: value for key, value in observed.items() if key != "labels"},
        "observed_size_bytes": stat.st_size,
        "observed_mtime_ns": stat.st_mtime_ns,
        "payload_sha256_before_loader": payload_sha256,
    }


def _pearson_components(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(signals, dtype=np.float64)
    centered = values - np.mean(values, axis=1, keepdims=True)
    gram = centered @ centered.T
    variances = np.diag(gram).copy()
    if not np.isfinite(gram).all() or np.any(variances <= 0.0):
        raise ValueError("pearson_requires_finite_nonconstant_channels")
    correlations = gram / np.sqrt(variances[:, None] * variances[None, :])
    covariance = gram / (values.shape[1] - 1)
    return covariance, correlations, variances / (values.shape[1] - 1)


def covariance_connectivity_diagnostics(
    signals: np.ndarray,
    *,
    raw_ed_value: float | None,
    nie_value: float | None,
) -> dict[str, Any]:
    """Compute SL-006 covariance and correlation diagnostics for one window."""
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        return {"status": "unsupported", "reason": "invalid_signal_matrix", "values": {}}
    try:
        covariance, correlations, channel_variances = _pearson_components(values)
    except (ValueError, FloatingPointError) as exc:
        return {"status": "unsupported", "reason": str(exc), "values": {}}

    upper = np.triu_indices(values.shape[0], k=1)
    pairwise = correlations[upper]
    denominator = float(np.sum(correlations * correlations))
    ed_r = float(np.trace(correlations) ** 2 / denominator) if denominator > 0.0 else None
    variance_sum = float(np.sum(channel_variances))
    corr_weighted = float(np.sum(channel_variances[:, None] * channel_variances[None, :] * correlations * correlations))
    # The double sum includes both off-diagonal orientations, so this is exactly
    # sum(sigma_i^4) + 2*sum_{i<j}(sigma_i^2*sigma_j^2*r_ij^2).
    formula_ed = float(variance_sum * variance_sum / corr_weighted) if corr_weighted > 0.0 else None
    return {
        "status": "computed",
        "reason": None,
        "values": {
            "ED_R": ed_r,
            "mean_absolute_correlation": float(np.mean(np.abs(pairwise))),
            "mean_squared_correlation": float(np.mean(pairwise * pairwise)),
            "raw_covariance_ED_formula": formula_ed,
            "raw_covariance_ED_identity_absolute_residual": (
                None if formula_ed is None or raw_ed_value is None else float(abs(formula_ed - raw_ed_value))
            ),
            "nie_graph_path_increment": (
                None if nie_value is None else float(nie_value - np.mean(np.abs(pairwise)))
            ),
            "mean_channel_variance_v2": float(np.mean(channel_variances)),
            "mean_pairwise_correlation": float(np.mean(pairwise)),
            "max_abs_correlation": float(np.max(np.abs(pairwise))),
            "covariance_matrix_trace_v2": float(np.trace(covariance)),
        },
    }


def amplitude_and_spectral_diagnostics(signals: np.ndarray) -> dict[str, Any]:
    """Summarize referenced amplitude and fixed Welch bands for one window."""
    from scipy.signal import welch

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape != (CHANNEL_COUNT, SAMPLE_COUNT) or not np.isfinite(values).all():
        return {"status": "unsupported", "reason": "invalid_signal_matrix", "amplitude": {}, "spectral": {}}
    values_uv = values * 1e6
    channel_sd_uv = np.std(values_uv, axis=1, ddof=1)
    channel_mean_uv = np.mean(values_uv, axis=1)
    residual = np.mean(values, axis=0)
    frequencies, density = welch(
        values_uv,
        fs=SAMPLE_RATE_HZ,
        window="hann",
        nperseg=1000,
        noverlap=500,
        nfft=1000,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        average="mean",
    )
    bin_width = float(frequencies[1] - frequencies[0])
    bands: dict[str, Any] = {}
    for band_id, (low, high) in PSD_BANDS_HZ.items():
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            bands[band_id] = {"power_mean_uv2": None, "power_median_uv2": None, "frequency_bin_count": 0}
            continue
        power_by_channel = np.sum(density[:, mask], axis=1) * bin_width
        bands[band_id] = {
            "power_mean_uv2": float(np.mean(power_by_channel)),
            "power_median_uv2": float(np.median(power_by_channel)),
            "frequency_bin_count": int(np.count_nonzero(mask)),
        }
    return {
        "status": "computed",
        "reason": None,
        "amplitude": {
            "channel_sd_median_uv": float(np.median(channel_sd_uv)),
            "channel_sd_min_uv": float(np.min(channel_sd_uv)),
            "channel_sd_max_uv": float(np.max(channel_sd_uv)),
            "global_rms_uv": float(np.sqrt(np.mean(values_uv * values_uv))),
            "channel_mean_abs_median_uv": float(np.median(np.abs(channel_mean_uv))),
            "post_average_reference_max_abs_channel_mean_uv": float(np.max(np.abs(residual)) * 1e6),
            "reference_id": "average_over_fixed_channel_set",
            "reference_set_sha256": EXPECTED_SET_SHA256,
        },
        "spectral": {
            "estimator": "scipy.signal.welch",
            "window": "hann",
            "nperseg": 1000,
            "noverlap": 500,
            "nfft": 1000,
            "detrend": "constant",
            "density_scaling": "uV^2/Hz",
            "band_power_units": "uV^2",
            "bin_width_hz": bin_width,
            "bands": bands,
        },
    }


def _kernel_results(signals: np.ndarray, labels: Sequence[str]) -> tuple[dict[str, Any], np.ndarray | None, dict[str, Any]]:
    from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
    from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency
    from mndm.mnps.v3_2.oi import compute_oi_matrix

    failures: dict[str, str] = {}
    try:
        ed = compute_ed_raw_sensor_covariance_participation_ratio(signals, labels)
    except Exception as exc:  # preserve OI/NIE if one estimator fails
        ed = None
        failures["ED"] = f"{type(exc).__name__}:{exc}"
    try:
        oi = compute_oi_matrix(signals, labels)
    except Exception as exc:  # preserve ED/NIE if one estimator fails
        oi = None
        failures["OI"] = f"{type(exc).__name__}:{exc}"
    try:
        weights, edge_audit = construct_absolute_pearson_weights(signals)
    except Exception as exc:  # graph construction is NIE-local
        weights = None
        edge_audit = {"status": "failed", "reason": f"{type(exc).__name__}:{exc}"}
        failures["NIE_edge_construction"] = edge_audit["reason"]
    nie = None
    if weights is not None:
        try:
            order_hash, _ = labels_hashes(labels)
            nie = compute_nie_raw_weighted_global_efficiency(
                weights,
                node_labels=labels,
                node_order_sha256=order_hash,
                edge_estimator_id=NIE_EDGE_ESTIMATOR_ID,
                edge_rule_id=NIE_EDGE_RULE_ID,
            )
            edge_audit = {**edge_audit, "matrix_kernel_reason": nie.reason, "matrix_kernel_valid": nie.valid}
        except Exception as exc:  # NIE must not discard computable ED/OI
            failures["NIE"] = f"{type(exc).__name__}:{exc}"
    return {"ED": ed, "OI": oi, "NIE": nie}, weights, {**edge_audit, "kernel_failures": failures}


def _kernel_value(kernel: Any | None) -> float | None:
    if kernel is None or getattr(kernel, "value", None) is None or not getattr(kernel, "valid", False):
        return None
    return float(kernel.value)


def _metric_equal(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and math.isclose(
        left, right, rel_tol=INVARIANCE_RTOL, abs_tol=INVARIANCE_ATOL
    )


def shared_time_permutation_control(
    signals: np.ndarray,
    labels: Sequence[str],
    *,
    subject_id: str,
    window_id: str,
    original: Mapping[str, Any],
) -> dict[str, Any]:
    """Permute one common time index across sensors and audit invariants."""
    matrix = np.asarray(signals, dtype=np.float64)
    serialized = "\0".join((CONTROL_ID, subject_id, window_id)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(serialized).digest()[:8], "big", signed=False)
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(matrix.shape[1])
    shuffled = matrix[:, permutation]
    original_cov, original_corr, _ = _pearson_components(matrix)
    shuffled_cov, shuffled_corr, _ = _pearson_components(shuffled)
    shuffled_results, _, shuffled_edge_audit = _kernel_results(shuffled, labels)
    original_values = {family: _kernel_value(result) for family, result in original.items()}
    shuffled_values = {family: _kernel_value(result) for family, result in shuffled_results.items()}
    covariance_equal = bool(np.allclose(original_cov, shuffled_cov, rtol=INVARIANCE_RTOL, atol=INVARIANCE_ATOL))
    correlation_equal = bool(np.allclose(original_corr, shuffled_corr, rtol=INVARIANCE_RTOL, atol=INVARIANCE_ATOL))
    ed_equal = _metric_equal(original_values.get("ED"), shuffled_values.get("ED"))
    nie_equal = _metric_equal(original_values.get("NIE"), shuffled_values.get("NIE"))
    return {
        "status": "passed" if covariance_equal and correlation_equal and ed_equal and nie_equal else "failed",
        "control_id": CONTROL_ID,
        "seed": seed,
        "seed_rule": "SHA256(NUL-joined control_id, subject_id, window_id); first 8 bytes big-endian; PCG64",
        "permutation_sha256_little_endian_int64": hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest(),
        "same_permutation_for_all_channels": True,
        "tolerance": {"rtol": INVARIANCE_RTOL, "atol": INVARIANCE_ATOL},
        "covariance_max_abs_difference": float(np.max(np.abs(original_cov - shuffled_cov))),
        "correlation_max_abs_difference": float(np.max(np.abs(original_corr - shuffled_corr))),
        "covariance_preserved": covariance_equal,
        "correlation_preserved": correlation_equal,
        "ED_preserved": ed_equal,
        "NIE_preserved": nie_equal,
        "original_values": original_values,
        "permuted_values": shuffled_values,
        "metric_deltas_permuted_minus_original": {
            family: None if original_values.get(family) is None or shuffled_values.get(family) is None
            else float(shuffled_values[family] - original_values[family])
            for family in ("ED", "OI", "NIE")
        },
        "permuted_NIE_edge_audit": shuffled_edge_audit,
        "qualified": False,
        "interpretation": "controlled temporal-order sensitivity only",
    }


def _support_record(
    window: Any,
    nominal: tuple[float, float],
    subject_id: str,
    labels: Sequence[str],
) -> tuple[dict[str, Any], str]:
    provenance = _mapping(getattr(window, "provenance", {}))
    audit = _mapping(provenance.get("support_audit"))
    source = _mapping(provenance.get("source"))
    dependency = _mapping(provenance.get("source_dependency"))
    filter_spec = _mapping(dependency.get("eeg_bandpass"))
    known = _support_audit_complete(window, provenance)
    support_status = "known" if known else "unsupported" if getattr(window, "status", None) != "ready" else "unknown"
    source_sha = source.get("sha256_after_read") or source.get("sha256_before_read")
    start = int(round(nominal[0] * SAMPLE_RATE_HZ))
    end = int(round(nominal[1] * SAMPLE_RATE_HZ))
    support = {
        "support_status": support_status,
        "source_id": f"ANPHY-Sleep/{subject_id}",
        "source_sha256": source_sha,
        "segment_id": f"edf:{str(source_sha or '')[:16]}:continuous_segment0",
        "sample_grid_id": f"ANPHY_SLEEP_{subject_id}_E1_G2_output_grid_250Hz",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "start_sample": start,
        "end_sample_exclusive": end,
        "sample_count": SAMPLE_COUNT if known else None,
        "nominal_window_seconds": WINDOW_SECONDS,
        "dependency_start_sample": dependency.get("source_sample_start"),
        "dependency_end_sample_exclusive": dependency.get("source_sample_end_exclusive"),
        "gap_free": audit.get("gap_free"),
        "coverage": 1.0 if known else None,
        "availability": "all nominal output samples and source dependencies audited" if known else (
            f"unsupported: {getattr(window, 'reason', None)}" if support_status == "unsupported" else "support audit incomplete"
        ),
        "filter_warmup_samples": None if filter_spec.get("tap_count") is None else int(filter_spec["tap_count"]) - 1,
        "filter_delay_samples": None,
    }
    return support, support_status


def _measurement_payload(
    *,
    family_id: str,
    estimator_id: str,
    schema_id: str,
    units: str,
    kernel: Any | None,
    reason: str | None,
    window: Any,
    nominal: tuple[float, float],
    subject_id: str,
    labels: Sequence[str],
    run_id: str,
    input_sha256: str | None,
    code_sha256: str,
    protocol_sha256: str,
    config_sha256: str,
    runtime: Mapping[str, str],
    method_parameters: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    from mndm.mnps.v3_2.contracts import MeasurementRecord, RealizationRecord, ReproducibilityRecord, TimeSupportRecord

    support_dict, support_status = _support_record(window, nominal, subject_id, labels)
    labels_tuple = tuple(str(label) for label in labels)
    order_hash, set_hash = labels_hashes(labels_tuple) if labels_tuple else (None, None)
    identity_ok = len(labels_tuple) == CHANNEL_COUNT and order_hash == EXPECTED_ORDER_SHA256 and set_hash == EXPECTED_SET_SHA256
    value = _kernel_value(kernel)
    computed = bool(getattr(kernel, "computed", False)) if kernel is not None else False
    valid = value is not None
    kernel_qc = tuple(str(item) for item in getattr(kernel, "qc", ())) if kernel is not None else ()
    qc = list(kernel_qc)
    if support_status == "unknown":
        qc.append("end_to_end_support_audit_incomplete")
    elif support_status == "unsupported":
        qc.append(f"window_unsupported:{reason or 'upstream_window_unsupported'}")
    if not identity_ok:
        qc.append("fixed83_realization_identity_incomplete")
    final_reason = None if valid else (getattr(kernel, "reason", None) or reason or "kernel_not_attempted")
    support = TimeSupportRecord(**support_dict)
    realization_id = {"ED": ED_REALIZATION_ID, "OI": OI_REALIZATION_ID, "NIE": NIE_REALIZATION_ID}[family_id]
    realization = RealizationRecord(
        audit_status="complete" if identity_ok else "incomplete" if labels_tuple else "unknown",
        kind="fixed_83_ANPHY_Sleep_position_file_surface",
        requested_id=realization_id,
        actual_id=realization_id if identity_ok else None,
        ordered_labels=labels_tuple or None,
        order_sha256=order_hash,
        set_sha256=set_hash,
        count=len(labels_tuple) if labels_tuple else None,
        coverage=1.0 if support_status == "known" else None,
        reference_id="average_over_fixed_channel_set" if labels_tuple else None,
        reference_set_sha256=set_hash if labels_tuple else None,
    )
    record = MeasurementRecord(
        family_id=family_id,
        estimator_id=str(getattr(kernel, "estimator_id", estimator_id)),
        schema_id=str(getattr(kernel, "schema_id", schema_id)),
        value=value,
        units=units,
        computed=computed,
        valid=valid,
        qualified=False,
        status="computed_exploratory" if valid else "unsupported",
        reason=final_reason,
        support=support,
        realization=realization,
        qc=tuple(dict.fromkeys(qc)),
        reproducibility=ReproducibilityRecord(
            status="complete" if input_sha256 is not None and code_sha256 and protocol_sha256 and config_sha256 else "partial",
            run_id=run_id,
            input_sha256=input_sha256,
            code_sha256=code_sha256,
            protocol_sha256=protocol_sha256,
            config_id=f"vector_feasibility_002_{subject_id}_e1_g2",
            config_sha256=config_sha256,
            dependency_versions=tuple(sorted((str(key), str(value)) for key, value in runtime.items())) or None,
        ),
        method_parameters=method_parameters,
        direction="higher_more_ordinally_irregular" if family_id == "OI" else None,
        claim_status="no_biological_claim",
    )
    return {"measurement_record": record.to_dict(), "matrix_kernel_result": None if kernel is None else kernel.to_dict()}


def _make_window_row(
    *,
    index: int,
    window: Any,
    nominal: tuple[float, float],
    labels: Sequence[str],
    subject_id: str,
    run_id: str,
    code_sha256: str,
    protocol_sha256: str,
    config_sha256: str,
    runtime: Mapping[str, str],
    edge_dir: Path,
) -> dict[str, Any]:
    provenance = _mapping(getattr(window, "provenance", {}))
    source = _mapping(provenance.get("source"))
    dependency = _mapping(provenance.get("source_dependency"))
    grid_id = f"ANPHY_SLEEP_{subject_id}_{int(nominal[0]):03d}_{int(nominal[1]):03d}s"
    matrix: np.ndarray | None = None
    matrix_error: str | None = None
    input_hash: str | None = None
    if getattr(window, "status", None) == "ready" and getattr(window, "signals", None) is not None:
        try:
            candidate = np.asarray(window.signals, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            candidate = np.empty((0, 0), dtype=np.float64)
        if candidate.shape != (CHANNEL_COUNT, SAMPLE_COUNT):
            matrix_error = "audited_matrix_shape_mismatch"
        elif not np.isfinite(candidate).all():
            matrix_error = "audited_matrix_non_finite"
        else:
            matrix = candidate
            input_hash = sha256_array(matrix)
    elif getattr(window, "status", None) == "ready":
        matrix_error = "ready_window_missing_matrix"
    else:
        matrix_error = str(getattr(window, "reason", None) or "upstream_window_unsupported")

    kernels: dict[str, Any] = {"ED": None, "OI": None, "NIE": None}
    edge_weights: np.ndarray | None = None
    edge_audit: dict[str, Any] = {"status": "not_attempted", "reason": matrix_error}
    diagnostics: dict[str, Any] = {"status": "unsupported", "reason": matrix_error, "values": {}}
    amplitude: dict[str, Any] = {"status": "unsupported", "reason": matrix_error, "amplitude": {}, "spectral": {}}
    permutation: dict[str, Any] = {"status": "not_attempted", "reason": matrix_error}
    if matrix is not None:
        kernels, edge_weights, edge_audit = _kernel_results(matrix, labels)
        if edge_weights is not None:
            edge_path = edge_dir / f"{subject_id}_window_{index:02d}_{int(nominal[0]):03d}_{int(nominal[1]):03d}s.npy"
            with edge_path.open("wb") as stream:
                np.save(stream, edge_weights, allow_pickle=False)
            edge_audit = {
                **edge_audit,
                "matrix_path": str(edge_path.relative_to(edge_dir.parent)).replace("\\", "/"),
                "matrix_sha256": sha256_file(edge_path),
            }
        try:
            diagnostics = covariance_connectivity_diagnostics(
                matrix,
                raw_ed_value=_kernel_value(kernels["ED"]),
                nie_value=_kernel_value(kernels["NIE"]),
            )
        except Exception as exc:  # diagnostics must not suppress primary metrics
            diagnostics = {"status": "failed", "reason": f"{type(exc).__name__}:{exc}", "values": {}}
        try:
            amplitude = amplitude_and_spectral_diagnostics(matrix)
        except Exception as exc:  # conventional comparators remain family-local
            amplitude = {"status": "failed", "reason": f"{type(exc).__name__}:{exc}", "amplitude": {}, "spectral": {}}
        try:
            permutation = shared_time_permutation_control(
                matrix,
                labels,
                subject_id=subject_id,
                window_id=grid_id,
                original=kernels,
            )
        except Exception as exc:  # keep the original computable measurements if a control fails
            permutation = {"status": "failed", "reason": f"{type(exc).__name__}:{exc}", "qualified": False}

    family_specs = {
        "ED": (
            "ED.sensor_covariance_participation_ratio_raw_matrix.v1",
            "nmdm.mnps_v3_2.ed_raw_matrix_result.v1",
            "sensor_directions",
            "centered_sample_covariance_ddof1_no_shrinkage_no_channel_normalization",
        ),
        "OI": (
            "OI.matrix_input_channel_median_normalized_PE5_delay1.v1",
            "nmdm.mnps_v3_2.oi_matrix_result.v1",
            "normalized_entropy",
            "order5_delay1_stable_mergesort_median_all_fixed83_channels",
        ),
        "NIE": (
            "D1.sensor_weighted_global_efficiency_invweight.v0",
            "nmdm.mnps_v3_2.nie_matrix_result.v1",
            "global_efficiency",
            "abs_pearson_no_threshold_inverse_positive_weight_edge_length",
        ),
    }
    measurements: dict[str, Any] = {}
    realization_ids = {"ED": ED_REALIZATION_ID, "OI": OI_REALIZATION_ID, "NIE": NIE_REALIZATION_ID}
    label_order_hash, label_set_hash = labels_hashes(labels) if labels else (None, None)
    realization_matches = (
        len(labels) == CHANNEL_COUNT
        and label_order_hash == EXPECTED_ORDER_SHA256
        and label_set_hash == EXPECTED_SET_SHA256
    )
    for family, (estimator_id, schema_id, units, definition) in family_specs.items():
        kernel = kernels[family]
        reason = matrix_error
        failures = _mapping(edge_audit.get("kernel_failures"))
        if family in failures:
            reason = str(failures[family])
        elif family == "NIE" and matrix is not None and edge_weights is None:
            reason = str(failures.get("NIE_edge_construction") or edge_audit.get("reason") or "edge_construction_unsupported")
        measurements[family] = {
            "realization_id": realization_ids[family],
            "requested_realization_id": realization_ids[family],
            "actual_realization_id": realization_ids[family] if realization_matches else None,
            "realization_identity_status": "matched" if realization_matches else "incomplete",
            **_measurement_payload(
                family_id=family,
                estimator_id=estimator_id,
                schema_id=schema_id,
                units=units,
                kernel=kernel,
                reason=reason,
                window=window,
                nominal=nominal,
                subject_id=subject_id,
                labels=labels,
                run_id=run_id,
                input_sha256=input_hash,
                code_sha256=code_sha256,
                protocol_sha256=protocol_sha256,
                config_sha256=config_sha256,
                runtime=runtime,
                method_parameters=(("definition", definition),),
            ),
            "qualified": False,
        }
        if family == "NIE":
            measurements[family]["edge_audit"] = edge_audit

    support_status = _support_record(window, nominal, subject_id, labels)[1]
    return {
        "schema": RESULT_SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "recording_id": f"ANPHY-Sleep/{subject_id}",
        "subject_id": subject_id,
        "window_id": grid_id,
        "window_index": index,
        "nominal_window_sec_half_open": list(nominal),
        "output_grid": {
            "sample_grid_id": f"ANPHY_SLEEP_{subject_id}_E1_G2_output_grid_250Hz",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "start_sample": int(round(nominal[0] * SAMPLE_RATE_HZ)),
            "end_sample_exclusive": int(round(nominal[1] * SAMPLE_RATE_HZ)),
            "sample_count": SAMPLE_COUNT,
        },
        "window_audit_status": str(getattr(window, "status", "unknown")),
        "window_audit_reason": getattr(window, "reason", None),
        "support_status": support_status,
        "input_matrix_sha256_little_endian_float64": input_hash,
        "source_sha256": source.get("sha256_after_read") or source.get("sha256_before_read"),
        "source_dependency": dependency,
        "support_audit": provenance.get("support_audit"),
        "window_audit_provenance": provenance,
        "realization_identity": {
            "ordered_labels": list(labels),
            "channel_count": len(labels),
            "order_sha256": label_order_hash,
            "set_sha256": label_set_hash,
            "reference_id": "average_over_fixed_channel_set",
            "reference_set_sha256": labels_hashes(labels)[1],
        },
        "measurements": measurements,
        "covariance_connectivity_diagnostics": diagnostics,
        "amplitude_and_spectral_diagnostics": amplitude,
        "time_permutation_control": permutation,
        "qualified": False,
        "claim_status": "no_biological_claim",
    }


def _pearson_pair(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all() or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _valid_window_value(row: Mapping[str, Any], family_or_metric: str) -> float | None:
    if family_or_metric in {"ED", "OI", "NIE"}:
        family = _mapping(_mapping(row.get("measurements")).get(family_or_metric))
        record = _mapping(family.get("measurement_record"))
        value = record.get("value")
        return float(value) if record.get("valid") is True and value is not None else None
    diagnostics = _mapping(row.get("covariance_connectivity_diagnostics"))
    value = _mapping(diagnostics.get("values")).get(family_or_metric)
    if value is not None:
        return float(value)
    amp = _mapping(row.get("amplitude_and_spectral_diagnostics"))
    spectral = _mapping(amp.get("spectral"))
    if "." in family_or_metric:
        band_id, stat = family_or_metric.split(".", maxsplit=1)
        value = _mapping(_mapping(spectral.get("bands")).get(band_id)).get(stat)
        return None if value is None else float(value)
    return None


def participant_summaries(participant_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Summarize windows within participant before between-participant analysis."""
    metric_ids = [*PARTICIPANT_SUMMARY_METRICS]
    for band_id in PSD_BANDS_HZ:
        metric_ids.extend((f"{band_id}.power_mean_uv2", f"{band_id}.power_median_uv2"))
    summaries: dict[str, Any] = {}
    within: dict[str, Any] = {}
    primary_pairs = (("ED", "OI"), ("ED", "NIE"), ("OI", "NIE"))
    for subject_id, rows in participant_rows.items():
        values_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_ids}
        metric_windows: dict[str, dict[str, float]] = {metric: {} for metric in metric_ids}
        for row in rows:
            for metric in metric_ids:
                value = _valid_window_value(row, metric)
                if value is not None and math.isfinite(value):
                    values_by_metric[metric].append(value)
                    metric_windows[metric][str(row["window_id"])] = value
        per_metric: dict[str, Any] = {}
        for metric, values in values_by_metric.items():
            array = np.asarray(values, dtype=np.float64)
            per_metric[metric] = {
                "valid_window_count": int(array.size),
                "median": None if not array.size else float(np.median(array)),
                "q25": None if not array.size else float(np.quantile(array, 0.25)),
                "q75": None if not array.size else float(np.quantile(array, 0.75)),
                "matched_window_ids": list(metric_windows[metric]),
            }
        summaries[subject_id] = {
            "subject_id": subject_id,
            "nominal_window_count": len(rows),
            "valid_measure_summaries": per_metric,
            "aggregation": "median and IQR across valid supported 20-second windows within participant",
            "windows_are_independent": False,
        }
        pairs: dict[str, Any] = {}
        for left, right in primary_pairs:
            common = sorted(set(metric_windows[left]) & set(metric_windows[right]))
            pairs[f"{left}_{right}"] = {
                "n_windows": len(common),
                "matched_window_ids": common,
                "pearson_r": _pearson_pair(
                    [metric_windows[left][window_id] for window_id in common],
                    [metric_windows[right][window_id] for window_id in common],
                ),
                "inference": "none_descriptive_overlapping_dependencies",
            }
        within[subject_id] = pairs

    participant_metric_values = {
        subject_id: {
            metric: _mapping(_mapping(summary.get("valid_measure_summaries")).get(metric)).get("median")
            for metric in metric_ids
        }
        for subject_id, summary in summaries.items()
    }
    association_metrics = ["ED", "OI", "NIE", "ED_R", "mean_absolute_correlation", "mean_squared_correlation", "nie_graph_path_increment"]
    association_metrics.extend(f"{band}.power_mean_uv2" for band in PSD_BANDS_HZ)
    matrix: dict[str, Any] = {}
    for left in association_metrics:
        matrix[left] = {}
        for right in association_metrics:
            matched = [
                subject_id for subject_id in summaries
                if participant_metric_values[subject_id].get(left) is not None
                and participant_metric_values[subject_id].get(right) is not None
            ]
            matrix[left][right] = {
                "n_participants": len(matched),
                "participant_ids": matched,
                "pearson_r": _pearson_pair(
                    [float(participant_metric_values[sid][left]) for sid in matched],
                    [float(participant_metric_values[sid][right]) for sid in matched],
                ),
            }

    redundancy_flags = []
    for left, right in primary_pairs:
        pair = matrix[left][right]
        value = pair["pearson_r"]
        if value is not None and pair["n_participants"] >= 10 and abs(value) >= 0.90:
            redundancy_flags.append({"flag": "participant_median_redundancy_review", "pair": [left, right], "n_participants": pair["n_participants"], "pearson_r": value})

    def participant_median_abs_difference(metric_values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
        per_subject = {sid: float(np.median(values)) for sid, values in metric_values.items() if values}
        triggering = [sid for sid, value in per_subject.items() if abs(value) <= 1e-10]
        return {"participant_median_values": per_subject, "n_participants_with_abs_median_le_1e-10": len(triggering), "triggered": len(triggering) >= 10}

    ed_deltas: dict[str, list[float]] = {}
    nie_increments: dict[str, list[float]] = {}
    oi_deltas: dict[str, list[float]] = {}
    for subject_id, rows in participant_rows.items():
        ed_deltas[subject_id] = []
        nie_increments[subject_id] = []
        oi_deltas[subject_id] = []
        for row in rows:
            measurement = _mapping(row.get("measurements"))
            ed = _mapping(_mapping(measurement.get("ED")).get("measurement_record"))
            oi = _mapping(_mapping(measurement.get("OI")).get("measurement_record"))
            nie = _mapping(_mapping(measurement.get("NIE")).get("measurement_record"))
            values = _mapping(_mapping(row.get("covariance_connectivity_diagnostics")).get("values"))
            ed_r = values.get("ED_R")
            mean_abs = values.get("mean_absolute_correlation")
            if ed.get("value") is not None and ed_r is not None:
                ed_deltas[subject_id].append(abs(float(ed["value"]) - float(ed_r)))
            if nie.get("value") is not None and mean_abs is not None:
                nie_increments[subject_id].append(abs(float(nie["value"]) - float(mean_abs)))
            control = _mapping(row.get("time_permutation_control"))
            delta = _mapping(control.get("metric_deltas_permuted_minus_original")).get("OI")
            if delta is not None:
                oi_deltas[subject_id].append(abs(float(delta)))
    ed_component = participant_median_abs_difference(ed_deltas)
    nie_component = participant_median_abs_difference(nie_increments)
    oi_subject_medians = {sid: float(np.median(values)) for sid, values in oi_deltas.items() if values}
    oi_triggering = [sid for sid, value in oi_subject_medians.items() if value <= 1e-12]
    review_triggers = {
        "redundancy_review": redundancy_flags,
        "ED_variance_component_review": ed_component,
        "NIE_path_contribution_review": nie_component,
        "OI_temporal_order_review": {
            "participant_median_absolute_OI_permutation_change": oi_subject_medians,
            "n_participants_with_median_le_1e-12": len(oi_triggering),
            "triggered": len(oi_triggering) >= 10,
        },
        "automatic_coordinate_removal": False,
        "inference": "predeclared descriptive review flags only; no qualification decision",
    }
    return {
        "schema": "mndm.mnps_v3_2.vector_feasibility_002_participant_summary.v1",
        "participant_summaries": summaries,
        "within_participant_primary_correlations": within,
        "participant_level_correlation_matrix": matrix,
        "predeclared_review_triggers": review_triggers,
        "inference": "none; participant medians are the between-participant rows; window overlap is retained and not treated as independent",
    }


def _write_long_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import csv

    fields = [
        "subject_id", "recording_id", "window_id", "window_index", "nominal_start_sec", "nominal_end_sec",
        "family_id", "estimator_id", "realization_id", "value", "units", "computed", "valid", "status",
        "reason", "qualified", "support_status", "dependency_start_sample", "dependency_end_sample_exclusive",
        "source_sha256", "input_matrix_sha256", "channel_count", "ordered_labels_json", "channel_order_sha256",
        "channel_set_sha256", "reference_id", "reference_set_sha256", "edge_matrix_sha256", "qc_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            dependency = _mapping(row.get("source_dependency"))
            identity = _mapping(row.get("realization_identity"))
            for family_id, family in _mapping(row.get("measurements")).items():
                record = _mapping(family.get("measurement_record"))
                support = _mapping(record.get("support"))
                realization = _mapping(record.get("realization"))
                writer.writerow({
                    "subject_id": row.get("subject_id"), "recording_id": row.get("recording_id"),
                    "window_id": row.get("window_id"), "window_index": row.get("window_index"),
                    "nominal_start_sec": row.get("nominal_window_sec_half_open", [None, None])[0],
                    "nominal_end_sec": row.get("nominal_window_sec_half_open", [None, None])[1],
                    "family_id": family_id, "estimator_id": record.get("estimator_id"),
                    "realization_id": family.get("realization_id"), "value": record.get("value"),
                    "units": record.get("units"), "computed": record.get("computed"), "valid": record.get("valid"),
                    "status": record.get("status"), "reason": record.get("reason"), "qualified": record.get("qualified"),
                    "support_status": support.get("support_status"),
                    "dependency_start_sample": dependency.get("source_sample_start"),
                    "dependency_end_sample_exclusive": dependency.get("source_sample_end_exclusive"),
                    "source_sha256": row.get("source_sha256"),
                    "input_matrix_sha256": row.get("input_matrix_sha256_little_endian_float64"),
                    "channel_count": realization.get("count", identity.get("channel_count")),
                    "ordered_labels_json": json.dumps(realization.get("ordered_labels", identity.get("ordered_labels")), ensure_ascii=False, separators=(",", ":")),
                    "channel_order_sha256": realization.get("order_sha256", identity.get("order_sha256")),
                    "channel_set_sha256": realization.get("set_sha256", identity.get("set_sha256")),
                    "reference_id": realization.get("reference_id", identity.get("reference_id")),
                    "reference_set_sha256": realization.get("reference_set_sha256", identity.get("reference_set_sha256")),
                    "edge_matrix_sha256": _mapping(family.get("edge_audit")).get("matrix_sha256"),
                    "qc_json": json.dumps(record.get("qc", []), ensure_ascii=False, separators=(",", ":")),
                })


def _write_config_overlay(
    *,
    subject_id: str,
    source_path: Path,
    positions_path: Path,
    output_path: Path,
    subject_output_root: Path,
) -> str:
    import yaml

    overlay = yaml.safe_load(BASE_OVERLAY_PATH.read_text(encoding="utf-8"))
    grid = overlay["eeg_grid"]
    grid.update({
        "dataset_id": "ANPHY-Sleep",
        "dataset_version": "OSF:10.17605/OSF.IO/R26FH; local snapshot revision unverified",
        "subject_id": subject_id,
        "session_id": "overnight_single_recording",
        "task_id": "sleep_PSG_overnight",
        "run_id": "single_EDF_in_subject_folder",
        "metadata_status": "dataset_doi_and_overnight_session_resolved_local_archive_revision_unverified",
        "metadata_note": (
            "Public dataset DOI and one overnight recording per participant folder are resolved. "
            "The exact local OSF archive revision is unavailable; local EDF header format is EDF+C "
            "while the paper describes EDF+D. Full source SHA-256 is audited during extraction."
        ),
        "source_edf": str(source_path.resolve()).replace("\\", "/"),
        "positions_file": str(positions_path.resolve()).replace("\\", "/"),
        "output_root": str(subject_output_root.resolve()).replace("\\", "/"),
    })
    payload = yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return sha256_file(output_path)


def _source_failure_rows(
    *,
    subject_id: str,
    reason: str,
    run_id: str,
    protocol_sha256: str,
    code_sha256: str,
) -> list[dict[str, Any]]:
    """Create all 15 explicit null rows when a frozen source cannot be read."""
    from mndm.mnps.v3_2.contracts import MeasurementRecord, RealizationRecord, ReproducibilityRecord, TimeSupportRecord

    rows = []
    for index, nominal in enumerate(WINDOWS_SEC):
        families = {}
        for family, estimator, schema, units, realization_id in (
            ("ED", "ED.sensor_covariance_participation_ratio_raw_matrix.v1", "nmdm.mnps_v3_2.ed_raw_matrix_result.v1", "sensor_directions", ED_REALIZATION_ID),
            ("OI", "OI.matrix_input_channel_median_normalized_PE5_delay1.v1", "nmdm.mnps_v3_2.oi_matrix_result.v1", "normalized_entropy", OI_REALIZATION_ID),
            ("NIE", "D1.sensor_weighted_global_efficiency_invweight.v0", "nmdm.mnps_v3_2.nie_matrix_result.v1", "global_efficiency", NIE_REALIZATION_ID),
        ):
            record = MeasurementRecord(
                family_id=family, estimator_id=estimator, schema_id=schema, value=None, units=units,
                computed=False, valid=False, qualified=False, status="unsupported", reason=reason,
                support=TimeSupportRecord(support_status="unsupported", source_id=f"ANPHY-Sleep/{subject_id}",
                    sample_grid_id=f"ANPHY_SLEEP_{subject_id}_E1_G2_output_grid_250Hz", sample_rate_hz=SAMPLE_RATE_HZ,
                    start_sample=int(nominal[0]*SAMPLE_RATE_HZ), end_sample_exclusive=int(nominal[1]*SAMPLE_RATE_HZ),
                    nominal_window_seconds=WINDOW_SECONDS, availability=reason),
                realization=RealizationRecord(audit_status="unknown", kind="fixed_83_ANPHY_Sleep_position_file_surface",
                    requested_id=realization_id, actual_id=None, ordered_labels=None, order_sha256=None,
                    set_sha256=None, count=None, reference_id=None, reference_set_sha256=None),
                qc=(reason,), reproducibility=ReproducibilityRecord(status="partial", run_id=run_id,
                    code_sha256=code_sha256, protocol_sha256=protocol_sha256), claim_status="no_biological_claim",
            )
            families[family] = {"realization_id": realization_id, "measurement_record": record.to_dict(), "matrix_kernel_result": None, "qualified": False}
        rows.append({
            "schema": RESULT_SCHEMA_ID, "experiment_id": EXPERIMENT_ID, "run_id": run_id,
            "recording_id": f"ANPHY-Sleep/{subject_id}", "subject_id": subject_id,
            "window_id": f"ANPHY_SLEEP_{subject_id}_{int(nominal[0]):03d}_{int(nominal[1]):03d}s",
            "window_index": index, "nominal_window_sec_half_open": list(nominal),
            "output_grid": {"sample_grid_id": f"ANPHY_SLEEP_{subject_id}_E1_G2_output_grid_250Hz",
                "sample_rate_hz": SAMPLE_RATE_HZ, "start_sample": int(nominal[0]*SAMPLE_RATE_HZ),
                "end_sample_exclusive": int(nominal[1]*SAMPLE_RATE_HZ), "sample_count": SAMPLE_COUNT},
            "window_audit_status": "unsupported", "window_audit_reason": reason,
            "support_status": "unsupported", "input_matrix_sha256_little_endian_float64": None,
            "source_sha256": None, "source_dependency": None, "support_audit": None,
            "measurements": families, "covariance_connectivity_diagnostics": {"status": "not_attempted", "reason": reason},
            "amplitude_and_spectral_diagnostics": {"status": "not_attempted", "reason": reason},
            "time_permutation_control": {"status": "not_attempted", "reason": reason},
            "qualified": False, "claim_status": "no_biological_claim",
        })
    return rows


def run(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    selection_path: Path = SELECTION_PATH,
    output_root: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Run the frozen twelve-participant study and return its manifest path."""
    source_root = source_root.resolve()
    selection_path = selection_path.resolve()
    if not selection_path.is_file() or not BASE_OVERLAY_PATH.is_file() or not PROTOCOL_PATH.is_file():
        raise FileNotFoundError("selection manifest, base overlay, and frozen protocol must exist")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_sha256 = sha256_file(selection_path)
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    base_overlay_sha256 = sha256_file(BASE_OVERLAY_PATH)
    validate_frozen_preregistration(
        selection,
        selection_sha256=selection_sha256,
        protocol_sha256=protocol_sha256,
        base_overlay_sha256=base_overlay_sha256,
    )
    subject_ids = _selected_ids(selection)
    if len(subject_ids) != 12 or len(set(subject_ids)) != 12:
        raise ValueError("frozen_selection_must_contain_twelve_unique_subjects")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_sl006_vectorfeasibility002")
    run_root = ((output_root or DEFAULT_OUTPUT_ROOT) / run_id).resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite VECTOR-FEASIBILITY-002 output root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    edge_dir = run_root / "edge_matrices"
    edge_dir.mkdir()
    config_dir = run_root / "config_snapshots"
    config_dir.mkdir()
    numba_cache = run_root / ".numba_cache"
    numba_cache.mkdir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    positions_path = source_root / "Co-registered average positions.pos"
    if not positions_path.is_file():
        raise FileNotFoundError(f"frozen positions file not found: {positions_path}")
    positions_sha256 = sha256_file(positions_path)
    if positions_sha256 != EXPECTED_POSITIONS_SHA256:
        raise RuntimeError("frozen_position_file_sha256_mismatch")
    code_paths = {
        "runner": Path(__file__).resolve(),
        "ed_kernel": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/ed.py",
        "oi_kernel": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/oi.py",
        "nie_kernel": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/nie.py",
        "contracts": REPO_ROOT / "mndm/src/mndm/mnps/v3_2/contracts.py",
        "audited_grid_loader": REPO_ROOT / "mndm/src/mndm/mnps/v3_1/eeg_windows.py",
        "preprocessing": REPO_ROOT / "mndm/src/mndm/preprocess.py",
        "base_overlay": BASE_OVERLAY_PATH,
        "selection_manifest": selection_path,
    }
    code_components = {name: sha256_file(path) for name, path in code_paths.items()}
    code_sha256 = sha256_json(code_components)
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__}
    participant_results: dict[str, Any] = {}
    participant_rows: dict[str, list[dict[str, Any]]] = {}
    source_audits: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for subject_id in subject_ids:
        position = subject_ids.index(subject_id) + 1
        print(f"VECTOR-FEASIBILITY-002 [{position}/12] checking frozen source {subject_id}", flush=True)
        frozen = next((row for row in _selection_rows(selection) if row.get("subject_id") == subject_id), None)
        if frozen is None or frozen.get("relative_path") is None:
            reason = "frozen_selected_source_metadata_missing"
            source_audits[subject_id] = {"valid": False, "reason": reason}
            rows = _source_failure_rows(subject_id=subject_id, reason=reason, run_id=run_id, protocol_sha256=protocol_sha256, code_sha256=code_sha256)
            participant_rows[subject_id] = rows
            all_rows.extend(rows)
            participant_results[subject_id] = {"status": "unsupported", "reason": reason}
            continue
        source_path = (source_root / str(frozen["relative_path"])).resolve()
        try:
            source_audit = validate_frozen_source(source_path, subject_id, selection)
        except Exception as exc:
            source_audit = {"valid": False, "reason": f"selected_header_audit_failed:{type(exc).__name__}:{exc}"}
        source_audits[subject_id] = source_audit
        if not source_audit["valid"]:
            reason = str(source_audit.get("reason") or "selected_source_identity_invalid")
            rows = _source_failure_rows(subject_id=subject_id, reason=reason, run_id=run_id, protocol_sha256=protocol_sha256, code_sha256=code_sha256)
            participant_rows[subject_id] = rows
            all_rows.extend(rows)
            participant_results[subject_id] = {"status": "unsupported", "reason": reason}
            continue

        participant_root = run_root / subject_id
        participant_root.mkdir()
        overlay_path = config_dir / f"{subject_id}_e1_g2_overlay.yaml"
        config_sha256 = _write_config_overlay(
            subject_id=subject_id,
            source_path=source_path,
            positions_path=positions_path,
            output_path=overlay_path,
            subject_output_root=participant_root,
        )
        try:
            from mndm.mnps.v3_1 import load_audited_edf_eeg_grid
            print(f"VECTOR-FEASIBILITY-002 [{position}/12] loading audited EDF grid {subject_id}", flush=True)
            grid = load_audited_edf_eeg_grid(source_path, positions_path, overlay_path)
        except Exception as exc:
            reason = f"audited_grid_loader_failed:{type(exc).__name__}:{exc}"
            rows = _source_failure_rows(subject_id=subject_id, reason=reason, run_id=run_id, protocol_sha256=protocol_sha256, code_sha256=code_sha256)
            participant_rows[subject_id] = rows
            all_rows.extend(rows)
            participant_results[subject_id] = {"status": "unsupported", "reason": reason}
            continue

        provenance = _mapping(getattr(grid, "provenance", {}))
        windows = list(getattr(grid, "windows", ()))
        window_provenances = [_mapping(getattr(window, "provenance", {})) for window in windows]
        first_window_provenance = next((item for item in window_provenances if item), {})
        source_info = _mapping(first_window_provenance.get("source"))
        configuration = _mapping(first_window_provenance.get("configuration"))
        channels = _mapping(first_window_provenance.get("channels"))
        loader_identity = loader_identity_checks(
            window_provenances,
            payload_sha256_before_loader=source_audit.get("payload_sha256_before_loader"),
            overlay_sha256=config_sha256,
            positions_sha256=EXPECTED_POSITIONS_SHA256,
        )
        source_audit = {**source_audit, "loader_identity_checks": loader_identity}
        source_audits[subject_id] = source_audit
        if not all(loader_identity.values()):
            reason = "loader_input_identity_changed_or_unverifiable"
            rows = _source_failure_rows(subject_id=subject_id, reason=reason, run_id=run_id, protocol_sha256=protocol_sha256, code_sha256=code_sha256)
            participant_rows[subject_id] = rows
            all_rows.extend(rows)
            participant_results[subject_id] = {"status": "unsupported", "reason": reason, "loader_identity_checks": loader_identity}
            continue
        labels: tuple[str, ...] = ()
        for candidate in windows:
            candidate_channels = _nested(_mapping(getattr(candidate, "provenance", {})), "channels")
            if candidate_channels.get("labels"):
                labels = tuple(str(label) for label in candidate_channels["labels"])
                break
        if not labels and windows:
            labels = tuple(str(label) for label in getattr(windows[0], "channel_names", ()))
        subject_runtime = _mapping(_nested(_mapping(getattr(windows[0], "provenance", {})), "runtime")) if windows else {}
        combined_runtime = {**runtime, **{str(key): str(value) for key, value in subject_runtime.items()}}
        if len(windows) != len(WINDOWS_SEC):
            reason = f"nominal_window_count_mismatch:{len(windows)}"
            missing_rows = _source_failure_rows(subject_id=subject_id, reason=reason, run_id=run_id, protocol_sha256=protocol_sha256, code_sha256=code_sha256)
            rows = []
            for index, nominal in enumerate(WINDOWS_SEC):
                if index < len(windows):
                    rows.append(_make_window_row(
                        index=index,
                        window=windows[index],
                        nominal=nominal,
                        labels=labels,
                        subject_id=subject_id,
                        run_id=run_id,
                        code_sha256=code_sha256,
                        protocol_sha256=protocol_sha256,
                        config_sha256=config_sha256,
                        runtime=combined_runtime,
                        edge_dir=edge_dir,
                    ))
                else:
                    rows.append(missing_rows[index])
        else:
            rows = [
                _make_window_row(
                    index=index,
                    window=window,
                    nominal=nominal,
                    labels=labels,
                    subject_id=subject_id,
                    run_id=run_id,
                    code_sha256=code_sha256,
                    protocol_sha256=protocol_sha256,
                    config_sha256=config_sha256,
                    runtime=combined_runtime,
                    edge_dir=edge_dir,
                )
                for index, (window, nominal) in enumerate(zip(windows, WINDOWS_SEC, strict=True))
            ]
        participant_rows[subject_id] = rows
        all_rows.extend(rows)
        identity = _mapping(provenance.get("recording_identity"))
        participant_result = {
            "participant_id": subject_id,
            "dataset_identity": identity,
            "source_audit": source_audit,
            "grid_status": getattr(grid, "status", "unknown"),
            "grid_reason": getattr(grid, "reason", None),
            "source_sha256_before_read": source_info.get("sha256_before_read"),
            "source_sha256_after_read": source_info.get("sha256_after_read"),
            "source_hash_stable": source_info.get("sha256_stable_after_read"),
            "positions_sha256": provenance.get("positions_sha256"),
            "positions_sha256_after_read": channels.get("positions_sha256_after_read"),
            "positions_hash_stable": channels.get("positions_hash_stable"),
            "config_hash_stable": configuration.get("overlay_hash_stable"),
            "ordered_channel_labels": list(labels),
            "channel_count": len(labels),
            "channel_order_sha256": labels_hashes(labels)[0] if labels else None,
            "channel_set_sha256": labels_hashes(labels)[1] if labels else None,
            "reference_id": "average_over_fixed_channel_set",
            "reference_set_sha256": labels_hashes(labels)[1] if labels else None,
            "config_snapshot": str(overlay_path.relative_to(run_root)).replace("\\", "/"),
            "config_sha256": config_sha256,
            "window_count": len(rows),
            "supported_window_count": sum(row.get("window_audit_status") == "ready" for row in rows),
            "rows": rows,
            "qualified": False,
            "claim_status": "no_biological_claim",
        }
        participant_results[subject_id] = participant_result
        target = participant_root / "participant_results.json"
        target.write_text(json.dumps(participant_result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            f"VECTOR-FEASIBILITY-002 [{position}/12] finished {subject_id}: "
            f"grid={participant_result['grid_status']} supported_windows={participant_result['supported_window_count']}/15",
            flush=True,
        )

    summary = participant_summaries(participant_rows)
    (run_root / "participant_level_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_root / "selected_sources_audit.json").write_text(json.dumps(source_audits, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_long_csv(all_rows, run_root / "measurements_long.csv")

    family_counts: dict[str, Any] = {}
    for family in ("ED", "OI", "NIE"):
        records = [
            _mapping(_mapping(_mapping(row.get("measurements")).get(family)).get("measurement_record"))
            for row in all_rows
        ]
        family_counts[family] = {
            "nominal_rows": len(records),
            "computed_valid_rows": sum(record.get("valid") is True for record in records),
            "unsupported_or_invalid_rows": sum(record.get("valid") is not True for record in records),
            "all_qualified_false": all(record.get("qualified") is False for record in records),
        }
    source_identity_verified = all(source_audits.get(subject_id, {}).get("valid") is True for subject_id in subject_ids)
    source_hash_stable = all(participant_results.get(subject_id, {}).get("source_hash_stable") is True for subject_id in subject_ids)
    config_hash_stable = all(participant_results.get(subject_id, {}).get("config_hash_stable") is True for subject_id in subject_ids)
    positions_hash_stable = all(participant_results.get(subject_id, {}).get("positions_hash_stable") is True for subject_id in subject_ids)
    complete_accounting = len(all_rows) == len(subject_ids) * len(WINDOWS_SEC) and all(
        sum(row.get("subject_id") == subject_id for row in all_rows) == len(WINDOWS_SEC) for subject_id in subject_ids
    )
    grid_execution_complete = all(participant_results.get(subject_id, {}).get("grid_status") == "completed" for subject_id in subject_ids)
    engineering_pass = bool(
        source_identity_verified and source_hash_stable and config_hash_stable
        and positions_hash_stable and complete_accounting and grid_execution_complete
    )
    output_files = [path for path in run_root.rglob("*") if path.is_file()]
    output_hashes = {
        str(path.relative_to(run_root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_files)
    }
    manifest = {
        "schema": MANIFEST_SCHEMA_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256,
        "selection_path": str(selection_path),
        "selection_sha256": selection_sha256,
        "selection_frozen_at_utc": selection.get("selection_frozen_at_utc"),
        "selected_subject_ids_in_draw_order": list(subject_ids),
        "source_root": str(source_root),
        "dataset_id": "ANPHY-Sleep",
        "dataset_doi": "10.17605/OSF.IO/R26FH",
        "local_archive_revision": "unverified; source identity preserved per selected EDF by stable full SHA-256",
        "published_vs_local_edf_format": {"publication": "EDF+D", "local_header": "EDF+C"},
        "base_overlay_path": str(BASE_OVERLAY_PATH),
        "base_overlay_sha256": base_overlay_sha256,
        "code_component_sha256": code_components,
        "code_sha256_composite": code_sha256,
        "positions_path": str(positions_path),
        "positions_sha256": positions_sha256,
        "channel_count": CHANNEL_COUNT,
        "channel_order_sha256": EXPECTED_ORDER_SHA256,
        "channel_set_sha256": EXPECTED_SET_SHA256,
        "reference_id": "average_over_fixed_channel_set",
        "reference_set_sha256": EXPECTED_SET_SHA256,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "sample_count_per_channel": SAMPLE_COUNT,
        "nominal_window_count_per_participant": len(WINDOWS_SEC),
        "nominal_window_count_total": len(all_rows),
        "family_counts": family_counts,
        "participant_grid_status": {sid: participant_results.get(sid, {}).get("grid_status", "unsupported") for sid in subject_ids},
        "participant_window_coverage": {
            sid: {
                "nominal_rows": len(participant_rows.get(sid, ())),
                "supported_rows": sum(row.get("window_audit_status") == "ready" for row in participant_rows.get(sid, ())),
                "unsupported_rows": sum(row.get("window_audit_status") != "ready" for row in participant_rows.get(sid, ())),
            }
            for sid in subject_ids
        },
        "source_identity_verified_for_all_selected": source_identity_verified,
        "source_hash_stable_for_all_selected": source_hash_stable,
        "config_hash_stable_for_all_selected": config_hash_stable,
        "positions_hash_stable_for_all_selected": positions_hash_stable,
        "complete_status_accounting": complete_accounting,
        "engineering_status": "PASS" if engineering_pass else "HOLD",
        "scientific_status": "HOLD",
        "scientific_claims": [],
        "participant_level_summary_sha256": sha256_file(run_root / "participant_level_summary.json"),
        "selected_sources_audit_sha256": sha256_file(run_root / "selected_sources_audit.json"),
        "measurements_long_csv_sha256": sha256_file(run_root / "measurements_long.csv"),
        "output_file_sha256": output_hashes,
        "runtime": runtime,
        "pipeline_export_integration": "not_performed",
        "jacobian_or_manifold_inference": "not_performed",
        "all_results_qualified_false": all(
            _mapping(_mapping(_mapping(row.get("measurements")).get(family)).get("measurement_record")).get("qualified") is False
            for row in all_rows for family in ("ED", "OI", "NIE")
        ),
        "claim_status": "no_biological_claim",
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    print(run(source_root=args.source_root, selection_path=args.selection, output_root=args.output_root, run_id=args.run_id))

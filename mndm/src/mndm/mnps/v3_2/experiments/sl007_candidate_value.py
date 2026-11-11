"""Bounded, frozen NIE and OI candidate-value analyses authorized by SL-007."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[6]
for _source_root in (REPO_ROOT / "core" / "src", REPO_ROOT / "mndm" / "src"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency  # noqa: E402
from mndm.mnps.v3_2.oi import compute_oi_matrix  # noqa: E402
from mndm.mnps.v3_2.experiments.vector_pilot_001 import (  # noqa: E402
    NIE_EDGE_ESTIMATOR_ID,
    NIE_EDGE_RULE_ID,
    labels_hashes,
    sha256_array,
    sha256_file,
)


PROTOCOL_PATH = REPO_ROOT / "docs" / "api" / "mnps_v3_2_sl_007_protocols.rst"
SL006_ROOT = REPO_ROOT / "outputs" / "VECTOR-FEASIBILITY-002" / "run_20261013T_sl006_vectorfeasibility002_final"
DEFAULT_NIE_OUTPUT = REPO_ROOT / "outputs" / "NIE-INCREMENTAL-001"
DEFAULT_OI_OUTPUT = REPO_ROOT / "outputs" / "OI-TEMPORAL-SPECIFICITY-001"
EXPECTED_SL006_MANIFEST_SHA256 = "b10f582844c49614d0238f6303947610345fea34f25ec79dc50f7a251a88d702"
EXPECTED_SL006_PROTOCOL_SHA256 = "422976eeaddcee4349aa8ee01d6badf5a5707470394a1045578eb4477521ce0c"
EXPECTED_SL007_PROTOCOL_SHA256 = "618edccd529a75b75f3fd5853a89aa918d4e9a579077a2e42d26e32800a5a82a"
NIE_RANDOMIZATIONS = 49
IAAFT_REPLICATES = 2
IAAFT_ITERATIONS = 50
SAMPLE_RATE_HZ = 250.0
SPECTRAL_BANDS_HZ = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "gamma_30_45": (30.0, 45.0),
}
AUTOCORRELATION_LAGS = (1, 25, 125)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected_json_object:{path}")
    return payload


def _seed(identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:16], "big")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 2:
        return None
    x = x[keep] - np.mean(x[keep])
    y = y[keep] - np.mean(y[keep])
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return None
    value = float(np.dot(x, y) / denominator)
    return value if math.isfinite(value) else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_run_root(parent: Path, run_id: str) -> Path:
    root = (parent / run_id).resolve()
    if root.exists():
        raise FileExistsError(f"refusing_to_overwrite:{root}")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _verify_sl006_artifacts(base_root: Path, artifact_hashes: Mapping[str, Any]) -> None:
    resolved_root = base_root.resolve()
    for relative_name, expected_hash in artifact_hashes.items():
        artifact_path = (resolved_root / str(relative_name)).resolve()
        try:
            artifact_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"sl006_artifact_path_escapes_run_root:{relative_name}") from error
        if not artifact_path.is_file():
            raise FileNotFoundError(f"sl006_artifact_missing:{relative_name}")
        if sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"sl006_artifact_hash_mismatch:{relative_name}")


def _validate_sl006(base_root: Path) -> dict[str, Any]:
    manifest_path = base_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SL-006 manifest missing: {manifest_path}")
    if sha256_file(manifest_path) != EXPECTED_SL006_MANIFEST_SHA256:
        raise ValueError("sl006_manifest_hash_mismatch")
    manifest = _json(manifest_path)
    if manifest.get("protocol_sha256") != EXPECTED_SL006_PROTOCOL_SHA256:
        raise ValueError("sl006_protocol_hash_mismatch")
    if manifest.get("engineering_status") != "PASS":
        raise ValueError("sl006_run_not_engineering_pass")
    if manifest.get("selected_subject_ids_in_draw_order") is None:
        raise ValueError("sl006_selection_missing")
    artifact_hashes = manifest.get("output_file_sha256")
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise ValueError("sl006_output_artifact_hashes_missing")
    _verify_sl006_artifacts(base_root, artifact_hashes)
    return manifest


def _runtime_dependencies() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0], "numpy": np.__version__}
    for package in ("scipy", "mne", "pyedflib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _estimator_source_hashes() -> dict[str, str]:
    source_files = {
        "nie_kernel": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "nie.py",
        "oi_kernel": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "oi.py",
        "shared_experiment_helpers": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "experiments" / "vector_pilot_001.py",
        "audited_eeg_loader": REPO_ROOT / "mndm" / "src" / "mndm" / "mnps" / "v3_1" / "eeg_windows.py",
        "preprocessing": REPO_ROOT / "mndm" / "src" / "mndm" / "preprocess.py",
    }
    return {name: sha256_file(path) for name, path in source_files.items()}


def _result_value(result: Any) -> float:
    if not getattr(result, "valid", False) or getattr(result, "value", None) is None:
        raise ValueError(f"nie_kernel_unsupported:{getattr(result, 'reason', None)}")
    return float(result.value)


def _permute_edge_positions(weights: np.ndarray, seed: int) -> np.ndarray:
    """Move an undirected edge multiset across positions without changing it."""
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.array_equal(matrix, matrix.T):
        raise ValueError("edge_position_control_requires_symmetric_square_matrix")
    upper_rows, upper_cols = np.triu_indices(matrix.shape[0], k=1)
    edge_multiset = matrix[upper_rows, upper_cols]
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = rng.permutation(edge_multiset)
    result = np.zeros_like(matrix)
    result[upper_rows, upper_cols] = shuffled
    result[upper_cols, upper_rows] = shuffled
    if not np.array_equal(np.sort(result[upper_rows, upper_cols]), np.sort(edge_multiset)):
        raise RuntimeError("exact_edge_multiset_preservation_failed")
    return result


def run_nie_incremental(
    *,
    base_root: Path = SL006_ROOT,
    output_parent: Path = DEFAULT_NIE_OUTPUT,
    run_id: str | None = None,
) -> Path:
    """Run exact-multiset graph-position controls and retain all source rows."""
    base_root = base_root.resolve()
    base_manifest = _validate_sl006(base_root)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    if protocol_hash != EXPECTED_SL007_PROTOCOL_SHA256:
        raise ValueError("nie_protocol_hash_mismatch")
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_sl007_nie_incremental001")
    run_root = _prepare_run_root(output_parent, run_id)
    participants = list(base_manifest["selected_subject_ids_in_draw_order"])
    window_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    per_participant: dict[str, list[dict[str, Any]]] = {str(sid): [] for sid in participants}
    upper_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for subject_id in participants:
        participant = _json(base_root / str(subject_id) / "participant_results.json")
        labels = tuple(str(label) for label in participant.get("ordered_channel_labels", ()))
        order_hash, _ = labels_hashes(labels)
        for row in participant.get("rows", ()):
            base_row = {
                "subject_id": subject_id,
                "window_id": row.get("window_id"),
                "window_index": row.get("window_index"),
                "window_audit_status": row.get("window_audit_status"),
                "input_matrix_sha256": row.get("input_matrix_sha256_little_endian_float64"),
                "edge_matrix_sha256": None,
                "channel_count": None,
                "metric_status": None,
                "metric_reason": None,
                "observed_nie": None,
                "mean_absolute_correlation": None,
                "delta_path": None,
                "control_count": 0,
                "control_delta_median": None,
                "control_delta_q025": None,
                "control_delta_q975": None,
                "observed_delta_fraction_control_le": None,
                "qualified": False,
                "claim_status": "no_biological_claim",
            }
            if row.get("window_audit_status") != "ready":
                base_row.update({
                    "metric_status": "unsupported",
                    "metric_reason": row.get("window_audit_reason") or "sl006_window_unsupported",
                })
                window_rows.append(base_row)
                continue
            measurement = row.get("measurements", {}).get("NIE", {})
            raw_record = measurement.get("matrix_kernel_result") or {}
            observed_nie = _finite_float(raw_record.get("value"))
            if observed_nie is None or not raw_record.get("valid"):
                base_row.update({
                    "metric_status": "unsupported",
                    "metric_reason": raw_record.get("reason") or measurement.get("reason") or "sl006_nie_value_invalid",
                })
                window_rows.append(base_row)
                continue
            edge_audit = measurement.get("edge_audit", {})
            matrix_relpath = edge_audit.get("matrix_path")
            expected_file_hash = edge_audit.get("matrix_sha256")
            if not matrix_relpath or not expected_file_hash:
                raise ValueError(f"missing_edge_matrix_audit:{subject_id}:{row.get('window_id')}")
            matrix_path = (base_root / str(matrix_relpath)).resolve()
            if sha256_file(matrix_path) != expected_file_hash:
                raise ValueError(f"edge_matrix_file_hash_mismatch:{subject_id}:{row.get('window_id')}")
            weights = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
            if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or not np.array_equal(weights, weights.T):
                raise ValueError(f"edge_matrix_shape_or_symmetry_mismatch:{subject_id}:{row.get('window_id')}")
            node_count = int(weights.shape[0])
            upper_rows, upper = upper_cache.setdefault(node_count, np.triu_indices(node_count, k=1))
            original_edge_multiset = weights[upper_rows, upper]
            mean_abs = float(np.mean(original_edge_multiset))
            stored_strength = _finite_float(
                row.get("covariance_connectivity_diagnostics", {}).get("values", {}).get("mean_absolute_correlation")
            )
            if stored_strength is None or not math.isclose(mean_abs, stored_strength, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"mean_absolute_strength_mismatch:{subject_id}:{row.get('window_id')}")
            calculated = compute_nie_raw_weighted_global_efficiency(
                weights,
                node_labels=labels,
                node_order_sha256=order_hash,
                edge_estimator_id=NIE_EDGE_ESTIMATOR_ID,
                edge_rule_id=NIE_EDGE_RULE_ID,
            )
            recalculated_nie = _result_value(calculated)
            if not math.isclose(recalculated_nie, observed_nie, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"stored_nie_recalculation_mismatch:{subject_id}:{row.get('window_id')}")
            observed_delta = recalculated_nie - mean_abs
            shuffled_nie_values: list[float] = []
            for permutation_index in range(NIE_RANDOMIZATIONS):
                control_seed = _seed(f"NIE-INCREMENTAL-001|{subject_id}|{row['window_id']}|{permutation_index}")
                permuted = _permute_edge_positions(weights, control_seed)
                control = compute_nie_raw_weighted_global_efficiency(
                    permuted,
                    node_labels=labels,
                    node_order_sha256=order_hash,
                    edge_estimator_id=NIE_EDGE_ESTIMATOR_ID,
                    edge_rule_id=NIE_EDGE_RULE_ID,
                )
                control_nie = _result_value(control)
                shuffled_nie_values.append(control_nie)
                null_rows.append({
                    "subject_id": subject_id,
                    "window_id": row["window_id"],
                    "permutation_index": permutation_index,
                "seed_sha256": hashlib.sha256(f"NIE-INCREMENTAL-001|{subject_id}|{row['window_id']}|{permutation_index}".encode("utf-8")).hexdigest(),
                    "control_nie": control_nie,
                    "control_delta_path": control_nie - mean_abs,
                    "mean_absolute_correlation_preserved": mean_abs,
                    "edge_count_upper_triangle": len(original_edge_multiset),
                })
            null_delta = np.asarray(shuffled_nie_values, dtype=np.float64) - mean_abs
            row_result = {
                **base_row,
                "subject_id": subject_id,
                "window_id": row["window_id"],
                "window_index": int(row["window_index"]),
                "input_matrix_sha256": row.get("input_matrix_sha256_little_endian_float64"),
                "edge_matrix_sha256": expected_file_hash,
                "channel_count": node_count,
                "metric_status": "computed_exploratory",
                "metric_reason": None,
                "observed_nie": recalculated_nie,
                "mean_absolute_correlation": mean_abs,
                "delta_path": observed_delta,
                "control_count": NIE_RANDOMIZATIONS,
                "control_delta_median": float(np.median(null_delta)),
                "control_delta_q025": float(np.quantile(null_delta, 0.025)),
                "control_delta_q975": float(np.quantile(null_delta, 0.975)),
                "observed_delta_fraction_control_le": float(np.mean(null_delta <= observed_delta)),
                "qualified": False,
                "claim_status": "no_biological_claim",
            }
            window_rows.append(row_result)
            per_participant[str(subject_id)].append(row_result)

    participant_summary: list[dict[str, Any]] = []
    for subject_id in participants:
        rows = per_participant[str(subject_id)]
        summary = {"subject_id": subject_id, "window_count": len(rows)}
        for metric in ("observed_nie", "mean_absolute_correlation", "delta_path", "control_delta_median"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            summary[f"{metric}_median"] = float(np.median(values)) if len(values) else None
            summary[f"{metric}_q25"] = float(np.quantile(values, 0.25)) if len(values) else None
            summary[f"{metric}_q75"] = float(np.quantile(values, 0.75)) if len(values) else None
        participant_summary.append(summary)
    cross_correlations: dict[str, Any] = {}
    for metric in ("observed_nie_median", "mean_absolute_correlation_median", "delta_path_median", "control_delta_median_median"):
        cross_correlations[metric] = {}
        for other in ("observed_nie_median", "mean_absolute_correlation_median", "delta_path_median", "control_delta_median_median"):
            matched = [row for row in participant_summary if row[metric] is not None and row[other] is not None]
            cross_correlations[metric][other] = {
                "pearson_r": _pearson([row[metric] for row in matched], [row[other] for row in matched]),
                "n_participants": len(matched),
                "subject_ids": [row["subject_id"] for row in matched],
            }
    _write_csv(run_root / "window_results.csv", window_rows)
    _write_csv(run_root / "edge_position_controls.csv", null_rows)
    summary_payload = {
        "schema": "mndm.mnps_v3_2.nie_incremental_001_summary.v1",
        "participant_summaries": participant_summary,
        "participant_median_correlations": cross_correlations,
        "window_count": len(window_rows),
        "unsupported_or_invalid_window_count": sum(row.get("metric_status") == "unsupported" for row in window_rows),
        "metric_valid_window_count": sum(row.get("metric_status") == "computed_exploratory" for row in window_rows),
        "inference_boundary": "descriptive_only_no_p_values_no_qualification",
        "qualified": False,
        "claim_status": "no_biological_claim",
    }
    (run_root / "participant_summary.json").write_text(json.dumps(summary_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {str(path.relative_to(run_root)).replace("\\", "/"): sha256_file(path) for path in sorted(run_root.rglob("*")) if path.is_file()}
    run_manifest = {
        "schema": "mndm.mnps_v3_2.nie_incremental_001_run_manifest.v1",
        "experiment_id": "NIE-INCREMENTAL-001",
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "sl006_manifest_sha256": sha256_file(base_root / "run_manifest.json"),
        "sl006_source_run_id": base_manifest.get("run_id"),
        "selected_subject_ids": participants,
        "nominal_window_count": len(window_rows),
        "computed_window_count": sum(row.get("metric_status") == "computed_exploratory" for row in window_rows),
        "unsupported_or_invalid_window_count": sum(row.get("metric_status") == "unsupported" for row in window_rows),
        "randomizations_per_window": NIE_RANDOMIZATIONS,
        "seed_derivation": "PCG64; first_128_sha256_bits_of_protocol_subject_window_permutation_index",
        "weight_multiset_preservation_checked": True,
        "runner_sha256": sha256_file(Path(__file__)),
        "estimator_and_loader_source_sha256": _estimator_source_hashes(),
        "dependency_versions": _runtime_dependencies(),
        "output_file_sha256": hashes,
        "qualified": False,
        "claim_status": "no_biological_claim",
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


def _iaaft_surrogate(signal: np.ndarray, seed: int, iterations: int = IAAFT_ITERATIONS) -> tuple[np.ndarray, float]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 5 or not np.isfinite(values).all():
        raise ValueError("iaaft_requires_finite_one_dimensional_signal_with_five_samples")
    target_sorted = np.sort(values, kind="mergesort")
    target_amplitudes = np.abs(np.fft.rfft(values))
    target_power = target_amplitudes * target_amplitudes
    rng = np.random.Generator(np.random.PCG64(seed))
    surrogate = rng.permutation(values)
    for _ in range(iterations):
        spectrum = np.fft.rfft(surrogate)
        phases = np.exp(1j * np.angle(spectrum))
        adjusted = np.fft.irfft(target_amplitudes * phases, n=values.size)
        rank_order = np.argsort(adjusted, kind="mergesort")
        next_surrogate = np.empty_like(adjusted)
        next_surrogate[rank_order] = target_sorted
        surrogate = next_surrogate
    achieved_amplitudes = np.abs(np.fft.rfft(surrogate))
    achieved_power = achieved_amplitudes * achieved_amplitudes
    denominator = float(np.linalg.norm(target_power))
    spectral_error = float(np.linalg.norm(achieved_power - target_power) / denominator) if denominator > 0 else 0.0
    if not np.array_equal(np.sort(surrogate, kind="mergesort"), target_sorted):
        raise RuntimeError("iaaft_amplitude_multiset_not_preserved")
    return surrogate, spectral_error


def _conventional_features(signals: np.ndarray) -> dict[str, float]:
    from scipy.signal import welch

    values_uv = np.asarray(signals, dtype=np.float64) * 1e6
    frequencies, density = welch(
        values_uv, fs=SAMPLE_RATE_HZ, window="hann", nperseg=1000, noverlap=500,
        nfft=1000, detrend="constant", return_onesided=True, scaling="density", axis=-1,
    )
    bin_width = float(frequencies[1] - frequencies[0])
    result: dict[str, float] = {}
    for name, (low, high) in SPECTRAL_BANDS_HZ.items():
        mask = (frequencies >= low) & (frequencies < high)
        channel_power = np.sum(density[:, mask], axis=1) * bin_width
        result[f"welch_power_{name}_uv2"] = float(np.median(channel_power))
    entropy_mask = (frequencies >= 1.0) & (frequencies < 45.0)
    spectral = np.maximum(density[:, entropy_mask], 0.0)
    sums = np.sum(spectral, axis=1, keepdims=True)
    probability = np.divide(spectral, sums, out=np.zeros_like(spectral), where=sums > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        channel_entropy = -np.sum(np.where(probability > 0.0, probability * np.log(probability), 0.0), axis=1)
    channel_entropy = channel_entropy / math.log(int(np.count_nonzero(entropy_mask)))
    result["normalized_spectral_entropy_1_45"] = float(np.median(channel_entropy))
    centered = np.asarray(signals, dtype=np.float64) - np.mean(signals, axis=1, keepdims=True)
    denominator = np.sum(centered * centered, axis=1)
    for lag in AUTOCORRELATION_LAGS:
        numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        values = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.0)
        finite = values[np.isfinite(values)]
        result[f"autocorrelation_lag_{lag}_samples"] = float(np.median(finite)) if finite.size else math.nan
    return result


def run_oi_temporal_specificity(
    *,
    source_root: Path = Path("K:/ExternalReceivedDatasets/ANPHY"),
    base_root: Path = SL006_ROOT,
    selection_path: Path = REPO_ROOT / "mndm" / "config" / "mnps_v3_2_vector_feasibility_002_selection_20261013.json",
    output_parent: Path = DEFAULT_OI_OUTPUT,
    run_id: str | None = None,
) -> Path:
    """Reconstruct and hash-check SL-006 signals, then run fixed OI controls."""
    base_root = base_root.resolve()
    source_root = source_root.resolve()
    base_manifest = _validate_sl006(base_root)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    if protocol_hash != EXPECTED_SL007_PROTOCOL_SHA256:
        raise ValueError("oi_protocol_hash_mismatch")
    if sha256_file(selection_path) != base_manifest.get("selection_sha256"):
        raise ValueError("frozen_selection_manifest_hash_mismatch")
    selection = _json(selection_path)
    selected_ids = tuple(str(sid) for sid in selection.get("random_selection", {}).get("selected_subject_ids_in_draw_order", ()))
    if list(selected_ids) != list(base_manifest["selected_subject_ids_in_draw_order"]):
        raise ValueError("selected_ids_do_not_match_sl006")
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_sl007_oi_specificity001")
    run_root = _prepare_run_root(output_parent, run_id)
    from mndm.mnps.v3_1 import load_audited_edf_eeg_grid

    positions_path = source_root / "Co-registered average positions.pos"
    if sha256_file(positions_path) != base_manifest.get("positions_sha256"):
        raise ValueError("frozen_positions_hash_mismatch")
    roster = {
        str(row.get("subject_id")): row
        for row in selection.get("source_roster_metadata_only", ())
        if isinstance(row, Mapping)
    }
    window_rows: list[dict[str, Any]] = []
    participant_summary: list[dict[str, Any]] = []
    inputs_audited = 0
    for subject_id in selected_ids:
        frozen = roster.get(subject_id)
        if not frozen or not frozen.get("relative_path"):
            raise ValueError(f"frozen_source_path_missing:{subject_id}")
        source_path = (source_root / str(frozen["relative_path"])).resolve()
        participant = _json(base_root / subject_id / "participant_results.json")
        overlay_path = base_root / "config_snapshots" / f"{subject_id}_e1_g2_overlay.yaml"
        expected_overlay_hash = participant.get("config_sha256")
        if not overlay_path.is_file() or sha256_file(overlay_path) != expected_overlay_hash:
            raise ValueError(f"frozen_overlay_hash_mismatch:{subject_id}")
        expected_source_hash = participant.get("source_sha256_before_read")
        if not expected_source_hash or sha256_file(source_path) != expected_source_hash:
            raise ValueError(f"frozen_source_hash_mismatch:{subject_id}")
        grid = load_audited_edf_eeg_grid(source_path, positions_path, overlay_path)
        if len(grid.windows) != len(participant.get("rows", ())):
            raise ValueError(f"window_count_mismatch:{subject_id}")
        labels = tuple(str(label) for label in participant.get("ordered_channel_labels", ()))
        if tuple(str(label) for label in grid.windows[0].channel_names) != labels:
            raise ValueError(f"channel_order_mismatch:{subject_id}")
        subject_rows: list[dict[str, Any]] = []
        for window, frozen_row in zip(grid.windows, participant["rows"], strict=True):
            if frozen_row.get("window_audit_status") != "ready":
                window_rows.append({
                    "subject_id": subject_id,
                    "window_id": frozen_row.get("window_id"),
                    "window_index": frozen_row.get("window_index"),
                    "input_audit_status": "source_window_unsupported",
                    "input_audit_reason": frozen_row.get("window_audit_reason"),
                    "oi": None,
                    "qualified": False,
                })
                continue
            signals = np.asarray(window.signals, dtype=np.float64)
            observed_input_hash = sha256_array(signals)
            expected_input_hash = frozen_row.get("input_matrix_sha256_little_endian_float64")
            if observed_input_hash != expected_input_hash:
                window_rows.append({
                    "subject_id": subject_id,
                    "window_id": frozen_row.get("window_id"),
                    "window_index": frozen_row.get("window_index"),
                    "input_audit_status": "hash_mismatch",
                    "input_audit_reason": "reconstructed_matrix_does_not_match_sl006",
                    "observed_input_sha256": observed_input_hash,
                    "expected_input_sha256": expected_input_hash,
                    "oi": None,
                    "qualified": False,
                })
                continue
            inputs_audited += 1
            original = compute_oi_matrix(signals, labels)
            original_value = _finite_float(original.value) if original.valid else None
            features = _conventional_features(signals)
            surrogate_values: list[float] = []
            spectral_errors: list[float] = []
            surrogate_reasons: list[str] = []
            for replicate in range(IAAFT_REPLICATES):
                surrogate_signals = np.empty_like(signals)
                channel_errors: list[float] = []
                for channel_index, channel_label in enumerate(labels):
                    seed_identity = f"OI-TEMPORAL-SPECIFICITY-001|{subject_id}|{frozen_row['window_id']}|{channel_label}|{replicate}"
                    surrogate_signals[channel_index], channel_error = _iaaft_surrogate(
                        signals[channel_index], _seed(seed_identity), IAAFT_ITERATIONS
                    )
                    channel_errors.append(channel_error)
                surrogate_result = compute_oi_matrix(surrogate_signals, labels)
                if not surrogate_result.valid or surrogate_result.value is None:
                    surrogate_values.append(math.nan)
                    surrogate_reasons.append(str(surrogate_result.reason or "iaaft_oi_invalid"))
                else:
                    surrogate_values.append(float(surrogate_result.value))
                spectral_errors.append(float(np.median(channel_errors)))
            finite_surrogate_values = [value for value in surrogate_values if math.isfinite(value)]
            mean_surrogate_oi = float(np.mean(finite_surrogate_values)) if finite_surrogate_values else None
            result = {
                "subject_id": subject_id,
                "window_id": frozen_row["window_id"],
                "window_index": int(frozen_row["window_index"]),
                "nominal_start_sec": frozen_row["nominal_window_sec_half_open"][0],
                "nominal_end_sec": frozen_row["nominal_window_sec_half_open"][1],
                "input_audit_status": "matched_sl006_matrix_hash",
                "input_matrix_sha256": observed_input_hash,
                "source_sha256": frozen_row.get("source_sha256"),
                "metric_status": "computed_exploratory" if original_value is not None else "unsupported",
                "metric_reason": None if original_value is not None else str(original.reason or "oi_kernel_invalid"),
                "oi": original_value,
                "mean_surrogate_oi": mean_surrogate_oi,
                "surrogate_oi_values_json": json.dumps([value if math.isfinite(value) else None for value in surrogate_values], separators=(",", ":")),
                "surrogate_status": "computed_exploratory" if not surrogate_reasons else "partial",
                "surrogate_reason": ";".join(surrogate_reasons) if surrogate_reasons else None,
                "surrogate_oi_count": len(finite_surrogate_values),
                "oi_minus_mean_surrogate": float(original_value - mean_surrogate_oi) if original_value is not None and mean_surrogate_oi is not None else None,
                "median_surrogate_spectral_relative_l2_error": float(np.median(spectral_errors)),
                "iaaft_replicates": IAAFT_REPLICATES,
                "iaaft_iterations": IAAFT_ITERATIONS,
                **features,
                "qualified": False,
                "claim_status": "no_biological_claim",
            }
            window_rows.append(result)
            subject_rows.append(result)
        participant_summary.append(_participant_oi_summary(subject_id, subject_rows))

    metrics = [
        "oi_median", "mean_surrogate_oi_median", "oi_minus_mean_surrogate_median",
        "normalized_spectral_entropy_1_45_median",
        *(f"welch_power_{band}_uv2_median" for band in SPECTRAL_BANDS_HZ),
        *(f"autocorrelation_lag_{lag}_samples_median" for lag in AUTOCORRELATION_LAGS),
    ]
    associations: dict[str, Any] = {}
    for left in metrics:
        associations[left] = {}
        for right in metrics:
            matched = [row for row in participant_summary if row.get(left) is not None and row.get(right) is not None]
            associations[left][right] = {
                "pearson_r": _pearson([row[left] for row in matched], [row[right] for row in matched]),
                "n_participants": len(matched),
                "subject_ids": [row["subject_id"] for row in matched],
            }
    _write_csv(run_root / "window_results.csv", window_rows)
    summary_payload = {
        "schema": "mndm.mnps_v3_2.oi_temporal_specificity_001_summary.v1",
        "participant_summaries": participant_summary,
        "participant_median_correlations": associations,
        "audited_reconstructed_signal_matrices": inputs_audited,
        "nominal_window_rows": len(window_rows),
        "supported_sl006_windows": sum(row.get("input_audit_status") == "matched_sl006_matrix_hash" for row in window_rows),
        "oi_valid_windows": sum(row.get("metric_status") == "computed_exploratory" for row in window_rows),
        "oi_unsupported_windows": sum(row.get("metric_status") == "unsupported" for row in window_rows),
        "surrogate_incomplete_windows": sum(row.get("surrogate_status") == "partial" for row in window_rows),
        "hash_mismatch_windows": sum(row.get("input_audit_status") == "hash_mismatch" for row in window_rows),
        "inference_boundary": "descriptive_only_no_p_values_no_qualification",
        "qualified": False,
        "claim_status": "no_biological_claim",
    }
    (run_root / "participant_summary.json").write_text(json.dumps(summary_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_hashes = {str(path.relative_to(run_root)).replace("\\", "/"): sha256_file(path) for path in sorted(run_root.rglob("*")) if path.is_file()}
    run_manifest = {
        "schema": "mndm.mnps_v3_2.oi_temporal_specificity_001_run_manifest.v1",
        "experiment_id": "OI-TEMPORAL-SPECIFICITY-001",
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "sl006_manifest_sha256": sha256_file(base_root / "run_manifest.json"),
        "sl006_source_run_id": base_manifest.get("run_id"),
        "source_root": str(source_root),
        "selected_subject_ids": list(selected_ids),
        "reconstructed_input_matrices_hash_checked": inputs_audited,
        "iaaft_replicates_per_channel_window": IAAFT_REPLICATES,
        "iaaft_iterations": IAAFT_ITERATIONS,
        "iaaft_spectral_error": "relative_l2_difference_of_power_spectrum_abs_rfft_squared",
        "runner_sha256": sha256_file(Path(__file__)),
        "estimator_and_loader_source_sha256": _estimator_source_hashes(),
        "dependency_versions": _runtime_dependencies(),
        "seed_derivation": "PCG64; first_128_sha256_bits_of_protocol_subject_window_channel_surrogate_index",
        "conventional_contract": {
            "welch": "hann_1000_samples_overlap500_nfft1000_detrend_constant_density_250Hz",
            "spectral_entropy_band_hz": [1.0, 45.0],
            "autocorrelation_lags_samples": list(AUTOCORRELATION_LAGS),
        },
        "output_file_sha256": output_hashes,
        "qualified": False,
        "claim_status": "no_biological_claim",
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


def _participant_oi_summary(subject_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        "oi", "mean_surrogate_oi", "oi_minus_mean_surrogate",
        "normalized_spectral_entropy_1_45",
        *(f"welch_power_{band}_uv2" for band in SPECTRAL_BANDS_HZ),
        *(f"autocorrelation_lag_{lag}_samples" for lag in AUTOCORRELATION_LAGS),
    ]
    summary: dict[str, Any] = {"subject_id": subject_id, "supported_window_count": len(rows)}
    for metric in metrics:
        values = np.asarray([value for row in rows if (value := _finite_float(row.get(metric))) is not None], dtype=np.float64)
        summary[f"{metric}_median"] = float(np.median(values)) if values.size else None
        summary[f"{metric}_q25"] = float(np.quantile(values, 0.25)) if values.size else None
        summary[f"{metric}_q75"] = float(np.quantile(values, 0.75)) if values.size else None
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    nie_parser = subparsers.add_parser("nie-incremental")
    nie_parser.add_argument("--base-root", type=Path, default=SL006_ROOT)
    nie_parser.add_argument("--output-parent", type=Path, default=DEFAULT_NIE_OUTPUT)
    nie_parser.add_argument("--run-id", type=str, default=None)
    oi_parser = subparsers.add_parser("oi-temporal-specificity")
    oi_parser.add_argument("--source-root", type=Path, default=Path("K:/ExternalReceivedDatasets/ANPHY"))
    oi_parser.add_argument("--base-root", type=Path, default=SL006_ROOT)
    oi_parser.add_argument("--output-parent", type=Path, default=DEFAULT_OI_OUTPUT)
    oi_parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    if args.experiment == "nie-incremental":
        result = run_nie_incremental(base_root=args.base_root, output_parent=args.output_parent, run_id=args.run_id)
    else:
        result = run_oi_temporal_specificity(source_root=args.source_root, base_root=args.base_root, output_parent=args.output_parent, run_id=args.run_id)
    print(result)


if __name__ == "__main__":
    _main()


__all__ = ["_conventional_features", "_iaaft_surrogate", "run_nie_incremental", "run_oi_temporal_specificity"]

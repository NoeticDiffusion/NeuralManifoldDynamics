"""Summarize existing MNPS 3.4 HDF5 exports by condition and participant.

This read-only audit characterizes estimator support and feature dependency. It
does not reprocess EEG, infer artifact-free data, or qualify the measurements.
Overlapping windows are summarized within participant before dependency
statistics are calculated.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


COORDINATE_ORDER = (
    "delta_theta_level",
    "alpha_level",
    "beta_alpha_ratio",
    "gamma_level",
    "hjorth_mobility",
    "alpha_theta_ratio",
    "permutation_entropy",
    "alpha_centroid_hz",
    "alpha_envelope_cv",
)
SOURCE_FEATURE_COUNT = 10
CORE_SCHEMA = {
    "raw_features/values": ("[W,F=10]", "float64"),
    "raw_features/support": ("[W,F=10]", "uint8"),
    "feature_scaled_values/values": ("[W,F=10]", "float64"),
    "raw_coordinates/values": ("[W,D=9]", "float64"),
    "raw_coordinates/support": ("[W,D=9]", "bool"),
    "coordinate_pre_reference_values": ("[W,D=9]", "float64"),
    "coordinates/values": ("[W,D=9]", "float64"),
    "coordinate_validity_mask": ("[W,D=9]", "uint8"),
    "complete_9d_mask": ("[W]", "uint8"),
    "window_start_sec": ("[W]", "float64"),
    "window_end_sec": ("[W]", "float64"),
    "support/qc/qc_ok_eeg": ("[W]", "int8"),
    "support/channel_labels": ("[C]", "string"),
    "per_channel_candidates/alpha_centroid_hz": ("[W,C]", "float64"),
    "per_channel_candidates/alpha_envelope_cv": ("[W,C]", "float64"),
    "qualified": ("scalar", "bool"),
}
OPTIONAL_SCHEMA = {
    "support/eeg_psd_method_id": ("[W]", "string"),
    "support/eeg_psd_fallback_reason": ("[W]", "string"),
}
HISTORICAL_ROOT_PATHS = ("mnps_3d", "coords_9d")


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read(h5: h5py.File, name: str) -> np.ndarray:
    return np.asarray(h5[name])


def _condition_and_subject(path: Path) -> tuple[str, str]:
    text = str(path).lower()
    if "eyesclosed" in text or "eyes_closed" in text:
        condition = "eyes_closed"
    elif "eyesopen" in text or "eyes_open" in text:
        condition = "eyes_open"
    else:
        raise ValueError(f"Cannot identify eyes condition from HDF5 path: {path}")
    match = re.search(r"(sub-[^_\\/]+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot identify participant from HDF5 path: {path}")
    return condition, match.group(1)


def _effective_rank(eigenvalues: np.ndarray) -> tuple[float, float]:
    positive = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
    total = float(positive.sum())
    if total <= 0:
        return float("nan"), float("nan")
    p = positive[positive > 0] / total
    spectral_entropy_rank = float(np.exp(-np.sum(p * np.log(p))))
    participation_ratio = float(total**2 / np.sum(positive**2))
    return spectral_entropy_rank, participation_ratio


def _process_file(path: Path) -> dict[str, Any]:
    condition, subject = _condition_and_subject(path)
    with h5py.File(path, "r") as h5:
        historical_root_paths_present = {
            name: name in h5 for name in HISTORICAL_ROOT_PATHS
        }
        names = [_decode(value) for value in _read(h5, "measurements/v3_4/coordinate_names")]
        if tuple(names) != COORDINATE_ORDER:
            raise ValueError(f"Unexpected v3.4 coordinate order in {path}: {names}")
        raw_support = _read(h5, "measurements/v3_4/raw_coordinates/support").astype(bool)
        valid = _read(h5, "measurements/v3_4/coordinate_validity_mask").astype(bool)
        complete = _read(h5, "measurements/v3_4/complete_9d_mask").astype(bool)
        values = _read(h5, "measurements/v3_4/coordinates/values").astype(float)
        qualified = bool(_read(h5, "measurements/v3_4/qualified").item())
        raw_feature_names = [
            _decode(value) for value in _read(h5, "measurements/v3_4/raw_features/names")
        ]
        raw_feature_support = _read(h5, "measurements/v3_4/raw_features/support").astype(bool)
        anchor_status = _decode(_read(h5, "measurements/v3_4/provenance/anchor_application_status").item())
        qc = _read(h5, "measurements/v3_4/support/qc/qc_ok_eeg").astype(int)
        degraded = _read(h5, "measurements/v3_4/support/eeg_entropy_degraded_mode").astype(bool)

        schema = {}
        group = h5["measurements/v3_4"]
        for relative_path, (shape_contract, dtype_contract) in CORE_SCHEMA.items():
            dataset = group[relative_path]
            if h5py.check_string_dtype(dataset.dtype) is not None:
                dtype = "string"
            else:
                dtype = str(dataset.dtype)
            if dtype != dtype_contract:
                raise ValueError(
                    f"Unexpected dtype for measurements/v3_4/{relative_path} in {path}: {dtype}"
                )
            if shape_contract == "[W,F=10]" and dataset.shape != (len(valid), SOURCE_FEATURE_COUNT):
                raise ValueError(f"Unexpected feature matrix shape in {path}: {relative_path} {dataset.shape}")
            if shape_contract == "[W,D=9]" and dataset.shape != (len(valid), len(COORDINATE_ORDER)):
                raise ValueError(f"Unexpected coordinate matrix shape in {path}: {relative_path} {dataset.shape}")
            if shape_contract == "[W]" and dataset.shape != (len(valid),):
                raise ValueError(f"Unexpected window vector shape in {path}: {relative_path} {dataset.shape}")
            if shape_contract == "[W,C]" and dataset.shape != (len(valid), len(_read(h5, "measurements/v3_4/support/channel_labels"))):
                raise ValueError(f"Unexpected channel matrix shape in {path}: {relative_path} {dataset.shape}")
            if shape_contract == "[C]" and dataset.ndim != 1:
                raise ValueError(f"Unexpected channel label shape in {path}: {dataset.shape}")
            if shape_contract == "scalar" and dataset.shape != ():
                raise ValueError(f"Unexpected scalar shape in {path}: {relative_path} {dataset.shape}")
            schema[relative_path] = {
                "shape_contract": shape_contract,
                "observed_shape": "x".join(str(size) for size in dataset.shape) or "scalar",
                "dtype": dtype,
            }
        for relative_path, (shape_contract, dtype_contract) in OPTIONAL_SCHEMA.items():
            if relative_path in group:
                dataset = group[relative_path]
                dtype = "string" if h5py.check_string_dtype(dataset.dtype) is not None else str(dataset.dtype)
                schema[relative_path] = {
                    "shape_contract": shape_contract,
                    "observed_shape": "x".join(str(size) for size in dataset.shape) or "scalar",
                    "dtype": dtype,
                }
        has_psd_method_id = "support/eeg_psd_method_id" in group
        has_psd_fallback_reason = "support/eeg_psd_fallback_reason" in group
        if has_psd_method_id != has_psd_fallback_reason:
            raise ValueError(f"Incomplete PSD estimator identity fields in {path}")
        if has_psd_method_id:
            psd_method_ids = [_decode(value) for value in _read(h5, "measurements/v3_4/support/eeg_psd_method_id")]
            psd_fallback_reasons = [_decode(value) for value in _read(h5, "measurements/v3_4/support/eeg_psd_fallback_reason")]
        else:
            psd_method_ids = ["not_recorded_in_export"] * len(valid)
            psd_fallback_reasons = ["not_recorded_in_export"] * len(valid)

        statuses: dict[str, list[str]] = {}
        for feature, dataset in (
            ("alpha_centroid_hz", "measurements/v3_4/support/alpha_centroid_status"),
            ("alpha_envelope_cv", "measurements/v3_4/support/alpha_envelope_status"),
        ):
            statuses[feature] = [_decode(v) for v in _read(h5, dataset)]

    if raw_support.shape != valid.shape or values.shape != valid.shape or complete.shape != (len(valid),):
        raise ValueError(f"Inconsistent v3.4 array shapes in {path}")
    values[~valid] = np.nan
    return {
        "path": path,
        "condition": condition,
        "subject": subject,
        "windows": int(valid.shape[0]),
        "raw_support": raw_support,
        "valid": valid,
        "complete": complete,
        "values": values,
        "qualified": qualified,
        "raw_feature_names": raw_feature_names,
        "raw_feature_support": raw_feature_support,
        "anchor_status": anchor_status,
        "qc": qc,
        "degraded": degraded,
        "candidate_statuses": statuses,
        "schema": schema,
        "psd_method_ids": psd_method_ids,
        "psd_fallback_reasons": psd_fallback_reasons,
        "historical_root_paths_present": historical_root_paths_present,
    }


def run(h5_root: Path, out_dir: Path, figure_out: Path | None = None) -> None:
    # The run directory also contains a v3.4-only measurement sidecar for one
    # recording. Audit one canonical combined HDF5 per BIDS recording instead
    # of counting that sidecar twice.
    files = sorted(path for path in h5_root.rglob("sub-*.h5") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {h5_root}")
    rows = [_process_file(path) for path in files]

    support_rows: list[dict[str, Any]] = []
    feature_support_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    reason_counts: Counter[tuple[str, str, str]] = Counter()
    participant_values_available: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in COORDINATE_ORDER}
    )
    participant_values_matched: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in COORDINATE_ORDER}
    )
    schema_forms: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        condition = row["condition"]
        subject = row["subject"]
        recording_rows.append({
            "condition": condition,
            "subject": subject,
            "recording": row["path"].stem,
            "n_windows": row["windows"],
            "raw_supported_coordinates": int(row["raw_support"].sum()),
            "final_valid_coordinates": int(row["valid"].sum()),
            "complete_9d_windows": int(row["complete"].sum()),
            "anchor_application_status": row["anchor_status"],
            "qualified": row["qualified"],
            "qc_ok_eeg_values": json.dumps(dict(Counter(map(int, row["qc"]))), sort_keys=True),
            "psd_method_ids": json.dumps(dict(Counter(row["psd_method_ids"])), sort_keys=True),
            "psd_fallback_reasons": json.dumps(dict(Counter(row["psd_fallback_reasons"])), sort_keys=True),
        })
        for name, details in row["schema"].items():
            schema_forms[name][(details["shape_contract"], details["dtype"])] += 1
        for k, name in enumerate(COORDINATE_ORDER):
            raw = row["raw_support"][:, k]
            final = row["valid"][:, k]
            support_rows.append({
                "condition": condition,
                "coordinate": name,
                "recording": row["path"].stem,
                "subject": subject,
                "n_windows": row["windows"],
                "raw_computable_windows": int(raw.sum()),
                "raw_computable_fraction": float(raw.mean()) if raw.size else float("nan"),
                "final_valid_windows": int(final.sum()),
                "final_valid_fraction": float(final.mean()) if final.size else float("nan"),
            })
            finite = final & np.isfinite(row["values"][:, k])
            participant_values_available[(condition, subject)][name].extend(
                row["values"][finite, k].tolist()
            )
            matched_finite = finite & row["complete"]
            participant_values_matched[(condition, subject)][name].extend(
                row["values"][matched_finite, k].tolist()
            )
        for j, name in enumerate(row["raw_feature_names"]):
            support = row["raw_feature_support"][:, j]
            feature_support_rows.append({
                "condition": condition,
                "source_feature": name,
                "recording": row["path"].stem,
                "subject": subject,
                "n_windows": row["windows"],
                "computable_windows": int(support.sum()),
                "computable_fraction": float(support.mean()) if support.size else float("nan"),
            })
        for feature, statuses in row["candidate_statuses"].items():
            for status, count in Counter(statuses).items():
                reason_counts[(condition, feature, status)] += count
        if row["degraded"].any():
            reason_counts[(condition, "permutation_entropy", "degraded_fallback")] += int(row["degraded"].sum())
        reason_counts[(condition, "anchor_application", row["anchor_status"])] += 1
        for qc_state, count in Counter(map(int, row["qc"])).items():
            reason_counts[(condition, "qc_ok_eeg", str(qc_state))] += count

    support = pd.DataFrame(support_rows)
    summary_rows = []
    for (condition, coordinate), group in support.groupby(["condition", "coordinate"], sort=False):
        summary_rows.append({
            "condition": condition,
            "coordinate": coordinate,
            "n_recordings": int(group["recording"].nunique()),
            "n_participants": int(group.loc[group["raw_computable_windows"] > 0, "subject"].nunique()),
            "n_windows": int(group["n_windows"].sum()),
            "raw_computable_windows": int(group["raw_computable_windows"].sum()),
            "raw_computable_fraction": float(group["raw_computable_windows"].sum() / group["n_windows"].sum()),
            "final_valid_windows": int(group["final_valid_windows"].sum()),
            "final_valid_fraction": float(group["final_valid_windows"].sum() / group["n_windows"].sum()),
            "final_valid_participants": int(group.loc[group["final_valid_windows"] > 0, "subject"].nunique()),
        })

    feature_support = pd.DataFrame(feature_support_rows)
    feature_summary = []
    for (condition, feature), group in feature_support.groupby(["condition", "source_feature"], sort=False):
        feature_summary.append({
            "condition": condition,
            "source_feature": feature,
            "n_recordings": int(group["recording"].nunique()),
            "n_participants": int(group.loc[group["computable_windows"] > 0, "subject"].nunique()),
            "n_windows": int(group["n_windows"].sum()),
            "computable_windows": int(group["computable_windows"].sum()),
            "computable_fraction": float(group["computable_windows"].sum() / group["n_windows"].sum()),
        })

    def _participant_median_frame(
        values_by_participant: dict[tuple[str, str], dict[str, list[float]]]
    ) -> pd.DataFrame:
        participant_rows = []
        for (condition, subject), features in sorted(values_by_participant.items()):
            participant_rows.append({
                "condition": condition,
                "subject": subject,
                **{
                    name: float(np.median(values)) if values else np.nan
                    for name, values in features.items()
                },
            })
        return pd.DataFrame(participant_rows)

    participant_frame = _participant_median_frame(participant_values_available)
    matched_participant_frame = _participant_median_frame(participant_values_matched)
    schema_inventory = pd.DataFrame([
        {
            "dataset_path": f"/measurements/v3_4/{name}",
            "shape_contract": shape_contract,
            "dtype": dtype,
            "recordings_present": count,
            "recordings_expected": len(rows),
        }
        for name, forms in schema_forms.items()
        for (shape_contract, dtype), count in forms.items()
    ])
    if (schema_inventory["recordings_present"] != len(rows)).any():
        raise ValueError("Core HDF5 schema varied across recordings; inspect hdf5_core_schema_inventory.csv")
    for relative_path, (shape_contract, dtype_contract) in OPTIONAL_SCHEMA.items():
        if relative_path not in schema_inventory["dataset_path"].str.removeprefix("/measurements/v3_4/").tolist():
            schema_inventory.loc[len(schema_inventory)] = {
                "dataset_path": f"/measurements/v3_4/{relative_path}",
                "shape_contract": shape_contract,
                "dtype": dtype_contract,
                "recordings_present": 0,
                "recordings_expected": len(rows),
            }
    reason_frame = pd.DataFrame([
        {"condition": condition, "measurement": measurement, "status_or_reason": status, "count": count}
        for (condition, measurement, status), count in sorted(reason_counts.items())
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "coordinate_support.csv", index=False)
    pd.DataFrame(feature_summary).to_csv(out_dir / "source_feature_support.csv", index=False)
    pd.DataFrame(recording_rows).to_csv(out_dir / "recording_support.csv", index=False)
    reason_frame.to_csv(out_dir / "support_status_counts.csv", index=False)
    participant_frame.to_csv(out_dir / "participant_condition_medians.csv", index=False)
    matched_participant_frame.to_csv(
        out_dir / "participant_condition_medians_matched_complete9.csv", index=False
    )
    schema_inventory.to_csv(out_dir / "hdf5_core_schema_inventory.csv", index=False)

    recording_frame = pd.DataFrame(recording_rows)
    completeness = (
        recording_frame.groupby("condition", sort=False)
        .agg(
            n_recordings=("recording", "nunique"),
            n_participants=("subject", "nunique"),
            n_windows=("n_windows", "sum"),
            complete_9d_windows=("complete_9d_windows", "sum"),
            recordings_with_complete_9d=("complete_9d_windows", lambda values: int((values > 0).sum())),
            qualified_recordings=("qualified", "sum"),
        )
        .reset_index()
    )
    completeness["complete_9d_fraction"] = (
        completeness["complete_9d_windows"] / completeness["n_windows"]
    )
    completeness.to_csv(out_dir / "complete_vector_support.csv", index=False)

    dependency_summaries = []
    dependency_comparisons = []
    dependency_rows = []
    loo_rows = []
    correlation_matrices = {}
    for condition in ("eyes_closed", "eyes_open"):
        method_results = {}
        method_frames = {
            "available_coordinate_medians": participant_frame,
            "matched_complete9_medians": matched_participant_frame,
        }
        method_subsets = {
            method: frame.loc[frame["condition"] == condition].set_index("subject")[
                list(COORDINATE_ORDER)
            ]
            for method, frame in method_frames.items()
        }
        common_subjects = (
            method_subsets["available_coordinate_medians"].dropna().index
            .intersection(method_subsets["matched_complete9_medians"].dropna().index)
        )
        for method, frame in method_frames.items():
            subset = method_subsets[method]
            complete = subset.loc[common_subjects].dropna(axis=0, how="any")
            corr = complete.corr(method="spearman")
            correlation_matrices[f"{method}:{condition}"] = corr
            corr.to_csv(
                out_dir / f"spearman_{method}_{condition}.csv", index_label="coordinate"
            )
            if method == "available_coordinate_medians":
                corr.to_csv(out_dir / f"spearman_{condition}.csv", index_label="coordinate")
            eig = np.linalg.eigvalsh(corr.to_numpy(dtype=float))
            eig = np.sort(np.clip(eig, 0.0, None))[::-1]
            effective_rank, participation_ratio = _effective_rank(eig)
            method_results[method] = {
                "n_participants": int(len(complete)),
                "correlation_matrix": corr,
                "spectral_entropy_effective_rank": effective_rank,
                "participation_ratio": participation_ratio,
            }
            for order, value in enumerate(eig, start=1):
                dependency_rows.append({
                    "condition": condition,
                    "aggregation_policy": method,
                    "matrix_type": "Spearman_correlation",
                    "eigenvalue_order": order,
                    "eigenvalue": float(value),
                    "fraction_of_total": float(value / eig.sum()) if eig.sum() else np.nan,
                    "complete_participants": int(len(complete)),
                    "spectral_entropy_effective_rank": effective_rank,
                    "participation_ratio": participation_ratio,
                })
            for omitted in COORDINATE_ORDER:
                keep = [name for name in COORDINATE_ORDER if name != omitted]
                loo_corr = complete[keep].corr(method="spearman")
                loo_eig = np.linalg.eigvalsh(loo_corr.to_numpy(dtype=float))
                loo_er, loo_pr = _effective_rank(loo_eig)
                loo_rows.append({
                    "condition": condition,
                    "aggregation_policy": method,
                    "omitted_coordinate": omitted,
                    "complete_participants": int(len(complete)),
                    "spectral_entropy_effective_rank_8d": loo_er,
                    "participation_ratio_8d": loo_pr,
                    "all_nine_spectral_entropy_effective_rank": effective_rank,
                    "all_nine_participation_ratio": participation_ratio,
                })
            dependency_summaries.append({
                "condition": condition,
                "aggregation_policy": method,
                "complete_participants": int(len(complete)),
                "participant_medians_available": int(len(subset)),
                "matrix_type": "Spearman correlation matrix",
                "spectral_entropy_effective_rank": effective_rank,
                "participation_ratio": participation_ratio,
            })
        available = method_results["available_coordinate_medians"]
        matched = method_results["matched_complete9_medians"]
        matrix_delta = (
            matched["correlation_matrix"].to_numpy()
            - available["correlation_matrix"].to_numpy()
        )
        dependency_comparisons.append({
            "condition": condition,
            "participants_available_medians": available["n_participants"],
            "participants_matched_complete9": matched["n_participants"],
            "mean_absolute_spearman_difference": float(np.mean(np.abs(matrix_delta))),
            "maximum_absolute_spearman_difference": float(np.max(np.abs(matrix_delta))),
            "frobenius_spearman_difference": float(np.linalg.norm(matrix_delta, ord="fro")),
            "available_spectral_entropy_effective_rank": available["spectral_entropy_effective_rank"],
            "matched_spectral_entropy_effective_rank": matched["spectral_entropy_effective_rank"],
            "matched_minus_available_effective_rank": (
                matched["spectral_entropy_effective_rank"]
                - available["spectral_entropy_effective_rank"]
            ),
            "available_participation_ratio": available["participation_ratio"],
            "matched_participation_ratio": matched["participation_ratio"],
            "matched_minus_available_participation_ratio": (
                matched["participation_ratio"] - available["participation_ratio"]
            ),
        })
    pd.DataFrame(dependency_rows).to_csv(out_dir / "dependency_eigenvalues.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(out_dir / "dependency_leave_one_coordinate_out.csv", index=False)
    comparison_frame = pd.DataFrame(dependency_comparisons)
    comparison_frame.to_csv(out_dir / "dependency_available_vs_matched.csv", index=False)
    _write_dependency_figure(correlation_matrices, out_dir / "dependency_heatmap.svg")
    if figure_out is not None:
        _write_dependency_figure(correlation_matrices, figure_out)

    condition_counts = Counter(row["condition"] for row in rows)
    qc_counts = Counter(
        (row["condition"], int(state))
        for row in rows
        for state in row["qc"]
    )
    psd_method_counts = Counter(
        method for row in rows for method in row["psd_method_ids"]
    )
    psd_fallback_counts = Counter(
        reason for row in rows for reason in row["psd_fallback_reasons"]
    )
    historical_path_counts = {
        name: sum(bool(row["historical_root_paths_present"][name]) for row in rows)
        for name in HISTORICAL_ROOT_PATHS
    }
    historical_path_frame = pd.DataFrame([
        {
            "dataset_path": f"/{name}",
            "recordings_present": historical_path_counts[name],
            "recordings_expected": len(rows),
        }
        for name in HISTORICAL_ROOT_PATHS
    ])
    historical_path_frame.to_csv(out_dir / "historical_root_path_audit.csv", index=False)
    report = {
        "schema": "mndm.mnps_v3_4_measurement_quality_audit.v2",
        "input_root": str(h5_root.resolve()),
        "hdf5_files_read": len(rows),
        "file_selection_rule": "canonical sub-*.h5 combined recording exports; separate mnps_v3_4_measurements.h5 sidecars excluded",
        "recordings_by_condition": dict(sorted(condition_counts.items())),
        "complete_vector_by_condition": completeness.to_dict(orient="records"),
        "core_hdf5_schema": schema_inventory.to_dict(orient="records"),
        "psd_method_id_window_counts": dict(sorted(psd_method_counts.items())),
        "psd_fallback_reason_window_counts": dict(sorted(psd_fallback_counts.items())),
        "historical_root_path_recording_counts": historical_path_counts,
        "qc_ok_eeg_window_state_counts": {
            f"{condition}:{state}": count for (condition, state), count in sorted(qc_counts.items())
        },
        "dependency_summaries": dependency_summaries,
        "dependency_available_vs_matched": dependency_comparisons,
        "interpretation_boundary": (
            "Descriptive support and participant-level dependency only. Overlapping windows were summarized "
            "within participant. Eigenvalues are from Spearman correlation matrices; effective rank is not evidence of a neural manifold, "
            "independent biological dimensions, reliability, artifact-free EEG, or scientific qualification."
        ),
    }
    (out_dir / "psd_method_identity.json").write_text(
        json.dumps(
            {
                "method_id_window_counts": report["psd_method_id_window_counts"],
                "fallback_reason_window_counts": report["psd_fallback_reason_window_counts"],
                "current_exports_contain_method_identity": (
                    "not_recorded_in_export" not in report["psd_method_id_window_counts"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report(
        out_dir,
        report,
        pd.DataFrame(summary_rows),
        pd.DataFrame(feature_summary),
        schema_inventory,
        comparison_frame,
    )


def _write_report(
    out_dir: Path,
    report: dict[str, Any],
    support: pd.DataFrame,
    features: pd.DataFrame,
    schema: pd.DataFrame,
    dependency_comparison: pd.DataFrame,
) -> None:
    lines = [
        "# MNPS 3.4 measurement-quality audit",
        "",
        f"Input: `{report['input_root']}`",
        f"HDF5 recordings read: {report['hdf5_files_read']}",
        "",
        "This read-only audit summarizes existing exports. It does not reprocess EEG or infer artifact-free data.",
        "Only canonical `sub-*.h5` combined recording files are included; the separate v3.4-only sidecar is excluded to avoid counting the same windows twice.",
        "Overlapping windows are pooled only for support counts; dependency analyses use participant-level medians.",
        "",
        "## Coordinate support",
        "",
        support.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Source-feature support",
        "",
        features.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Complete-nine support",
        "",
        pd.DataFrame(report["complete_vector_by_condition"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Participant-level dependency",
        "",
        "Two participant-median policies are compared on the same participants: each coordinate uses all windows where it is valid, or all nine medians use only windows where the complete-nine mask is true. Eigenvalues are from the Spearman correlation matrix (not a covariance matrix) and are descriptive diagnostics, not intrinsic-dimension estimates.",
        "",
        "![Available-coordinate and matched-complete-nine Spearman matrices by condition](dependency_heatmap.svg)",
        "",
        pd.DataFrame(report["dependency_summaries"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Available-coordinate versus matched-window comparison",
        "",
        dependency_comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Core HDF5 schema audit",
        "",
        "The paths below were checked across every selected combined recording export. `W` is the recording's window count, `D=9` is the coordinate count, `F=10` is the raw-feature count, and `C` is the stored channel count.",
        "",
        schema.to_markdown(index=False),
        "",
        "### Root-level historical paths in the combined exports",
        "",
        pd.DataFrame([
            {
                "dataset_path": f"/{name}",
                "recordings_present": count,
                "recordings_expected": report["hdf5_files_read"],
            }
            for name, count in report["historical_root_path_recording_counts"].items()
        ]).to_markdown(index=False),
        "These counts describe the examined combined exports, not all archived historical releases.",
        "",
        "## PSD estimator identity",
        "",
        f"Method IDs by window: `{json.dumps(report['psd_method_id_window_counts'], sort_keys=True)}`",
        "",
        f"Fallback reasons by window: `{json.dumps(report['psd_fallback_reason_window_counts'], sort_keys=True)}`",
        "",
        "## Interpretation boundary",
        "",
        report["interpretation_boundary"],
        "",
        "QC field states and estimator support reasons are in `support_status_counts.csv`; per-recording denominators are in `recording_support.csv`.",
        "Legacy path counts in these combined files are in `historical_root_path_audit.csv`; they do not establish the schema of every archived historical release.",
        "PSD backend identity in these cached exports is summarized in `psd_method_identity.json`; missing backend fields are reported as not recorded.",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dependency_figure(matrices: dict[str, pd.DataFrame], path: Path) -> None:
    """Write condition- and support-policy-stratified Spearman matrices."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "δ/θ", "α level", "β/α", "γ level", "Hjorth",
        "α/θ", "Perm. entropy", "α centroid", "α env. CV",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    image = None
    policies = (
        ("available_coordinate_medians", "Available-coordinate medians"),
        ("matched_complete9_medians", "Matched complete-nine medians"),
    )
    for row_index, (policy, policy_label) in enumerate(policies):
        for column_index, condition in enumerate(("eyes_closed", "eyes_open")):
            axis = axes[row_index, column_index]
            matrix = matrices[f"{policy}:{condition}"].loc[
                list(COORDINATE_ORDER), list(COORDINATE_ORDER)
            ].to_numpy()
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
            condition_label = "Eyes closed" if condition == "eyes_closed" else "Eyes open"
            axis.set_title(f"{condition_label} · {policy_label}")
            axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, ha="center")
            axis.set_yticks(np.arange(len(labels)), labels=labels)
            axis.tick_params(axis="both", labelsize=8)
            axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.7)
            axis.tick_params(which="minor", bottom=False, left=False)
            for matrix_row in range(len(labels)):
                for matrix_column in range(len(labels)):
                    value = matrix[matrix_row, matrix_column]
                    axis.text(
                        matrix_column,
                        matrix_row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                    )
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.72, label="Spearman correlation (participant medians)")
    fig.savefig(path, format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-root", type=Path, required=True, help="Root containing the selected v3.4 HDF5 exports")
    parser.add_argument("--out", type=Path, required=True, help="New audit output directory")
    parser.add_argument("--figure-out", type=Path, help="Optional second path for the SVG correlation figure")
    args = parser.parse_args()
    run(args.h5_root, args.out, args.figure_out)


if __name__ == "__main__":
    main()

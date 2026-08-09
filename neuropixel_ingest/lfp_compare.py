"""Conservative descriptive comparison for shared-depth probe feature ensembles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import h5py


def write_shared_depth_comparison(
    probe_b_features: str | Path, probe_f_features: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Write a descriptive comparison restricted to depth ensembles present in both probes."""
    b = pd.read_parquet(probe_b_features)
    f = pd.read_parquet(probe_f_features)
    shared = sorted(
        set(_depth_columns(b)).intersection(_depth_columns(f)),
        key=lambda name: int(name.removeprefix("depth_")),
    )
    result: dict[str, Any] = {
        "schema": "mndm.lfp_shared_depth_comparison.v1",
        "interpretation_limit": "Descriptive feature agreement only; no cross-probe geometric inference.",
        "n_epochs_probe_b": len(b),
        "n_epochs_probe_f": len(f),
        "shared_depth_ensembles": {},
    }
    for depth in shared:
        suffix = f"__g_{depth}"
        b_columns = {column.removesuffix(suffix): column for column in b if column.endswith(suffix)}
        f_columns = {column.removesuffix(suffix): column for column in f if column.endswith(suffix)}
        features = sorted(set(b_columns).intersection(f_columns))
        result["shared_depth_ensembles"][depth] = {
            feature: float(b[b_columns[feature]].corr(f[f_columns[feature]])) for feature in features
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_shared_depth_state_comparison(
    probe_b_features: str | Path, probe_f_features: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Write descriptive shared-depth agreement separately for inter-stim state rows."""
    b = pd.read_parquet(probe_b_features)
    f = pd.read_parquet(probe_f_features)
    output: dict[str, Any] = {
        "schema": "mndm.lfp_shared_depth_state_comparison.v1",
        "interpretation_limit": "Descriptive feature agreement only; no cross-probe geometric inference.",
        "states": {},
    }
    for state in ("awake", "isoflurane", "recovery"):
        b_state = b.loc[b["lfp_interstim_primary"] & b["lfp_behavioral_state"].eq(state)].reset_index(drop=True)
        f_state = f.loc[f["lfp_interstim_primary"] & f["lfp_behavioral_state"].eq(state)].reset_index(drop=True)
        n_rows = min(len(b_state), len(f_state))
        if n_rows == 0:
            continue
        b_state, f_state = b_state.iloc[:n_rows], f_state.iloc[:n_rows]
        shared = sorted(set(_depth_columns(b_state)).intersection(_depth_columns(f_state)))
        output["states"][state] = {
            "n_epochs_probe_b": len(b_state),
            "n_epochs_probe_f": len(f_state),
            "shared_depth_ensembles": {
                depth: {
                    base: float(b_state[b_column].corr(f_state[f_column]))
                    for base, b_column in {
                        column.removesuffix(f"__g_{depth}"): column
                        for column in b_state if column.endswith(f"__g_{depth}")
                    }.items()
                    if (f_column := f"{base}__g_{depth}") in f_state
                }
                for depth in shared
            },
        }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def write_condition_sensitivity(
    baseline_features: str | Path,
    variant_features: str | Path,
    baseline_h5: str | Path,
    variant_h5: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Describe feature and MNPS agreement for a single sensitivity condition."""
    baseline = pd.read_parquet(baseline_features)
    variant = pd.read_parquet(variant_features)
    common_rows = baseline.merge(variant, on=["epoch_id", "file"], suffixes=("_base", "_variant"))
    base_columns = {
        column.removesuffix("_base"): column
        for column in common_rows
        if column.endswith("_base") and column.startswith("eeg_")
    }
    feature_corr = {
        name: float(common_rows[column].corr(common_rows[f"{name}_variant"]))
        for name, column in base_columns.items()
        if f"{name}_variant" in common_rows
        and pd.api.types.is_numeric_dtype(common_rows[column])
        and pd.api.types.is_numeric_dtype(common_rows[f"{name}_variant"])
    }
    with h5py.File(baseline_h5, "r") as base_handle, h5py.File(variant_h5, "r") as variant_handle:
        base_mnps = np.asarray(base_handle["mnps_3d"])
        variant_mnps = np.asarray(variant_handle["mnps_3d"])
    rows = min(len(base_mnps), len(variant_mnps))
    output = {
        "schema": "mndm.lfp_condition_sensitivity.v1",
        "interpretation_limit": "Within-session descriptive stability; MNPS coordinate orientation is not identified across independently normalized runs.",
        "n_matched_feature_epochs": int(len(common_rows)),
        "n_matched_mnps_rows": int(rows),
        "feature_correlations": feature_corr,
        "mnps_coordinate_correlations": [
            float(pd.Series(base_mnps[:rows, index]).corr(pd.Series(variant_mnps[:rows, index])))
            for index in range(3)
        ],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def write_condition_state_feature_sensitivity(
    baseline_features: str | Path, variant_features: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Describe selected depth-feature stability within each annotated state."""
    baseline = pd.read_parquet(baseline_features)
    variant = pd.read_parquet(variant_features)
    result: dict[str, Any] = {
        "schema": "mndm.lfp_condition_state_feature_sensitivity.v1",
        "interpretation_limit": "Feature-level, within-session stability only; no cross-run MNPS coordinate identity is assumed.",
        "states": {},
    }
    for state in ("awake", "isoflurane", "recovery"):
        left = baseline.loc[baseline["lfp_interstim_primary"] & baseline["lfp_behavioral_state"].eq(state)]
        right = variant.loc[variant["lfp_interstim_primary"] & variant["lfp_behavioral_state"].eq(state)]
        rows = left.merge(right, on=["epoch_id", "file"], suffixes=("_base", "_variant"))
        columns = [
            column
            for column in rows
            if column.endswith("_base")
            and "__g_depth_" in column
            and pd.api.types.is_numeric_dtype(rows[column])
            and pd.api.types.is_numeric_dtype(rows[column.removesuffix("_base") + "_variant"])
        ]
        result["states"][state] = {
            "n_matched_epochs": int(len(rows)),
            "depth_feature_correlations": {
                column.removesuffix("_base"): float(
                    rows[column].corr(rows[column.removesuffix("_base") + "_variant"])
                )
                for column in columns
            },
        }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_cross_session_comparison(
    primary_features: str | Path,
    replicate_features: str | Path,
    primary_h5: str | Path,
    replicate_h5: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare sessions without assuming aligned MNPS coordinate orientation."""
    from scipy.stats import wasserstein_distance

    first, second = pd.read_parquet(primary_features), pd.read_parquet(replicate_features)
    states: dict[str, Any] = {}
    for state in ("awake", "isoflurane", "recovery"):
        left = first.loc[first["lfp_interstim_primary"] & first["lfp_behavioral_state"].eq(state)]
        right = second.loc[second["lfp_interstim_primary"] & second["lfp_behavioral_state"].eq(state)]
        columns = sorted(set(_depth_columns(left)).intersection(_depth_columns(right)))
        states[state] = {
            "n_epochs_primary": int(len(left)),
            "n_epochs_replicate": int(len(right)),
            "depth_feature_wasserstein": {
                depth: {
                    name: float(wasserstein_distance(left[column].dropna(), right[column].dropna()))
                    for name, column in {
                        value.removesuffix(f"__g_{depth}"): value
                        for value in left
                        if value.endswith(f"__g_{depth}") and pd.api.types.is_numeric_dtype(left[value])
                    }.items()
                    if f"{name}__g_{depth}" in right
                    and len(left[column].dropna())
                    and len(right[f"{name}__g_{depth}"].dropna())
                }
                for depth in columns
            },
        }
    def _spectra(path: str | Path) -> np.ndarray:
        with h5py.File(path, "r") as handle:
            jacobian = np.asarray(handle["jacobian/J_hat"])
        return np.sort(np.abs(np.linalg.eigvals(jacobian)), axis=1)
    left_spectra, right_spectra = _spectra(primary_h5), _spectra(replicate_h5)
    output = {
        "schema": "mndm.lfp_cross_session_replication.v1",
        "interpretation_limit": "Orientation-invariant descriptive comparison; not population inference.",
        "states": states,
        "jacobian_spectrum": {
            "primary_mean_abs_eigenvalues": np.nanmean(left_spectra, axis=0).tolist(),
            "replicate_mean_abs_eigenvalues": np.nanmean(right_spectra, axis=0).tolist(),
            "wasserstein_by_rank": [
                float(wasserstein_distance(left_spectra[:, index], right_spectra[:, index]))
                for index in range(min(left_spectra.shape[1], right_spectra.shape[1]))
            ],
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _depth_columns(frame: pd.DataFrame) -> set[str]:
    return {
        column.rsplit("__g_", 1)[1]
        for column in frame.columns
        if "__g_depth_" in column
    }

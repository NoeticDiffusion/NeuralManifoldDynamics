"""Secondary cohort-coordinate age analysis for the Figshare infant EEG run.

This module intentionally models cohort-anchored coordinate positions, not
subject-anchored Jacobian/reachability measures.  The latter remain the
within-child primary analysis contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


COORDINATE_GROUPS = ("coords_3d_cohort_anchored", "coords_9d_cohort_anchored")


def _decode_names(values: Iterable[Any]) -> list[str]:
    """Decode HDF5 coordinate names."""
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _participant_meta(h5_path: Path) -> dict[str, Any]:
    """Read the summary-side session metadata paired with one HDF5 output."""
    summary_path = h5_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return dict(json.loads(summary_path.read_text(encoding="utf-8")).get("participant_meta") or {})
    except (OSError, ValueError, TypeError):
        return {}


def build_session_coordinate_table(run_dir: Path, sessions_path: Path) -> pd.DataFrame:
    """Summarize cohort-anchored HDF5 trajectories to one row per session."""
    sessions = pd.read_csv(sessions_path, sep="\t")
    required = {"participant_id", "session_id", "age_days", "age_source", "excluded_from_age_analysis"}
    if missing := required.difference(sessions.columns):
        raise ValueError(f"Sessions table lacks required columns: {sorted(missing)}")
    session_lookup = {
        (str(row.participant_id), str(row.session_id)): row.to_dict()
        for _, row in sessions.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for h5_path in sorted(run_dir.glob("**/*.h5")):
        meta = _participant_meta(h5_path)
        participant_id = str(meta.get("participant_id", "")).strip()
        session_id = str(meta.get("session_id", "")).strip()
        session = session_lookup.get((participant_id, session_id))
        if not session:
            continue
        row: dict[str, Any] = {
            "h5_path": str(h5_path),
            "participant_id": participant_id,
            "session_id": session_id,
            "age_days": session.get("age_days"),
            "age_source": session.get("age_source"),
            "excluded_from_age_analysis": bool(session.get("excluded_from_age_analysis", False)),
        }
        with h5py.File(h5_path, "r") as h5:
            for group_name in COORDINATE_GROUPS:
                if group_name not in h5 or "values" not in h5[group_name]:
                    continue
                values = np.asarray(h5[group_name]["values"], dtype=float)
                names = _decode_names(h5[group_name]["names"][()])
                row[f"{group_name}_n_timepoints"] = int(values.shape[0])
                for index, name in enumerate(names):
                    row[f"{group_name}__{name}_median"] = float(np.nanmedian(values[:, index]))
        rows.append(row)
    return pd.DataFrame(rows)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values retaining missing entries."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def _fit_gee_spline(
    table: pd.DataFrame,
    coordinate_column: str,
    analysis_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit one Gaussian GEE with child clusters and a cubic age spline."""
    import statsmodels.api as sm

    model_data = table.loc[
        table[coordinate_column].notna()
        & table["age_days"].notna()
        & table["participant_id"].notna()
    ].copy()
    n_clusters = int(model_data["participant_id"].nunique())
    if len(model_data) < 6 or n_clusters < 3:
        return (
            {
                "analysis": analysis_name,
                "coordinate": coordinate_column,
                "status": "insufficient_data",
                "n_sessions": int(len(model_data)),
                "n_children": n_clusters,
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    formula = f"{coordinate_column} ~ bs(age_days, df=3, degree=3, include_intercept=False)"
    result = sm.GEE.from_formula(
        formula,
        groups="participant_id",
        data=model_data,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    spline_terms = [name for name in result.params.index if name != "Intercept"]
    constraint = np.zeros((len(spline_terms), len(result.params)))
    for row_index, term in enumerate(spline_terms):
        constraint[row_index, list(result.params.index).index(term)] = 1.0
    omnibus = result.wald_test(constraint, scalar=False)
    omnibus_p = float(np.asarray(omnibus.pvalue).squeeze())
    omnibus_chi2 = float(np.asarray(omnibus.statistic).squeeze())

    coefficient_rows = pd.DataFrame(
        {
            "analysis": analysis_name,
            "coordinate": coordinate_column,
            "term": result.params.index,
            "estimate": result.params.to_numpy(),
            "robust_se": result.bse.to_numpy(),
            "z_value": result.tvalues.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "n_sessions": len(model_data),
            "n_children": n_clusters,
        }
    )
    age_grid = np.linspace(float(model_data["age_days"].min()), float(model_data["age_days"].max()), 100)
    prediction_data = pd.DataFrame({"age_days": age_grid, "participant_id": model_data["participant_id"].iloc[0]})
    prediction = result.get_prediction(prediction_data).summary_frame()
    prediction_rows = pd.DataFrame(
        {
            "analysis": analysis_name,
            "coordinate": coordinate_column,
            "age_days": age_grid,
            "predicted_mean": prediction["mean"].to_numpy(),
            "ci_lower": prediction["mean_ci_lower"].to_numpy(),
            "ci_upper": prediction["mean_ci_upper"].to_numpy(),
        }
    )
    summary = {
        "analysis": analysis_name,
        "coordinate": coordinate_column,
        "status": "fit",
        "n_sessions": int(len(model_data)),
        "n_children": n_clusters,
        "age_min_days": float(model_data["age_days"].min()),
        "age_max_days": float(model_data["age_days"].max()),
        "spline_df": 3,
        "working_correlation": "exchangeable",
        "converged": bool(result.converged),
        "omnibus_wald_chi2": omnibus_chi2,
        "omnibus_p_value": omnibus_p,
    }
    return summary, coefficient_rows, prediction_rows


def fit_cohort_coordinate_age_models(session_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit resolved-age and direct-age-only GEE sensitivity analyses."""
    eligible = session_table.loc[
        ~session_table["excluded_from_age_analysis"].astype(bool)
        & session_table["age_days"].notna()
    ].copy()
    coordinate_columns = [
        column
        for column in eligible.columns
        if column.startswith("coords_") and column.endswith("_median")
    ]
    outputs: list[dict[str, Any]] = []
    coefficients: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    analyses = {
        "resolved_age_primary": eligible,
        "spreadsheet_direct_age_sensitivity": eligible.loc[eligible["age_source"] == "spreadsheet_direct"].copy(),
    }
    for analysis_name, subset in analyses.items():
        for column in coordinate_columns:
            summary, coefficient_frame, prediction_frame = _fit_gee_spline(subset, column, analysis_name)
            outputs.append(summary)
            if not coefficient_frame.empty:
                coefficients.append(coefficient_frame)
            if not prediction_frame.empty:
                predictions.append(prediction_frame)
    results = pd.DataFrame(outputs)
    if not results.empty:
        results["omnibus_p_fdr_bh"] = results.groupby("analysis")["omnibus_p_value"].transform(_bh_fdr)
    return (
        results,
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
    )


def write_qc_plot(session_table: pd.DataFrame, predictions: pd.DataFrame, out_path: Path) -> None:
    """Write a compact age-versus-cohort-coordinate QC figure."""
    import matplotlib.pyplot as plt

    coordinates = [
        "coords_3d_cohort_anchored__m_median",
        "coords_3d_cohort_anchored__d_median",
        "coords_3d_cohort_anchored__e_median",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True)
    for axis, column in zip(axes, coordinates):
        axis.scatter(session_table["age_days"], session_table[column], color="0.35", alpha=0.8, s=24)
        model = predictions.loc[
            (predictions["analysis"] == "resolved_age_primary")
            & (predictions["coordinate"] == column)
        ]
        if not model.empty:
            axis.plot(model["age_days"], model["predicted_mean"], color="C0")
            axis.fill_between(
                model["age_days"],
                model["ci_lower"],
                model["ci_upper"],
                color="C0",
                alpha=0.2,
            )
        axis.set_title(column.rsplit("__", 1)[-1].replace("_median", ""))
        axis.set_xlabel("Age (days)")
    axes[0].set_ylabel("Cohort-anchored coordinate")
    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def run_analysis(run_dir: Path, sessions_path: Path, out_dir: Path) -> dict[str, Any]:
    """Build cohort table, fit age models, and export auditable artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_table = build_session_coordinate_table(run_dir, sessions_path)
    session_table.to_csv(out_dir / "cohort_coordinate_session_table.csv", index=False)
    results, coefficients, predictions = fit_cohort_coordinate_age_models(session_table)
    results.to_csv(out_dir / "cohort_coordinate_age_gee_results.csv", index=False)
    coefficients.to_csv(out_dir / "cohort_coordinate_age_gee_coefficients.csv", index=False)
    predictions.to_csv(out_dir / "cohort_coordinate_age_gee_predictions.csv", index=False)
    eligible = session_table.loc[
        ~session_table["excluded_from_age_analysis"].astype(bool)
        & session_table["age_days"].notna()
    ].copy()
    write_qc_plot(eligible, predictions, out_dir / "cohort_coordinate_age_qc.png")
    summary = {
        "schema": "mndm.figshare_infant_cohort_age_gee.v1",
        "run_dir": str(run_dir),
        "sessions_path": str(sessions_path),
        "n_h5_sessions": int(len(session_table)),
        "n_age_eligible_sessions": int(len(eligible)),
        "n_age_eligible_children": int(eligible["participant_id"].nunique()),
        "n_direct_age_sessions": int((eligible["age_source"] == "spreadsheet_direct").sum()),
        "model": {
            "family": "Gaussian",
            "estimator": "GEE",
            "cluster": "participant_id",
            "working_correlation": "exchangeable",
            "age_basis": "cubic_bspline_df3",
            "coordinate_contract": "cohort_anchored",
        },
        "interpretation": (
            "Secondary between-child/age coordinate associations. "
            "Subject-anchored MNJ and reachability remain within-child analyses."
        ),
    }
    (out_dir / "cohort_coordinate_age_gee_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Run cohort-coordinate GEE analysis from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.run_dir, args.sessions, args.out_dir), indent=2))


if __name__ == "__main__":
    main()

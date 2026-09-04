"""Run the JQ1-B synthetic replay through the production MNDM CLI path.

This is a truth-known translation qualification, not an empirical ingest
workflow.  It writes an isolated features table, invokes ``mndm.cli
summarize``, then reads the generated HDF5 metric series back for comparison.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.linalg import expm

from .. import cli

DT = 0.01
N_STEPS = 200
SEEDS = (611, 612, 613, 614)
J1_TOLERANCES = {"alpha": 0.10, "omega": 0.10, "delta_react": 0.10}
J2_TOLERANCES = {"alpha": 0.08, "omega": 0.08, "delta_react": 0.15}


@dataclass(frozen=True)
class TruthCase:
    family: str
    identifier: str
    matrix_2d: np.ndarray
    sigma_process: float
    seed: int

    @property
    def task(self) -> str:
        return f"jq1{self.family.lower()}{self.identifier.lower()}p{int(self.sigma_process > 0)}"

    @property
    def matrix_3d(self) -> np.ndarray:
        return np.block(
            [
                [self.matrix_2d, np.zeros((2, 1))],
                [np.zeros((1, 2)), np.array([[-4.0]])],
            ]
        )


def _matched_omega_pair() -> tuple[np.ndarray, np.ndarray]:
    def _k(a: float, b: float, omega: float) -> float:
        return 2.0 * np.sqrt((omega + (a + b) / 2.0) ** 2 - ((a - b) / 2.0) ** 2)

    return (
        np.array([[-1.0, _k(1.0, 3.0, 1.0)], [0.0, -3.0]]),
        np.array([[-1.0, _k(1.0, 20.0, 1.0)], [0.0, -20.0]]),
    )


def truth_cases() -> list[TruthCase]:
    """Return predeclared clean J1/J2 systems and J2 process strata."""
    m0, m1 = _matched_omega_pair()
    j1 = {
        "T0": np.diag([-1.0, -3.0]),
        "T1": np.array([[-1.0, 6.0], [0.0, -3.0]]),
        "T2": np.array([[-1.0, 2.0], [0.0, -3.0]]),
        "T3": np.array([[0.5, 3.0], [0.0, -1.0]]),
        "T4": -2.0 * np.eye(2),
        "T5": np.array([[-2.0, -1.5], [1.5, -2.0]]),
        "T8": m0,
        "T9": m1,
    }
    cases = [
        # Stage J1's reference family used additive process noise (sigma=0.02);
        # preserve that family distinction rather than making seed stability
        # vacuous through four identical deterministic trajectories.
        TruthCase("J1", identifier, matrix, sigma_process=0.02, seed=seed)
        for identifier, matrix in j1.items()
        for seed in SEEDS
    ]
    j2 = {
        "S0": np.diag([-1.0, -3.0]),
        "S1": np.array([[-1.0, 6.0], [0.0, -3.0]]),
        "S2": np.array([[-2.0, -1.5], [1.5, -2.0]]),
    }
    for stratum, sigma_process in (("P0_zero_process_noise", 0.0), ("P1_reference_process_noise", 0.02)):
        for identifier, matrix in j2.items():
            for seed in SEEDS:
                cases.append(
                    TruthCase(
                        "J2",
                        f"{identifier}_{stratum}",
                        matrix,
                        sigma_process=sigma_process,
                        seed=seed,
                    )
                )
    return cases


def _truth_metrics(jacobian: np.ndarray) -> dict[str, float]:
    alpha = float(np.max(np.linalg.eigvals(jacobian).real))
    omega = float(np.max(np.linalg.eigvalsh(0.5 * (jacobian + jacobian.T))))
    return {"alpha": alpha, "omega": omega, "delta_react": omega - alpha}


def _simulate(case: TruthCase) -> np.ndarray:
    """Produce one exact-transition 3D trajectory from a fixed nonzero state."""
    transition = expm(case.matrix_3d * DT)
    rng = np.random.default_rng(case.seed)
    x = np.empty((N_STEPS + 1, 3), dtype=float)
    x[0] = np.array([1.0, -0.7, 0.45])
    for index in range(N_STEPS):
        noise = rng.normal(scale=case.sigma_process * np.sqrt(DT), size=3)
        x[index + 1] = transition @ x[index] + noise
    return x


def _feature_rows(cases: list[TruthCase]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        trajectory = _simulate(case)
        file_name = f"sub-001_task-{case.task}_run-{case_index + 1:02d}_eeg.csv"
        for index, value in enumerate(trajectory):
            start = index * DT
            rows.append(
                {
                    "file": file_name,
                    "subject": "001",
                    "t_start": start,
                    "t_end": start + DT,
                    "qc_ok_eeg": 1,
                    "synthetic_m": value[0],
                    "synthetic_d": value[1],
                    "synthetic_e": value[2],
                    "jq1_case_index": case_index,
                }
            )
    return pd.DataFrame(rows)


def _config(processed_dir: Path, received_dir: Path) -> dict[str, Any]:
    """Return the complete, predeclared identity-projection JQ1 config."""
    return {
        "datasets": ["jq1_synthetic"],
        "paths": {"processed_dir": str(processed_dir), "received_dir": str(received_dir)},
        "modality": "eeg",
        "feature_storage": {"read_prefer": "csv"},
        "summarize": {"qc_policy": "eeg_only", "allow_group_collisions": False},
        "mnps_projection": {
            "normalize": None,
            "feature_standardization": {},
            "clip_threshold": 6.0,
            "min_axis_coverage": 1.0,
            "anchor": {"enabled": False},
            "weights": {
                "m": {"synthetic_m": 1.0},
                "d": {"synthetic_d": 1.0},
                "e": {"synthetic_e": 1.0},
            },
            "export_contracts": {"subject_anchored": True, "cohort_anchored": False},
        },
        "mnps_3d": {"mode": "direct_features"},
        "mnps_9d": {"enabled": False},
        "mnps": {
            "fs_out": 100.0,
            "window_sec": DT,
            "overlap": 0.0,
            "derivative": {"method": "sav_gol", "window": 7, "polyorder": 3},
            "derivative_robust": {"enabled": True, "max_jump": 5.0, "min_seg": 9},
            "knn": {"k": 20, "metric": "euclidean"},
            "ridge": {"alpha": 1.0, "distance_weighted": True},
            "whiten": True,
            "super_window": 3,
            "jacobian": {"enabled": True},
        },
        "robustness": {"coverage": {"min_seconds": 0.0, "min_epochs": 1}},
        "regional_mnps": {"enabled": False},
        "normalization": {"enabled": False},
    }


def _find_h5(run_dir: Path, task: str, run_index: int) -> Path:
    # The production BIDS parser normalizes the synthetic task label in the
    # HDF5 basename, but preserves the unique run entity.
    matches = sorted(run_dir.rglob(f"*run-{run_index + 1:02d}*.h5"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one H5 for task {task}, found {matches}")
    return matches[0]


def _read_case(h5_path: Path, case: TruthCase) -> dict[str, Any]:
    analysis_root = Path("J:/repos/nmd-analysis/nmd-analysis")
    if analysis_root.exists() and str(analysis_root) not in sys.path:
        sys.path.insert(0, str(analysis_root))
    try:
        from nmd_analysis.local_dynamics.adapters import jacobian_metrics_window_rows
    except ImportError as exc:
        raise RuntimeError(
            "JQ1 requires the nmd-analysis local-dynamics adapters; "
            "set up that repository or classify the adapter contract as blocked"
        ) from exc

    adapter_rows = jacobian_metrics_window_rows(h5_path, dimension=3)
    with h5py.File(h5_path, "r") as h5:
        j_hat = np.asarray(h5["jacobian/J_hat"], dtype=float)
        centers = np.asarray(h5["jacobian/centers"], dtype=int)
        diagnostics = h5["jacobian/diagnostics"].attrs
        exported = {
            "alpha": np.asarray([row.get("spectral_abscissa", np.nan) for row in adapter_rows], dtype=float),
            "omega": np.asarray([row.get("numerical_abscissa", np.nan) for row in adapter_rows], dtype=float),
            "delta_react": np.asarray([row.get("reactivity_gap", np.nan) for row in adapter_rows], dtype=float),
        }
        truth = _truth_metrics(case.matrix_3d)
        valid = np.isfinite(exported["alpha"]) & np.isfinite(exported["omega"]) & np.isfinite(exported["delta_react"])
        n_valid = int(np.sum(valid))
        errors = {
            metric: np.abs(values[valid] - truth[metric])
            for metric, values in exported.items()
        }
        tolerances = J1_TOLERANCES if case.family == "J1" else J2_TOLERANCES
        return {
            "family": case.family,
            "identifier": case.identifier,
            "task": case.task,
            "seed": case.seed,
            "sigma_process": case.sigma_process,
            "h5_path": str(h5_path),
            "truth": truth,
            "shape": {"j_hat": list(j_hat.shape), "centers": list(centers.shape)},
            "window_qc": {
                "windows_raw": int(diagnostics["windows_raw"]),
                "windows_retained": int(diagnostics["windows"]),
                "invalid_windows": int(diagnostics["hard_invalid_windows"]),
                "n_adapter_rows": len(adapter_rows),
                "n_valid_metric_rows": n_valid,
                "j_dot_dt": float(diagnostics["j_dot_dt"]),
            },
            "metric_error": {
                metric: {
                    "max": float(np.max(values)) if values.size else float("nan"),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "median": float(np.median(values)) if values.size else float("nan"),
                    "sd": float(np.std(values, ddof=0)) if values.size else float("nan"),
                    "tolerance": tolerances[metric],
                }
                for metric, values in errors.items()
            },
            "pass": bool(
                n_valid >= 3
                and all(
                    values.size > 0 and float(np.max(values)) <= tolerances[metric]
                    for metric, values in errors.items()
                )
            ),
        }


def run(output_root: Path) -> dict[str, Any]:
    """Generate fixtures, invoke production summarize, and compare HDF5 output."""
    output_root = output_root.resolve()
    processed = output_root / "processed"
    received = output_root / "received"
    dataset_dir = processed / "jq1_synthetic"
    if output_root.exists():
        shutil.rmtree(output_root)
    dataset_dir.mkdir(parents=True)
    received.mkdir(parents=True)

    cases = truth_cases()
    feature_path = dataset_dir / "features.csv"
    feature_rows = _feature_rows(cases)
    feature_rows.to_csv(feature_path, index=False)
    per_file_dt = feature_rows.groupby("file", sort=False)["t_start"].apply(
        lambda values: float(np.median(np.diff(values.to_numpy(dtype=float))))
    )
    configured_dt = DT * (1.0 - 0.0)
    if not np.allclose(per_file_dt.to_numpy(), DT) or not np.isclose(configured_dt, DT):
        raise RuntimeError("synthetic time-base contract failed before production invocation")
    config_path = output_root / "config_ingest_jq1_synthetic.yaml"
    config_path.write_text(yaml.safe_dump(_config(processed, received), sort_keys=False), encoding="utf-8")
    (output_root / "config_snapshot" / "jq1_seeds.json").parent.mkdir(parents=True, exist_ok=True)
    spent_pools = {
        "j1_development": list(range(3000, 3020)),
        "j1_held_out": [211, 222, 233, 244, 255, 266, 277, 288, 299, 311],
        "j2_development": list(range(4000, 4010)),
        "j2_held_out": [411, 422, 433, 444, 455],
        "f1_development": [0, 1, 2, 3],
        "f1_held_out": [101, 102, 103, 104],
    }
    collisions = {
        pool_name: sorted(set(SEEDS) & set(pool))
        for pool_name, pool in spent_pools.items()
    }
    if any(collisions.values()):
        raise RuntimeError(f"JQ1 seeds collide with spent pool(s): {collisions}")
    (output_root / "config_snapshot" / "jq1_seeds.json").write_text(
        json.dumps(
            {
                "jq1_seeds": list(SEEDS),
                "purpose": "fresh JQ1 translation qualification",
                "pools_checked": spent_pools,
                "collisions": collisions,
                "collision_check_passed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "summarize",
            "--dataset",
            "jq1_synthetic",
            "--config",
            str(config_path),
            "--out-dir",
            str(processed),
            "--data-dir",
            str(received),
            "--h5-mode",
            "subject",
            "--n-jobs",
            "1",
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"production MNDM summarize failed with exit code {exit_code}")

    run_dirs = sorted(path for path in dataset_dir.glob("neuralmanifolddynamics_*") if path.is_dir())
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected exactly one MNDM run directory, found {run_dirs}")
    rows = [
        _read_case(_find_h5(run_dirs[0], case.task, case_index), case)
        for case_index, case in enumerate(cases)
    ]
    seed_stability: dict[str, Any] = {}
    for identifier in sorted({row["identifier"] for row in rows if row["family"] == "J1"}):
        system_rows = [row for row in rows if row["family"] == "J1" and row["identifier"] == identifier]
        metric_sd = {
            metric: float(np.std([row["metric_error"][metric]["max"] for row in system_rows], ddof=0))
            for metric in J1_TOLERANCES
        }
        seed_stability[identifier] = {
            "n_seeds": len(system_rows),
            "max_error_sd": max(metric_sd.values()),
            "per_metric_error_sd": metric_sd,
            "pass": bool(len(system_rows) == len(SEEDS) and max(metric_sd.values()) <= 0.05),
        }
    primary_rows_pass = bool(rows) and all(row["pass"] for row in rows)
    seed_stability_pass = bool(seed_stability) and all(row["pass"] for row in seed_stability.values())
    result = {
        "status": "complete",
        "dt_truth": DT,
        "time_base_assertion": {
            "measured_median_t_start_step_per_file": per_file_dt.tolist(),
            "configured_window_step": configured_dt,
            "derivative_delta_from_hdf5": sorted(
                {row["window_qc"]["j_dot_dt"] for row in rows}
            ),
            "passed": bool(
                np.allclose(per_file_dt.to_numpy(), DT)
                and np.isclose(configured_dt, DT)
                and all(np.isclose(row["window_qc"]["j_dot_dt"], DT) for row in rows)
            ),
        },
        "config_path": str(config_path),
        "feature_path": str(feature_path),
        "run_dir": str(run_dirs[0]),
        "rows": rows,
        "j1_seed_stability": seed_stability,
        "primary_rows_pass": primary_rows_pass,
        "j1_seed_stability_pass": seed_stability_pass,
        "overall_pass": bool(primary_rows_pass and seed_stability_pass),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
    }
    (output_root / "effect_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    default_output = Path(__file__).resolve().parents[3] / "results" / "nmd_jq1" / "synthetic_output"
    print(json.dumps(run(default_output), indent=2))

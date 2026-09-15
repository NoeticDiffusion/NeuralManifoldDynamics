"""Family B SL-002: temporal-contract / operator qualification (test-only).

Truth-known linear SDEs are sampled onto the production 8 s / 4 s grid
(and an 8 s / 8 s hop control), then scored with a time-blocked embargo
so Savitzky–Golay support does not leak across train/test.

This is not a production measurement module and is not wired into summarize.
A next-state map A is diagnostic only; it is not spectral abscissa.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from mndm.jacobian import _affine_rel_mse, _fit_ridge, estimate_local_jacobians
from mndm.projection import build_knn_indices, estimate_derivatives


DT_4_SEC = 4.0
DT_8_SEC = 8.0
SG_WINDOW = 7
SG_POLYORDER = 3
KNN_K = 20
SUPER_WINDOW = 3
RIDGE_ALPHA = 1.0
OOS_THRESHOLD = 0.9
TRAIN_FRAC = 0.6
DT_FINE = 0.05
N_GRID_4 = 320
PROCESS_SIGMA = 0.05

ICARE_0284_RUN = Path(
    r"K:\processed\physionet_icare_2_1_part1_0_12h_dynamical_families"
    r"\physionet_icare_2_1\neuralmanifolddynamics_physionet_icare_2_1_20260912_094016"
)

REGIMES: Dict[str, np.ndarray] = {
    "stable": np.array(
        [[-0.08, 0.02, 0.0], [-0.01, -0.06, 0.01], [0.0, 0.02, -0.05]],
        dtype=np.float64,
    ),
    "marginal": np.array(
        [[0.0, -0.06, 0.0], [0.06, 0.0, 0.0], [0.0, 0.0, -0.001]],
        dtype=np.float64,
    ),
    "unstable": np.array(
        [[0.0025, -0.01, 0.0], [0.005, 0.002, 0.0], [0.0, 0.0, -0.02]],
        dtype=np.float64,
    ),
    "nonnormal": np.array(
        [[-0.06, 3.5, 0.0], [0.0, -0.06, 0.0], [0.0, 0.0, -0.08]],
        dtype=np.float64,
    ),
}


def spectral_abscissa(jacobian: np.ndarray) -> float:
    eig = np.linalg.eigvals(np.asarray(jacobian, dtype=np.float64))
    return float(np.max(np.real(eig)))


def alpha_class(alpha: float, *, eps: float = 1e-3) -> str:
    if alpha < -float(eps):
        return "stable"
    if alpha > float(eps):
        return "unstable"
    return "marginal"


def embargo_samples(sg_window: int = SG_WINDOW, super_window: int = SUPER_WINDOW) -> int:
    return int(sg_window) // 2 + int(super_window) // 2


def blocked_slices(
    n: int,
    *,
    train_frac: float = TRAIN_FRAC,
    embargo: Optional[int] = None,
    min_side: int = 16,
) -> Tuple[slice, slice, int]:
    """Train prefix and test suffix with an unused embargo gap between them."""
    requested = embargo_samples() if embargo is None else int(embargo)
    cut = int(n * float(train_frac))
    candidates = [requested, max(SG_WINDOW // 2, 1), 1]
    for gap in candidates:
        train_end = cut - gap
        test_start = cut + gap
        if train_end >= int(min_side) and n - test_start >= int(min_side) and train_end + gap <= test_start:
            return slice(0, train_end), slice(test_start, n), int(gap)
    raise ValueError(f"blocked split too small for n={n}, embargo={requested}")


def slices_disjoint_with_embargo(n: int, train: slice, test: slice, embargo: int) -> bool:
    return int(train.stop) + int(embargo) <= int(test.start) and int(test.start) < n


def simulate_linear_sde(
    jacobian: np.ndarray,
    *,
    dt_fine: float,
    n_fine: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    j = np.asarray(jacobian, dtype=np.float64)
    dim = int(j.shape[0])
    x = np.zeros((n_fine, dim), dtype=np.float64)
    x[0] = rng.normal(size=dim)
    scale = float(sigma) * np.sqrt(float(dt_fine))
    for t in range(n_fine - 1):
        x[t + 1] = x[t] + float(dt_fine) * (j @ x[t]) + scale * rng.normal(size=dim)
        if not np.isfinite(x[t + 1]).all() or np.max(np.abs(x[t + 1])) > 1e6:
            x[t + 1 :] = np.nan
            break
    return x


def downsample_to_grid(x_fine: np.ndarray, *, dt_fine: float, grid_dt: float) -> np.ndarray:
    stride = int(round(float(grid_dt) / float(dt_fine)))
    if stride < 1:
        raise ValueError("grid_dt must be >= dt_fine")
    return np.asarray(x_fine[::stride], dtype=np.float32)


def _trim_sg_edges(x: np.ndarray, y: np.ndarray, window: int = SG_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    half = int(window) // 2
    if x.shape[0] <= 2 * half:
        return x[:0], y[:0]
    return x[half:-half], y[half:-half]


def fit_global_affine(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_alpha: float = RIDGE_ALPHA,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[finite]
    y = y[finite]
    if x.shape[0] < x.shape[1] + 1:
        return None
    x_mean = np.mean(x, axis=0, keepdims=True)
    design = x - x_mean
    col_scale = np.std(design, axis=0, ddof=0)
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0).astype(np.float32)
    design_std = design / col_scale[None, :]
    design_aug = np.hstack([design_std, np.ones((design.shape[0], 1), dtype=np.float32)])
    a, b = _fit_ridge(design_aug, y, ridge_alpha)
    mapped = (a / col_scale[None, :]).astype(np.float32)
    return mapped, b.astype(np.float32), x_mean.reshape(-1).astype(np.float32)


def _score_affine(
    x: np.ndarray,
    y: np.ndarray,
    mapped: np.ndarray,
    intercept: np.ndarray,
    reference: np.ndarray,
) -> float:
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if int(finite.sum()) < 2:
        return float("nan")
    return float(_affine_rel_mse(x[finite], y[finite], mapped, intercept, reference))


def production_local_j(x: np.ndarray, x_dot: np.ndarray, *, dt: float):
    nn_idx = build_knn_indices(x, k=KNN_K, metric="euclidean", whiten=True)
    return estimate_local_jacobians(
        x,
        x_dot,
        nn_idx,
        super_window=SUPER_WINDOW,
        ridge_alpha=RIDGE_ALPHA,
        distance_weighted=True,
        j_dot_dt=float(dt),
        knn_k=KNN_K,
    )


def qualify_grid_series(
    x: np.ndarray,
    *,
    dt: float,
    j_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Blocked-OOS J (SG and oracle ẋ) vs discrete A on one series."""
    x = np.asarray(x, dtype=np.float32)
    train_sl, test_sl, embargo = blocked_slices(int(x.shape[0]))
    x_train_raw, x_test_raw = x[train_sl], x[test_sl]
    xdot_train = estimate_derivatives(
        x_train_raw, float(dt), method="sav_gol", window=SG_WINDOW, polyorder=SG_POLYORDER
    )
    xdot_test = estimate_derivatives(
        x_test_raw, float(dt), method="sav_gol", window=SG_WINDOW, polyorder=SG_POLYORDER
    )
    x_train, xdot_train = _trim_sg_edges(x_train_raw, xdot_train)
    x_test, xdot_test = _trim_sg_edges(x_test_raw, xdot_test)

    jac = production_local_j(x_train, xdot_train, dt=dt)
    if jac.j_hat.shape[0] == 0 or jac.affine_intercept is None or jac.affine_reference is None:
        j_sg_rel = float("nan")
        alpha_hat = float("nan")
    else:
        j_mean = np.mean(jac.j_hat, axis=0)
        b_mean = np.mean(jac.affine_intercept, axis=0)
        ref_mean = np.mean(jac.affine_reference, axis=0)
        j_sg_rel = _score_affine(x_test, xdot_test, j_mean, b_mean, ref_mean)
        alpha_hat = spectral_abscissa(j_mean)

    j_sg_global_rel = float("nan")
    alpha_hat_global = float("nan")
    fitted_sg = fit_global_affine(x_train, xdot_train)
    if fitted_sg is not None:
        j_sg_global_rel = _score_affine(x_test, xdot_test, *fitted_sg)
        alpha_hat_global = spectral_abscissa(fitted_sg[0])

    oracle_rel = float("nan")
    if j_true is not None:
        j_true = np.asarray(j_true, dtype=np.float32)
        y_train = (x_train @ j_true.T).astype(np.float32)
        y_test = (x_test @ j_true.T).astype(np.float32)
        fitted = fit_global_affine(x_train, y_train)
        if fitted is not None:
            oracle_rel = _score_affine(x_test, y_test, *fitted)

    a_rel = float("nan")
    if x_train.shape[0] > 2 and x_test.shape[0] > 2:
        fitted_a = fit_global_affine(x_train[:-1], x_train[1:])
        if fitted_a is not None:
            a_rel = _score_affine(x_test[:-1], x_test[1:], *fitted_a)

    alpha_true = spectral_abscissa(j_true) if j_true is not None else float("nan")
    return {
        "dt_sec": float(dt),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "embargo_samples": int(embargo),
        "embargo_requested": int(embargo_samples()),
        "split_ok": slices_disjoint_with_embargo(int(x.shape[0]), train_sl, test_sl, embargo),
        "rel_mse_j_sg": float(j_sg_rel),
        "rel_mse_j_sg_global": float(j_sg_global_rel),
        "rel_mse_j_oracle_xdot": float(oracle_rel),
        "rel_mse_a_next": float(a_rel),
        "j_sg_oos_pass": bool(np.isfinite(j_sg_rel) and j_sg_rel < OOS_THRESHOLD),
        "j_sg_global_oos_pass": bool(np.isfinite(j_sg_global_rel) and j_sg_global_rel < OOS_THRESHOLD),
        "j_oracle_oos_pass": bool(np.isfinite(oracle_rel) and oracle_rel < OOS_THRESHOLD),
        "a_oos_pass": bool(np.isfinite(a_rel) and a_rel < OOS_THRESHOLD),
        "alpha_true": float(alpha_true),
        "alpha_hat": float(alpha_hat),
        "alpha_hat_global": float(alpha_hat_global),
        "alpha_true_class": alpha_class(alpha_true) if np.isfinite(alpha_true) else "unknown",
        "alpha_hat_class": alpha_class(alpha_hat) if np.isfinite(alpha_hat) else "unknown",
        "alpha_hat_global_class": (
            alpha_class(alpha_hat_global) if np.isfinite(alpha_hat_global) else "unknown"
        ),
        "alpha_class_match": (
            alpha_class(alpha_true) == alpha_class(alpha_hat)
            if np.isfinite(alpha_true) and np.isfinite(alpha_hat)
            else False
        ),
        "alpha_global_class_match": (
            alpha_class(alpha_true) == alpha_class(alpha_hat_global)
            if np.isfinite(alpha_true) and np.isfinite(alpha_hat_global)
            else False
        ),
        "local_j_windows": int(jac.j_hat.shape[0]) if jac.j_hat.ndim == 3 else 0,
    }


def run_truth_known(
    *,
    rng_seed: int = 602,
    grids: Sequence[float] = (DT_4_SEC, DT_8_SEC),
) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    duration_sec = N_GRID_4 * DT_4_SEC
    n_fine = int(round(duration_sec / DT_FINE))
    rows = []
    for name, j_true in REGIMES.items():
        x_fine = simulate_linear_sde(
            j_true, dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng
        )
        for grid_dt in grids:
            x = downsample_to_grid(x_fine, dt_fine=DT_FINE, grid_dt=float(grid_dt))
            row = qualify_grid_series(x, dt=float(grid_dt), j_true=j_true)
            row["regime"] = name
            rows.append(row)
    return {
        "threshold": OOS_THRESHOLD,
        "sigma": PROCESS_SIGMA,
        "rng_seed": int(rng_seed),
        "knn_k": KNN_K,
        "super_window": SUPER_WINDOW,
        "ridge_alpha": RIDGE_ALPHA,
        "sg_window": SG_WINDOW,
        "rows": rows,
    }


def _load_mnps_3d(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["mnps_3d"][()], dtype=np.float32)


def run_icare_overlap_ablation(
    run_dir: Path = ICARE_0284_RUN,
    *,
    pattern: str = "sub-0284_*.h5",
) -> Dict[str, Any]:
    files = sorted(run_dir.rglob(pattern))
    rows = []
    for path in files:
        x4 = _load_mnps_3d(path)
        q4 = qualify_grid_series(x4, dt=DT_4_SEC, j_true=None)
        try:
            q8 = qualify_grid_series(x4[::2], dt=DT_8_SEC, j_true=None)
        except ValueError as exc:
            q8 = {
                "dt_sec": DT_8_SEC,
                "skip_reason": str(exc),
                "rel_mse_j_sg": float("nan"),
                "rel_mse_j_sg_global": float("nan"),
                "rel_mse_a_next": float("nan"),
                "j_sg_oos_pass": False,
                "j_sg_global_oos_pass": False,
                "a_oos_pass": False,
            }
        rows.append(
            {
                "file": path.name,
                "grid_8_4": q4,
                "grid_8_8": q8,
            }
        )
    return {
        "run_dir": str(run_dir),
        "n_files": len(files),
        "threshold": OOS_THRESHOLD,
        "oos": "time_blocked_embargo",
        "knn_k": KNN_K,
        "ridge_alpha": RIDGE_ALPHA,
        "sg_window": SG_WINDOW,
        "rows": rows,
    }


def summarize_icare(report: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for grid in ("grid_8_4", "grid_8_8"):
        j_vals: list[float] = []
        jg_vals: list[float] = []
        a_vals: list[float] = []
        j_pass = 0
        jg_pass = 0
        a_pass = 0
        n = 0
        for row in report.get("rows", []):
            block = row[grid]
            n += 1
            j_vals.append(block["rel_mse_j_sg"])
            jg_vals.append(block["rel_mse_j_sg_global"])
            a_vals.append(block["rel_mse_a_next"])
            j_pass += int(bool(block["j_sg_oos_pass"]))
            jg_pass += int(bool(block["j_sg_global_oos_pass"]))
            a_pass += int(bool(block["a_oos_pass"]))
        finite_j = np.asarray(j_vals, dtype=np.float64)
        finite_j = finite_j[np.isfinite(finite_j)]
        finite_jg = np.asarray(jg_vals, dtype=np.float64)
        finite_jg = finite_jg[np.isfinite(finite_jg)]
        finite_a = np.asarray(a_vals, dtype=np.float64)
        finite_a = finite_a[np.isfinite(finite_a)]
        out[grid] = {
            "n": n,
            "j_sg_oos_median": float(np.median(finite_j)) if finite_j.size else float("nan"),
            "j_sg_global_oos_median": float(np.median(finite_jg)) if finite_jg.size else float("nan"),
            "a_oos_median": float(np.median(finite_a)) if finite_a.size else float("nan"),
            "j_sg_oos_pass": f"{j_pass}/{n}",
            "j_sg_global_oos_pass": f"{jg_pass}/{n}",
            "a_oos_pass": f"{a_pass}/{n}",
        }
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    import json

    truth = run_truth_known()
    print("truth-known")
    for row in truth["rows"]:
        print(
            f"  {row['regime']:10s} dt={row['dt_sec']:.0f} "
            f"J_SG_local={row['rel_mse_j_sg']:.3f} "
            f"J_SG_global={row['rel_mse_j_sg_global']:.3f} "
            f"J_oracle={row['rel_mse_j_oracle_xdot']:.3f} "
            f"A={row['rel_mse_a_next']:.3f} passA={row['a_oos_pass']} "
            f"alpha {row['alpha_true_class']}->{row['alpha_hat_class']} "
            f"match={row['alpha_class_match']}"
        )
    if ICARE_0284_RUN.exists():
        icare = run_icare_overlap_ablation()
        summary = summarize_icare(icare)
        print("0284 blocked overlap")
        for key, val in summary.items():
            print(
                f"  {key} J_local={val['j_sg_oos_median']:.3f} pass={val['j_sg_oos_pass']} "
                f"J_global={val['j_sg_global_oos_median']:.3f} pass={val['j_sg_global_oos_pass']} "
                f"A={val['a_oos_median']:.3f} pass={val['a_oos_pass']}"
            )
        out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl002_20260914"
        out.mkdir(parents=True, exist_ok=True)
        payload = {"truth_known": truth, "icare": icare, "icare_summary": summary}
        (out / "sl002_report.json").write_text(
            json.dumps(_json_ready(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"wrote {out / 'sl002_report.json'}")
    else:
        print("0284 run dir missing; synthetic only")


if __name__ == "__main__":
    main()

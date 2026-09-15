"""Family B SL-003: epoch generator qualification (test-only).

Compares a Savitzky–Golay epoch-J against a transition-domain generator

    x_{k+1} = F x_k + c,   J = log(F) / Δ

Primary licensing is on non-overlapping 8 s hops. Alpha is licensed only
when F beats the next-state baseline out of sample AND the real logarithm
is admissible. This is not wired into summarize.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

TESTS = Path(__file__).resolve().parent
_SRC = TESTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import numpy as np
from scipy.linalg import logm

from family_b_sl002 import (
    DT_4_SEC,
    DT_8_SEC,
    ICARE_0284_RUN,
    OOS_THRESHOLD,
    PROCESS_SIGMA,
    REGIMES,
    RIDGE_ALPHA,
    SG_POLYORDER,
    SG_WINDOW,
    _json_ready,
    _score_affine,
    alpha_class,
    downsample_to_grid,
    fit_global_affine,
    simulate_linear_sde,
    spectral_abscissa,
)
from mndm.projection import estimate_derivatives


STATUS_TRANSITION_OOS_FAIL = "TRANSITION_OOS_FAIL"
STATUS_TRANSITION_INSUFFICIENT = "TRANSITION_INSUFFICIENT"
STATUS_NO_REAL_GENERATOR = "NO_REAL_GENERATOR"
STATUS_LOG_BRANCH_UNSTABLE = "LOG_BRANCH_UNSTABLE"
STATUS_GENERATOR_ILL_CONDITIONED = "GENERATOR_ILL_CONDITIONED"
STATUS_VALID_EPOCH_GENERATOR = "VALID_EPOCH_GENERATOR"
STATUS_SG_OOS_FAIL = "SG_OOS_FAIL"
STATUS_VALID_SG_EPOCH_J = "VALID_SG_EPOCH_J"
STATUS_SG_INSUFFICIENT = "SG_INSUFFICIENT"

EPOCH_SEC = {"primary": 300.0, "sensitivity": 180.0, "exploratory": 120.0}
COND_MAX = 1e8
IMAG_TOL = 1e-8
BRANCH_ANGLE_DEG = 15.0
DT_FINE = 0.05
TRUTH_DURATION_SEC = 30 * 60.0


def n_epoch_samples(epoch_sec: float, dt: float) -> int:
    return max(4, int(float(epoch_sec) / float(dt)))


def consecutive_one_step_pairs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Adjacent finite pairs only. Never stitch across a missing sample."""
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x).all(axis=1)
    src = []
    tgt = []
    for i in range(int(x.shape[0]) - 1):
        if bool(finite[i]) and bool(finite[i + 1]):
            src.append(x[i])
            tgt.append(x[i + 1])
    if not src:
        empty = np.empty((0, int(x.shape[1])), dtype=np.float32)
        return empty, empty
    return np.stack(src, axis=0), np.stack(tgt, axis=0)


def pair_blocked_slices(
    n_pairs: int,
    *,
    train_frac: float = 0.6,
    min_train: int = 12,
    min_test: int = 4,
    gap: int = 1,
) -> Tuple[slice, slice, int]:
    n_pairs = int(n_pairs)
    embargo = int(gap)
    if embargo < 1:
        raise ValueError("blocked OOS requires a positive pair embargo")
    cut = int(n_pairs * float(train_frac))
    train_end = cut - embargo
    test_start = cut + embargo
    if train_end >= int(min_train) and n_pairs - test_start >= int(min_test):
        return slice(0, train_end), slice(test_start, n_pairs), embargo
    raise ValueError(f"epoch pair split too small for n_pairs={n_pairs}")


def homogeneous_F(mapped: np.ndarray, intercept: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Convert y ≈ B(x − x̄) + b into x+ ≈ F x + c, returning F (= B)."""
    return np.asarray(mapped, dtype=np.float64)


def real_generator_from_F(
    F: np.ndarray,
    dt: float,
    *,
    cond_max: float = COND_MAX,
    imag_tol: float = IMAG_TOL,
    branch_angle_deg: float = BRANCH_ANGLE_DEG,
) -> Tuple[Optional[np.ndarray], str]:
    """Principal real logarithm J = log(F)/Δ, or an explicit failure status."""
    F = np.asarray(F, dtype=np.float64)
    if F.shape != (F.shape[0], F.shape[0]) or F.shape[0] < 1:
        return None, STATUS_NO_REAL_GENERATOR
    if not np.isfinite(F).all():
        return None, STATUS_NO_REAL_GENERATOR
    eigs = np.linalg.eigvals(F)
    if np.any(np.abs(eigs) < 1e-12):
        return None, STATUS_NO_REAL_GENERATOR
    branch_limit = np.pi - np.deg2rad(float(branch_angle_deg))
    for lam in eigs:
        if lam.real < 0.0 and abs(lam.imag) <= 1e-10:
            return None, STATUS_NO_REAL_GENERATOR
        if lam.real < 0.0 and abs(np.angle(lam)) >= branch_limit:
            return None, STATUS_LOG_BRANCH_UNSTABLE
    try:
        cond_f = float(np.linalg.cond(F))
    except np.linalg.LinAlgError:
        return None, STATUS_GENERATOR_ILL_CONDITIONED
    if not np.isfinite(cond_f) or cond_f > float(cond_max):
        return None, STATUS_GENERATOR_ILL_CONDITIONED
    log_f = logm(F)
    imag = np.max(np.abs(np.imag(np.asarray(log_f)))) if np.iscomplexobj(log_f) else 0.0
    if float(imag) > float(imag_tol):
        return None, STATUS_NO_REAL_GENERATOR
    jacobian = np.real(np.asarray(log_f, dtype=np.complex128)) / float(dt)
    if not np.isfinite(jacobian).all():
        return None, STATUS_NO_REAL_GENERATOR
    try:
        cond_j = float(np.linalg.cond(jacobian))
    except np.linalg.LinAlgError:
        return None, STATUS_GENERATOR_ILL_CONDITIONED
    if not np.isfinite(cond_j) or cond_j > float(cond_max):
        return None, STATUS_GENERATOR_ILL_CONDITIONED
    return jacobian.astype(np.float64), STATUS_VALID_EPOCH_GENERATOR


def _transition_row(
    *,
    status: str,
    family_b_scope: bool,
    rel_mse_oos: float = float("nan"),
    alpha: float = float("nan"),
    embargo_pairs: int = 0,
    n_pairs: int = 0,
    F_condition: float = float("nan"),
) -> Dict[str, Any]:
    generator_qualified = status == STATUS_VALID_EPOCH_GENERATOR
    return {
        "status": status,
        "rel_mse_oos": float(rel_mse_oos),
        "alpha": float(alpha),
        "generator_qualified": bool(generator_qualified),
        "alpha_licensed": bool(generator_qualified),
        "family_b_alpha_licensed": bool(family_b_scope and generator_qualified),
        "embargo_pairs": int(embargo_pairs),
        "n_pairs": int(n_pairs),
        "F_condition": float(F_condition),
    }


def qualify_transition_epoch(
    x: np.ndarray,
    *,
    dt: float,
    family_b_scope: bool = False,
) -> Dict[str, Any]:
    src, tgt = consecutive_one_step_pairs(x)
    n_pairs = int(src.shape[0])
    try:
        train_sl, test_sl, gap = pair_blocked_slices(n_pairs)
    except ValueError:
        return _transition_row(
            status=STATUS_TRANSITION_INSUFFICIENT,
            family_b_scope=family_b_scope,
            n_pairs=n_pairs,
        )
    fitted = fit_global_affine(src[train_sl], tgt[train_sl])
    if fitted is None:
        return _transition_row(
            status=STATUS_TRANSITION_OOS_FAIL,
            family_b_scope=family_b_scope,
            embargo_pairs=gap,
            n_pairs=n_pairs,
        )
    rel = _score_affine(src[test_sl], tgt[test_sl], *fitted)
    if not np.isfinite(rel) or float(rel) >= OOS_THRESHOLD:
        return _transition_row(
            status=STATUS_TRANSITION_OOS_FAIL,
            family_b_scope=family_b_scope,
            rel_mse_oos=float(rel) if np.isfinite(rel) else float("nan"),
            embargo_pairs=gap,
            n_pairs=n_pairs,
        )
    F = homogeneous_F(*fitted)
    jacobian, status = real_generator_from_F(F, float(dt))
    alpha = spectral_abscissa(jacobian) if jacobian is not None else float("nan")
    return _transition_row(
        status=status,
        family_b_scope=family_b_scope,
        rel_mse_oos=float(rel),
        alpha=float(alpha),
        embargo_pairs=gap,
        n_pairs=n_pairs,
        F_condition=float(np.linalg.cond(F)),
    )


def qualify_sg_epoch(
    x: np.ndarray,
    *,
    dt: float,
    family_b_scope: bool = False,
) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float32)
    n = int(x.shape[0])
    if n < 1 or not np.isfinite(x).all():
        return {
            "status": STATUS_SG_INSUFFICIENT,
            "rel_mse_oos": float("nan"),
            "alpha": float("nan"),
            "alpha_licensed": False,
            "family_b_alpha_licensed": False,
        }
    half = SG_WINDOW // 2
    cut = int(0.6 * n)
    train_end = cut - half
    test_start = cut + half
    if train_end < 12 or n - test_start < 8:
        return {
            "status": STATUS_SG_INSUFFICIENT,
            "rel_mse_oos": float("nan"),
            "alpha": float("nan"),
            "alpha_licensed": False,
            "family_b_alpha_licensed": False,
        }
    x_train = x[:train_end]
    x_test = x[test_start:]
    y_train = estimate_derivatives(
        x_train, float(dt), method="sav_gol", window=SG_WINDOW, polyorder=SG_POLYORDER
    )
    y_test = estimate_derivatives(
        x_test, float(dt), method="sav_gol", window=SG_WINDOW, polyorder=SG_POLYORDER
    )
    if x_train.shape[0] > 2 * half:
        x_train, y_train = x_train[half:-half], y_train[half:-half]
    if x_test.shape[0] > 2 * half:
        x_test, y_test = x_test[half:-half], y_test[half:-half]
    fitted = fit_global_affine(x_train, y_train)
    if fitted is None:
        return {
            "status": STATUS_SG_OOS_FAIL,
            "rel_mse_oos": float("nan"),
            "alpha": float("nan"),
            "alpha_licensed": False,
            "family_b_alpha_licensed": False,
        }
    rel = _score_affine(x_test, y_test, *fitted)
    if not np.isfinite(rel) or float(rel) >= OOS_THRESHOLD:
        return {
            "status": STATUS_SG_OOS_FAIL,
            "rel_mse_oos": float(rel) if np.isfinite(rel) else float("nan"),
            "alpha": float("nan"),
            "alpha_licensed": False,
            "family_b_alpha_licensed": False,
        }
    residual_pass = True
    return {
        "status": STATUS_VALID_SG_EPOCH_J,
        "rel_mse_oos": float(rel),
        "alpha": float(spectral_abscissa(fitted[0])),
        "alpha_licensed": residual_pass,
        "family_b_alpha_licensed": False,
    }


def epoch_windows(x: np.ndarray, n_epoch: int) -> List[np.ndarray]:
    x = np.asarray(x)
    out = []
    start = 0
    while start + int(n_epoch) <= int(x.shape[0]):
        out.append(x[start : start + int(n_epoch)])
        start += int(n_epoch)
    return out


def qualify_series_epochs(
    x: np.ndarray,
    *,
    dt: float,
    epoch_sec: float,
    j_true: Optional[np.ndarray] = None,
    family_b_scope: bool = False,
) -> Dict[str, Any]:
    n_epoch = n_epoch_samples(epoch_sec, dt)
    rows = []
    for epoch in epoch_windows(x, n_epoch):
        trans = qualify_transition_epoch(epoch, dt=dt, family_b_scope=family_b_scope)
        sg = qualify_sg_epoch(epoch, dt=dt, family_b_scope=False)
        trans_alpha_match = False
        if j_true is not None and trans["alpha_licensed"]:
            trans_alpha_match = alpha_class(trans["alpha"]) == alpha_class(spectral_abscissa(j_true))
        sg_alpha_match = False
        if j_true is not None and sg["alpha_licensed"]:
            sg_alpha_match = alpha_class(sg["alpha"]) == alpha_class(spectral_abscissa(j_true))
        rows.append(
            {
                "transition": trans,
                "sg": sg,
                "transition_alpha_class_match": trans_alpha_match,
                "sg_alpha_class_match": sg_alpha_match,
            }
        )
    trans_status = Counter(row["transition"]["status"] for row in rows)
    sg_status = Counter(row["sg"]["status"] for row in rows)
    licensed = [row["transition"]["alpha"] for row in rows if row["transition"]["alpha_licensed"]]
    return {
        "dt_sec": float(dt),
        "epoch_sec": float(epoch_sec),
        "n_epoch_samples": int(n_epoch),
        "n_epochs": len(rows),
        "transition_status_counts": dict(trans_status),
        "sg_status_counts": dict(sg_status),
        "family_b_scope": bool(family_b_scope),
        "n_generator_qualified": int(sum(row["transition"]["generator_qualified"] for row in rows)),
        "n_family_b_alpha_licensed": int(
            sum(row["transition"]["family_b_alpha_licensed"] for row in rows)
        ),
        "n_transition_licensed": int(sum(row["transition"]["generator_qualified"] for row in rows)),
        "n_sg_residual_pass": int(sum(row["sg"]["alpha_licensed"] for row in rows)),
        "n_sg_licensed": int(sum(row["sg"]["alpha_licensed"] for row in rows)),
        "n_transition_alpha_match": int(sum(row["transition_alpha_class_match"] for row in rows)),
        "n_sg_alpha_match": int(sum(row["sg_alpha_class_match"] for row in rows)),
        "licensed_alpha_median": float(np.median(licensed)) if licensed else float("nan"),
        "epochs": rows,
    }


def run_truth_known(*, rng_seed: int = 703) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    n_fine = int(round(TRUTH_DURATION_SEC / DT_FINE))
    rows = []
    for name, j_true in REGIMES.items():
        x_fine = simulate_linear_sde(
            j_true, dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng
        )
        x8 = downsample_to_grid(x_fine, dt_fine=DT_FINE, grid_dt=DT_8_SEC)
        x4 = downsample_to_grid(x_fine, dt_fine=DT_FINE, grid_dt=DT_4_SEC)
        for label, epoch_sec in EPOCH_SEC.items():
            q8 = qualify_series_epochs(x8, dt=DT_8_SEC, epoch_sec=epoch_sec, j_true=j_true)
            q8["regime"] = name
            q8["grid"] = "8/8"
            q8["epoch_role"] = label
            rows.append(q8)
            if label == "primary":
                q4 = qualify_series_epochs(
                    x4, dt=DT_4_SEC, epoch_sec=epoch_sec, j_true=j_true, family_b_scope=False
                )
                q4["regime"] = name
                q4["grid"] = "8/4"
                q4["epoch_role"] = "sensitivity_overlap"
                rows.append(q4)
    return {
        "threshold": OOS_THRESHOLD,
        "rng_seed": int(rng_seed),
        "primary_grid": "8/8",
        "epoch_sec": EPOCH_SEC,
        "rows": rows,
    }


def _load_mnps_3d(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["mnps_3d"][()], dtype=np.float32)


def run_icare_epochs(
    run_dir: Path = ICARE_0284_RUN,
    *,
    pattern: str = "sub-0284_*.h5",
) -> Dict[str, Any]:
    files = sorted(run_dir.rglob(pattern))
    rows = []
    for path in files:
        x4 = _load_mnps_3d(path)
        x8 = x4[::2]
        q8 = qualify_series_epochs(
            x8, dt=DT_8_SEC, epoch_sec=EPOCH_SEC["primary"], family_b_scope=True
        )
        q4 = qualify_series_epochs(
            x4, dt=DT_4_SEC, epoch_sec=EPOCH_SEC["primary"], family_b_scope=False
        )
        rows.append({"file": path.name, "grid_8_8": q8, "grid_8_4": q4})
    return {
        "run_dir": str(run_dir),
        "n_files": len(files),
        "threshold": OOS_THRESHOLD,
        "primary_grid": "8/8",
        "epoch_sec_primary": EPOCH_SEC["primary"],
        "rows": rows,
    }


def summarize_icare(report: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for grid in ("grid_8_8", "grid_8_4"):
        trans = Counter()
        sg = Counter()
        n_epochs = 0
        n_files = 0
        trans_rel: List[float] = []
        sg_rel: List[float] = []
        n_f_oos_pass = 0
        n_family_b = 0
        for row in report.get("rows", []):
            n_files += 1
            block = row[grid]
            n_epochs += int(block.get("n_epochs", 0))
            trans.update(block.get("transition_status_counts", {}))
            sg.update(block.get("sg_status_counts", {}))
            for epoch in block.get("epochs", []):
                f_rel = epoch["transition"].get("rel_mse_oos")
                s_rel = epoch["sg"].get("rel_mse_oos")
                if isinstance(f_rel, (int, float)) and np.isfinite(f_rel):
                    trans_rel.append(float(f_rel))
                    if float(f_rel) < OOS_THRESHOLD:
                        n_f_oos_pass += 1
                if isinstance(s_rel, (int, float)) and np.isfinite(s_rel):
                    sg_rel.append(float(s_rel))
                if bool(epoch["transition"].get("family_b_alpha_licensed")):
                    n_family_b += 1
        n_qualified = int(trans.get(STATUS_VALID_EPOCH_GENERATOR, 0))
        n_sg_pass = int(sg.get(STATUS_VALID_SG_EPOCH_J, 0))
        out[grid] = {
            "n_files": n_files,
            "n_epochs": n_epochs,
            "transition_status_counts": dict(trans),
            "sg_status_counts": dict(sg),
            "n_generator_qualified": n_qualified,
            "n_family_b_alpha_licensed": n_family_b,
            "n_sg_residual_pass": n_sg_pass,
            "n_transition_oos_pass": int(n_f_oos_pass),
            "transition_rel_mse_median": float(np.median(trans_rel)) if trans_rel else float("nan"),
            "sg_rel_mse_median": float(np.median(sg_rel)) if sg_rel else float("nan"),
        }
    return out


def main() -> None:
    import json

    truth = run_truth_known()
    print("truth-known 8/8 primary (5 min)")
    for row in truth["rows"]:
        if row.get("epoch_role") != "primary":
            continue
        print(
            f"  {row['regime']:10s} epochs={row['n_epochs']} "
            f"F_licensed={row['n_transition_licensed']}/{row['n_epochs']} "
            f"F_match={row['n_transition_alpha_match']} "
            f"SG_licensed={row['n_sg_licensed']}/{row['n_epochs']} "
            f"F_status={row['transition_status_counts']} "
            f"SG_status={row['sg_status_counts']}"
        )
    if ICARE_0284_RUN.exists():
        icare = run_icare_epochs()
        summary = summarize_icare(icare)
        print("0284 5-min epochs")
        for key, val in summary.items():
            print(
                f"  {key} epochs={val['n_epochs']} "
                f"family_b_alpha={val['n_family_b_alpha_licensed']} "
                f"F_qualified={val['n_generator_qualified']} "
                f"SG_residual={val['n_sg_residual_pass']} "
                f"F={val['transition_status_counts']} "
                f"SG={val['sg_status_counts']}"
            )
        out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl003_20260914"
        out.mkdir(parents=True, exist_ok=True)
        payload = {"truth_known": truth, "icare": icare, "icare_summary": summary}
        (out / "sl003_report.json").write_text(
            json.dumps(_json_ready(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"wrote {out / 'sl003_report.json'}")
    else:
        print("0284 run dir missing; synthetic only")


if __name__ == "__main__":
    main()

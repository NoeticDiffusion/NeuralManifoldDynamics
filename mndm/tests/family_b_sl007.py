"""Family B SL-007: history-augmented Type-C operator F_H (test-only).

Builds the companion of the SL-006 M1 map

    y_{t+1} = F_H y_t + c̃,   y_t = [z_t, z_{t-1}],
    F_H = [[B0, B1], [I, 0]]

and asks whether F_H is an identified finite-time operator, not whether
it has a matrix logarithm. Central test: held-out multi-step
propagation at h=1,2,4 versus persistence, reversed order, and a
shuffle-fit companion.

No alpha, no logm, no CPC. Not wired into summarize.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

TESTS = Path(__file__).resolve().parent
_SRC = TESTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import numpy as np

from family_b_sl002 import (
    DT_FINE,
    OOS_THRESHOLD,
    PROCESS_SIGMA,
    RIDGE_ALPHA,
    _json_ready,
    _score_affine,
    downsample_to_grid,
    fit_global_affine,
    simulate_linear_sde,
)
from family_b_sl003 import (
    EPOCH_SEC,
    STATUS_TRANSITION_INSUFFICIENT,
    STATUS_TRANSITION_OOS_FAIL,
    epoch_windows,
    n_epoch_samples,
)
from family_b_sl004 import DT_2_SEC, TYPE_C_STEP_SEC
from family_b_sl005 import (
    BROAD_COVERAGE_P_R,
    NAME_TYPE_C_STATE,
    PANELS,
    PRIMARY_PANEL,
    STATUS_TRANSITION_OOS_PASS,
    TYPE_C_FEATURES_DIR,
    TYPE_C_OVERLAY,
    TRUTH_DURATION_SEC,
    apply_type_c_transforms,
    assert_not_mnps_state,
    coverage_summary,
    drop_cpc_columns,
    _load_type_c_features,
    _recording_groups,
)
from family_b_sl006 import (
    M1_N_PARAMS,
    N_TARGET,
    _shuffle_lag_block,
    history_split,
    simulate_lagged_linear,
)

HORIZONS = (1, 2, 4)
MODELS_H = ("fh", "persist", "reverse", "shuffle")
N_AUG = 12
N_BOOTSTRAP = 8
EQUIV_ATOL = 1e-5
OPERATOR_NAME = "history_augmented_type_c_operator"
SHUFFLE_SEED = 907


def affine_to_companion(
    mapped: np.ndarray,
    intercept: np.ndarray,
    x_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Companion F_H and intercept for y=[z_t, z_{t-1}]. Not a generator."""
    mapped = np.asarray(mapped, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64).reshape(-1)
    x_mean = np.asarray(x_mean, dtype=np.float64).reshape(-1)
    if mapped.shape != (N_TARGET, N_AUG):
        raise ValueError(f"M1 map must be {N_TARGET}x{N_AUG}, got {mapped.shape}")
    c6 = intercept - mapped @ x_mean
    f_h = np.zeros((N_AUG, N_AUG), dtype=np.float64)
    f_h[:N_TARGET, :] = mapped
    f_h[N_TARGET:, :N_TARGET] = np.eye(N_TARGET, dtype=np.float64)
    c_aug = np.zeros(N_AUG, dtype=np.float64)
    c_aug[:N_TARGET] = c6
    return f_h, c_aug


def companion_predict_z(
    f_h: np.ndarray,
    c_aug: np.ndarray,
    now: np.ndarray,
    lag: np.ndarray,
    horizon: int,
) -> np.ndarray:
    now = np.asarray(now, dtype=np.float64)
    lag = np.asarray(lag, dtype=np.float64)
    if now.ndim == 1:
        now = now.reshape(1, -1)
        lag = lag.reshape(1, -1)
    y = np.hstack([now, lag])
    shift = np.asarray(c_aug, dtype=np.float64).reshape(1, -1)
    f_t = np.asarray(f_h, dtype=np.float64).T
    for _ in range(int(horizon)):
        y = y @ f_t + shift
        bad = ~np.isfinite(y).all(axis=1) | (np.max(np.abs(y), axis=1) > 1e8)
        if np.any(bad):
            y[bad] = np.nan
    return y[:, :N_TARGET].astype(np.float32)


def affine_predict_z(
    mapped: np.ndarray,
    intercept: np.ndarray,
    x_mean: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    mapped = np.asarray(mapped, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64).reshape(-1)
    x_mean = np.asarray(x_mean, dtype=np.float64).reshape(-1)
    return ((pred - x_mean) @ mapped.T + intercept).astype(np.float32)


def reversed_test_block(z: np.ndarray, test_times: set) -> Tuple[np.ndarray, set]:
    """Contiguous held-out block, time-reversed. Column-swapping [z_{t-1},z_t]
    is algebraically the same 12D one-step map; reversing the trajectory is
    the directional control.
    """
    times = sorted(int(t) for t in test_times)
    if not times:
        empty = np.empty((0, int(z.shape[1])), dtype=np.float32)
        return empty, set()
    if times != list(range(times[0], times[-1] + 1)):
        empty = np.empty((0, int(z.shape[1])), dtype=np.float32)
        return empty, set()
    block = np.asarray(z[times[0] : times[-1] + 1], dtype=np.float32)
    return block[::-1].copy(), set(range(int(block.shape[0])))


def rel_mse_z(pred: np.ndarray, actual: np.ndarray) -> float:
    """Relative MSE vs test-set mean. Any nonfinite rollout fails the score."""
    pred = np.asarray(pred, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if pred.shape[0] != actual.shape[0] or pred.shape[0] < 2:
        return float("nan")
    pred_ok = np.isfinite(pred).all(axis=1)
    actual_ok = np.isfinite(actual).all(axis=1)
    if int((~pred_ok).sum()) > 0:
        return float("nan")
    finite = pred_ok & actual_ok
    if int(finite.sum()) < 2:
        return float("nan")
    target = actual[finite]
    residual = target - pred[finite]
    mse_model = float(np.mean(residual**2))
    baseline = target - np.mean(target, axis=0, keepdims=True)
    mse_baseline = float(np.mean(baseline**2))
    if not np.isfinite(mse_baseline) or mse_baseline <= 1e-12:
        return float("nan")
    return float(mse_model / mse_baseline)


def spectral_radius(f_h: np.ndarray) -> float:
    """Discrete |λ|_max of F_H. Not Re λ of log(F)/Δ."""
    eig = np.linalg.eigvals(np.asarray(f_h, dtype=np.float64))
    if eig.size == 0 or not np.isfinite(eig).all():
        return float("nan")
    return float(np.max(np.abs(eig)))


def frobenius_rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = 0.5 * (np.linalg.norm(a, ord="fro") + np.linalg.norm(b, ord="fro"))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(a - b, ord="fro") / denom)


def consecutive_triples_indexed(
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite (z_{t-1}, z_t, z_{t+1}) plus original now-index t. No stitching."""
    z = np.asarray(z, dtype=np.float32)
    finite = np.isfinite(z).all(axis=1)
    lag, now, nxt, t_now = [], [], [], []
    for t in range(1, int(z.shape[0]) - 1):
        if bool(finite[t - 1]) and bool(finite[t]) and bool(finite[t + 1]):
            lag.append(z[t - 1])
            now.append(z[t])
            nxt.append(z[t + 1])
            t_now.append(t)
    if not now:
        empty = np.empty((0, int(z.shape[1])), dtype=np.float32)
        return empty, empty, empty, np.empty((0,), dtype=np.int32)
    return np.stack(lag), np.stack(now), np.stack(nxt), np.asarray(t_now, dtype=np.int32)


def original_triple_times(t_now: np.ndarray) -> set:
    times: set = set()
    for t in np.asarray(t_now, dtype=int).ravel():
        times.update((int(t) - 1, int(t), int(t) + 1))
    return times


def horizon_starts(
    z: np.ndarray,
    test_times: set,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Held-out starts whose entire span {t-1,...,t+h} is finite and in test."""
    z = np.asarray(z, dtype=np.float32)
    h = int(horizon)
    n = int(z.shape[0])
    dim = int(z.shape[1])
    lags, nows, tgts, t_list = [], [], [], []
    finite = np.isfinite(z).all(axis=1)
    for t in range(1, n - h):
        span = range(t - 1, t + h + 1)
        if any(int(k) not in test_times for k in span):
            continue
        if not all(bool(finite[k]) for k in span):
            continue
        lags.append(z[t - 1])
        nows.append(z[t])
        tgts.append(z[t + h])
        t_list.append(t)
    if not nows:
        empty = np.empty((0, dim), dtype=np.float32)
        return empty, empty, empty, np.empty((0,), dtype=np.int32)
    return np.stack(lags), np.stack(nows), np.stack(tgts), np.asarray(t_list, dtype=np.int32)


def _score_pack(rel: float) -> Dict[str, Any]:
    oos_pass = bool(np.isfinite(rel) and float(rel) < OOS_THRESHOLD)
    return {
        "rel_mse_oos": float(rel) if np.isfinite(rel) else float("nan"),
        "oos_pass": oos_pass,
        "status": STATUS_TRANSITION_OOS_PASS if oos_pass else STATUS_TRANSITION_OOS_FAIL,
    }


def _empty_horizon_models() -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "rel_mse_oos": float("nan"),
            "oos_pass": False,
            "status": STATUS_TRANSITION_INSUFFICIENT,
            "n_starts": 0,
        }
        for name in MODELS_H
    }


def operator_stability(
    y_train: np.ndarray,
    tgt_train: np.ndarray,
    f_full: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n_tr = int(y_train.shape[0])
    mid = n_tr // 2
    fro_halves = float("nan")
    rho_halves = float("nan")
    if mid >= N_AUG + 1 and n_tr - mid >= N_AUG + 1:
        a = fit_global_affine(y_train[:mid], tgt_train[:mid], ridge_alpha=RIDGE_ALPHA)
        b = fit_global_affine(y_train[mid:], tgt_train[mid:], ridge_alpha=RIDGE_ALPHA)
        if a is not None and b is not None:
            fa, _ = affine_to_companion(*a)
            fb, _ = affine_to_companion(*b)
            fro_halves = frobenius_rel(fa, fb)
            rho_halves = float(np.median([spectral_radius(fa), spectral_radius(fb)]))
    boot = []
    rho_boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_tr, size=n_tr)
        fitted = fit_global_affine(y_train[idx], tgt_train[idx], ridge_alpha=RIDGE_ALPHA)
        if fitted is None:
            continue
        fb, _ = affine_to_companion(*fitted)
        boot.append(frobenius_rel(fb, f_full))
        rho_boot.append(spectral_radius(fb))
    boot_arr = np.asarray(boot, dtype=float)
    boot_arr = boot_arr[np.isfinite(boot_arr)]
    rho_boot_arr = np.asarray(rho_boot, dtype=float)
    rho_boot_arr = rho_boot_arr[np.isfinite(rho_boot_arr)]
    return {
        "fro_rel_halves": float(fro_halves) if np.isfinite(fro_halves) else float("nan"),
        "fro_rel_bootstrap_median": float(np.median(boot_arr)) if boot_arr.size else float("nan"),
        "n_bootstrap": int(boot_arr.size),
        "rho": spectral_radius(f_full),
        "rho_halves_median": float(rho_halves) if np.isfinite(rho_halves) else float("nan"),
        "rho_bootstrap_median": float(np.median(rho_boot_arr)) if rho_boot_arr.size else float("nan"),
        "alpha_computed": False,
        "logm_computed": False,
    }


def qualify_operator_epoch(
    z: np.ndarray,
    *,
    columns: Sequence[str],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    assert_not_mnps_state(columns)
    z = apply_type_c_transforms(z, columns)
    lag, now, nxt, t_now = consecutive_triples_indexed(z)
    n_triples = int(now.shape[0])
    n_dim = int(now.shape[1]) if now.ndim == 2 else 0
    base = {
        "n_triples": n_triples,
        "n_dim": n_dim,
        "operator_name": OPERATOR_NAME,
        "state_name": NAME_TYPE_C_STATE,
    }
    try:
        train_sl, test_sl, gap = history_split(n_triples, n_dim)
    except ValueError:
        return {
            **base,
            "status": STATUS_TRANSITION_INSUFFICIENT,
            "embargo_pairs": 0,
            "n_train": 0,
            "n_test": 0,
            "equivalence_ok": False,
            "horizons": {str(h): _empty_horizon_models() for h in HORIZONS},
            "stability": {
                "fro_rel_halves": float("nan"),
                "fro_rel_bootstrap_median": float("nan"),
                "n_bootstrap": 0,
                "rho": float("nan"),
                "rho_halves_median": float("nan"),
                "rho_bootstrap_median": float("nan"),
                "alpha_computed": False,
                "logm_computed": False,
            },
        }
    y = np.hstack([now, lag])
    y_tr, y_te = y[train_sl], y[test_sl]
    tgt_tr, tgt_te = nxt[train_sl], nxt[test_sl]
    fitted = fit_global_affine(y_tr, tgt_tr, ridge_alpha=RIDGE_ALPHA)
    y_sh_tr = _shuffle_lag_block(y_tr, rng)
    fitted_sh = fit_global_affine(y_sh_tr, tgt_tr, ridge_alpha=RIDGE_ALPHA)
    if fitted is None:
        return {
            **base,
            "status": STATUS_TRANSITION_OOS_FAIL,
            "embargo_pairs": int(gap),
            "n_train": int(tgt_tr.shape[0]),
            "n_test": int(tgt_te.shape[0]),
            "equivalence_ok": False,
            "horizons": {str(h): _empty_horizon_models() for h in HORIZONS},
            "stability": {
                "fro_rel_halves": float("nan"),
                "fro_rel_bootstrap_median": float("nan"),
                "n_bootstrap": 0,
                "rho": float("nan"),
                "rho_halves_median": float("nan"),
                "rho_bootstrap_median": float("nan"),
                "alpha_computed": False,
                "logm_computed": False,
            },
        }
    mapped, intercept, x_mean = fitted
    f_h, c_aug = affine_to_companion(mapped, intercept, x_mean)
    pred_m1 = affine_predict_z(mapped, intercept, x_mean, y_te)
    pred_fh1 = companion_predict_z(f_h, c_aug, now[test_sl], lag[test_sl], 1)
    m1_rel = _score_affine(y_te, tgt_te, mapped, intercept, x_mean)
    fh1_rel = rel_mse_z(pred_fh1, tgt_te)
    equiv_err = float(np.max(np.abs(pred_m1 - pred_fh1))) if pred_m1.size else float("nan")
    equivalence_ok = bool(
        np.isfinite(equiv_err)
        and equiv_err < 1e-4
        and np.isfinite(m1_rel)
        and np.isfinite(fh1_rel)
        and abs(float(m1_rel) - float(fh1_rel)) < EQUIV_ATOL
    )
    test_times = original_triple_times(t_now[test_sl])
    z_rev, rev_times = reversed_test_block(z, test_times)
    horizons: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        if h == 1:
            lag_h, now_h, tgt_h = lag[test_sl], now[test_sl], tgt_te
        else:
            lag_h, now_h, tgt_h, _ = horizon_starts(z, test_times, h)
        lag_r, now_r, tgt_r, _ = horizon_starts(z_rev, rev_times, h)
        n_starts = int(now_h.shape[0])
        n_rev = int(now_r.shape[0])
        models = {}
        if n_starts < 2:
            models = _empty_horizon_models()
            for pack in models.values():
                pack["n_starts"] = n_starts
        else:
            pred_fh = companion_predict_z(f_h, c_aug, now_h, lag_h, h)
            pred_persist = now_h.astype(np.float32)
            if fitted_sh is not None:
                f_sh, c_sh = affine_to_companion(*fitted_sh)
                pred_sh = companion_predict_z(f_sh, c_sh, now_h, lag_h, h)
            else:
                pred_sh = np.full_like(tgt_h, np.nan)
            if n_rev >= 2:
                pred_rev = companion_predict_z(f_h, c_aug, now_r, lag_r, h)
                rev_rel = rel_mse_z(pred_rev, tgt_r)
            else:
                rev_rel = float("nan")
            models["fh"] = {**_score_pack(rel_mse_z(pred_fh, tgt_h)), "n_starts": n_starts}
            models["persist"] = {**_score_pack(rel_mse_z(pred_persist, tgt_h)), "n_starts": n_starts}
            models["reverse"] = {**_score_pack(rev_rel), "n_starts": n_rev}
            models["shuffle"] = {**_score_pack(rel_mse_z(pred_sh, tgt_h)), "n_starts": n_starts}
        horizons[str(h)] = models
    stability = operator_stability(y_tr, tgt_tr, f_h, rng)
    return {
        **base,
        "status": "SCORED",
        "embargo_pairs": int(gap),
        "n_train": int(tgt_tr.shape[0]),
        "n_test": int(tgt_te.shape[0]),
        "equivalence_ok": equivalence_ok,
        "equivalence_max_abs": float(equiv_err) if np.isfinite(equiv_err) else float("nan"),
        "m1_rel_mse": float(m1_rel) if np.isfinite(m1_rel) else float("nan"),
        "horizons": horizons,
        "stability": stability,
        "alpha_computed": False,
        "logm_computed": False,
        "cpc_inspected": False,
    }


def _model_recording_stats(
    epoch_rows: Sequence[Mapping[str, Any]],
    horizon: int,
    model: str,
) -> Dict[str, Any]:
    eligible = []
    for row in epoch_rows:
        if row.get("status") == STATUS_TRANSITION_INSUFFICIENT:
            continue
        pack = row.get("horizons", {}).get(str(horizon), {}).get(model)
        if pack is None:
            continue
        eligible.append(pack)
    n_pass = int(sum(bool(m["oos_pass"]) for m in eligible))
    n_eligible = len(eligible)
    rels = [
        1.0 if not np.isfinite(m.get("rel_mse_oos", np.nan)) else float(m["rel_mse_oos"])
        for m in eligible
    ]
    p_r = float(n_pass / n_eligible) if n_eligible else float("nan")
    return {
        "n_epochs": len(epoch_rows),
        "n_eligible": n_eligible,
        "n_oos_pass": n_pass,
        "p_r": p_r,
        "rel_mse_median": float(np.median(rels)) if rels else float("nan"),
        "status_counts": dict(Counter(m["status"] for m in eligible)) if eligible else {},
    }


def qualify_recording_operator(
    z: np.ndarray,
    *,
    columns: Sequence[str],
    rng: np.random.Generator,
    dt: float = DT_2_SEC,
) -> Dict[str, Any]:
    n_epoch = n_epoch_samples(EPOCH_SEC["primary"], dt)
    epoch_rows = []
    for i, epoch in enumerate(epoch_windows(z, n_epoch)):
        epoch_rng = np.random.default_rng(rng.integers(0, 2**31 - 1) + i)
        epoch_rows.append(qualify_operator_epoch(epoch, columns=columns, rng=epoch_rng))
    by_h = {
        str(h): {name: _model_recording_stats(epoch_rows, h, name) for name in MODELS_H}
        for h in HORIZONS
    }
    equiv = [bool(row.get("equivalence_ok")) for row in epoch_rows if row.get("status") == "SCORED"]
    rhos = [
        float(row["stability"]["rho"])
        for row in epoch_rows
        if row.get("status") == "SCORED" and np.isfinite(row.get("stability", {}).get("rho", np.nan))
    ]
    fro_h = [
        float(row["stability"]["fro_rel_halves"])
        for row in epoch_rows
        if row.get("status") == "SCORED"
        and np.isfinite(row.get("stability", {}).get("fro_rel_halves", np.nan))
    ]
    fro_b = [
        float(row["stability"]["fro_rel_bootstrap_median"])
        for row in epoch_rows
        if row.get("status") == "SCORED"
        and np.isfinite(row.get("stability", {}).get("fro_rel_bootstrap_median", np.nan))
    ]
    rho_h = [
        float(row["stability"]["rho_halves_median"])
        for row in epoch_rows
        if row.get("status") == "SCORED"
        and np.isfinite(row.get("stability", {}).get("rho_halves_median", np.nan))
    ]
    rho_b = [
        float(row["stability"]["rho_bootstrap_median"])
        for row in epoch_rows
        if row.get("status") == "SCORED"
        and np.isfinite(row.get("stability", {}).get("rho_bootstrap_median", np.nan))
    ]
    fh1 = by_h["1"]["fh"]
    return {
        "state_name": NAME_TYPE_C_STATE,
        "operator_name": OPERATOR_NAME,
        "panel": PRIMARY_PANEL,
        "horizons": by_h,
        "p_r": fh1["p_r"],
        "rel_mse_median": fh1["rel_mse_median"],
        "equivalence_rate": float(np.mean(equiv)) if equiv else float("nan"),
        "stability": {
            "rho_median": float(np.median(rhos)) if rhos else float("nan"),
            "rho_halves_median": float(np.median(rho_h)) if rho_h else float("nan"),
            "rho_bootstrap_median": float(np.median(rho_b)) if rho_b else float("nan"),
            "fro_rel_halves_median": float(np.median(fro_h)) if fro_h else float("nan"),
            "fro_rel_bootstrap_median": float(np.median(fro_b)) if fro_b else float("nan"),
            "alpha_computed": False,
            "logm_computed": False,
        },
        "epochs": epoch_rows,
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def operator_decision(
    coverages: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    def rel(h: int, model: str) -> float:
        return float(coverages[str(h)][model]["recording_rel_mse_median"])

    def beats(h: int) -> bool:
        fh_rel = rel(h, "fh")
        controls = [rel(h, "persist"), rel(h, "reverse"), rel(h, "shuffle")]
        return bool(np.isfinite(fh_rel) and all(np.isfinite(v) for v in controls) and fh_rel < min(controls))

    h1 = coverages["1"]["fh"]
    h1_pass = bool(h1["closure_pass"])
    h2_licensed = bool(rel(2, "fh") < OOS_THRESHOLD and beats(2))
    h4_beats = beats(4)
    h4_closed = bool(rel(4, "fh") < OOS_THRESHOLD)
    order_sensitive = bool(np.isfinite(rel(1, "reverse")) and np.isfinite(rel(1, "fh")) and rel(1, "reverse") > rel(1, "fh"))
    if h1_pass and h2_licensed and h4_beats and h4_closed:
        branch = "OPERATOR_LICENSED"
    elif h1_pass and h2_licensed and h4_beats:
        branch = "OPERATOR_PARTIAL"
    elif h1_pass:
        branch = "AR_ONLY"
    else:
        branch = "OPERATOR_FAIL"
    return {
        "branch": branch,
        "h1_pass": h1_pass,
        "h2_licensed": h2_licensed,
        "h4_beats_controls": h4_beats,
        "h4_closed": h4_closed,
        "order_sensitive": order_sensitive,
        "reverse_is_time_reversed_test": True,
        "delta_rel_h2_vs_h1": float(rel(2, "fh") - rel(1, "fh")),
        "delta_rel_h4_vs_h1": float(rel(4, "fh") - rel(1, "fh")),
        "alpha_computed": False,
        "logm_computed": False,
        "cpc_inspected": False,
        "not_mnps_abscissa": True,
        "not_instantaneous_type_c_abscissa": True,
    }


def run_c0_operator(*, rng_seed: int = 906) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    n_grid = int(round(TRUTH_DURATION_SEC / DT_2_SEC))
    columns = tuple(f"z{i}" for i in range(N_TARGET))
    j_true = np.diag(np.linspace(-0.08, -0.04, N_TARGET)).astype(np.float64)
    n_fine = int(round(TRUTH_DURATION_SEC / DT_FINE))
    first = downsample_to_grid(
        simulate_linear_sde(j_true, dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng),
        dt_fine=DT_FINE,
        grid_dt=DT_2_SEC,
    )
    a1 = np.diag(np.linspace(0.55, 0.40, N_TARGET))
    second = simulate_lagged_linear(
        a0=0.15 * np.eye(N_TARGET), a1=a1, n=n_grid, sigma=PROCESS_SIGMA, rng=rng
    )
    rows = {}
    for name, series in (("first_order", first), ("second_order", second)):
        rec = qualify_recording_operator(series, columns=columns, rng=rng)
        rec["regime"] = name
        rec.pop("epochs", None)
        rows[name] = rec
    return {"grid": "2/2", "n_params_m1": M1_N_PARAMS, "ridge_alpha": RIDGE_ALPHA, "rows": rows}


def _load_icare_recordings() -> Dict[str, Any]:
    frame = _load_type_c_features()
    if frame is None:
        return {
            "status": "BLOCKED",
            "reason": "Type-C 2/2 features missing; reuse the SL-005 extract. Do not slice mnps_3d.",
            "features_dir": str(TYPE_C_FEATURES_DIR),
            "alpha_computed": False,
            "cpc_inspected": False,
        }
    frame = drop_cpc_columns(frame)
    columns = PANELS[PRIMARY_PANEL]
    missing = [c for c in columns if c not in frame.columns]
    return {"status": "READY", "frame": frame, "columns": columns, "missing": missing}


def run_icare_operator(*, rng_seed: int = 906) -> Dict[str, Any]:
    loaded = _load_icare_recordings()
    if loaded["status"] != "READY":
        return loaded
    frame = loaded["frame"]
    columns = loaded["columns"]
    missing = loaded["missing"]
    rng = np.random.default_rng(rng_seed)
    recs = []
    for rec_name, part in _recording_groups(frame):
        if missing:
            recs.append(
                {
                    "recording": rec_name,
                    "status": "MISSING_FEATURES",
                    "p_r": float("nan"),
                    "rel_mse_median": float("nan"),
                }
            )
            continue
        part = part.copy()
        if "t_start" in part.columns:
            part = part.sort_values(["t_start", "epoch_id"] if "epoch_id" in part.columns else ["t_start"])
            starts = np.asarray(part["t_start"], dtype=float)
            hops = np.diff(starts[np.isfinite(starts)])
            if hops.size and not np.allclose(np.median(hops), TYPE_C_STEP_SEC, atol=0.05):
                recs.append(
                    {
                        "recording": rec_name,
                        "status": "GRID_INVALID",
                        "p_r": float("nan"),
                        "rel_mse_median": float("nan"),
                    }
                )
                continue
        z = part.loc[:, list(columns)].to_numpy(dtype=np.float32)
        rec_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        row = qualify_recording_operator(z, columns=columns, rng=rec_rng)
        row["recording"] = rec_name
        row.pop("epochs", None)
        recs.append(row)
    coverages: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        coverages[str(h)] = {}
        for name in MODELS_H:
            model_rows = []
            for rec in recs:
                stats = rec.get("horizons", {}).get(str(h), {}).get(name, {})
                model_rows.append(
                    {
                        "p_r": stats.get("p_r", rec.get("p_r", float("nan"))),
                        "rel_mse_median": stats.get("rel_mse_median", rec.get("rel_mse_median", float("nan"))),
                    }
                )
            coverages[str(h)][name] = coverage_summary(model_rows)
    decision = operator_decision(coverages)
    equiv_rates = [float(r["equivalence_rate"]) for r in recs if np.isfinite(r.get("equivalence_rate", np.nan))]
    rhos = [
        float(r["stability"]["rho_median"])
        for r in recs
        if np.isfinite(r.get("stability", {}).get("rho_median", np.nan))
    ]
    return {
        "status": "SCORED",
        "panel": PRIMARY_PANEL,
        "columns": list(columns),
        "operator_name": OPERATOR_NAME,
        "missing_columns": missing,
        "n_recordings": len(recs),
        "recordings": recs,
        "coverage": coverages,
        "decision": decision,
        "equivalence_rate_median": float(np.median(equiv_rates)) if equiv_rates else float("nan"),
        "rho_median": float(np.median(rhos)) if rhos else float("nan"),
        "features_dir": str(TYPE_C_FEATURES_DIR),
        "overlay": str(TYPE_C_OVERLAY),
        "alpha_computed": False,
        "logm_computed": False,
        "cpc_inspected": False,
    }


def sl007_report(*, rng_seed: int = 906) -> Dict[str, Any]:
    return {
        "gate": "TYPE_C_AUGMENTED_OPERATOR",
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "state_name": NAME_TYPE_C_STATE,
        "operator_name": OPERATOR_NAME,
        "not_mnps": True,
        "alpha_outside_acceptance": True,
        "logm_outside_acceptance": True,
        "cpc_blind": True,
        "threshold": OOS_THRESHOLD,
        "horizons": list(HORIZONS),
        "predictor": "y_{t+h}=F_H^h y_t + (I+...+F_H^{h-1})c , y=[z_t, z_{t-1}]",
        "c0": run_c0_operator(rng_seed=rng_seed),
        "icare": run_icare_operator(rng_seed=rng_seed),
        "next_if_licensed": "only then consider whether F_H=exp(J_H Δ) is well-posed; α_H would be of the 12D augmented state, still CPC-blind",
        "next_if_partial": "short-horizon operator only; do not take log(F_H)",
        "next_if_ar_only": "history helps one-step regression, not a dynamical operator; latent SSM next",
    }


def main() -> None:
    import json

    report = sl007_report()
    print("C0 operator horizons (F_H relMSE)")
    for name, row in report["c0"]["rows"].items():
        parts = []
        for h in HORIZONS:
            fh = row["horizons"][str(h)]["fh"]
            parts.append(f"h={h} p_r={fh['p_r']:.3f} rel={fh['rel_mse_median']:.3f}")
        print(f"  {name:13s} " + " | ".join(parts))
    icare = report["icare"]
    print(f"0284 status={icare['status']}")
    if icare.get("status") == "SCORED":
        dec = icare["decision"]
        for h in HORIZONS:
            print(f"  h={h}")
            for name, cov in icare["coverage"][str(h)].items():
                print(
                    f"    {name:8s} median_p_r={cov['median_p_r']:.3f} "
                    f"P>0.5={cov['p_recordings_p_r_gt_half']:.3f} "
                    f"rel={cov['recording_rel_mse_median']:.3f} closure={cov['closure_pass']}"
                )
        print(
            f"  branch={dec['branch']} order_sensitive={dec['order_sensitive']} "
            f"d_h2={dec['delta_rel_h2_vs_h1']:.3f} d_h4={dec['delta_rel_h4_vs_h1']:.3f} "
            f"rho_median={icare['rho_median']:.3f} equiv={icare['equivalence_rate_median']:.3f}"
        )
    out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl007_20260914"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sl007_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'sl007_report.json'}")


if __name__ == "__main__":
    main()

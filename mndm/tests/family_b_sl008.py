"""Family B SL-008: latent linear Gaussian SSM on Type-C observations (test-only).

z_t is treated as an observation of a hidden state s_t:

    s_{t+1} = A s_t + w_t
    z_t     = C s_t + v_t

The question is whether filtered ŝ_{t|t} and ẑ_{t+h}=C A^h ŝ_{t|t}
rescue held-out h=2 (and retain h=4 information) beyond M0, AR_ONLY/M1,
and a static latent denoising control.

Latent sizes {2,3,4,6} are frozen. No CPC, no log(A), no alpha_s.
Not wired into summarize.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from family_b_sl006 import M1_N_PARAMS, N_TARGET, history_split, simulate_lagged_linear
from family_b_sl007 import (
    HORIZONS,
    affine_to_companion,
    companion_predict_z,
    consecutive_triples_indexed,
    horizon_starts,
    original_triple_times,
    rel_mse_z,
    spectral_radius,
)

LATENT_DIMS = (2, 3, 4, 6)
N_EM = 3
JITTER = 1e-6
OBS_NOISE = 0.25
H1_AR_REF = 0.83
SSM_NAME = "latent_type_c_ssm"
MODELS_BASE = ("m0", "m1", "persist")


def _sym(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    return 0.5 * (mat + mat.T)


def _pd(mat: np.ndarray, jitter: float = JITTER) -> np.ndarray:
    mat = _sym(mat)
    n = int(mat.shape[0])
    return mat + float(jitter) * np.eye(n, dtype=np.float64)


def _score_pack(rel: float) -> Dict[str, Any]:
    oos_pass = bool(np.isfinite(rel) and float(rel) < OOS_THRESHOLD)
    return {
        "rel_mse_oos": float(rel) if np.isfinite(rel) else float("nan"),
        "oos_pass": oos_pass,
        "status": STATUS_TRANSITION_OOS_PASS if oos_pass else STATUS_TRANSITION_OOS_FAIL,
    }


def _block(z: np.ndarray, times: set) -> Optional[Tuple[np.ndarray, int]]:
    ordered = sorted(int(t) for t in times)
    if not ordered:
        return None
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        return None
    block = np.asarray(z[ordered[0] : ordered[-1] + 1], dtype=np.float64)
    return block, int(ordered[0])


def pca_var_init(z: np.ndarray, d_s: int) -> Optional[Dict[str, np.ndarray]]:
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z).all(axis=1)
    if int(finite.sum()) < max(d_s + 2, 12):
        return None
    zc = z.copy()
    mu = np.nanmean(zc[finite], axis=0)
    zc[finite] = zc[finite] - mu
    zc[~finite] = 0.0
    _, _, vt = np.linalg.svd(zc[finite], full_matrices=False)
    if vt.shape[0] < d_s:
        return None
    c = vt[:d_s].T.copy()
    scores = zc[finite] @ c
    if scores.shape[0] < d_s + 2:
        return None
    s_src, s_tgt = scores[:-1], scores[1:]
    design = np.hstack([s_src, np.ones((s_src.shape[0], 1))])
    ridge = RIDGE_ALPHA * np.eye(design.shape[1], dtype=np.float64)
    ridge[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + ridge, design.T @ s_tgt)
    a = coef[:-1].T
    resid_s = s_tgt - s_src @ a.T
    q = _pd(np.cov(resid_s.T) if resid_s.shape[0] > 1 else np.eye(d_s))
    resid_z = zc[finite] - scores @ c.T
    r = _pd(np.cov(resid_z.T) if resid_z.shape[0] > 1 else np.eye(z.shape[1]))
    return {"A": a, "C": c, "Q": q, "R": r, "mu": mu.astype(np.float64), "d_s": np.array([d_s])}


def kalman_filter(
    z: np.ndarray,
    *,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Q = _pd(Q)
    R = _pd(R)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    t_len = int(z.shape[0])
    d_s = int(A.shape[0])
    s_f = np.zeros((t_len, d_s), dtype=np.float64)
    p_f = np.zeros((t_len, d_s, d_s), dtype=np.float64)
    s_p = np.zeros((t_len, d_s), dtype=np.float64)
    p_p = np.zeros((t_len, d_s, d_s), dtype=np.float64)
    s = np.zeros(d_s, dtype=np.float64)
    p = 10.0 * np.eye(d_s, dtype=np.float64)
    eye = np.eye(d_s, dtype=np.float64)
    for t in range(t_len):
        if t == 0:
            s_pred, p_pred = s, p
        else:
            s_pred = A @ s
            p_pred = _pd(A @ p @ A.T + Q)
        s_p[t], p_p[t] = s_pred, p_pred
        innov = z[t] - mu - C @ s_pred
        if np.isfinite(innov).all():
            s_mat = _pd(C @ p_pred @ C.T + R)
            gain = p_pred @ C.T @ np.linalg.pinv(s_mat)
            s = s_pred + gain @ innov
            p = _pd((eye - gain @ C) @ p_pred)
        else:
            s, p = s_pred, p_pred
        s_f[t], p_f[t] = s, p
    return s_f, p_f, s_p, p_p


def kalman_smooth(
    z: np.ndarray,
    params: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, c, q, r, mu = params["A"], params["C"], params["Q"], params["R"], params["mu"]
    s_f, p_f, s_p, p_p = kalman_filter(z, A=a, C=c, Q=q, R=r, mu=mu)
    t_len, d_s = s_f.shape
    s_s = s_f.copy()
    p_s = p_f.copy()
    p_lag = np.zeros((t_len, d_s, d_s), dtype=np.float64)
    for t in range(t_len - 2, -1, -1):
        j_t = p_f[t] @ a.T @ np.linalg.pinv(p_p[t + 1])
        s_s[t] = s_f[t] + j_t @ (s_s[t + 1] - s_p[t + 1])
        p_s[t] = _pd(p_f[t] + j_t @ (p_s[t + 1] - p_p[t + 1]) @ j_t.T)
        p_lag[t + 1] = p_s[t + 1] @ j_t.T
    return s_s, p_s, p_lag


def em_update(z: np.ndarray, params: Mapping[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    z = np.asarray(z, dtype=np.float64)
    s_s, p_s, p_lag = kalman_smooth(z, params)
    t_len, d_s = s_s.shape
    d_z = int(z.shape[1])
    mu = np.asarray(params["mu"], dtype=np.float64).reshape(-1)
    finite = np.isfinite(z).all(axis=1)
    if int(finite.sum()) < d_s + 2 or t_len < 3:
        return None
    e_ss0 = np.zeros((d_s, d_s), dtype=np.float64)
    e_ss1 = np.zeros((d_s, d_s), dtype=np.float64)
    e_cross = np.zeros((d_s, d_s), dtype=np.float64)
    n_pair = 0
    for t in range(t_len - 1):
        e_ss0 += p_s[t] + np.outer(s_s[t], s_s[t])
        e_ss1 += p_s[t + 1] + np.outer(s_s[t + 1], s_s[t + 1])
        e_cross += p_lag[t + 1] + np.outer(s_s[t + 1], s_s[t])
        n_pair += 1
    a = e_cross @ np.linalg.pinv(e_ss0 + RIDGE_ALPHA * np.eye(d_s))
    q = _pd((e_ss1 - a @ e_cross.T - e_cross @ a.T + a @ e_ss0 @ a.T) / max(n_pair, 1))
    e_sz = np.zeros((d_z, d_s), dtype=np.float64)
    e_ss_obs = np.zeros((d_s, d_s), dtype=np.float64)
    e_zz = np.zeros((d_z, d_z), dtype=np.float64)
    n_obs = 0
    for t in range(t_len):
        if not bool(finite[t]):
            continue
        zt = z[t] - mu
        e_sz += np.outer(zt, s_s[t])
        e_ss_obs += p_s[t] + np.outer(s_s[t], s_s[t])
        e_zz += np.outer(zt, zt)
        n_obs += 1
    if n_obs < d_s + 1:
        return None
    c = e_sz @ np.linalg.pinv(e_ss_obs)
    r = _pd((e_zz - c @ e_sz.T - e_sz @ c.T + c @ e_ss_obs @ c.T) / n_obs)
    if not np.isfinite(a).all() or not np.isfinite(c).all():
        return None
    return {"A": a, "C": c, "Q": q, "R": r, "mu": mu, "d_s": np.array([d_s])}


def fit_lgssm(z: np.ndarray, d_s: int, *, n_em: int = N_EM) -> Optional[Dict[str, np.ndarray]]:
    params = pca_var_init(z, d_s)
    if params is None:
        return None
    for _ in range(int(n_em)):
        updated = em_update(z, params)
        if updated is None:
            break
        params = updated
    return params


def power_A(A: np.ndarray, h: int) -> np.ndarray:
    a = np.asarray(A, dtype=np.float64)
    out = np.eye(a.shape[0], dtype=np.float64)
    for _ in range(int(h)):
        out = a @ out
        if not np.isfinite(out).all() or np.max(np.abs(out)) > 1e8:
            out[:] = np.nan
            break
    return out


def ssm_predict_z(
    params: Mapping[str, np.ndarray],
    s_filt: np.ndarray,
    local_t: np.ndarray,
    horizon: int,
) -> np.ndarray:
    a_h = power_A(params["A"], horizon)
    c = np.asarray(params["C"], dtype=np.float64)
    mu = np.asarray(params["mu"], dtype=np.float64).reshape(-1)
    s = np.asarray(s_filt, dtype=np.float64)[np.asarray(local_t, dtype=int)]
    if not np.isfinite(a_h).all():
        return np.full((s.shape[0], c.shape[0]), np.nan, dtype=np.float32)
    s_h = s @ a_h.T
    pred = s_h @ c.T + mu
    bad = ~np.isfinite(pred).all(axis=1) | (np.max(np.abs(pred), axis=1) > 1e8)
    if np.any(bad):
        pred[bad] = np.nan
    return pred.astype(np.float32)


def static_predict_z(params: Mapping[str, np.ndarray], z_now: np.ndarray) -> np.ndarray:
    """Denoised current observation; no A^h. Static latent control."""
    c = np.asarray(params["C"], dtype=np.float64)
    mu = np.asarray(params["mu"], dtype=np.float64).reshape(-1)
    zc = np.asarray(z_now, dtype=np.float64) - mu
    gram = c.T @ c + JITTER * np.eye(c.shape[1], dtype=np.float64)
    scores = np.linalg.solve(gram, c.T @ zc.T).T
    return (scores @ c.T + mu).astype(np.float32)


def iterate_m0(
    mapped: np.ndarray,
    intercept: np.ndarray,
    x_mean: np.ndarray,
    now: np.ndarray,
    horizon: int,
) -> np.ndarray:
    z = np.asarray(now, dtype=np.float64)
    mapped = np.asarray(mapped, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64).reshape(-1)
    x_mean = np.asarray(x_mean, dtype=np.float64).reshape(-1)
    for _ in range(int(horizon)):
        z = (z - x_mean) @ mapped.T + intercept
        bad = ~np.isfinite(z).all(axis=1) | (np.max(np.abs(z), axis=1) > 1e8)
        if np.any(bad):
            z[bad] = np.nan
    return z.astype(np.float32)


def _empty_models(d_s_list: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    names = list(MODELS_BASE) + [f"ssm_{d}" for d in d_s_list] + [f"static_{d}" for d in d_s_list]
    return {
        name: {
            "rel_mse_oos": float("nan"),
            "oos_pass": False,
            "status": STATUS_TRANSITION_INSUFFICIENT,
            "n_starts": 0,
        }
        for name in names
    }


def qualify_ssm_epoch(
    z: np.ndarray,
    *,
    columns: Sequence[str],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    del rng
    assert_not_mnps_state(columns)
    z = apply_type_c_transforms(z, columns)
    lag, now, nxt, t_now = consecutive_triples_indexed(z)
    n_triples = int(now.shape[0])
    n_dim = int(now.shape[1]) if now.ndim == 2 else 0
    base = {
        "n_triples": n_triples,
        "n_dim": n_dim,
        "state_name": NAME_TYPE_C_STATE,
        "ssm_name": SSM_NAME,
        "latent_dims": list(LATENT_DIMS),
        "alpha_computed": False,
        "logm_computed": False,
        "cpc_inspected": False,
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
            "horizons": {str(h): _empty_models(LATENT_DIMS) for h in HORIZONS},
            "rho": {str(d): float("nan") for d in LATENT_DIMS},
        }
    train_times = original_triple_times(t_now[train_sl])
    test_times = original_triple_times(t_now[test_sl])
    train_block = _block(z, train_times)
    test_block = _block(z, test_times)
    y = np.hstack([now, lag])
    fitted_m1 = fit_global_affine(y[train_sl], nxt[train_sl], ridge_alpha=RIDGE_ALPHA)
    fitted_m0 = fit_global_affine(now[train_sl], nxt[train_sl], ridge_alpha=RIDGE_ALPHA)
    params_by_d: Dict[int, Optional[Dict[str, np.ndarray]]] = {}
    filt_by_d: Dict[int, Optional[np.ndarray]] = {}
    rho_by_d: Dict[str, float] = {}
    t0_test = None
    if train_block is not None:
        z_tr, _ = train_block
        for d_s in LATENT_DIMS:
            params_by_d[d_s] = fit_lgssm(z_tr, d_s)
            rho_by_d[str(d_s)] = (
                spectral_radius(params_by_d[d_s]["A"]) if params_by_d[d_s] is not None else float("nan")
            )
    else:
        for d_s in LATENT_DIMS:
            params_by_d[d_s] = None
            rho_by_d[str(d_s)] = float("nan")
    if test_block is not None:
        z_te, t0_test = test_block
        for d_s in LATENT_DIMS:
            par = params_by_d[d_s]
            if par is None:
                filt_by_d[d_s] = None
                continue
            s_f, _, _, _ = kalman_filter(z_te, A=par["A"], C=par["C"], Q=par["Q"], R=par["R"], mu=par["mu"])
            filt_by_d[d_s] = s_f
    else:
        for d_s in LATENT_DIMS:
            filt_by_d[d_s] = None
    f_h = c_aug = None
    if fitted_m1 is not None:
        f_h, c_aug = affine_to_companion(*fitted_m1)
    horizons: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        if h == 1:
            now_h, lag_h, tgt_h, t_h = now[test_sl], lag[test_sl], nxt[test_sl], t_now[test_sl]
        else:
            lag_h, now_h, tgt_h, t_h = horizon_starts(z, test_times, h)
        n_starts = int(now_h.shape[0])
        models = _empty_models(LATENT_DIMS)
        if n_starts >= 2:
            models["persist"] = {**_score_pack(rel_mse_z(now_h, tgt_h)), "n_starts": n_starts}
            if fitted_m0 is not None:
                pred_m0 = iterate_m0(*fitted_m0, now_h, h)
                models["m0"] = {**_score_pack(rel_mse_z(pred_m0, tgt_h)), "n_starts": n_starts}
            if f_h is not None:
                pred_m1 = companion_predict_z(f_h, c_aug, now_h, lag_h, h)
                models["m1"] = {**_score_pack(rel_mse_z(pred_m1, tgt_h)), "n_starts": n_starts}
            if t0_test is not None:
                local = np.asarray(t_h, dtype=int) - int(t0_test)
                in_block = (local >= 0) & (local < (test_block[0].shape[0] if test_block else 0))
                for d_s in LATENT_DIMS:
                    par = params_by_d[d_s]
                    s_f = filt_by_d[d_s]
                    if par is None or s_f is None or int(in_block.sum()) < 2:
                        continue
                    loc = local[in_block]
                    tgt = tgt_h[in_block]
                    now_ok = now_h[in_block]
                    pred_ssm = ssm_predict_z(par, s_f, loc, h)
                    pred_static = static_predict_z(par, now_ok)
                    models[f"ssm_{d_s}"] = {**_score_pack(rel_mse_z(pred_ssm, tgt)), "n_starts": int(in_block.sum())}
                    models[f"static_{d_s}"] = {**_score_pack(rel_mse_z(pred_static, tgt)), "n_starts": int(in_block.sum())}
        horizons[str(h)] = models
    return {
        **base,
        "status": "SCORED",
        "embargo_pairs": int(gap),
        "n_train": int(nxt[train_sl].shape[0]),
        "n_test": int(nxt[test_sl].shape[0]),
        "horizons": horizons,
        "rho": rho_by_d,
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


def qualify_recording_ssm(
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
        epoch_rows.append(qualify_ssm_epoch(epoch, columns=columns, rng=epoch_rng))
    names = list(MODELS_BASE) + [f"ssm_{d}" for d in LATENT_DIMS] + [f"static_{d}" for d in LATENT_DIMS]
    by_h = {
        str(h): {name: _model_recording_stats(epoch_rows, h, name) for name in names}
        for h in HORIZONS
    }
    rhos = {str(d): [] for d in LATENT_DIMS}
    for row in epoch_rows:
        if row.get("status") != "SCORED":
            continue
        for d in LATENT_DIMS:
            val = row.get("rho", {}).get(str(d), float("nan"))
            if np.isfinite(val):
                rhos[str(d)].append(float(val))
    ssm3 = by_h["1"].get("ssm_3", {})
    return {
        "state_name": NAME_TYPE_C_STATE,
        "ssm_name": SSM_NAME,
        "panel": PRIMARY_PANEL,
        "horizons": by_h,
        "p_r": ssm3.get("p_r", float("nan")),
        "rel_mse_median": ssm3.get("rel_mse_median", float("nan")),
        "rho_median": {d: float(np.median(v)) if v else float("nan") for d, v in rhos.items()},
        "epochs": epoch_rows,
        "alpha_computed": False,
        "logm_computed": False,
        "cpc_inspected": False,
    }


def ssm_decision(coverages: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    def rel(h: int, model: str) -> float:
        return float(coverages[str(h)][model]["recording_rel_mse_median"])

    def beats(h: int, model: str, controls: Sequence[str]) -> bool:
        fh_rel = rel(h, model)
        vals = [rel(h, name) for name in controls]
        return bool(np.isfinite(fh_rel) and all(np.isfinite(v) for v in vals) and fh_rel < min(vals))

    licensed: List[int] = []
    per_d = {}
    for d_s in LATENT_DIMS:
        name = f"ssm_{d_s}"
        static = f"static_{d_s}"
        h1 = rel(1, name)
        h2 = rel(2, name)
        h1_ok = bool(np.isfinite(h1) and h1 < OOS_THRESHOLD and h1 <= H1_AR_REF + 0.05)
        h2_ok = bool(np.isfinite(h2) and h2 < OOS_THRESHOLD)
        h2_beats = beats(2, name, ("m0", "m1", "persist", static))
        h4_beats = beats(4, name, ("m0", "m1", "persist", static))
        h2_broad = bool(coverages["2"][name]["closure_pass"])
        row = {
            "h1_rel": h1,
            "h2_rel": h2,
            "h4_rel": rel(4, name),
            "h1_ok": h1_ok,
            "h2_ok": h2_ok,
            "h2_beats_controls": h2_beats,
            "h4_beats_controls": h4_beats,
            "h2_closure_pass": h2_broad,
        }
        per_d[str(d_s)] = row
        if h1_ok and h2_ok and h2_beats:
            licensed.append(int(d_s))
    selected = min(licensed) if licensed else None
    if selected is None:
        h2_any = any(np.isfinite(rel(2, f"ssm_{d}")) and rel(2, f"ssm_{d}") < rel(2, "m1") for d in LATENT_DIMS)
        branch = "REPACKAGED_AR" if h2_any or any(per_d[str(d)]["h1_ok"] for d in LATENT_DIMS) else "SSM_FAIL"
    else:
        info = per_d[str(selected)]
        if info["h2_closure_pass"] and info["h4_beats_controls"]:
            branch = "SSM_LICENSED" if info["h4_rel"] < OOS_THRESHOLD else "SSM_PARTIAL"
        elif info["h2_closure_pass"]:
            branch = "SSM_H2"
        else:
            branch = "SSM_H2_NARROW"
    return {
        "branch": branch,
        "selected_d_s": selected,
        "latent_dims": list(LATENT_DIMS),
        "by_d_s": per_d,
        "selection_rule": "smallest d_s with h1<0.9, h1<=0.88, h2<0.9, and beating m0/m1/persist/static at h=2",
        "alpha_computed": False,
        "logm_computed": False,
        "cpc_inspected": False,
        "cpc_not_used_for_d_s": True,
    }


def simulate_partial_observations(*, rng: np.random.Generator, d_s: int = 3, d_z: int = 6) -> np.ndarray:
    """Latent first-order AR observed through a tall linear map plus noise."""
    n_grid = int(round(TRUTH_DURATION_SEC / DT_2_SEC))
    a0 = np.diag(np.linspace(0.92, 0.80, d_s)).astype(np.float64)
    latent = simulate_lagged_linear(a0=a0, a1=None, n=n_grid, sigma=0.30, rng=rng)
    raw = rng.normal(size=(d_z, d_s))
    q_mat, _ = np.linalg.qr(raw)
    c_true = q_mat[:, :d_s]
    noise = float(OBS_NOISE) * rng.normal(size=(latent.shape[0], d_z))
    return (latent @ c_true.T + noise).astype(np.float32)


def run_c0_ssm(*, rng_seed: int = 908) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    columns = tuple(f"z{i}" for i in range(N_TARGET))
    hidden = simulate_partial_observations(rng=rng, d_s=3, d_z=N_TARGET)
    j_full = np.diag(np.linspace(-0.08, -0.04, N_TARGET)).astype(np.float64)
    n_fine = int(round(TRUTH_DURATION_SEC / DT_FINE))
    observed = downsample_to_grid(
        simulate_linear_sde(j_full, dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng),
        dt_fine=DT_FINE,
        grid_dt=DT_2_SEC,
    )
    rows = {}
    for name, series in (("hidden_3d", hidden), ("fully_observed_6d", observed)):
        rec = qualify_recording_ssm(series, columns=columns, rng=rng)
        rec["regime"] = name
        rec.pop("epochs", None)
        rows[name] = rec
    return {"grid": "2/2", "n_params_m1": M1_N_PARAMS, "latent_dims": list(LATENT_DIMS), "rows": rows}


def run_icare_ssm(*, rng_seed: int = 908) -> Dict[str, Any]:
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
    rng = np.random.default_rng(rng_seed)
    recs = []
    for rec_name, part in _recording_groups(frame):
        if missing:
            recs.append({"recording": rec_name, "status": "MISSING_FEATURES", "p_r": float("nan"), "rel_mse_median": float("nan")})
            continue
        part = part.copy()
        if "t_start" in part.columns:
            part = part.sort_values(["t_start", "epoch_id"] if "epoch_id" in part.columns else ["t_start"])
            starts = np.asarray(part["t_start"], dtype=float)
            hops = np.diff(starts[np.isfinite(starts)])
            if hops.size and not np.allclose(np.median(hops), TYPE_C_STEP_SEC, atol=0.05):
                recs.append({"recording": rec_name, "status": "GRID_INVALID", "p_r": float("nan"), "rel_mse_median": float("nan")})
                continue
        z = part.loc[:, list(columns)].to_numpy(dtype=np.float32)
        rec_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        row = qualify_recording_ssm(z, columns=columns, rng=rec_rng)
        row["recording"] = rec_name
        row.pop("epochs", None)
        recs.append(row)
    names = list(MODELS_BASE) + [f"ssm_{d}" for d in LATENT_DIMS] + [f"static_{d}" for d in LATENT_DIMS]
    coverages: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        coverages[str(h)] = {}
        for name in names:
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
    decision = ssm_decision(coverages)
    return {
        "status": "SCORED",
        "panel": PRIMARY_PANEL,
        "columns": list(columns),
        "ssm_name": SSM_NAME,
        "missing_columns": missing,
        "n_recordings": len(recs),
        "recordings": recs,
        "coverage": coverages,
        "decision": decision,
        "features_dir": str(TYPE_C_FEATURES_DIR),
        "overlay": str(TYPE_C_OVERLAY),
        "alpha_computed": False,
        "logm_computed": False,
        "cpc_inspected": False,
    }


def sl008_report(*, rng_seed: int = 908) -> Dict[str, Any]:
    return {
        "gate": "TYPE_C_LATENT_SSM",
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "state_name": NAME_TYPE_C_STATE,
        "ssm_name": SSM_NAME,
        "not_mnps": True,
        "alpha_outside_acceptance": True,
        "logm_outside_acceptance": True,
        "cpc_blind": True,
        "threshold": OOS_THRESHOLD,
        "horizons": list(HORIZONS),
        "latent_dims": list(LATENT_DIMS),
        "predictor": "zhat_{t+h}=C A^h s_{t|t}  (Kalman filtered latent state)",
        "c0": run_c0_ssm(rng_seed=rng_seed),
        "icare": run_icare_ssm(rng_seed=rng_seed),
        "next_if_licensed": "only then consider rho(A) as a discrete operator, and only later whether A=exp(J_s Δ) is well-posed; still CPC-blind",
        "next_if_repackaged": "SSM did not rescue h=2; do not take log(A); nonlinear/switching SSM is a new family, not a CPC search",
    }


def main() -> None:
    import json

    report = sl008_report()
    print("C0 SSM (ssm_3 vs m1)")
    for name, row in report["c0"]["rows"].items():
        parts = []
        for h in HORIZONS:
            ssm = row["horizons"][str(h)]["ssm_3"]
            m1 = row["horizons"][str(h)]["m1"]
            parts.append(f"h={h} ssm3={ssm['rel_mse_median']:.3f} m1={m1['rel_mse_median']:.3f}")
        print(f"  {name:20s} " + " | ".join(parts))
    icare = report["icare"]
    print(f"0284 status={icare['status']}")
    if icare.get("status") == "SCORED":
        dec = icare["decision"]
        for h in HORIZONS:
            print(f"  h={h}")
            for key in ("m0", "m1", "persist", "ssm_2", "ssm_3", "ssm_4", "ssm_6", "static_3"):
                cov = icare["coverage"][str(h)][key]
                print(
                    f"    {key:10s} median_p_r={cov['median_p_r']:.3f} "
                    f"P>0.5={cov['p_recordings_p_r_gt_half']:.3f} "
                    f"rel={cov['recording_rel_mse_median']:.3f} closure={cov['closure_pass']}"
                )
        print(f"  branch={dec['branch']} selected_d_s={dec['selected_d_s']}")
    out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl008_20260914"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sl008_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'sl008_report.json'}")


if __name__ == "__main__":
    main()

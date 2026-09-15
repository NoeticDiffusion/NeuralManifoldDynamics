"""HC-OER-001: history-conditioned one-step expansion rate (test-only).

Observed finite-step local separation on frozen Type-C C-primary:

    y_t = [z_t, z_{t-1}]
    gamma_HC_1(t) = (1/dt) median_j log((||z_{t+1}-z_{j+1}||+eps) / (||z_t-z_j||+eps))

Not spectral abscissa. Not F_H eigenvalues. Not CPC. Not summarize.
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
    PROCESS_SIGMA,
    REGIMES,
    _json_ready,
    downsample_to_grid,
    simulate_linear_sde,
    spectral_abscissa,
)
from family_b_sl003 import EPOCH_SEC, n_epoch_samples, epoch_windows
from family_b_sl004 import DT_2_SEC, TYPE_C_STEP_SEC
from family_b_sl005 import (
    NAME_TYPE_C_STATE,
    PANELS,
    PRIMARY_PANEL,
    TRUTH_DURATION_SEC,
    TYPE_C_FEATURES_DIR,
    TYPE_C_OVERLAY,
    apply_type_c_transforms,
    assert_not_mnps_state,
    drop_cpc_columns,
    _load_type_c_features,
    _recording_groups,
)
from family_b_sl006 import consecutive_triples

GATE = "HC_OER_001"
DT_SEC = DT_2_SEC
K_NEIGHBORS = 20
K_MIN = 8
MIN_INDEX_GAP = 3
EPS = 1e-6
D_MIN_STEP_FRAC = 0.5
D_MAX_STEP_FRAC = 5.0
SHELL_Q_LO = 40.0
SHELL_Q_HI = 80.0
WHITE_ABS_TOL = 0.05
MAD_SCALE = 1.4826
ISOTROPIC_TOL = 0.05
COVERAGE_FRAC = 0.5
N_RECORDINGS_COVERAGE = 8
RNG_SEED = 1001
MODES = ("hc", "z_only", "temporal", "shuffle_next", "reverse")
STATUS_INSUFFICIENT = "HC_OER_INSUFFICIENT"
STATUS_SCORED = "SCORED"


def _robust_center_scale(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z).all(axis=1)
    pool = z[finite] if np.any(finite) else z
    center = np.median(pool, axis=0)
    mad = np.median(np.abs(pool - center), axis=0)
    scale = np.where(mad > 1e-12, MAD_SCALE * mad, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def _apply_robust_z(z: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(z, dtype=np.float64) - center) / scale


def _pairwise_euclid(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    grams = a @ a.T
    nrm = np.sum(a * a, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * grams
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, dtype=np.float64)


def _step_scale(now: np.ndarray, nxt: np.ndarray) -> float:
    step = np.linalg.norm(np.asarray(nxt, dtype=np.float64) - np.asarray(now, dtype=np.float64), axis=1)
    step = step[np.isfinite(step)]
    if step.size == 0:
        return 1.0
    med = float(np.median(step))
    return med if med > 1e-12 else 1.0


def _finite_scale_mask(now: np.ndarray, i: int, step_scale: float) -> np.ndarray:
    d_now = np.linalg.norm(now - now[int(i)], axis=1)
    lo = D_MIN_STEP_FRAC * float(step_scale)
    hi = D_MAX_STEP_FRAC * float(step_scale)
    return (d_now >= lo) & (d_now <= hi)


def _eligible_mask(n: int, index: int, gap: int = MIN_INDEX_GAP) -> np.ndarray:
    idx = np.arange(n)
    return np.abs(idx - int(index)) >= int(gap)


def _knn_indices(distances: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    d = np.where(eligible, distances, np.inf)
    finite = np.isfinite(d)
    n_ok = int(np.sum(finite))
    if n_ok <= 0:
        return np.empty(0, dtype=int)
    take = min(int(k), n_ok)
    order = np.argpartition(d, take - 1)[:take]
    return order[np.argsort(d[order])]


def _log_ratio(d_next: np.ndarray, d_now: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.log((np.asarray(d_next, dtype=np.float64) + float(eps)) / (np.asarray(d_now, dtype=np.float64) + float(eps)))


def _summarize_gammas(gammas: np.ndarray) -> Dict[str, Any]:
    g = np.asarray(gammas, dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {
            "n_scored": 0,
            "gamma_median": float("nan"),
            "f_plus": float("nan"),
            "gamma_mean": float("nan"),
        }
    return {
        "n_scored": int(g.size),
        "gamma_median": float(np.median(g)),
        "f_plus": float(np.mean(g > 0.0)),
        "gamma_mean": float(np.mean(g)),
    }


def _shell_indices(distances: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool) & np.isfinite(d)
    vals = d[mask]
    if int(vals.size) < K_MIN:
        return np.empty(0, dtype=int)
    lo, hi = np.percentile(vals, [SHELL_Q_LO, SHELL_Q_HI])
    shell = mask & (d >= lo) & (d <= hi)
    return np.where(shell)[0]


def _neighbor_sets(
    *,
    y: np.ndarray,
    now: np.ndarray,
    nxt: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, list]:
    n = int(y.shape[0])
    d_y = _pairwise_euclid(y)
    d_z = _pairwise_euclid(now)
    time_d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(np.float64)
    step_scale = _step_scale(now, nxt)
    hc: list = []
    z_only: list = []
    temporal: list = []
    shuffle: list = []
    for i in range(n):
        eligible = _eligible_mask(n, i) & _finite_scale_mask(now, i, step_scale)
        hc_idx = _shell_indices(d_y[i], eligible)
        z_idx = _shell_indices(d_z[i], eligible)
        t_idx = _knn_indices(time_d[i], eligible, K_NEIGHBORS)
        hc.append(hc_idx)
        z_only.append(z_idx)
        temporal.append(t_idx)
        if hc_idx.size:
            pool = np.where(_eligible_mask(n, i))[0]
            take = min(int(hc_idx.size), int(pool.size))
            if take <= 0:
                shuffle.append(np.empty(0, dtype=int))
            else:
                shuffle.append(np.asarray(rng.choice(pool, size=take, replace=False), dtype=int))
        else:
            shuffle.append(np.empty(0, dtype=int))
    return {"hc": hc, "z_only": z_only, "temporal": temporal, "shuffle_next": shuffle, "step_scale": step_scale}


def _gamma_from_neighbors(
    now: np.ndarray,
    nxt: np.ndarray,
    neighbor_sets: Sequence[np.ndarray],
    *,
    dt: float,
    next_sets: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    n = int(now.shape[0])
    gammas = np.full(n, np.nan, dtype=np.float64)
    for i, neigh in enumerate(neighbor_sets):
        j = np.asarray(neigh, dtype=int)
        if next_sets is not None:
            j_next = np.asarray(next_sets[i], dtype=int)
            take = min(int(j.size), int(j_next.size))
            if take < K_MIN:
                continue
            j_now = j[:take]
            j_n = j_next[:take]
        else:
            if int(j.size) < K_MIN:
                continue
            j_now = j
            j_n = j
        d_now = np.linalg.norm(now[i] - now[j_now], axis=1)
        d_next = np.linalg.norm(nxt[i] - nxt[j_n], axis=1)
        ratios = _log_ratio(d_next, d_now)
        finite = np.isfinite(ratios)
        if int(np.sum(finite)) < K_MIN:
            continue
        gammas[i] = float(np.median(ratios[finite]) / float(dt))
    return gammas


def score_epoch_hc_oer(
    z: np.ndarray,
    *,
    columns: Sequence[str],
    rng: np.random.Generator,
    dt: float = DT_SEC,
) -> Dict[str, Any]:
    assert_not_mnps_state(columns)
    z = apply_type_c_transforms(np.asarray(z, dtype=np.float32), columns)
    lag, now_raw, nxt_raw = consecutive_triples(z)
    n_triples = int(now_raw.shape[0])
    base = {"n_triples": n_triples, "n_dim": int(now_raw.shape[1]) if n_triples else 0, "dt_sec": float(dt)}
    if n_triples < K_NEIGHBORS + MIN_INDEX_GAP + 1:
        empty = _summarize_gammas(np.array([]))
        return {
            **base,
            "status": STATUS_INSUFFICIENT,
            "coverage": 0.0,
            "modes": {name: empty for name in MODES},
        }
    stacked = np.vstack([now_raw, nxt_raw, lag])
    center, scale = _robust_center_scale(stacked)
    now = _apply_robust_z(now_raw, center, scale)
    nxt = _apply_robust_z(nxt_raw, center, scale)
    lag_s = _apply_robust_z(lag, center, scale)
    y = np.hstack([now, lag_s])
    neighbors = _neighbor_sets(y=y, now=now, nxt=nxt, rng=rng)
    modes: Dict[str, Any] = {}
    hc_g = _gamma_from_neighbors(now, nxt, neighbors["hc"], dt=dt)
    modes["hc"] = _summarize_gammas(hc_g)
    modes["z_only"] = _summarize_gammas(_gamma_from_neighbors(now, nxt, neighbors["z_only"], dt=dt))
    modes["temporal"] = _summarize_gammas(_gamma_from_neighbors(now, nxt, neighbors["temporal"], dt=dt))
    modes["shuffle_next"] = _summarize_gammas(
        _gamma_from_neighbors(now, nxt, neighbors["hc"], dt=dt, next_sets=neighbors["shuffle_next"])
    )
    lag_r, now_r_raw, nxt_r_raw = consecutive_triples(z[::-1])
    center_r, scale_r = _robust_center_scale(np.vstack([now_r_raw, nxt_r_raw, lag_r]))
    now_r = _apply_robust_z(now_r_raw, center_r, scale_r)
    nxt_r = _apply_robust_z(nxt_r_raw, center_r, scale_r)
    lag_r_s = _apply_robust_z(lag_r, center_r, scale_r)
    y_r = np.hstack([now_r, lag_r_s])
    rev_neighbors = _neighbor_sets(y=y_r, now=now_r, nxt=nxt_r, rng=rng)
    modes["reverse"] = _summarize_gammas(_gamma_from_neighbors(now_r, nxt_r, rev_neighbors["hc"], dt=dt))
    n_scored = int(modes["hc"]["n_scored"])
    coverage = float(n_scored / n_triples) if n_triples else 0.0
    status = STATUS_SCORED if n_scored >= K_MIN else STATUS_INSUFFICIENT
    return {
        **base,
        "status": status,
        "coverage": coverage,
        "modes": modes,
        "step_scale": float(neighbors["step_scale"]),
    }


def qualify_recording_hc_oer(
    z: np.ndarray,
    *,
    columns: Sequence[str],
    rng: np.random.Generator,
    dt: float = DT_SEC,
) -> Dict[str, Any]:
    n_epoch = n_epoch_samples(EPOCH_SEC["primary"], dt)
    epoch_rows = []
    for i, epoch in enumerate(epoch_windows(z, n_epoch)):
        epoch_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i)
        epoch_rows.append(score_epoch_hc_oer(epoch, columns=columns, rng=epoch_rng, dt=dt))
    by_mode: Dict[str, Any] = {}
    for name in MODES:
        scored = [row["modes"][name] for row in epoch_rows if row["status"] == STATUS_SCORED]
        meds = [float(s["gamma_median"]) for s in scored if np.isfinite(s["gamma_median"])]
        fplus = [float(s["f_plus"]) for s in scored if np.isfinite(s["f_plus"])]
        coverages = [float(row["coverage"]) for row in epoch_rows]
        by_mode[name] = {
            "n_epochs": len(epoch_rows),
            "n_scored_epochs": len(scored),
            "gamma_median": float(np.median(meds)) if meds else float("nan"),
            "f_plus_median": float(np.median(fplus)) if fplus else float("nan"),
            "coverage_median": float(np.median(coverages)) if coverages else float("nan"),
            "status_counts": dict(Counter(row["status"] for row in epoch_rows)),
        }
    return {
        "state_name": NAME_TYPE_C_STATE,
        "panel": PRIMARY_PANEL,
        "modes": by_mode,
        "coverage_median": by_mode["hc"]["coverage_median"],
        "gamma_median": by_mode["hc"]["gamma_median"],
        "f_plus_median": by_mode["hc"]["f_plus_median"],
        "n_epochs": len(epoch_rows),
    }


def isotropic_series(rho: float, *, n: int, dim: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    z = np.zeros((n, dim), dtype=np.float64)
    z[0] = rng.normal(size=dim)
    for t in range(n - 1):
        z[t + 1] = float(rho) * z[t] + float(sigma) * rng.normal(size=dim)
    return z.astype(np.float32)


def expected_isotropic_gamma(rho: float, dt: float = DT_SEC) -> float:
    return float(np.log(abs(float(rho))) / float(dt))


def run_c0_hc_oer(*, rng_seed: int = RNG_SEED) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    columns = tuple(f"z{i}" for i in range(6))
    n_det = 150
    n_iso = 300
    rows: Dict[str, Any] = {}

    for name, rho in (("contract_0p8", 0.8), ("expand_1p15", 1.15)):
        series = isotropic_series(rho, n=n_det, dim=6, sigma=0.0, rng=rng)
        rec = qualify_recording_hc_oer(series, columns=columns, rng=rng)
        rec["regime"] = name
        rec["rho"] = float(rho)
        rec["sigma"] = 0.0
        rec["gamma_theory"] = expected_isotropic_gamma(rho)
        rec.pop("epochs", None)
        rows[name] = rec

    sto = isotropic_series(0.8, n=n_iso, dim=6, sigma=0.02, rng=rng)
    rec_s = qualify_recording_hc_oer(sto, columns=columns, rng=rng)
    rec_s["regime"] = "contract_0p8_noise"
    rec_s["rho"] = 0.8
    rec_s["sigma"] = 0.02
    rec_s.pop("epochs", None)
    rows["contract_0p8_noise"] = rec_s

    n_fine = int(round(TRUTH_DURATION_SEC / DT_FINE))
    for regime in ("stable", "unstable"):
        x = downsample_to_grid(
            simulate_linear_sde(REGIMES[regime], dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng),
            dt_fine=DT_FINE,
            grid_dt=DT_2_SEC,
        )
        rec = qualify_recording_hc_oer(x, columns=columns, rng=rng)
        rec["regime"] = regime
        rec["alpha_true"] = spectral_abscissa(REGIMES[regime])
        rec.pop("epochs", None)
        rows[regime] = rec

    white = rng.normal(size=(n_iso, 6)).astype(np.float32)
    rec_w = qualify_recording_hc_oer(white, columns=columns, rng=rng)
    rec_w["regime"] = "white"
    rec_w.pop("epochs", None)
    rows["white"] = rec_w

    contract = rows["contract_0p8"]["modes"]
    expand = rows["expand_1p15"]["modes"]
    g_c = float(contract["hc"]["gamma_median"])
    g_e = float(expand["hc"]["gamma_median"])
    theory_c = expected_isotropic_gamma(0.8)
    theory_e = expected_isotropic_gamma(1.15)
    g_sto = float(rows["contract_0p8_noise"]["modes"]["hc"]["gamma_median"])
    g_white = float(rows["white"]["modes"]["hc"]["gamma_median"])
    g_stable = float(rows["stable"]["modes"]["hc"]["gamma_median"])
    g_unstable = float(rows["unstable"]["modes"]["hc"]["gamma_median"])
    sde_sign_pass = bool(np.isfinite(g_stable) and np.isfinite(g_unstable) and g_stable < 0.0 and g_unstable > 0.0)
    c0_pass = bool(
        np.isfinite(g_c)
        and np.isfinite(g_e)
        and g_c < 0.0
        and g_e > 0.0
        and abs(g_c - theory_c) < ISOTROPIC_TOL
        and abs(g_e - theory_e) < ISOTROPIC_TOL
        and float(contract["shuffle_next"]["gamma_median"]) > g_c
        and float(contract["reverse"]["gamma_median"]) > 0.0
        and np.isfinite(g_sto)
        and g_sto < 0.0
        and np.isfinite(g_white)
        and abs(g_white) < WHITE_ABS_TOL
    )
    return {
        "grid": "2/2",
        "k": K_NEIGHBORS,
        "shell_q": [SHELL_Q_LO, SHELL_Q_HI],
        "k_min": K_MIN,
        "min_index_gap": MIN_INDEX_GAP,
        "dt_sec": DT_SEC,
        "isotropic_tol": ISOTROPIC_TOL,
        "c0_pass": c0_pass,
        "sde_sign_pass": sde_sign_pass,
        "rows": rows,
        "alpha_computed": False,
        "cpc_inspected": False,
        "not_spectral_abscissa": True,
    }


def _recording_coverage_pass(rec: Mapping[str, Any]) -> bool:
    cov = rec.get("coverage_median", float("nan"))
    return bool(np.isfinite(cov) and float(cov) >= COVERAGE_FRAC)


def hc_oer_decision(*, c0_pass: bool, recordings: Sequence[Mapping[str, Any]], blocked: bool) -> Dict[str, Any]:
    if blocked:
        branch = "BLOCKED"
    elif not c0_pass:
        branch = "C0_FAIL"
    else:
        n_cov = int(sum(_recording_coverage_pass(r) for r in recordings))
        branch = "QUALIFIED" if n_cov >= N_RECORDINGS_COVERAGE else "WEAK_COVERAGE"
    return {
        "branch": branch,
        "c0_pass": bool(c0_pass),
        "n_recordings_coverage_ge_half": int(sum(_recording_coverage_pass(r) for r in recordings)),
        "coverage_required": N_RECORDINGS_COVERAGE,
        "alpha_computed": False,
        "cpc_inspected": False,
        "not_spectral_abscissa": True,
        "next_if_qualified": "freeze HC-OER; do not open CPC in this gate",
        "next_if_weak": "do not open CPC; inspect coverage, do not retune k against 0284",
        "next_if_c0_fail": "estimator not qualified; do not score I-CARE",
    }


def run_icare_hc_oer(*, rng_seed: int = RNG_SEED) -> Dict[str, Any]:
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
            recs.append(
                {
                    "recording": rec_name,
                    "status": "MISSING_FEATURES",
                    "coverage_median": float("nan"),
                    "gamma_median": float("nan"),
                    "f_plus_median": float("nan"),
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
                        "coverage_median": float("nan"),
                        "gamma_median": float("nan"),
                        "f_plus_median": float("nan"),
                    }
                )
                continue
        z = part.loc[:, list(columns)].to_numpy(dtype=np.float32)
        rec_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        row = qualify_recording_hc_oer(z, columns=columns, rng=rec_rng)
        row["recording"] = rec_name
        recs.append(row)
    hc_meds = [float(r.get("gamma_median", np.nan)) for r in recs]
    hc_f = [float(r.get("f_plus_median", np.nan)) for r in recs]
    return {
        "status": "SCORED",
        "panel": PRIMARY_PANEL,
        "columns": list(columns),
        "missing_columns": missing,
        "n_recordings": len(recs),
        "recordings": recs,
        "gamma_median_across_recordings": float(np.nanmedian(hc_meds)) if hc_meds else float("nan"),
        "f_plus_median_across_recordings": float(np.nanmedian(hc_f)) if hc_f else float("nan"),
        "n_recordings_coverage_ge_half": int(sum(_recording_coverage_pass(r) for r in recs)),
        "features_dir": str(TYPE_C_FEATURES_DIR),
        "overlay": str(TYPE_C_OVERLAY),
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def hc_oer_001_report(*, rng_seed: int = RNG_SEED) -> Dict[str, Any]:
    c0 = run_c0_hc_oer(rng_seed=rng_seed)
    icare = run_icare_hc_oer(rng_seed=rng_seed)
    blocked = icare.get("status") == "BLOCKED"
    recs = icare.get("recordings", []) if not blocked else []
    decision = hc_oer_decision(c0_pass=bool(c0["c0_pass"]), recordings=recs, blocked=blocked)
    return {
        "gate": GATE,
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "state_name": NAME_TYPE_C_STATE,
        "not_mnps": True,
        "not_spectral_abscissa": True,
        "alpha_outside_acceptance": True,
        "cpc_blind": True,
        "k": K_NEIGHBORS,
        "k_min": K_MIN,
        "min_index_gap": MIN_INDEX_GAP,
        "dt_sec": DT_SEC,
        "c0": c0,
        "icare": icare,
        "decision": decision,
    }


def main() -> None:
    import json

    report = hc_oer_001_report()
    print("C0")
    print(f"  c0_pass={report['c0']['c0_pass']}")
    for name, row in report["c0"]["rows"].items():
        hc = row["modes"]["hc"]
        extra = ""
        if "gamma_theory" in row:
            extra = f" theory={row['gamma_theory']:.4f}"
        if "alpha_true" in row:
            extra = f" alpha={row['alpha_true']:.4f}"
        print(
            f"  {name:14s} gamma={hc['gamma_median']:.4f} f+={hc['f_plus_median']:.3f} "
            f"cov={row['coverage_median']:.3f}{extra}"
        )
        sh = row["modes"]["shuffle_next"]["gamma_median"]
        rv = row["modes"]["reverse"]["gamma_median"]
        print(f"                 shuffle={sh:.4f} reverse={rv:.4f}")
    icare = report["icare"]
    print(f"0284 status={icare['status']} branch={report['decision']['branch']}")
    if icare.get("status") == "SCORED":
        print(
            f"  median_gamma={icare['gamma_median_across_recordings']:.4f} "
            f"median_f+={icare['f_plus_median_across_recordings']:.3f} "
            f"n_cov={icare['n_recordings_coverage_ge_half']}/12"
        )
        mode_med = {}
        for mode in MODES:
            vals = [
                rec.get("modes", {}).get(mode, {}).get("gamma_median", float("nan"))
                for rec in icare["recordings"]
            ]
            finite = [v for v in vals if np.isfinite(v)]
            mode_med[mode] = float(np.median(finite)) if finite else float("nan")
        print(
            "  controls "
            + " ".join(f"{name}={mode_med[name]:.4f}" for name in MODES)
        )
        for rec in icare["recordings"]:
            print(
                f"  {str(rec.get('recording', '?'))[-20:]:20s} "
                f"g={rec.get('gamma_median', float('nan')):.4f} "
                f"f+={rec.get('f_plus_median', float('nan')):.3f} "
                f"cov={rec.get('coverage_median', float('nan')):.3f}"
            )
    out = Path(__file__).resolve().parents[2] / ".work" / "hc_oer_001_20260915"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hc_oer_001_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'hc_oer_001_report.json'}")


if __name__ == "__main__":
    main()

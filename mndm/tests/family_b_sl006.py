"""Family B SL-006: Type-C short-history closure (test-only).

Compares instantaneous M0  z_{t+1}=B0 z_t+c  against one-lag M1
z_{t+1}=B0 z_t+B1 z_{t-1}+c on the frozen C-primary 6D panel.

Controls: duplicate state [z_t,z_t] and within-block shuffled lag.
No alpha, no CPC, no 12D eigenvalue object. Not wired into summarize.
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
    pair_blocked_slices,
)
from family_b_sl004 import DT_2_SEC, TYPE_C_STEP_SEC
from family_b_sl005 import (
    BROAD_COVERAGE_FRACTION,
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

MODELS = ("m0", "m1", "duplicate", "shuffle")
N_TARGET = 6
N_LAG_PRED = 12
M1_N_PARAMS = N_TARGET * N_LAG_PRED + N_TARGET  # 78
M0_N_PARAMS = N_TARGET * N_TARGET + N_TARGET  # 42
HISTORY_STRONG_P_R = 0.6
SHUFFLE_SEED = 906


def consecutive_triples(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite (z_{t-1}, z_t, z_{t+1}) rows. Never stitch across a gap."""
    z = np.asarray(z, dtype=np.float32)
    finite = np.isfinite(z).all(axis=1)
    lag = []
    now = []
    nxt = []
    for t in range(1, int(z.shape[0]) - 1):
        if bool(finite[t - 1]) and bool(finite[t]) and bool(finite[t + 1]):
            lag.append(z[t - 1])
            now.append(z[t])
            nxt.append(z[t + 1])
    if not now:
        empty = np.empty((0, int(z.shape[1])), dtype=np.float32)
        return empty, empty, empty
    return np.stack(lag), np.stack(now), np.stack(nxt)


def history_split(n_triples: int, n_dim: int) -> Tuple[slice, slice, int]:
    """Shared blocked split for M0, M1, and both controls. min_train is M1's 78."""
    return pair_blocked_slices(
        n_triples,
        min_train=M1_N_PARAMS,
        min_test=max(4, n_dim),
    )


def triple_times(index: int) -> Tuple[int, int, int]:
    """Sample indices (t-1, t, t+1) owned by consecutive_triples row `index`."""
    return (int(index), int(index) + 1, int(index) + 2)


def split_sample_times(sl: slice) -> set:
    times: set = set()
    start = 0 if sl.start is None else int(sl.start)
    stop = int(sl.stop)
    for index in range(start, stop):
        times.update(triple_times(index))
    return times


def _predictors(lag: np.ndarray, now: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "m0": now,
        "m1": np.hstack([now, lag]),
        "duplicate": np.hstack([now, now]),
        "shuffle": np.hstack([now, lag]),
    }


def _shuffle_lag_block(pred: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()
    if out.shape[0] < 2 or out.shape[1] < 2:
        return out
    dim = out.shape[1] // 2
    out[:, dim:] = out[rng.permutation(out.shape[0]), dim:]
    return out


def _score_model(
    pred_train: np.ndarray,
    tgt_train: np.ndarray,
    pred_test: np.ndarray,
    tgt_test: np.ndarray,
) -> Dict[str, Any]:
    fitted = fit_global_affine(pred_train, tgt_train, ridge_alpha=RIDGE_ALPHA)
    if fitted is None:
        return {
            "status": STATUS_TRANSITION_OOS_FAIL,
            "rel_mse_oos": float("nan"),
            "oos_pass": False,
            "n_pred": int(pred_train.shape[1]),
        }
    rel = _score_affine(pred_test, tgt_test, *fitted)
    oos_pass = bool(np.isfinite(rel) and float(rel) < OOS_THRESHOLD)
    return {
        "status": STATUS_TRANSITION_OOS_PASS if oos_pass else STATUS_TRANSITION_OOS_FAIL,
        "rel_mse_oos": float(rel) if np.isfinite(rel) else float("nan"),
        "oos_pass": oos_pass,
        "n_pred": int(pred_train.shape[1]),
    }


def qualify_history_epoch(
    z: np.ndarray,
    *,
    columns: Sequence[str],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    assert_not_mnps_state(columns)
    z = apply_type_c_transforms(z, columns)
    lag, now, nxt = consecutive_triples(z)
    n_triples = int(now.shape[0])
    n_dim = int(now.shape[1]) if now.ndim == 2 else 0
    base = {"n_triples": n_triples, "n_dim": n_dim, "embargo_pairs": 0}
    try:
        train_sl, test_sl, gap = history_split(n_triples, n_dim)
    except ValueError:
        models = {
            name: {
                "status": STATUS_TRANSITION_INSUFFICIENT,
                "rel_mse_oos": float("nan"),
                "oos_pass": False,
                "n_pred": N_LAG_PRED if name != "m0" else N_TARGET,
            }
            for name in MODELS
        }
        return {
            **base,
            "models": models,
            "n_train": 0,
            "n_test": 0,
            "status": STATUS_TRANSITION_INSUFFICIENT,
        }
    preds = _predictors(lag, now)
    tgt_tr, tgt_te = nxt[train_sl], nxt[test_sl]
    models: Dict[str, Any] = {}
    for name in MODELS:
        pred = preds[name]
        pred_tr, pred_te = pred[train_sl], pred[test_sl]
        if name == "shuffle":
            pred_tr = _shuffle_lag_block(pred_tr, rng)
            pred_te = _shuffle_lag_block(pred_te, rng)
        models[name] = _score_model(pred_tr, tgt_tr, pred_te, tgt_te)
    return {
        **base,
        "models": models,
        "embargo_pairs": int(gap),
        "n_train": int(tgt_tr.shape[0]),
        "n_test": int(tgt_te.shape[0]),
        "status": "SCORED",
    }


def _model_recording_stats(epoch_rows: Sequence[Mapping[str, Any]], model: str) -> Dict[str, Any]:
    eligible = [
        row["models"][model]
        for row in epoch_rows
        if row.get("status") != STATUS_TRANSITION_INSUFFICIENT
    ]
    n_pass = int(sum(bool(m["oos_pass"]) for m in eligible))
    n_eligible = len(eligible)
    rels = [float(m["rel_mse_oos"]) for m in eligible if np.isfinite(m.get("rel_mse_oos", np.nan))]
    p_r = float(n_pass / n_eligible) if n_eligible else float("nan")
    return {
        "n_epochs": len(epoch_rows),
        "n_eligible": n_eligible,
        "n_oos_pass": n_pass,
        "p_r": p_r,
        "rel_mse_median": float(np.median(rels)) if rels else float("nan"),
        "status_counts": dict(Counter(m["status"] for m in eligible)) if eligible else {},
    }


def qualify_recording_history(
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
        epoch_rows.append(qualify_history_epoch(epoch, columns=columns, rng=epoch_rng))
    by_model = {name: _model_recording_stats(epoch_rows, name) for name in MODELS}
    m0 = by_model["m0"]
    m1 = by_model["m1"]
    return {
        "state_name": NAME_TYPE_C_STATE,
        "panel": PRIMARY_PANEL,
        "models": by_model,
        "delta_rel_mse": (
            float(m1["rel_mse_median"] - m0["rel_mse_median"])
            if np.isfinite(m1["rel_mse_median"]) and np.isfinite(m0["rel_mse_median"])
            else float("nan")
        ),
        "delta_p_r": (
            float(m1["p_r"] - m0["p_r"])
            if np.isfinite(m1["p_r"]) and np.isfinite(m0["p_r"])
            else float("nan")
        ),
        "p_r": m1["p_r"],
        "rel_mse_median": m1["rel_mse_median"],
        "epochs": epoch_rows,
    }


def _median_finite(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def history_decision(coverages: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    m0 = coverages["m0"]
    m1 = coverages["m1"]
    dup = coverages["duplicate"]
    shuf = coverages["shuffle"]
    m1_rel = float(m1["recording_rel_mse_median"])
    m1_p = float(m1["median_p_r"])
    control_rel = [
        float(m0["recording_rel_mse_median"]),
        float(dup["recording_rel_mse_median"]),
        float(shuf["recording_rel_mse_median"]),
    ]
    control_p = [float(m0["median_p_r"]), float(dup["median_p_r"]), float(shuf["median_p_r"])]
    beats_controls = bool(
        np.isfinite(m1_rel)
        and np.isfinite(m1_p)
        and all(np.isfinite(v) for v in control_rel + control_p)
        and m1_rel < min(control_rel)
        and m1_p > max(control_p)
    )
    strong = bool(
        beats_controls
        and bool(m1["closure_pass"])
        and m1_p >= HISTORY_STRONG_P_R
    )
    if strong:
        branch = "STRONG_HISTORY"
    elif beats_controls and (m1_rel >= 0.90 or m1_p < BROAD_COVERAGE_P_R):
        branch = "MODEST_HISTORY"
    elif beats_controls:
        branch = "MODEST_HISTORY"
    else:
        branch = "NO_HISTORY"
    return {
        "beats_controls": beats_controls,
        "strong_history_pass": strong,
        "branch": branch,
        "delta_rel_mse_vs_m0": float(m1_rel - control_rel[0]) if np.isfinite(m1_rel) else float("nan"),
        "delta_p_r_vs_m0": float(m1_p - control_p[0]) if np.isfinite(m1_p) else float("nan"),
        "history_strong_p_r": HISTORY_STRONG_P_R,
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def simulate_lagged_linear(
    *,
    a0: np.ndarray,
    a1: Optional[np.ndarray],
    n: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    a0 = np.asarray(a0, dtype=np.float64)
    dim = int(a0.shape[0])
    z = np.zeros((n, dim), dtype=np.float64)
    z[0] = rng.normal(size=dim)
    z[1] = rng.normal(size=dim)
    a1_m = np.zeros_like(a0) if a1 is None else np.asarray(a1, dtype=np.float64)
    for t in range(1, n - 1):
        z[t + 1] = a0 @ z[t] + a1_m @ z[t - 1] + float(sigma) * rng.normal(size=dim)
        if not np.isfinite(z[t + 1]).all() or np.max(np.abs(z[t + 1])) > 1e6:
            z[t + 1 :] = np.nan
            break
    return z.astype(np.float32)


def run_c0_history(*, rng_seed: int = 906) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    n_grid = int(round(TRUTH_DURATION_SEC / DT_2_SEC))
    columns = tuple(f"z{i}" for i in range(N_TARGET))
    # First-order: same 6D stable SDE family as SL-005 C0, then 2/2 downsample.
    j_true = np.diag(np.linspace(-0.08, -0.04, N_TARGET)).astype(np.float64)
    n_fine = int(round(TRUTH_DURATION_SEC / DT_FINE))
    first = downsample_to_grid(
        simulate_linear_sde(j_true, dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng),
        dt_fine=DT_FINE,
        grid_dt=DT_2_SEC,
    )
    a1 = np.diag(np.linspace(0.55, 0.40, N_TARGET))
    second = simulate_lagged_linear(a0=0.15 * np.eye(N_TARGET), a1=a1, n=n_grid, sigma=PROCESS_SIGMA, rng=rng)
    rows = {}
    for name, series in (("first_order", first), ("second_order", second)):
        rec = qualify_recording_history(series, columns=columns, rng=rng)
        rec["regime"] = name
        rec.pop("epochs", None)
        rows[name] = rec
    return {
        "grid": "2/2",
        "n_params_m0": M0_N_PARAMS,
        "n_params_m1": M1_N_PARAMS,
        "ridge_alpha": RIDGE_ALPHA,
        "rows": rows,
    }


def run_icare_history(*, rng_seed: int = 906) -> Dict[str, Any]:
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
        row = qualify_recording_history(z, columns=columns, rng=rec_rng)
        row["recording"] = rec_name
        row.pop("epochs", None)
        recs.append(row)
    coverages = {}
    for name in MODELS:
        model_rows = []
        for rec in recs:
            stats = rec.get("models", {}).get(name, {})
            model_rows.append(
                {
                    "p_r": stats.get("p_r", rec.get("p_r", float("nan"))),
                    "rel_mse_median": stats.get("rel_mse_median", rec.get("rel_mse_median", float("nan"))),
                }
            )
        coverages[name] = coverage_summary(model_rows)
    decision = history_decision(coverages)
    return {
        "status": "SCORED",
        "panel": PRIMARY_PANEL,
        "columns": list(columns),
        "n_params_m0": M0_N_PARAMS,
        "n_params_m1": M1_N_PARAMS,
        "missing_columns": missing,
        "n_recordings": len(recs),
        "recordings": recs,
        "coverage": coverages,
        "decision": decision,
        "features_dir": str(TYPE_C_FEATURES_DIR),
        "overlay": str(TYPE_C_OVERLAY),
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def sl006_report(*, rng_seed: int = 906) -> Dict[str, Any]:
    return {
        "gate": "TYPE_C_SHORT_HISTORY",
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "state_name": NAME_TYPE_C_STATE,
        "not_mnps": True,
        "alpha_outside_acceptance": True,
        "cpc_blind": True,
        "threshold": OOS_THRESHOLD,
        "history_strong_p_r": HISTORY_STRONG_P_R,
        "predictor": "z_{t+1}=B0 z_t + B1 z_{t-1}+c  (6D target, not 12D generator)",
        "c0": run_c0_history(rng_seed=rng_seed),
        "icare": run_icare_history(rng_seed=rng_seed),
        "next_if_strong": "augmented operator, then generator licensing of the delay system, still CPC-blind",
        "next_if_modest": "do not stack lags; hidden state not a one-step delay",
        "next_if_none": "latent state-space model; stop manual feature-chart Jacobians",
    }


def main() -> None:
    import json

    report = sl006_report()
    print("C0 first-order vs second-order")
    for name, row in report["c0"]["rows"].items():
        m0 = row["models"]["m0"]
        m1 = row["models"]["m1"]
        print(
            f"  {name:13s} M0 p_r={m0['p_r']:.3f} rel={m0['rel_mse_median']:.3f} "
            f"M1 p_r={m1['p_r']:.3f} rel={m1['rel_mse_median']:.3f} "
            f"d_rel={row['delta_rel_mse']:.3f} d_p={row['delta_p_r']:.3f}"
        )
    icare = report["icare"]
    print(f"0284 status={icare['status']}")
    if icare.get("status") == "SCORED":
        dec = icare["decision"]
        for name, cov in icare["coverage"].items():
            print(
                f"  {name:10s} median_p_r={cov['median_p_r']:.3f} "
                f"P>0.5={cov['p_recordings_p_r_gt_half']:.3f} "
                f"rel={cov['recording_rel_mse_median']:.3f} closure={cov['closure_pass']}"
            )
        print(
            f"  branch={dec['branch']} beats_controls={dec['beats_controls']} "
            f"d_rel={dec['delta_rel_mse_vs_m0']:.3f} d_p={dec['delta_p_r_vs_m0']:.3f}"
        )
    out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl006_20260914"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sl006_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'sl006_report.json'}")


if __name__ == "__main__":
    main()

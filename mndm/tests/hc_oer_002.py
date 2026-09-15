"""HC-OER-002: history increment vs z-only shell (test-only).

    delta_gamma = gamma_HC - gamma_z_only   (epoch-paired)

Frozen HC-OER-001 estimator. Not alpha, not F_H, not CPC, not summarize.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

TESTS = Path(__file__).resolve().parent
_SRC = TESTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import numpy as np

from family_b_sl002 import PROCESS_SIGMA, _json_ready
from family_b_sl003 import EPOCH_SEC, n_epoch_samples, epoch_windows
from family_b_sl004 import TYPE_C_STEP_SEC
from family_b_sl005 import (
    NAME_TYPE_C_STATE,
    PANELS,
    PRIMARY_PANEL,
    TRUTH_DURATION_SEC,
    TYPE_C_FEATURES_DIR,
    TYPE_C_OVERLAY,
    drop_cpc_columns,
    _load_type_c_features,
    _recording_groups,
)
from family_b_sl006 import N_TARGET, simulate_lagged_linear
from hc_oer_001 import (
    COVERAGE_FRAC,
    DT_SEC,
    N_RECORDINGS_COVERAGE,
    STATUS_SCORED,
    isotropic_series,
    score_epoch_hc_oer,
)

GATE = "HC_OER_002"
RNG_SEED = 1002
DELTA_FLOOR = 0.02
WHITE_DELTA_TOL = 0.03
N_SIGN_CONSISTENT = 8


def _finite(value: Any) -> bool:
    return bool(np.isfinite(value))


def epoch_delta_gamma(epoch_row: Mapping[str, Any]) -> float:
    modes = epoch_row.get("modes", {})
    hc = modes.get("hc", {}).get("gamma_median", float("nan"))
    z_only = modes.get("z_only", {}).get("gamma_median", float("nan"))
    if not (_finite(hc) and _finite(z_only)):
        return float("nan")
    return float(hc) - float(z_only)


def qualify_recording_delta(
    z: np.ndarray,
    *,
    columns: Sequence[str],
    rng: np.random.Generator,
    dt: float = DT_SEC,
) -> Dict[str, Any]:
    n_epoch = n_epoch_samples(EPOCH_SEC["primary"], dt)
    epoch_rows = []
    deltas: List[float] = []
    for i, epoch in enumerate(epoch_windows(z, n_epoch)):
        epoch_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i)
        row = score_epoch_hc_oer(epoch, columns=columns, rng=epoch_rng, dt=dt)
        epoch_rows.append(row)
        if row.get("status") == STATUS_SCORED:
            delta = epoch_delta_gamma(row)
            if _finite(delta):
                deltas.append(float(delta))
    hc_meds = [
        float(row["modes"]["hc"]["gamma_median"])
        for row in epoch_rows
        if row.get("status") == STATUS_SCORED and _finite(row["modes"]["hc"]["gamma_median"])
    ]
    z_meds = [
        float(row["modes"]["z_only"]["gamma_median"])
        for row in epoch_rows
        if row.get("status") == STATUS_SCORED and _finite(row["modes"]["z_only"]["gamma_median"])
    ]
    coverages = [float(row.get("coverage", float("nan"))) for row in epoch_rows]
    return {
        "state_name": NAME_TYPE_C_STATE,
        "panel": PRIMARY_PANEL,
        "n_epochs": len(epoch_rows),
        "n_scored_epochs": len(deltas),
        "gamma_hc_median": float(np.median(hc_meds)) if hc_meds else float("nan"),
        "gamma_z_only_median": float(np.median(z_meds)) if z_meds else float("nan"),
        "delta_gamma_median": float(np.median(deltas)) if deltas else float("nan"),
        "delta_gamma_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "coverage_median": float(np.nanmedian(coverages)) if coverages else float("nan"),
        "n_delta_positive": int(sum(d > 0.0 for d in deltas)),
        "n_delta_negative": int(sum(d < 0.0 for d in deltas)),
    }


def adaptive_threshold(first_order_abs_delta: float) -> float:
    return float(max(DELTA_FLOOR, 2.0 * abs(float(first_order_abs_delta))))


def increment_decision(
    *,
    c0: Mapping[str, Any],
    recordings: Sequence[Mapping[str, Any]],
    blocked: bool,
) -> Dict[str, Any]:
    if blocked:
        branch = "BLOCKED"
    elif not bool(c0.get("first_order_ok")) or not bool(c0.get("white_ok")):
        branch = "C0_FAIL"
    elif not bool(c0.get("ar2_ok")):
        branch = "METHOD_LIMITED"
    else:
        deltas = [float(r.get("delta_gamma_median", np.nan)) for r in recordings]
        finite = [d for d in deltas if _finite(d)]
        med = float(np.median(finite)) if finite else float("nan")
        thresh = float(c0["threshold"])
        if not _finite(med) or abs(med) < thresh:
            branch = "NO_INCREMENT"
        else:
            sign = 1.0 if med > 0.0 else -1.0
            n_same = int(sum(np.sign(d) == sign for d in finite))
            branch = "HISTORY_INCREMENT" if n_same >= N_SIGN_CONSISTENT else "MIXED_SIGN"
    return {
        "branch": branch,
        "c0_first_order_ok": bool(c0.get("first_order_ok")),
        "c0_ar2_ok": bool(c0.get("ar2_ok")),
        "c0_white_ok": bool(c0.get("white_ok")),
        "threshold": float(c0.get("threshold", DELTA_FLOOR)),
        "n_sign_required": N_SIGN_CONSISTENT,
        "alpha_computed": False,
        "cpc_inspected": False,
        "not_spectral_abscissa": True,
        "next_if_increment": "replicate Δγ on another CPC-blind subject; still no CPC",
        "next_if_none": "HC-OER remains a one-step expansion rate without a licensed history increment",
        "next_if_method_limited": "expansion increment cannot see AR(2) history; do not retune on 0284",
    }


def run_c0_delta(*, rng_seed: int = RNG_SEED) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    columns = tuple(f"z{i}" for i in range(N_TARGET))
    first = isotropic_series(0.8, n=150, dim=N_TARGET, sigma=0.0, rng=rng)
    rec_first = qualify_recording_delta(first, columns=columns, rng=rng)
    rec_first["regime"] = "first_order_isotropic"

    n_grid = int(round(TRUTH_DURATION_SEC / DT_SEC))
    a1 = np.diag(np.linspace(0.55, 0.40, N_TARGET))
    ar2 = simulate_lagged_linear(a0=0.15 * np.eye(N_TARGET), a1=a1, n=n_grid, sigma=PROCESS_SIGMA, rng=rng)
    rec_ar2 = qualify_recording_delta(ar2, columns=columns, rng=rng)
    rec_ar2["regime"] = "ar2"

    white = rng.normal(size=(300, N_TARGET)).astype(np.float32)
    rec_white = qualify_recording_delta(white, columns=columns, rng=rng)
    rec_white["regime"] = "white"

    d1 = float(rec_first["delta_gamma_median"])
    d2 = float(rec_ar2["delta_gamma_median"])
    dw = float(rec_white["delta_gamma_median"])
    thresh = adaptive_threshold(d1 if _finite(d1) else 0.0)
    first_ok = bool(_finite(d1) and abs(d1) < DELTA_FLOOR)
    ar2_ok = bool(_finite(d2) and abs(d2) > abs(d1 if _finite(d1) else 0.0) and abs(d2) >= DELTA_FLOOR)
    white_ok = bool(_finite(dw) and abs(dw) < WHITE_DELTA_TOL)
    return {
        "delta_floor": DELTA_FLOOR,
        "threshold": thresh,
        "first_order_ok": first_ok,
        "ar2_ok": ar2_ok,
        "white_ok": white_ok,
        "c0_increment_identifiable": bool(first_ok and ar2_ok and white_ok),
        "rows": {
            "first_order_isotropic": rec_first,
            "ar2": rec_ar2,
            "white": rec_white,
        },
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def run_icare_delta(*, rng_seed: int = RNG_SEED) -> Dict[str, Any]:
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
                    "delta_gamma_median": float("nan"),
                    "coverage_median": float("nan"),
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
                        "delta_gamma_median": float("nan"),
                        "coverage_median": float("nan"),
                    }
                )
                continue
        z = part.loc[:, list(columns)].to_numpy(dtype=np.float32)
        rec_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        row = qualify_recording_delta(z, columns=columns, rng=rec_rng)
        row["recording"] = rec_name
        recs.append(row)
    deltas = [float(r.get("delta_gamma_median", np.nan)) for r in recs]
    finite = [d for d in deltas if _finite(d)]
    med = float(np.median(finite)) if finite else float("nan")
    n_pos = int(sum(d > 0.0 for d in finite))
    n_neg = int(sum(d < 0.0 for d in finite))
    n_cov = int(
        sum(
            _finite(r.get("coverage_median", np.nan)) and float(r["coverage_median"]) >= COVERAGE_FRAC
            for r in recs
        )
    )
    return {
        "status": "SCORED",
        "panel": PRIMARY_PANEL,
        "columns": list(columns),
        "missing_columns": missing,
        "n_recordings": len(recs),
        "recordings": recs,
        "delta_gamma_median_across_recordings": med,
        "n_recordings_delta_positive": n_pos,
        "n_recordings_delta_negative": n_neg,
        "n_recordings_coverage_ge_half": n_cov,
        "coverage_required": N_RECORDINGS_COVERAGE,
        "features_dir": str(TYPE_C_FEATURES_DIR),
        "overlay": str(TYPE_C_OVERLAY),
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def hc_oer_002_report(*, rng_seed: int = RNG_SEED) -> Dict[str, Any]:
    c0 = run_c0_delta(rng_seed=rng_seed)
    icare = run_icare_delta(rng_seed=rng_seed)
    blocked = icare.get("status") == "BLOCKED"
    recs = icare.get("recordings", []) if not blocked else []
    decision = increment_decision(c0=c0, recordings=recs, blocked=blocked)
    return {
        "gate": GATE,
        "parent_gate": "HC_OER_001",
        "estimator_frozen": True,
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "state_name": NAME_TYPE_C_STATE,
        "not_spectral_abscissa": True,
        "alpha_outside_acceptance": True,
        "cpc_blind": True,
        "delta_floor": DELTA_FLOOR,
        "c0": c0,
        "icare": icare,
        "decision": decision,
    }


def main() -> None:
    import json

    report = hc_oer_002_report()
    c0 = report["c0"]
    print("C0 increment")
    for name, row in c0["rows"].items():
        print(
            f"  {name:24s} dgamma={row['delta_gamma_median']:.4f} "
            f"hc={row['gamma_hc_median']:.4f} z={row['gamma_z_only_median']:.4f} "
            f"cov={row['coverage_median']:.3f}"
        )
    print(
        f"  first_ok={c0['first_order_ok']} ar2_ok={c0['ar2_ok']} "
        f"white_ok={c0['white_ok']} thresh={c0['threshold']:.4f}"
    )
    icare = report["icare"]
    dec = report["decision"]
    print(f"0284 status={icare['status']} branch={dec['branch']}")
    if icare.get("status") == "SCORED":
        print(
            f"  median_dgamma={icare['delta_gamma_median_across_recordings']:.4f} "
            f"+={icare['n_recordings_delta_positive']} "
            f"-={icare['n_recordings_delta_negative']} "
            f"n_cov={icare['n_recordings_coverage_ge_half']}/12"
        )
        for rec in icare["recordings"]:
            print(
                f"  {str(rec.get('recording', '?'))[-20:]:20s} "
                f"dgamma={rec.get('delta_gamma_median', float('nan')):.4f} "
                f"hc={rec.get('gamma_hc_median', float('nan')):.4f} "
                f"z={rec.get('gamma_z_only_median', float('nan')):.4f}"
            )
    out = Path(__file__).resolve().parents[2] / ".work" / "hc_oer_002_20260915"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hc_oer_002_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'hc_oer_002_report.json'}")


if __name__ == "__main__":
    main()

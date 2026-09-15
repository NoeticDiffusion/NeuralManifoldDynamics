"""Family B SL-005 / Type-C-1: frozen z^C panels, transition closure only.

The scientific object is type_c_state (z^C), not MNPS. Spectral abscissa
and CPC are out of scope. This module is test-only and is not wired into
summarize.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

TESTS = Path(__file__).resolve().parent
_SRC = TESTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import numpy as np
import pandas as pd

from family_b_sl002 import (
    DT_FINE,
    OOS_THRESHOLD,
    PROCESS_SIGMA,
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
    consecutive_one_step_pairs,
    epoch_windows,
    n_epoch_samples,
    pair_blocked_slices,
)
from family_b_sl004 import DT_2_SEC, TYPE_C_STEP_SEC, TYPE_C_WINDOW_SEC

STATUS_TRANSITION_OOS_PASS = "TRANSITION_OOS_PASS"
NAME_TYPE_C_STATE = "type_c_state"
BROAD_COVERAGE_P_R = 0.5
BROAD_COVERAGE_FRACTION = 0.5
TRUTH_DURATION_SEC = 30 * 60.0

TYPE_C_OVERLAY = (
    TESTS.parent
    / "config"
    / "sources"
    / "other"
    / "config_ingest_physionet_i-care_2_1_part1_0_12h_dynamical_families_type_c.yaml"
)
TYPE_C_PROCESSED = Path(r"K:/processed/physionet_icare_2_1_part1_0_12h_dynamical_families_type_c_20260914")
TYPE_C_FEATURES_DIR = TYPE_C_PROCESSED / "physionet_icare_2_1"

PANELS: Dict[str, Tuple[str, ...]] = {
    "c_primary": (
        "eeg_theta",
        "eeg_alpha",
        "eeg_beta",
        "eeg_permutation_entropy",
        "eeg_hjorth_mobility",
        "eeg_hjorth_complexity",
    ),
    "c_spectral_sensitivity": (
        "eeg_theta",
        "eeg_alpha",
        "eeg_beta",
        "eeg_beta_alpha",
        "eeg_alpha_theta",
    ),
    "c_dynamical_sensitivity": (
        "eeg_theta",
        "eeg_alpha",
        "eeg_beta",
        "eeg_permutation_entropy",
        "eeg_hjorth_activity",
        "eeg_hjorth_mobility",
        "eeg_hjorth_complexity",
    ),
}
PRIMARY_PANEL = "c_primary"
LOG10_FEATURES = frozenset({"eeg_theta", "eeg_alpha", "eeg_beta", "eeg_hjorth_activity"})
FORBIDDEN_FEATURE_TOKENS = (
    "eeg_delta",
    "eeg_gamma",
    "mnps",
    "coords_9d",
    "type_c_is_mnps",
)
CPC_BLIND_COLUMN_RE = re.compile(r"(cpc|outcome|prognos)", re.IGNORECASE)


def affine_n_params(n_dim: int) -> int:
    return int(n_dim) * (int(n_dim) + 1)


def panel_min_train(n_dim: int) -> int:
    return affine_n_params(n_dim)


def assert_not_mnps_state(columns: Sequence[str]) -> None:
    lowered = [str(c).lower() for c in columns]
    if any(name.startswith("m_") or name.startswith("d_") or name.startswith("e_") for name in lowered):
        raise ValueError("Type-C state must not use MNPS axis names")
    for token in FORBIDDEN_FEATURE_TOKENS:
        if any(token in name for name in lowered):
            raise ValueError(f"Type-C state contains forbidden token {token}")


def drop_cpc_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in frame.columns if CPC_BLIND_COLUMN_RE.search(str(c)) is None]
    return frame.loc[:, keep]


def apply_type_c_transforms(x: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    for j, name in enumerate(columns):
        if name in LOG10_FEATURES:
            col = np.asarray(out[:, j], dtype=np.float64)
            positive = np.isfinite(col) & (col > 0.0)
            col[positive] = np.log10(col[positive])
            col[~positive] = np.nan
            out[:, j] = col.astype(np.float32)
    return out


def qualify_type_c_closure(
    z: np.ndarray,
    *,
    columns: Sequence[str],
) -> Dict[str, Any]:
    """Blocked OOS next-state closure for z^C. Does not compute log(F) or alpha."""
    assert_not_mnps_state(columns)
    raw = np.asarray(z, dtype=np.float32)
    n_log10_nonpositive = 0
    for j, name in enumerate(columns):
        if name in LOG10_FEATURES and j < raw.shape[1]:
            col = raw[:, j]
            n_log10_nonpositive += int(np.sum(np.isfinite(col) & (col <= 0.0)))
    z = apply_type_c_transforms(raw, columns)
    src, tgt = consecutive_one_step_pairs(z)
    n_pairs = int(src.shape[0])
    n_dim = int(z.shape[1]) if z.ndim == 2 else 0
    base = {
        "n_pairs": n_pairs,
        "n_dim": n_dim,
        "n_log10_nonpositive": int(n_log10_nonpositive),
        "oos_pass": False,
    }
    try:
        train_sl, test_sl, gap = pair_blocked_slices(
            n_pairs,
            min_train=panel_min_train(n_dim),
            min_test=max(4, n_dim),
        )
    except ValueError:
        return {
            **base,
            "status": STATUS_TRANSITION_INSUFFICIENT,
            "rel_mse_oos": float("nan"),
            "embargo_pairs": 0,
        }
    fitted = fit_global_affine(src[train_sl], tgt[train_sl])
    if fitted is None:
        return {
            **base,
            "status": STATUS_TRANSITION_OOS_FAIL,
            "rel_mse_oos": float("nan"),
            "embargo_pairs": int(gap),
        }
    rel = _score_affine(src[test_sl], tgt[test_sl], *fitted)
    oos_pass = bool(np.isfinite(rel) and float(rel) < OOS_THRESHOLD)
    return {
        **base,
        "status": STATUS_TRANSITION_OOS_PASS if oos_pass else STATUS_TRANSITION_OOS_FAIL,
        "rel_mse_oos": float(rel) if np.isfinite(rel) else float("nan"),
        "embargo_pairs": int(gap),
        "oos_pass": oos_pass,
    }


def qualify_recording_panel(z: np.ndarray, *, columns: Sequence[str], dt: float = DT_2_SEC) -> Dict[str, Any]:
    n_epoch = n_epoch_samples(EPOCH_SEC["primary"], dt)
    epoch_rows = []
    for epoch in epoch_windows(z, n_epoch):
        epoch_rows.append(qualify_type_c_closure(epoch, columns=columns))
    eligible = [row for row in epoch_rows if row["status"] != STATUS_TRANSITION_INSUFFICIENT]
    n_pass = int(sum(bool(row["oos_pass"]) for row in eligible))
    n_eligible = len(eligible)
    rels = [float(row["rel_mse_oos"]) for row in eligible if np.isfinite(row["rel_mse_oos"])]
    p_r = float(n_pass / n_eligible) if n_eligible else float("nan")
    return {
        "state_name": NAME_TYPE_C_STATE,
        "n_epochs": len(epoch_rows),
        "n_eligible": n_eligible,
        "n_oos_pass": n_pass,
        "p_r": p_r,
        "rel_mse_median": float(np.median(rels)) if rels else float("nan"),
        "status_counts": dict(Counter(row["status"] for row in epoch_rows)),
        "epochs": epoch_rows,
    }


def coverage_summary(recording_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    p_vals = []
    rels = []
    for row in recording_rows:
        p = row.get("p_r", float("nan"))
        p_vals.append(0.0 if not np.isfinite(p) else float(p))
        rel = row.get("rel_mse_median", float("nan"))
        rels.append(1.0 if not np.isfinite(rel) else float(rel))
    n = len(recording_rows)
    n_broad = int(sum(p > BROAD_COVERAGE_P_R for p in p_vals))
    p_broad = float(n_broad / n) if n else float("nan")
    median_p = float(np.median(p_vals)) if p_vals else float("nan")
    median_rel = float(np.median(rels)) if rels else float("nan")
    closure_pass = bool(
        np.isfinite(median_rel)
        and median_rel < OOS_THRESHOLD
        and np.isfinite(p_broad)
        and p_broad >= BROAD_COVERAGE_FRACTION
    )
    return {
        "n_recordings": n,
        "median_p_r": median_p,
        "p_recordings_p_r_gt_half": p_broad,
        "n_recordings_p_r_gt_half": n_broad,
        "recording_rel_mse_median": median_rel,
        "threshold": OOS_THRESHOLD,
        "broad_coverage_p_r": BROAD_COVERAGE_P_R,
        "broad_coverage_fraction": BROAD_COVERAGE_FRACTION,
        "closure_pass": closure_pass,
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def run_c0_type_c_dimension(*, rng_seed: int = 805, n_dim: int = 6) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    j_true = np.diag(np.linspace(-0.08, -0.04, n_dim)).astype(np.float64)
    n_fine = int(round(TRUTH_DURATION_SEC / DT_FINE))
    x_fine = simulate_linear_sde(j_true, dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng)
    x2 = downsample_to_grid(x_fine, dt_fine=DT_FINE, grid_dt=DT_2_SEC)
    columns = tuple(f"z{i}" for i in range(n_dim))
    rec = qualify_recording_panel(x2, columns=columns, dt=DT_2_SEC)
    rec["regime"] = "stable_diag"
    rec["c0_pass"] = bool(np.isfinite(rec["p_r"]) and rec["p_r"] >= 0.5 and rec["rel_mse_median"] < OOS_THRESHOLD)
    return rec


def _load_type_c_features() -> Optional[pd.DataFrame]:
    parquet = TYPE_C_FEATURES_DIR / "features.parquet"
    csv = TYPE_C_FEATURES_DIR / "features.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _recording_groups(frame: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    key = "file" if "file" in frame.columns else None
    if key is None:
        return [("recording", frame)]
    out = []
    for name, part in frame.groupby(key, sort=True):
        out.append((str(name), part))
    return out


def qualify_features_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    frame = drop_cpc_columns(frame)
    panel_reports: Dict[str, Any] = {}
    for panel_name, columns in PANELS.items():
        missing = [c for c in columns if c not in frame.columns]
        recs = []
        for rec_name, part in _recording_groups(frame):
            if missing:
                recs.append(
                    {
                        "recording": rec_name,
                        "status": "MISSING_FEATURES",
                        "missing": missing,
                        "p_r": float("nan"),
                        "rel_mse_median": float("nan"),
                        "n_eligible": 0,
                        "n_oos_pass": 0,
                    }
                )
                continue
            part = part.copy()
            if "t_start" in part.columns:
                part = part.sort_values(["t_start", "epoch_id"] if "epoch_id" in part.columns else ["t_start"])
                starts = pd.to_numeric(part["t_start"], errors="coerce").to_numpy(dtype=float)
                hops = np.diff(starts[np.isfinite(starts)])
                if hops.size and not np.allclose(np.median(hops), TYPE_C_STEP_SEC, atol=0.05):
                    recs.append(
                        {
                            "recording": rec_name,
                            "status": "GRID_INVALID",
                            "p_r": float("nan"),
                            "rel_mse_median": float("nan"),
                            "n_eligible": 0,
                            "n_oos_pass": 0,
                            "median_hop_sec": float(np.median(hops)),
                        }
                    )
                    continue
            z = part.loc[:, list(columns)].to_numpy(dtype=np.float32)
            row = qualify_recording_panel(z, columns=columns)
            row["recording"] = rec_name
            recs.append(row)
        panel_reports[panel_name] = {
            "columns": list(columns),
            "n_dim": len(columns),
            "n_affine_params": affine_n_params(len(columns)),
            "missing_columns": missing,
            "recordings": recs,
            "coverage": coverage_summary(recs),
        }
    return {
        "state_name": NAME_TYPE_C_STATE,
        "not_mnps": True,
        "primary_panel": PRIMARY_PANEL,
        "panels": panel_reports,
        "primary_closure_pass": bool(panel_reports[PRIMARY_PANEL]["coverage"]["closure_pass"]),
    }


def run_c2_icare_closure() -> Dict[str, Any]:
    frame = _load_type_c_features()
    if frame is None:
        return {
            "status": "BLOCKED",
            "reason": (
                "Type-C 2/2 features are not on disk. Run a CPC-blind features extract "
                "with the Type-C overlay before closure scoring. Do not slice 8 s mnps_3d."
            ),
            "overlay": str(TYPE_C_OVERLAY),
            "features_dir": str(TYPE_C_FEATURES_DIR),
            "alpha_computed": False,
            "cpc_inspected": False,
        }
    payload = qualify_features_frame(frame)
    payload["status"] = "SCORED"
    payload["n_rows"] = int(len(frame))
    payload["n_conventional_columns"] = int(sum("conventional" in str(c) for c in frame.columns))
    payload["overlay"] = str(TYPE_C_OVERLAY)
    payload["overlay_sha256_16"] = hashlib.sha256(TYPE_C_OVERLAY.read_bytes()).hexdigest()[:16]
    payload["features_dir"] = str(TYPE_C_FEATURES_DIR)
    payload["alpha_computed"] = False
    payload["cpc_inspected"] = False
    return payload


def sl005_report(*, rng_seed: int = 805) -> Dict[str, Any]:
    return {
        "gate": "TYPE_C_1",
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "state_name": NAME_TYPE_C_STATE,
        "not_mnps": True,
        "alpha_outside_acceptance": True,
        "cpc_blind": True,
        "threshold": OOS_THRESHOLD,
        "grid": "2/2",
        "epoch_sec": EPOCH_SEC["primary"],
        "panels": {name: list(cols) for name, cols in PANELS.items()},
        "c0_6d_stable": run_c0_type_c_dimension(rng_seed=rng_seed, n_dim=6),
        "c2": run_c2_icare_closure(),
        "next_if_closure_pass": "C3 generator licensing (log F / Δ), still CPC-blind",
        "next_if_closure_fail": "history augmentation / latent SSM; not 1/1 or feature search",
    }


def main() -> None:
    import json

    report = sl005_report()
    c0 = report["c0_6d_stable"]
    print(
        f"C0 6D stable p_r={c0['p_r']:.3f} relMSE={c0['rel_mse_median']:.3f} "
        f"pass={c0['c0_pass']} epochs={c0['n_oos_pass']}/{c0['n_eligible']}"
    )
    c2 = report["c2"]
    print(f"C2 status={c2['status']}")
    if c2.get("status") == "SCORED":
        for name, panel in c2["panels"].items():
            cov = panel["coverage"]
            print(
                f"  {name}: closure_pass={cov['closure_pass']} "
                f"median_p_r={cov['median_p_r']:.3f} "
                f"P(p_r>0.5)={cov['p_recordings_p_r_gt_half']:.3f} "
                f"rec_relMSE={cov['recording_rel_mse_median']:.3f}"
            )
    else:
        print(f"  {c2.get('reason')}")
    out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl005_20260914"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sl005_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'sl005_report.json'}")


if __name__ == "__main__":
    main()

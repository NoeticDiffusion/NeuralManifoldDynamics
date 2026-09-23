"""Family B SL-010: CPC-blind Type-C history-gain replication (test-only).

Reuses the frozen SL-006 M0/M1/duplicate/shuffle scorer. Discovery subject
0284 is a scorer-drift lock. Replication subject 0286 is selected by a
CPC-blind filename-hour rule, not by outcome.

Not alpha, not CPC, not an operator, not MNPS history_predictive_gain.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional

TESTS = Path(__file__).resolve().parent
_SRC = TESTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import numpy as np
import pandas as pd

from family_b_sl002 import OOS_THRESHOLD, RIDGE_ALPHA, _json_ready
from family_b_sl003 import EPOCH_SEC
from family_b_sl004 import TYPE_C_STEP_SEC
from family_b_sl005 import (
    PANELS,
    PRIMARY_PANEL,
    TYPE_C_FEATURES_DIR,
    TYPE_C_OVERLAY,
    _recording_groups,
    coverage_summary,
    drop_cpc_columns,
)
from family_b_sl006 import (
    HISTORY_STRONG_P_R,
    M0_N_PARAMS,
    M1_N_PARAMS,
    MODELS,
    history_decision,
    qualify_recording_history,
    run_c0_history,
    run_icare_history,
)

DISCOVERY_SUBJECT = "0284"
MIN_HOUR_FILES = 10
HOUR_LO = 1
HOUR_HI = 12
RECEIVED_TRAINING = Path(
    r"E:/Science_Datasets/physionet/received/i-care_2_1_random100_longitudinal/training"
)
TYPE_C_SL010_OVERLAY = (
    TESTS.parent
    / "config"
    / "sources"
    / "other"
    / "config_ingest_physionet_i-care_2_1_part1_0_12h_dynamical_families_type_c_sl010.yaml"
)
TYPE_C_SL010_PROCESSED = Path(
    r"K:/processed/physionet_icare_2_1_part1_0_12h_dynamical_families_type_c_sl010_20260918"
)
TYPE_C_SL010_FEATURES_DIR = TYPE_C_SL010_PROCESSED / "physionet_icare_2_1"


def select_replication_subject(
    received_training: Path,
    *,
    exclude: str = DISCOVERY_SUBJECT,
    min_hours: int = MIN_HOUR_FILES,
) -> str:
    """Lexicographically first subject with enough 001-012 hours, excluding discovery.

    Reads only ``*_EEG.hea`` filenames. Does not open CPC/outcome tables.
    """
    hours: dict[str, set[int]] = defaultdict(set)
    root = Path(received_training)
    if not root.exists():
        raise FileNotFoundError(f"received training tree missing: {root}")
    for path in root.rglob("*_EEG.hea"):
        parts = path.name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        hour = int(parts[1])
        if HOUR_LO <= hour <= HOUR_HI:
            hours[parts[0]].add(hour)
    candidates = [
        subject
        for subject in sorted(hours)
        if subject != exclude and len(hours[subject]) >= int(min_hours)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no CPC-blind replication subject under {root} "
            f"(exclude={exclude}, min_hours={min_hours})"
        )
    return str(candidates[0])


REPLICATION_SUBJECT = "0286"
FROZEN_SCORER = {
    "threshold": float(OOS_THRESHOLD),
    "ridge_alpha": float(RIDGE_ALPHA),
    "min_train_m1": int(M1_N_PARAMS),
    "n_params_m0": int(M0_N_PARAMS),
    "n_params_m1": int(M1_N_PARAMS),
    "grid_step_sec": float(TYPE_C_STEP_SEC),
    "epoch_sec": float(EPOCH_SEC["primary"]),
    "history_strong_p_r": float(HISTORY_STRONG_P_R),
    "hop_rule": "all_adjacent_hops",
}


def recording_subject_id(name: str) -> str:
    token = Path(str(name)).name.split("_")[0]
    return token.removeprefix("sub-")


def adjacent_hops_canonical(starts: np.ndarray, *, step_sec: float = TYPE_C_STEP_SEC) -> Dict[str, Any]:
    finite = np.asarray(starts, dtype=float)
    finite = finite[np.isfinite(finite)]
    hops = np.diff(finite)
    n_bad = int(np.sum(~np.isclose(hops, float(step_sec), atol=0.05))) if hops.size else 0
    return {
        "n_hops": int(hops.size),
        "n_noncanonical_hops": n_bad,
        "median_hop_sec": float(np.median(hops)) if hops.size else float("nan"),
        "canonical": bool(hops.size == 0 or n_bad == 0),
    }


def require_frozen_subject(frame: pd.DataFrame, subject: str) -> Optional[str]:
    if "file" not in frame.columns:
        return "feature table has no file column; cannot enforce subject freeze"
    tokens = sorted({recording_subject_id(name) for name in frame["file"].astype(str)})
    if tokens != [str(subject)]:
        return f"feature table subjects {tokens} != freeze {subject}"
    return None


def _load_features(features_dir: Path) -> Optional[pd.DataFrame]:
    parquet = Path(features_dir) / "features.parquet"
    csv = Path(features_dir) / "features.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def score_type_c_history(
    frame: pd.DataFrame,
    *,
    rng_seed: int,
    features_dir: Path,
    overlay: Path,
) -> Dict[str, Any]:
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
                    "p_r": float("nan"),
                    "rel_mse_median": float("nan"),
                }
            )
            continue
        hop_audit = None
        part = part.copy()
        if "t_start" in part.columns:
            part = part.sort_values(
                ["t_start", "epoch_id"] if "epoch_id" in part.columns else ["t_start"]
            )
            hop_audit = adjacent_hops_canonical(np.asarray(part["t_start"], dtype=float))
            if not hop_audit["canonical"]:
                recs.append(
                    {
                        "recording": rec_name,
                        "status": "GRID_INVALID",
                        "p_r": float("nan"),
                        "rel_mse_median": float("nan"),
                        "hop_audit": hop_audit,
                    }
                )
                continue
        z = part.loc[:, list(columns)].to_numpy(dtype=np.float32)
        rec_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        row = qualify_recording_history(z, columns=columns, rng=rec_rng)
        row["recording"] = rec_name
        if hop_audit is not None:
            row["hop_audit"] = hop_audit
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
                    "rel_mse_median": stats.get(
                        "rel_mse_median", rec.get("rel_mse_median", float("nan"))
                    ),
                }
            )
        coverages[name] = coverage_summary(model_rows)
    return {
        "status": "SCORED",
        "panel": PRIMARY_PANEL,
        "columns": list(columns),
        "missing_columns": missing,
        "n_recordings": len(recs),
        "recordings": recs,
        "coverage": coverages,
        "decision": history_decision(coverages),
        "frozen_scorer": dict(FROZEN_SCORER),
        "features_dir": str(features_dir),
        "overlay": str(overlay),
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def replication_decision(coverages: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    hist = history_decision(coverages)
    delta_p = hist["delta_p_r_vs_m0"]
    replicate_pass = bool(
        hist["beats_controls"] and np.isfinite(delta_p) and float(delta_p) > 0.0
    )
    if replicate_pass and hist["branch"] == "STRONG_HISTORY":
        replication_branch = "REPLICATED_STRONG"
    elif replicate_pass:
        replication_branch = "REPLICATED_MODEST"
    else:
        replication_branch = "NOT_REPLICATED"
    return {
        **hist,
        "replicate_pass": replicate_pass,
        "replication_branch": replication_branch,
        "discovery_subject": DISCOVERY_SUBJECT,
        "replication_subject": REPLICATION_SUBJECT,
        "alpha_computed": False,
        "cpc_inspected": False,
    }


def run_replication_history(*, rng_seed: int = 906) -> Dict[str, Any]:
    frame = _load_features(TYPE_C_SL010_FEATURES_DIR)
    blocked = {
        "features_dir": str(TYPE_C_SL010_FEATURES_DIR),
        "overlay": str(TYPE_C_SL010_OVERLAY),
        "subject": REPLICATION_SUBJECT,
        "frozen_scorer": dict(FROZEN_SCORER),
        "alpha_computed": False,
        "cpc_inspected": False,
    }
    if frame is None:
        return {
            "status": "BLOCKED",
            "reason": (
                "Type-C 2/2 SL-010 features missing; run the Type-C overlay on "
                f"{REPLICATION_SUBJECT} into a new processed root. Do not slice mnps_3d."
            ),
            **blocked,
        }
    purity = require_frozen_subject(frame, REPLICATION_SUBJECT)
    if purity is not None:
        return {"status": "BLOCKED", "reason": purity, **blocked}
    scored = score_type_c_history(
        frame,
        rng_seed=rng_seed,
        features_dir=TYPE_C_SL010_FEATURES_DIR,
        overlay=TYPE_C_SL010_OVERLAY,
    )
    scored["subject"] = REPLICATION_SUBJECT
    scored["replication"] = replication_decision(scored["coverage"])
    return scored


def sl010_report(*, rng_seed: int = 906) -> Dict[str, Any]:
    selected = None
    selection_error = None
    try:
        selected = select_replication_subject(RECEIVED_TRAINING)
    except FileNotFoundError as exc:
        selection_error = str(exc)
    selection_matches = bool(selected == REPLICATION_SUBJECT)
    if selection_matches:
        replication = run_replication_history(rng_seed=rng_seed)
    else:
        replication = {
            "status": "BLOCKED",
            "reason": (
                "CPC-blind selection does not match frozen replication subject "
                f"{REPLICATION_SUBJECT}; selected={selected}"
            ),
            "subject": REPLICATION_SUBJECT,
            "selected_subject": selected,
            "frozen_scorer": dict(FROZEN_SCORER),
            "alpha_computed": False,
            "cpc_inspected": False,
        }
    return {
        "gate": "TYPE_C_HISTORY_REPLICATION",
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "not_alpha_rescue": True,
        "not_operator": True,
        "not_mnps_history_gain": True,
        "cpc_blind": True,
        "alpha_outside_acceptance": True,
        "discovery_subject": DISCOVERY_SUBJECT,
        "replication_subject": REPLICATION_SUBJECT,
        "selected_subject": selected,
        "selection_matches_freeze": selection_matches,
        "selection_error": selection_error,
        "frozen_scorer": dict(FROZEN_SCORER),
        "c0": run_c0_history(rng_seed=rng_seed),
        "discovery_lock": run_icare_history(rng_seed=rng_seed),
        "replication": replication,
        "discovery_features_dir": str(TYPE_C_FEATURES_DIR),
        "replication_features_dir": str(TYPE_C_SL010_FEATURES_DIR),
        "discovery_overlay": str(TYPE_C_OVERLAY),
        "replication_overlay": str(TYPE_C_SL010_OVERLAY),
        "next_if_replicated": (
            "Type-C short-history gain is no longer 0284-only; still not alpha, still CPC-blind"
        ),
        "next_if_not": (
            "treat SL-006 as single-subject; do not hunt a third subject; do not open CPC"
        ),
    }


def main() -> None:
    import json

    report = sl010_report()
    print(
        f"selection freeze={report['replication_subject']} "
        f"selected={report['selected_subject']} "
        f"match={report['selection_matches_freeze']}"
    )
    lock = report["discovery_lock"]
    if lock.get("status") == "SCORED":
        dec = lock["decision"]
        print(
            f"0284 lock branch={dec['branch']} "
            f"d_p={dec['delta_p_r_vs_m0']:.3f} beats={dec['beats_controls']}"
        )
    else:
        print(f"0284 lock status={lock.get('status')}")
    icare = report["replication"]
    print(f"0286 status={icare['status']}")
    if icare.get("status") == "SCORED":
        dec = icare["replication"]
        for name, cov in icare["coverage"].items():
            print(
                f"  {name:10s} median_p_r={cov['median_p_r']:.3f} "
                f"P>0.5={cov['p_recordings_p_r_gt_half']:.3f} "
                f"rel={cov['recording_rel_mse_median']:.3f} closure={cov['closure_pass']}"
            )
        print(
            f"  replication_branch={dec['replication_branch']} "
            f"replicate_pass={dec['replicate_pass']} "
            f"d_rel={dec['delta_rel_mse_vs_m0']:.3f} d_p={dec['delta_p_r_vs_m0']:.3f}"
        )
    else:
        print(f"  {icare.get('reason')}")
    out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl010_20260918"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sl010_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'sl010_report.json'}")


if __name__ == "__main__":
    main()

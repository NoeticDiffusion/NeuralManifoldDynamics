"""Family B SL-004 / Type-C-0: 2/2 truth-known, feature audit, 0284 status.

Type-C is a new measurement family, not MNPS-v2-but-faster. Spectral
abscissa is reported but is not an acceptance criterion. Empirical 0284
OOS is blocked until a dedicated 2 s / 2 s ingest exists. Do not slice
existing 8 s mnps_3d. This module is test-only.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

TESTS = Path(__file__).resolve().parent
_SRC = TESTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import numpy as np

from family_b_sl002 import (
    DT_FINE,
    ICARE_0284_RUN,
    OOS_THRESHOLD,
    PROCESS_SIGMA,
    REGIMES,
    _json_ready,
    downsample_to_grid,
    qualify_grid_series,
    simulate_linear_sde,
)
from family_b_sl003 import EPOCH_SEC, qualify_series_epochs

DT_2_SEC = 2.0
TRUTH_DURATION_SEC = 30 * 60.0
TYPE_C_WINDOW_SEC = 2.0
TYPE_C_STEP_SEC = 2.0
ICARE_SFREQ_HZ = 100.0
ICARE_BANDPASS_HZ = (0.5, 40.0)
MNPS_WINDOW_SEC = 8.0
MNPS_STEP_SEC = 4.0

VERDICT_CANDIDATE = "CANDIDATE"
VERDICT_REDEFINE = "REDEFINE"
VERDICT_DROP = "DROP"


def n_cycles(freq_hz: float, window_sec: float = TYPE_C_WINDOW_SEC) -> float:
    return float(freq_hz) * float(window_sec)


def _band_row(
    name: str,
    lo_hz: float,
    hi_hz: float,
    *,
    verdict: str,
    reason: str,
    mnps_9d_axes: List[str],
) -> Dict[str, Any]:
    return {
        "feature": name,
        "kind": "spectral_bandpower",
        "band_hz": [float(lo_hz), float(hi_hz)],
        "n_cycles_lo": n_cycles(lo_hz),
        "n_cycles_hi": n_cycles(hi_hz),
        "freq_bin_hz": 1.0 / TYPE_C_WINDOW_SEC,
        "icare_bandpass_hz": list(ICARE_BANDPASS_HZ),
        "mnps_9d_axes": mnps_9d_axes,
        "verdict": verdict,
        "reason": reason,
        "new_family": True,
    }


def feature_support_audit() -> Dict[str, Any]:
    """C1: can each current MNPS-9D feature be measured at 2 s?

    Verdicts are physics/support judgments for a *new* Type-C family.
    CANDIDATE does not mean the 8 s MNPS label may be reused.
    """
    rows = [
        _band_row(
            "eeg_delta",
            1.0,
            4.0,
            verdict=VERDICT_REDEFINE,
            reason=(
                "2–8 cycles in 2 s; Rayleigh 0.5 Hz; I-CARE high-pass 0.5 Hz sits "
                "on the low-frequency edge. Not a well-posed 8 s delta morphology."
            ),
            mnps_9d_axes=["m_a"],
        ),
        _band_row(
            "eeg_theta",
            4.0,
            8.0,
            verdict=VERDICT_CANDIDATE,
            reason="8–16 cycles in 2 s. Empirically re-estimate variance; new axis, not MNPS m_a.",
            mnps_9d_axes=["m_a"],
        ),
        _band_row(
            "eeg_alpha",
            8.0,
            12.0,
            verdict=VERDICT_CANDIDATE,
            reason="16–24 cycles in 2 s. Fast spectral morphology candidate.",
            mnps_9d_axes=["m_e"],
        ),
        {
            "feature": "eeg_beta",
            "kind": "spectral_bandpower",
            "band_hz": [13.0, 30.0],
            "n_cycles_lo": n_cycles(13.0),
            "n_cycles_hi": n_cycles(30.0),
            "freq_bin_hz": 1.0 / TYPE_C_WINDOW_SEC,
            "icare_bandpass_hz": list(ICARE_BANDPASS_HZ),
            "mnps_9d_axes": ["m_o (via eeg_beta_alpha)"],
            "verdict": VERDICT_CANDIDATE,
            "reason": (
                "26–60 cycles in 2 s; 13–30 Hz lies inside I-CARE 0.5–40 Hz. "
                "Empirically re-estimate variance; new Type-C axis, not MNPS m_o."
            ),
            "new_family": True,
        },
        _band_row(
            "eeg_gamma",
            30.0,
            45.0,
            verdict=VERDICT_REDEFINE,
            reason="Configured 30–45 Hz vs I-CARE bandpass 0.5–40 Hz. Not a 2 s gamma contract.",
            mnps_9d_axes=["d_n"],
        ),
        {
            "feature": "eeg_alpha_theta",
            "kind": "bandpower_ratio",
            "band_hz": None,
            "parents": ["eeg_alpha", "eeg_theta"],
            "mnps_9d_axes": ["d_s"],
            "verdict": VERDICT_CANDIDATE,
            "reason": "Inherits parent support. Recompute on Type-C PSD; do not copy 8 s ratios.",
            "new_family": True,
        },
        {
            "feature": "eeg_beta_alpha",
            "kind": "bandpower_ratio",
            "parents": ["eeg_beta", "eeg_alpha"],
            "mnps_9d_axes": ["m_o"],
            "verdict": VERDICT_CANDIDATE,
            "reason": "Inherits parent support after beta high-cut is redefined.",
            "new_family": True,
        },
        {
            "feature": "eeg_permutation_entropy",
            "kind": "ordinal_complexity",
            "order": 5,
            "delay": 1,
            "n_samples_2s": int(ICARE_SFREQ_HZ * TYPE_C_WINDOW_SEC),
            "n_samples_8s": int(ICARE_SFREQ_HZ * MNPS_WINDOW_SEC),
            "mnps_9d_axes": ["e_e"],
            "verdict": VERDICT_CANDIDATE,
            "reason": (
                "Computable at 200 samples, but 8 s used 800. Short-window variance "
                "must be measured before it is a Type-C axis."
            ),
            "new_family": True,
        },
        {
            "feature": "eeg_sample_entropy",
            "kind": "alias",
            "alias_of": "eeg_permutation_entropy",
            "mnps_9d_axes": ["legacy v1 d"],
            "verdict": VERDICT_DROP,
            "reason": "Not a separate estimator; current code aliases permutation entropy.",
            "new_family": True,
        },
        {
            "feature": "eeg_hjorth_mobility",
            "kind": "derivative_complexity",
            "mnps_9d_axes": ["d_l"],
            "verdict": VERDICT_CANDIDATE,
            "reason": "Defined on ≥4 samples; 2 s is noise-sensitive. Qualify variance, new axis.",
            "new_family": True,
        },
        {
            "feature": "eeg_hjorth_complexity",
            "kind": "derivative_complexity",
            "mnps_9d_axes": ["e_s"],
            "verdict": VERDICT_CANDIDATE,
            "reason": "Same as mobility: computable, not automatically an 8 s e_s transplant.",
            "new_family": True,
        },
        {
            "feature": "embodied_arousal_proxy",
            "kind": "embodiment",
            "preferred": "ecg_rmssd",
            "hrv_superwindow_s": 60.0,
            "mnps_9d_axes": ["e_m"],
            "verdict": VERDICT_DROP,
            "reason": "HRV contract is 60 s / ≥20 NN. Invalid as a 2 s Type-C feature.",
            "new_family": True,
        },
    ]
    counts = {
        VERDICT_CANDIDATE: sum(row["verdict"] == VERDICT_CANDIDATE for row in rows),
        VERDICT_REDEFINE: sum(row["verdict"] == VERDICT_REDEFINE for row in rows),
        VERDICT_DROP: sum(row["verdict"] == VERDICT_DROP for row in rows),
    }
    return {
        "gate": "C1",
        "window_sec": TYPE_C_WINDOW_SEC,
        "step_sec": TYPE_C_STEP_SEC,
        "not_mnps_v2_but_faster": True,
        "automatic_8s_carry_forbidden": True,
        "icare_sfreq_hz": ICARE_SFREQ_HZ,
        "n_features": len(rows),
        "verdict_counts": counts,
        "rows": rows,
        "safe_claim": (
            "A Type-C chart must be a smaller, temporally well-posed feature set, "
            "then a new projection. alpha_C is not alpha_MNPS8s."
        ),
    }


def run_c0_truth_known(*, rng_seed: int = 804) -> Dict[str, Any]:
    """C0: 2/2 truth-known ẋ→J and F→log(F)/Δ. Acceptance is generator OOS, not α."""
    rng = np.random.default_rng(rng_seed)
    n_fine = int(round(TRUTH_DURATION_SEC / DT_FINE))
    rows = []
    for name, j_true in REGIMES.items():
        x_fine = simulate_linear_sde(
            j_true, dt_fine=DT_FINE, n_fine=n_fine, sigma=PROCESS_SIGMA, rng=rng
        )
        x2 = downsample_to_grid(x_fine, dt_fine=DT_FINE, grid_dt=DT_2_SEC)
        series = qualify_grid_series(x2, dt=DT_2_SEC, j_true=j_true)
        epochs = qualify_series_epochs(
            x2,
            dt=DT_2_SEC,
            epoch_sec=EPOCH_SEC["primary"],
            j_true=j_true,
            family_b_scope=False,
        )
        oracle_ok = bool(series["j_oracle_oos_pass"])
        f_series_ok = bool(series["a_oos_pass"])
        f_epoch_ok = int(epochs["n_generator_qualified"]) >= 1
        sg_series_ok = bool(series["j_sg_global_oos_pass"])
        generator_identified = bool(f_series_ok and f_epoch_ok)
        c0_pass = bool(oracle_ok and generator_identified)
        rows.append(
            {
                "regime": name,
                "grid": "2/2",
                "dt_sec": DT_2_SEC,
                "n_grid": int(x2.shape[0]),
                "series": series,
                "epoch_5min": {
                    "n_epochs": epochs["n_epochs"],
                    "n_epoch_samples": epochs["n_epoch_samples"],
                    "n_generator_qualified": epochs["n_generator_qualified"],
                    "n_family_b_alpha_licensed": epochs["n_family_b_alpha_licensed"],
                    "n_sg_residual_pass": epochs["n_sg_residual_pass"],
                    "n_transition_alpha_match": epochs["n_transition_alpha_match"],
                    "transition_status_counts": epochs["transition_status_counts"],
                    "sg_status_counts": epochs["sg_status_counts"],
                },
                "oracle_xdot_identified": oracle_ok,
                "sg_xdot_identified": sg_series_ok,
                "transition_series_identified": f_series_ok,
                "transition_epoch_identified": f_epoch_ok,
                "generator_identified": generator_identified,
                "c0_pass": c0_pass,
                "alpha_is_acceptance": False,
            }
        )
    stable = next(row for row in rows if row["regime"] == "stable")
    return {
        "gate": "C0",
        "threshold": OOS_THRESHOLD,
        "rng_seed": int(rng_seed),
        "grid": "2/2",
        "window_sec": TYPE_C_WINDOW_SEC,
        "step_sec": TYPE_C_STEP_SEC,
        "duration_sec": TRUTH_DURATION_SEC,
        "acceptance": (
            "oracle_xdot_J AND transition F OOS on the series AND "
            "≥1 five-min epoch generator-qualified; alpha is not acceptance"
        ),
        "stable_required": True,
        "stable_c0_pass": bool(stable["c0_pass"]),
        "rows": rows,
    }


def run_c2_icare_status() -> Dict[str, Any]:
    """C2 cannot run on 8 s MNPS. Do not subsample existing dumps."""
    dump_exists = ICARE_0284_RUN.exists()
    return {
        "gate": "C2",
        "status": "BLOCKED",
        "reason": (
            "No Type-C 2 s/2 s ingest exists. Existing 0284 HDF5 is 8 s/4 s MNPS "
            "and is not a Type-C chart."
        ),
        "mnps_8s_dump_present": bool(dump_exists),
        "mnps_8s_dump": str(ICARE_0284_RUN) if dump_exists else None,
        "forbidden": [
            "slice mnps_3d to pretend 2 s hops",
            "reuse 8 s feature caches without --force-features on a Type-C overlay",
            "treat alpha_C as alpha_MNPS8s",
            "open CPC labels before generator OOS",
        ],
        "required_before_oos": {
            "epoching.length_s": TYPE_C_WINDOW_SEC,
            "epoching.step_s": TYPE_C_STEP_SEC,
            "mnps.window_sec": TYPE_C_WINDOW_SEC,
            "mnps.overlap": 0.0,
            "feature_set": "Type-C C1 survivors only; new projection",
            "overlay_name_must_contain": "dynamical_families",
            "separate_output_root": True,
            "force_features": True,
            "subject": "0284",
            "cpc_blind": True,
            "acceptance": "relMSE_OOS distribution and P(licensed epochs|recording); not alpha",
        },
        "n_family_b_alpha_licensed": 0,
    }


def sl004_report(*, rng_seed: int = 804) -> Dict[str, Any]:
    c0 = run_c0_truth_known(rng_seed=rng_seed)
    c1 = feature_support_audit()
    c2 = run_c2_icare_status()
    return {
        "family_b_8s": "CLOSED_METHOD_LIMITED",
        "type_c_is_new_family": True,
        "alpha_outside_acceptance": True,
        "threshold": OOS_THRESHOLD,
        "c0": c0,
        "c1": c1,
        "c2": c2,
    }


def main() -> None:
    import json

    report = sl004_report()
    c0 = report["c0"]
    print("C0 truth-known 2/2 (acceptance = generator OOS, not alpha)")
    for row in c0["rows"]:
        s = row["series"]
        e = row["epoch_5min"]
        print(
            f"  {row['regime']:10s} c0_pass={row['c0_pass']} "
            f"oracle={s['rel_mse_j_oracle_xdot']:.3f} "
            f"SG={s['rel_mse_j_sg_global']:.3f} "
            f"F_series={s['rel_mse_a_next']:.3f} "
            f"F_epochs={e['n_generator_qualified']}/{e['n_epochs']} "
            f"SG_epochs={e['n_sg_residual_pass']}/{e['n_epochs']}"
        )
    print(f"  stable_required_pass={c0['stable_c0_pass']}")
    c1 = report["c1"]
    print(
        f"C1 feature audit n={c1['n_features']} "
        f"counts={c1['verdict_counts']} carry_forbidden={c1['automatic_8s_carry_forbidden']}"
    )
    c2 = report["c2"]
    print(f"C2 {c2['status']}: {c2['reason']}")
    out = Path(__file__).resolve().parents[2] / ".work" / "family_b_sl004_20260914"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sl004_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out / 'sl004_report.json'}")


if __name__ == "__main__":
    main()

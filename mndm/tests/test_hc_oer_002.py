"""HC-OER-002 tests: history increment vs z-only, not alpha, not CPC."""

from pathlib import Path
import json
import sys

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from hc_oer_001 import isotropic_series, score_epoch_hc_oer  # noqa: E402
from hc_oer_002 import (  # noqa: E402
    DELTA_FLOOR,
    adaptive_threshold,
    epoch_delta_gamma,
    hc_oer_002_report,
    increment_decision,
    qualify_recording_delta,
    run_c0_delta,
)


def test_estimator_is_frozen_001_shell():
    source_001 = Path(__file__).resolve().parent.joinpath("hc_oer_001.py").read_text(encoding="utf-8")
    source_002 = Path(__file__).resolve().parent.joinpath("hc_oer_002.py").read_text(encoding="utf-8")
    assert "SHELL_Q_LO" in source_001
    assert "SHELL_Q_LO =" not in source_002
    assert "from scipy.linalg import logm" not in source_002
    assert "cmd_summarize" not in source_002
    assert "cpc_inspected" in source_002


def test_epoch_delta_is_hc_minus_z_only():
    rng = np.random.default_rng(2)
    z = isotropic_series(0.8, n=160, dim=6, sigma=0.0, rng=rng)
    columns = tuple(f"z{i}" for i in range(6))
    row = score_epoch_hc_oer(z[:150], columns=columns, rng=rng)
    delta = epoch_delta_gamma(row)
    expected = row["modes"]["hc"]["gamma_median"] - row["modes"]["z_only"]["gamma_median"]
    assert abs(delta - expected) < 1e-12


def test_first_order_has_no_history_increment():
    rng = np.random.default_rng(1002)
    z = isotropic_series(0.8, n=150, dim=6, sigma=0.0, rng=rng)
    columns = tuple(f"z{i}" for i in range(6))
    rec = qualify_recording_delta(z, columns=columns, rng=rng)
    assert abs(rec["delta_gamma_median"]) < DELTA_FLOOR


def test_c0_ar2_increment_stays_below_predeclared_floor():
    """SL-006 AR(2) is visible as prediction, not as a 0.02 expansion increment."""
    report = run_c0_delta(rng_seed=1002)
    d_first = abs(report["rows"]["first_order_isotropic"]["delta_gamma_median"])
    d_ar2 = abs(report["rows"]["ar2"]["delta_gamma_median"])
    assert report["first_order_ok"] is True
    assert report["white_ok"] is True
    assert report["ar2_ok"] is False
    assert report["c0_increment_identifiable"] is False
    assert d_ar2 > d_first
    assert d_ar2 < DELTA_FLOOR


def test_decision_branches_use_c0_threshold_not_cpc():
    def c0(*, first=True, ar2=True, white=True, thresh=0.02):
        return {
            "first_order_ok": first,
            "ar2_ok": ar2,
            "white_ok": white,
            "threshold": thresh,
        }

    rec = {"delta_gamma_median": 0.05, "coverage_median": 0.9}
    assert increment_decision(c0=c0(), recordings=[], blocked=True)["branch"] == "BLOCKED"
    assert increment_decision(c0=c0(first=False), recordings=[rec] * 12, blocked=False)["branch"] == "C0_FAIL"
    assert increment_decision(c0=c0(ar2=False), recordings=[rec] * 12, blocked=False)["branch"] == "METHOD_LIMITED"
    none = increment_decision(
        c0=c0(),
        recordings=[{"delta_gamma_median": 0.001, "coverage_median": 0.9}] * 12,
        blocked=False,
    )
    assert none["branch"] == "NO_INCREMENT"
    mixed = increment_decision(
        c0=c0(),
        recordings=[{"delta_gamma_median": 0.05}] * 7 + [{"delta_gamma_median": -0.05}] * 5,
        blocked=False,
    )
    assert mixed["branch"] == "MIXED_SIGN"
    ok = increment_decision(c0=c0(), recordings=[rec] * 12, blocked=False)
    assert ok["branch"] == "HISTORY_INCREMENT"
    assert ok["cpc_inspected"] is False
    assert ok["alpha_computed"] is False


def test_adaptive_threshold_never_below_floor():
    assert adaptive_threshold(0.0) == DELTA_FLOOR
    assert adaptive_threshold(0.005) == DELTA_FLOOR
    assert adaptive_threshold(0.03) == 0.06


def test_c0_report_json_ready():
    report = run_c0_delta(rng_seed=1002)
    payload = _json_ready({"c0": report, "cpc_inspected": False})
    json.dumps(payload, allow_nan=False)


def test_full_report_is_method_limited_even_if_icare_looks_positive(monkeypatch):
    monkeypatch.setattr(
        "hc_oer_002.run_icare_delta",
        lambda rng_seed=1002: {
            "status": "SCORED",
            "recordings": [{"delta_gamma_median": 0.05, "coverage_median": 1.0}] * 12,
            "delta_gamma_median_across_recordings": 0.05,
            "alpha_computed": False,
            "cpc_inspected": False,
        },
    )
    report = hc_oer_002_report(rng_seed=1002)
    assert report["c0"]["ar2_ok"] is False
    assert report["decision"]["branch"] == "METHOD_LIMITED"
    assert report["decision"]["cpc_inspected"] is False

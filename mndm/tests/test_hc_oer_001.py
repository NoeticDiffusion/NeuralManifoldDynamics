"""HC-OER-001 tests: truth-known sign recovery, not alpha, not CPC."""

from pathlib import Path
import json
import sys

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from family_b_sl005 import PANELS, PRIMARY_PANEL  # noqa: E402
from family_b_sl006 import consecutive_triples  # noqa: E402
from hc_oer_001 import (  # noqa: E402
    K_MIN,
    K_NEIGHBORS,
    MIN_INDEX_GAP,
    expected_isotropic_gamma,
    hc_oer_decision,
    isotropic_series,
    run_c0_hc_oer,
    score_epoch_hc_oer,
    _eligible_mask,
)


def _assert_no_none(value, path: str = "root") -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            _assert_no_none(inner, f"{path}.{key}")
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            _assert_no_none(inner, f"{path}[{i}]")


def test_frozen_settings_and_not_alpha_pipeline():
    assert K_NEIGHBORS == 20
    assert K_MIN == 8
    assert MIN_INDEX_GAP == 3
    assert PANELS[PRIMARY_PANEL][0] == "eeg_theta"
    source = Path(__file__).resolve().parent.joinpath("hc_oer_001.py").read_text(encoding="utf-8")
    assert "from scipy.linalg import logm" not in source
    assert "cmd_summarize" not in source
    assert "cpc_inspected" in source
    assert "not_spectral_abscissa" in source
    assert "SHELL_Q_LO" in Path(__file__).resolve().parent.joinpath("hc_oer_001.py").read_text(encoding="utf-8")


def test_index_gap_prevents_shared_triple_samples():
    n = 20
    i = 10
    eligible = _eligible_mask(n, i)
    lag, now, nxt = consecutive_triples(np.arange(n * 2, dtype=np.float32).reshape(n, 2))
    query_times = {i, i + 1, i + 2}
    for j in np.where(eligible)[0]:
        other = {int(j), int(j) + 1, int(j) + 2}
        assert query_times.isdisjoint(other)
    assert not eligible[i]
    assert not eligible[i + 1]
    assert not eligible[i + 2]
    assert eligible[i - 3]
    assert now.shape[0] == n - 2


def test_isotropic_contract_matches_log_rho_over_dt():
    rng = np.random.default_rng(1001)
    z = isotropic_series(0.8, n=160, dim=6, sigma=0.0, rng=rng)
    columns = tuple(f"z{i}" for i in range(6))
    row = score_epoch_hc_oer(z[:150], columns=columns, rng=rng)
    g = row["modes"]["hc"]["gamma_median"]
    theory = expected_isotropic_gamma(0.8)
    assert row["status"] == "SCORED"
    assert g < 0.0
    assert abs(g - theory) < 0.05
    assert row["modes"]["shuffle_next"]["gamma_median"] > g
    assert row["modes"]["reverse"]["gamma_median"] > 0.0


def test_c0_sign_recovery_and_white_near_zero():
    report = run_c0_hc_oer(rng_seed=1001)
    assert report["c0_pass"] is True
    assert report["rows"]["contract_0p8"]["modes"]["hc"]["gamma_median"] < 0.0
    assert report["rows"]["expand_1p15"]["modes"]["hc"]["gamma_median"] > 0.0
    assert report["rows"]["contract_0p8_noise"]["modes"]["hc"]["gamma_median"] < 0.0
    assert abs(report["rows"]["white"]["modes"]["hc"]["gamma_median"]) < 0.05
    assert report["alpha_computed"] is False
    assert report["cpc_inspected"] is False


def test_decision_requires_coverage_and_c0():
    rec_ok = {"coverage_median": 0.8, "gamma_median": 0.01}
    rec_bad = {"coverage_median": 0.1, "gamma_median": 0.01}
    blocked = hc_oer_decision(c0_pass=True, recordings=[], blocked=True)
    assert blocked["branch"] == "BLOCKED"
    fail = hc_oer_decision(c0_pass=False, recordings=[rec_ok] * 12, blocked=False)
    assert fail["branch"] == "C0_FAIL"
    weak = hc_oer_decision(c0_pass=True, recordings=[rec_ok] * 7 + [rec_bad] * 5, blocked=False)
    assert weak["branch"] == "WEAK_COVERAGE"
    ok = hc_oer_decision(c0_pass=True, recordings=[rec_ok] * 8 + [rec_bad] * 4, blocked=False)
    assert ok["branch"] == "QUALIFIED"
    assert ok["cpc_inspected"] is False
    assert ok["alpha_computed"] is False


def test_report_json_ready_without_icare_extract(tmp_path, monkeypatch):
    import hc_oer_001 as mod

    monkeypatch.setattr(mod, "run_icare_hc_oer", lambda rng_seed=1001: {
        "status": "BLOCKED",
        "reason": "missing",
        "alpha_computed": False,
        "cpc_inspected": False,
    })
    report = mod.hc_oer_001_report(rng_seed=1001)
    assert report["decision"]["branch"] == "BLOCKED"
    payload = _json_ready(report)
    json.dumps(payload, allow_nan=False)
    _assert_no_none(payload)

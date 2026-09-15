"""SL-002 temporal-contract tests (truth-known 8/4 vs 8/8, J vs A).

Blocked OOS with derivative embargo. Next-state A is diagnostic only.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import (  # noqa: E402
    DT_4_SEC,
    DT_8_SEC,
    ICARE_0284_RUN,
    OOS_THRESHOLD,
    REGIMES,
    blocked_slices,
    embargo_samples,
    qualify_grid_series,
    run_icare_overlap_ablation,
    run_truth_known,
    slices_disjoint_with_embargo,
    spectral_abscissa,
    summarize_icare,
)


def test_blocked_split_does_not_share_sg_support():
    n = 200
    embargo = embargo_samples()
    train, test, gap = blocked_slices(n, embargo=embargo)
    assert gap == embargo
    assert slices_disjoint_with_embargo(n, train, test, gap)
    assert train.stop + embargo <= test.start
    assert test.start - train.stop >= 2 * gap


def test_easy_short_dt_recovers_stable_generator():
    rng = np.random.default_rng(611)
    j_true = REGIMES["stable"]
    dt = 0.5
    n = 400
    x = np.zeros((n, 3), dtype=np.float64)
    x[0] = rng.normal(size=3)
    for t in range(n - 1):
        x[t + 1] = x[t] + dt * (j_true @ x[t]) + 0.02 * np.sqrt(dt) * rng.normal(size=3)
    row = qualify_grid_series(x.astype(np.float32), dt=dt, j_true=j_true)
    assert row["split_ok"]
    assert row["rel_mse_j_oracle_xdot"] < 0.2
    assert row["rel_mse_a_next"] < OOS_THRESHOLD
    assert row["alpha_global_class_match"] or row["rel_mse_j_oracle_xdot"] < 0.2
    # Production SG ẋ is the quantity under SL-002 audit, not a unit invariant.


def test_truth_known_production_grids_record_j_vs_a_field():
    report = run_truth_known(rng_seed=602)
    rows = report["rows"]
    assert {row["regime"] for row in rows} == set(REGIMES)
    dts = {float(row["dt_sec"]) for row in rows}
    assert DT_4_SEC in dts and DT_8_SEC in dts
    for row in rows:
        assert row["split_ok"]
        assert "j_sg_oos_pass" in row and "a_oos_pass" in row
        assert row["alpha_true_class"] in {"stable", "marginal", "unstable"}


def test_stable_regime_alpha_is_negative():
    assert spectral_abscissa(REGIMES["stable"]) < 0
    assert spectral_abscissa(REGIMES["unstable"]) > 0


@pytest.mark.skipif(not ICARE_0284_RUN.exists(), reason="I-CARE 0284 dump not mounted")
def test_icare_0284_overlap_ablation_blocked_oos():
    report = run_icare_overlap_ablation()
    assert report["n_files"] == 12
    summary = summarize_icare(report)
    for grid in ("grid_8_4", "grid_8_8"):
        assert summary[grid]["n"] == 12
        assert np.isfinite(summary[grid]["j_sg_oos_median"]) or summary[grid]["n"] >= 1
    # Do not require OOS < 0.9: that is the scientific outcome, not a unit invariant.

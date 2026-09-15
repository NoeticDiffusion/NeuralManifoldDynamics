"""SL-006 Type-C short-history tests: M1 vs M0, duplicate, shuffle."""

from pathlib import Path
import json
import sys

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from family_b_sl005 import PANELS, PRIMARY_PANEL  # noqa: E402
from family_b_sl006 import (  # noqa: E402
    M0_N_PARAMS,
    M1_N_PARAMS,
    N_TARGET,
    _predictors,
    _shuffle_lag_block,
    consecutive_triples,
    history_decision,
    history_split,
    qualify_history_epoch,
    run_c0_history,
    sl006_report,
    split_sample_times,
)


def _assert_no_none(value, path: str = "root") -> None:
    if value is None:
        raise AssertionError(f"unexpected None at {path}")
    if isinstance(value, dict):
        for key, inner in value.items():
            _assert_no_none(inner, f"{path}.{key}")
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            _assert_no_none(inner, f"{path}[{i}]")


def test_primary_panel_frozen_and_m1_is_six_d_target():
    assert PANELS[PRIMARY_PANEL] == (
        "eeg_theta",
        "eeg_alpha",
        "eeg_beta",
        "eeg_permutation_entropy",
        "eeg_hjorth_mobility",
        "eeg_hjorth_complexity",
    )
    assert "eeg_delta" not in PANELS[PRIMARY_PANEL]
    assert "eeg_gamma" not in PANELS[PRIMARY_PANEL]
    assert M0_N_PARAMS == 42
    assert M1_N_PARAMS == 78


def test_shuffle_permutes_only_the_lag_half_within_a_block():
    rng = np.random.default_rng(1)
    now = np.arange(24, dtype=np.float32).reshape(4, 6)
    lag = now + 100.0
    pred = np.hstack([now, lag])
    out = _shuffle_lag_block(pred, rng)
    np.testing.assert_array_equal(out[:, :6], now)
    # Rows of the lag block are a permutation, not a new mix of coordinates.
    assert sorted(map(tuple, out[:, 6:])) == sorted(map(tuple, lag))


def test_duplicate_copies_now_and_m1_keeps_true_lag():
    lag = np.arange(18, dtype=np.float32).reshape(3, 6)
    now = lag + 50.0
    preds = _predictors(lag, now)
    assert preds["m0"].shape[1] == 6
    assert preds["m1"].shape[1] == 12
    assert preds["duplicate"].shape[1] == 12
    np.testing.assert_array_equal(preds["duplicate"][:, :6], now)
    np.testing.assert_array_equal(preds["duplicate"][:, 6:], now)
    np.testing.assert_array_equal(preds["m1"][:, :6], now)
    np.testing.assert_array_equal(preds["m1"][:, 6:], lag)


def test_blocked_triples_share_no_timestamps():
    n_samples = 150
    n_triples = n_samples - 2
    train_sl, test_sl, gap = history_split(n_triples, N_TARGET)
    assert gap == 1
    assert train_sl.stop - (train_sl.start or 0) >= M1_N_PARAMS
    train_times = split_sample_times(train_sl)
    test_times = split_sample_times(test_sl)
    assert train_times.isdisjoint(test_times)
    last_train = max(train_times)
    first_test = min(test_times)
    assert first_test == last_train + 1


def test_models_share_one_split_and_embargo():
    rng = np.random.default_rng(6)
    z = rng.normal(size=(150, 6)).astype(np.float32)
    columns = tuple(f"z{i}" for i in range(6))
    row = qualify_history_epoch(z, columns=columns, rng=rng)
    assert row["status"] == "SCORED"
    assert row["embargo_pairs"] == 1
    assert row["n_train"] >= M1_N_PARAMS
    assert row["n_test"] >= 6
    assert row["models"]["m0"]["n_pred"] == 6
    assert row["models"]["m1"]["n_pred"] == 12
    assert row["models"]["duplicate"]["n_pred"] == 12
    assert row["models"]["shuffle"]["n_pred"] == 12


def test_shuffle_train_and_test_stay_in_their_own_blocks():
    rng = np.random.default_rng(11)
    n_triples = 148
    lag = rng.normal(size=(n_triples, 6)).astype(np.float32)
    now = rng.normal(size=(n_triples, 6)).astype(np.float32)
    train_sl, test_sl, _ = history_split(n_triples, 6)
    pred = np.hstack([now, lag])
    pred_tr = _shuffle_lag_block(pred[train_sl], rng)
    pred_te = _shuffle_lag_block(pred[test_sl], rng)
    train_lags = {tuple(row) for row in lag[train_sl]}
    test_lags = {tuple(row) for row in lag[test_sl]}
    assert {tuple(row) for row in pred_tr[:, 6:]} == train_lags
    assert {tuple(row) for row in pred_te[:, 6:]} == test_lags
    np.testing.assert_array_equal(pred_tr[:, :6], now[train_sl])
    np.testing.assert_array_equal(pred_te[:, :6], now[test_sl])


def test_triples_do_not_stitch_gaps():
    z = np.ones((8, 6), dtype=np.float32)
    z[3] = np.nan
    lag, now, nxt = consecutive_triples(z)
    # Valid t in {1,5,6} because t=2 and t=4 touch the gap at index 3.
    assert now.shape[0] == 3


def test_second_order_truth_known_history_beats_controls():
    report = run_c0_history(rng_seed=906)
    second = report["rows"]["second_order"]
    m0 = second["models"]["m0"]["rel_mse_median"]
    m1 = second["models"]["m1"]["rel_mse_median"]
    dup = second["models"]["duplicate"]["rel_mse_median"]
    shuf = second["models"]["shuffle"]["rel_mse_median"]
    assert m1 < m0
    assert m1 < dup
    assert m1 < shuf
    assert second["models"]["m1"]["p_r"] > second["models"]["m0"]["p_r"]


def test_first_order_does_not_need_history():
    report = run_c0_history(rng_seed=906)
    first = report["rows"]["first_order"]
    assert first["models"]["m0"]["p_r"] >= 0.5
    assert first["models"]["m0"]["rel_mse_median"] < 0.9
    assert first["delta_p_r"] < 0.3


def test_history_decision_requires_all_controls():
    def cov(p, rel, closure=False):
        return {
            "median_p_r": p,
            "recording_rel_mse_median": rel,
            "closure_pass": closure,
            "p_recordings_p_r_gt_half": 1.0 if p > 0.5 else 0.0,
        }

    weak = history_decision(
        {
            "m0": cov(0.21, 0.94),
            "m1": cov(0.27, 0.92),
            "duplicate": cov(0.22, 0.93),
            "shuffle": cov(0.20, 0.94),
        }
    )
    assert weak["strong_history_pass"] is False
    strong = history_decision(
        {
            "m0": cov(0.21, 0.94),
            "m1": cov(0.70, 0.55, closure=True),
            "duplicate": cov(0.25, 0.90),
            "shuffle": cov(0.22, 0.91),
        }
    )
    assert strong["strong_history_pass"] is True
    assert strong["branch"] == "STRONG_HISTORY"


def test_sl006_report_is_cpc_blind_and_finite():
    report = sl006_report(rng_seed=906)
    assert report["alpha_outside_acceptance"] is True
    assert report["cpc_blind"] is True
    assert report["not_mnps"] is True
    icare = report["icare"]
    assert icare["alpha_computed"] is False
    assert icare["cpc_inspected"] is False
    ready = _json_ready(report)
    json.dumps(ready, allow_nan=False)
    _assert_no_none(ready)
    if icare.get("status") == "SCORED":
        assert icare["panel"] == PRIMARY_PANEL
        assert icare["n_recordings"] == 12
        assert "alpha" not in icare["decision"]
        assert icare["columns"] == list(PANELS[PRIMARY_PANEL])

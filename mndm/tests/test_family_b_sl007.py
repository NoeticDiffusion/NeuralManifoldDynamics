"""SL-007 Type-C augmented-operator tests: F_H multi-step, not log(F)."""

from pathlib import Path
import json
import sys

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from family_b_sl005 import PANELS, PRIMARY_PANEL  # noqa: E402
from family_b_sl006 import N_TARGET, history_split, split_sample_times  # noqa: E402
from family_b_sl007 import (  # noqa: E402
    EQUIV_ATOL,
    HORIZONS,
    OPERATOR_NAME,
    affine_predict_z,
    affine_to_companion,
    companion_predict_z,
    horizon_starts,
    operator_decision,
    qualify_operator_epoch,
    run_c0_operator,
    sl007_report,
    spectral_radius,
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


def test_primary_panel_frozen_and_operator_is_not_alpha():
    assert PANELS[PRIMARY_PANEL] == (
        "eeg_theta",
        "eeg_alpha",
        "eeg_beta",
        "eeg_permutation_entropy",
        "eeg_hjorth_mobility",
        "eeg_hjorth_complexity",
    )
    assert OPERATOR_NAME == "history_augmented_type_c_operator"
    assert HORIZONS == (1, 2, 4)


def test_companion_one_step_matches_m1_affine():
    rng = np.random.default_rng(7)
    mapped = rng.normal(size=(6, 12)).astype(np.float32) * 0.1
    intercept = rng.normal(size=6).astype(np.float32)
    x_mean = rng.normal(size=12).astype(np.float32)
    now = rng.normal(size=(20, 6)).astype(np.float32)
    lag = rng.normal(size=(20, 6)).astype(np.float32)
    y = np.hstack([now, lag])
    pred_m1 = affine_predict_z(mapped, intercept, x_mean, y)
    f_h, c_aug = affine_to_companion(mapped, intercept, x_mean)
    pred_fh = companion_predict_z(f_h, c_aug, now, lag, 1)
    np.testing.assert_allclose(pred_fh, pred_m1, atol=1e-5, rtol=1e-5)
    assert f_h.shape == (12, 12)
    np.testing.assert_array_equal(f_h[6:, :6], np.eye(6))
    np.testing.assert_array_equal(f_h[6:, 6:], np.zeros((6, 6)))


def test_companion_second_block_copies_current_state():
    mapped = np.zeros((6, 12), dtype=np.float64)
    intercept = np.zeros(6)
    x_mean = np.zeros(12)
    f_h, c_aug = affine_to_companion(mapped, intercept, x_mean)
    now = np.arange(6, dtype=np.float64).reshape(1, 6)
    lag = now + 10.0
    y1 = np.hstack([now, lag]) @ f_h.T + c_aug.reshape(1, -1)
    np.testing.assert_allclose(y1[:, 6:], now)


def test_spectral_radius_is_not_a_logarithm():
    f_h = 0.5 * np.eye(12)
    assert abs(spectral_radius(f_h) - 0.5) < 1e-12
    source = Path(__file__).resolve().parent.joinpath("family_b_sl007.py").read_text(encoding="utf-8")
    assert "from scipy.linalg import logm" not in source
    assert "real_generator_from_F" not in source
    assert "spectral_abscissa" not in source


def test_h_step_starts_stay_inside_held_out_times():
    n_samples = 150
    n_triples = n_samples - 2
    train_sl, test_sl, gap = history_split(n_triples, N_TARGET)
    assert gap == 1
    train_times = split_sample_times(train_sl)
    test_times = split_sample_times(test_sl)
    assert train_times.isdisjoint(test_times)
    z = np.ones((n_samples, 6), dtype=np.float32)
    lag, now, tgt, t_now = horizon_starts(z, test_times, 4)
    assert now.shape[0] > 10
    tmax = max(test_times)
    tmin = min(test_times)
    assert now.shape[0] <= (tmax - tmin - 4)
    assert set(int(t) for t in t_now).isdisjoint(train_times)


def test_exploding_rollout_is_a_failed_score_not_dropped():
    from family_b_sl007 import rel_mse_z

    f_h = 1.0e3 * np.eye(12)
    c_aug = np.zeros(12)
    now = np.ones((8, 6), dtype=np.float32)
    lag = np.ones((8, 6), dtype=np.float32)
    pred = companion_predict_z(f_h, c_aug, now, lag, 4)
    assert not np.isfinite(pred).all()
    actual = np.ones((8, 6), dtype=np.float32)
    assert np.isnan(rel_mse_z(pred, actual))


def test_horizon_starts_do_not_cross_a_nan_gap():
    z = np.ones((80, 6), dtype=np.float32)
    z[50] = np.nan
    test_times = set(range(40, 80))
    _, _, _, t_now = horizon_starts(z, test_times, 4)
    for t in t_now:
        span = set(range(int(t) - 1, int(t) + 4 + 1))
        assert 50 not in span


def test_reversed_test_block_rejects_gappy_times():
    from family_b_sl007 import reversed_test_block

    z = np.ones((6, 6), dtype=np.float32)
    block, times = reversed_test_block(z, {0, 1, 3, 4})
    assert block.shape[0] == 0
    assert times == set()


def test_reversed_test_block_flips_time_not_coordinates():
    from family_b_sl007 import reversed_test_block

    z = np.arange(24, dtype=np.float32).reshape(4, 6)
    block, times = reversed_test_block(z, {0, 1, 2, 3})
    np.testing.assert_array_equal(block[0], z[-1])
    np.testing.assert_array_equal(block[-1], z[0])
    assert times == {0, 1, 2, 3}


def test_second_order_truth_known_fh_beats_controls_at_h1():
    report = run_c0_operator(rng_seed=906)
    second = report["rows"]["second_order"]
    h1 = second["horizons"]["1"]
    assert h1["fh"]["rel_mse_median"] < h1["persist"]["rel_mse_median"]
    assert h1["fh"]["rel_mse_median"] < h1["shuffle"]["rel_mse_median"]
    assert h1["fh"]["rel_mse_median"] < 0.9
    assert second["equivalence_rate"] == 1.0
    # Time-reversed test is a diagnostic, not a C0 requirement: a stationary
    # AR(2) is nearly second-order reversible.


def test_first_order_h1_still_closes():
    report = run_c0_operator(rng_seed=906)
    first = report["rows"]["first_order"]
    assert first["horizons"]["1"]["fh"]["rel_mse_median"] < 0.9
    assert first["horizons"]["1"]["fh"]["p_r"] >= 0.5


def test_operator_decision_requires_multistep_not_just_h1():
    def cov(rel, closure=False, p=0.8):
        return {
            "median_p_r": p,
            "recording_rel_mse_median": rel,
            "closure_pass": closure,
            "p_recordings_p_r_gt_half": 1.0 if p > 0.5 else 0.0,
        }

    ar_only = operator_decision(
        {
            "1": {"fh": cov(0.83, True), "persist": cov(1.0), "reverse": cov(0.95), "shuffle": cov(1.0)},
            "2": {"fh": cov(1.05), "persist": cov(1.1), "reverse": cov(1.08), "shuffle": cov(1.2)},
            "4": {"fh": cov(1.4), "persist": cov(1.2), "reverse": cov(1.3), "shuffle": cov(1.5)},
        }
    )
    assert ar_only["branch"] == "AR_ONLY"
    licensed = operator_decision(
        {
            "1": {"fh": cov(0.83, True), "persist": cov(1.0), "reverse": cov(0.95), "shuffle": cov(1.0)},
            "2": {"fh": cov(0.85, True), "persist": cov(1.05), "reverse": cov(0.98), "shuffle": cov(1.1)},
            "4": {"fh": cov(0.88, True), "persist": cov(1.1), "reverse": cov(1.02), "shuffle": cov(1.2)},
        }
    )
    assert licensed["branch"] == "OPERATOR_LICENSED"
    assert licensed["order_sensitive"] is True
    assert licensed["reverse_is_time_reversed_test"] is True


def test_sl007_report_is_cpc_blind_has_no_logm_and_is_finite():
    report = sl007_report(rng_seed=906)
    assert report["alpha_outside_acceptance"] is True
    assert report["logm_outside_acceptance"] is True
    assert report["cpc_blind"] is True
    assert report["not_mnps"] is True
    icare = report["icare"]
    assert icare["alpha_computed"] is False
    assert icare["logm_computed"] is False
    assert icare["cpc_inspected"] is False
    ready = _json_ready(report)
    json.dumps(ready, allow_nan=False)
    _assert_no_none(ready)
    if icare.get("status") == "SCORED":
        assert icare["panel"] == PRIMARY_PANEL
        assert icare["n_recordings"] == 12
        assert icare["columns"] == list(PANELS[PRIMARY_PANEL])
        assert icare["decision"]["alpha_computed"] is False
        assert icare["decision"]["logm_computed"] is False
        assert icare["decision"]["reverse_is_time_reversed_test"] is True
        blob = json.dumps(icare["decision"])
        assert "alpha_H" not in blob


def test_epoch_equivalence_on_random_series():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(150, 6)).astype(np.float32)
    row = qualify_operator_epoch(z, columns=tuple(f"z{i}" for i in range(6)), rng=rng)
    assert row["status"] == "SCORED"
    assert row["equivalence_ok"] is True
    assert row["equivalence_max_abs"] < 1e-4
    assert abs(row["m1_rel_mse"] - row["horizons"]["1"]["fh"]["rel_mse_oos"]) < EQUIV_ATOL

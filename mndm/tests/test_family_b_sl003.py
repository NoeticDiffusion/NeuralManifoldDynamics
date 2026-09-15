"""SL-003 epoch generator tests: SG vs F→log(F)/Δ, explicit failure states."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import DT_8_SEC, ICARE_0284_RUN, REGIMES, spectral_abscissa  # noqa: E402
from family_b_sl003 import (  # noqa: E402
    EPOCH_SEC,
    STATUS_GENERATOR_ILL_CONDITIONED,
    STATUS_LOG_BRANCH_UNSTABLE,
    STATUS_NO_REAL_GENERATOR,
    STATUS_TRANSITION_INSUFFICIENT,
    STATUS_TRANSITION_OOS_FAIL,
    STATUS_VALID_EPOCH_GENERATOR,
    consecutive_one_step_pairs,
    n_epoch_samples,
    pair_blocked_slices,
    qualify_transition_epoch,
    real_generator_from_F,
    run_icare_epochs,
    run_truth_known,
    summarize_icare,
)


def test_log_of_expm_recovers_stable_generator():
    j_true = REGIMES["stable"]
    dt = DT_8_SEC
    F = expm(j_true * dt)
    j_hat, status = real_generator_from_F(F, dt)
    assert status == STATUS_VALID_EPOCH_GENERATOR
    assert j_hat is not None
    assert np.allclose(j_hat, j_true, atol=1e-8)
    assert spectral_abscissa(j_hat) < 0


def test_negative_real_eigenvalue_has_no_real_generator():
    F = np.diag([-0.5, 0.8, 0.9])
    j_hat, status = real_generator_from_F(F, DT_8_SEC)
    assert j_hat is None
    assert status == STATUS_NO_REAL_GENERATOR


def test_left_half_plane_near_branch_cut_is_unstable():
    F = np.array(
        [[-1.0, 0.1, 0.0], [-0.1, -1.0, 0.0], [0.0, 0.0, 0.5]],
        dtype=np.float64,
    )
    j_hat, status = real_generator_from_F(F, DT_8_SEC)
    assert j_hat is None
    assert status == STATUS_LOG_BRANCH_UNSTABLE


def test_ill_conditioned_F_is_rejected():
    F = np.diag([1.0e4, 1.0e-5, 1.0])
    j_hat, status = real_generator_from_F(F, DT_8_SEC)
    assert j_hat is None
    assert status == STATUS_GENERATOR_ILL_CONDITIONED


def test_three_minute_8_8_split_is_insufficient_with_embargo():
    with pytest.raises(ValueError, match="too small"):
        pair_blocked_slices(n_epoch_samples(EPOCH_SEC["sensitivity"], DT_8_SEC) - 1)


def test_gap_does_not_stitch_nonadjacent_samples():
    rng = np.random.default_rng(712)
    x = rng.normal(size=(n_epoch_samples(EPOCH_SEC["primary"], DT_8_SEC), 3)).astype(np.float32)
    x[10] = np.nan
    src, tgt = consecutive_one_step_pairs(x)
    assert src.shape[0] == x.shape[0] - 1 - 2
    compact = x[np.isfinite(x).all(axis=1)]
    assert src.shape[0] != compact.shape[0] - 1
    row = qualify_transition_epoch(x, dt=DT_8_SEC, family_b_scope=True)
    assert row["n_pairs"] == src.shape[0]
    assert row["family_b_alpha_licensed"] is False


def test_family_b_scope_false_never_licenses_alpha():
    rng = np.random.default_rng(713)
    x = rng.normal(size=(n_epoch_samples(EPOCH_SEC["primary"], DT_8_SEC), 3)).astype(np.float32)
    row = qualify_transition_epoch(x, dt=DT_8_SEC, family_b_scope=False)
    assert row["family_b_alpha_licensed"] is False



def test_white_noise_epoch_is_transition_oos_fail():
    rng = np.random.default_rng(710)
    x = rng.normal(size=(n_epoch_samples(EPOCH_SEC["primary"], DT_8_SEC), 3)).astype(np.float32)
    row = qualify_transition_epoch(x, dt=DT_8_SEC)
    assert row["status"] == STATUS_TRANSITION_OOS_FAIL
    assert row["alpha_licensed"] is False


def test_short_epoch_is_transition_insufficient_not_oos_fail():
    rng = np.random.default_rng(711)
    x = rng.normal(size=(n_epoch_samples(EPOCH_SEC["exploratory"], DT_8_SEC), 3)).astype(np.float32)
    row = qualify_transition_epoch(x, dt=DT_8_SEC)
    assert row["status"] == STATUS_TRANSITION_INSUFFICIENT
    assert row["alpha_licensed"] is False


def test_truth_known_primary_8_8_licenses_transition_on_stable():
    report = run_truth_known(rng_seed=703)
    stable = [
        row
        for row in report["rows"]
        if row["regime"] == "stable" and row["grid"] == "8/8" and row["epoch_role"] == "primary"
    ]
    assert len(stable) == 1
    row = stable[0]
    assert row["n_epochs"] >= 1
    assert row["n_transition_licensed"] >= 1
    assert row["n_transition_alpha_match"] >= 1


@pytest.mark.skipif(not ICARE_0284_RUN.exists(), reason="I-CARE 0284 dump not mounted")
def test_icare_primary_is_nonoverlapping_8_8():
    report = run_icare_epochs()
    assert report["primary_grid"] == "8/8"
    assert report["n_files"] == 12
    summary = summarize_icare(report)
    assert summary["grid_8_8"]["n_epochs"] >= 1
    assert summary["grid_8_4"]["n_family_b_alpha_licensed"] == 0
    assert summary["grid_8_8"]["n_family_b_alpha_licensed"] == summary["grid_8_8"]["n_generator_qualified"]

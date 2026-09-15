"""Independent truth-known edge checks for the offline W_Q diagnostic."""

import numpy as np

from mndm.tools.wq_horizon_diagnostic import ordinary_recurrence, scaled_recurrence


def test_zero_phi_and_zero_q_are_stable_zero_cases():
    zero_phi = np.repeat(np.zeros((1, 2, 2)), 4, axis=0)
    identity_q = np.eye(2)
    ordinary = ordinary_recurrence(zero_phi, identity_q, 4)
    scaled = scaled_recurrence(zero_phi, identity_q, 4)
    np.testing.assert_allclose(ordinary["covariance"], identity_q)
    assert scaled["representable"]
    np.testing.assert_allclose(scaled["covariance"], identity_q)

    zero_q = np.zeros((2, 2))
    ordinary_zero = ordinary_recurrence(zero_phi, zero_q, 4)
    scaled_zero = scaled_recurrence(zero_phi, zero_q, 4)
    np.testing.assert_allclose(ordinary_zero["covariance"], zero_q)
    assert scaled_zero["representable"]
    np.testing.assert_allclose(scaled_zero["covariance"], zero_q)
    assert scaled_zero["first_unrepresentable_step"] is None


def test_large_finite_propagator_is_scaled_without_intermediate_overflow():
    phi = np.repeat((1.0e200 * np.eye(1))[None], 4, axis=0)
    q = np.ones((1, 1))
    scaled = scaled_recurrence(phi, q, 4)
    assert scaled["scaled_finite"]
    assert scaled["first_unrepresentable_step"] is None
    assert scaled["first_range_exceed_step"] is not None
    assert not scaled["representable"]
    assert np.isfinite(scaled["normalized_covariance"]).all()


def test_scaled_recurrence_distinguishes_transient_range_exceed_from_final_recovery():
    phi = np.asarray(
        [
            [[1.0]],
            [[1.0e200]],
            [[1.0e-200]],
            [[1.0]],
        ]
    )
    q = np.ones((1, 1))
    scaled = scaled_recurrence(phi, q, 4)
    assert scaled["first_range_exceed_step"] == 2
    assert scaled["first_unrepresentable_step"] is None
    assert scaled["representable"]
    assert np.isfinite(scaled["covariance"]).all()


def test_q_scaling_is_homogeneous_when_both_results_are_representable():
    phi = np.repeat((0.7 * np.eye(2))[None], 8, axis=0)
    q = np.diag([1.0, 2.0])
    scale = 3.5
    a = scaled_recurrence(phi, q, 8)
    b = scaled_recurrence(phi, scale * q, 8)
    assert a["representable"] and b["representable"]
    np.testing.assert_allclose(b["covariance"], scale * a["covariance"], rtol=1e-12, atol=1e-12)


def test_tiny_q_reports_relative_term_loss_in_extreme_scale_separation():
    phi = np.asarray([[[1.0]], [[1.0e200]], [[1.0e-200]]])
    q = np.asarray([[1.0e-300]])
    scaled = scaled_recurrence(phi, q, 3)
    assert scaled["scaled_finite"]
    assert isinstance(scaled["underflow_loss"], bool)
    assert scaled["underflow_loss"]

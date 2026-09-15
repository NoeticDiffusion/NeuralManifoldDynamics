"""SL-008 Type-C latent SSM tests: rescue h=2, not CPC, not log(A)."""

from pathlib import Path
import json
import sys

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from family_b_sl005 import PANELS, PRIMARY_PANEL  # noqa: E402
from family_b_sl008 import (  # noqa: E402
    H1_AR_REF,
    LATENT_DIMS,
    SSM_NAME,
    fit_lgssm,
    kalman_filter,
    run_c0_ssm,
    sl008_report,
    ssm_decision,
    ssm_predict_z,
    static_predict_z,
)


def _assert_no_none(value, path: str = "root") -> None:
    if value is None:
        if path.endswith("selected_d_s"):
            return
        raise AssertionError(f"unexpected None at {path}")
    if isinstance(value, dict):
        for key, inner in value.items():
            _assert_no_none(inner, f"{path}.{key}")
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            _assert_no_none(inner, f"{path}[{i}]")


def test_latent_dims_are_frozen_and_cpc_blind():
    assert LATENT_DIMS == (2, 3, 4, 6)
    assert SSM_NAME == "latent_type_c_ssm"
    assert PANELS[PRIMARY_PANEL][0] == "eeg_theta"
    source = Path(__file__).resolve().parent.joinpath("family_b_sl008.py").read_text(encoding="utf-8")
    assert "from scipy.linalg import logm" not in source
    assert "spectral_abscissa" not in source
    assert "real_generator_from_F" not in source


def test_static_control_ignores_A_power():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(80, 6))
    params = fit_lgssm(z, 3)
    assert params is not None
    now = z[10:20]
    a = static_predict_z(params, now)
    b = static_predict_z({**params, "A": 0.0 * params["A"]}, now)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_ssm_h_step_uses_filtered_state_not_future_z():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(40, 6))
    params = fit_lgssm(z, 2, n_em=2)
    assert params is not None
    s_f, _, _, _ = kalman_filter(z, A=params["A"], C=params["C"], Q=params["Q"], R=params["R"], mu=params["mu"])
    pred = ssm_predict_z(params, s_f, np.array([5, 6, 7]), 2)
    assert pred.shape == (3, 6)
    z_future = z.copy()
    z_future[8:] = 1.0e3
    s_f2, _, _, _ = kalman_filter(
        z_future, A=params["A"], C=params["C"], Q=params["Q"], R=params["R"], mu=params["mu"]
    )
    pred2 = ssm_predict_z(params, s_f2, np.array([5, 6, 7]), 2)
    np.testing.assert_allclose(pred, pred2, atol=1e-6)


def test_hidden_state_c0_ssm_beats_m1_at_h2():
    report = run_c0_ssm(rng_seed=908)
    hidden = report["rows"]["hidden_3d"]
    h2 = hidden["horizons"]["2"]
    assert h2["ssm_3"]["rel_mse_median"] < 0.9
    assert h2["ssm_3"]["rel_mse_median"] < h2["m1"]["rel_mse_median"]
    assert h2["ssm_3"]["rel_mse_median"] < h2["static_3"]["rel_mse_median"]


def test_ssm_decision_requires_h2_not_just_h1():
    def cov(rel, closure=False, p=0.8):
        return {
            "median_p_r": p,
            "recording_rel_mse_median": rel,
            "closure_pass": closure,
            "p_recordings_p_r_gt_half": 1.0 if p > 0.5 else 0.0,
        }

    def pack(h1_ssm, h2_ssm, h4_ssm):
        out = {}
        for h, ssm, m1, stat in (
            (1, h1_ssm, 0.83, 0.95),
            (2, h2_ssm, 0.96, 1.1),
            (4, h4_ssm, 1.04, 1.2),
        ):
            out[str(h)] = {
                "m0": cov(1.0),
                "m1": cov(m1),
                "persist": cov(1.4),
                "ssm_2": cov(ssm, closure=ssm < 0.9),
                "ssm_3": cov(ssm, closure=ssm < 0.9),
                "ssm_4": cov(ssm, closure=ssm < 0.9),
                "ssm_6": cov(ssm, closure=ssm < 0.9),
                "static_2": cov(stat),
                "static_3": cov(stat),
                "static_4": cov(stat),
                "static_6": cov(stat),
            }
        return out

    repack = ssm_decision(pack(0.80, 0.95, 1.03))
    assert repack["branch"] == "REPACKAGED_AR"
    assert repack["selected_d_s"] is None
    licensed = ssm_decision(pack(0.78, 0.84, 0.88))
    assert licensed["selected_d_s"] == 2
    assert licensed["branch"] == "SSM_LICENSED"
    assert licensed["cpc_not_used_for_d_s"] is True


def test_noncontiguous_train_block_is_rejected():
    from family_b_sl008 import _block

    z = np.ones((10, 6), dtype=np.float32)
    assert _block(z, {0, 1, 2}) is not None
    assert _block(z, {0, 1, 3}) is None


def test_exploding_A_fails_ssm_score():
    from family_b_sl008 import rel_mse_z

    params = {
        "A": 200.0 * np.eye(2),
        "C": np.ones((6, 2)),
        "mu": np.zeros(6),
        "Q": np.eye(2),
        "R": np.eye(6),
    }
    s_f = np.ones((5, 2))
    pred = ssm_predict_z(params, s_f, np.array([0, 1, 2]), 4)
    assert not np.isfinite(pred).all()
    actual = np.ones((3, 6), dtype=np.float32)
    assert np.isnan(rel_mse_z(pred, actual))
    assert abs(H1_AR_REF - 0.83) < 1e-9


def test_sl008_report_is_cpc_blind_and_finite():
    report = sl008_report(rng_seed=908)
    assert report["alpha_outside_acceptance"] is True
    assert report["logm_outside_acceptance"] is True
    assert report["cpc_blind"] is True
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
        assert icare["decision"]["cpc_not_used_for_d_s"] is True
        assert "alpha_s" not in json.dumps(icare["decision"])

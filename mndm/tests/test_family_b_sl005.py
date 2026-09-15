"""SL-005 Type-C-1 tests: frozen z^C panels, closure only, no alpha/CPC."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from family_b_sl005 import (  # noqa: E402
    NAME_TYPE_C_STATE,
    PANELS,
    PRIMARY_PANEL,
    STATUS_TRANSITION_OOS_FAIL,
    TYPE_C_OVERLAY,
    affine_n_params,
    assert_not_mnps_state,
    coverage_summary,
    drop_cpc_columns,
    panel_min_train,
    qualify_type_c_closure,
    run_c0_type_c_dimension,
    sl005_report,
)


def test_primary_panel_is_six_d_without_delta_gamma_or_mnps():
    cols = PANELS[PRIMARY_PANEL]
    assert len(cols) == 6
    assert affine_n_params(6) == 42
    assert panel_min_train(6) == 42
    blob = " ".join(cols)
    assert "delta" not in blob
    assert "gamma" not in blob
    assert NAME_TYPE_C_STATE == "type_c_state"
    assert_not_mnps_state(cols)
    with pytest.raises(ValueError):
        assert_not_mnps_state(["eeg_delta", "eeg_theta"])
    with pytest.raises(ValueError):
        assert_not_mnps_state(["m_a", "d_n"])


def test_overlay_is_dynamical_families_type_c_not_mnps_summarize():
    assert "dynamical_families" in TYPE_C_OVERLAY.name
    raw = yaml.safe_load(TYPE_C_OVERLAY.read_text(encoding="utf-8"))
    assert raw["epoching"]["length_s"] == 2.0
    assert raw["epoching"]["step_s"] == 2.0
    assert raw["mnps"]["overlap"] == 0.0
    assert raw["mnps"]["jacobian"]["enabled"] is False
    assert raw["mnps_9d"]["enabled"] is False
    assert raw["dynamical_families"]["enabled"] is False
    assert raw["conventional_eeg"]["enabled"] is False
    assert raw["conventional_eeg"]["datasets"]["physionet_icare_2_1"]["enabled"] is False
    assert raw["features"]["hjorth_activity"] is True
    assert raw.get("summarize_forbidden") is True


def test_nonpositive_bandpower_is_counted_not_silent():
    rng = np.random.default_rng(811)
    z = rng.lognormal(mean=0.0, sigma=0.5, size=(150, 6)).astype(np.float32)
    z[:20, 0] = -1.0
    row = qualify_type_c_closure(z, columns=PANELS[PRIMARY_PANEL])
    assert row["n_log10_nonpositive"] == 20
    assert row["oos_pass"] is False


def test_summarize_forbidden_aborts_mnps_projection():
    from mndm.orchestrate import cmd_summarize

    assert cmd_summarize({"summarize_forbidden": True}, ["physionet_icare_2_1"], None) == 1


def test_white_noise_six_d_epoch_fails_closure():
    rng = np.random.default_rng(810)
    z = rng.lognormal(mean=0.0, sigma=0.5, size=(150, 6)).astype(np.float32)
    row = qualify_type_c_closure(z, columns=PANELS[PRIMARY_PANEL])
    assert row["status"] == STATUS_TRANSITION_OOS_FAIL
    assert row["oos_pass"] is False
    assert row["n_log10_nonpositive"] == 0
    assert "alpha" not in row
    assert "family_b_alpha_licensed" not in row


def test_c0_six_d_stable_identifies_transition_without_alpha():
    rec = run_c0_type_c_dimension(rng_seed=805, n_dim=6)
    assert rec["c0_pass"] is True
    assert rec["p_r"] >= 0.5
    assert rec["rel_mse_median"] < 0.9
    assert "alpha" not in rec


def test_cpc_columns_are_dropped_before_state_build():
    frame = pd.DataFrame(
        {
            "eeg_theta": np.ones(20),
            "cpc": np.arange(20),
            "hospital": ["A"] * 20,
            "group": ["x"] * 20,
        }
    )
    cleaned = drop_cpc_columns(frame)
    assert "cpc" not in cleaned.columns
    assert "eeg_theta" in cleaned.columns


def test_coverage_gate_requires_broad_recording_support():
    recs = [
        {"p_r": 0.8, "rel_mse_median": 0.6},
        {"p_r": 0.7, "rel_mse_median": 0.5},
        {"p_r": 0.1, "rel_mse_median": 1.2},
    ]
    summary = coverage_summary(recs)
    assert summary["alpha_computed"] is False
    assert summary["cpc_inspected"] is False
    assert summary["closure_pass"] is True
    tail = [{"p_r": 1.0, "rel_mse_median": 0.4}] + [{"p_r": 0.0, "rel_mse_median": 1.3}] * 5
    assert coverage_summary(tail)["closure_pass"] is False


def test_sl005_report_json_finite_and_c2_does_not_invent_alpha():
    report = sl005_report(rng_seed=805)
    assert report["not_mnps"] is True
    assert report["alpha_outside_acceptance"] is True
    assert report["cpc_blind"] is True
    c2 = report["c2"]
    assert c2["alpha_computed"] is False
    assert c2["cpc_inspected"] is False
    json.dumps(_json_ready(report), allow_nan=False)
    if c2.get("status") == "SCORED":
        assert c2["primary_panel"] == PRIMARY_PANEL
        assert "alpha" not in c2["panels"][PRIMARY_PANEL]["coverage"]
        assert c2["panels"][PRIMARY_PANEL]["coverage"]["n_recordings"] == 12
        assert c2["n_conventional_columns"] == 0


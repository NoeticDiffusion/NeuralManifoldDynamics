"""SL-004 Type-C-0 tests: 2/2 truth-known, feature audit, blocked 0284."""

from pathlib import Path
import json
import sys

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from family_b_sl004 import (  # noqa: E402
    DT_2_SEC,
    VERDICT_CANDIDATE,
    VERDICT_DROP,
    VERDICT_REDEFINE,
    feature_support_audit,
    n_cycles,
    run_c0_truth_known,
    run_c2_icare_status,
    sl004_report,
)


def test_two_second_window_has_one_cycle_at_half_hertz():
    assert n_cycles(0.5, 2.0) == 1.0
    assert n_cycles(1.0, 2.0) == 2.0


def test_c1_refuses_automatic_eight_second_carry():
    audit = feature_support_audit()
    assert audit["automatic_8s_carry_forbidden"] is True
    assert audit["not_mnps_v2_but_faster"] is True
    by_name = {row["feature"]: row for row in audit["rows"]}
    assert by_name["eeg_delta"]["verdict"] == VERDICT_REDEFINE
    assert by_name["eeg_gamma"]["verdict"] == VERDICT_REDEFINE
    assert by_name["embodied_arousal_proxy"]["verdict"] == VERDICT_DROP
    assert by_name["eeg_sample_entropy"]["verdict"] == VERDICT_DROP
    assert by_name["eeg_alpha"]["verdict"] == VERDICT_CANDIDATE
    assert "cuts at 40" not in by_name["eeg_beta"]["reason"]
    assert by_name["eeg_gamma"]["band_hz"][1] > 40.0
    assert audit["verdict_counts"][VERDICT_DROP] >= 1
    assert audit["verdict_counts"][VERDICT_CANDIDATE] < audit["n_features"]


def test_c0_stable_identifies_generator_without_using_alpha():
    report = run_c0_truth_known(rng_seed=804)
    assert report["grid"] == "2/2"
    assert report["stable_required"] is True
    assert report["stable_c0_pass"] is True
    stable = next(row for row in report["rows"] if row["regime"] == "stable")
    assert stable["alpha_is_acceptance"] is False
    assert stable["oracle_xdot_identified"] is True
    assert stable["generator_identified"] is True
    assert stable["epoch_5min"]["n_family_b_alpha_licensed"] == 0
    assert stable["series"]["dt_sec"] == DT_2_SEC
    assert stable["epoch_5min"]["n_epoch_samples"] == 150
    assert np.isfinite(stable["series"]["rel_mse_j_oracle_xdot"])
    assert stable["series"]["rel_mse_j_oracle_xdot"] < 0.9
    assert stable["series"]["rel_mse_a_next"] < 0.9


def test_c2_is_blocked_and_does_not_slice_mnps():
    c2 = run_c2_icare_status()
    assert c2["status"] == "BLOCKED"
    assert c2["n_family_b_alpha_licensed"] == 0
    assert any("slice mnps_3d" in item for item in c2["forbidden"])
    report = sl004_report(rng_seed=804)
    assert report["family_b_8s"] == "CLOSED_METHOD_LIMITED"
    assert report["type_c_is_new_family"] is True
    assert report["alpha_outside_acceptance"] is True
    assert report["c2"]["status"] == "BLOCKED"
    json.dumps(_json_ready(report), allow_nan=False)

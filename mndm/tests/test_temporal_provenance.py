import numpy as np

from mndm.features.epoch_selection import resolve_epoch_params
from mndm.pipeline.robustness_helpers import compute_window_time_audit
from mndm.pipeline.summary import resolve_effective_epoch_contract


def test_effective_epoch_override_is_distinct_from_generic_mnps_defaults():
    cfg = {"epoching": {"length_s": 8.0, "step_s": 4.0, "datasets": {"ds005555": {"length_s": 30.0, "step_s": 30.0}}}}
    assert resolve_epoch_params(cfg, "ds005555") == (30.0, 30.0)
    audit = compute_window_time_audit(
        time=np.array([15.0, 45.0, 75.0]),
        window_start=np.array([0.0, 30.0, 60.0]),
        window_end=np.array([30.0, 60.0, 90.0]),
        dt_sec_runtime=30.0,
        dt_sec_config=4.0,
        window_sec_config=8.0,
        dt_sec_epoch_config=30.0,
        window_sec_epoch_config=30.0,
    )
    assert audit["dt_sec_config"] == 4.0
    assert audit["window_sec_config"] == 8.0
    assert audit["dt_sec_epoch_config"] == 30.0
    assert audit["window_sec_epoch_config"] == 30.0
    assert audit["status"] == "ok"
    assert resolve_effective_epoch_contract(cfg, "ds005555", "eeg") == (30.0, 30.0)


def test_generic_epoch_fallback_remains_explicit():
    assert resolve_epoch_params({"epoching": {"length_s": 8.0, "step_s": 4.0}}, "other") == (8.0, 4.0)
    audit = compute_window_time_audit(
        time=np.array([4.0, 8.0]), window_start=np.array([0.0, 4.0]), window_end=np.array([8.0, 12.0]),
        dt_sec_runtime=4.0, dt_sec_config=4.0, window_sec_config=8.0,
    )
    assert audit["dt_sec_epoch_config"] == 4.0
    assert audit["window_sec_epoch_config"] == 8.0
    assert resolve_effective_epoch_contract({"mnps": {"window_sec": 12.0}, "epoching": {}}, "other", "eeg") is None
    assert resolve_effective_epoch_contract({"epoching": {"length_s": 30.0, "step_s": 30.0}}, "other", "fmri") is None
    assert resolve_effective_epoch_contract({"epoching": {"datasets": {"ds": {"sampling": {"enabled": False}}}}}, "ds", "eeg") is None


def test_observed_mismatch_and_gap_remain_warnings():
    audit = compute_window_time_audit(
        time=np.array([15.0, 45.0, 105.0]),
        window_start=np.array([0.0, 30.0, 90.0]),
        window_end=np.array([30.0, 60.0, 120.0]),
        dt_sec_runtime=30.0,
        dt_sec_config=4.0,
        window_sec_config=8.0,
        dt_sec_epoch_config=30.0,
        window_sec_epoch_config=30.0,
    )
    assert audit["status"] == "warning"
    assert audit["dt_matches_epoch_config"] is False
    assert any("effective epoching" not in warning for warning in audit["warnings"])

    length_mismatch = compute_window_time_audit(
        time=np.array([15.0, 45.0]), window_start=np.array([0.0, 30.0]), window_end=np.array([20.0, 50.0]),
        dt_sec_runtime=30.0, dt_sec_config=4.0, window_sec_config=8.0,
        dt_sec_epoch_config=30.0, window_sec_epoch_config=30.0,
    )
    assert length_mismatch["window_len_matches_epoch_config"] is False
    assert any("window length disagrees with effective epoching" in warning for warning in length_mismatch["warnings"])

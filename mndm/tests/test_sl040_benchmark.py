"""Focused tests for frozen SL-040 selection and fold-local v2 calibration."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCIENCE = ROOT / "project/new_mnps_v.3.2/science_lead"
ANALYSIS_PATH = Path(r"J:/repos/nmd-analysis/project/sl040_boas_benchmark_20261104/analyze.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_window_selection_falls_back_to_first_last_for_spacing():
    select = load_module("sl040_select_under_test", SCIENCE / "sl040_benchmark_select.py")
    candidates = []
    for stage in select.STAGES:
        for second in (0.0, 200.0, 400.0, 600.0, 3600.0):
            candidates.append({"stage": stage, "duration_s": 20, "target_start_s": second})
    result = select.choose_targets(candidates)
    assert result is not None
    windows, methods = result
    assert len(windows) == 8
    assert set(methods.values()) == {"first_last_fallback"}
    for stage in select.STAGES:
        pair = [row["target_start_s"] for row in windows if row["stage"] == stage]
        assert pair == [0.0, 3600.0]


def _analysis_module():
    return load_module("sl040_analysis_under_test", ANALYSIS_PATH)


def _training_rows(analysis, *, constant_feature: str | None = None):
    rows = []
    idx = 0
    for pid in ("8", "9"):
        for stage in analysis.STAGES:
            for window in range(2):
                raw = {}
                for feature in analysis.ANCHOR_FEATURES:
                    if feature == constant_feature:
                        raw[feature] = 2.0
                    elif feature in analysis.LOG10_FEATURES:
                        raw[feature] = 1.0 + idx + window
                    else:
                        raw[feature] = 0.5 + idx + window
                rows.append({"pid": pid, "stage": stage, "window_id": f"{pid}_{stage}_{window}", "v2_raw": raw})
            idx += 1
    return rows


def test_v2_anchors_use_balanced_training_pool_and_preserve_map():
    analysis = _analysis_module()
    rows = _training_rows(analysis)
    anchors = analysis.fit_v2_anchors(rows)
    assert all(value["n_training_windows"] == 16 for value in anchors.values())
    coords = analysis.project_v2_row(rows[0], anchors)
    assert tuple(coords) == analysis.V2_COORDS
    z_delta = (analysis.transform_v2_raw("eeg_delta", rows[0]["v2_raw"]["eeg_delta"]) - anchors["eeg_delta"]["center"]) / anchors["eeg_delta"]["scale"]
    z_theta = (analysis.transform_v2_raw("eeg_theta", rows[0]["v2_raw"]["eeg_theta"]) - anchors["eeg_theta"]["center"]) / anchors["eeg_theta"]["scale"]
    assert coords["m_a"] == pytest.approx(-(z_delta + z_theta) / 2.0)


def test_zero_mad_and_nonpositive_log_power_are_unsupported():
    analysis = _analysis_module()
    rows = _training_rows(analysis, constant_feature="eeg_delta")
    with pytest.raises(analysis.AnchorFailure, match="anchor_median_or_mad_invalid:eeg_delta"):
        analysis.fit_v2_anchors(rows)
    with pytest.raises(ValueError, match="log10_nonpositive:eeg_gamma"):
        analysis.transform_v2_raw("eeg_gamma", 0.0)


def test_participant_bootstrap_is_reproducible_and_paired():
    analysis = _analysis_module()
    rows = []
    for pid in ("8", "9", "10"):
        for stage in analysis.STAGES:
            rows.append({"pid": pid, "stage": stage, "predictions": {
                "better": stage, "baseline": analysis.STAGES[(analysis.STAGES.index(stage) + (pid == "9")) % 4]}})
    first = analysis.participant_bootstrap_difference(rows, "better", "baseline", n_resamples=200, seed=40040)
    second = analysis.participant_bootstrap_difference(rows, "better", "baseline", n_resamples=200, seed=40040)
    assert first == second
    assert first["mean_difference"] > 0
    assert first["conditional_on_oof_predictions"] is True


def test_frozen_four_class_logistic_configuration_on_sklearn_180():
    analysis = _analysis_module()
    import sklearn
    from sklearn.linear_model import LogisticRegression

    assert sklearn.__version__ == analysis.SKLEARN_VERSION == "1.8.0"
    x = [[0.0, 0.1], [0.1, 0.0], [1.0, 1.1], [1.1, 1.0],
         [2.0, 2.1], [2.1, 2.0], [3.0, 3.1], [3.1, 3.0]]
    y = ["Wake", "Wake", "N2", "N2", "N3", "N3", "REM", "REM"]
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-4,
                               fit_intercept=True, class_weight=None, random_state=40040)
    model.fit(x, y)
    assert set(model.classes_) == set(analysis.STAGES)
    assert model.predict_proba(x).shape == (8, 4)

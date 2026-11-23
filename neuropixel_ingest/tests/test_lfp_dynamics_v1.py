from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))


def _dynamics_spec() -> dict:
    return {
        "estimands": {},
        "lag_policy": {
            "max_relative_tau_deviation_from_realized_median": 1e-4,
            "predeclared_sensitivity_lag_steps": [1, 2],
        },
        "transition_eligibility": {
            "no_crossing_of_missing_windows_or_source_timestamp_gaps": True,
        },
        "envelope_processing_block_policy": {
            "minimum_absolute_batch_id_difference_from_query": 2,
        },
        "neighborhood": {
            "minimum_eligible_transitions_per_behavior_state_and_group": 30,
            "k_unique_source_batches": 32,
            "minimum_unique_source_batches": 28,
        },
    }


def test_observed_chart_uses_separate_robust_calibration_and_keeps_zero_power_missing() -> None:
    from neuropixel_ingest.lfp_dynamics_v1 import build_observed_chart

    n = 40
    centroid = 3.0 + np.linspace(0.0, 0.5, n)
    power = np.power(10.0, np.linspace(-13.0, -12.0, n))
    envelope = 0.1 + np.linspace(0.0, 0.2, n)
    power[0] = 0.0
    support = {name: np.ones(n, dtype=bool) for name in ("centroid_hz", "bandpower_v2", "envelope_cv")}
    chart = build_observed_chart(
        centroid_hz=centroid,
        bandpower_v2=power,
        envelope_cv=envelope,
        support_pass=support,
    )

    assert chart["calibration_window_count"] == n - 1
    assert not chart["complete_mask"][0]
    assert chart["missing_reason"][0] == "zero_power_log_unsupported"
    assert np.isnan(chart["values"][0]).all()
    assert np.median(chart["values"][chart["complete_mask"]], axis=0) == pytest.approx(
        np.zeros(3), abs=1e-12
    )
    assert np.all(chart["calibration_scale"] > 0.0)


def test_observed_chart_does_not_fallback_for_zero_robust_scale() -> None:
    from neuropixel_ingest.lfp_dynamics_v1 import build_observed_chart

    n = 35
    support = {name: np.ones(n, dtype=bool) for name in ("centroid_hz", "bandpower_v2", "envelope_cv")}
    chart = build_observed_chart(
        centroid_hz=np.full(n, 4.0),
        bandpower_v2=np.power(10.0, np.linspace(-13.0, -12.0, n)),
        envelope_cv=np.linspace(0.05, 0.3, n),
        support_pass=support,
    )
    assert chart["calibration_scale_valid"].tolist() == [False, True, True]
    assert not chart["complete_mask"].any()
    assert np.isnan(chart["values"]).all()
    assert set(chart["missing_reason"]) == {"zero_or_nonfinite_robust_scale"}


def test_finite_lag_estimates_observed_rates_and_excludes_shared_or_invalid_batches() -> None:
    from neuropixel_ingest.lfp_dynamics_v1 import estimate_finite_lag_group

    n = 1800
    times = np.arange(n, dtype=np.float64) * 2.0
    slope = np.asarray([0.01, -0.02, 0.005])
    chart = times[:, None] * slope[None, :]
    complete = np.ones(n, dtype=bool)
    complete[700] = False
    states = np.full(n, "rest", dtype=object)
    states[900:] = "run"
    segments = np.full(n, "segment-0", dtype=object)
    batch_ids = np.arange(n, dtype=np.int32) // 30
    flags = np.zeros(n, dtype=bool)

    result = estimate_finite_lag_group(
        chart=chart,
        complete_mask=complete,
        center_time_sec=times,
        behavioral_state=states,
        stim_adjacent=flags,
        stim_contains_onset=flags,
        induction_excluded=flags,
        segment_id=segments,
        processing_batch_id=batch_ids,
        spec=_dynamics_spec(),
    )

    lag1 = result["state_results"]["run"]["lags"]["lag_1"]
    lag2 = result["state_results"]["run"]["lags"]["lag_2"]
    query = 900
    assert lag1["support_count"][query] == 28
    assert lag2["support_count"][query] == 28
    assert lag1["drift"][query] == pytest.approx(slope, abs=1e-12)
    assert lag2["drift"][query] == pytest.approx(slope, abs=1e-12)
    assert np.allclose(lag1["covariance_rate"][query], 0.0, atol=1e-12)
    chosen_batches = lag1["selected_source_batch_id"][query]
    chosen_batches = chosen_batches[chosen_batches >= 0]
    assert np.unique(chosen_batches).size == chosen_batches.size
    assert np.all(np.abs(chosen_batches - batch_ids[query]) >= 2)
    assert lag1["tau_sec"] == pytest.approx(2.0)
    assert lag2["tau_sec"] == pytest.approx(4.0)
    assert result["transition_summary"]["rest|lag_1"]["eligible_transitions"] < 870


def test_finite_lag_missingness_does_not_bridge_gap_or_state_boundaries() -> None:
    from neuropixel_ingest.lfp_dynamics_v1 import estimate_finite_lag_group

    n = 1800
    times = np.arange(n, dtype=np.float64) * 2.0
    times[1200:] += 12.0
    chart = np.column_stack((np.sin(times / 20.0), np.cos(times / 17.0), np.sin(times / 13.0)))
    complete = np.ones(n, dtype=bool)
    complete[1000] = False
    states = np.full(n, "rest", dtype=object)
    states[900:] = "run"
    segments = np.full(n, "segment-0", dtype=object)
    segments[1200:] = "segment-1"
    batch_ids = np.arange(n, dtype=np.int32) // 30
    flags = np.zeros(n, dtype=bool)
    flags[1100] = True

    result = estimate_finite_lag_group(
        chart=chart,
        complete_mask=complete,
        center_time_sec=times,
        behavioral_state=states,
        stim_adjacent=flags,
        stim_contains_onset=np.zeros(n, dtype=bool),
        induction_excluded=np.zeros(n, dtype=bool),
        segment_id=segments,
        processing_batch_id=batch_ids,
        spec=_dynamics_spec(),
    )
    eligible = result["transition_summary"]["run|lag_1"]["eligible_transitions"]
    assert eligible < 870
    lag1 = result["state_results"]["run"]["lags"]["lag_1"]
    eligible_sources = set(lag1["eligible_source_transition_index"].tolist())
    assert eligible_sources.isdisjoint({899, 999, 1000, 1099, 1100, 1199, 1499})
    eligible_source_ids = lag1["eligible_source_transition_index"]
    assert np.all(batch_ids[eligible_source_ids] == batch_ids[eligible_source_ids + 1])
    assert lag1["query_valid"][1100] == 0


def test_lfp_dynamics_hdf5_roundtrip_keeps_chart_and_transition_support(tmp_path: Path) -> None:
    from neuropixel_ingest.lfp_dynamics_v1 import DYNAMICS_SCHEMA, write_lfp_dynamics_h5

    n = 2
    chart_result = {
        "coordinate_names": ["c", "p", "v"],
        "raw_values": np.ones((n, 3)),
        "values": np.asarray([[0.0, 0.1, -0.1], [0.2, 0.0, 0.3]]),
        "component_support": np.ones((n, 3), dtype=bool),
        "complete_mask": np.ones(n, dtype=bool),
        "missing_reason": np.asarray(["", ""], dtype=object),
        "calibration_median": np.zeros(3),
        "calibration_scale": np.ones(3),
        "calibration_scale_valid": np.ones(3, dtype=bool),
    }
    lag_result = {
        "lag_steps": 1,
        "tau_sec": 2.0,
        "query_window_index": np.arange(n, dtype=np.int32),
        "query_valid": np.ones(n, dtype=np.uint8),
        "drift": np.ones((n, 3)),
        "covariance_rate": np.ones((n, 3, 3)),
        "support_count": np.asarray([28, 27], dtype=np.int32),
        "local_radius": np.asarray([0.5, np.nan]),
        "missing_reason": np.asarray(["", "insufficient_local_support"], dtype=object),
        "ineligible_transition_counts": {"missing": 2, "gap": 1},
        "selected_source_transition_index": np.full((n, 32), -1, dtype=np.int32),
        "selected_source_batch_id": np.full((n, 32), -1, dtype=np.int32),
        "selected_source_distance": np.full((n, 32), np.nan),
    }
    result = {
        "chart": chart_result,
        "center_time_sec": np.asarray([1.0, 3.0]),
        "source_sample_start": np.asarray([0, 5000]),
        "segment_id": np.asarray(["segment-0", "segment-0"], dtype=object),
        "behavioral_state": np.asarray(["rest", "rest"], dtype=object),
        "stim_adjacent": np.zeros(n, dtype=bool),
        "stim_contains_onset": np.zeros(n, dtype=bool),
        "induction_excluded": np.zeros(n, dtype=bool),
        "dynamics": {
            "processing_batch_ids": np.asarray([0, 0]),
            "state_results": {
                "rest": {
                    "behavioral_state": "rest",
                    "lags": {"lag_1": lag_result},
                }
            },
            "transition_summary": {"rest|lag_1": {"eligible_transitions": 1}},
        },
    }
    target = tmp_path / "lfp_dynamics.h5"
    write_lfp_dynamics_h5(
        target,
        groups={"probeF__MOs__depth_bin_2": result},
        chart_spec={"chart_id": "chart-v1"},
        dynamics_spec={"estimator_id": "finite-lag-v1"},
        provenance={"condition_comparisons_performed": False},
    )
    with h5py.File(target, "r") as handle:
        assert handle.attrs["schema"] == DYNAMICS_SCHEMA
        assert handle["groups/probeF__MOs__depth_bin_2/chart/values"].shape == (n, 3)
        assert handle[
            "groups/probeF__MOs__depth_bin_2/dynamical_families/state_000/lag_1/covariance_rate"
        ].shape == (n, 3, 3)
        assert handle[
            "groups/probeF__MOs__depth_bin_2/dynamical_families/state_000/lag_1/support_count"
        ][1] == 27
        exclusions = handle[
            "groups/probeF__MOs__depth_bin_2/dynamical_families/state_000/lag_1/ineligible_transition_counts"
        ][()].decode("utf-8")
        assert json.loads(exclusions) == {"gap": 1, "missing": 2}
        encoded = handle["contracts/chart_spec_json"][()].decode("utf-8")
        assert json.loads(encoded)["chart_id"] == "chart-v1"


def test_blocked_evaluator_uses_training_only_scale_and_neighbor_batch_embargo() -> None:
    from neuropixel_ingest.lfp_dynamics_evaluation_v1 import (
        _eligible_transition_starts,
        _fit_fold_chart,
    )

    n = 1200
    batch = np.arange(n, dtype=np.int32) // 30
    chart = _fit_fold_chart(
        centroid=np.r_[np.arange(6.0), np.full(n - 6, 1000.0)],
        power=np.power(10.0, np.linspace(-13.0, -12.0, n)),
        envelope=np.linspace(0.05, 0.3, n),
        support=np.ones((n, 3), dtype=bool),
        training_windows=np.arange(n) < 6,
    )
    assert chart["median"][0] == pytest.approx(2.5)
    assert chart["values"][10, 0] > 100.0

    complete = np.ones(n, dtype=bool)
    time = np.arange(n, dtype=np.float64) * 2.0
    state = np.full(n, "rest", dtype=object)
    segment = np.full(n, "segment-0", dtype=object)
    flags = np.zeros(n, dtype=bool)
    train_idx, train_dt, _ = _eligible_transition_starts(
        complete=complete,
        time=time,
        state=state,
        segment=segment,
        batch=batch,
        stim_adjacent=flags,
        stim_onset=flags,
        induction=flags,
        state_label="rest",
        lag=1,
        mode="train",
        heldout_batch=20,
        max_relative_tau_deviation=1e-4,
    )
    test_idx, test_dt, _ = _eligible_transition_starts(
        complete=complete,
        time=time,
        state=state,
        segment=segment,
        batch=batch,
        stim_adjacent=flags,
        stim_onset=flags,
        induction=flags,
        state_label="rest",
        lag=1,
        mode="test",
        heldout_batch=20,
        max_relative_tau_deviation=1e-4,
    )
    assert train_idx.size == 37 * 29
    assert not np.isin(batch[train_idx], [19, 20, 21]).any()
    assert test_idx.size == 29
    assert np.all(batch[test_idx] == 20)
    assert np.allclose(train_dt, 2.0)
    assert np.allclose(test_dt, 2.0)


def test_blocked_evaluator_scores_a_synthetic_supported_session(tmp_path: Path) -> None:
    from neuropixel_ingest.lfp_dynamics_evaluation_v1 import evaluate_lfp_dynamics_blocked_v1

    n = 35 * 30
    index = np.arange(n, dtype=np.float64)
    samples = np.arange(n, dtype=np.int64) * 5000
    sample_rate = 2500.0
    starts = samples / sample_rate
    group_ids = ["probeF__MOs__depth_bin_2", "probeF__MOs__depth_bin_3"]
    candidate_path = tmp_path / "candidate.h5"
    chart_path = tmp_path / "chart.json"
    dynamics_path = tmp_path / "dynamics.json"
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "evaluation.json"

    centroid = np.column_stack(
        (
            3.5 + 0.2 * np.sin(index / 17.0) + 0.03 * np.cos(index / 5.0),
            3.6 + 0.2 * np.sin(index / 17.0 + 0.2) + 0.03 * np.cos(index / 5.0),
        )
    )
    power = np.column_stack(
        (
            1e-12 * 10.0 ** (0.4 * np.sin(index / 23.0)),
            1.2e-12 * 10.0 ** (0.4 * np.sin(index / 23.0 + 0.2)),
        )
    )
    envelope = np.column_stack(
        (
            0.2 + 0.05 * np.cos(index / 29.0) + 0.01 * np.sin(index / 7.0),
            0.21 + 0.05 * np.cos(index / 29.0 + 0.1) + 0.01 * np.sin(index / 7.0),
        )
    )

    with h5py.File(candidate_path, "w") as handle:
        handle.attrs["schema"] = "mndm.experimental_lfp_mnps_3_4_candidates.v1"
        contracts = handle.create_group("contracts")
        contracts.create_dataset(
            "estimator_spec_json",
            data=json.dumps(
                {
                    "spec_id": "candidate-v1",
                    "timebase": {"sample_rate_hz": sample_rate},
                    "contact_groups": {"contact_set_sha256": "synthetic-contact-hash"},
                }
            ),
            dtype=h5py.string_dtype("utf-8"),
        )
        windows = handle.create_group("windows")
        windows.create_dataset("source_sample_start", data=samples)
        windows.create_dataset("source_time_start_sec", data=starts)
        windows.create_dataset("source_time_stop_sec", data=starts + 4.0)
        windows.create_dataset(
            "segment_id", data=np.full(n, "segment-0", dtype=object), dtype=h5py.string_dtype("utf-8")
        )
        annotations = handle.create_group("window_annotations")
        annotations.create_dataset(
            "lfp_behavioral_state", data=np.full(n, "rest", dtype=object), dtype=h5py.string_dtype("utf-8")
        )
        for name in ("lfp_stim_adjacent", "lfp_stim_contains_onset", "lfp_induction_excluded"):
            annotations.create_dataset(name, data=np.zeros(n, dtype=bool))
        metrics = handle.create_group("group_metrics")
        metrics.create_dataset("group_id", data=np.asarray(group_ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
        for name, values in (
            ("centroid_hz_median", centroid),
            ("bandpower_v2_median", power),
            ("envelope_cv_median", envelope),
        ):
            metrics.create_dataset(name, data=values)
            metrics.create_dataset(name.replace("_median", "_support_pass"), data=np.ones((n, 2), dtype=bool))

    chart_spec = {
        "status": "frozen_before_lfp_dynamics_computation",
        "chart_id": "chart-v1",
        "source_candidate_estimator_spec_id": "candidate-v1",
        "trajectory_policy": {"group_ids": group_ids},
    }
    dynamics_spec = {
        "status": "frozen_before_lfp_dynamics_computation",
        "candidate_dynamics_values_examined": False,
        "estimator_id": "dynamics-v1",
        "chart_spec_id": "chart-v1",
        "lag_policy": {
            "max_relative_tau_deviation_from_realized_median": 1e-4,
            "predeclared_sensitivity_lag_steps": [1, 2],
        },
        "neighborhood": {
            "k_unique_source_batches": 32,
            "minimum_unique_source_batches": 28,
            "minimum_eligible_transitions_per_behavior_state_and_group": 30,
        },
    }
    plan = {
        "status": "frozen_before_lfp_dynamics_computation",
        "candidate_dynamics_values_examined": False,
        "evaluation_id": "evaluation-v1",
        "chart_spec_id": "chart-v1",
        "dynamics_estimator_id": "dynamics-v1",
        "metrics": {"minimum_scorable_heldout_batches": 20},
    }
    for path, obj in ((chart_path, chart_spec), (dynamics_path, dynamics_spec), (plan_path, plan)):
        path.write_text(json.dumps(obj), encoding="utf-8")

    report = evaluate_lfp_dynamics_blocked_v1(
        candidate_h5_path=candidate_path,
        chart_spec_path=chart_path,
        dynamics_spec_path=dynamics_path,
        evaluation_plan_path=plan_path,
        output_json_path=report_path,
    )
    first_group = report["group_state_lag_summaries"][group_ids[0]]["rest|lag_1"]
    assert first_group["status"] == "evaluable"
    assert first_group["paired_transition_count"] > 0
    assert first_group["scorable_heldout_batch_count"] >= 20
    assert set(first_group["macro_mean_batch_mse"]) == {"full_chart", "bandpower_only", "persistence"}
    assert report["jacobian_followup_screen"]["overall_pass"] is False
    assert report_path.exists()

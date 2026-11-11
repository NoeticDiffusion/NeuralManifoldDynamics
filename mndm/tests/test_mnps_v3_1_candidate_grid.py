"""Synthetic checks for the opt-in three-surface MNPS v3.1 runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_1.candidate_grid import (
    DS004148_CORE60_CHANNELS,
    CandidateGridPlugin,
    CandidateGridValue,
    ds004148_candidate_surfaces,
    run_mnps31_candidate_surface_grid,
)
from mndm.mnps.v3_1.candidates import MNPS31WindowSupport
from mndm.mnps.v3_1 import eeg_metrics
from mndm.mnps.v3_1.realizations import MNPS31WindowSlice


@pytest.fixture(autouse=True)
def _stub_phrase_counter_for_grid_tests(monkeypatch):
    # The exact LZ78 algorithm has its own deterministic synthetic tests. These
    # grid tests exercise row wiring and support without repeating thousands of
    # 5,000-symbol reference parses for every surface.
    monkeypatch.setattr(eeg_metrics, "_lz78_phrase_count", lambda symbols: 100)


def _support(**overrides) -> MNPS31WindowSupport:
    values = {
        "window_start_sec": 20.0,
        "window_end_sec": 40.0,
        "sample_rate_hz": 250.0,
        "sample_count": 5000,
        "segment_id": "synthetic-recording-segment0",
        "source_is_continuous": True,
        "timestamps_uniform": True,
        "gap_free": True,
        "bad_interval_free": True,
        "within_recording_bounds": True,
        "effective_filter_support_valid": True,
    }
    values.update(overrides)
    return MNPS31WindowSupport(**values)


def _window(support: MNPS31WindowSupport | None = None) -> MNPS31WindowSlice:
    support = _support() if support is None else support
    start_sample = int(round(support.window_start_sec * support.sample_rate_hz))
    end_sample = start_sample + support.sample_count
    return MNPS31WindowSlice(
        window_id=f"w{support.window_start_sec:g}_{support.window_end_sec:g}",
        start_sample=start_sample,
        end_sample_exclusive=end_sample,
        support=support,
        provenance={"synthetic": True},
    )


def _inputs(source_labels: tuple[str, ...] | None = None):
    labels = tuple(DS004148_CORE60_CHANNELS) + ("fpz",) if source_labels is None else source_labels
    rng = np.random.default_rng(48148)
    filtered = rng.normal(size=(len(labels), 5000))
    sample = np.arange(5188, dtype=np.float64) / 250.0
    phase_offsets = np.linspace(-1.1, 1.3, len(labels))
    context = np.asarray([
        np.sin(2 * np.pi * 10.2 * sample + phase) + 0.2 * np.sin(2 * np.pi * 12.1 * sample + 0.2 * phase)
        for phase in phase_offsets
    ])
    surfaces = ds004148_candidate_surfaces(labels)
    return labels, filtered, context, surfaces


def _run(*, source_labels=None, bad_channel_names=(), support=None, m1_support=None, m1_context="present", plugins=(), surfaces_override=None, reference_channels=("FCz",)):
    labels, filtered, context, surfaces = _inputs(source_labels)
    active_support = _support() if support is None else support
    if surfaces_override is not None:
        surfaces = surfaces_override
    return run_mnps31_candidate_surface_grid(
        filtered,
        labels,
        surfaces,
        _window(active_support),
        source_identity="ds004148:synthetic-recording:sourcehash",
        recording_identity={
            "dataset_id": "ds004148",
            "subject_id": "sub-synthetic",
            "session_id": "ses-01",
            "task_id": "eyesclosed",
            "run_id": "run-01",
        },
        preprocessing_id="synthetic-source-grid-1_45-hz-v0",
        recorded_reference_context=(context if m1_context == "present" else None),
        recorded_reference_channel_names=labels,
        recorded_reference_id="FCz",
        recorded_reference_channels=reference_channels,
        m1_support=_support() if m1_support is None else m1_support,
        bad_channel_names=bad_channel_names,
        source_provenance={"source_sha256": "a" * 64},
        extension_plugins=plugins,
    )


def test_candidate_grid_emits_nine_outputs_per_surface_with_branch_provenance():
    run = _run()
    document = run.to_dict()
    assert document["schema"] == "mndm.mnps_v3_1_candidate_surface_grid.v1"
    assert document["row_count"] == 27
    assert document["computed_count"] == 27
    assert document["unsupported_count"] == 0
    assert document["qualified"] is False

    by_key = {(row["surface_id"], row["candidate"], row["estimator_id"]): row for row in run.rows}
    assert by_key[("core60", "E1", "E1.normalized_PE5_delay1.median.core60.v1")]["computed"] is True
    assert by_key[("core19", "E2", "E2.median_LZ78_exactMultiset.core19.v1")]["qualified"] is False
    assert by_key[("native", "E3", "E3.covariance_PR.native.v1")]["details"]["reference"]["mode"] == "average_over_effective_surface_channels"

    m1 = by_key[("core60", "M1", "M1.alpha_phase_order_parameter_SD.core60.v1")]
    assert m1["computed"] is True
    assert m1["metric_estimator_id"] == "M1.sensor_phase_order_parameter_sd.v0"
    assert m1["details"]["branch"] == "m1_recorded_reference_8_13_hz"
    assert m1["details"]["reference"]["mode"] == "recorded_input_reference"
    for row in run.rows:
        details = row["details"]
        reference = details["reference"]
        assert details["reference_mode"] == reference["mode"]
        for key in ("reference_id", "reference_channel_labels", "reference_channel_count",
                    "reference_channel_order_sha256", "reference_channel_set_sha256"):
            assert details[key] == reference[key]
    assert m1["details"]["reference_channel_labels"] == ["FCz"]
    assert m1["details"]["reference_channel_count"] == 1
    assert m1["details"]["metric_details"]["average_reference_applied"] is False
    assert m1["details"]["metric_details"]["source_reference_id"] == "FCz"
    assert len(m1["details"]["effective_channel_labels"]) == 60

    car = by_key[("core60", "E3", "E3.covariance_PR.core60.v1")]
    assert car["details"]["branch"] == "surface_CAR_after_1_45_hz_filter"
    assert car["details"]["reference"]["reference_channel_count"] == 60
    assert len(car["details"]["reference"]["reference_channel_set_sha256"]) == 64


def test_m1_unknown_reference_set_remains_unknown_in_flat_and_nested_metadata():
    run = _run(reference_channels=None)
    for row in run.rows:
        if row["candidate"] != "M1":
            continue
        details = row["details"]
        assert details["reference_id"] == "FCz"
        for key in ("reference_channel_labels", "reference_channel_count",
                    "reference_channel_order_sha256", "reference_channel_set_sha256"):
            assert details[key] is None
            assert details["reference"][key] is None


def test_exact_core_missing_channel_only_blocks_that_surface():
    labels = tuple(label for label in DS004148_CORE60_CHANNELS if label != "af3") + ("fpz",)
    run = _run(source_labels=labels)
    core60 = [row for row in run.rows if row["surface_id"] == "core60"]
    core19 = [row for row in run.rows if row["surface_id"] == "core19"]
    native = [row for row in run.rows if row["surface_id"] == "native"]
    assert len(core60) == len(core19) == len(native) == 9
    assert all(row["status"] == "unsupported" for row in core60)
    assert all(
        row["reason"] == ("m1_required_surface_channel_missing" if row["candidate"] == "M1" else "required_surface_channel_missing")
        for row in core60
    )
    assert all(row["value"] is None for row in core60)
    assert all(row["status"] == "computed_exploratory" for row in core19)
    assert all(row["status"] == "computed_exploratory" for row in native)


def test_native_surface_cannot_omit_or_reorder_source_channels():
    labels, _, _, surfaces = _inputs()
    omitted = replace(surfaces[0], channel_names=surfaces[0].channel_names[:-1])
    reordered = replace(
        surfaces[0], channel_names=tuple(reversed(surfaces[0].channel_names))
    )
    for native in (omitted, reordered):
        with pytest.raises(ValueError, match="native_surface_must_match_source_channel_order"):
            _run(surfaces_override=(native, surfaces[1], surfaces[2]))


def test_native_surface_comparison_ignores_outer_label_whitespace():
    labels, *_ = _inputs()
    padded_labels = tuple(f" {label} " for label in labels)
    run = _run(source_labels=padded_labels)
    native = [row for row in run.rows if row["surface_id"] == "native"]
    assert len(native) == 9
    assert all(row["details"]["effective_channel_count"] == len(labels) for row in native)


def test_candidate_grid_accepts_other_sample_aligned_20s_windows_and_updates_m3_times():
    support = _support(window_start_sec=60.0, window_end_sec=80.0)
    run = _run(support=support, m1_support=support)
    m3 = next(
        row for row in run.rows
        if row["surface_id"] == "core60"
        and row["estimator_id"] == "M3.activity_RMS_adjacent_change_vector9.core60.v1"
    )
    intervals = m3["details"]["metric_details"]["block_intervals_sec"]
    assert intervals[0] == [60.0, 62.0]
    assert intervals[-1] == [78.0, 80.0]
    assert m3["window"]["start_sec"] == 60.0


def test_candidate_grid_keeps_the_250hz_support_contract():
    support = _support(sample_rate_hz=200.0, sample_count=4000, window_start_sec=60.0, window_end_sec=80.0)
    with pytest.raises(ValueError, match="sample_aligned_20_second_250hz_window"):
        _run(support=support)


def test_native_drops_bad_channel_with_coverage_qc_while_fixed_cores_keep_rows():
    run = _run(bad_channel_names=("fz",))
    native = next(row for row in run.rows if row["surface_id"] == "native" and row["candidate"] == "E3")
    core60 = next(row for row in run.rows if row["surface_id"] == "core60" and row["candidate"] == "E3")
    core19 = next(row for row in run.rows if row["surface_id"] == "core19" and row["candidate"] == "E3")
    assert native["status"] == "computed_exploratory"
    assert native["details"]["effective_channel_count"] == 60
    assert "fz" in [name.casefold() for name in native["details"]["rejected_channel_labels"]]
    assert "one_or_more_surface_channels_unavailable" in native["details"]["qc_flags"]
    assert core60["status"] == "unsupported" and core60["reason"] == "required_surface_channel_unavailable"
    assert core19["status"] == "unsupported" and core19["reason"] == "required_surface_channel_unavailable"


def test_m1_missing_recorded_context_is_unsupported_without_dropping_other_metrics():
    run = _run(m1_context="missing")
    m1_rows = [row for row in run.rows if row["candidate"] == "M1"]
    other_rows = [row for row in run.rows if row["candidate"] != "M1"]
    assert len(m1_rows) == 3
    assert all(row["status"] == "unsupported" for row in m1_rows)
    assert all(row["reason"] == "recorded_reference_context_missing" for row in m1_rows)
    assert all(row["status"] == "computed_exploratory" for row in other_rows)


def test_extension_plugin_runs_once_per_surface_and_gets_pre_and_post_car_inputs():
    seen: list[str] = []

    def evaluate(context):
        seen.append(context.surface.surface_id)
        assert context.filtered_surface_signals_unreferenced is not None
        assert context.filtered_car_signals is not None
        assert np.allclose(np.mean(context.filtered_car_signals, axis=0), 0.0, atol=1e-12)
        estimator_id = f"M2.synthetic_probe.{context.surface.surface_id}.v0"
        return {estimator_id: CandidateGridValue(
            value=float(len(context.effective_channel_names)),
            units="channels",
            details={"input_reference": "runner-provided_exact_surface_CAR"},
        )}

    plugin = CandidateGridPlugin(
        candidate_id="M2",
        estimator_id_templates=("M2.synthetic_probe.{surface_id}.v0",),
        evaluate=evaluate,
    )
    run = _run(plugins=(plugin,))
    plugin_rows = [row for row in run.rows if row["candidate"] == "M2"]
    assert seen == ["native", "core60", "core19"]
    assert len(plugin_rows) == 3
    assert {row["surface_id"] for row in plugin_rows} == {"native", "core60", "core19"}
    assert all(row["qualified"] is False and row["status"] == "computed_exploratory" for row in plugin_rows)
    implementation = run.provenance["implementation_sha256"]
    assert len(implementation["plugin_M2_evaluator"]) == 64
    assert len(implementation["m2_acf_mobility"]) == 64


def test_extension_plugin_cannot_compute_after_a_support_failure():
    seen: list[str] = []

    def evaluate(context):
        seen.append(context.surface.surface_id)
        estimator_id = f"M2.synthetic_probe.{context.surface.surface_id}.v0"
        return {estimator_id: CandidateGridValue(value=1.0)}

    plugin = CandidateGridPlugin(
        candidate_id="M2",
        estimator_id_templates=("M2.synthetic_probe.{surface_id}.v0",),
        evaluate=evaluate,
    )
    run = _run(plugins=(plugin,), support=_support(gap_free=False))
    rows = [row for row in run.rows if row["candidate"] == "M2"]
    assert seen == []
    assert len(rows) == 3
    assert all(row["status"] == "unsupported" for row in rows)
    assert all(row["reason"] == "gap_or_segment_boundary" for row in rows)


def test_m1_support_source_controls_plugin_support_and_output_branch():
    def evaluate(context):
        estimator_id = f"D2.m1_support_probe.{context.surface.surface_id}.v0"
        return {estimator_id: CandidateGridValue(value=0.5, units="1")}

    plugin = CandidateGridPlugin(
        candidate_id="D2",
        estimator_id_templates=("D2.m1_support_probe.{surface_id}.v0",),
        evaluate=evaluate,
        support_source="m1",
    )
    run = _run(plugins=(plugin,), support=_support(gap_free=False))
    rows = [row for row in run.rows if row["candidate"] == "D2"]
    assert len(rows) == 3
    assert all(row["status"] == "computed_exploratory" for row in rows)
    assert all(row["details"]["branch"] == "m1_recorded_reference_8_13_hz" for row in rows)
    assert all(row["details"]["support"]["gap_free"] is True for row in rows)
    assert all(row["details"]["reference"]["mode"] == "recorded_input_reference" for row in rows)


def test_extension_plugin_bad_numeric_value_or_details_becomes_row_local_unsupported():
    def bad_value(context):
        estimator_id = f"M2.bad_value.{context.surface.surface_id}.v0"
        return {estimator_id: CandidateGridValue(value="not-a-number")}

    def bad_details(context):
        estimator_id = f"D2.bad_details.{context.surface.surface_id}.v0"
        return {estimator_id: CandidateGridValue(value=0.5, details={"bad": np.nan})}

    def matrix_value(context):
        estimator_id = f"M4.matrix_value.{context.surface.surface_id}.v0"
        return {estimator_id: CandidateGridValue(value=np.ones((2, 2)))}

    def nested_list_value(context):
        estimator_id = f"D4.nested_list_value.{context.surface.surface_id}.v0"
        return {estimator_id: CandidateGridValue(value=[[1.0, 2.0]])}

    plugins = (
        CandidateGridPlugin("M2", ("M2.bad_value.{surface_id}.v0",), bad_value),
        CandidateGridPlugin("D2", ("D2.bad_details.{surface_id}.v0",), bad_details),
        CandidateGridPlugin("M4", ("M4.matrix_value.{surface_id}.v0",), matrix_value),
        CandidateGridPlugin("D4", ("D4.nested_list_value.{surface_id}.v0",), nested_list_value),
    )
    run = _run(plugins=plugins)
    bad_values = [row for row in run.rows if row["candidate"] == "M2"]
    bad_detail_rows = [row for row in run.rows if row["candidate"] == "D2"]
    matrix_rows = [row for row in run.rows if row["candidate"] == "M4"]
    nested_rows = [row for row in run.rows if row["candidate"] == "D4"]
    assert all(row["status"] == "unsupported" for row in bad_values)
    assert all(row["reason"] == "candidate_returned_nonnumeric_or_nonfinite_value" for row in bad_values)
    assert all(row["status"] == "unsupported" for row in bad_detail_rows)
    assert all(row["reason"] == "candidate_details_not_finite_json" for row in bad_detail_rows)
    assert all(row["value"] is None for row in bad_detail_rows)
    assert all(row["status"] == "unsupported" for row in matrix_rows + nested_rows)
    assert all(
        row["reason"] == "candidate_returned_nonnumeric_or_nonfinite_value"
        for row in matrix_rows + nested_rows
    )

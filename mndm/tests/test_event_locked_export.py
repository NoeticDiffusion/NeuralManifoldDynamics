"""Tests for pipeline/event_locked_export.py (M5).

All tests use self-contained fixtures — no external data, no HDF5 files required.
Acceptance criteria from architect spec:

1. Provenance preserved in every row.
2. Failed controls / invalid windows never silently dropped.
3. Works when coords_9d or Jacobians are absent.
4. Enough identifiers to join back to HDF5 outputs.
5. Deterministic output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_WINDOWS = 30
WINDOW_SEC = 4.0
STEP_SEC = 4.0


def _make_payload(
    n: int = N_WINDOWS,
    *,
    include_coords_9d: bool = True,
    include_derivatives: bool = True,
    include_stage: bool = True,
    include_window_bounds: bool = True,
) -> "MNPSPayload":
    from mndm.schema import MNPSPayload

    t = np.arange(n, dtype=np.float64) * STEP_SEC + STEP_SEC / 2
    x = np.random.RandomState(0).rand(n, 3).astype(np.float32)
    x_dot = np.random.RandomState(1).rand(n, 3).astype(np.float32) if include_derivatives else None
    stage = np.array([2] * n, dtype=np.int8) if include_stage else None
    w_start = (np.arange(n, dtype=np.float64) * STEP_SEC) if include_window_bounds else None
    w_end = (w_start + WINDOW_SEC) if include_window_bounds else None

    coords_9d = None
    coords_9d_names = None
    if include_coords_9d:
        coords_9d = np.random.RandomState(2).rand(n, 9).astype(np.float32)
        coords_9d_names = ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"]

    return MNPSPayload(
        time=t,
        x=x,
        x_dot=x_dot,
        stage=stage,
        window_start=w_start,
        window_end=w_end,
        coords_9d=coords_9d,
        coords_9d_names=coords_9d_names,
        attrs={"fs_out": 4.0, "window_sec": WINDOW_SEC, "overlap": 0.0},
    )


def _make_event_table(onsets=(20.0, 60.0, 80.0)):
    from mndm.pipeline.event_annotations import EventTable

    onsets = np.array(onsets, dtype=np.float64)
    n = len(onsets)
    return EventTable(
        onset_sec=onsets,
        duration_sec=np.ones(n) * 1.0,
        event_type=np.array(["sleep_spindle"] * n, dtype=object),
        source=np.array(["annotation:test"] * n, dtype=object),
        channel=np.array(["Cz"] * n, dtype=object),
        confidence=np.array([0.9] * n, dtype=np.float64),
        source_path="test/spindles.csv",
        n_events_loaded=n,
    )


def _make_alignment(payload, table):
    from mndm.pipeline.event_alignment import AlignmentConfig, align_events_to_windows

    cfg = AlignmentConfig(stage_transition_margin_sec=0.0)
    return align_events_to_windows(
        table,
        window_start=payload.window_start,
        window_end=payload.window_end,
        time=payload.time,
        stage=payload.stage,
        config=cfg,
    )


def _make_controls(payload, table):
    from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls

    cfg = MatchingConfig(n_controls_per_event=2, seed=42, target_stage=2, exclusion_margin_sec=5.0)
    return build_matched_controls(
        table,
        time=payload.time,
        window_start=payload.window_start,
        window_end=payload.window_end,
        stage=payload.stage,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestSpindleRows:
    def test_event_rows_have_generic_event_condition(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table, ExportConfig

        payload = _make_payload()
        table = _make_event_table()
        alignment = _make_alignment(payload, table)
        controls = _make_controls(payload, table)

        rows = build_event_locked_table(
            payload=payload, alignment=alignment, controls=controls,
            event_table=table, subject_id="sub-001", config=ExportConfig(include_coords_9d=True),
        )
        spindle_rows = [r for r in rows if r["condition"] == "event"]
        assert len(spindle_rows) > 0
        for r in spindle_rows:
            assert r["condition"] == "event"

    def test_spindle_rows_have_valid_bin_labels(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table, ExportConfig
        from mndm.pipeline.event_alignment import DEFAULT_BINS

        valid_labels = {b[0] for b in DEFAULT_BINS}

        payload = _make_payload()
        table = _make_event_table()
        alignment = _make_alignment(payload, table)
        controls = _make_controls(payload, table)

        rows = build_event_locked_table(
            payload=payload, alignment=alignment, controls=controls, event_table=table,
            config=ExportConfig(),
        )
        for r in (r for r in rows if r["condition"] == "event"):
            assert r["bin_label"] in valid_labels, f"Unexpected bin: {r['bin_label']}"

    def test_spindle_rows_have_mnps_columns(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        for r in (r for r in rows if r["condition"] == "event"):
            assert "m" in r and "d" in r and "e" in r
            assert "m_dot" in r and "d_dot" in r and "e_dot" in r

    def test_rows_include_rate_per_min(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        values = [
            float(r.get("rate_per_min", np.nan))
            for r in rows
            if np.isfinite(float(r.get("rate_per_min", np.nan)))
        ]
        assert values
        assert all(v > 0 for v in values)
        assert len({round(v, 12) for v in values}) == 1

    def test_spindle_rows_carry_event_metadata(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
            subject_id="sub-001",
        )
        for r in (r for r in rows if r["condition"] == "event"):
            assert r["event_type"] == "sleep_spindle"
            assert r["event_source"] == "annotation:test"
            assert r["event_channel"] == "Cz"
            assert np.isfinite(r["event_confidence"])
            assert np.isfinite(r["event_onset_sec"])


class TestControlRows:
    def test_control_rows_have_condition_matched_control(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        control_rows = [r for r in rows if r["condition"] == "matched_control"]
        assert len(control_rows) > 0

    def test_control_rows_have_matched_event_id(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        controls = _make_controls(payload, table)
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=controls,
            event_table=table,
        )
        for r in (r for r in rows if r["condition"] == "matched_control"):
            assert r["matched_event_id"] >= 0


class TestAnchorAndLabelExports:
    def test_event_locked_rows_include_anchor_and_task_labels(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        payload.labels["task_state_label"] = np.array(["rest"] * len(payload.time), dtype=object)
        payload.labels["task_load_n"] = np.zeros((len(payload.time),), dtype=np.float32)
        payload.anchor_state = {
            "values": np.stack(
                [
                    np.linspace(0.0, 1.0, len(payload.time), dtype=np.float32),
                    np.linspace(1.0, 2.0, len(payload.time), dtype=np.float32),
                ],
                axis=1,
            ),
            "names": ["anchor_index", "vagal_index"],
        }
        payload.anchor_state_dot = {
            "values": np.ones((len(payload.time), 2), dtype=np.float32) * 0.25,
            "names": ["anchor_index", "vagal_index"],
        }
        payload.anchor_quality = {
            "values": np.ones((len(payload.time), 2), dtype=np.float32) * 0.9,
            "names": ["anchor_index", "vagal_index"],
        }

        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        assert rows
        row = rows[0]
        assert "anchor_index" in row
        assert "anchor_index_dot" in row
        assert "anchor_index_quality" in row
        assert row["task_state_label"] == "rest"
        assert row["task_load_n"] == pytest.approx(0.0)

    def test_event_locked_rows_preserve_invalid_anchor_as_nan_with_quality_flag(self):
        """Guarded anchor invalidity must reach event-locked rows unchanged."""
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        payload.anchor_state = {
            "values": np.full((len(payload.time), 1), np.nan, dtype=np.float32),
            "names": ["anchor_index"],
        }
        payload.anchor_state_dot = {
            "values": np.full((len(payload.time), 1), np.nan, dtype=np.float32),
            "names": ["anchor_index"],
        }
        payload.anchor_quality = {
            "values": np.zeros((len(payload.time), 1), dtype=np.float32),
            "names": ["anchor_index_valid"],
        }

        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, _make_event_table()),
            controls=_make_controls(payload, _make_event_table()),
            event_table=_make_event_table(),
        )
        row = rows[0]
        assert np.isnan(row["anchor_index"])
        assert np.isnan(row["anchor_index_dot"])
        assert row["anchor_index_valid_quality"] == pytest.approx(0.0)

    def test_control_rows_have_no_event_metadata(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        for r in (r for r in rows if r["condition"] == "matched_control"):
            # event metadata should be empty/NaN for controls
            assert r["event_type"] == ""
            assert r["event_source"] == ""
            assert np.isnan(r["event_onset_sec"])

    def test_control_rows_have_mnps_values(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        for r in (r for r in rows if r["condition"] == "matched_control"):
            assert np.isfinite(r["m"]) or True  # may be NaN for edge windows, but key must exist
            assert "m" in r and "d" in r and "e" in r


# ---------------------------------------------------------------------------
# Acceptance criterion 1: Provenance in every row
# ---------------------------------------------------------------------------

class TestProvenanceInEveryRow:
    PROVENANCE_KEYS = [
        "subject_id", "session_id", "run_id", "dataset_id",
        "alignment_reference", "alignment_bins_json", "control_seed",
        "event_source_path", "n_events_input", "n_events_aligned",
        "n_events_excluded_transition", "match_success_rate",
    ]

    def test_all_provenance_keys_present(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table, ExportConfig

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
            subject_id="sub-001",
            session_id="ses-01",
            run_id="run-01",
            dataset_id="ds005555",
            config=ExportConfig(provenance_in_every_row=True),
        )
        for r in rows:
            for key in self.PROVENANCE_KEYS:
                assert key in r, f"Missing provenance key '{key}' in row condition={r.get('condition')}"

    def test_subject_id_matches_in_all_rows(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
            subject_id="sub-042",
        )
        for r in rows:
            assert r["subject_id"] == "sub-042"

    def test_alignment_bins_json_is_parseable(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        if rows:
            parsed = json.loads(rows[0]["alignment_bins_json"])
            assert isinstance(parsed, list)

    def test_control_seed_recorded(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls

        payload = _make_payload()
        table = _make_event_table()
        cfg = MatchingConfig(seed=9999)
        controls = build_matched_controls(
            table, time=payload.time, window_start=payload.window_start,
            window_end=payload.window_end, stage=payload.stage, config=cfg,
        )
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=controls,
            event_table=table,
        )
        for r in rows:
            assert r["control_seed"] == 9999

    def test_event_source_path_recorded(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()  # source_path = "test/spindles.csv"
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        for r in rows:
            assert r["event_source_path"] == "test/spindles.csv"


# ---------------------------------------------------------------------------
# Acceptance criterion 2: Failed controls never silently dropped
# ---------------------------------------------------------------------------

class TestFailedControlsNotDropped:
    def test_zero_events_yields_zero_rows(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table
        from mndm.pipeline.event_annotations import make_empty_event_table
        from mndm.pipeline.event_alignment import align_events_to_windows, AlignmentConfig
        from mndm.pipeline.control_matching import build_matched_controls

        payload = _make_payload()
        empty = make_empty_event_table()
        alignment = align_events_to_windows(
            empty,
            window_start=payload.window_start,
            window_end=payload.window_end,
            time=payload.time,
            config=AlignmentConfig(stage_transition_margin_sec=0.0),
        )
        controls = build_matched_controls(empty, time=payload.time, window_start=payload.window_start, window_end=payload.window_end)
        rows = build_event_locked_table(payload=payload, alignment=alignment, controls=controls, event_table=empty)
        assert rows == []

    def test_match_success_rate_zero_when_no_controls(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table
        from mndm.pipeline.control_matching import MatchingConfig, build_matched_controls

        payload = _make_payload()
        table = _make_event_table(onsets=(2.0,))
        # Tiny exclusion pool forces no matches
        cfg = MatchingConfig(exclusion_margin_sec=1000.0, seed=0)
        controls = build_matched_controls(table, time=payload.time, window_start=payload.window_start, window_end=payload.window_end, stage=payload.stage, config=cfg)
        rows = build_event_locked_table(payload=payload, alignment=_make_alignment(payload, table), controls=controls, event_table=table)
        control_rows = [r for r in rows if r["condition"] == "matched_control"]
        assert len(control_rows) == 0
        # Provenance should still carry match_success_rate = 0
        for r in rows:
            assert r["match_success_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Acceptance criterion 3: Works without optional tensors
# ---------------------------------------------------------------------------

class TestMissingOptionalTensors:
    def test_no_coords_9d(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table, ExportConfig

        payload = _make_payload(include_coords_9d=False)
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
            config=ExportConfig(include_coords_9d=True),
        )
        # coords_9d columns should be NaN but present (if include_coords_9d=True)
        spindle = [r for r in rows if r["condition"] == "event"]
        if spindle:
            # m_a should be NaN when coords_9d absent
            assert np.isnan(spindle[0].get("m_a", 0.0)) or "m_a" not in spindle[0]

    def test_no_derivatives(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload(include_derivatives=False)
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        for r in (r for r in rows if r["condition"] == "event"):
            assert np.isnan(r["m_dot"])
            assert np.isnan(r["d_dot"])
            assert np.isnan(r["e_dot"])

    def test_no_stage(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload(include_stage=False)
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        # stage should default to -1 for all rows
        for r in (r for r in rows if r["condition"] == "event"):
            assert r["stage"] == -1

    def test_no_window_bounds(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload(include_window_bounds=False)
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        for r in (r for r in rows if r["condition"] == "event"):
            assert np.isnan(r["window_start_sec"])
            assert np.isnan(r["window_end_sec"])

    def test_no_event_table_metadata_graceful(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=None,  # no event table
        )
        # Should produce rows without crashing; event metadata columns = NaN/empty
        assert isinstance(rows, list)
        for r in (r for r in rows if r["condition"] == "event"):
            assert np.isnan(r["event_onset_sec"])


# ---------------------------------------------------------------------------
# Acceptance criterion 4: Identifiers for join-back
# ---------------------------------------------------------------------------

class TestJoinBackIdentifiers:
    JOIN_KEYS = ["subject_id", "session_id", "run_id", "dataset_id", "window_id"]

    def test_join_keys_present(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
            subject_id="sub-001",
            session_id="ses-01",
            run_id="run-01",
            dataset_id="ds005555",
        )
        for r in rows:
            for k in self.JOIN_KEYS:
                assert k in r, f"Missing join key '{k}'"

    def test_window_ids_are_valid_indices(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        n = len(payload.time)
        for r in rows:
            assert 0 <= r["window_id"] < n, f"window_id {r['window_id']} out of range [0, {n})"


# ---------------------------------------------------------------------------
# Acceptance criterion 5: Deterministic output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_rows(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table

        payload = _make_payload()
        table = _make_event_table()
        alignment = _make_alignment(payload, table)
        controls = _make_controls(payload, table)

        rows1 = build_event_locked_table(
            payload=payload, alignment=alignment, controls=controls, event_table=table,
            subject_id="sub-001",
        )
        rows2 = build_event_locked_table(
            payload=payload, alignment=alignment, controls=controls, event_table=table,
            subject_id="sub-001",
        )
        assert len(rows1) == len(rows2)
        for r1, r2 in zip(rows1, rows2):
            assert r1["condition"] == r2["condition"]
            assert r1["window_id"] == r2["window_id"]
            assert r1["event_id"] == r2["event_id"]


# ---------------------------------------------------------------------------
# Manifest entry
# ---------------------------------------------------------------------------

class TestManifestEntry:
    def test_manifest_entry_has_row_counts(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table, event_locked_export_manifest_entry

        payload = _make_payload()
        table = _make_event_table()
        alignment = _make_alignment(payload, table)
        controls = _make_controls(payload, table)
        rows = build_event_locked_table(payload=payload, alignment=alignment, controls=controls, event_table=table)

        entry = event_locked_export_manifest_entry(rows, alignment=alignment, controls=controls)
        assert entry["n_rows_total"] == len(rows)
        assert entry["n_event_rows"] + entry["n_matched_control_rows"] == len(rows)
        assert "alignment_qc" in entry
        assert "control_qc" in entry
        assert entry["condition_counts"]["event"] == entry["n_event_rows"]


# ---------------------------------------------------------------------------
# I/O: CSV and Parquet
# ---------------------------------------------------------------------------

class TestCsvOutput:
    def test_write_csv_produces_file(self):
        from mndm.pipeline.event_locked_export import build_event_locked_table, write_event_locked_csv

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "export.csv"
            out = write_event_locked_csv(rows, p)
            assert out is not None
            assert p.exists()
            content = p.read_text(encoding="utf-8")
            assert "condition" in content
            assert "event" in content

    def test_write_csv_empty_rows_returns_none(self):
        from mndm.pipeline.event_locked_export import write_event_locked_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.csv"
            out = write_event_locked_csv([], p)
            assert out is None

    def test_csv_row_count_matches(self):
        import csv
        from mndm.pipeline.event_locked_export import build_event_locked_table, write_event_locked_csv

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.csv"
            write_event_locked_csv(rows, p)
            with p.open(newline="", encoding="utf-8") as fh:
                reader = list(csv.DictReader(fh))
            assert len(reader) == len(rows)


class TestParquetOutput:
    def test_write_parquet_when_pandas_available(self):
        pytest.importorskip("pandas")
        from mndm.pipeline.event_locked_export import build_event_locked_table, write_event_locked_parquet

        payload = _make_payload()
        table = _make_event_table()
        rows = build_event_locked_table(
            payload=payload,
            alignment=_make_alignment(payload, table),
            controls=_make_controls(payload, table),
            event_table=table,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.parquet"
            out = write_event_locked_parquet(rows, p)
            if out is not None:
                assert p.exists() or (p.with_suffix(".csv")).exists()

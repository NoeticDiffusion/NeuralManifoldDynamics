"""Unit tests for pipeline/event_annotations.py."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], sep: str = ",") -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=sep)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# EventTable schema
# ---------------------------------------------------------------------------

class TestEventTableSchema:
    def test_empty_table_is_detected(self):
        from mndm.pipeline.event_annotations import make_empty_event_table

        t = make_empty_event_table()
        assert t.is_empty()
        assert len(t) == 0
        assert t.n == 0

    def test_table_len_matches_onset(self):
        from mndm.pipeline.event_annotations import EventTable

        t = EventTable(onset_sec=np.array([1.0, 2.0, 3.0]))
        assert len(t) == 3
        assert not t.is_empty()

    def test_validate_warns_on_empty(self):
        from mndm.pipeline.event_annotations import make_empty_event_table, validate_event_table

        warnings = validate_event_table(make_empty_event_table())
        assert any("zero rows" in w for w in warnings)

    def test_validate_warns_missing_source(self):
        from mndm.pipeline.event_annotations import EventTable, validate_event_table

        t = EventTable(onset_sec=np.array([1.0, 2.0]))
        warnings = validate_event_table(t)
        assert any("source" in w for w in warnings)

    def test_validate_warns_negative_duration(self):
        from mndm.pipeline.event_annotations import EventTable, validate_event_table

        t = EventTable(
            onset_sec=np.array([1.0, 2.0]),
            duration_sec=np.array([-0.5, 1.0]),
            source=np.array(["test", "test"], dtype=object),
            event_type=np.array(["sleep_spindle", "sleep_spindle"], dtype=object),
        )
        warnings = validate_event_table(t)
        assert any("negative duration" in w for w in warnings)

    def test_validate_passes_clean_table(self):
        from mndm.pipeline.event_annotations import EventTable, validate_event_table

        t = EventTable(
            onset_sec=np.array([1.0, 2.0, 3.0]),
            duration_sec=np.array([0.8, 1.2, 0.9]),
            event_type=np.array(["sleep_spindle"] * 3, dtype=object),
            source=np.array(["annotation:test"] * 3, dtype=object),
        )
        warnings = validate_event_table(t)
        assert warnings == []


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadEventTableFromCsv:
    def test_load_basic_csv(self):
        from mndm.pipeline.event_annotations import load_event_table_from_csv

        rows = [
            {"onset_sec": "3.4", "duration_sec": "1.2", "event_type": "sleep_spindle",
             "source": "annotation:test", "channel": "Cz", "confidence": "0.9"},
            {"onset_sec": "10.0", "duration_sec": "0.8", "event_type": "sleep_spindle",
             "source": "annotation:test", "channel": "Fz", "confidence": "0.7"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.csv"
            _write_csv(p, rows)
            t = load_event_table_from_csv(p)

        assert t.n == 2
        assert t.n_events_loaded == 2
        assert np.isclose(t.onset_sec[0], 3.4)
        assert t.event_type is not None
        assert t.event_type[0] == "sleep_spindle"
        assert t.confidence is not None
        assert np.isclose(t.confidence[1], 0.7)

    def test_load_tsv(self):
        from mndm.pipeline.event_annotations import load_event_table_from_csv

        rows = [
            {"onset_sec": "5.0", "duration_sec": "1.0", "event_type": "sleep_spindle"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.tsv"
            _write_csv(p, rows, sep="\t")
            t = load_event_table_from_csv(p)

        assert t.n == 1

    def test_event_type_filter(self):
        from mndm.pipeline.event_annotations import load_event_table_from_csv

        rows = [
            {"onset_sec": "1.0", "event_type": "sleep_spindle"},
            {"onset_sec": "2.0", "event_type": "slow_oscillation"},
            {"onset_sec": "3.0", "event_type": "sleep_spindle"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.csv"
            _write_csv(p, rows)
            t = load_event_table_from_csv(p, event_type_filter="sleep_spindle")

        assert t.n == 2
        assert t.n_events_loaded == 3

    def test_duration_bounds(self):
        from mndm.pipeline.event_annotations import load_event_table_from_csv

        rows = [
            {"onset_sec": "1.0", "duration_sec": "0.3"},  # too short
            {"onset_sec": "2.0", "duration_sec": "1.0"},  # ok
            {"onset_sec": "3.0", "duration_sec": "4.0"},  # too long
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.csv"
            _write_csv(p, rows)
            t = load_event_table_from_csv(p, min_duration_sec=0.5, max_duration_sec=3.0)

        assert t.n == 1
        assert np.isclose(t.onset_sec[0], 2.0)

    def test_missing_file_returns_empty(self):
        from mndm.pipeline.event_annotations import load_event_table_from_csv

        t = load_event_table_from_csv(Path("/nonexistent/path/events.csv"))
        assert t.is_empty()

    def test_extra_columns_stored_as_metadata_json(self):
        from mndm.pipeline.event_annotations import load_event_table_from_csv

        rows = [
            {"onset_sec": "1.0", "event_type": "sleep_spindle", "detector_version": "v2", "rms": "0.5"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.csv"
            _write_csv(p, rows)
            t = load_event_table_from_csv(p)

        assert t.metadata_json is not None
        parsed = json.loads(t.metadata_json[0])
        assert "detector_version" in parsed


# ---------------------------------------------------------------------------
# HDF5 serialization
# ---------------------------------------------------------------------------

class TestEventTableHdf5:
    def test_to_hdf5_columns_empty(self):
        from mndm.pipeline.event_annotations import make_empty_event_table, event_table_to_hdf5_columns

        assert event_table_to_hdf5_columns(make_empty_event_table()) == {}

    def test_to_hdf5_columns_fields(self):
        from mndm.pipeline.event_annotations import EventTable, event_table_to_hdf5_columns

        t = EventTable(
            onset_sec=np.array([1.0, 2.0]),
            duration_sec=np.array([0.8, 1.0]),
            event_type=np.array(["sleep_spindle", "sleep_spindle"], dtype=object),
            source=np.array(["annotation:test", "annotation:test"], dtype=object),
        )
        cols = event_table_to_hdf5_columns(t)
        assert "onset_sec" in cols
        assert "duration_sec" in cols
        assert "event_type" in cols
        assert "_schema_version" in cols

    def test_manifest_entry(self):
        from mndm.pipeline.event_annotations import EventTable, event_table_manifest_entry

        t = EventTable(
            onset_sec=np.array([1.0, 2.0, 3.0]),
            duration_sec=np.array([0.8, 1.0, 1.2]),
            event_type=np.array(["sleep_spindle"] * 3, dtype=object),
            source=np.array(["annotation:test"] * 3, dtype=object),
            n_events_loaded=3,
        )
        entry = event_table_manifest_entry(t)
        assert entry["n_events"] == 3
        assert "sleep_spindle" in entry["event_types"]
        assert "duration_mean_sec" in entry

    def test_schema_validates_required_fields(self):
        """Canonical acceptance criterion: schema validates required fields."""
        from mndm.pipeline.event_annotations import EventTable, validate_event_table

        t = EventTable(
            onset_sec=np.array([5.0]),
            event_type=np.array(["sleep_spindle"], dtype=object),
            source=np.array(["annotation:test"], dtype=object),
        )
        warnings = validate_event_table(t)
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_h5_roundtrip(self):
        """EventTable columns survive a write → read cycle through HDF5."""
        pytest.importorskip("h5py")
        import h5py
        from mndm.pipeline.event_annotations import EventTable, event_table_to_hdf5_columns

        t = EventTable(
            onset_sec=np.array([3.0, 7.5, 12.1]),
            duration_sec=np.array([1.0, 0.8, 1.3]),
            event_type=np.array(["sleep_spindle"] * 3, dtype=object),
            source=np.array(["annotation:test"] * 3, dtype=object),
        )
        cols = event_table_to_hdf5_columns(t)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.h5"
            with h5py.File(p, "w") as f:
                grp = f.require_group("events")
                for k, v in cols.items():
                    if isinstance(v, (bytes, np.bytes_)):
                        grp.attrs[k] = v.decode("utf-8")
                    else:
                        arr = np.asarray(v)
                        if arr.dtype.kind == "O":
                            str_dt = h5py.string_dtype(encoding="utf-8")
                            grp.create_dataset(k, data=arr.astype(str).astype(object), dtype=str_dt)
                        else:
                            grp.create_dataset(k, data=arr)

            with h5py.File(p, "r") as f:
                assert "events" in f
                assert "onset_sec" in f["events"]
                loaded = f["events"]["onset_sec"][()]
                assert np.allclose(loaded, [3.0, 7.5, 12.1])
                assert "_schema_version" in f["events"].attrs


# ---------------------------------------------------------------------------
# MNPSPayload integration
# ---------------------------------------------------------------------------

class TestMNPSPayloadEventTable:
    def test_payload_accepts_event_table_columns(self):
        from mndm.schema import MNPSPayload
        from mndm.pipeline.event_annotations import EventTable, event_table_to_hdf5_columns

        t = EventTable(onset_sec=np.array([1.0, 2.0]))
        cols = event_table_to_hdf5_columns(t)

        payload = MNPSPayload(
            time=np.arange(5, dtype=np.float64),
            x=np.zeros((5, 3), dtype=np.float32),
            x_dot=np.zeros((5, 3), dtype=np.float32),
            event_table_columns=cols,
        )
        assert "onset_sec" in payload.event_table_columns

    def test_manifest_records_event_config(self):
        """Acceptance criterion: manifest records event config."""
        from mndm.pipeline.event_annotations import EventTable, event_table_manifest_entry

        t = EventTable(
            onset_sec=np.array([1.0]),
            source=np.array(["annotation:test"], dtype=object),
            event_type=np.array(["sleep_spindle"], dtype=object),
            n_events_loaded=1,
        )
        entry = event_table_manifest_entry(t)
        assert entry["n_events"] == 1
        assert entry["sources"] == ["annotation:test"]

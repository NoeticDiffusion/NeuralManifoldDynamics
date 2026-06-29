"""Tests for event-locked CSV source resolution in summary pipeline."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline import summary as summary_mod


def test_resolve_event_locked_csv_sources_discovers_spindle_siblings(tmp_path: Path):
    """Sleep spindle default should resolve sibling YASA CSV files."""
    events_path = tmp_path / "sub-001_task-Sleep_acq-psg_events.tsv"
    events_path.write_text("onset\tduration\tstage_hum\n", encoding="utf-8")

    f3_path = tmp_path / "sub-001_task-Sleep_acq-psg_spindles_yasa_v1_psg_f3.csv"
    c3_path = tmp_path / "sub-001_task-Sleep_acq-psg_spindles_yasa_v1_psg_c3.csv"
    f3_path.write_text("onset_sec,event_type\n0.0,sleep_spindle\n", encoding="utf-8")
    c3_path.write_text("onset_sec,event_type\n0.0,sleep_spindle\n", encoding="utf-8")

    resolved = summary_mod._resolve_event_locked_csv_sources(
        stage_events_path=str(events_path),
        event_locked_cfg={"event_types": ["sleep_spindle"]},
    )

    assert {str(path) for path, _ in resolved} == {str(f3_path), str(c3_path)}
    assert {slug for _, slug in resolved} == {"psg_f3", "psg_c3"}


def test_resolve_event_locked_csv_sources_uses_custom_pattern(tmp_path: Path):
    """Dataset-level csv_source_glob should override default spindle discovery."""
    events_path = tmp_path / "sub-001_task-Sleep_acq-psg_events.tsv"
    events_path.write_text("onset\tduration\tstage_hum\n", encoding="utf-8")

    custom = tmp_path / "sub-001_task-Sleep_acq-psg_custom_annotations.csv"
    custom.write_text("onset_sec,event_type\n0.0,sleep_spindle\n", encoding="utf-8")

    resolved = summary_mod._resolve_event_locked_csv_sources(
        stage_events_path=str(events_path),
        event_locked_cfg={"csv_source_glob": "{events_core}_custom_annotations.csv"},
    )

    assert resolved == [(custom, "")]

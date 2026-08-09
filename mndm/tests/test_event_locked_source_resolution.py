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


def test_resolve_event_locked_csv_sources_uses_explicit_v3_spindle_pattern(tmp_path: Path):
    """The Phase 2 overlay must source strength and SO fields from v3 CSVs."""
    v1_path = tmp_path / "xn001_spindles_yasa_v1_eeg.csv"
    v3_path = tmp_path / "xn001_spindles_yasa_v3_eeg.csv"
    v1_path.write_text("onset_sec,event_type\n0.0,sleep_spindle\n", encoding="utf-8")
    v3_path.write_text(
        "onset_sec,event_type,sigma_power,so_partner_missing\n0.0,sleep_spindle,5.0,0\n",
        encoding="utf-8",
    )

    resolved = summary_mod._resolve_event_locked_csv_sources(
        stage_events_path=None,
        fallback_search_dirs=[tmp_path],
        subject_id="sub-xn001",
        event_locked_cfg={"csv_source_glob": "{events_core}_spindles_yasa_v3_*.csv"},
    )

    assert resolved == [(v3_path, "eeg")]


def test_resolve_event_locked_csv_sources_ds004587_absolute_ig_trial_pattern(tmp_path: Path):
    """ds004587's LUX-recovered trial CSVs live outside the raw BIDS tree.

    ``csv_source_glob`` must resolve to an absolute, non-wildcard,
    ``{events_core}``-templated path pointing at the sidecar directory
    written by ``project/scripts/28_ds004587_lux_trial_sync.py`` --
    mirroring how ``event_locked.datasets.ds004587.csv_source_glob`` is
    configured in ``config_ingest_ds004587.yaml``.
    """
    events_path = tmp_path / "raw" / "sub-FFE057_ses-01_task-IG_run-01_events.tsv"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("onset\tduration\ttrial_type\n0.0\tn/a\trecording_start\n", encoding="utf-8")

    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    trial_csv = sidecar_dir / "sub-FFE057_ses-01_task-IG_run-01_ig_trials_v1.csv"
    trial_csv.write_text("onset_sec,event_type\n12.3,ig_trial\n", encoding="utf-8")

    other_run_csv = sidecar_dir / "sub-FFE058_ses-01_task-IG_run-01_ig_trials_v1.csv"
    other_run_csv.write_text("onset_sec,event_type\n99.0,ig_trial\n", encoding="utf-8")

    resolved = summary_mod._resolve_event_locked_csv_sources(
        stage_events_path=str(events_path),
        event_locked_cfg={
            "csv_source_glob": str(sidecar_dir / "{events_core}_ig_trials_v1.csv").replace("\\", "/"),
            "event_types": ["ig_trial"],
        },
    )

    assert resolved == [(trial_csv, "")]


def test_event_source_config_source_path_must_not_be_used_for_templated_csv_paths():
    """``event_source.source_path`` is read verbatim (no ``{events_core}``
    substitution) by ``resolve_event_table_for_event_locked``, so it must
    stay empty for datasets (like ds004587) that rely on the per-run
    templated ``csv_source_glob`` path instead. Regression guard for the
    ds004587 wiring bug where nesting the template under
    ``event_source.source_path`` caused every run to resolve to the same
    literal, unrendered path.
    """
    from mndm.pipeline.event_locked_config import event_source_config_from_config

    config = {
        "event_locked": {
            "datasets": {
                "ds004587": {
                    "event_source": {"kind": "csv"},
                    "csv_source_glob": "J:/processed/ds004587_ig_trial_sync/{events_core}_ig_trials_v1.csv",
                    "event_types": ["ig_trial"],
                }
            }
        }
    }
    source_cfg = event_source_config_from_config(config, "ds004587")
    assert source_cfg.source_path == ""

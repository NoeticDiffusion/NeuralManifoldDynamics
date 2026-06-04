"""Focused tests for the reusable event-locked runner."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))


def _make_payload():
    from mndm.schema import MNPSPayload

    window_start = np.array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0], dtype=np.float64)
    window_end = window_start + 4.0
    time = (window_start + window_end) / 2.0
    x = np.stack([time, time + 1.0, time + 2.0], axis=1).astype(np.float32)
    x_dot = np.ones_like(x, dtype=np.float32)
    stage = np.full((len(time),), 2, dtype=np.int16)
    return MNPSPayload(
        time=time,
        x=x,
        x_dot=x_dot,
        stage=stage,
        window_start=window_start,
        window_end=window_end,
        attrs={"window_sec": 4.0, "overlap": 0.0},
    )


def _make_config() -> dict:
    return {
        "mnps": {
            "stage_codebook": {
                "PHOTO 10Hz": 50,
                "PHOTO 15Hz": 51,
            }
        },
        "epoching": {
            "datasets": {
                "ds006036": {
                    "length_s": 4.0,
                    "step_s": 4.0,
                    "sampling": {
                        "onset_column": "onset",
                        "duration_column": "duration",
                        "stage_columns": ["trial_type"],
                        "stage_blocking": {
                            "enabled": True,
                            "stage_event_regex": r"^PHOTO\s+(\d+)Hz$",
                            "bridge_marker_labels": ["Photo/HV mark"],
                            "use_bridge_markers": True,
                            "min_block_sec": 2.0,
                            "max_block_sec": 20.0,
                            "bridge_tail_sec": 0.5,
                            "bridge_tail_cap_sec": 1.0,
                        },
                    },
                }
            }
        },
        "event_locked": {
            "datasets": {
                "ds006036": {
                    "enabled": True,
                    "event_source": {
                        "kind": "derived_stage_block_end",
                        "stage_codes": [50],
                        "block_parameters": [10],
                    },
                    "event_types": ["stage_block_end"],
                    "reference": "onset",
                    "stage_filter": ["N2"],
                    "bins": {
                        "in_block_tail_ms": [-8.0, 0.0],
                        "post_block_early_ms": [0.0, 8.0],
                    },
                    "controls": {
                        "n_controls_per_event": 1,
                        "exclusion_margin_sec": 0.0,
                        "seed": 7,
                    },
                    "export": {
                        "write_parquet": False,
                        "write_csv": True,
                    },
                }
            }
        },
    }


def _write_events_tsv(path: Path) -> None:
    rows = [
        {"onset": 10.0, "duration": 0.0, "trial_type": "PHOTO 10Hz"},
        {"onset": 12.0, "duration": 0.0, "trial_type": "Photo/HV mark"},
        {"onset": 14.0, "duration": 0.0, "trial_type": "Photo/HV mark"},
        {"onset": 30.0, "duration": 0.0, "trial_type": "PHOTO 15Hz"},
        {"onset": 32.0, "duration": 0.0, "trial_type": "Photo/HV mark"},
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["onset", "duration", "trial_type"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class TestDerivedStageBlockEndResolver:
    def test_resolves_block_end_event_table_from_stage_blocking(self):
        from mndm.pipeline.event_locked_runner import resolve_event_table_for_event_locked

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "sub-01_task-rest_events.tsv"
            _write_events_tsv(events_path)
            source_cfg, table = resolve_event_table_for_event_locked(
                config=_make_config(),
                dataset_id="ds006036",
                source_path=events_path,
            )

        assert source_cfg.kind == "derived_stage_block_end"
        assert table.n == 1
        assert table.onset_sec[0] == 15.0
        assert table.event_type is not None and table.event_type[0] == "stage_block_end"
        assert table.frequency_hz is not None and table.frequency_hz[0] == 10.0
        assert table.metadata_json is not None
        payload = json.loads(table.metadata_json[0])
        assert payload["derived_from"] == "stage_blocking"
        assert payload["is_inferred"] is True
        assert payload["end_reason"] == "bridge_tail"
        assert payload["membership_mode"] == "midpoint_in_interval"
        assert payload["bridge_tail_sec"] == 0.5
        assert payload["bridge_tail_cap_sec"] == 1.0
        assert payload["bridge_tail_ms"] == 500.0
        assert payload["bridge_tail_cap_ms"] == 1000.0
        assert payload["block_id"] == 0
        assert payload["block_start_sec"] == 10.0
        assert payload["block_end_sec"] == 15.0
        assert payload["block_start_ms"] == 10000.0
        assert payload["block_end_ms"] == 15000.0
        assert payload["block_duration_ms"] == 5000.0


class TestDerivedStageBlockEndRunner:
    def test_runs_alignment_and_csv_export_for_derived_block_end_events(self):
        from mndm.pipeline.event_locked_runner import run_event_locked_export

        payload = _make_payload()
        config = _make_config()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "sub-01_task-rest_events.tsv"
            _write_events_tsv(events_path)
            out_prefix = tmp_path / "sub-01_stage_block_end_event_locked"

            result = run_event_locked_export(
                payload=payload,
                config=config,
                dataset_id="ds006036",
                source_path=events_path,
                subject_id="sub-01",
                run_id="run-01",
                out_prefix=out_prefix,
            )

            csv_paths = [p for p in result.output_paths if p.suffix == ".csv"]
            assert len(csv_paths) == 1
            content = csv_paths[0].read_text(encoding="utf-8")

        event_rows = [row for row in result.rows if row["condition"] == "event"]
        control_rows = [row for row in result.rows if row["condition"] == "matched_control"]

        assert result.source_config.kind == "derived_stage_block_end"
        assert result.event_table.n == 1
        assert len(event_rows) >= 2
        assert len(control_rows) >= 1
        assert any(row["bin_label"] == "post_block_early_ms" for row in event_rows)
        assert result.manifest_entry["n_event_rows"] == len(event_rows)
        assert result.manifest_entry["event_source"]["kind"] == "derived_stage_block_end"
        assert result.manifest_entry["profile"]["event_source_kind"] == "derived_stage_block_end"
        assert "stage_block_end" in content

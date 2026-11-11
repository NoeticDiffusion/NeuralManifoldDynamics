from pathlib import Path

import numpy as np
import pandas as pd

from mndm.pipeline.summary_events import build_bids_event_stage_provenance


def test_bids_event_annotations_can_be_omitted_without_changing_default(tmp_path: Path):
    dataset_root = tmp_path / "received" / "ds005555"
    eeg_dir = dataset_root / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    eeg_name = "sub-001_task-Sleep_acq-psg_eeg.edf"
    (eeg_dir / eeg_name).write_bytes(b"synthetic-edf")
    events_path = eeg_dir / "sub-001_task-Sleep_acq-psg_events.tsv"
    events_path.write_text("onset\tduration\tstage_hum\n0\t30\tN2\n", encoding="utf-8")

    frame = pd.DataFrame(
        {"file": [eeg_name], "t_start": [0.0], "t_end": [30.0]}
    )
    common = {
        "sub_frame": frame,
        "stage_for_windows": None,
        "index_df": None,
        "dataset_root": dataset_root,
        "lookup_rel_paths_by_file_value": lambda _name: [
            str(Path("sub-001") / "eeg" / eeg_name)
        ],
        "mnps_cfg": {
            "window_sec": 30.0,
            "overlap": 0.0,
            "stage_codebook": {"N2": 2},
        },
        "dataset_id": "ds005555",
    }

    default = build_bids_event_stage_provenance(
        **common,
        ctx_config={
            "epoching": {
                "datasets": {
                    "ds005555": {"sampling": {"stage_columns": ["stage_hum"]}}
                }
            }
        },
    )
    assert default.events_path == str(events_path)
    assert default.stage_inferred is not None
    assert default.event_table_columns["raw_event_label"].tolist() == ["N2"]

    suppressed = build_bids_event_stage_provenance(
        **common,
        ctx_config={
            "summarize": {"include_bids_event_annotations": False},
            "epoching": {
                "datasets": {
                    "ds005555": {"sampling": {"stage_columns": ["stage_hum"]}}
                }
            },
        },
    )
    assert suppressed.events_path is None
    assert suppressed.stage_inferred is None
    assert suppressed.event_table_columns == {}
    assert suppressed.legacy_events == {}
    assert suppressed.stage_mapping_qc == {}

from __future__ import annotations

import pandas as pd

from mndm.features import epoch_selection


def test_label_epochs_with_stages_maps_point_events_to_containing_epoch():
    """Point events (duration=0) should map to the epoch containing onset."""
    epoch_meta = [
        (0, 0, 800),      # [0.0, 8.0)
        (1, 400, 1200),   # [4.0, 12.0)
        (2, 800, 1600),   # [8.0, 16.0)
    ]
    sfreq = 100.0
    events_df = pd.DataFrame(
        {
            "onset": [3.9, 8.1],
            "duration": [0.0, 0.0],
            "value": ["PHOTO 5Hz", "open eyes"],
        }
    )

    out = epoch_selection.label_epochs_with_stages(
        epoch_meta=epoch_meta,
        sfreq=sfreq,
        events_df=events_df,
        stage_columns=["value"],
        onset_column="onset",
        duration_column="duration",
        stage_map={"PHOTO 5Hz": 50, "open eyes": 60},
    )

    assert out is not None
    # onset=3.9 maps to epoch 0, onset=8.1 maps to epoch 2.
    assert int(out[0]) == 50
    assert int(out[2]) == 60


def test_label_epochs_with_stages_normalizes_noisy_event_labels():
    """Control chars/whitespace noise should still match stage_map keys."""
    epoch_meta = [(0, 0, 800)]  # [0.0, 8.0)
    sfreq = 100.0
    events_df = pd.DataFrame(
        {
            "onset": [1.0],
            "duration": [0.0],
            "value": ["PAT 2\u001a\u001a CAL"],
        }
    )

    out = epoch_selection.label_epochs_with_stages(
        epoch_meta=epoch_meta,
        sfreq=sfreq,
        events_df=events_df,
        stage_columns=["value"],
        onset_column="onset",
        duration_column="duration",
        stage_map={"PAT 2 CAL": 69},
    )

    assert out is not None
    assert int(out[0]) == 69


def test_label_epochs_with_stages_expands_photic_blocks_with_hv_markers():
    """Photic stage should remain stable across inferred stimulation blocks."""
    epoch_meta = [
        (0, 0, 800),      # [0.0, 8.0), mid=4
        (1, 800, 1600),   # [8.0, 16.0), mid=12
        (2, 1600, 2400),  # [16.0, 24.0), mid=20
        (3, 2400, 3200),  # [24.0, 32.0), mid=28
    ]
    sfreq = 100.0
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 0.2, 4.0, 8.0, 20.0, 20.2, 28.0],
            "duration": [0.0] * 7,
            "value": [
                "PHOTO 5Hz",
                "Photo/HV mark",
                "open eyes",
                "Photo/HV mark",
                "PHOTO 10Hz",
                "Photo/HV mark",
                "Photo/HV mark",
            ],
        }
    )

    out = epoch_selection.label_epochs_with_stages(
        epoch_meta=epoch_meta,
        sfreq=sfreq,
        events_df=events_df,
        stage_columns=["value"],
        onset_column="onset",
        duration_column="duration",
        stage_map={
            "PHOTO 5Hz": 50,
            "PHOTO 10Hz": 51,
            "Photo/HV mark": 54,
            "open eyes": 60,
        },
        sampling_cfg={
            "stage_blocking": {
                "enabled": True,
                "stage_event_regex": r"(?i)^PHOTO\s*(\d+)\s*Hz$",
                "bridge_marker_labels": ["Photo/HV mark"],
                "use_bridge_markers": True,
                "hv_tail_sec": 0.5,
                "min_block_sec": 2.0,
                "max_block_sec": 20.0,
                "preserve_block_assignments": True,
            }
        },
    )

    assert out is not None
    # Block 1 covers epoch 0 (PHOTO 5Hz) and should resist "open eyes" overwrite.
    assert int(out[0]) == 50
    # Epoch 1 remains unlabeled after first inferred block ends.
    assert int(out[1]) == -1
    # Block 2 (PHOTO 10Hz) should cover epochs 2-3.
    assert int(out[2]) == 51
    assert int(out[3]) == 51


def test_stage_blocking_legacy_keys_remain_supported():
    """Legacy stage_blocking key names should remain backward compatible."""
    epoch_meta = [
        (0, 0, 800),      # [0.0, 8.0)
        (1, 800, 1600),   # [8.0, 16.0)
    ]
    sfreq = 100.0
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 0.2, 8.0],
            "duration": [0.0, 0.0, 0.0],
            "value": ["PHOTO 5Hz", "Photo/HV mark", "Photo/HV mark"],
        }
    )
    out = epoch_selection.label_epochs_with_stages(
        epoch_meta=epoch_meta,
        sfreq=sfreq,
        events_df=events_df,
        stage_columns=["value"],
        onset_column="onset",
        duration_column="duration",
        stage_map={"PHOTO 5Hz": 50, "Photo/HV mark": 54},
        sampling_cfg={
            "stage_blocking": {
                "enabled": True,
                "photic_regex": r"(?i)^PHOTO\s*(\d+)\s*Hz$",
                "hv_mark_labels": ["Photo/HV mark"],
                "use_hv_marks": True,
                "preserve_photic_blocks": True,
            }
        },
    )
    assert out is not None
    assert int(out[0]) == 50


def test_label_epochs_with_stages_supports_fully_contained_block_membership():
    """Strict membership should only keep epochs fully inside inferred blocks."""
    epoch_meta = [
        (0, 0, 800),      # [0.0, 8.0)
        (1, 800, 1600),   # [8.0, 16.0)
        (2, 1600, 2400),  # [16.0, 24.0)
        (3, 2400, 3200),  # [24.0, 32.0)
    ]
    sfreq = 100.0
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 0.2, 8.0, 20.0, 20.2, 28.0],
            "duration": [0.0] * 6,
            "value": [
                "PHOTO 5Hz",
                "Photo/HV mark",
                "Photo/HV mark",
                "PHOTO 10Hz",
                "Photo/HV mark",
                "Photo/HV mark",
            ],
        }
    )

    out = epoch_selection.label_epochs_with_stages(
        epoch_meta=epoch_meta,
        sfreq=sfreq,
        events_df=events_df,
        stage_columns=["value"],
        onset_column="onset",
        duration_column="duration",
        stage_map={
            "PHOTO 5Hz": 50,
            "PHOTO 10Hz": 51,
            "Photo/HV mark": 54,
        },
        sampling_cfg={
            "stage_blocking": {
                "enabled": True,
                "stage_event_regex": r"(?i)^PHOTO\s*(\d+)\s*Hz$",
                "bridge_marker_labels": ["Photo/HV mark"],
                "use_bridge_markers": True,
                "bridge_tail_sec": 0.5,
                "min_block_sec": 2.0,
                "max_block_sec": 20.0,
                "preserve_block_assignments": True,
                "window_membership": {"mode": "fully_contained"},
            }
        },
    )

    assert out is not None
    assert out.tolist() == [50, -1, -1, -1]


def test_label_epochs_with_stages_supports_overlap_fraction_block_membership():
    """Overlap-threshold membership should keep only strongly overlapping epochs."""
    epoch_meta = [
        (0, 0, 800),      # [0.0, 8.0)
        (1, 800, 1600),   # [8.0, 16.0)
        (2, 1600, 2400),  # [16.0, 24.0)
        (3, 2400, 3200),  # [24.0, 32.0)
    ]
    sfreq = 100.0
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 0.2, 8.0, 20.0, 20.2, 28.0],
            "duration": [0.0] * 6,
            "value": [
                "PHOTO 5Hz",
                "Photo/HV mark",
                "Photo/HV mark",
                "PHOTO 10Hz",
                "Photo/HV mark",
                "Photo/HV mark",
            ],
        }
    )

    out = epoch_selection.label_epochs_with_stages(
        epoch_meta=epoch_meta,
        sfreq=sfreq,
        events_df=events_df,
        stage_columns=["value"],
        onset_column="onset",
        duration_column="duration",
        stage_map={
            "PHOTO 5Hz": 50,
            "PHOTO 10Hz": 51,
            "Photo/HV mark": 54,
        },
        sampling_cfg={
            "stage_blocking": {
                "enabled": True,
                "stage_event_regex": r"(?i)^PHOTO\s*(\d+)\s*Hz$",
                "bridge_marker_labels": ["Photo/HV mark"],
                "use_bridge_markers": True,
                "bridge_tail_sec": 0.5,
                "min_block_sec": 2.0,
                "max_block_sec": 20.0,
                "preserve_block_assignments": True,
                "window_membership": {
                    "mode": "overlap_frac_ge",
                    "min_overlap_fraction": 0.55,
                },
            }
        },
    )

    assert out is not None
    assert out.tolist() == [50, -1, -1, 51]


def test_label_epochs_with_stages_supports_ds003490_eye_blocks_with_oddball_tones():
    """EO/EC block labels should survive alongside unmapped oddball tones."""
    epoch_meta = [
        (0, 0, 800),      # [0.0, 8.0)
        (1, 400, 1200),   # [4.0, 12.0)
        (2, 800, 1600),   # [8.0, 16.0)
        (3, 1200, 2000),  # [12.0, 20.0)
    ]
    sfreq = 100.0
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 8.0, 2.0, 10.0, 14.0],
            "duration": [8.0, 8.0, 0.0, 0.0, 0.0],
            "trial_type": [
                "Eyes Closed: Every 1000 ms",
                "Eyes Open: Every 1000 ms",
                "Standard Tone",
                "Novel Tone",
                "Target Tone",
            ],
        }
    )

    out = epoch_selection.label_epochs_with_stages(
        epoch_meta=epoch_meta,
        sfreq=sfreq,
        events_df=events_df,
        stage_columns=["trial_type"],
        onset_column="onset",
        duration_column="duration",
        stage_map={
            "Eyes Closed: Every 1000 ms": 10,
            "Eyes Open: Every 1000 ms": 11,
        },
    )

    assert out is not None
    assert int(out[0]) == 10
    assert int(out[1]) == 11
    assert int(out[2]) == 11
    assert int(out[3]) == -1

